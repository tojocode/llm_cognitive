# Devlog — Hybrid Cognitive Model

Chronologisches Protokoll aller Optimierungen und Änderungen am Modell.
Neueste Einträge oben.

---

## 2026-07-10 — Performance-Optimierung Denk-Pipeline (Memoization + Hot-Path-Umbau)

### Motivation
Profiling der Laufzeitpipeline zeigte, dass pro Frage tausendfach dieselben
reinen String-Funktionen mit Regex-Aufrufen liefen und der Triangle-Check in
der Spreading Activation pro Kante lineare Kantenlisten-Scans machte.
Keine Verhaltensänderung beabsichtigt — nur schnellere Berechnung identischer
Ergebnisse.

### Messergebnisse (vorher → nachher)

| Messung | Baseline | Optimiert | Speedup |
|---|---|---|---|
| Denk-Latenz (`denken()`, 100 Fragen, warm) | 72,1 ms/Frage | 11,8 ms/Frage | **6,1×** |
| Eval-Suite gesamt (`run_eval_suite.py`) | 11,9 s | 5,3 s | **2,2×** |

Qualität unverändert (identische Werte vor/nach):
- gold-train: Hit@1 93,75 %, Hit@3/5 100 %, Unknown-Precision 100 %, Leakage-Free 100 %
- gold-holdout: Hit@1 85,29 %, Hit@3 97,06 %, Hit@5 100 %, Unknown-Precision 100 %, Leakage-Free 100 %, Repeat-Stability 100 %
- `SUITE PASS`, `ruff check` sauber.

Die verbleibende Suite-Laufzeit wird vom Holdout-Design dominiert
(`fresh_per_case=True` lädt pro Fall eine frische Engine inkl. JSONL-Parsing,
Cluster- und Ensemble-Rebuild) — das ist beabsichtigte Testisolation, kein
Modell-Overhead.

### Änderungen im Detail

#### 1. Memoization reiner String-Funktionen (`llm_speech/wernicke.py`)
Diese Funktionen sind deterministische Abbildungen ihres String-Inputs und
liefen vorher pro Frage über alle Lexikon-Aliase bzw. alle Konzepte × Labels:

- `_norm_label(s)` — Cache `_norm_label_cache` (Bound 200k, dann Clear).
  Größter Einzelgewinn: wurde pro Frage für jedes Label jedes Konzepts mit
  zwei Regex-Substitutionen aufgerufen (u. a. in `_cue_set`,
  `_phrase_to_concept_id`, Benchmark-Label-Index).
- `_token_variants(token)` — Cache `_token_variants_cache` (Bound 100k).
  Achtung: hängt über den Komposita-Split von `self.lexikon` ab, daher
  Invalidierung nötig (siehe Bugfix unten).
- `_is_junk_concept_id(cid)` — Cache `_junk_id_cache` (Bound 200k); die
  eigentliche Logik liegt jetzt in `_is_junk_concept_id_uncached`.

#### 2. Memoization in der Engine (`llm_core/engine.py`)
- `_canon_type(t)` und `_schema_rel(rel_type)` — Caches im `__init__`
  (`_canon_type_cache`, `_schema_rel_cache`, Bound je 50k); Logik unverändert
  in `*_uncached`-Methoden. Beide liefen pro Kante pro Tick in
  `_gate`, `_cluster_transition_bias` und beim Laden/Speichern.
  Hinweis: `_schema_rel` liest `self.schema_core_relations` — wird diese
  Konfiguration zur Laufzeit mutiert, muss `_schema_rel_cache` geleert werden
  (aktuell wird sie nur im `__init__` gesetzt).
- `_concept_query_tokens(kid)` — Cache `_concept_tokens_cache` mit
  `len(labels)` als Versionsschlüssel (Labels werden nur angehängt, nie
  entfernt — `_ensure_label` deckelt bei 10). Wird pro Frage für jeden
  Cue-Kandidaten in `_split_cues_content_context` aufgerufen.

#### 3. Triangle-Check über WM-Nachbar-Sets (`llm_core/engine.py`)
Vorher: `_triangle_adjust(src, dst)` rief pro propagierter Kante für jeden
WM-Knoten zweimal `_has_edge` auf — jeweils ein linearer Scan über
semantische + episodische Kantenlisten → O(|WM| × Grad) pro Kante.

Neu: `_wm_out_neighbors(wm_ids)` baut einmal pro Tick die
Ausgangs-Nachbar-Sets aller WM-Knoten (semantic komplett, episodic ohne
Sequenzkanten — exakt die `_has_edge`-Semantik). `_triangle_adjust`
macht dann nur noch Set-Lookups. Ergebnisidentisch, da sich innerhalb eines
Ticks keine Kanten ändern (Sequenzkanten entstehen erst nach `_wm_refresh`
am Tick-Ende und sind ohnehin ausgeschlossen).

Angepasst in beiden Aktivierungsschleifen: `spreading_activation` und
`predictive_activation`.

#### 4. Gate-Cache pro Aktivierungslauf (`llm_core/engine.py`)
`_gate(edge_typ, intent)` ist innerhalb eines Aktivierungslaufs eine reine
Funktion des Kantentyps (Intent ist fix). Beide Schleifen halten jetzt ein
lokales `gate_cache: Dict[str, float]` pro Lauf — kein Invalidierungsrisiko,
da der Cache mit dem Lauf endet. Der Seq-Boost-Override in
`predictive_activation` wird wie zuvor nach dem Lookup angewandt.

#### 5. Bugfix: veralteter Komposita-Alias-Cache
`_single_word_aliases` (Basis für den Komposita-Split in `_token_variants`,
z. B. „Vulkanausbrüche" → „vulkan") wurde einmalig lazy aufgebaut und nie
invalidiert. Nach `lexikon_add(...)` oder erneutem `lade_lexikon(...)`
arbeitete der Split mit veraltetem Alias-Bestand. Neu:
`_invalidate_lexikon_caches()` setzt `_single_word_aliases`,
`_token_variants_cache` und `_concept_tokens_cache` bei jeder
Lexikon-Änderung zurück.

#### 6. Cleanup
- `_erkenne_intent`: redundantes Präfix-Tupel
  `("woraus", "woraus besteht", "woraus setzt", "woraus besteht")` auf
  `"woraus"` reduziert (Präfix subsumiert die übrigen Einträge, doppelter
  Eintrag entfernt). Verhalten identisch.

### Bewusst nicht angefasst
- NFC/NFD-Duplikat-Keys in `_schema_rel`-Mapping, `cluster_bridge_types`,
  `ensemble_rel_types` (z. B. „gehört_zu" in beiden Unicode-Formen): nach
  `_canon_type` ist der Input immer NFC, die NFD-Einträge sind tote, aber
  harmlose Keys. Entfernen wäre reine Kosmetik mit Encoding-Risiko beim
  Editieren.
- Invertierter Label-Index für `_cue_set` (statt Scan über alle Konzepte):
  nach der Memoization kein dominanter Kostenpunkt mehr; bei künftig deutlich
  größeren Graphen (>10k Konzepte) der nächste sinnvolle Schritt.
- `_build_label_index` in `run_benchmark.py` wird weiterhin pro Fall gebaut;
  durch den `_norm_label`-Cache jetzt billig.

### Verifikation
```
python debug_analyse/run_eval_suite.py   # SUITE PASS, Metriken identisch
python debug_analyse/run_benchmark.py    # Hit@1 93,75 %, Hit@3/5/focus/trace 100 %
ruff check llm_core/engine.py llm_speech/wernicke.py   # All checks passed
```

### Invarianten für künftige Änderungen
- Alle Memo-Caches gelten nur für reine Funktionen. Wer `lexikon`,
  `schema_core_relations` oder Label-Semantik zur Laufzeit ändert, muss die
  betroffenen Caches invalidieren (Lexikon: `_invalidate_lexikon_caches()`).
- `_concept_tokens_cache` verlässt sich darauf, dass Labels append-only sind.
- `_wm_out_neighbors` muss die `_has_edge`-Semantik spiegeln (semantic alle
  Typen, episodic ohne `seq_type`), sonst driftet der Triangle-Boost.
