# Hybrid Cognitive Model - Architektur

## Zweck
Dieses Dokument beschreibt die Funktionsweise des `hybrid_cognition_model` auf Systemebene.
Das Modell kombiniert symbolisches Graph-Denken, episodisches Lernen und optionale Embedding-Retrievals.

## Designziele
- Trennung von **Denken** (interne Aktivierung/Selektion) und **Sprechen** (Antwortformulierung).
- Hohe Antwortstabilitaet durch Intent-Gating, Fokus-Anchor und Unknown-Gate.
- Lernfaehigkeit ueber episodische Spuren mit spaeter Konsolidierung in semantisches Wissen.
- Debugbarkeit durch Trace, Focus, Pattern, Memory-Hits und Benchmark-Suiten.

## Komponenten
- `llm_core/engine.py`
  - Zentrale Laufzeit: Laden/Speichern, Denken, Antworten, Lernen, Konsolidierung.
- `llm_core/types.py`
  - Datentypen fuer Knoten/Kanten/Trace/Working-Memory.
- `llm_speech/wernicke.py`
  - Textnormalisierung, Lexikon, Frage->Cues, Import-Hilfen.
- `llm_speech/broca.py`
  - Antwortgenerierung (regelbasiert + optional LM), Filter fuer schwache Relationen.
- `llm_memory/*`
  - Persistente Daten: semantischer Graph, episodischer Graph, Lexikon, Embeddings.
- `data_import/wikipedia_to_core.py`
  - Import-Pipeline fuer Wikipedia-TXT in beide Memory-Layer inkl. Pruning/Lexikon-Update.
- `debug_analyse/run_benchmark.py`
  - Qualitaetsmessung (Hit@k, Focus/Trace, Unknown-Precision, Leakage-Checks, Repeat-Stability).
- `debug_analyse/run_eval_suite.py`
  - Train/Holdout-Suite mit festen Thresholds fuer Regression-Checks (CI-tauglich).

## Datenmodell
### Knoten (`Konzept`)
- `id`: kanonische Konzept-ID
- `labels`: Aliasnamen
- `semantische_features`: optionale Merkmale
- `verbindungen`: semantische Kanten
- `cluster_id`: Primar-Cluster
- `context_ids`: Kontext-Tags
- `ensemble_ids`: Mehrfachzuordnung zu Ensembles

### Kanten (`Verbindung`)
- `ziel`, `gewicht`, `typ`
- `evidence`: Beobachtungsmenge
- `confidence`: Zuverlaessigkeit [0..1]
- `context_stability`: Kontextstabilitaet [0..1]

### Speichertrennung
- **Semantic Memory** (`memory_semantic.jsonl`): langsamer, stabiler Wissensgraph.
- **Episodic Memory** (`memory_episodic.jsonl`): schneller, situativer Lernspeicher mit staerkerem Decay.

## Laufzeitpipeline (Frage -> Antwort)
### 1) Vorverarbeitung
- Intent-Erkennung (`DEF`, `CAUSE`, `HOW`, `COMPARE`, ...).
- Frage wird in `content_tokens` (Inhalt) und `context_tokens` (Fragekontext) getrennt.

### 2) Cue-Bildung
- Lexikon-Matches, Token-Varianten, optionale Embedding-Hits, Goal-Cues.
- Content-Cues und Context-Cues werden separat bewertet und danach gemerged.
- Tokenvarianten enthalten Plural/Flexion, Umlaut-Transliteration und Komposita-Splits (z. B. `Weltklima` -> `Klima`).

### 3) Strukturierte Aktivierung
- Cluster-First: lokale Aktivierung im relevanten Cluster, Cross-Cluster nur ueber Bridge-Gating.
- Ensemble-Layer: aktive Ensembles verstaerken kompatible Knoten.
- Workspace + Inhibition: begrenztes Aktivierungsbudget, Konkurrenzunterdrueckung.
- Intent-Gating nach Relationsschema (`is_a`, `part_of`, `causes`, `has_property`, ...).
- Inferenzmodus nutzt standardmäßig keine Predictive-Gewichtsupdates im Fragebetrieb (`pred_update_on_think=False`).

### 4) Musterselektion
- Denkmuster (`pattern`) mit Top-K Knoten.
- Focus-IDs und Trace werden aufgebaut.
- Generic-Gate reduziert zu generische Knoten, wenn kein starker Frageanker vorliegt.

### 5) Antwortgenerierung
- Unknown-Gate: bei niedriger Relevanz/Unsicherheit -> "Das weiss ich nicht...".
- Unknown-Stubs sind standardmaessig deaktiviert (`create_unknown_stubs=False`),
  damit Wiederholungen unbekannter Fragen nicht ungewollt neue Pseudo-Konzepte erzeugen.
- Sonst Broca-Ausgabe:
  - bevorzugt intent-passende Relationen
  - filtert schwache Strukturrelationen (`cooccur`, `coactive`, `similar`, `cluster_of`)
  - optional LM-Formulierung mit Trace-Transparenz

## Lernen und Konsolidierung
### Online-Lernen
- `lerne_episodisch_trace(...)`
  - schreibt nur episodische Kanten aus aktuellem Trace (sicherer bei Predictive-Mode).
- `lerne_aus_aktivierung(...)`
  - Hebbian-Verstaerkung, leichte Anti-Hebb-Korrektur, Sequenz- und Triangle-Updates.

### Evidenzmodell
- Jede episodische Kante sammelt `evidence`, `confidence`, `context_stability`.
- Decay reduziert Gewicht/Evidenz ueber Zeit, damit Rauschen abgebaut wird.

### Konsolidierung
- `konsolidiere_episodisch()` ueberfuehrt nur starke episodische Kanten in Semantic,
  basierend auf Score aus Gewicht + Evidenz + Confidence + Kontextstabilitaet.

## Import-Architektur (Wikipedia)
- Eingabe: `data_import/wikipedia/*.txt`
- Pipeline:
  1. Text-Extraktion in Konzepte + Relationen
  2. Import in Semantic und Episodic
  3. Pruning von Rauschknoten/-kanten
  4. Lexikon-Neuaufbau aus Labels/IDs
  5. optionales Kanten-Augmenting
  6. Persistenz aller Memory-Dateien

## Qualitaetssicherung
- Standard-Benchmark: `debug_analyse/benchmark_questions.json`
- Cluster/Context-Benchmark: `debug_analyse/benchmark_cluster_context.json`
- Erweiterter Qualitaetsbenchmark: `debug_analyse/benchmark_quality_extended.json`
  - bekannte Faelle: Hit@k, Focus, Trace
  - unknown-Faelle: Unknown-Precision
  - Leakage-Pruefung ueber `forbid`
- Gold-Split fuer robuste Generalisierung:
  - Train: `debug_analyse/benchmark_gold_train.json`
  - Holdout: `debug_analyse/benchmark_gold_holdout.json`
  - Holdout enthaelt Paraphrasen und Repeat-Faelle (`repeat`, `min_repeat_pass`).
- CI-Suite: `python debug_analyse/run_eval_suite.py`
  - prueft Train/Holdout gegen feste Mindestwerte
  - liefert Exit-Code != 0 bei Regression

## Staerken
- Sehr gute Nachvollziehbarkeit (Trace + Focus + Pattern).
- Robustes Unknown-Verhalten statt Halluzinations-Output.
- Klare Trennung von stabilem Wissen vs. kurzfristigem Lernen.
- Gute Erweiterbarkeit (neue Intents, neue Edge-Typen, neue Gating-Regeln).

## Grenzen
- Kein vollstaendiges Sprachverstehen wie bei grossen End-to-End-LLMs.
- Antwortqualitaet haengt stark von Importqualitaet und Kantenstruktur ab.
- Weltwissen ist auf importierte Inhalte begrenzt.

## Erweiterungspunkte
- Bessere temporale Modelle fuer Ursachenketten.
- Feedback-getriebenes Reward/Value-Lernen.
- Automatische Ontologie-Pruefung fuer konsistente Relationen.
- Visualisierung mit Cluster- und Ensemble-Layern im Export.
