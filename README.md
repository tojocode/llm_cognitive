# mindgraph_model

## Struktur
```
mindgraph_model/
  data_import/
    wikipedia_to_core.py     # Wikipedia-TXT -> Memory (semantic/episodic)
    augment_brain_edges.py   # Zusätzliche Kanten (cooccur/coactive/similar/cluster)
    wikipedia/
  llm_core/
    engine.py        # Denk-Kern
    types.py         # Datentypen
  llm_speech/
    wernicke.py      # Verstehen / Lexikon / Import
    broca.py         # Ausgabe / LM-Hybrid
  llm_export/
    cytoscape_view.html
  llm_memory/
    memory_semantic.jsonl
    memory_episodic.jsonl
    lexikon.json
    embeddings.json
    embedding_db.py
  client.py          # Minimaler CLI-Client
  llm_think/
    think_controler.py # Think-Controller (CLI)
  cognitive_engine.py# Legacy-Wrapper
  debug_analyse/
    test_questions.py
    run_benchmark.py
    export_cytoscape.py
```

## Kurzstart
```
python client.py "Warum ist der Himmel blau?"
python llm_think/think_controler.py
python data_import/wikipedia_to_core.py
python data_import/wikipedia_to_core.py --augment
python debug_analyse/run_benchmark.py
python debug_analyse/export_cytoscape.py
```

## Hinweise
- Warum-Fragen nutzen ein stärkeres Ursache-Gating und eine feste Fokus-Ankerung auf den Frageterm.
- Junk-Knoten werden in Frage-Cues und Antwortmustern gefiltert.
- Warum-Fragen ohne passende Anker vermeiden Embedding-Antworten (lieber „unbekannt“ als Drift).
- `trace` enthält einen expliziten Fokus-Anker (`layer=anchor`) für besseres Debugging.
- Der Benchmark (`debug_analyse/benchmark_questions.json`) enthält jetzt Fragetypen: Definition, Warum, Wie, Vergleich, Abgrenzung, Ursache/Wirkung, Mehrhop.
- Query-Matching nutzt einen leichten Plural/Singular-Fallback (z. B. `Vulkane` -> `Vulkan`) in Cues und Fokus-Ankerung.
- Antworten nutzen ein Relevanz-Gate: bei zu geringer Fragepassung wird sauber mit "Das weiß ich nicht..." abgebrochen statt Schrottantwort.
- Import extrahiert Farb-Relationen (`type=farbe`) aus Formulierungen wie "... Färbung ... ist ...", kontextualisiert über den Dateitopic-Knoten.
- `PROPS` priorisiert Eigenschaft/Farbe stärker als reine `ist`-Kanten, und blendet Rausch-Extras in der Ausgabe aus.
- Import hat robustere Subjektauflösung (Pronomen/Alternativen wie `... oder ...`) und zusätzliche Muster (`gilt als`, `zählt zu`, `befasst sich mit`, `Zu den ... gehören ...`, `wird ... genannt`).
- Vergleichsfragen werden als eigener Intent (`COMPARE`) behandelt; generische Knoten (`System`, `Wissenschaft` usw.) werden im Ranking stärker abgewertet.
- Zusätzliche Junk-Filter reduzieren Rauschknoten wie `Sich`, `Wurde`, `Nicht`.
- Import-IDs werden aus **Content-Tokens** aufgebaut: Füllwörter und Hilfsverben (`der`, `die`, `ist`, `so`, `etwa`, `noch`, ...) werden bei der Knotenbildung aktiv entfernt.
- Broca blendet schwache Strukturkanten (`cooccur`, `coactive`, `similar`, `cluster_of`) in der Sprachausgabe aus, damit Antworten primär auf Wissensrelationen basieren.

