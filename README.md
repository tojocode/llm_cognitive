# mindgraph_model

## Struktur
```
mindgraph_model/
  llm_core/
    engine.py        # Denk-Kern
    types.py         # Datentypen
  llm_speech/
    wernicke.py      # Verstehen / Lexikon / Import
    broca.py         # Ausgabe / LM-Hybrid
  data/
    memory_semantic.jsonl
    memory_episodic.jsonl
    lexikon.json
    embeddings.json
    embedding_db.py
  client.py          # Minimaler CLI-Client
  llm_train/
    think_controler.py # Think-Controller (CLI)
  cognitive_engine.py# Legacy-Wrapper
  debug_analyse/
    test_questions.py
```

## Kurzstart
```
python client.py "Warum ist der Himmel blau?"
python llm_train/think_controler.py
```

