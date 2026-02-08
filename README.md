# llm_cognitive

## Struktur
```
llm_cognitive/
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
  main.py            # Interaktiver Runner
  client.py          # Minimaler CLI-Client
  cognitive_engine.py# Legacy-Wrapper
```

## Kurzstart
```
python main.py
python client.py "Warum ist der Himmel blau?"
```
