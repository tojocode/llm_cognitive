# -*- coding: utf-8 -*-

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_core.engine import KognitivesModell


def main():
    engine = KognitivesModell()
    questions = [
        "Warum ist der Himmel blau?",
        "Was ist ein Axolotl?",
        "Wie funktioniert Lernen?",
    ]
    for i, q in enumerate(questions, 1):
        print("\n" + "▬" * 86)
        print(f"TEST {i}: {q}")
        print("▬" * 86)
        print(engine.antworte(q))


if __name__ == "__main__":
    main()
