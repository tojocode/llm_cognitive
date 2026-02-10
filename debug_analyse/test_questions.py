# -*- coding: utf-8 -*-

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_core.engine import KognitivesModell  # noqa: E402


def main():
    engine = KognitivesModell()
    questions = [
        "Warum ist Pluto ein Planet?",
        "Was ist ein Axolotl?",
        "Was ist Nachhaltigkeit?",
        "Was ist künstliche Intelligenz?"
    ]
    for i, q in enumerate(questions, 1):
        print("\n" + "▬" * 86)
        print(f"TEST {i}: {q}")
        print("▬" * 86)
        print(engine.antworte(q))


if __name__ == "__main__":
    main()
