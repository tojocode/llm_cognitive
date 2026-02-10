# -*- coding: utf-8 -*-

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_core.engine import KognitivesModell  # noqa: E402


def _apply_umlauts(text: str) -> str:
    if not text:
        return text
    t = text
    t = t.replace("ae", "ä").replace("oe", "ö")
    t = t.replace("Ae", "Ä").replace("Oe", "Ö")
    t = t.replace("ue", "ü").replace("Ue", "Ü")
    t = t.replace("fussball", "fußball")
    t = t.replace("Fussball", "Fußball")
    t = t.replace("koerper", "körper").replace("Koerper", "Körper")
    t = t.replace("elektromobilitaet", "elektromobilität")
    t = t.replace("Elektromobilitaet", "Elektromobilität")
    return t


def _topic_from_filename(stem: str) -> str:
    name = stem
    while name and name[0].isdigit():
        name = name[1:]
    if name.startswith("_"):
        name = name[1:]
    name = name.replace("_", " ").strip()
    return name


def _candidate_aliases(stem: str) -> list[str]:
    base = _topic_from_filename(stem)
    if not base:
        return []
    candidates = {base, _apply_umlauts(base)}
    return [c for c in candidates if c]


def _build_auto_definitions(engine: KognitivesModell, limit: int = 30) -> list[dict]:
    wiki_dir = ROOT / "data_import" / "wikipedia"
    if not wiki_dir.exists():
        return []
    out = []
    for p in sorted(wiki_dir.glob("*.txt")):
        chosen = None
        for cand in _candidate_aliases(p.stem):
            norm = engine._norm_label(cand)
            if norm in engine.lexikon:
                chosen = cand
                break
        if chosen:
            out.append({"typ": "Definition", "q": f"Was ist {chosen}?"})
        if limit and len(out) >= limit:
            break
    return out


def _dedupe(questions: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for item in questions:
        q = item.get("q", "").strip().lower()
        if not q or q in seen:
            continue
        out.append(item)
        seen.add(q)
    return out


def main():
    engine = KognitivesModell()
    questions = [
        {"typ": "Definition", "q": "Was ist Künstliche Intelligenz?"},
        {"typ": "Definition", "q": "Was ist das Sonnensystem?"},
        {"typ": "Definition", "q": "Was ist Pluto?"},
        {"typ": "Definition", "q": "Was ist ein Axolotl?"},
        {"typ": "Definition", "q": "Was ist Quantenmechanik?"},
        {"typ": "Definition", "q": "Was ist die deutsche Sprache?"},
        {"typ": "Definition", "q": "Was ist Philosophie?"},
        {"typ": "Definition", "q": "Was ist Psychologie?"},
        {"typ": "Definition", "q": "Was ist Geologie?"},
        {"typ": "Definition", "q": "Was ist Informatik?"},
        {"typ": "Definition", "q": "Was ist Mathematik?"},
        {"typ": "Definition", "q": "Was ist Medizin?"},
        {"typ": "Definition", "q": "Was ist Fußball?"},
        {"typ": "Definition", "q": "Was ist Nachhaltigkeit?"},
        {"typ": "Definition", "q": "Was ist Demokratie?"},
        {"typ": "Definition", "q": "Was ist Europa?"},
        {"typ": "Definition", "q": "Was ist das Internet?"},
        {"typ": "Definition", "q": "Was ist Klimawandel?"},
        {"typ": "Definition", "q": "Was ist ein Ozean?"},
        {"typ": "Definition", "q": "Was ist ein Vulkan?"},
        {"typ": "Ursache", "q": "Warum gibt es den Klimawandel?"},
        {"typ": "Ursache", "q": "Warum ist Pluto ein Zwergplanet?"},
        {"typ": "Prozess", "q": "Wie funktioniert das Internet?"},
        {"typ": "Prozess", "q": "Wie funktioniert das Sonnensystem?"},
        {"typ": "Prozess", "q": "Wie entsteht ein Vulkan?"},
        {"typ": "Zusammensetzung", "q": "Woraus besteht das Sonnensystem?"},
        {"typ": "Zusammensetzung", "q": "Woraus besteht der menschliche Körper?"},
        {"typ": "Eigenschaften", "q": "Welche Eigenschaften hat der Axolotl?"},
        {"typ": "Ort", "q": "Wo lebt der Axolotl?"},
        {"typ": "Ja/Nein", "q": "Ist Pluto ein Zwergplanet?"},
        {"typ": "Ja/Nein", "q": "Ist Mathematik eine Wissenschaft?"},
    ]

    auto_defs = _build_auto_definitions(engine, limit=30)
    questions.extend(auto_defs)
    questions = _dedupe(questions)

    for i, item in enumerate(questions, 1):
        q = item["q"]
        typ = item.get("typ", "Frage")
        print("\n" + "-" * 86)
        print(f"TEST {i} ({typ}): {q}")
        print("-" * 86)
        print(engine.antworte(q))


if __name__ == "__main__":
    main()
