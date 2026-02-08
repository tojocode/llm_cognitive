# -*- coding: utf-8 -*-
# ============================================================
# Minimal CLI client for KognitivesModell
# ============================================================

import argparse
import sys

from llm_core.engine import KognitivesModell


def _read_stdin() -> str:
    if sys.stdin.isatty():
        return ""
    data = sys.stdin.read()
    return data.strip()


def main():
    parser = argparse.ArgumentParser(description="Minimaler CLI-Client für das kognitive Modell")
    parser.add_argument("query", nargs="*", help="Frage als Text (oder über STDIN)")
    parser.add_argument("--semantic", default="data/memory_semantic.jsonl", help="Pfad zu memory_semantic.jsonl")
    parser.add_argument("--episodic", default="data/memory_episodic.jsonl", help="Pfad zu memory_episodic.jsonl")
    parser.add_argument("--lexikon", default="data/lexikon.json", help="Pfad zu lexikon.json")
    parser.add_argument("--llm-cmd", dest="llm_cmd", default=None, help="Shell-Command für lokales LM (liest Prompt von STDIN)")
    parser.add_argument("--no-lm", action="store_true", help="LM-Output deaktivieren (Templates verwenden)")
    parser.add_argument("--import", dest="import_path", default=None, help="Textdatei importieren (UTF-8)")
    parser.add_argument("--to", dest="import_target", default="episodic", choices=["semantic", "episodic"], help="Ziel-Layer für Import")
    parser.add_argument("--think", type=int, default=0, help="Autonom denken (n Zyklen)")
    parser.add_argument("--save", action="store_true", help="Memory/lexikon speichern")
    args = parser.parse_args()

    engine = KognitivesModell(
        args.semantic,
        episodic_datei=args.episodic,
        lm_cmd=args.llm_cmd,
        lexikon_datei=args.lexikon,
    )
    if args.no_lm:
        engine.use_lm_default = False

    if args.import_path:
        stats = engine.import_text_file(args.import_path, target=args.import_target)
        print(f"✓ Import ok: +{stats['nodes_added']} nodes, +{stats['edges_added']} edges ({args.import_target})")

    if args.think and args.think > 0:
        results = engine.autonom_denken(steps=args.think)
        for i, r in enumerate(results, 1):
            text = engine.versprachliche(
                r.get("denkmuster") or [],
                intent="OTHER",
                trace=r.get("trace") or [],
                use_lm=engine.use_lm_default,
            )
            print(f"[THINK {i}] {text}")

    query = " ".join(args.query).strip()
    if not query:
        query = _read_stdin()

    if query:
        ans = engine.antworte(query, use_lm=engine.use_lm_default)
        print(ans)

    if args.save:
        engine.speichere_model(args.semantic, episodic_datei=args.episodic)
        engine.speichere_lexikon(args.lexikon)


if __name__ == "__main__":
    main()
