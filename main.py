# -*- coding: utf-8 -*-
# ============================================================
# Cognitive Engine – Demo Runner
# Datei: main.py
#
# VERSION: 3.1.4
# STATUS: STABIL
#
# ROLLE:
# - Startet das kognitive Modell und führt kurze Tests aus
# - Optional: Import-Lernen aus Textdateien (JSONL Memory)
# - Danach interaktiver Modus
# ============================================================


import os
import argparse


from cognitive_engine import KognitivesModell




def _print_banner(ver: str):
    print("\n" + "=" * 70)
    print(f"🧠 KOGNITIVES DENKSYSTEM v{ver}")
    print("   WM + Inhibition + Intent-Gating + Lernen (Episodic)")
    print("=" * 70)




def _default_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    semantic = os.path.join(base_dir, "memory_semantic.jsonl")
    episodic = os.path.join(base_dir, "memory_episodic.jsonl")
    return semantic, episodic




def _run_tests(engine: KognitivesModell):
    tests = [
        "Warum ist der Himmel blau?",
        "Was ist ein Axolotl?",
        "Wie funktioniert Lernen?",
    ]
    for i, q in enumerate(tests, 1):
        print("\n" + "▬" * 86)
        print(f"TEST {i}: {q}")
        print("▬" * 86)
        ans = engine.antworte(q)
        print("Antwort:", ans)




def _interactive(engine: KognitivesModell, semantic_path: str, episodic_path: str):
    print("\n" + "-" * 70)
    print("Interaktiv: Frage eingeben (ENTER = Ende)")
    print("Commands:")
    print("  /import <pfad> [semantic|episodic]   -> Text importieren & lernen")
    print("  /save                               -> Memory speichern")
    print("  /think [n]                          -> Autonom denken (n Zyklen)")
    print("  /help                               -> Hilfe")
    print("  /exit                               -> Ende")
    print("-" * 70)


    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break


        if not q:
            break


        if q.startswith("/"):
            parts = q.split()
            cmd = parts[0].lower()


            if cmd in {"/exit", "/quit"}:
                break


            if cmd == "/help":
                print("Commands:")
                print("  /import <pfad> [semantic|episodic]")
                print("  /save")
                print("  /think [n]")
                print("  /exit")
                continue


            if cmd == "/save":
                engine.speichere_model(semantic_path, episodic_datei=episodic_path)
                continue


            if cmd == "/import":
                if len(parts) < 2:
                    print("⚠️ Nutzung: /import <pfad> [semantic|episodic]")
                    continue
                p = parts[1]
                target = parts[2].lower() if len(parts) >= 3 else "episodic"
                try:
                    stats = engine.import_text_file(p, target=target)
                    engine.speichere_model(semantic_path, episodic_datei=episodic_path)
                    print(f"✓ Import ok: +{stats['nodes_added']} nodes, +{stats['edges_added']} edges ({target})")
                except FileNotFoundError:
                    print(f"❌ Datei nicht gefunden: {p}")
                except Exception as e:
                    print(f"❌ Import-Fehler: {e}")
                continue

            if cmd == "/think":
                n = 1
                if len(parts) >= 2:
                    try:
                        n = int(parts[1])
                    except Exception:
                        n = 1
                results = engine.autonom_denken(steps=n)
                for i, r in enumerate(results, 1):
                    seeds = [engine._label_for_output(s) for s in (r.get("seeds") or [])]
                    if seeds:
                        print(f"[THINK {i}] Seeds: {', '.join(seeds)}")
                    text = engine.versprachliche(
                        r.get("denkmuster") or [],
                        intent="OTHER",
                        trace=r.get("trace") or [],
                        use_lm=engine.use_lm_default,
                    )
                    print(f"[THINK {i}] {text}")
                continue


            print("⚠️ Unbekannter Command. /help")
            continue


        ans = engine.antworte(q)
        print(ans)




def main():
    parser = argparse.ArgumentParser(description="Kognitives Denksystem – Demo + Import-Lernen")
    parser.add_argument("--semantic", default=None, help="Pfad zu memory_semantic.jsonl")
    parser.add_argument("--episodic", default=None, help="Pfad zu memory_episodic.jsonl")
    parser.add_argument("--import", dest="import_path", default=None, help="Textdatei importieren (UTF-8)")
    parser.add_argument("--to", dest="import_target", default="episodic", choices=["semantic", "episodic"], help="Ziel-Layer für Import")
    parser.add_argument("--llm-cmd", dest="llm_cmd", default=None, help="Shell-Command für lokales LM (liest Prompt von STDIN)")
    parser.add_argument("--no-lm", action="store_true", help="LM-Output deaktivieren (Templates verwenden)")
    parser.add_argument("--no-tests", action="store_true", help="Starttests überspringen")
    parser.add_argument("--no-interactive", action="store_true", help="Interaktivmodus überspringen")
    args = parser.parse_args()


    semantic, episodic = _default_paths()
    if args.semantic:
        semantic = args.semantic
    if args.episodic:
        episodic = args.episodic


    _print_banner("3.1.4")


    engine = KognitivesModell(semantic, episodic_datei=episodic, lm_cmd=args.llm_cmd)
    if args.no_lm:
        engine.use_lm_default = False


    if args.import_path:
        stats = engine.import_text_file(args.import_path, target=args.import_target)
        engine.speichere_model(semantic, episodic_datei=episodic)
        print(f"✓ Import ok: +{stats['nodes_added']} nodes, +{stats['edges_added']} edges ({args.import_target})")


    if not args.no_tests:
        _run_tests(engine)


    if not args.no_interactive:
        _interactive(engine, semantic, episodic)




if __name__ == "__main__":
    main()


# ============================================================
# ROWCOUNT FOOTER
# ============================================================
# ROWCOUNT_OLD: 66
# ROWCOUNT_NEW: 149
