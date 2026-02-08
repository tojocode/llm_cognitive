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


def _run_think(engine: KognitivesModell, n: int):
    results = engine.autonom_denken(steps=n)
    for i, r in enumerate(results, 1):
        mem = _format_memory_event(r)
        print(f"[THINK {i}] {mem}")


def _format_memory_event(result: dict) -> str:
    trace = result.get("trace") or []
    if trace:
        t0 = trace[0]
        return f"Erinnerung: {t0.src} -> {t0.dst} ({t0.typ}, {t0.layer})"
    pattern = result.get("denkmuster") or []
    if pattern:
        top = [k for k, _ in pattern[:3]]
        return "Erinnerung: aktiviert " + ", ".join(top)
    return "Erinnerung: kein Ereignis"


def _chat_loop(engine: KognitivesModell, semantic_path: str, episodic_path: str, lexikon_path: str):
    print("Chat-Modus (ENTER ignoriert / /exit beendet)")
    print("Commands: /import <pfad> [semantic|episodic], /save, /think [n], /lex add <id> <alias...>, /lex save, /exit")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.startswith("/"):
            parts = q.split()
            cmd = parts[0].lower()
            if cmd in {"/exit", "/quit"}:
                break
            if cmd == "/save":
                engine.speichere_model(semantic_path, episodic_datei=episodic_path)
                engine.speichere_lexikon(lexikon_path)
                engine.speichere_embeddings()
                print("✓ Gespeichert")
                continue
            if cmd == "/import":
                if len(parts) < 2:
                    print("⚠️ Nutzung: /import <pfad> [semantic|episodic]")
                    continue
                p = parts[1]
                target = parts[2].lower() if len(parts) >= 3 else "episodic"
                try:
                    stats = engine.import_text_file(p, target=target)
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
                _run_think(engine, n)
                continue
            if cmd == "/lex":
                if len(parts) < 2:
                    print("⚠️ Nutzung: /lex add <id> <alias...> | /lex save")
                    continue
                sub = parts[1].lower()
                if sub == "add":
                    if len(parts) < 4:
                        print("⚠️ Nutzung: /lex add <id> <alias...>")
                        continue
                    cid = parts[2]
                    alias = " ".join(parts[3:]).strip()
                    ok = engine.lexikon_add(cid, alias)
                    if ok:
                        print(f"✓ Lexikon: {alias} -> {cid}")
                    else:
                        print("❌ Lexikon-Fehler: id/alias ungültig")
                    continue
                if sub == "save":
                    engine.speichere_lexikon(lexikon_path)
                    print("✓ Lexikon gespeichert")
                    continue
                print("⚠️ Nutzung: /lex add <id> <alias...> | /lex save")
                continue
            if cmd == "/goal":
                if len(parts) < 2:
                    if engine.goal_state:
                        print("Goals:", ", ".join(engine.goal_state))
                    else:
                        print("Goals: (leer)")
                    continue
                sub = parts[1].lower()
                if sub == "set":
                    g = " ".join(parts[2:]).strip()
                    engine.set_goals([g] if g else [])
                    print("✓ Goal gesetzt" if g else "✓ Goals geleert")
                    continue
                if sub == "add":
                    g = " ".join(parts[2:]).strip()
                    if not g:
                        print("⚠️ Nutzung: /goal add <text>")
                        continue
                    engine.add_goal(g)
                    print("✓ Goal hinzugefügt")
                    continue
                if sub == "clear":
                    engine.clear_goals()
                    print("✓ Goals geleert")
                    continue
                if sub == "show":
                    if engine.goal_state:
                        print("Goals:", ", ".join(engine.goal_state))
                    else:
                        print("Goals: (leer)")
                    continue
                print("⚠️ Nutzung: /goal set <text> | /goal add <text> | /goal clear | /goal show")
                continue
            print("⚠️ Unbekannter Command")
            continue
        ans = engine.antworte(q, use_lm=engine.use_lm_default)
        print(ans)


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
    parser.add_argument("--chat", action="store_true", help="Interaktiver Chat-Modus")
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
        _run_think(engine, args.think)

    query = " ".join(args.query).strip()
    if not query:
        query = _read_stdin()

    if query:
        ans = engine.antworte(query, use_lm=engine.use_lm_default)
        print(ans)

    # Start chat when requested or when no query/stdin was provided
    if (args.chat or not query) and sys.stdin.isatty():
        _chat_loop(engine, args.semantic, args.episodic, args.lexikon)

    if args.save:
        engine.speichere_model(args.semantic, episodic_datei=args.episodic)
        engine.speichere_lexikon(args.lexikon)
        engine.speichere_embeddings()


if __name__ == "__main__":
    main()
