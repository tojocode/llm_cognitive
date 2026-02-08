# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json
import random
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_core.engine import KognitivesModell
from llm_core.types import Verbindung


def _format_memory_event(result: dict) -> str:
    trace = result.get("trace") or []
    if trace:
        t0 = trace[0]
        return f"Memory: {t0.src} -> {t0.dst} ({t0.typ}, {t0.layer})"
    pattern = result.get("denkmuster") or []
    if pattern:
        top = [k for k, _ in pattern[:3]]
        return "Memory: activated " + ", ".join(top)
    return "Memory: no event"


def _format_pattern(pattern: List[tuple], max_items: int) -> str:
    parts = []
    for k, v in pattern[: max(1, max_items)]:
        parts.append(f"{k}={v:.3f}")
    return "Pattern: " + ", ".join(parts) if parts else "Pattern: empty"


def _format_trace(trace: List[object], max_items: int) -> str:
    if not trace:
        return "Trace: empty"
    items = []
    for t in trace[: max(1, max_items)]:
        items.append(f"{t.src}->{t.dst} ({t.typ}, {t.layer}, {t.contrib:.3f})")
    return "Trace: " + " | ".join(items)


def _jsonable(result: dict, idx: int) -> Dict[str, object]:
    pattern = [{"id": k, "a": float(v)} for k, v in (result.get("denkmuster") or [])]
    trace = []
    for t in (result.get("trace") or []):
        trace.append({
            "tick": t.tick,
            "src": t.src,
            "dst": t.dst,
            "typ": t.typ,
            "contrib": float(t.contrib),
            "layer": t.layer,
        })
    return {
        "index": idx,
        "seeds": result.get("seeds") or [],
        "pattern": pattern,
        "trace": trace,
        "timestamp": result.get("timestamp"),
    }


def _rebuild_from_semantic(semantic_path: str, episodic_path: str, lexikon_path: str, embeddings_path: str):
    engine = KognitivesModell(
        semantic_path,
        episodic_datei=None,
        lexikon_datei=None,
        embed_db_path=embeddings_path,
    )

    # rebuild episodic from semantic
    engine.episodic_edges = {}
    for src, k in engine.konzepte.items():
        for e in k.verbindungen:
            engine.episodic_edges.setdefault(src, []).append(
                Verbindung(ziel=e.ziel, gewicht=e.gewicht, typ=e.typ)
            )

    # rebuild lexikon from labels
    lex = {}
    for cid, k in engine.konzepte.items():
        labels = k.labels or [cid]
        for lab in labels:
            norm = engine._norm_label(lab)
            if norm and norm not in lex:
                lex[norm] = cid
    engine.lexikon = lex
    engine.lexikon_datei = lexikon_path
    engine.speichere_lexikon(lexikon_path)

    # rebuild embeddings from semantic nodes/edges
    if engine.embed_db:
        engine.embed_db.items = []
        for cid, k in engine.konzepte.items():
            labels = k.labels or [cid]
            feats = k.semantische_features or []
            if feats:
                text = f"{' / '.join(labels)}. Features: {', '.join(feats)}."
            else:
                text = f"{' / '.join(labels)}."
            engine.embed_db.add(text, meta={"type": "node", "id": cid})
        for src, k in engine.konzepte.items():
            for e in k.verbindungen:
                text = f"{src} {e.typ} {e.ziel}"
                engine.embed_db.add(text, meta={"type": "edge", "src": src, "dst": e.ziel, "rel": e.typ})
        engine.embed_db.save()

    # save rebuilt semantic + episodic
    engine.speichere_model(semantic_path, episodic_datei=episodic_path)


def _make_engine(args) -> KognitivesModell:
    return KognitivesModell(
        args.semantic,
        episodic_datei=args.episodic,
        lexikon_datei=args.lexikon,
        embed_db_path=args.embeddings,
    )


def _print_help():
    print("Kommandos:")
    print("  help                      -> diese Hilfe")
    print("  think [n]                 -> autonom denken (n Iterationen, default 1)")
    print("  new                       -> JSON neu aus memory_semantic aufbauen")
    print("  reload                    -> Modell aus Dateien neu laden")
    print("  save                      -> Modelle + Lexikon + Embeddings speichern")
    print("  seed <n>                  -> Zufallssaat setzen")
    print("  learn on|off              -> Lernen an/aus")
    print("  embed on|off              -> Embedding-Memory an/aus")
    print("  seeds on|off              -> Seeds anzeigen an/aus")
    print("  pattern on|off|<n>         -> Pattern-Ausgabe an/aus, optional max n")
    print("  trace on|off|<n>           -> Trace-Ausgabe an/aus, optional max n")
    print("  json on|off               -> JSON-Ausgabe an/aus")
    print("  config                    -> aktuelle Pfade und Werte")
    print("  status                    -> aktuelle Einstellungen")
    print("  exit                      -> beenden")


def main():
    parser = argparse.ArgumentParser(description="Think CLI")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--semantic", default="data/memory_semantic.jsonl", help="path to semantic memory")
    parser.add_argument("--episodic", default="data/memory_episodic.jsonl", help="path to episodic memory")
    parser.add_argument("--lexikon", default="data/lexikon.json", help="path to lexicon")
    parser.add_argument("--embeddings", default="data/embeddings.json", help="path to embeddings db")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    engine = _make_engine(args)
    learn_fn = engine.lerne_aus_aktivierung

    state = {
        "show_seeds": False,
        "print_pattern": False,
        "max_pattern": 5,
        "print_trace": False,
        "max_trace": 5,
        "json": False,
    }

    _print_help()
    while True:
        try:
            line = input("think> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lstrip("/").lower()

        if cmd in {"exit", "quit"}:
            break
        if cmd in {"help", "commands", "?"}:
            _print_help()
            continue
        if cmd == "status":
            print(f"learn: {'on' if engine.lerne_aus_aktivierung == learn_fn else 'off'}")
            print(f"embed: {'on' if engine.embed_enabled else 'off'}")
            print(f"seeds: {'on' if state['show_seeds'] else 'off'}")
            print(f"pattern: {'on' if state['print_pattern'] else 'off'} (max {state['max_pattern']})")
            print(f"trace: {'on' if state['print_trace'] else 'off'} (max {state['max_trace']})")
            print(f"json: {'on' if state['json'] else 'off'}")
            continue
        if cmd == "config":
            print(f"semantic: {args.semantic}")
            print(f"episodic: {args.episodic}")
            print(f"lexikon: {args.lexikon}")
            print(f"embeddings: {args.embeddings}")
            print(f"embed_enabled: {engine.embed_enabled}")
            print(f"embed_top_k: {engine.embed_top_k}")
            print(f"embed_min_score: {engine.embed_min_score}")
            print(f"embed_cue_boost: {engine.embed_cue_boost}")
            print(f"embed_store_on_import: {engine.embed_store_on_import}")
            continue
        if cmd == "seed" and len(parts) >= 2:
            try:
                random.seed(int(parts[1]))
                print("✓ seed gesetzt")
            except Exception:
                print("⚠️ ungültiger seed")
            continue
        if cmd == "learn" and len(parts) >= 2:
            val = parts[1].lower()
            if val == "on":
                engine.lerne_aus_aktivierung = learn_fn
                print("✓ learn on")
            elif val == "off":
                engine.lerne_aus_aktivierung = lambda *_, **__: None
                print("✓ learn off")
            else:
                print("⚠️ learn on|off")
            continue
        if cmd == "embed" and len(parts) >= 2:
            val = parts[1].lower()
            if val == "on":
                engine.embed_enabled = True
                print("✓ embed on")
            elif val == "off":
                engine.embed_enabled = False
                print("✓ embed off")
            else:
                print("⚠️ embed on|off")
            continue
        if cmd == "seeds" and len(parts) >= 2:
            val = parts[1].lower()
            state["show_seeds"] = (val == "on")
            print(f"✓ seeds {val}")
            continue
        if cmd == "pattern" and len(parts) >= 2:
            val = parts[1].lower()
            if val in {"on", "off"}:
                state["print_pattern"] = (val == "on")
                print(f"✓ pattern {val}")
            else:
                try:
                    state["max_pattern"] = max(1, int(val))
                    state["print_pattern"] = True
                    print(f"✓ pattern on (max {state['max_pattern']})")
                except Exception:
                    print("⚠️ pattern on|off|<n>")
            continue
        if cmd == "trace" and len(parts) >= 2:
            val = parts[1].lower()
            if val in {"on", "off"}:
                state["print_trace"] = (val == "on")
                print(f"✓ trace {val}")
            else:
                try:
                    state["max_trace"] = max(1, int(val))
                    state["print_trace"] = True
                    print(f"✓ trace on (max {state['max_trace']})")
                except Exception:
                    print("⚠️ trace on|off|<n>")
            continue
        if cmd == "json" and len(parts) >= 2:
            val = parts[1].lower()
            state["json"] = (val == "on")
            print(f"✓ json {val}")
            continue
        if cmd == "new":
            _rebuild_from_semantic(args.semantic, args.episodic, args.lexikon, args.embeddings)
            engine = _make_engine(args)
            learn_fn = engine.lerne_aus_aktivierung
            print("✓ rebuild abgeschlossen")
            continue
        if cmd == "reload":
            engine = _make_engine(args)
            learn_fn = engine.lerne_aus_aktivierung
            print("✓ neu geladen")
            continue
        if cmd == "save":
            engine.speichere_model(args.semantic, episodic_datei=args.episodic)
            engine.speichere_lexikon(args.lexikon)
            engine.speichere_embeddings()
            print("✓ gespeichert")
            continue
        if cmd == "think":
            n = 1
            if len(parts) >= 2:
                try:
                    n = max(1, int(parts[1]))
                except Exception:
                    n = 1
            results = engine.autonom_denken(steps=n)
            if state["json"]:
                payload = [_jsonable(r, i + 1) for i, r in enumerate(results)]
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                for i, r in enumerate(results, 1):
                    msg = _format_memory_event(r)
                    if state["show_seeds"]:
                        seeds = r.get("seeds") or []
                        if seeds:
                            msg += " | Seeds: " + ", ".join(seeds)
                    print(f"[THINK {i}] {msg}")
                    if state["print_pattern"]:
                        print(_format_pattern(r.get("denkmuster") or [], state["max_pattern"]))
                    if state["print_trace"]:
                        print(_format_trace(r.get("trace") or [], state["max_trace"]))
            continue

        print("⚠️ unbekanntes Kommando. 'help' zeigt alle Optionen.")


if __name__ == "__main__":
    main()
