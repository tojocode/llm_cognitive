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


def main():
    # allow "/new" as alias for "--new"
    sys.argv = ["--new" if a.lower() == "/new" else a for a in sys.argv]

    parser = argparse.ArgumentParser(description="Autonomous thinking controller")
    parser.add_argument("-n", "--iterations", type=int, default=5, help="number of think iterations")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--semantic", default="data/memory_semantic.jsonl", help="path to semantic memory")
    parser.add_argument("--episodic", default="data/memory_episodic.jsonl", help="path to episodic memory")
    parser.add_argument("--lexikon", default="data/lexikon.json", help="path to lexicon")
    parser.add_argument("--embeddings", default="data/embeddings.json", help="path to embeddings db")
    parser.add_argument("--new", action="store_true", help="rebuild all json files from semantic memory")
    parser.add_argument("--no-learn", action="store_true", help="disable learning during think")
    parser.add_argument("--no-embed", action="store_true", help="disable embedding memory")
    parser.add_argument("--show-seeds", action="store_true", help="print selected seeds")
    parser.add_argument("--print-pattern", action="store_true", help="print top pattern concepts")
    parser.add_argument("--max-pattern", type=int, default=5, help="max pattern items to print")
    parser.add_argument("--print-trace", action="store_true", help="print trace items")
    parser.add_argument("--max-trace", type=int, default=5, help="max trace items to print")
    parser.add_argument("--json", action="store_true", help="json output per iteration")
    parser.add_argument("--save", action="store_true", help="save memories after run")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.new:
        _rebuild_from_semantic(args.semantic, args.episodic, args.lexikon, args.embeddings)
        return

    engine = KognitivesModell(
        args.semantic,
        episodic_datei=args.episodic,
        lexikon_datei=args.lexikon,
        embed_db_path=args.embeddings,
    )

    if args.no_embed:
        engine.embed_enabled = False

    if args.no_learn:
        engine.lerne_aus_aktivierung = lambda *_, **__: None

    results = engine.autonom_denken(steps=max(1, args.iterations))

    if args.json:
        payload = [_jsonable(r, i + 1) for i, r in enumerate(results)]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for i, r in enumerate(results, 1):
            msg = _format_memory_event(r)
            if args.show_seeds:
                seeds = r.get("seeds") or []
                if seeds:
                    msg += " | Seeds: " + ", ".join(seeds)
            print(f"[THINK {i}] {msg}")
            if args.print_pattern:
                print(_format_pattern(r.get("denkmuster") or [], args.max_pattern))
            if args.print_trace:
                print(_format_trace(r.get("trace") or [], args.max_trace))

    if args.save:
        engine.speichere_model(args.semantic, episodic_datei=args.episodic)
        engine.speichere_lexikon(args.lexikon)
        engine.speichere_embeddings()


if __name__ == "__main__":
    main()
