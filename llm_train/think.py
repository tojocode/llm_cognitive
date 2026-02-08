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


def main():
    parser = argparse.ArgumentParser(description="Autonomous thinking controller")
    parser.add_argument("-n", "--iterations", type=int, default=5, help="number of think iterations")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--semantic", default="data/memory_semantic.jsonl", help="path to semantic memory")
    parser.add_argument("--episodic", default="data/memory_episodic.jsonl", help="path to episodic memory")
    parser.add_argument("--lexikon", default="data/lexikon.json", help="path to lexicon")
    parser.add_argument("--embeddings", default="data/embeddings.json", help="path to embeddings db")
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
