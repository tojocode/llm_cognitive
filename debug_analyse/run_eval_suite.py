# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from debug_analyse.run_benchmark import evaluate_benchmark  # noqa: E402

SUITE = [
    {
        "name": "gold-train",
        "file": "debug_analyse/benchmark_gold_train.json",
        "fresh_per_case": False,
        "fresh_per_run": False,
        "thresholds": {
            "min_hit1": 0.93,
            "min_hit3": 0.99,
            "min_hit5": 0.99,
            "min_unknown_precision": 1.0,
            "min_leakage_free": 0.95,
        },
    },
    {
        "name": "gold-holdout",
        "file": "debug_analyse/benchmark_gold_holdout.json",
        "fresh_per_case": True,
        "fresh_per_run": False,
        "thresholds": {
            "min_hit1": 0.85,
            "min_hit3": 0.94,
            "min_hit5": 0.94,
            "min_unknown_precision": 1.0,
            "min_leakage_free": 0.95,
            "min_repeat_stability": 1.0,
        },
    },
]


def _rate(num: int, den: int) -> float:
    return num / max(1, den)


def _metric_rates(summary: Dict[str, object]) -> Dict[str, float]:
    known_total = int(summary.get("known_total", 0))
    unknown_total = int(summary.get("unknown_total", 0))
    hits = dict(summary.get("hits", {}))

    leakage_cases = int(summary.get("leakage_cases", 0))
    repeat_cases = int(summary.get("repeat_cases", 0))

    return {
        "hit1": _rate(int(hits.get(1, 0)), known_total) if known_total else 0.0,
        "hit3": _rate(int(hits.get(3, 0)), known_total) if known_total else 0.0,
        "hit5": _rate(int(hits.get(5, 0)), known_total) if known_total else 0.0,
        "unknown_precision": (
            _rate(int(summary.get("unknown_ok", 0)), unknown_total) if unknown_total else 1.0
        ),
        "leakage_free": (
            _rate(int(summary.get("leakage_clean", 0)), leakage_cases)
            if leakage_cases
            else 1.0
        ),
        "repeat_stability": (
            _rate(int(summary.get("repeat_stable", 0)), repeat_cases)
            if repeat_cases
            else 1.0
        ),
    }


def _check_thresholds(
    name: str,
    rates: Dict[str, float],
    thresholds: Dict[str, float],
) -> List[str]:
    failures: List[str] = []

    mapping = {
        "min_hit1": "hit1",
        "min_hit3": "hit3",
        "min_hit5": "hit5",
        "min_unknown_precision": "unknown_precision",
        "min_leakage_free": "leakage_free",
        "min_repeat_stability": "repeat_stability",
    }

    for t_key, m_key in mapping.items():
        th = thresholds.get(t_key)
        if th is None:
            continue
        val = rates.get(m_key, 0.0)
        if val < float(th):
            failures.append(f"{name}: {m_key} {val:.2%} < {float(th):.2%}")

    return failures


def _print_block(title: str, summary: Dict[str, object], rates: Dict[str, float]):
    known_total = int(summary.get("known_total", 0))
    unknown_total = int(summary.get("unknown_total", 0))
    hits = dict(summary.get("hits", {}))

    print()
    print(f"=== {title} ===")
    if known_total:
        print(f"Hit@1: {int(hits.get(1, 0))}/{known_total} = {rates['hit1']:.2%}")
        print(f"Hit@3: {int(hits.get(3, 0))}/{known_total} = {rates['hit3']:.2%}")
        print(f"Hit@5: {int(hits.get(5, 0))}/{known_total} = {rates['hit5']:.2%}")

    if unknown_total:
        ok = int(summary.get("unknown_ok", 0))
        print(f"Unknown-Precision: {ok}/{unknown_total} = {rates['unknown_precision']:.2%}")

    leakage_cases = int(summary.get("leakage_cases", 0))
    if leakage_cases:
        clean = int(summary.get("leakage_clean", 0))
        print(f"Leakage-Free: {clean}/{leakage_cases} = {rates['leakage_free']:.2%}")

    repeat_cases = int(summary.get("repeat_cases", 0))
    if repeat_cases:
        stable = int(summary.get("repeat_stable", 0))
        print(f"Repeat-Stability: {stable}/{repeat_cases} = {rates['repeat_stability']:.2%}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gold Eval Suite (Train/Holdout)")
    parser.add_argument(
        "--json-dir",
        default="",
        help="Optionales Verzeichnis für JSON-Summaries pro Suite-Teil",
    )
    args = parser.parse_args()

    failures: List[str] = []

    for spec in SUITE:
        file_path = Path(spec["file"])
        if not file_path.is_absolute():
            file_path = (ROOT / file_path).resolve()
        if not file_path.exists():
            print(f"FEHLER Datei fehlt: {file_path}")
            return 1

        summary = evaluate_benchmark(
            bench_path=file_path,
            fresh_per_case=bool(spec.get("fresh_per_case", False)),
            fresh_per_run=bool(spec.get("fresh_per_run", False)),
            repeat_default=1,
        )
        rates = _metric_rates(summary)
        _print_block(str(spec["name"]), summary, rates)
        failures.extend(_check_thresholds(str(spec["name"]), rates, dict(spec["thresholds"])))

        if args.json_dir:
            out_dir = Path(args.json_dir)
            if not out_dir.is_absolute():
                out_dir = (ROOT / out_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "suite": spec["name"],
                "file": str(file_path),
                "rates": rates,
                "summary": summary,
                "thresholds": spec["thresholds"],
            }
            out_file = out_dir / f"{spec['name']}.json"
            out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"JSON geschrieben: {out_file}")

    if failures:
        print()
        print("SUITE FAIL:")
        for item in failures:
            print(f"  - {item}")
        return 2

    print()
    print("SUITE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
