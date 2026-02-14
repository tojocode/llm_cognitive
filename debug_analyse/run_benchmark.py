# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_core.engine import KognitivesModell  # noqa: E402


def _load_cases(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Benchmark-Datei muss eine Liste sein.")
    return data


def _build_label_index(engine: KognitivesModell) -> Dict[str, Set[str]]:
    label_map: Dict[str, Set[str]] = {}
    for cid, konzept in engine.konzepte.items():
        for lab in konzept.labels or []:
            norm = engine._norm_label(lab)
            if not norm:
                continue
            label_map.setdefault(norm, set()).add(cid)
    return label_map


def _candidate_ids(
    engine: KognitivesModell,
    label_map: Dict[str, Set[str]],
    alias: str,
) -> Set[str]:
    norm = engine._norm_label(alias)
    if not norm:
        return set()
    out: Set[str] = set()
    if norm in engine.lexikon:
        out.add(engine.lexikon[norm])
    out.update(label_map.get(norm, set()))
    return out


def _any_hit(candidates: Iterable[str], top_ids: List[str], k: int) -> bool:
    if k <= 0:
        return False
    top = set(top_ids[:k])
    return any(cid in top for cid in candidates)


def _extract_focus_ids(res: dict) -> List[str]:
    focus = res.get("focus") or []
    if isinstance(focus, list):
        return [str(x) for x in focus if str(x)]
    return []


def _extract_trace_src_ids(res: dict) -> List[str]:
    trace = res.get("trace") or []
    out: List[str] = []
    for item in trace:
        src = getattr(item, "src", None)
        if src:
            out.append(str(src))
    return out


def _is_unknown_answer(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    prefixes = (
        "das weiß ich nicht",
        "das weiss ich nicht",
        "ich weiß nicht",
        "ich weiss nicht",
        "unbekannt",
    )
    return any(t.startswith(p) for p in prefixes)


def _build_engine() -> KognitivesModell:
    engine = KognitivesModell()
    # Deterministische Inferenz fuer Benchmarks
    if hasattr(engine, "pred_update_on_think"):
        engine.pred_update_on_think = False
    if hasattr(engine, "pred_lr"):
        engine.pred_lr = 0.0
    if hasattr(engine, "seq_lr"):
        engine.seq_lr = 0.0
    return engine


def _rate(num: int, den: int) -> float:
    return num / max(1, den)


def _eval_known_run(
    engine: KognitivesModell,
    q: str,
    expect: List[str],
    forbid: List[str],
) -> Dict[str, object]:
    label_map = _build_label_index(engine)

    candidates: Set[str] = set()
    for ex in expect:
        candidates.update(_candidate_ids(engine, label_map, str(ex)))

    res = engine.denken(q)
    pattern = res.get("denkmuster") or []
    top_ids = [kid for kid, _ in pattern]
    focus_ids = _extract_focus_ids(res)
    trace_srcs = _extract_trace_src_ids(res)

    hits = {
        1: _any_hit(candidates, top_ids, 1),
        3: _any_hit(candidates, top_ids, 3),
        5: _any_hit(candidates, top_ids, 5),
    }
    focus_hit = any(cid in candidates for cid in focus_ids)
    trace_hit = any(cid in candidates for cid in trace_srcs)

    forbid_ids: Set[str] = set()
    for fb in forbid:
        forbid_ids.update(_candidate_ids(engine, label_map, str(fb)))

    leak_pool = set(top_ids[:3]) | set(focus_ids) | set(trace_srcs[:3])
    leaked = bool(forbid_ids and any(cid in leak_pool for cid in forbid_ids))
    ok = bool((hits[5] or focus_hit or trace_hit) and not leaked)

    return {
        "ok": ok,
        "hits": hits,
        "focus_hit": focus_hit,
        "trace_hit": trace_hit,
        "leaked": leaked,
        "candidates_empty": not candidates,
        "top_ids": top_ids,
        "focus_ids": focus_ids,
        "trace_srcs": trace_srcs,
    }


def _eval_unknown_run(engine: KognitivesModell, q: str) -> Dict[str, object]:
    answer = engine.antworte(q, auto_lernen=False)
    return {"ok": _is_unknown_answer(answer), "answer": answer}


def _print_case(idx: int, status: str, q: str, ok_count: int, repeat: int):
    if repeat > 1:
        print(f"[{idx:02d}] {status} | {q} | repeats: {ok_count}/{repeat}")
    else:
        print(f"[{idx:02d}] {status} | {q}")


def evaluate_benchmark(
    bench_path: Path,
    fresh_per_case: bool,
    fresh_per_run: bool,
    repeat_default: int,
) -> Dict[str, object]:
    cases = _load_cases(bench_path)

    ks = [1, 3, 5]
    hits = {k: 0 for k in ks}
    focus_hits = 0
    trace_hits = 0
    missing = 0
    missing_list = []

    known_total = 0
    unknown_total = 0
    unknown_ok = 0

    leakage_cases = 0
    leakage_clean = 0

    repeat_cases = 0
    repeat_stable = 0

    by_type: Dict[str, Dict[str, int]] = {}

    shared_engine: Optional[KognitivesModell] = None
    if not fresh_per_case and not fresh_per_run:
        shared_engine = _build_engine()
        print(
            f"OK Modell geladen (semantic): {shared_engine.semantic_datei} | "
            f"Konzepte: {len(shared_engine.konzepte)}"
        )
        print(
            f"OK Modell geladen (episodic): {shared_engine.episodic_datei} | "
            f"Konzepte: {len(shared_engine.konzepte)}"
        )

    first_case_load_printed = shared_engine is not None

    for i, case in enumerate(cases, 1):
        q = str(case.get("q", "")).strip()
        if not q:
            continue

        expect = [str(x) for x in (case.get("expect") or [])]
        forbid = [str(x) for x in (case.get("forbid") or [])]
        expect_unknown = bool(case.get("expect_unknown"))
        ctype = str(case.get("type") or "(ohne typ)")

        repeat = max(1, int(case.get("repeat", repeat_default or 1)))
        required = int(case.get("min_repeat_pass", repeat))
        required = max(1, min(repeat, required))

        case_engine: Optional[KognitivesModell]
        if fresh_per_case and not fresh_per_run:
            case_engine = _build_engine()
            if not first_case_load_printed:
                print(
                    f"OK Modell geladen (semantic): {case_engine.semantic_datei} | "
                    f"Konzepte: {len(case_engine.konzepte)}"
                )
                print(
                    f"OK Modell geladen (episodic): {case_engine.episodic_datei} | "
                    f"Konzepte: {len(case_engine.konzepte)}"
                )
                first_case_load_printed = True
        else:
            case_engine = shared_engine

        run_results: List[Dict[str, object]] = []
        for _ in range(repeat):
            engine = _build_engine() if fresh_per_run else case_engine
            if engine is None:
                engine = _build_engine()

            if expect_unknown:
                run_results.append(_eval_unknown_run(engine, q))
            else:
                run_results.append(_eval_known_run(engine, q, expect, forbid))

        ok_count = sum(1 for rr in run_results if bool(rr.get("ok")))
        status = "OK" if ok_count >= required else "MISS"
        _print_case(i, status, q, ok_count, repeat)

        tstats = by_type.setdefault(ctype, {"total": 0, "ok": 0})
        tstats["total"] += 1
        if ok_count >= required:
            tstats["ok"] += 1

        if repeat > 1:
            repeat_cases += 1
            if expect_unknown:
                states = [bool(rr.get("ok")) for rr in run_results]
                stable = len(set(states)) == 1
            else:
                states = [bool(rr.get("ok")) for rr in run_results]
                top1 = []
                for rr in run_results:
                    tops = rr.get("top_ids") or []
                    top1.append(tops[0] if tops else "")
                stable = len(set(states)) == 1 and len(set(top1)) == 1
            if stable:
                repeat_stable += 1

        sample = run_results[0] if run_results else {}

        if expect_unknown:
            unknown_total += 1
            if ok_count >= required:
                unknown_ok += 1
            if ok_count < required:
                short = str(sample.get("answer", "")).strip().replace("\n", " ")
                if len(short) > 220:
                    short = short[:220] + "..."
                print(f"     answer: {short}")
            continue

        if not expect:
            continue

        known_total += 1

        for k in ks:
            run_hit = sum(1 for rr in run_results if bool((rr.get("hits") or {}).get(k)))
            if run_hit >= required:
                hits[k] += 1

        run_focus = sum(1 for rr in run_results if bool(rr.get("focus_hit")))
        run_trace = sum(1 for rr in run_results if bool(rr.get("trace_hit")))
        if run_focus >= required:
            focus_hits += 1
        if run_trace >= required:
            trace_hits += 1

        if forbid:
            leakage_cases += 1
            clean = sum(1 for rr in run_results if not bool(rr.get("leaked")))
            if clean >= required:
                leakage_clean += 1

        if bool(sample.get("candidates_empty")):
            missing += 1
            missing_list.append({"q": q, "expect": expect})

        if ok_count < required:
            top_ids = sample.get("top_ids") or []
            focus_ids = sample.get("focus_ids") or []
            trace_srcs = sample.get("trace_srcs") or []

            top3 = ", ".join(top_ids[:3]) if top_ids else "(leer)"
            focus = ", ".join(focus_ids) if focus_ids else "(leer)"
            tr = ", ".join(trace_srcs[:3]) if trace_srcs else "(leer)"

            print(f"     top3: {top3}")
            print(f"     focus: {focus}")
            print(f"     trace_src: {tr}")
            print(f"     expect: {', '.join(expect)}")
            if bool(sample.get("leaked")):
                print(f"     leak_forbid: {', '.join(forbid)}")

    return {
        "known_total": known_total,
        "unknown_total": unknown_total,
        "hits": hits,
        "focus_hits": focus_hits,
        "trace_hits": trace_hits,
        "unknown_ok": unknown_ok,
        "leakage_cases": leakage_cases,
        "leakage_clean": leakage_clean,
        "repeat_cases": repeat_cases,
        "repeat_stable": repeat_stable,
        "missing": missing,
        "missing_list": missing_list,
        "by_type": by_type,
    }


def _print_summary(summary: Dict[str, object]):
    known_total = int(summary.get("known_total", 0))
    unknown_total = int(summary.get("unknown_total", 0))
    hits = dict(summary.get("hits", {}))

    focus_hits = int(summary.get("focus_hits", 0))
    trace_hits = int(summary.get("trace_hits", 0))

    unknown_ok = int(summary.get("unknown_ok", 0))
    leakage_cases = int(summary.get("leakage_cases", 0))
    leakage_clean = int(summary.get("leakage_clean", 0))

    repeat_cases = int(summary.get("repeat_cases", 0))
    repeat_stable = int(summary.get("repeat_stable", 0))

    missing = int(summary.get("missing", 0))
    missing_list = list(summary.get("missing_list", []))

    by_type = dict(summary.get("by_type", {}))

    print()
    print("=== Benchmark Summary ===")
    if known_total:
        for k in [1, 3, 5]:
            hk = int(hits.get(k, 0))
            print(f"Hit@{k}: {hk}/{known_total} = {_rate(hk, known_total):.2%}")
        print(f"Hit@focus: {focus_hits}/{known_total} = {_rate(focus_hits, known_total):.2%}")
        print(f"Hit@trace-src: {trace_hits}/{known_total} = {_rate(trace_hits, known_total):.2%}")

    if unknown_total:
        print(
            f"Unknown-Precision: {unknown_ok}/{unknown_total} = "
            f"{_rate(unknown_ok, unknown_total):.2%}"
        )

    if leakage_cases:
        print(
            "Leakage-Free@3/focus/trace: "
            f"{leakage_clean}/{leakage_cases} = "
            f"{_rate(leakage_clean, leakage_cases):.2%}"
        )

    if repeat_cases:
        print(
            f"Repeat-Stability: {repeat_stable}/{repeat_cases} = "
            f"{_rate(repeat_stable, repeat_cases):.2%}"
        )

    if by_type:
        print("Type-Accuracy:")
        for t in sorted(by_type.keys()):
            total = int(by_type[t].get("total", 0))
            ok = int(by_type[t].get("ok", 0))
            print(f"  - {t}: {ok}/{total} = {_rate(ok, total):.2%}")

    if missing:
        print(f"WARN Fehlende Aliase (nicht im Lexikon/Labels gefunden): {missing}")
        for item in missing_list:
            exp = ", ".join(str(x) for x in (item.get("expect") or []))
            print(f"   - {item.get('q')}: {exp}")


def _threshold_failures(summary: Dict[str, object], args: argparse.Namespace) -> List[str]:
    failures: List[str] = []

    known_total = int(summary.get("known_total", 0))
    unknown_total = int(summary.get("unknown_total", 0))

    hits = dict(summary.get("hits", {}))

    hit1 = _rate(int(hits.get(1, 0)), known_total) if known_total else 0.0
    hit3 = _rate(int(hits.get(3, 0)), known_total) if known_total else 0.0
    hit5 = _rate(int(hits.get(5, 0)), known_total) if known_total else 0.0

    if unknown_total:
        unknown_precision = _rate(int(summary.get("unknown_ok", 0)), unknown_total)
    else:
        unknown_precision = 1.0

    leakage_cases = int(summary.get("leakage_cases", 0))
    if leakage_cases:
        leakage_free = _rate(int(summary.get("leakage_clean", 0)), leakage_cases)
    else:
        leakage_free = 1.0

    repeat_cases = int(summary.get("repeat_cases", 0))
    if repeat_cases:
        repeat_stability = _rate(int(summary.get("repeat_stable", 0)), repeat_cases)
    else:
        repeat_stability = 1.0

    def check(name: str, val: float, th: Optional[float]):
        if th is None:
            return
        if val < th:
            failures.append(f"{name}: {val:.2%} < {th:.2%}")

    check("Hit@1", hit1, args.min_hit1)
    check("Hit@3", hit3, args.min_hit3)
    check("Hit@5", hit5, args.min_hit5)
    check("Unknown-Precision", unknown_precision, args.min_unknown_precision)
    check("Leakage-Free", leakage_free, args.min_leakage_free)
    check("Repeat-Stability", repeat_stability, args.min_repeat_stability)

    return failures


def _to_json_summary(bench_path: Path, summary: Dict[str, object]) -> Dict[str, object]:
    known_total = int(summary.get("known_total", 0))
    unknown_total = int(summary.get("unknown_total", 0))
    hits = dict(summary.get("hits", {}))

    out = {
        "file": str(bench_path),
        "known_total": known_total,
        "unknown_total": unknown_total,
        "hit_at": {
            "1": _rate(int(hits.get(1, 0)), known_total) if known_total else None,
            "3": _rate(int(hits.get(3, 0)), known_total) if known_total else None,
            "5": _rate(int(hits.get(5, 0)), known_total) if known_total else None,
            "focus": (
                _rate(int(summary.get("focus_hits", 0)), known_total) if known_total else None
            ),
            "trace": (
                _rate(int(summary.get("trace_hits", 0)), known_total) if known_total else None
            ),
        },
        "unknown_precision": (
            _rate(int(summary.get("unknown_ok", 0)), unknown_total) if unknown_total else None
        ),
        "leakage_free": None,
        "repeat_stability": None,
        "missing_aliases": int(summary.get("missing", 0)),
        "by_type": summary.get("by_type", {}),
    }

    leakage_cases = int(summary.get("leakage_cases", 0))
    if leakage_cases:
        out["leakage_free"] = _rate(int(summary.get("leakage_clean", 0)), leakage_cases)

    repeat_cases = int(summary.get("repeat_cases", 0))
    if repeat_cases:
        out["repeat_stability"] = _rate(int(summary.get("repeat_stable", 0)), repeat_cases)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark für den Denk-Kern")
    parser.add_argument(
        "--file",
        default=str(Path(__file__).with_name("benchmark_questions.json")),
        help="Pfad zur Benchmark-JSON-Datei",
    )
    parser.add_argument(
        "--fresh-per-case",
        action="store_true",
        help="Jeden Fall mit frischer Modellinstanz evaluieren.",
    )
    parser.add_argument(
        "--fresh-per-run",
        action="store_true",
        help="Jeden Repeat-Lauf mit frischer Modellinstanz evaluieren.",
    )
    parser.add_argument(
        "--repeat-default",
        type=int,
        default=1,
        help="Standardanzahl Repeat-Läufe je Fall.",
    )

    parser.add_argument("--min-hit1", type=float, default=None)
    parser.add_argument("--min-hit3", type=float, default=None)
    parser.add_argument("--min-hit5", type=float, default=None)
    parser.add_argument("--min-unknown-precision", type=float, default=None)
    parser.add_argument("--min-leakage-free", type=float, default=None)
    parser.add_argument("--min-repeat-stability", type=float, default=None)
    parser.add_argument("--json-out", default=None, help="Optionaler Pfad für JSON-Summary")

    args = parser.parse_args()

    if args.fresh_per_run:
        args.fresh_per_case = True

    bench_path = Path(args.file)
    if not bench_path.is_absolute():
        bench_path = (ROOT / bench_path).resolve()
    if not bench_path.exists():
        print(f"FEHLER Benchmark-Datei fehlt: {bench_path}")
        return 1

    summary = evaluate_benchmark(
        bench_path=bench_path,
        fresh_per_case=bool(args.fresh_per_case),
        fresh_per_run=bool(args.fresh_per_run),
        repeat_default=max(1, int(args.repeat_default)),
    )

    if int(summary.get("known_total", 0)) == 0 and int(summary.get("unknown_total", 0)) == 0:
        print("WARN Keine gültigen Fälle im Benchmark.")
        return 1

    _print_summary(summary)

    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = (ROOT / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _to_json_summary(bench_path, summary)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON Summary geschrieben: {out_path}")

    failures = _threshold_failures(summary, args)
    if failures:
        print()
        print("THRESHOLD-FAIL:")
        for item in failures:
            print(f"  - {item}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
