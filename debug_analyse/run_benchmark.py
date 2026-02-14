# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

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
    for cid, k in engine.konzepte.items():
        for lab in (k.labels or []):
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
    for t in trace:
        try:
            src = getattr(t, "src", None)
        except Exception:
            src = None
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark für den Denk-Kern")
    parser.add_argument(
        "--file",
        default=str(Path(__file__).with_name("benchmark_questions.json")),
        help="Pfad zur Benchmark-JSON-Datei",
    )
    args = parser.parse_args()

    bench_path = Path(args.file)
    if not bench_path.is_absolute():
        bench_path = (ROOT / bench_path).resolve()
    if not bench_path.exists():
        print(f"FEHLER Benchmark-Datei fehlt: {bench_path}")
        return 1

    cases = _load_cases(bench_path)
    engine = KognitivesModell()
    label_map = _build_label_index(engine)

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

    for i, case in enumerate(cases, 1):
        q = str(case.get("q", "")).strip()
        expect = case.get("expect") or []
        expect_unknown = bool(case.get("expect_unknown"))
        forbid = case.get("forbid") or []

        if not q:
            continue

        if expect_unknown:
            unknown_total += 1
            answer = engine.antworte(q, auto_lernen=False)
            ok_unknown = _is_unknown_answer(answer)
            if ok_unknown:
                unknown_ok += 1
            status = "OK" if ok_unknown else "MISS"
            print(f"[{i:02d}] {status} | {q}")
            if not ok_unknown:
                short = answer.strip().replace("\n", " ")
                if len(short) > 220:
                    short = short[:220] + "..."
                print(f"     answer: {short}")
            continue

        if not expect:
            continue

        known_total += 1

        res = engine.denken(q)
        pattern = res.get("denkmuster") or []
        top_ids = [k for k, _ in pattern]
        focus_ids = _extract_focus_ids(res)
        trace_srcs = _extract_trace_src_ids(res)

        candidates: Set[str] = set()
        for ex in expect:
            candidates.update(_candidate_ids(engine, label_map, str(ex)))

        if not candidates:
            missing += 1
            missing_list.append({"q": q, "expect": expect})

        for k in ks:
            if _any_hit(candidates, top_ids, k):
                hits[k] += 1

        focus_hit = any(cid in candidates for cid in focus_ids)
        trace_hit = any(cid in candidates for cid in trace_srcs)
        if focus_hit:
            focus_hits += 1
        if trace_hit:
            trace_hits += 1

        forbid_ids: Set[str] = set()
        for fb in forbid:
            forbid_ids.update(_candidate_ids(engine, label_map, str(fb)))

        leaked = False
        if forbid_ids:
            leakage_cases += 1
            leak_pool = set(top_ids[:3]) | set(focus_ids) | set(trace_srcs[:3])
            leaked = any(cid in leak_pool for cid in forbid_ids)
            if not leaked:
                leakage_clean += 1

        ok = _any_hit(candidates, top_ids, 5) or focus_hit or trace_hit
        status = "OK" if ok and not leaked else "MISS"
        print(f"[{i:02d}] {status} | {q}")
        if not ok or leaked:
            top3 = ", ".join(top_ids[:3]) if top_ids else "(leer)"
            focus = ", ".join(focus_ids) if focus_ids else "(leer)"
            tr = ", ".join(trace_srcs[:3]) if trace_srcs else "(leer)"
            print(f"     top3: {top3}")
            print(f"     focus: {focus}")
            print(f"     trace_src: {tr}")
            print(f"     expect: {', '.join(str(x) for x in expect)}")
            if leaked:
                print(f"     leak_forbid: {', '.join(str(x) for x in forbid)}")

    if known_total == 0 and unknown_total == 0:
        print("WARN Keine gültigen Fälle im Benchmark.")
        return 1

    print("\n=== Benchmark Summary ===")
    if known_total:
        for k in ks:
            rate = hits[k] / max(1, known_total)
            print(f"Hit@{k}: {hits[k]}/{known_total} = {rate:.2%}")
        print(
            f"Hit@focus: {focus_hits}/{known_total} = "
            f"{focus_hits / max(1, known_total):.2%}"
        )
        print(
            f"Hit@trace-src: {trace_hits}/{known_total} = "
            f"{trace_hits / max(1, known_total):.2%}"
        )

    if unknown_total:
        print(
            f"Unknown-Precision: {unknown_ok}/{unknown_total} = "
            f"{unknown_ok / max(1, unknown_total):.2%}"
        )

    if leakage_cases:
        print(
            f"Leakage-Free@3/focus/trace: {leakage_clean}/{leakage_cases} = "
            f"{leakage_clean / max(1, leakage_cases):.2%}"
        )

    if missing:
        print(f"WARN Fehlende Aliase (nicht im Lexikon/Labels gefunden): {missing}")
        for m in missing_list:
            exp = ", ".join(str(x) for x in (m.get("expect") or []))
            print(f"   - {m.get('q')}: {exp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
