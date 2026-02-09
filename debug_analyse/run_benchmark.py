# -*- coding: utf-8 -*-

from pathlib import Path
import json
import sys
from typing import Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_core.engine import KognitivesModell


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


def _candidate_ids(engine: KognitivesModell, label_map: Dict[str, Set[str]], alias: str) -> Set[str]:
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


def main() -> int:
    bench_path = Path(__file__).with_name("benchmark_questions.json")
    if not bench_path.exists():
        print(f"❌ Benchmark-Datei fehlt: {bench_path}")
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
    total = 0

    for i, case in enumerate(cases, 1):
        q = str(case.get("q", "")).strip()
        expect = case.get("expect") or []
        if not q or not expect:
            continue
        total += 1

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

        if any(cid in candidates for cid in focus_ids):
            focus_hits += 1
        if any(cid in candidates for cid in trace_srcs):
            trace_hits += 1

        ok = _any_hit(candidates, top_ids, 5)
        status = "OK" if ok else "MISS"
        print(f"[{i:02d}] {status} | {q}")
        if not ok:
            top3 = ", ".join(top_ids[:3]) if top_ids else "(leer)"
            focus = ", ".join(focus_ids) if focus_ids else "(leer)"
            tr = ", ".join(trace_srcs[:3]) if trace_srcs else "(leer)"
            print(f"     top3: {top3}")
            print(f"     focus: {focus}")
            print(f"     trace_src: {tr}")
            print(f"     expect: {', '.join(str(x) for x in expect)}")

    if total == 0:
        print("⚠️ Keine gültigen Fälle im Benchmark.")
        return 1

    print("\n=== Benchmark Summary ===")
    for k in ks:
        rate = hits[k] / max(1, total)
        print(f"Hit@{k}: {hits[k]}/{total} = {rate:.2%}")
    print(f"Hit@focus: {focus_hits}/{total} = {focus_hits / max(1, total):.2%}")
    print(f"Hit@trace-src: {trace_hits}/{total} = {trace_hits / max(1, total):.2%}")
    if missing:
        print(f"⚠️ Fehlende Aliase (nicht im Lexikon/Labels gefunden): {missing}")
        for m in missing_list:
            exp = ", ".join(str(x) for x in (m.get("expect") or []))
            print(f"   - {m.get('q')}: {exp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
