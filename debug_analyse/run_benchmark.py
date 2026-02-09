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
    missing = 0
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

        candidates: Set[str] = set()
        for ex in expect:
            candidates.update(_candidate_ids(engine, label_map, str(ex)))

        if not candidates:
            missing += 1

        for k in ks:
            if _any_hit(candidates, top_ids, k):
                hits[k] += 1

        ok = _any_hit(candidates, top_ids, 5)
        status = "OK" if ok else "MISS"
        print(f"[{i:02d}] {status} | {q}")
        if not ok:
            top3 = ", ".join(top_ids[:3]) if top_ids else "(leer)"
            print(f"     top3: {top3}")
            print(f"     expect: {', '.join(str(x) for x in expect)}")

    if total == 0:
        print("⚠️ Keine gültigen Fälle im Benchmark.")
        return 1

    print("\n=== Benchmark Summary ===")
    for k in ks:
        rate = hits[k] / max(1, total)
        print(f"Hit@{k}: {hits[k]}/{total} = {rate:.2%}")
    if missing:
        print(f"⚠️ Fehlende Aliase (nicht im Lexikon/Labels gefunden): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
