# -*- coding: utf-8 -*-

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_core.engine import KognitivesModell  # noqa: E402
from llm_memory.embedding_db import HashingEmbedder, cosine_sparse  # noqa: E402


AUG_TYPES = {"cooccur", "coactive", "similar", "cluster_of"}
STOP_TOKENS = {
    "der", "die", "das", "ein", "eine", "einen", "einem", "einer",
    "ist", "sind", "und", "oder", "zu", "im", "in", "am", "an", "von", "mit",
    "fuer", "für", "den", "dem", "des", "hat", "haben", "was", "wie", "warum",
    "wieso", "weshalb", "woraus", "womit", "wodurch", "wo", "wann", "wer",
    "wen", "wem", "wessen", "welche", "welcher", "welches", "welchen", "welchem",
    "auch", "aber", "denn", "dann", "daher", "deshalb", "daneben",
    "außerdem", "ausserdem", "hierbei", "dabei", "somit", "jedoch",
}


def _remove_augmented(engine: KognitivesModell) -> int:
    removed = 0
    # remove edges
    for cid, k in engine.konzepte.items():
        before = len(k.verbindungen)
        k.verbindungen = [e for e in k.verbindungen if e.typ not in AUG_TYPES]
        removed += max(0, before - len(k.verbindungen))
    for src, edges in list(engine.episodic_edges.items()):
        before = len(edges)
        engine.episodic_edges[src] = [e for e in edges if e.typ not in AUG_TYPES]
        removed += max(0, before - len(engine.episodic_edges[src]))

    # remove cluster nodes
    cluster_nodes = [cid for cid in engine.konzepte if cid.startswith("Cluster_")]
    for cid in cluster_nodes:
        engine.konzepte.pop(cid, None)
        engine.episodic_edges.pop(cid, None)
    # remove edges pointing to removed cluster nodes
    if cluster_nodes:
        cset = set(cluster_nodes)
        for cid, k in engine.konzepte.items():
            k.verbindungen = [e for e in k.verbindungen if e.ziel not in cset]
        for src, edges in list(engine.episodic_edges.items()):
            engine.episodic_edges[src] = [e for e in edges if e.ziel not in cset]

    return removed + len(cluster_nodes)


def _alias_index(engine: KognitivesModell, max_tokens: int = 3) -> Dict[str, List[Tuple[List[str], str]]]:
    idx: Dict[str, List[Tuple[List[str], str]]] = defaultdict(list)
    for alias, cid in engine.lexikon.items():
        if not alias or len(alias) < 3:
            continue
        parts = alias.split()
        if not parts or len(parts) > max_tokens:
            continue
        if parts[0] in STOP_TOKENS:
            continue
        if engine._is_junk_concept_id(cid):
            continue
        idx[parts[0]].append((parts, cid))
    return idx


def _sentence_concepts(
    tokens: List[str],
    idx: Dict[str, List[Tuple[List[str], str]]],
    max_nodes: int,
) -> List[str]:
    found: List[str] = []
    used = set()
    n = len(tokens)
    i = 0
    while i < n:
        key = tokens[i]
        matched = False
        for parts, cid in idx.get(key, []):
            ln = len(parts)
            if ln == 1:
                if cid not in used:
                    found.append(cid)
                    used.add(cid)
                matched = True
                continue
            if i + ln <= n and tokens[i:i + ln] == parts:
                if cid not in used:
                    found.append(cid)
                    used.add(cid)
                matched = True
        i += 1 if not matched else 1
        if max_nodes and len(found) >= max_nodes:
            break
    return found


def _iter_sentences(text: str) -> Iterable[str]:
    parts = re.split(r"[\r\n]+", text)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        for s in re.split(r"(?<=[\.!?])\s+", p):
            s = s.strip()
            if s:
                yield s


def _add_sem_edge(engine: KognitivesModell, src: str, dst: str, typ: str, weight: float) -> None:
    if src not in engine.konzepte or dst not in engine.konzepte:
        return
    k = engine.konzepte[src]
    for e in k.verbindungen:
        if e.ziel == dst and e.typ == typ:
            e.gewicht = max(e.gewicht, weight)
            return
    from llm_core.types import Verbindung
    k.verbindungen.append(Verbindung(ziel=dst, gewicht=weight, typ=typ))


def _add_epi_edge(engine: KognitivesModell, src: str, dst: str, typ: str, weight: float) -> None:
    if src not in engine.konzepte or dst not in engine.konzepte:
        return
    engine.episodic_edges.setdefault(src, [])
    for e in engine.episodic_edges[src]:
        if e.ziel == dst and e.typ == typ:
            e.gewicht = max(e.gewicht, weight)
            return
    from llm_core.types import Verbindung
    engine.episodic_edges[src].append(Verbindung(ziel=dst, gewicht=weight, typ=typ))


def _add_cooccurrence(
    engine: KognitivesModell,
    wiki_dir: Path,
    max_tokens: int,
    max_nodes_per_sentence: int,
    min_count: int,
    weight_base: float,
    weight_scale: float,
) -> int:
    idx = _alias_index(engine, max_tokens=max_tokens)
    counts: Dict[Tuple[str, str], int] = Counter()

    for p in sorted(wiki_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        for s in _iter_sentences(text):
            tokens = engine._tokenize(s)
            if not tokens:
                continue
            cids = _sentence_concepts(tokens, idx, max_nodes_per_sentence)
            if len(cids) < 2:
                continue
            cids = list(dict.fromkeys(cids))
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    a, b = sorted((cids[i], cids[j]))
                    counts[(a, b)] += 1

    added = 0
    for (a, b), cnt in counts.items():
        if cnt < min_count:
            continue
        w = min(0.35, weight_base + weight_scale * cnt)
        _add_sem_edge(engine, a, b, "cooccur", w)
        _add_sem_edge(engine, b, a, "cooccur", w)
        added += 2
    return added


def _add_coactivation(
    engine: KognitivesModell,
    wiki_dir: Path,
    max_tokens: int,
    max_nodes_per_sentence: int,
    min_count: int,
    weight_base: float,
    weight_scale: float,
) -> int:
    idx = _alias_index(engine, max_tokens=max_tokens)
    counts: Dict[Tuple[str, str], int] = Counter()

    for p in sorted(wiki_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        prev = []
        for s in _iter_sentences(text):
            tokens = engine._tokenize(s)
            if not tokens:
                continue
            cur = _sentence_concepts(tokens, idx, max_nodes_per_sentence)
            if prev and cur:
                for a in prev:
                    for b in cur:
                        if a == b:
                            continue
                        key = (a, b)
                        counts[key] += 1
            prev = cur

    added = 0
    for (a, b), cnt in counts.items():
        if cnt < min_count:
            continue
        w = min(0.3, weight_base + weight_scale * cnt)
        _add_epi_edge(engine, a, b, "coactive", w)
        added += 1
    return added


def _add_similarity(
    engine: KognitivesModell,
    min_score: float,
    top_k: int,
    weight_scale: float,
) -> int:
    embedder = HashingEmbedder(dim=512)
    vectors: Dict[str, Dict[int, float]] = {}
    labels: Dict[str, str] = {}
    for cid, k in engine.konzepte.items():
        if cid.startswith("Cluster_"):
            continue
        label = engine._label_for_output(cid)
        labels[cid] = label
        vectors[cid] = embedder.embed_sparse(label)

    ids = list(vectors.keys())
    added = 0
    for i, a in enumerate(ids):
        sims: List[Tuple[float, str]] = []
        va = vectors[a]
        if not va:
            continue
        for b in ids[i + 1:]:
            vb = vectors[b]
            if not vb:
                continue
            s = cosine_sparse(va, vb)
            if s >= min_score:
                sims.append((s, b))
        sims.sort(key=lambda x: x[0], reverse=True)
        for s, b in sims[: max(1, top_k)]:
            w = min(0.35, s * weight_scale)
            _add_sem_edge(engine, a, b, "similar", w)
            _add_sem_edge(engine, b, a, "similar", w)
            added += 2
    return added


def _add_clusters(engine: KognitivesModell, min_cluster_size: int, max_clusters: int) -> int:
    # Simple label propagation on semantic edges (excluding augmented)
    nodes = [cid for cid in engine.konzepte if not cid.startswith("Cluster_")]
    label = {cid: cid for cid in nodes}

    neighbors = {cid: set() for cid in nodes}
    for cid, k in engine.konzepte.items():
        if cid not in neighbors:
            continue
        for e in k.verbindungen:
            if e.typ in AUG_TYPES:
                continue
            if e.ziel in neighbors:
                neighbors[cid].add(e.ziel)
                neighbors[e.ziel].add(cid)

    for _ in range(6):
        for cid in nodes:
            neigh = neighbors.get(cid) or set()
            if not neigh:
                continue
            counts = Counter(label[n] for n in neigh)
            label[cid] = counts.most_common(1)[0][0]

    groups: Dict[str, List[str]] = defaultdict(list)
    for cid, lab in label.items():
        groups[lab].append(cid)
    # keep only sizable clusters
    clusters = [g for g in groups.values() if len(g) >= min_cluster_size]
    clusters.sort(key=len, reverse=True)
    if max_clusters and len(clusters) > max_clusters:
        clusters = clusters[:max_clusters]

    added = 0
    for i, members in enumerate(clusters, 1):
        cluster_id = f"Cluster_{i}"
        if cluster_id not in engine.konzepte:
            from llm_core.types import Konzept
            engine.konzepte[cluster_id] = Konzept(
                id=cluster_id,
                labels=[f"Cluster {i}"],
                semantische_features=["cluster"],
            )
        for m in members:
            _add_sem_edge(engine, m, cluster_id, "cluster_of", 0.08)
            added += 1
    return added


def run_augmentation(
    semantic: str = "llm_memory/memory_semantic.jsonl",
    episodic: str = "llm_memory/memory_episodic.jsonl",
    wiki_dir: str = "data_import/wikipedia",
    *,
    co_min_count: int = 2,
    co_max_tokens: int = 3,
    co_max_nodes: int = 6,
    co_weight_base: float = 0.05,
    co_weight_scale: float = 0.02,
    ca_min_count: int = 2,
    ca_weight_base: float = 0.04,
    ca_weight_scale: float = 0.02,
    sim_min: float = 0.78,
    sim_topk: int = 3,
    sim_weight_scale: float = 0.35,
    cluster_min_size: int = 6,
    cluster_max: int = 20,
) -> dict:
    engine = KognitivesModell(semantic, episodic_datei=episodic)

    removed = _remove_augmented(engine)

    wdir = Path(wiki_dir)
    if not wdir.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {wdir}")

    co = _add_cooccurrence(
        engine,
        wdir,
        max_tokens=co_max_tokens,
        max_nodes_per_sentence=co_max_nodes,
        min_count=co_min_count,
        weight_base=co_weight_base,
        weight_scale=co_weight_scale,
    )
    ca = _add_coactivation(
        engine,
        wdir,
        max_tokens=co_max_tokens,
        max_nodes_per_sentence=co_max_nodes,
        min_count=ca_min_count,
        weight_base=ca_weight_base,
        weight_scale=ca_weight_scale,
    )
    sim = _add_similarity(
        engine,
        min_score=sim_min,
        top_k=sim_topk,
        weight_scale=sim_weight_scale,
    )
    clu = _add_clusters(
        engine,
        min_cluster_size=cluster_min_size,
        max_clusters=cluster_max,
    )

    engine.speichere_model(semantic, episodic_datei=episodic)
    return {
        "removed": removed,
        "cooccur": co,
        "coactive": ca,
        "similar": sim,
        "cluster": clu,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Augment MindGraph with structured edges")
    parser.add_argument("--semantic", default="llm_memory/memory_semantic.jsonl")
    parser.add_argument("--episodic", default="llm_memory/memory_episodic.jsonl")
    parser.add_argument("--wiki-dir", default="data_import/wikipedia")

    # co-occurrence
    parser.add_argument("--co-min-count", type=int, default=2)
    parser.add_argument("--co-max-tokens", type=int, default=3)
    parser.add_argument("--co-max-nodes", type=int, default=6)
    parser.add_argument("--co-weight-base", type=float, default=0.05)
    parser.add_argument("--co-weight-scale", type=float, default=0.02)

    # co-activation
    parser.add_argument("--ca-min-count", type=int, default=2)
    parser.add_argument("--ca-weight-base", type=float, default=0.04)
    parser.add_argument("--ca-weight-scale", type=float, default=0.02)

    # similarity
    parser.add_argument("--sim-min", type=float, default=0.78)
    parser.add_argument("--sim-topk", type=int, default=3)
    parser.add_argument("--sim-weight-scale", type=float, default=0.35)

    # clusters
    parser.add_argument("--cluster-min-size", type=int, default=6)
    parser.add_argument("--cluster-max", type=int, default=20)

    args = parser.parse_args()

    try:
        stats = run_augmentation(
            semantic=args.semantic,
            episodic=args.episodic,
            wiki_dir=args.wiki_dir,
            co_min_count=args.co_min_count,
            co_max_tokens=args.co_max_tokens,
            co_max_nodes=args.co_max_nodes,
            co_weight_base=args.co_weight_base,
            co_weight_scale=args.co_weight_scale,
            ca_min_count=args.ca_min_count,
            ca_weight_base=args.ca_weight_base,
            ca_weight_scale=args.ca_weight_scale,
            sim_min=args.sim_min,
            sim_topk=args.sim_topk,
            sim_weight_scale=args.sim_weight_scale,
            cluster_min_size=args.cluster_min_size,
            cluster_max=args.cluster_max,
        )
    except FileNotFoundError as e:
        print(f"FEHLER {e}")
        return 1
    print(
        "OK Augment: "
        f"-removed {stats['removed']} | +cooccur {stats['cooccur']} | "
        f"+coactive {stats['coactive']} | +similar {stats['similar']} | "
        f"+cluster {stats['cluster']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
