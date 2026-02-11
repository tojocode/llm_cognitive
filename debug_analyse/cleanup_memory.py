# -*- coding: utf-8 -*-

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_core.engine import KognitivesModell  # noqa: E402


def _rebuild_lexikon(engine: KognitivesModell) -> None:
    lex = {}
    def _add_alias(cid: str, alias: str) -> None:
        a = engine._norm_label(alias)
        if not a or a in lex:
            return
        if len(a) < 2:
            return
        lex[a] = cid

    for cid, k in engine.konzepte.items():
        if engine._is_junk_concept_id(cid):
            continue
        labels = k.labels or [cid]
        for lab in labels:
            if not lab or engine._is_junk_concept_id(lab):
                continue
            _add_alias(cid, lab)

        # Zusatz: Alias aus Concept-ID (underscores -> spaces)
        if "_" in cid:
            alias_from_id = cid.replace("_", " ")
            _add_alias(cid, alias_from_id)
            # Acronym aus Mehrwort-Alias
            parts = [p for p in alias_from_id.split(" ") if p]
            if 2 <= len(parts) <= 4:
                acronym = "".join([p[0] for p in parts if p and p[0].isalpha()]).upper()
                if len(acronym) >= 2:
                    _add_alias(cid, acronym)
    engine.lexikon = lex


def _cleanup_engine(engine: KognitivesModell) -> dict:
    to_drop = {cid for cid in engine.konzepte if engine._is_junk_concept_id(cid)}
    edges_removed_sem = 0
    edges_removed_epi = 0

    for cid in to_drop:
        engine.konzepte.pop(cid, None)
        engine.episodic_edges.pop(cid, None)

    for cid, k in list(engine.konzepte.items()):
        new_edges = [e for e in k.verbindungen if e.ziel not in to_drop]
        edges_removed_sem += max(0, len(k.verbindungen) - len(new_edges))
        k.verbindungen = new_edges

    for src, edges in list(engine.episodic_edges.items()):
        new_edges = [e for e in edges if e.ziel not in to_drop]
        edges_removed_epi += max(0, len(edges) - len(new_edges))
        engine.episodic_edges[src] = new_edges

    for cid, k in engine.konzepte.items():
        labels = [lab for lab in (k.labels or []) if lab and not engine._is_junk_concept_id(lab)]
        k.labels = list(dict.fromkeys(labels)) if labels else [cid]

    _rebuild_lexikon(engine)
    return {
        "nodes_removed": len(to_drop),
        "edges_removed_sem": edges_removed_sem,
        "edges_removed_epi": edges_removed_epi,
    }


def _rebuild_embeddings(engine: KognitivesModell) -> None:
    if not engine.embed_db:
        return
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
            engine.embed_db.add(
                text,
                meta={"type": "edge", "src": src, "dst": e.ziel, "rel": e.typ},
            )
    engine.embed_db.save()


def main() -> int:
    engine = KognitivesModell()
    stats = _cleanup_engine(engine)
    engine.speichere_model(engine.semantic_datei, episodic_datei=engine.episodic_datei)
    engine.speichere_lexikon(engine.lexikon_datei)
    _rebuild_embeddings(engine)
    print(
        f"OK Cleanup: -{stats['nodes_removed']} nodes | "
        f"-{stats['edges_removed_sem']} sem-edges | "
        f"-{stats['edges_removed_epi']} epi-edges"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
