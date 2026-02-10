# -*- coding: utf-8 -*-
# ============================================================
# Wikipedia -> Memory Import (Semantic + Episodic)
# Datei: data_import/wikipedia_to_core.py
#
# Liest alle .txt Dateien aus data_import/wikipedia und importiert
# sie in llm_memory/memory_semantic.jsonl und llm_memory/memory_episodic.jsonl
# via KognitivesModell.import_text_file.
# Optional können einzelne Dateien per --files angegeben werden.
# Zusätzlich wird ein sauberes Lexikon aus Labels aufgebaut.
# ============================================================

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_core.engine import KognitivesModell  # noqa: E402


def _default_paths() -> tuple[Path, Path, Path, Path, Path]:
    wiki_dir = ROOT / "data_import" / "wikipedia"
    semantic = ROOT / "llm_memory" / "memory_semantic.jsonl"
    episodic = ROOT / "llm_memory" / "memory_episodic.jsonl"
    lexikon = ROOT / "llm_memory" / "lexikon.json"
    embeddings = ROOT / "llm_memory" / "embeddings.json"
    return wiki_dir, semantic, episodic, lexikon, embeddings


def _build_clean_lexikon(engine: KognitivesModell) -> tuple[dict, dict]:
    stop_first = {
        "bei", "in", "im", "am", "an", "von", "mit", "für", "fuer", "durch", "aus",
        "auf", "unter", "über", "ueber", "zwischen", "ohne", "als", "seit", "nach",
        "vor", "wegen", "gegen", "während", "waehrend", "wenn", "weil", "dass",
        "was", "wie", "warum", "wieso", "weshalb", "welche", "welcher", "welches",
    }
    stop_verbs = {
        "ist", "sind", "war", "wurde", "werden", "behandelt", "gilt", "hat", "haben",
        "beschreibt", "besteht", "gehört", "lebt", "kommt", "geht", "dient", "zeigt",
        "gibt", "führt", "steht", "liegt", "heißt", "heisst",
    }
    lex = {}
    stats = {"candidates": 0, "added": 0, "skipped": 0, "collisions": 0}

    def _add_alias(raw: str, cid: str) -> bool:
        alias = (raw or "").strip()
        if not alias:
            return False
        if "," in alias:
            alias = alias.split(",", 1)[0].strip()
        if "(" in alias:
            alias = alias.split("(", 1)[0].strip()
        if not alias:
            return False
        if "_" in alias:
            return False

        is_acronym = len(alias) == 2 and alias.isalpha() and alias.isupper()
        if (len(alias) < 3 and not is_acronym) or len(alias) > 60:
            return False
        if alias and not alias[0].isalpha():
            return False

        norm = engine._norm_label(alias)
        if not norm or norm.isdigit():
            return False

        tokens = norm.split()
        if not tokens or len(tokens) > 3:
            return False
        if tokens[0] in stop_first:
            return False
        if any(t in stop_verbs for t in tokens):
            return False
        if any(len(t) < 2 for t in tokens):
            return False

        if norm in lex:
            stats["collisions"] += 1
            return False

        lex[norm] = cid
        stats["added"] += 1
        return True

    for cid in sorted(engine.konzepte.keys()):
        labels = engine.konzepte[cid].labels or []
        uniq_labels = sorted(set(labels), key=lambda x: (len(x), x))
        for lab in uniq_labels:
            stats["candidates"] += 1
            if _add_alias(lab, cid):
                continue
            stats["skipped"] += 1

        # Zusatz: Alias aus Concept-ID (underscores -> spaces)
        if "_" in cid:
            alias_from_id = cid.replace("_", " ")
            ok = _add_alias(alias_from_id, cid)
            if ok:
                # Acronym aus Mehrwort-Alias (z.B. "Künstliche Intelligenz" -> "KI")
                parts = [p for p in alias_from_id.split(" ") if p]
                if 2 <= len(parts) <= 4:
                    acronym = "".join([p[0] for p in parts if p and p[0].isalpha()]).upper()
                    if 2 <= len(acronym) <= 4:
                        _add_alias(acronym, cid)

    return lex, stats


def _prune_engine(engine: KognitivesModell) -> dict:
    punct_start = set("-/\"'“”.,:;!?()[]{}")
    stop_first = {
        "bei", "in", "im", "am", "an", "von", "mit", "für", "fuer", "durch", "aus",
        "auf", "unter", "über", "ueber", "zwischen", "ohne", "als", "seit", "nach",
        "vor", "wegen", "gegen", "während", "waehrend", "wenn", "weil", "dass",
        "was", "wie", "warum", "wieso", "weshalb", "welche", "welcher", "welches",
    }

    def is_bad_id(cid: str) -> bool:
        if not cid:
            return True
        if cid[0] in punct_start:
            return True
        if cid.isdigit():
            return True
        if len(cid) < 3 and not (cid.isalpha() and cid.isupper()):
            return True
        if re.search(r"[^A-Za-zÄÖÜäöüß0-9_\-]", cid):
            return True
        base = re.split(r"[_\-]", cid.lower(), maxsplit=1)[0]
        if base in stop_first:
            return True
        return False

    # Incoming degree
    incoming = {cid: 0 for cid in engine.konzepte.keys()}
    for src, k in engine.konzepte.items():
        for e in k.verbindungen:
            incoming[e.ziel] = incoming.get(e.ziel, 0) + 1
    for src, edges in engine.episodic_edges.items():
        for e in edges:
            incoming[e.ziel] = incoming.get(e.ziel, 0) + 1

    # Label cleanup
    labels_removed = 0
    for cid, k in engine.konzepte.items():
        new_labels = []
        for lab in (k.labels or []):
            if not lab:
                labels_removed += 1
                continue
            if len(lab) > 80:
                labels_removed += 1
                continue
            if lab[0] in punct_start:
                labels_removed += 1
                continue
            if "\n" in lab or "\r" in lab:
                labels_removed += 1
                continue
            new_labels.append(lab)
        if not new_labels:
            new_labels = [cid]
        k.labels = list(dict.fromkeys(new_labels))

    # Node pruning
    to_drop = set()
    for cid, k in engine.konzepte.items():
        deg_out = len(k.verbindungen)
        deg_epi = len(engine.episodic_edges.get(cid, []))
        deg_in = incoming.get(cid, 0)
        degree = deg_out + deg_epi + deg_in

        base = re.split(r"[_\-]", cid.lower(), maxsplit=1)[0]
        if base in stop_first:
            to_drop.add(cid)
            continue

        if is_bad_id(cid) and degree <= 1:
            to_drop.add(cid)
            continue

        if degree == 0 and len((k.labels or [])) == 0:
            to_drop.add(cid)

    # Remove nodes
    for cid in to_drop:
        engine.konzepte.pop(cid, None)
        engine.episodic_edges.pop(cid, None)

    # Remove edges to pruned nodes
    edges_removed_sem = 0
    for cid, k in list(engine.konzepte.items()):
        new_edges = [e for e in k.verbindungen if e.ziel not in to_drop]
        edges_removed_sem += max(0, len(k.verbindungen) - len(new_edges))
        k.verbindungen = new_edges

    edges_removed_epi = 0
    for src, edges in list(engine.episodic_edges.items()):
        new_edges = [e for e in edges if e.ziel not in to_drop]
        edges_removed_epi += max(0, len(edges) - len(new_edges))
        engine.episodic_edges[src] = new_edges

    return {
        "nodes_removed": len(to_drop),
        "edges_removed_sem": edges_removed_sem,
        "edges_removed_epi": edges_removed_epi,
        "labels_removed": labels_removed,
    }


def main() -> int:
    wiki_dir, semantic, episodic, lexikon, embeddings = _default_paths()

    parser = argparse.ArgumentParser(
        description=(
            "Importiert Wikipedia-TXT Dateien in memory_semantic.jsonl "
            "und memory_episodic.jsonl"
        )
    )
    parser.add_argument("--wiki-dir", default=str(wiki_dir), help="Ordner mit .txt Dateien")
    parser.add_argument("--semantic", default=str(semantic), help="Pfad zu memory_semantic.jsonl")
    parser.add_argument("--episodic", default=str(episodic), help="Pfad zu memory_episodic.jsonl")
    parser.add_argument("--lexikon", default=str(lexikon), help="Pfad zu lexikon.json")
    parser.add_argument("--embeddings", default=str(embeddings), help="Pfad zu embeddings.json")
    parser.add_argument(
        "--target",
        choices=["semantic", "episodic", "both"],
        default="both",
        help="Ziel-Layer (default: both)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max. Anzahl Dateien (0 = alle)")
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Eine oder mehrere .txt Dateien (überschreibt --wiki-dir/--limit)",
    )
    parser.add_argument("--no-lexikon", action="store_true", help="Lexikon nicht neu aufbauen")
    parser.add_argument("--no-prune", action="store_true", help="Pruning nicht ausführen")
    args = parser.parse_args()

    if args.files:
        raw_files = [Path(p) for p in args.files]
        files = []
        for p in raw_files:
            if p.exists():
                files.append(p)
            else:
                print(f"⚠️ Datei nicht gefunden: {p}")
    else:
        wdir = Path(args.wiki_dir)
        if not wdir.exists():
            print(f"❌ Ordner nicht gefunden: {wdir}")
            return 1
        files = sorted(wdir.glob("*.txt"))
        if args.limit and args.limit > 0:
            files = files[: args.limit]

    if not files:
        if args.files:
            print("⚠️ Keine gültigen Dateien angegeben.")
        else:
            print(f"⚠️ Keine .txt Dateien in: {wdir}")
        return 1

    engine = KognitivesModell(
        args.semantic,
        episodic_datei=args.episodic,
        lexikon_datei=args.lexikon,
        embed_db_path=args.embeddings,
    )

    total_nodes = 0
    total_edges_sem = 0
    total_edges_epi = 0

    for p in files:
        try:
            if args.target == "both":
                stats_sem = engine.import_text_file(str(p), target="semantic")
                stats_epi = engine.import_text_file(str(p), target="episodic")
                total_nodes += stats_sem.get("nodes_added", 0)
                total_edges_sem += stats_sem.get("edges_added", 0)
                total_edges_epi += stats_epi.get("edges_added", 0)
                print(
                    f"✓ {p.name}: +{stats_sem['nodes_added']} nodes, "
                    f"+{stats_sem['edges_added']} edges (semantic), "
                    f"+{stats_epi['edges_added']} edges (episodic)"
                )
            else:
                stats = engine.import_text_file(str(p), target=args.target)
                total_nodes += stats.get("nodes_added", 0)
                if args.target == "semantic":
                    total_edges_sem += stats.get("edges_added", 0)
                else:
                    total_edges_epi += stats.get("edges_added", 0)
                print(
                    f"✓ {p.name}: +{stats['nodes_added']} nodes, "
                    f"+{stats['edges_added']} edges ({args.target})"
                )
        except Exception as e:
            print(f"❌ {p.name}: {e}")

    if not args.no_prune:
        pstats = _prune_engine(engine)
        print(
            "=== Prune: "
            f"-{pstats['nodes_removed']} nodes | "
            f"-{pstats['edges_removed_sem']} sem-edges | "
            f"-{pstats['edges_removed_epi']} epi-edges | "
            f"-{pstats['labels_removed']} labels ==="
        )

    engine.speichere_model(args.semantic, episodic_datei=args.episodic)
    if not args.no_lexikon:
        lex, lstats = _build_clean_lexikon(engine)
        engine.lexikon = lex
        engine.speichere_lexikon(args.lexikon)
        print(
            f"=== Lexikon: +{lstats['added']} Aliases | "
            f"{lstats['collisions']} Kollisionen | "
            f"{lstats['skipped']} verworfen ==="
        )
    engine.speichere_embeddings()

    print(
        "=== Fertig: "
        f"{len(files)} Dateien | +{total_nodes} nodes | "
        f"+{total_edges_sem} edges (semantic) | +{total_edges_epi} edges (episodic) ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
