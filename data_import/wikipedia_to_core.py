# -*- coding: utf-8 -*-
# ============================================================
# Wikipedia -> Memory Import (Semantic + Episodic)
# Datei: data_import/wikipedia_to_core.py
#
# Liest alle .txt Dateien aus data_import/wikipedia und importiert
# sie in data/memory_semantic.jsonl und data/memory_episodic.jsonl
# via KognitivesModell.import_text_file.
# Optional können einzelne Dateien per --files angegeben werden.
# ============================================================

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_core.engine import KognitivesModell


def _default_paths() -> tuple[Path, Path, Path, Path, Path]:
    wiki_dir = ROOT / "data_import" / "wikipedia"
    semantic = ROOT / "data" / "memory_semantic.jsonl"
    episodic = ROOT / "data" / "memory_episodic.jsonl"
    lexikon = ROOT / "data" / "lexikon.json"
    embeddings = ROOT / "data" / "embeddings.json"
    return wiki_dir, semantic, episodic, lexikon, embeddings


def main() -> int:
    wiki_dir, semantic, episodic, lexikon, embeddings = _default_paths()

    parser = argparse.ArgumentParser(
        description="Importiert Wikipedia-TXT Dateien in memory_semantic.jsonl und memory_episodic.jsonl"
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
                print(f"✓ {p.name}: +{stats['nodes_added']} nodes, +{stats['edges_added']} edges ({args.target})")
        except Exception as e:
            print(f"❌ {p.name}: {e}")

    engine.speichere_model(args.semantic, episodic_datei=args.episodic)
    engine.speichere_lexikon(args.lexikon)
    engine.speichere_embeddings()

    print(
        "=== Fertig: "
        f"{len(files)} Dateien | +{total_nodes} nodes | "
        f"+{total_edges_sem} edges (semantic) | +{total_edges_epi} edges (episodic) ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
