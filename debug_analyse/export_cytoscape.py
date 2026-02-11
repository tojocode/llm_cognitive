# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_core.engine import KognitivesModell  # noqa: E402


def _build_elements(
    engine: KognitivesModell,
    include_episodic: bool = True,
    max_nodes: int = 0,
    min_degree: int = 0,
    min_weight: float = 0.0,
):
    # Collect edges and degrees
    edges = []
    degree = {cid: 0 for cid in engine.konzepte.keys()}

    def add_edge(src: str, dst: str, typ: str, w: float, layer: str):
        if w < min_weight:
            return
        if src not in engine.konzepte or dst not in engine.konzepte:
            return
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1
        edges.append({
            "data": {
                "id": f"{src}::{typ}::{dst}::{layer}",
                "source": src,
                "target": dst,
                "type": typ,
                "weight": round(float(w), 4),
                "layer": layer,
            }
        })

    for src, k in engine.konzepte.items():
        for e in k.verbindungen:
            add_edge(src, e.ziel, e.typ, e.gewicht, "semantic")

    if include_episodic:
        for src, eds in engine.episodic_edges.items():
            for e in eds:
                add_edge(src, e.ziel, e.typ, e.gewicht, "episodic")

    # Node filtering
    nodes = []
    items = []
    for cid in engine.konzepte.keys():
        deg = degree.get(cid, 0)
        if deg < min_degree:
            continue
        items.append((deg, cid))

    items.sort(reverse=True)
    if max_nodes and max_nodes > 0:
        items = items[:max_nodes]
    keep = {cid for _, cid in items}

    for cid in keep:
        label = engine._label_for_output(cid)
        nodes.append({
            "data": {
                "id": cid,
                "label": label,
                "degree": degree.get(cid, 0),
            }
        })

    # Filter edges to kept nodes
    edges = [e for e in edges if e["data"]["source"] in keep and e["data"]["target"] in keep]

    return nodes + edges


def _html_template(elements_json: str) -> str:
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>MindGraph – Cytoscape</title>
    <script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
    <style>
      body {{
        margin: 0;
        font-family: "Segoe UI", Arial, sans-serif;
        background: #0b0c10;
        color: #e6edf3;
      }}
      #toolbar {{
        padding: 10px 12px;
        display: flex;
        gap: 12px;
        align-items: center;
        border-bottom: 1px solid #222;
        background: #11131a;
        flex-wrap: wrap;
      }}
      #toolbar input[type="text"] {{
        width: 240px;
        padding: 6px 8px;
        background: #0f1116;
        color: #e6edf3;
        border: 1px solid #2b2f3a;
        border-radius: 6px;
      }}
      #toolbar label {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 13px;
      }}
      #toolbar input[type="range"] {{
        width: 160px;
      }}
      #toolbar button {{
        padding: 6px 10px;
        background: #1f2937;
        color: #e6edf3;
        border: 1px solid #2b2f3a;
        border-radius: 6px;
        cursor: pointer;
      }}
      #toolbar button:hover {{
        background: #2b3647;
      }}
      #legend {{
        padding: 6px 12px;
        font-size: 12px;
        color: #9aa4b2;
      }}
      #legend span {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-right: 10px;
      }}
      #legend i {{
        display: inline-block;
        width: 12px;
        height: 2px;
        background: #64748b;
      }}
      #cy {{
        width: 100vw;
        height: calc(100vh - 84px);
        display: block;
      }}
    </style>
  </head>
  <body>
    <div id="toolbar">
      <strong>MindGraph</strong>
      <label>Suche <input id="search" type="text" placeholder="Knotenlabel..."></label>
      <label><input id="toggle-sem" type="checkbox" checked> Semantik</label>
      <label><input id="toggle-epi" type="checkbox" checked> Episodisch</label>
      <label>Min Weight <input id="weight" type="range" min="0" max="1" step="0.05" value="0"></label>
      <label><input id="toggle-community" type="checkbox" checked> Community-Farben</label>
      <button id="btn-cluster">Cluster</button>
      <button id="btn-reset">Reset Fokus</button>
    </div>
    <div id="legend"></div>
    <div id="cy"></div>
    <script>
      const elements = {elements_json};

      const cy = cytoscape({{
        container: document.getElementById("cy"),
        elements,
        layout: {{ name: "cose", animate: false, padding: 30 }},
        style: [
          {{
            selector: "node",
            style: {{
              "label": "data(label)",
              "background-color": "#3b82f6",
              "text-valign": "center",
              "text-halign": "center",
              "color": "#e6edf3",
              "font-size": 10,
              "width": "mapData(degree, 0, 20, 18, 38)",
              "height": "mapData(degree, 0, 20, 18, 38)",
              "border-width": 1,
              "border-color": "#1f2a44"
            }}
          }},
          {{
            selector: "edge",
            style: {{
              "width": "mapData(weight, 0, 1, 0.5, 3)",
              "line-color": "#64748b",
              "curve-style": "bezier",
              "target-arrow-shape": "triangle",
              "target-arrow-color": "#64748b",
              "opacity": 0.7
            }}
          }},
          {{
            selector: "edge[layer = 'episodic']",
            style: {{
              "line-style": "dashed"
            }}
          }},
          {{
            selector: ".match",
            style: {{
              "background-color": "#f97316",
              "border-color": "#fbbf24",
              "border-width": 2
            }}
          }},
          {{
            selector: ".faded",
            style: {{
              "opacity": 0.08,
              "text-opacity": 0.1
            }}
          }},
          {{
            selector: ".focus",
            style: {{
              "border-color": "#f59e0b",
              "border-width": 3
            }}
          }}
        ]
      }});

      const palette = [
        "#22c55e", "#3b82f6", "#f97316", "#eab308", "#a855f7",
        "#14b8a6", "#f43f5e", "#10b981", "#6366f1", "#0ea5e9",
        "#f59e0b", "#84cc16"
      ];

      function mapEdgeColors() {{
        const typeColors = {{}};
        let idx = 0;
        cy.edges().forEach(e => {{
          const t = (e.data("type") || "rel").toString();
          if (!typeColors[t]) {{
            typeColors[t] = palette[idx % palette.length];
            idx += 1;
          }}
          e.style("line-color", typeColors[t]);
          e.style("target-arrow-color", typeColors[t]);
        }});

        const legend = document.getElementById("legend");
        legend.innerHTML = "";
        Object.keys(typeColors).slice(0, 14).forEach(t => {{
          const span = document.createElement("span");
          const i = document.createElement("i");
          i.style.background = typeColors[t];
          span.appendChild(i);
          span.appendChild(document.createTextNode(t));
          legend.appendChild(span);
        }});
      }}

      function communityCluster(iterations=8) {{
        // Label Propagation
        cy.nodes().forEach(n => n.data("community", n.id()));
        for (let i=0; i<iterations; i++) {{
          cy.nodes().forEach(n => {{
            const counts = {{}};
            n.neighborhood("node").forEach(nb => {{
              const c = nb.data("community");
              counts[c] = (counts[c] || 0) + 1;
            }});
            let best = n.data("community");
            let bestCount = -1;
            Object.keys(counts).forEach(c => {{
              if (counts[c] > bestCount) {{
                bestCount = counts[c];
                best = c;
              }}
            }});
            n.data("community", best);
          }});
        }}
        const commColors = {{}};
        let idx = 0;
        cy.nodes().forEach(n => {{
          const c = n.data("community");
          if (!commColors[c]) {{
            commColors[c] = palette[idx % palette.length];
            idx += 1;
          }}
          n.data("commColor", commColors[c]);
        }});
        applyCommunityColors();
      }}

      function applyCommunityColors() {{
        const enabled = document.getElementById("toggle-community").checked;
        cy.nodes().forEach(n => {{
          const col = enabled ? (n.data("commColor") || "#3b82f6") : "#3b82f6";
          n.style("background-color", col);
        }});
      }}

      function resetFocus() {{
        cy.elements().removeClass("faded");
        cy.nodes().removeClass("focus");
      }}

      const search = document.getElementById("search");
      search.addEventListener("input", () => {{
        const q = search.value.trim().toLowerCase();
        cy.nodes().removeClass("match");
        if (!q) return;
        cy.nodes().filter(n => n.data("label").toLowerCase().includes(q)).addClass("match");
      }});

      cy.on("tap", "node", (evt) => {{
        const node = evt.target;
        const neighborhood = node.closedNeighborhood();
        cy.elements().addClass("faded");
        neighborhood.removeClass("faded");
        cy.nodes().removeClass("focus");
        node.addClass("focus");
      }});
      cy.on("tap", (evt) => {{
        if (evt.target === cy) {{
          resetFocus();
        }}
      }});

      const toggleSem = document.getElementById("toggle-sem");
      const toggleEpi = document.getElementById("toggle-epi");
      function updateLayers() {{
        cy.edges("[layer = 'semantic']").style("display", toggleSem.checked ? "element" : "none");
        cy.edges("[layer = 'episodic']").style("display", toggleEpi.checked ? "element" : "none");
      }}
      toggleSem.addEventListener("change", updateLayers);
      toggleEpi.addEventListener("change", updateLayers);
      updateLayers();

      const weight = document.getElementById("weight");
      function updateWeight() {{
        const minW = parseFloat(weight.value || "0");
        cy.edges().forEach(e => {{
          const w = e.data("weight") || 0;
          e.style("display", w >= minW ? "element" : "none");
        }});
      }}
      weight.addEventListener("input", updateWeight);
      updateWeight();

      document.getElementById("toggle-community").addEventListener("change", applyCommunityColors);
      document.getElementById("btn-cluster").addEventListener("click", () => {{
        communityCluster(10);
      }});
      document.getElementById("btn-reset").addEventListener("click", resetFocus);

      mapEdgeColors();
      communityCluster(8);
    </script>
  </body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MindGraph to Cytoscape HTML")
    parser.add_argument("--semantic", default="llm_memory/memory_semantic.jsonl")
    parser.add_argument("--episodic", default="llm_memory/memory_episodic.jsonl")
    parser.add_argument("--output", default="llm_export/cytoscape_view.html")
    parser.add_argument("--max-nodes", type=int, default=0, help="0 = alle")
    parser.add_argument("--min-degree", type=int, default=0)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--no-episodic", action="store_true")
    args = parser.parse_args()

    engine = KognitivesModell(args.semantic, episodic_datei=args.episodic)
    elements = _build_elements(
        engine,
        include_episodic=not args.no_episodic,
        max_nodes=args.max_nodes,
        min_degree=args.min_degree,
        min_weight=args.min_weight,
    )

    html = _html_template(json.dumps(elements, ensure_ascii=False))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"OK Export: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
