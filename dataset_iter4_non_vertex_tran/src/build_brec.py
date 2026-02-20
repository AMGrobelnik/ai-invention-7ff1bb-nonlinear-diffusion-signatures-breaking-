"""Phase 1: Parse all BREC .npy files and save as JSON (no CFI generation)."""

import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

WS = Path(__file__).resolve().parent
BREC_DIR = WS / "temp" / "datasets" / "brec_raw"
OUT_DIR = WS / "temp" / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def is_vertex_transitive_wl(G):
    nodes = sorted(G.nodes())
    if not nodes:
        return True
    colors = {v: G.degree(v) for v in nodes}
    color_map = {}
    nc = max(colors.values()) + 1
    for _ in range(min(len(nodes), 20)):
        new_c = {}
        for v in nodes:
            nb = tuple(sorted(colors[u] for u in G.neighbors(v)))
            key = (colors[v], nb)
            if key not in color_map:
                color_map[key] = nc; nc += 1
            new_c[v] = color_map[key]
        if new_c == colors:
            break
        colors = new_c
    return len(set(colors.values())) == 1


def compute_node_features(G):
    nodes = sorted(G.nodes())
    n = len(nodes)
    degrees = [G.degree(v) for v in nodes]
    cl = nx.clustering(G)
    cl_vals = [round(cl[v], 6) for v in nodes]
    if n <= 60:
        bt = nx.betweenness_centrality(G)
        bt_vals = [round(bt[v], 6) for v in nodes]
    else:
        bt_vals = [0.0] * n
    return {"degree": degrees, "clustering_coefficient": cl_vals,
            "betweenness_centrality": bt_vals}


def features_vary(features):
    for k in ["degree", "clustering_coefficient"]:
        if len(set(features[k])) > 1:
            return True
    bv = features["betweenness_centrality"]
    if any(v != 0.0 for v in bv) and len(set(bv)) > 1:
        return True
    return False


def is_regular(G):
    return len(set(d for _, d in G.degree())) == 1


def graph_to_dict(G):
    nodes = sorted(G.nodes())
    n = len(nodes)
    nm = {v: i for i, v in enumerate(nodes)}
    el = sorted([[nm[u], nm[v]] for u, v in G.edges()])
    result = {"num_nodes": n, "edge_list": el}
    if n <= 50:
        adj = [[0]*n for _ in range(n)]
        for u, v in G.edges():
            i, j = nm[u], nm[v]
            adj[i][j] = 1; adj[j][i] = 1
        result["adjacency_matrix"] = adj
    Gr = nx.relabel_nodes(G, nm)
    result["node_features"] = compute_node_features(Gr)
    return result


def to_bytes(val):
    return val if isinstance(val, bytes) else val.encode()


def parse_consecutive(path):
    data = np.load(path, allow_pickle=True)
    pairs = []
    for i in range(0, len(data), 2):
        g1, g2 = to_bytes(data[i]), to_bytes(data[i+1])
        pairs.append((nx.from_graph6_bytes(g1), nx.from_graph6_bytes(g2), g1, g2))
    return pairs


def parse_paired(path):
    data = np.load(path, allow_pickle=True)
    pairs = []
    for i in range(len(data)):
        g1, g2 = to_bytes(data[i][0]), to_bytes(data[i][1])
        pairs.append((nx.from_graph6_bytes(g1), nx.from_graph6_bytes(g2), g1, g2))
    return pairs


def build_record(G1, G2, pair_id, source, category, difficulty,
                 brec_idx=None, g6_1=None, g6_2=None):
    g1d = graph_to_dict(G1)
    g2d = graph_to_dict(G2)
    vt1 = is_vertex_transitive_wl(G1)
    vt2 = is_vertex_transitive_wl(G2)
    fv1 = features_vary(g1d["node_features"])
    fv2 = features_vary(g2d["node_features"])
    r1, r2 = is_regular(G1), is_regular(G2)
    rec = {
        "input": {"pair_id": pair_id, "graph_1": g1d, "graph_2": g2d},
        "output": {"is_isomorphic": False, "is_1wl_equivalent": True,
                    "wl_equivalence_level": difficulty},
        "metadata_fold": "test",
        "metadata": {
            "source": source, "category": category, "difficulty": difficulty,
            "graph_1_vertex_transitive": vt1, "graph_2_vertex_transitive": vt2,
            "graph_1_features_vary": fv1, "graph_2_features_vary": fv2,
            "graph_1_is_regular": r1, "graph_2_is_regular": r2,
            "num_nodes_g1": G1.number_of_nodes(),
            "num_nodes_g2": G2.number_of_nodes(),
            "num_edges_g1": G1.number_of_edges(),
            "num_edges_g2": G2.number_of_edges(),
        },
    }
    if brec_idx is not None:
        rec["metadata"]["brec_pair_index"] = brec_idx
    for label, g6 in [("graph6_g1", g6_1), ("graph6_g2", g6_2)]:
        if g6 is not None:
            try: rec["metadata"][label] = g6.decode("ascii", errors="replace")
            except: pass
    return rec


def main():
    t0 = time.time()
    records = []

    cats = [
        ("basic.npy", "consecutive", "Basic", "1-WL", 0),
        ("regular.npy", "paired", "Simple_Regular", "1-WL", 60),
        ("str.npy", "consecutive", "Strongly_Regular", "3-WL", 110),
        ("extension.npy", "paired", "Extension", "between-1WL-3WL", 160),
        ("cfi.npy", "paired", "CFI", None, 260),
        ("4vtx.npy", "consecutive", "4-Vertex_Condition", "4-vertex", 360),
        ("dr.npy", "consecutive", "Distance_Regular", "distance-regular", 380),
    ]

    for ci, (fn, pt, cat, diff, si) in enumerate(cats):
        print(f"[{ci+1}/7] {fn}...", flush=True)
        t1 = time.time()
        pairs = parse_consecutive(BREC_DIR / fn) if pt == "consecutive" else parse_paired(BREC_DIR / fn)
        for idx, (G1, G2, g1, g2) in enumerate(pairs):
            d = diff
            if cat == "CFI":
                d = "1-WL" if idx < 60 else ("3-WL" if idx < 80 else "4-WL")
            rec = build_record(G1, G2, f"brec_{cat.lower()}_{idx:03d}",
                               "BREC", cat, d, si + idx, g1, g2)
            records.append(rec)
        print(f"  {len(pairs)} pairs in {time.time()-t1:.1f}s", flush=True)

    # Save intermediate result
    brec_path = OUT_DIR / "brec_records.json"
    with open(brec_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} BREC records to {brec_path}")
    print(f"Size: {brec_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
