"""
Build Non-Vertex-Transitive 1-WL-Equivalent Graph Pairs + BREC Benchmark Dataset.

Optimized: skips expensive operations for large graphs (>60 nodes).
"""

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

WS = Path(__file__).resolve().parent
BREC_DIR = WS / "temp" / "datasets" / "brec_raw"
OUT_DIR = WS / "temp" / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def wl_color_refinement(G, max_iters=100):
    """Run 1-WL color refinement. Returns sorted multiset of stable colors."""
    nodes = sorted(G.nodes())
    colors = {v: G.degree(v) for v in nodes}
    color_map = {}
    next_color = max(colors.values()) + 1 if colors else 0
    for _ in range(max_iters):
        new_colors = {}
        for v in nodes:
            nc = tuple(sorted(colors[u] for u in G.neighbors(v)))
            key = (colors[v], nc)
            if key not in color_map:
                color_map[key] = next_color
                next_color += 1
            new_colors[v] = color_map[key]
        if new_colors == colors:
            break
        colors = new_colors
    return sorted(colors.values())


def are_1wl_equivalent(G1, G2):
    return wl_color_refinement(G1) == wl_color_refinement(G2)


def is_vertex_transitive_wl(G):
    """WL proxy for vertex-transitivity."""
    nodes = sorted(G.nodes())
    if not nodes:
        return True
    colors = {v: G.degree(v) for v in nodes}
    color_map = {}
    next_color = max(colors.values()) + 1
    for _ in range(min(len(nodes), 20)):
        new_colors = {}
        for v in nodes:
            nc = tuple(sorted(colors[u] for u in G.neighbors(v)))
            key = (colors[v], nc)
            if key not in color_map:
                color_map[key] = next_color
                next_color += 1
            new_colors[v] = color_map[key]
        if new_colors == colors:
            break
        colors = new_colors
    return len(set(colors.values())) == 1


def compute_node_features(G):
    """Compute node features. Skip betweenness for large graphs (>60 nodes)."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    degrees = [G.degree(v) for v in nodes]
    clustering = nx.clustering(G)
    clustering_vals = [round(clustering[v], 6) for v in nodes]
    if n <= 60:
        betweenness = nx.betweenness_centrality(G)
        betweenness_vals = [round(betweenness[v], 6) for v in nodes]
    else:
        betweenness_vals = [0.0] * n
    return {
        "degree": degrees,
        "clustering_coefficient": clustering_vals,
        "betweenness_centrality": betweenness_vals,
    }


def features_vary(features):
    for key in ["degree", "clustering_coefficient"]:
        if len(set(features[key])) > 1:
            return True
    bv = features["betweenness_centrality"]
    if any(v != 0.0 for v in bv) and len(set(bv)) > 1:
        return True
    return False


def is_regular(G):
    degrees = [d for _, d in G.degree()]
    return len(set(degrees)) == 1


def graph_to_dict(G, include_adj=True):
    nodes = sorted(G.nodes())
    n = len(nodes)
    node_map = {v: i for i, v in enumerate(nodes)}
    edge_list = sorted([[node_map[u], node_map[v]] for u, v in G.edges()])
    result = {"num_nodes": n, "edge_list": edge_list}
    if include_adj and n <= 50:
        adj = [[0] * n for _ in range(n)]
        for u, v in G.edges():
            i, j = node_map[u], node_map[v]
            adj[i][j] = 1
            adj[j][i] = 1
        result["adjacency_matrix"] = adj
    G_r = nx.relabel_nodes(G, node_map)
    result["node_features"] = compute_node_features(G_r)
    return result


def to_bytes(val):
    if isinstance(val, bytes):
        return val
    if isinstance(val, str):
        return val.encode()
    return val


def parse_consecutive_pairs(path):
    data = np.load(path, allow_pickle=True)
    pairs = []
    for i in range(0, len(data), 2):
        g6_1, g6_2 = to_bytes(data[i]), to_bytes(data[i + 1])
        G1 = nx.from_graph6_bytes(g6_1)
        G2 = nx.from_graph6_bytes(g6_2)
        pairs.append((G1, G2, g6_1, g6_2))
    return pairs


def parse_paired_array(path):
    data = np.load(path, allow_pickle=True)
    pairs = []
    for i in range(len(data)):
        g6_1, g6_2 = to_bytes(data[i][0]), to_bytes(data[i][1])
        G1 = nx.from_graph6_bytes(g6_1)
        G2 = nx.from_graph6_bytes(g6_2)
        pairs.append((G1, G2, g6_1, g6_2))
    return pairs


def build_cfi_pair(base_graph, twist_edge=None):
    base = base_graph.copy()
    nodes_base = sorted(base.nodes())
    incident_edges = {v: sorted(base.edges(v)) for v in nodes_base}

    def even_parity_subsets(d):
        result = []
        for r in range(0, d + 1, 2):
            for combo in combinations(range(d), r):
                result.append(frozenset(combo))
        return result

    def build_single_cfi(twist_edge_pair=None):
        G = nx.Graph()
        nc = 0
        a_nodes = {}
        b_nodes = {}
        for v in nodes_base:
            d = base.degree(v)
            a_nodes[v] = {}
            b_nodes[v] = {}
            for i in range(d):
                a_nodes[v][i] = nc; G.add_node(nc); nc += 1
            for i in range(d):
                b_nodes[v][i] = nc; G.add_node(nc); nc += 1
            for S in even_parity_subsets(d):
                mid_id = nc; G.add_node(nc); nc += 1
                for i in range(d):
                    G.add_edge(mid_id, a_nodes[v][i] if i in S else b_nodes[v][i])
        for u, v in base.edges():
            idx_u = next(j for j, e in enumerate(incident_edges[u]) if set(e) == {u, v})
            idx_v = next(j for j, e in enumerate(incident_edges[v]) if set(e) == {u, v})
            is_tw = twist_edge_pair is not None and set(twist_edge_pair) == {u, v}
            if is_tw:
                G.add_edge(a_nodes[u][idx_u], b_nodes[v][idx_v])
                G.add_edge(b_nodes[u][idx_u], a_nodes[v][idx_v])
            else:
                G.add_edge(a_nodes[u][idx_u], a_nodes[v][idx_v])
                G.add_edge(b_nodes[u][idx_u], b_nodes[v][idx_v])
        return G

    if twist_edge is None:
        twist_edge = list(base.edges())[0]
    return build_single_cfi(None), build_single_cfi(twist_edge)


def generate_custom_cfi_pairs():
    pairs = []
    np.random.seed(42)
    base_graphs = []
    for n in [5, 6, 7, 8, 10, 12]:
        base_graphs.append((nx.cycle_graph(n), f"cycle_{n}"))
    for n in [5, 6, 8]:
        base_graphs.append((nx.path_graph(n), f"path_{n}"))
    for m, k in [(3, 3), (3, 4), (4, 4)]:
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(m, k))
        base_graphs.append((G, f"grid_{m}x{k}"))
    base_graphs.append((nx.petersen_graph(), "petersen"))
    for m, k in [(3, 3), (4, 4)]:
        base_graphs.append((nx.complete_bipartite_graph(m, k), f"K_{m}_{k}"))
    for n in [8, 10, 12, 14]:
        try:
            G = nx.random_regular_graph(3, n, seed=42 + n)
            base_graphs.append((G, f"random_3reg_{n}"))
        except Exception:
            pass

    print(f"  Generating from {len(base_graphs)} base graphs...")
    for base_G, name in base_graphs:
        if not nx.is_connected(base_G):
            print(f"    Skip {name}: not connected"); continue
        try:
            G1, G2 = build_cfi_pair(base_G)
            if nx.is_isomorphic(G1, G2):
                print(f"    Skip {name}: isomorphic"); continue
            if G1.number_of_nodes() <= 100:
                if not are_1wl_equivalent(G1, G2):
                    print(f"    Skip {name}: NOT 1-WL equiv"); continue
            pairs.append((G1, G2, name))
            print(f"    {name}: ({G1.number_of_nodes()}n,{G1.number_of_edges()}e) OK")
        except Exception as e:
            print(f"    Error {name}: {e}")
    print(f"  Generated {len(pairs)} custom CFI pairs")
    return pairs


def build_pair_record(G1, G2, pair_id, source, category, difficulty,
                      brec_pair_index=None, g6_1=None, g6_2=None):
    n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
    g1_dict = graph_to_dict(G1, include_adj=(n1 <= 50))
    g2_dict = graph_to_dict(G2, include_adj=(n2 <= 50))
    vt1 = is_vertex_transitive_wl(G1)
    vt2 = is_vertex_transitive_wl(G2)
    fv1 = features_vary(g1_dict["node_features"])
    fv2 = features_vary(g2_dict["node_features"])
    reg1, reg2 = is_regular(G1), is_regular(G2)

    record = {
        "input": {"pair_id": pair_id, "graph_1": g1_dict, "graph_2": g2_dict},
        "output": {"is_isomorphic": False, "is_1wl_equivalent": True,
                    "wl_equivalence_level": difficulty},
        "metadata_fold": "test",
        "metadata": {
            "source": source, "category": category, "difficulty": difficulty,
            "graph_1_vertex_transitive": vt1, "graph_2_vertex_transitive": vt2,
            "graph_1_features_vary": fv1, "graph_2_features_vary": fv2,
            "graph_1_is_regular": reg1, "graph_2_is_regular": reg2,
            "num_nodes_g1": n1, "num_nodes_g2": n2,
            "num_edges_g1": G1.number_of_edges(),
            "num_edges_g2": G2.number_of_edges(),
        },
    }
    if brec_pair_index is not None:
        record["metadata"]["brec_pair_index"] = brec_pair_index
    if g6_1 is not None:
        try: record["metadata"]["graph6_g1"] = g6_1.decode("ascii", errors="replace")
        except Exception: pass
    if g6_2 is not None:
        try: record["metadata"]["graph6_g2"] = g6_2.decode("ascii", errors="replace")
        except Exception: pass
    return record


def main():
    t0 = time.time()
    all_records = []

    print("=" * 60)
    print("Phase 1: Parsing BREC .npy files")
    print("=" * 60)

    categories = [
        ("basic.npy", "consecutive", "Basic", "1-WL", 0),
        ("regular.npy", "paired", "Simple_Regular", "1-WL", 60),
        ("str.npy", "consecutive", "Strongly_Regular", "3-WL", 110),
        ("extension.npy", "paired", "Extension", "between-1WL-3WL", 160),
        ("cfi.npy", "paired", "CFI", None, 260),
        ("4vtx.npy", "consecutive", "4-Vertex_Condition", "4-vertex", 360),
        ("dr.npy", "consecutive", "Distance_Regular", "distance-regular", 380),
    ]

    for ci, (fname, parse_type, category, difficulty, start_idx) in enumerate(categories):
        path = BREC_DIR / fname
        print(f"\n[{ci + 1}/7] {fname}...", flush=True)
        t1 = time.time()
        pairs = parse_consecutive_pairs(path) if parse_type == "consecutive" else parse_paired_array(path)
        for idx, (G1, G2, g6_1, g6_2) in enumerate(pairs):
            if category == "CFI":
                diff = "1-WL" if idx < 60 else ("3-WL" if idx < 80 else "4-WL")
            else:
                diff = difficulty
            rec = build_pair_record(
                G1, G2,
                pair_id=f"brec_{category.lower()}_{idx:03d}",
                source="BREC", category=category, difficulty=diff,
                brec_pair_index=start_idx + idx, g6_1=g6_1, g6_2=g6_2,
            )
            all_records.append(rec)
            if (idx + 1) % 10 == 0:
                print(f"    {idx + 1}/{len(pairs)}", flush=True)
        print(f"  {len(pairs)} pairs in {time.time() - t1:.1f}s", flush=True)

    print(f"\n  Total BREC: {len(all_records)}")

    print("\n" + "=" * 60)
    print("Phase 2: Custom CFI pairs")
    print("=" * 60)
    cfi_custom = generate_custom_cfi_pairs()
    for idx, (G1, G2, base_name) in enumerate(cfi_custom):
        rec = build_pair_record(
            G1, G2, pair_id=f"custom_cfi_{base_name}",
            source="Custom_CFI", category="Custom_CFI", difficulty="1-WL",
        )
        rec["metadata"]["base_graph"] = base_name
        all_records.append(rec)

    total = len(all_records)
    print(f"\n  Total pairs: {total}")

    print("\n" + "=" * 60)
    print("Phase 3: Summary")
    print("=" * 60)
    cat_counts = {}
    nvt = fv = 0
    for r in all_records:
        c = r["metadata"]["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
        if not r["metadata"]["graph_1_vertex_transitive"] or not r["metadata"]["graph_2_vertex_transitive"]:
            nvt += 1
        if r["metadata"]["graph_1_features_vary"] or r["metadata"]["graph_2_features_vary"]:
            fv += 1
    for c, n in sorted(cat_counts.items()):
        print(f"  {c}: {n}")
    print(f"  Non-vertex-transitive: {nvt}")
    print(f"  Feature-varying: {fv}")

    print("\n" + "=" * 60)
    print("Phase 4: Saving")
    print("=" * 60)

    full_path = OUT_DIR / "data_out.json"
    with open(full_path, "w") as f:
        json.dump(all_records, f, indent=2)
    fs = full_path.stat().st_size / (1024 * 1024)
    print(f"  data_out.json: {total} records, {fs:.1f} MB")

    mini = []
    limits = {"Basic": 20, "Extension": 10, "CFI": 10, "Custom_CFI": 10}
    cc = {}
    for r in all_records:
        c = r["metadata"]["category"]
        lim = limits.get(c, 0)
        if lim > 0 and cc.get(c, 0) < lim:
            mini.append(r)
            cc[c] = cc.get(c, 0) + 1
    mp = OUT_DIR / "data_out_mini.json"
    with open(mp, "w") as f:
        json.dump(mini, f, indent=2)
    print(f"  data_out_mini.json: {len(mini)} records, {mp.stat().st_size / (1024 * 1024):.1f} MB")

    prev = []
    for r in all_records[:5]:
        pr = json.loads(json.dumps(r))
        for gk in ["graph_1", "graph_2"]:
            g = pr["input"][gk]
            if "adjacency_matrix" in g:
                g["adjacency_matrix"] = g["adjacency_matrix"][:5]
            for fk in g["node_features"]:
                g["node_features"][fk] = g["node_features"][fk][:10]
        prev.append(pr)
    pp = OUT_DIR / "data_out_preview.json"
    with open(pp, "w") as f:
        json.dump(prev, f, indent=2)
    print(f"  data_out_preview.json: {len(prev)} records, {pp.stat().st_size / 1024:.1f} KB")

    print(f"\n  Total time: {time.time() - t0:.1f}s")
    if fs > 50:
        print(f"  WARNING: data_out.json is {fs:.1f}MB (>50MB)")
    print("\nDone!")


if __name__ == "__main__":
    main()
