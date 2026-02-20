"""Phase 2: Generate custom CFI graph pairs and verify them.

Uses signal-based timeout per pair to avoid hanging on large graphs.
"""

import json
import signal
import time
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

WS = Path(__file__).resolve().parent
OUT_DIR = WS / "temp" / "datasets"

PAIR_TIMEOUT = 60  # seconds per pair


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Pair timed out")


def wl_color_refinement(G, max_iters=100):
    nodes = sorted(G.nodes())
    colors = {v: G.degree(v) for v in nodes}
    color_map = {}
    nc = max(colors.values()) + 1 if colors else 0
    for _ in range(max_iters):
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
    return sorted(colors.values())


def are_1wl_equivalent(G1, G2):
    return wl_color_refinement(G1) == wl_color_refinement(G2)


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
            tw = twist_edge_pair is not None and set(twist_edge_pair) == {u, v}
            if tw:
                G.add_edge(a_nodes[u][idx_u], b_nodes[v][idx_v])
                G.add_edge(b_nodes[u][idx_u], a_nodes[v][idx_v])
            else:
                G.add_edge(a_nodes[u][idx_u], a_nodes[v][idx_v])
                G.add_edge(b_nodes[u][idx_u], b_nodes[v][idx_v])
        return G

    if twist_edge is None:
        twist_edge = list(base.edges())[0]
    return build_single_cfi(None), build_single_cfi(twist_edge)


def build_record(G1, G2, pair_id, base_name):
    g1d = graph_to_dict(G1)
    g2d = graph_to_dict(G2)
    vt1 = is_vertex_transitive_wl(G1)
    vt2 = is_vertex_transitive_wl(G2)
    fv1 = features_vary(g1d["node_features"])
    fv2 = features_vary(g2d["node_features"])
    r1, r2 = is_regular(G1), is_regular(G2)
    return {
        "input": {"pair_id": pair_id, "graph_1": g1d, "graph_2": g2d},
        "output": {"is_isomorphic": False, "is_1wl_equivalent": True,
                    "wl_equivalence_level": "1-WL"},
        "metadata_fold": "test",
        "metadata": {
            "source": "Custom_CFI", "category": "Custom_CFI", "difficulty": "1-WL",
            "base_graph": base_name,
            "graph_1_vertex_transitive": vt1, "graph_2_vertex_transitive": vt2,
            "graph_1_features_vary": fv1, "graph_2_features_vary": fv2,
            "graph_1_is_regular": r1, "graph_2_is_regular": r2,
            "num_nodes_g1": G1.number_of_nodes(), "num_nodes_g2": G2.number_of_nodes(),
            "num_edges_g1": G1.number_of_edges(), "num_edges_g2": G2.number_of_edges(),
        },
    }


def main():
    t0 = time.time()
    np.random.seed(42)
    base_graphs = []

    # Small base graphs only (keep CFI output <= ~150 nodes)
    for n in [5, 6, 7, 8]:
        base_graphs.append((nx.cycle_graph(n), f"cycle_{n}"))
    for n in [5, 6, 8]:
        base_graphs.append((nx.path_graph(n), f"path_{n}"))
    for m, k in [(3, 3), (3, 4)]:
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(m, k))
        base_graphs.append((G, f"grid_{m}x{k}"))
    base_graphs.append((nx.petersen_graph(), "petersen"))
    for m, k in [(3, 3)]:
        base_graphs.append((nx.complete_bipartite_graph(m, k), f"K_{m}_{k}"))
    for n in [8, 10, 12]:
        try:
            G = nx.random_regular_graph(3, n, seed=42 + n)
            base_graphs.append((G, f"random_3reg_{n}"))
        except Exception:
            pass
    # Also some bigger cycles for diversity
    for n in [10, 12]:
        base_graphs.append((nx.cycle_graph(n), f"cycle_{n}"))
    # Wheel graphs
    for n in [5, 6, 7, 8]:
        base_graphs.append((nx.wheel_graph(n), f"wheel_{n}"))

    # Remove duplicates
    seen = set()
    unique = []
    for g, name in base_graphs:
        if name not in seen:
            seen.add(name)
            unique.append((g, name))
    base_graphs = unique

    records = []
    signal.signal(signal.SIGALRM, timeout_handler)

    print(f"Generating CFI pairs from {len(base_graphs)} base graphs...")
    for base_G, name in base_graphs:
        if not nx.is_connected(base_G):
            print(f"  Skip {name}: not connected"); continue

        signal.alarm(PAIR_TIMEOUT)
        try:
            G1, G2 = build_cfi_pair(base_G)
            n = G1.number_of_nodes()
            print(f"  {name}: CFI has {n} nodes, {G1.number_of_edges()} edges", flush=True)

            # Skip iso check for large graphs (>100 nodes) — CFI construction guarantees non-iso
            if n <= 100:
                if nx.is_isomorphic(G1, G2):
                    print(f"    WARNING: isomorphic! Skipping"); continue

            # WL check for small graphs only
            if n <= 80:
                if not are_1wl_equivalent(G1, G2):
                    print(f"    WARNING: NOT 1-WL equiv! Skipping"); continue

            rec = build_record(G1, G2, f"custom_cfi_{name}", name)
            records.append(rec)
            print(f"    OK ({n}n)", flush=True)

        except TimeoutError:
            print(f"  Skip {name}: timeout ({PAIR_TIMEOUT}s)")
        except Exception as e:
            print(f"  Error {name}: {e}")
        finally:
            signal.alarm(0)

    print(f"\nGenerated {len(records)} custom CFI pairs")

    cfi_path = OUT_DIR / "cfi_records.json"
    with open(cfi_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved to {cfi_path}")
    print(f"Size: {cfi_path.stat().st_size / 1024:.1f} KB")
    print(f"Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
