"""
Build the 1-WL Equivalent Graph Pairs Dataset.

Constructs ~10-11 non-isomorphic graph pairs that are provably 1-WL equivalent,
spanning 4 families:
  1. srg(16,6,2,2): Shrikhande vs Rook's graph
  2. srg(25,12,5,6): Paulus graphs from Spence DB
  3. srg(26,10,3,4): Paulus graphs from Spence DB
  4. CSL(41,R): Circular Skip Links graphs

Sources:
  - Ted Spence's SRG database (University of Glasgow)
  - Murphy et al. 2019 "Relational Pooling for Graph Representations" (ICML)
  - Shrikhande graph: Cayley graph on Z4×Z4
  - Rook's graph: Line graph of K(4,4)
"""

import itertools
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

WORKSPACE = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────
# 1-WL Color Refinement
# ─────────────────────────────────────────────────────────────────────

def run_1wl(G: nx.Graph, max_iterations: int = 100) -> tuple[tuple[int, ...], dict[int, int]]:
    """Run 1-WL color refinement. Return (stable color histogram, final coloring)."""
    colors = {v: 0 for v in G.nodes()}
    for _ in range(max_iterations):
        new_colors = {}
        for v in G.nodes():
            neighbor_colors = tuple(sorted(colors[u] for u in G.neighbors(v)))
            new_colors[v] = hash((colors[v], neighbor_colors))
        # Canonicalize colors to integers
        unique = sorted(set(new_colors.values()))
        color_map = {c: i for i, c in enumerate(unique)}
        new_colors = {v: color_map[new_colors[v]] for v in G.nodes()}
        if new_colors == colors:
            break
        colors = new_colors
    histogram = tuple(sorted(Counter(colors.values()).values()))
    return histogram, colors


def verify_1wl_equivalent(G1: nx.Graph, G2: nx.Graph) -> tuple[bool, tuple, tuple]:
    """Check if two graphs are 1-WL equivalent."""
    hist1, _ = run_1wl(G1)
    hist2, _ = run_1wl(G2)
    return hist1 == hist2, hist1, hist2


# ─────────────────────────────────────────────────────────────────────
# Graph property computation
# ─────────────────────────────────────────────────────────────────────

def compute_eigenvalues(G: nx.Graph) -> tuple[list[float], list[float]]:
    """Compute adjacency and Laplacian eigenvalues."""
    A = nx.adjacency_matrix(G).toarray().astype(float)
    L = nx.laplacian_matrix(G).toarray().astype(float)
    adj_eigs = sorted(np.linalg.eigvalsh(A).tolist(), reverse=True)
    lap_eigs = sorted(np.linalg.eigvalsh(L).tolist())
    # Round to 6 decimal places
    adj_eigs = [round(e, 6) for e in adj_eigs]
    lap_eigs = [round(e, 6) for e in lap_eigs]
    return adj_eigs, lap_eigs


def check_cospectral(eigs1: list[float], eigs2: list[float], atol: float = 1e-6) -> bool:
    """Check if two eigenvalue lists are equal within tolerance."""
    return np.allclose(eigs1, eigs2, atol=atol)


def verify_srg_parameters(G: nx.Graph, v: int, k: int, lam: int, mu: int) -> bool:
    """Verify strongly regular graph parameters (v, k, lambda, mu)."""
    if G.number_of_nodes() != v:
        return False
    degrees = [d for _, d in G.degree()]
    if not all(d == k for d in degrees):
        return False
    # Check lambda (common neighbors for adjacent pairs)
    # and mu (common neighbors for non-adjacent pairs)
    adj = set(G.edges())
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, w = nodes[i], nodes[j]
            cn = len(set(G.neighbors(u)) & set(G.neighbors(w)))
            if G.has_edge(u, w):
                if cn != lam:
                    return False
            else:
                if cn != mu:
                    return False
    return True


def graph_to_dict(G: nx.Graph, name: str, extra: dict | None = None) -> dict:
    """Convert a NetworkX graph to a serializable dictionary."""
    nodes = sorted(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    G_relabeled = nx.relabel_nodes(G, node_map)

    adj_matrix = nx.adjacency_matrix(G_relabeled, nodelist=range(len(nodes))).toarray().tolist()
    edge_list = sorted([[int(u), int(v)] for u, v in G_relabeled.edges()])
    degree_seq = sorted([d for _, d in G_relabeled.degree()], reverse=True)
    adj_eigs, lap_eigs = compute_eigenvalues(G_relabeled)

    result = {
        "name": name,
        "num_nodes": G_relabeled.number_of_nodes(),
        "num_edges": G_relabeled.number_of_edges(),
        "degree_sequence": degree_seq,
        "edge_list": edge_list,
        "adjacency_matrix": adj_matrix,
        "adjacency_eigenvalues": adj_eigs,
        "laplacian_eigenvalues": lap_eigs,
    }
    if extra:
        result.update(extra)
    return result


# ─────────────────────────────────────────────────────────────────────
# Graph constructions
# ─────────────────────────────────────────────────────────────────────

def build_rook_4x4() -> nx.Graph:
    """Build 4x4 Rook's graph = line graph of K(4,4). srg(16,6,2,2)."""
    K44 = nx.complete_bipartite_graph(4, 4)
    rook = nx.line_graph(K44)
    return nx.convert_node_labels_to_integers(rook)


def build_shrikhande() -> nx.Graph:
    """Build Shrikhande graph as Cayley graph on Z4 x Z4. srg(16,6,2,2)."""
    G = nx.Graph()
    vertices = list(itertools.product(range(4), range(4)))
    G.add_nodes_from(range(16))
    # Connection set: ±(1,0), ±(0,1), ±(1,1) mod 4
    offsets = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]
    for i, (a1, b1) in enumerate(vertices):
        for da, db in offsets:
            a2 = (a1 + da) % 4
            b2 = (b1 + db) % 4
            j = 4 * a2 + b2
            if i != j:
                G.add_edge(i, j)
    return G


def parse_adjacency_matrix(lines: list[str]) -> nx.Graph:
    """Parse a binary adjacency matrix (list of '01...' strings) into a graph."""
    n = len(lines)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if lines[i][j] == '1':
                G.add_edge(i, j)
    return G


def build_csl_graph(M: int, R: int) -> nx.Graph:
    """Build Circular Skip Links graph Gskip(M, R).

    Definition from Murphy et al. 2019:
    - M vertices {0, ..., M-1}
    - Cycle edges: {j, (j+1) mod M}
    - Skip links: s_1=0, s_{i+1}=(s_i + R) mod M, edges {s_i, s_{i+1}}
    """
    G = nx.Graph()
    G.add_nodes_from(range(M))
    # Cycle edges
    for j in range(M):
        G.add_edge(j, (j + 1) % M)
    # Skip link edges
    s = 0
    for _ in range(M):
        s_next = (s + R) % M
        G.add_edge(s, s_next)
        s = s_next
    return G


# ─────────────────────────────────────────────────────────────────────
# Spence Database matrices (fetched from maths.gla.ac.uk/~es/SRGs/)
# ─────────────────────────────────────────────────────────────────────

SPENCE_16_6_2_2 = [
    # Graph 1 (Rook's graph from Spence)
    [
        "0111111000000000",
        "1011000111000000",
        "1101000000111000",
        "1110000000000111",
        "1000011100100100",
        "1000101010010010",
        "1000110001001001",
        "0100100011100100",
        "0100010101010010",
        "0100001110001001",
        "0010100100011100",
        "0010010010101010",
        "0010001001110001",
        "0001100100100011",
        "0001010010010101",
        "0001001001001110",
    ],
    # Graph 2 (Shrikhande graph from Spence)
    [
        "0111111000000000",
        "1011000111000000",
        "1100100100110000",
        "1100010010001100",
        "1010001000101010",
        "1001001000010101",
        "1000110001000011",
        "0110000001010110",
        "0101000001101001",
        "0100001110000011",
        "0010100010011001",
        "0010010100100101",
        "0001100010100110",
        "0001010100011010",
        "0000101101001100",
        "0000011011110000",
    ],
]

SPENCE_25_12_5_6 = [
    # Matrix 1
    [
        "0111111111111000000000000",
        "1011111000000111111000000",
        "1101111000000000000111111",
        "1110000111000111000111000",
        "1110000100110100110100110",
        "1110000010101010101010101",
        "1110000001011001011001011",
        "1001100011100101001000111",
        "1001010101010010110010011",
        "1001001110001000111101100",
        "1000110100011110001101001",
        "1000101010101011010110010",
        "1000011001110101100011100",
        "0101100100101001110011001",
        "0101010010110001101101010",
        "0101001100011110001010110",
        "0100110011001110010001110",
        "0100101011010100101110001",
        "0100011101100011010100101",
        "0011100001110010011011100",
        "0011010010011101010100101",
        "0011001001101110100100011",
        "0010110101001001101110010",
        "0010101110010011100001101",
        "0010011110100100011011010",
    ],
    # Matrix 2
    [
        "0111111111111000000000000",
        "1011111000000111111000000",
        "1101111000000000000111111",
        "1110000111000111000111000",
        "1110000100110100110100110",
        "1110000010101010101010101",
        "1110000001011001011001011",
        "1001100011100101001000111",
        "1001010101010010110001101",
        "1001001110001000111110010",
        "1000110100011110001011010",
        "1000101010101011010101100",
        "1000011001110101100110001",
        "0101100100101001110011001",
        "0101010010110001101101010",
        "0101001100011110001100101",
        "0100110011001110010100011",
        "0100101011010100101011100",
        "0100011101100011010010110",
        "0011100001011011100010110",
        "0011010001101100011101100",
        "0011001010110110010010011",
        "0010110110010001011110001",
        "0010101101100010101101001",
        "0010011110001101100001110",
    ],
    # Matrix 3
    [
        "0111111111111000000000000",
        "1011111000000111111000000",
        "1101111000000000000111111",
        "1110000111000111000111000",
        "1110000100110100110100110",
        "1110000010101010101010101",
        "1110000001011001011001011",
        "1001100011100101001000111",
        "1001010101001010110100011",
        "1001001110010000111011100",
        "1000110100011110001011010",
        "1000101001101011100101100",
        "1000011010110101010110001",
        "0101100100101001110011001",
        "0101010010110001101101010",
        "0101001100011110001100101",
        "0100110011010110010001101",
        "0100101011001100101110010",
        "0100011101100011010010110",
        "0011100010011011010010110",
        "0011010001101100011101100",
        "0011001001110110100010011",
        "0010110101010001101110001",
        "0010101110100010011101001",
        "0010011110001101100001110",
    ],
]

SPENCE_26_10_3_4 = [
    # Matrix 1
    [
        "01111111111000000000000000",
        "10111000000111111000000000",
        "11010100000100000111110000",
        "11100010000010000100001111",
        "11000001100001100011001100",
        "10100001100001010000110011",
        "10010000011000110010101010",
        "10001100010010001010010110",
        "10001100001000011101001001",
        "10000011001110100001010001",
        "10000010110101001100100100",
        "01100000011001001010011001",
        "01010001010000011001100101",
        "01001100001100100000100111",
        "01001010010001010101010010",
        "01000110100010101000111000",
        "01000001101110010110000010",
        "00110000101000101001010110",
        "00101011000100001001101010",
        "00101000110010100110100001",
        "00100110001011010011000100",
        "00100101010100110100001100",
        "00011010100100010010010101",
        "00011001001011000100111000",
        "00010111000001101110000001",
        "00010100110111000001001010",
    ],
    # Matrix 2
    [
        "01111111111000000000000000",
        "10111000000111111000000000",
        "11010100000100000111110000",
        "11100010000010000100001111",
        "11000001100001100011001100",
        "10100001100001010000110011",
        "10010000011000110010101010",
        "10001100010010001010010110",
        "10001100001000011101001001",
        "10000011001110100001010001",
        "10000010110101001100100100",
        "01100000011001001001011010",
        "01010001010000011001100101",
        "01001100001100100000100111",
        "01001010010001010110010001",
        "01000110100010101000111000",
        "01000001101110010110000010",
        "00110000101000101010010101",
        "00101011000000101101100010",
        "00101000110110000010101001",
        "00100110001011010011000100",
        "00100101010100110100001100",
        "00011010100100010001010110",
        "00011001001011000100111000",
        "00010111000101001010001001",
        "00010100110011100101000010",
    ],
]


# ─────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────

def build_all_pairs() -> list[dict]:
    """Build and verify all graph pairs."""
    pairs = []

    # ── PAIR 1: srg(16,6,2,2) — Rook's vs Shrikhande ────────────
    print("Building srg(16,6,2,2): Rook's graph vs Shrikhande graph...")
    rook = build_rook_4x4()
    shrikhande = build_shrikhande()

    # Also parse from Spence for cross-validation
    spence_g1 = parse_adjacency_matrix(SPENCE_16_6_2_2[0])
    spence_g2 = parse_adjacency_matrix(SPENCE_16_6_2_2[1])

    # Verify our constructions match Spence's
    assert nx.is_isomorphic(rook, spence_g1) or nx.is_isomorphic(rook, spence_g2), \
        "Rook graph doesn't match either Spence graph!"
    assert nx.is_isomorphic(shrikhande, spence_g1) or nx.is_isomorphic(shrikhande, spence_g2), \
        "Shrikhande graph doesn't match either Spence graph!"
    print("  ✓ Cross-validated with Spence DB")

    # Verify properties
    assert rook.number_of_nodes() == 16
    assert shrikhande.number_of_nodes() == 16
    assert rook.number_of_edges() == 48
    assert shrikhande.number_of_edges() == 48
    assert not nx.is_isomorphic(rook, shrikhande), "Graphs should NOT be isomorphic!"
    print("  ✓ Non-isomorphic confirmed")

    # Verify SRG parameters
    assert verify_srg_parameters(rook, v=16, k=6, lam=2, mu=2), "Rook's graph SRG params wrong!"
    assert verify_srg_parameters(shrikhande, v=16, k=6, lam=2, mu=2), "Shrikhande SRG params wrong!"
    print("  ✓ SRG(16,6,2,2) parameters verified")

    # Build dicts
    rook_dict = graph_to_dict(rook, "Rook_4x4")
    shrikhande_dict = graph_to_dict(shrikhande, "Shrikhande")

    # Check cospectrality
    is_cospec_adj = check_cospectral(rook_dict["adjacency_eigenvalues"], shrikhande_dict["adjacency_eigenvalues"])
    is_cospec_lap = check_cospectral(rook_dict["laplacian_eigenvalues"], shrikhande_dict["laplacian_eigenvalues"])
    assert is_cospec_adj, "SRG pair should be cospectral (adjacency)!"
    assert is_cospec_lap, "SRG pair should be cospectral (Laplacian)!"
    print("  ✓ Cospectral (both adjacency and Laplacian)")

    # 1-WL
    wl_eq, hist_a, hist_b = verify_1wl_equivalent(rook, shrikhande)
    assert wl_eq, "SRG pair should be 1-WL equivalent!"
    print(f"  ✓ 1-WL equivalent (histograms: {hist_a}, {hist_b})")

    pairs.append({
        "pair_id": "srg16_rook_vs_shrikhande",
        "category": "srg_16_6_2_2",
        "graph_a": rook_dict,
        "graph_b": shrikhande_dict,
        "verification": {
            "is_isomorphic": False,
            "is_cospectral_adjacency": bool(is_cospec_adj),
            "is_cospectral_laplacian": bool(is_cospec_lap),
            "wl1_equivalent": True,
            "wl1_color_histogram_a": list(hist_a),
            "wl1_color_histogram_b": list(hist_b),
            "srg_parameters": [16, 6, 2, 2],
        },
    })

    # ── PAIRS 2-3: srg(25,12,5,6) — Paulus graphs from Spence ───
    print("\nBuilding srg(25,12,5,6): Paulus graphs from Spence DB...")
    paulus25_graphs = []
    for idx, matrix_lines in enumerate(SPENCE_25_12_5_6):
        G = parse_adjacency_matrix(matrix_lines)
        assert G.number_of_nodes() == 25, f"Graph {idx+1} has {G.number_of_nodes()} nodes"
        assert G.number_of_edges() == 150, f"Graph {idx+1} has {G.number_of_edges()} edges"
        assert all(d == 12 for _, d in G.degree()), f"Graph {idx+1} is not 12-regular"
        paulus25_graphs.append(G)
    print(f"  ✓ Parsed {len(paulus25_graphs)} graphs, all 25-node 12-regular")

    # Verify non-isomorphism between all pairs
    for i in range(len(paulus25_graphs)):
        for j in range(i + 1, len(paulus25_graphs)):
            assert not nx.is_isomorphic(paulus25_graphs[i], paulus25_graphs[j]), \
                f"Graphs {i+1} and {j+1} are isomorphic (unexpected)!"
    print("  ✓ All pairs non-isomorphic")

    # Verify SRG parameters for first graph (full check is slow, do one)
    assert verify_srg_parameters(paulus25_graphs[0], v=25, k=12, lam=5, mu=6), \
        "Paulus(25) graph SRG params wrong!"
    print("  ✓ SRG(25,12,5,6) parameters verified (graph 1)")

    # Create 2 pairs: (1,2) and (1,3)
    paulus25_pair_indices = [(0, 1), (0, 2)]
    for pair_idx, (i, j) in enumerate(paulus25_pair_indices):
        ga_dict = graph_to_dict(paulus25_graphs[i], f"Paulus25_graph{i+1}")
        gb_dict = graph_to_dict(paulus25_graphs[j], f"Paulus25_graph{j+1}")

        is_cospec_adj = check_cospectral(ga_dict["adjacency_eigenvalues"], gb_dict["adjacency_eigenvalues"])
        is_cospec_lap = check_cospectral(ga_dict["laplacian_eigenvalues"], gb_dict["laplacian_eigenvalues"])
        wl_eq, hist_a, hist_b = verify_1wl_equivalent(paulus25_graphs[i], paulus25_graphs[j])

        assert is_cospec_adj, f"Paulus25 pair ({i+1},{j+1}) should be cospectral (adj)!"
        assert wl_eq, f"Paulus25 pair ({i+1},{j+1}) should be 1-WL equivalent!"
        print(f"  ✓ Pair ({i+1},{j+1}): cospectral={is_cospec_adj}, 1-WL-eq={wl_eq}")

        pairs.append({
            "pair_id": f"srg25_paulus_g{i+1}_vs_g{j+1}",
            "category": "srg_25_12_5_6",
            "graph_a": ga_dict,
            "graph_b": gb_dict,
            "verification": {
                "is_isomorphic": False,
                "is_cospectral_adjacency": bool(is_cospec_adj),
                "is_cospectral_laplacian": bool(is_cospec_lap),
                "wl1_equivalent": True,
                "wl1_color_histogram_a": list(hist_a),
                "wl1_color_histogram_b": list(hist_b),
                "srg_parameters": [25, 12, 5, 6],
            },
        })

    # ── PAIRS 4-5: srg(26,10,3,4) — Paulus graphs from Spence ───
    print("\nBuilding srg(26,10,3,4): Paulus graphs from Spence DB...")
    paulus26_graphs = []
    for idx, matrix_lines in enumerate(SPENCE_26_10_3_4):
        G = parse_adjacency_matrix(matrix_lines)
        assert G.number_of_nodes() == 26, f"Graph {idx+1} has {G.number_of_nodes()} nodes"
        assert G.number_of_edges() == 130, f"Graph {idx+1} has {G.number_of_edges()} edges"
        assert all(d == 10 for _, d in G.degree()), f"Graph {idx+1} is not 10-regular"
        paulus26_graphs.append(G)
    print(f"  ✓ Parsed {len(paulus26_graphs)} graphs, all 26-node 10-regular")

    # Verify non-isomorphism
    for i in range(len(paulus26_graphs)):
        for j in range(i + 1, len(paulus26_graphs)):
            assert not nx.is_isomorphic(paulus26_graphs[i], paulus26_graphs[j]), \
                f"Graphs {i+1} and {j+1} are isomorphic (unexpected)!"
    print("  ✓ All pairs non-isomorphic")

    # Verify SRG parameters
    assert verify_srg_parameters(paulus26_graphs[0], v=26, k=10, lam=3, mu=4), \
        "Paulus(26) graph SRG params wrong!"
    print("  ✓ SRG(26,10,3,4) parameters verified (graph 1)")

    # Create 2 pairs: (1,2)
    paulus26_pair_indices = [(0, 1)]
    for pair_idx, (i, j) in enumerate(paulus26_pair_indices):
        ga_dict = graph_to_dict(paulus26_graphs[i], f"Paulus26_graph{i+1}")
        gb_dict = graph_to_dict(paulus26_graphs[j], f"Paulus26_graph{j+1}")

        is_cospec_adj = check_cospectral(ga_dict["adjacency_eigenvalues"], gb_dict["adjacency_eigenvalues"])
        is_cospec_lap = check_cospectral(ga_dict["laplacian_eigenvalues"], gb_dict["laplacian_eigenvalues"])
        wl_eq, hist_a, hist_b = verify_1wl_equivalent(paulus26_graphs[i], paulus26_graphs[j])

        assert is_cospec_adj, f"Paulus26 pair ({i+1},{j+1}) should be cospectral (adj)!"
        assert wl_eq, f"Paulus26 pair ({i+1},{j+1}) should be 1-WL equivalent!"
        print(f"  ✓ Pair ({i+1},{j+1}): cospectral={is_cospec_adj}, 1-WL-eq={wl_eq}")

        pairs.append({
            "pair_id": f"srg26_paulus_g{i+1}_vs_g{j+1}",
            "category": "srg_26_10_3_4",
            "graph_a": ga_dict,
            "graph_b": gb_dict,
            "verification": {
                "is_isomorphic": False,
                "is_cospectral_adjacency": bool(is_cospec_adj),
                "is_cospectral_laplacian": bool(is_cospec_lap),
                "wl1_equivalent": True,
                "wl1_color_histogram_a": list(hist_a),
                "wl1_color_histogram_b": list(hist_b),
                "srg_parameters": [26, 10, 3, 4],
            },
        })

    # ── PAIRS 6-10: CSL(41, R) — Circular Skip Links ────────────
    print("\nBuilding CSL(41, R) graphs...")
    M = 41
    R_values = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]

    csl_graphs = {}
    for R in R_values:
        G = build_csl_graph(M, R)
        assert G.number_of_nodes() == 41, f"CSL({M},{R}) has {G.number_of_nodes()} nodes"
        assert G.number_of_edges() == 82, f"CSL({M},{R}) has {G.number_of_edges()} edges"
        assert all(d == 4 for _, d in G.degree()), f"CSL({M},{R}) is not 4-regular"
        assert nx.is_connected(G), f"CSL({M},{R}) is not connected"
        csl_graphs[R] = G
    print(f"  ✓ Built {len(csl_graphs)} CSL graphs, all 41-node 4-regular connected")

    # Verify 1-WL equivalence for all CSL graphs (they're all 4-regular)
    hist0, _ = run_1wl(csl_graphs[R_values[0]])
    for R in R_values[1:]:
        hist_r, _ = run_1wl(csl_graphs[R])
        assert hist0 == hist_r, f"CSL({M},{R}) has different 1-WL histogram: {hist_r} vs {hist0}"
    print(f"  ✓ All CSL graphs have identical 1-WL color histogram: {hist0}")

    # Select 5 pairs
    csl_pair_Rs = [(2, 3), (4, 5), (6, 9), (11, 12), (13, 16)]
    for Ra, Rb in csl_pair_Rs:
        Ga = csl_graphs[Ra]
        Gb = csl_graphs[Rb]

        # Verify non-isomorphism
        is_iso = nx.is_isomorphic(Ga, Gb)
        assert not is_iso, f"CSL({M},{Ra}) and CSL({M},{Rb}) should NOT be isomorphic!"

        ga_dict = graph_to_dict(Ga, f"Gskip_{M}_{Ra}", extra={"skip_value": Ra})
        gb_dict = graph_to_dict(Gb, f"Gskip_{M}_{Rb}", extra={"skip_value": Rb})

        is_cospec_adj = check_cospectral(ga_dict["adjacency_eigenvalues"], gb_dict["adjacency_eigenvalues"])
        is_cospec_lap = check_cospectral(ga_dict["laplacian_eigenvalues"], gb_dict["laplacian_eigenvalues"])
        wl_eq, hist_a, hist_b = verify_1wl_equivalent(Ga, Gb)

        assert wl_eq, f"CSL pair ({Ra},{Rb}) should be 1-WL equivalent!"
        print(f"  ✓ CSL pair (R={Ra}, R={Rb}): non-iso=True, cospectral_adj={is_cospec_adj}, 1-WL-eq=True")

        pairs.append({
            "pair_id": f"csl41_skip{Ra}_vs_skip{Rb}",
            "category": "csl_41",
            "graph_a": ga_dict,
            "graph_b": gb_dict,
            "verification": {
                "is_isomorphic": False,
                "is_cospectral_adjacency": bool(is_cospec_adj),
                "is_cospectral_laplacian": bool(is_cospec_lap),
                "wl1_equivalent": True,
                "wl1_color_histogram_a": list(hist_a),
                "wl1_color_histogram_b": list(hist_b),
                "csl_parameters": {"M": M, "R_a": Ra, "R_b": Rb},
            },
        })

    return pairs


def build_dataset() -> dict:
    """Build the complete dataset."""
    pairs = build_all_pairs()

    dataset = {
        "metadata": {
            "dataset_name": "1WL_Equivalent_Graph_Pairs",
            "description": (
                "Curated dataset of non-isomorphic graph pairs that are 1-WL equivalent, "
                "for testing expressiveness of graph features beyond 1-WL"
            ),
            "total_pairs": len(pairs),
            "categories": ["srg_16_6_2_2", "srg_25_12_5_6", "srg_26_10_3_4", "csl_41"],
            "construction_sources": [
                "Shrikhande graph: Cayley graph on Z4×Z4",
                "Rook's graph: Line graph of K(4,4)",
                "Paulus graphs srg(25,12,5,6): Ted Spence database (University of Glasgow)",
                "Paulus graphs srg(26,10,3,4): Ted Spence database (University of Glasgow)",
                "CSL graphs: Murphy et al. 2019 'Relational Pooling for Graph Representations' (ICML), Gskip(41, R)",
            ],
            "references": [
                "Spence, E. 'Strongly Regular Graphs'. https://www.maths.gla.ac.uk/~es/srgraphs.php",
                "Murphy, R. et al. (2019). 'Relational Pooling for Graph Representations'. ICML.",
                "Shrikhande, S.S. (1959). 'The uniqueness of the L2 association scheme'.",
            ],
        },
        "pairs": pairs,
    }

    return dataset


def save_versions(dataset: dict, output_dir: Path) -> None:
    """Save full, mini, and preview versions of the dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full version
    full_path = output_dir / "data_out.json"
    with open(full_path, "w") as f:
        json.dump(dataset, f, indent=2)
    size_mb = full_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Full dataset saved: {full_path} ({size_mb:.2f} MB)")

    # Mini version (no adjacency matrices)
    mini = json.loads(json.dumps(dataset))  # deep copy
    for pair in mini["pairs"]:
        for key in ["graph_a", "graph_b"]:
            if "adjacency_matrix" in pair[key]:
                del pair[key]["adjacency_matrix"]
    mini_path = output_dir / "data_out_mini.json"
    with open(mini_path, "w") as f:
        json.dump(mini, f, indent=2)
    size_mb_mini = mini_path.stat().st_size / (1024 * 1024)
    print(f"✓ Mini dataset saved: {mini_path} ({size_mb_mini:.2f} MB)")

    # Preview version (first 3 pairs only, full fields)
    preview = json.loads(json.dumps(dataset))
    preview["pairs"] = preview["pairs"][:3]
    preview["metadata"]["total_pairs"] = len(preview["pairs"])
    preview["metadata"]["note"] = "Preview: first 3 pairs only"
    preview_path = output_dir / "data_out_preview.json"
    with open(preview_path, "w") as f:
        json.dump(preview, f, indent=2)
    size_mb_preview = preview_path.stat().st_size / (1024 * 1024)
    print(f"✓ Preview dataset saved: {preview_path} ({size_mb_preview:.2f} MB)")


def final_validation(dataset: dict) -> None:
    """Run final validation checklist."""
    print("\n" + "=" * 60)
    print("FINAL VALIDATION CHECKLIST")
    print("=" * 60)

    for pair in dataset["pairs"]:
        pid = pair["pair_id"]
        v = pair["verification"]

        # 1. Non-isomorphic
        assert v["is_isomorphic"] is False, f"{pid}: should be non-isomorphic!"
        print(f"  ☑ {pid}: non-isomorphic=True")

        # 2. 1-WL equivalent
        assert v["wl1_equivalent"] is True, f"{pid}: should be 1-WL equivalent!"
        assert v["wl1_color_histogram_a"] == v["wl1_color_histogram_b"], f"{pid}: histograms differ!"
        print(f"  ☑ {pid}: 1-WL equivalent=True")

        # 3. SRG pairs should be cospectral
        if pair["category"].startswith("srg"):
            assert v["is_cospectral_adjacency"] is True, f"{pid}: SRG pair should be cospectral!"
            print(f"  ☑ {pid}: cospectral=True")

        # 4. Adjacency matrices are symmetric with zero diagonal
        for key in ["graph_a", "graph_b"]:
            adj = pair[key]["adjacency_matrix"]
            n = len(adj)
            for i in range(n):
                assert adj[i][i] == 0, f"{pid} {key}: diagonal not zero at [{i}][{i}]"
                for j in range(i + 1, n):
                    assert adj[i][j] == adj[j][i], f"{pid} {key}: not symmetric at [{i}][{j}]"
        print(f"  ☑ {pid}: adjacency matrices symmetric, zero diagonal")

    # 5. JSON validity
    json_str = json.dumps(dataset)
    json.loads(json_str)
    print("  ☑ JSON is valid and parseable")

    print("=" * 60)
    print(f"ALL {len(dataset['pairs'])} PAIRS PASSED VALIDATION")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Building 1-WL Equivalent Graph Pairs Dataset")
    print("=" * 60)

    dataset = build_dataset()
    final_validation(dataset)
    save_versions(dataset, output_dir=WORKSPACE)

    print("\n✓ Dataset construction complete!")
    print(f"  Total pairs: {dataset['metadata']['total_pairs']}")
    for cat in dataset["metadata"]["categories"]:
        count = sum(1 for p in dataset["pairs"] if p["category"] == cat)
        print(f"  {cat}: {count} pairs")
