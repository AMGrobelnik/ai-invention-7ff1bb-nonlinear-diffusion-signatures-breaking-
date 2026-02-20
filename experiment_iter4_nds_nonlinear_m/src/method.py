#!/usr/bin/env python3
"""
NDS Nonlinear Mode-Coupling Analysis with Spectral Decomposition and Baseline Comparison.

Three-part experiment:
  (A) NDS distinguishability on graph pairs with full linear-vs-nonlinear ablation
  (B) Spectral decomposition of NDS outputs into Laplacian eigenbasis to measure
      cross-frequency coupling energy
  (C) Head-to-head comparison of NDS against RWSE, LapPE, and substructure counting
      baselines, measuring both distinguishing power and wall-clock time.

Additionally constructs synthetic non-vertex-transitive regular graph pairs
(Fallback 3 from plan) because all 9 dependency pairs are vertex-transitive.
"""

import json
import time
import resource
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import linalg as sp_linalg

from loguru import logger

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Resource limits  (14 GB RAM, 3600 s CPU)
# ---------------------------------------------------------------------------
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
DEP_DIR = WORKSPACE / "dependencies" / "data_id3_it1__opus"
OUTPUT_FILE = WORKSPACE / "method_out.json"
MAX_EXAMPLES: int | None = None  # Set to int for gradual scaling, None for all


# ============================================================================
# STEP 0: DATA LOADING AND GRAPH RECONSTRUCTION
# ============================================================================

def compute_normalized_adjacency(A: np.ndarray) -> np.ndarray:
    """Compute D^{-1/2} A D^{-1/2} (symmetric normalized adjacency)."""
    degrees = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-15)))
    return D_inv_sqrt @ A @ D_inv_sqrt


def compute_laplacian_eigen(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Laplacian eigenvalues and eigenvectors: L = D - A."""
    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvals, eigvecs


def build_graph(graph_dict: dict) -> dict:
    """Build a graph data structure from the JSON representation."""
    A = np.array(graph_dict["adjacency_matrix"], dtype=np.float64)
    n = graph_dict["num_nodes"]
    m = graph_dict["num_edges"]
    degree = np.array(graph_dict["degree_sequence"], dtype=np.float64)
    A_norm = compute_normalized_adjacency(A)
    eigvals, eigvecs = compute_laplacian_eigen(A)
    return {
        "A": A,
        "A_norm": A_norm,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "degree": degree,
        "n": n,
        "m": m,
        "name": graph_dict.get("name", "unknown"),
    }


def load_dependency_data(max_examples: int | None = None) -> tuple[dict, dict]:
    """Load full_data_out.json and reconstruct graphs.

    Returns:
        graphs_by_pair: {pair_id: {0: graph_a, 1: graph_b}}
        pair_metadata:  {pair_id: {...}}
    """
    data_path = DEP_DIR / "full_data_out.json"
    logger.info(f"Loading data from {data_path}")
    data = json.loads(data_path.read_text())

    graphs_by_pair: dict[str, dict] = {}
    pair_metadata: dict[str, dict] = {}

    count = 0
    for ds_entry in data["datasets"]:
        dataset_name = ds_entry["dataset"]
        for example in ds_entry["examples"]:
            if max_examples is not None and count >= max_examples:
                break
            inp = json.loads(example["input"])
            pair_id = inp["pair_id"]
            category = inp["category"]

            g_a = build_graph(inp["graph_a"])
            g_b = build_graph(inp["graph_b"])

            graphs_by_pair[pair_id] = {0: g_a, 1: g_b}
            pair_metadata[pair_id] = {
                "dataset": dataset_name,
                "category": category,
                "subcategory": category,
                "n": g_a["n"],
                "m": g_a["m"],
                "graph_a_name": g_a["name"],
                "graph_b_name": g_b["name"],
            }
            count += 1

    logger.info(f"Loaded {len(graphs_by_pair)} pairs from dependency data")
    return graphs_by_pair, pair_metadata


# ============================================================================
# SYNTHETIC NON-VERTEX-TRANSITIVE PAIRS (Fallback 3)
# ============================================================================

def _frucht_graph() -> np.ndarray:
    """Frucht graph: 12 nodes, 3-regular, trivial automorphism group (non-VT).
    Standard edge list from graph theory references."""
    edges = [
        (0,1),(0,6),(0,11),
        (1,2),(1,7),
        (2,3),(2,8),
        (3,4),(3,9),
        (4,5),(4,10),
        (5,6),(5,11),
        (6,7),
        (7,8),
        (8,9),
        (9,10),
        (10,11),
    ]
    n = 12
    A = np.zeros((n, n), dtype=np.float64)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def _tietze_graph() -> np.ndarray:
    """Tietze graph: 12 nodes, 3-regular, non-vertex-transitive.
    Standard edge list from graph theory references."""
    edges = [
        (0,1),(0,4),(0,9),
        (1,2),(1,6),
        (2,3),(2,11),
        (3,4),(3,7),
        (4,5),
        (5,6),(5,10),
        (6,7),
        (7,8),
        (8,9),(8,11),
        (9,10),
        (10,11),
    ]
    n = 12
    A = np.zeros((n, n), dtype=np.float64)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def _random_cubic_graph(n: int, seed: int) -> np.ndarray:
    """Generate a random 3-regular (cubic) graph on n nodes using a pairing model.
    Returns adjacency matrix. May fail and retry with different seeds."""
    rng = np.random.RandomState(seed)
    for attempt in range(50):
        points = list(range(n)) * 3
        rng.shuffle(points)
        A = np.zeros((n, n), dtype=np.float64)
        valid = True
        for k in range(0, len(points), 2):
            u, v = points[k], points[k + 1]
            if u == v or A[u, v] > 0:
                valid = False
                break
            A[u, v] = 1.0
            A[v, u] = 1.0
        if valid and np.all(A.sum(axis=1) == 3):
            return A
        rng = np.random.RandomState(seed * 100 + attempt + 1)
    raise RuntimeError(f"Failed to generate cubic graph with n={n}, seed={seed}")


def _non_iso_cubic_pair_1() -> tuple[np.ndarray, np.ndarray]:
    """Frucht vs Tietze — both 12-node cubic, provably non-isomorphic."""
    return _frucht_graph(), _tietze_graph()


def _non_iso_cubic_pair_seeds(n: int, s1: int, s2: int) -> tuple[np.ndarray, np.ndarray]:
    """Two random cubic graphs on n nodes with different seeds.
    High probability of being non-isomorphic for n >= 20."""
    return _random_cubic_graph(n, s1), _random_cubic_graph(n, s2)


def generate_synthetic_non_vt_pairs() -> tuple[dict, dict]:
    """Generate synthetic non-vertex-transitive regular graph pairs.
    These have non-uniform clustering coefficients, so scalar inits vary."""
    graphs_by_pair: dict[str, dict] = {}
    pair_metadata: dict[str, dict] = {}

    # Pair 1: Frucht vs Tietze (12 nodes, cubic)
    A_a, A_b = _non_iso_cubic_pair_1()
    for idx, (label, A) in enumerate([(0, A_a), (1, A_b)]):
        n = A.shape[0]
        g = {
            "A": A,
            "A_norm": compute_normalized_adjacency(A),
            "eigvals": compute_laplacian_eigen(A)[0],
            "eigvecs": compute_laplacian_eigen(A)[1],
            "degree": A.sum(axis=1),
            "n": n,
            "m": int(A.sum()) // 2,
            "name": ["Frucht", "Tietze"][idx],
        }
        if "synth_frucht_vs_tietze" not in graphs_by_pair:
            graphs_by_pair["synth_frucht_vs_tietze"] = {}
        graphs_by_pair["synth_frucht_vs_tietze"][label] = g
    pair_metadata["synth_frucht_vs_tietze"] = {
        "dataset": "synthetic_non_vt",
        "category": "cubic_12",
        "subcategory": "frucht_vs_tietze",
        "n": 12,
        "m": 18,
        "graph_a_name": "Frucht",
        "graph_b_name": "Tietze",
    }

    # Pairs 2-6: random cubic pairs on n=20,30,40 (multiple seeds)
    configs = [
        ("synth_cubic_20_s1s2", 20, 42, 137),
        ("synth_cubic_20_s3s4", 20, 271, 503),
        ("synth_cubic_30_s1s2", 30, 42, 137),
        ("synth_cubic_30_s3s4", 30, 271, 503),
        ("synth_cubic_40_s1s2", 40, 42, 137),
    ]
    for pair_id, n, s1, s2 in configs:
        try:
            A_a, A_b = _non_iso_cubic_pair_seeds(n, s1, s2)
        except RuntimeError:
            logger.warning(f"Failed to generate cubic pair {pair_id}, skipping")
            continue
        for label, A in [(0, A_a), (1, A_b)]:
            nn = A.shape[0]
            eig_vals, eig_vecs = compute_laplacian_eigen(A)
            g = {
                "A": A,
                "A_norm": compute_normalized_adjacency(A),
                "eigvals": eig_vals,
                "eigvecs": eig_vecs,
                "degree": A.sum(axis=1),
                "n": nn,
                "m": int(A.sum()) // 2,
                "name": f"Cubic_{n}_s{s1 if label == 0 else s2}",
            }
            if pair_id not in graphs_by_pair:
                graphs_by_pair[pair_id] = {}
            graphs_by_pair[pair_id][label] = g
        pair_metadata[pair_id] = {
            "dataset": "synthetic_non_vt",
            "category": f"cubic_{n}",
            "subcategory": f"seeds_{s1}_vs_{s2}",
            "n": n,
            "m": int(A_a.sum()) // 2,
            "graph_a_name": f"Cubic_{n}_s{s1}",
            "graph_b_name": f"Cubic_{n}_s{s2}",
        }

    logger.info(f"Generated {len(graphs_by_pair)} synthetic non-VT pairs")
    return graphs_by_pair, pair_metadata


# ============================================================================
# VERTEX-TRANSITIVITY CLASSIFICATION
# ============================================================================

def compute_clustering_coefficients(graph: dict) -> np.ndarray:
    """Per-node clustering coefficient."""
    A = graph["A"]
    n = graph["n"]
    cc = np.zeros(n)
    for i in range(n):
        neighbors = np.where(A[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            cc[i] = 0.0
        else:
            subA = A[np.ix_(neighbors, neighbors)]
            triangles = np.sum(subA) / 2.0
            cc[i] = triangles / (k * (k - 1) / 2.0)
    return cc


def classify_vertex_transitivity(graph: dict) -> bool:
    """Check if graph is likely vertex-transitive.
    Uses degree regularity + multiple local invariant uniformity checks:
    clustering coefficient, betweenness centrality, subgraph centrality.
    All must be uniform for the graph to be classified as VT."""
    degrees = graph["degree"]
    is_regular = (np.min(degrees) == np.max(degrees))
    if not is_regular:
        return False
    cc = compute_clustering_coefficients(graph)
    if np.std(cc) > 1e-10:
        return False
    bt = compute_betweenness_centrality(graph)
    if np.std(bt) > 1e-6:
        return False
    sc = compute_subgraph_centrality(graph)
    if np.std(sc) > 1e-6:
        return False
    return True


# ============================================================================
# NDS COMPUTATION
# ============================================================================

def compute_nds(
    A_norm: np.ndarray,
    init_features: np.ndarray,
    nonlinearity: str,
    T: int,
) -> np.ndarray:
    """Compute NDS trajectory: alternate diffusion + nonlinearity T times.

    Returns:
        trajectory: (n, T+1) matrix — node features at each step
    """
    n = len(init_features)
    trajectory = np.zeros((n, T + 1))
    x = init_features.copy().astype(np.float64)
    trajectory[:, 0] = x

    for t in range(1, T + 1):
        # Diffusion step
        x = A_norm @ x
        # Nonlinearity step
        if nonlinearity == "linear":
            pass
        elif nonlinearity == "relu":
            x = np.maximum(x, 0.0)
        elif nonlinearity == "tanh":
            x = np.tanh(x)
        elif nonlinearity == "leaky_relu":
            x = np.where(x > 0, x, 0.01 * x)
        elif nonlinearity == "abs":
            x = np.abs(x)
        elif nonlinearity == "square":
            x = np.clip(x, -1e6, 1e6) ** 2
            x = np.clip(x, -1e12, 1e12)
        elif nonlinearity == "sin":
            x = np.sin(x)
        # Clip to prevent NaN/Inf accumulation
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=1e12, neginf=-1e12)
        trajectory[:, t] = x

    return trajectory


# ============================================================================
# SCALAR INIT FUNCTIONS
# ============================================================================

def compute_betweenness_centrality(graph: dict) -> np.ndarray:
    """Approximate betweenness via BFS-based shortest paths (Brandes algorithm)."""
    A = graph["A"]
    n = graph["n"]
    betweenness = np.zeros(n)

    for s in range(n):
        dist = -np.ones(n, dtype=int)
        dist[s] = 0
        queue = [s]
        order = []
        sigma = np.zeros(n)
        sigma[s] = 1.0
        parents: list[list[int]] = [[] for _ in range(n)]

        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            order.append(v)
            for w in range(n):
                if A[v, w] == 0:
                    continue
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    parents[w].append(v)

        delta = np.zeros(n)
        for v in reversed(order):
            for p in parents[v]:
                if sigma[v] > 0:
                    delta[p] += (sigma[p] / sigma[v]) * (1.0 + delta[v])
            if v != s:
                betweenness[v] += delta[v]

    betweenness /= 2.0  # undirected
    return betweenness


def compute_pagerank(graph: dict, alpha: float = 0.85, max_iter: int = 100) -> np.ndarray:
    """Compute PageRank via power iteration."""
    A = graph["A"]
    n = graph["n"]
    degrees = A.sum(axis=1)
    D_inv = np.diag(1.0 / np.maximum(degrees, 1e-15))
    M = D_inv @ A  # row-stochastic transition
    pr = np.ones(n) / n
    for _ in range(max_iter):
        pr_new = (1.0 - alpha) / n + alpha * (M.T @ pr)
        if np.linalg.norm(pr_new - pr) < 1e-10:
            break
        pr = pr_new
    return pr


def compute_eigenvector_centrality(graph: dict) -> np.ndarray:
    """Leading eigenvector of adjacency matrix."""
    A = graph["A"]
    eigvals, eigvecs = np.linalg.eigh(A)
    return np.abs(eigvecs[:, -1])  # Largest eigenvalue's eigenvector


def compute_subgraph_centrality(graph: dict) -> np.ndarray:
    """Subgraph centrality: sc(i) = sum_k (A^k)_{ii} / k! = (e^A)_{ii}."""
    A = graph["A"]
    expA = sp_linalg.expm(A)
    return np.diag(expA)


def compute_heat_kernel_diag(graph: dict, t: float = 1.0) -> np.ndarray:
    """Heat kernel diagonal: h(i) = (e^{-tL})_{ii}."""
    A = graph["A"]
    n = graph["n"]
    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A
    expL = sp_linalg.expm(-t * L)
    return np.diag(expL)


SCALAR_INITS = {
    "degree": lambda g: g["degree"].astype(np.float64),
    "clustering_coeff": compute_clustering_coefficients,
    "betweenness": compute_betweenness_centrality,
    "pagerank": compute_pagerank,
    "eigvec_centrality": compute_eigenvector_centrality,
    "subgraph_centrality": compute_subgraph_centrality,
    "heat_kernel_diag": compute_heat_kernel_diag,
}

NONLINEARITIES = ["linear", "relu", "tanh", "leaky_relu", "abs", "square", "sin"]
T_VALUES = [1, 2, 5, 10, 15, 20]


# ============================================================================
# PART A: NDS DISTINGUISHABILITY ABLATION
# ============================================================================

def compare_multisets(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    threshold: float = 1e-6,
) -> tuple[float, bool]:
    """Compare two multisets of node features by sorting rows and computing
    Frobenius distance. Returns (distance, distinguished).

    Uses a relative threshold: distinguishes if dist > threshold * max(||a||, ||b||, 1).
    This avoids false positives from floating-point amplification in nonlinearities
    like square that magnify tiny numerical differences."""
    sorted_a = np.sort(feat_a, axis=0)
    sorted_b = np.sort(feat_b, axis=0)
    diff = sorted_a - sorted_b
    if not np.all(np.isfinite(diff)):
        diff = np.nan_to_num(diff, nan=0.0, posinf=1e12, neginf=-1e12)
    dist = float(np.linalg.norm(diff, "fro"))
    scale = max(float(np.linalg.norm(sorted_a, "fro")),
                float(np.linalg.norm(sorted_b, "fro")), 1.0)
    return dist, dist > threshold * scale


@logger.catch
def run_part_a(
    graphs_by_pair: dict,
    pair_metadata: dict,
    vt_cache: dict,
) -> tuple[list[dict], dict]:
    """Part A: NDS distinguishability on all pairs with full ablation."""
    logger.info("=== PART A: NDS Distinguishability Ablation ===")
    results = []
    total_configs = len(graphs_by_pair) * len(SCALAR_INITS) * len(NONLINEARITIES) * len(T_VALUES)
    logger.info(f"Total configurations: {total_configs}")

    # Precompute init features for all graphs
    init_cache: dict[str, dict[str, np.ndarray]] = {}
    for pair_id, pair_graphs in graphs_by_pair.items():
        for label in [0, 1]:
            g = pair_graphs[label]
            cache_key = f"{pair_id}_{label}"
            init_cache[cache_key] = {}
            for init_name, init_fn in SCALAR_INITS.items():
                init_cache[cache_key][init_name] = init_fn(g)
    logger.info("Precomputed init features for all graphs")

    done = 0
    for pair_id, pair_graphs in graphs_by_pair.items():
        g_a, g_b = pair_graphs[0], pair_graphs[1]
        is_vt = vt_cache.get(pair_id, True)
        meta = pair_metadata[pair_id]

        for init_name in SCALAR_INITS:
            feat_a = init_cache[f"{pair_id}_0"][init_name]
            feat_b = init_cache[f"{pair_id}_1"][init_name]
            init_varies_a = bool(np.std(feat_a) > 1e-10)
            init_varies_b = bool(np.std(feat_b) > 1e-10)

            for nonlin in NONLINEARITIES:
                for T in T_VALUES:
                    traj_a = compute_nds(
                        A_norm=g_a["A_norm"],
                        init_features=feat_a,
                        nonlinearity=nonlin,
                        T=T,
                    )
                    traj_b = compute_nds(
                        A_norm=g_b["A_norm"],
                        init_features=feat_b,
                        nonlinearity=nonlin,
                        T=T,
                    )
                    frob_dist, dist_traj = compare_multisets(traj_a, traj_b)
                    step_a = np.sort(traj_a[:, -1])
                    step_b = np.sort(traj_b[:, -1])
                    step_dist = float(np.linalg.norm(step_a - step_b))
                    step_distinguished = step_dist > 1e-8

                    results.append({
                        "pair_id": pair_id,
                        "dataset": meta["dataset"],
                        "subcategory": meta["subcategory"],
                        "is_vertex_transitive": is_vt,
                        "init": init_name,
                        "nonlinearity": nonlin,
                        "T": T,
                        "frobenius_distance": frob_dist,
                        "distinguished_trajectory": dist_traj,
                        "step_frobenius": step_dist,
                        "distinguished_single_step": step_distinguished,
                        "init_varies_a": init_varies_a,
                        "init_varies_b": init_varies_b,
                    })
                    done += 1

        if done % 500 == 0 or pair_id == list(graphs_by_pair.keys())[-1]:
            logger.info(f"  Part A progress: {done}/{total_configs}")

    logger.info(f"Part A complete: {len(results)} results")

    # Compute DELTA analysis
    delta_analysis = _compute_delta_analysis(
        results=results,
        pair_metadata=pair_metadata,
        graphs_by_pair=graphs_by_pair,
        vt_cache=vt_cache,
    )
    return results, delta_analysis


def _compute_delta_analysis(
    results: list[dict],
    pair_metadata: dict,
    graphs_by_pair: dict,
    vt_cache: dict,
) -> dict:
    """Compute per-(init, nonlinearity) delta: nonlinear vs linear distinguishing."""
    vt_pairs = {pid for pid in pair_metadata if vt_cache.get(pid, True)}
    non_vt_pairs = set(pair_metadata.keys()) - vt_pairs

    # Collect best-T results per (pair, init, nonlin)
    best_results: dict[tuple, bool] = {}
    for r in results:
        key = (r["pair_id"], r["init"], r["nonlinearity"])
        if key not in best_results:
            best_results[key] = r["distinguished_trajectory"]
        else:
            best_results[key] = best_results[key] or r["distinguished_trajectory"]

    delta_analysis = {}
    for init_name in SCALAR_INITS:
        # Linear baseline
        linear_distinguished = {
            pid for pid in pair_metadata
            if best_results.get((pid, init_name, "linear"), False)
        }
        for nonlin in [nl for nl in NONLINEARITIES if nl != "linear"]:
            nonlin_distinguished = {
                pid for pid in pair_metadata
                if best_results.get((pid, init_name, nonlin), False)
            }
            delta_analysis[f"{init_name}__{nonlin}"] = {
                "nonlin_distinguished_total": len(nonlin_distinguished),
                "linear_distinguished_total": len(linear_distinguished),
                "delta_total": len(nonlin_distinguished) - len(linear_distinguished),
                "nonlin_distinguished_non_vt": len(nonlin_distinguished & non_vt_pairs),
                "linear_distinguished_non_vt": len(linear_distinguished & non_vt_pairs),
                "delta_non_vt": len(nonlin_distinguished & non_vt_pairs) - len(linear_distinguished & non_vt_pairs),
                "nonlin_distinguished_vt": len(nonlin_distinguished & vt_pairs),
                "linear_distinguished_vt": len(linear_distinguished & vt_pairs),
                "delta_vt": len(nonlin_distinguished & vt_pairs) - len(linear_distinguished & vt_pairs),
                "positive_gap_pairs": sorted(nonlin_distinguished - linear_distinguished),
                "negative_gap_pairs": sorted(linear_distinguished - nonlin_distinguished),
            }

    return delta_analysis


# ============================================================================
# PART B: SPECTRAL DECOMPOSITION & MODE COUPLING
# ============================================================================

def compute_spectral_coupling(
    graph: dict,
    nds_trajectory: np.ndarray,
    eigvecs: np.ndarray,
) -> dict:
    """Project NDS trajectory onto Laplacian eigenbasis and measure coupling."""
    T_plus_1 = nds_trajectory.shape[1]
    V = eigvecs  # (n, n)

    # Project: c(t) = V^T @ x(t)
    traj_clean = np.nan_to_num(nds_trajectory, nan=0.0, posinf=1e12, neginf=-1e12)
    coefficients = V.T @ traj_clean  # (n, T+1)

    # Coupling matrix: C[i,j] = mean_t |c_i(t) * c_j(t)|
    n = V.shape[0]
    coupling_matrix = np.zeros((n, n))
    for t in range(T_plus_1):
        c_t = np.clip(coefficients[:, t], -1e12, 1e12)
        coupling_matrix += np.outer(np.abs(c_t), np.abs(c_t))
    coupling_matrix /= T_plus_1
    coupling_matrix = np.nan_to_num(coupling_matrix, nan=0.0, posinf=1e12, neginf=0.0)

    diag_energy = float(np.sum(np.diag(coupling_matrix)))
    total_energy = float(np.sum(coupling_matrix))
    offdiag_energy = total_energy - diag_energy

    coupling_ratio = offdiag_energy / total_energy if total_energy > 1e-15 else 0.0

    return {
        "diagonal_energy": diag_energy,
        "offdiagonal_energy": offdiag_energy,
        "total_energy": total_energy,
        "coupling_ratio": float(coupling_ratio),
    }


@logger.catch
def run_part_b(
    graphs_by_pair: dict,
    pair_metadata: dict,
    vt_cache: dict,
) -> tuple[list[dict], dict]:
    """Part B: Spectral decomposition — measure cross-frequency coupling."""
    logger.info("=== PART B: Spectral Decomposition & Mode Coupling ===")
    spectral_results = []
    T_FIXED = 10
    INITS_FOR_SPECTRAL = ["degree", "clustering_coeff", "betweenness"]
    NONLINS_FOR_SPECTRAL = ["linear", "relu", "tanh", "abs", "square", "sin"]

    for pair_id, pair_graphs in graphs_by_pair.items():
        g_a, g_b = pair_graphs[0], pair_graphs[1]
        is_vt = vt_cache.get(pair_id, True)
        meta = pair_metadata[pair_id]

        for init_name in INITS_FOR_SPECTRAL:
            feat_a = SCALAR_INITS[init_name](g_a)
            feat_b = SCALAR_INITS[init_name](g_b)

            for nonlin in NONLINS_FOR_SPECTRAL:
                traj_a = compute_nds(
                    A_norm=g_a["A_norm"],
                    init_features=feat_a,
                    nonlinearity=nonlin,
                    T=T_FIXED,
                )
                traj_b = compute_nds(
                    A_norm=g_b["A_norm"],
                    init_features=feat_b,
                    nonlinearity=nonlin,
                    T=T_FIXED,
                )

                coupling_a = compute_spectral_coupling(g_a, traj_a, g_a["eigvecs"])
                coupling_b = compute_spectral_coupling(g_b, traj_b, g_b["eigvecs"])
                coupling_diff = abs(coupling_a["coupling_ratio"] - coupling_b["coupling_ratio"])

                spectral_results.append({
                    "pair_id": pair_id,
                    "dataset": meta["dataset"],
                    "is_vt": is_vt,
                    "init": init_name,
                    "nonlinearity": nonlin,
                    "T": T_FIXED,
                    "graph_a_coupling_ratio": coupling_a["coupling_ratio"],
                    "graph_a_offdiag_energy": coupling_a["offdiagonal_energy"],
                    "graph_b_coupling_ratio": coupling_b["coupling_ratio"],
                    "graph_b_offdiag_energy": coupling_b["offdiagonal_energy"],
                    "coupling_ratio_diff": coupling_diff,
                    "avg_coupling_ratio": (coupling_a["coupling_ratio"] + coupling_b["coupling_ratio"]) / 2.0,
                    "avg_offdiag_energy": (coupling_a["offdiagonal_energy"] + coupling_b["offdiagonal_energy"]) / 2.0,
                })

    logger.info(f"Part B complete: {len(spectral_results)} spectral results")

    # Aggregate coupling summary
    coupling_summary = {}
    for init_name in INITS_FOR_SPECTRAL:
        linear_ratios = [
            r["avg_coupling_ratio"]
            for r in spectral_results
            if r["init"] == init_name and r["nonlinearity"] == "linear"
        ]
        for nonlin in [nl for nl in NONLINS_FOR_SPECTRAL if nl != "linear"]:
            nonlin_ratios = [
                r["avg_coupling_ratio"]
                for r in spectral_results
                if r["init"] == init_name and r["nonlinearity"] == nonlin
            ]
            mean_linear = float(np.mean(linear_ratios)) if linear_ratios else 0.0
            mean_nonlin = float(np.mean(nonlin_ratios)) if nonlin_ratios else 0.0
            coupling_summary[f"{init_name}__{nonlin}_vs_linear"] = {
                "mean_linear_coupling": mean_linear,
                "mean_nonlin_coupling": mean_nonlin,
                "coupling_delta": mean_nonlin - mean_linear,
            }

    return spectral_results, coupling_summary


# ============================================================================
# PART C: BASELINE COMPARISON
# ============================================================================

def compute_rwse(A: np.ndarray, k_max: int = 20) -> np.ndarray:
    """Random Walk Structural Encoding: diagonal of (D^{-1}A)^k."""
    n = A.shape[0]
    degrees = A.sum(axis=1)
    D_inv = np.diag(1.0 / np.maximum(degrees, 1e-10))
    P = D_inv @ A
    features = np.zeros((n, k_max))
    P_k = np.eye(n)
    for k in range(k_max):
        P_k = P_k @ P
        features[:, k] = np.diag(P_k)
    return features


def compute_lappe(eigvecs: np.ndarray, k: int = 20) -> np.ndarray:
    """Laplacian Positional Encoding: first k non-trivial eigenvectors (absolute values)."""
    n = eigvecs.shape[0]
    k_actual = min(k, n - 1)
    features = np.abs(eigvecs[:, 1:k_actual + 1])
    if k_actual < k:
        pad = np.zeros((n, k - k_actual))
        features = np.concatenate([features, pad], axis=1)
    return features


def compute_substructure_counts(A: np.ndarray) -> np.ndarray:
    """Count local substructures per node: triangles + raw 4-walk diagonal."""
    A2 = A @ A
    A3 = A2 @ A
    triangles = np.diag(A3) / 2.0
    A4 = A3 @ A
    raw_4walks = np.diag(A4)
    return np.column_stack([triangles, raw_4walks])


def compute_node_degree_histogram(graph: dict) -> np.ndarray:
    """Degree as 1D features — baseline that captures no structural info beyond degree."""
    return graph["degree"].reshape(-1, 1).astype(np.float64)


@logger.catch
def run_part_c(
    graphs_by_pair: dict,
    pair_metadata: dict,
    vt_cache: dict,
) -> tuple[list[dict], dict]:
    """Part C: Baseline comparison — NDS vs RWSE vs LapPE vs substructure counting."""
    logger.info("=== PART C: Baseline Comparison ===")

    methods = {
        "nds_relu_degree_T10": lambda g: compute_nds(g["A_norm"], g["degree"].astype(np.float64), "relu", 10),
        "nds_relu_betweenness_T10": lambda g: compute_nds(g["A_norm"], compute_betweenness_centrality(g), "relu", 10),
        "nds_tanh_betweenness_T10": lambda g: compute_nds(g["A_norm"], compute_betweenness_centrality(g), "tanh", 10),
        "nds_tanh_clustering_T10": lambda g: compute_nds(g["A_norm"], compute_clustering_coefficients(g), "tanh", 10),
        "nds_abs_subgraph_centrality_T10": lambda g: compute_nds(g["A_norm"], compute_subgraph_centrality(g), "abs", 10),
        "linear_degree_T10": lambda g: compute_nds(g["A_norm"], g["degree"].astype(np.float64), "linear", 10),
        "linear_betweenness_T10": lambda g: compute_nds(g["A_norm"], compute_betweenness_centrality(g), "linear", 10),
        "linear_clustering_T10": lambda g: compute_nds(g["A_norm"], compute_clustering_coefficients(g), "linear", 10),
        "rwse_k20": lambda g: compute_rwse(g["A"], k_max=20),
        "lappe_k20": lambda g: compute_lappe(g["eigvecs"], k=20),
        "substructure_tri_4cycle": lambda g: compute_substructure_counts(g["A"]),
        "degree_only": compute_node_degree_histogram,
    }

    comparison_results = []
    timing_results = {}

    for method_name, method_fn in methods.items():
        distinguished_pairs: set[str] = set()
        total_time = 0.0
        n_graphs = 0
        method_results_local = []

        for pair_id, pair_graphs in graphs_by_pair.items():
            g_a, g_b = pair_graphs[0], pair_graphs[1]
            meta = pair_metadata[pair_id]
            is_vt = vt_cache.get(pair_id, True)

            t0 = time.perf_counter()
            feat_a = method_fn(g_a)
            feat_b = method_fn(g_b)
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            n_graphs += 2

            # Ensure 2D
            if feat_a.ndim == 1:
                feat_a = feat_a.reshape(-1, 1)
            if feat_b.ndim == 1:
                feat_b = feat_b.reshape(-1, 1)

            # Handle shape mismatch
            if feat_a.shape != feat_b.shape:
                max_cols = max(feat_a.shape[1], feat_b.shape[1])
                if feat_a.shape[1] < max_cols:
                    feat_a = np.pad(feat_a, ((0, 0), (0, max_cols - feat_a.shape[1])))
                if feat_b.shape[1] < max_cols:
                    feat_b = np.pad(feat_b, ((0, 0), (0, max_cols - feat_b.shape[1])))

            dist, distinguished = compare_multisets(feat_a, feat_b)
            if distinguished:
                distinguished_pairs.add(pair_id)

            method_results_local.append({
                "method": method_name,
                "pair_id": pair_id,
                "dataset": meta["dataset"],
                "subcategory": meta["subcategory"],
                "distinguished": distinguished,
                "frobenius_distance": dist,
                "preprocessing_time_s": float(elapsed),
                "is_vt": is_vt,
            })

        comparison_results.extend(method_results_local)

        vt_pairs = {pid for pid in pair_metadata if vt_cache.get(pid, True)}
        non_vt_pairs = set(pair_metadata.keys()) - vt_pairs

        timing_results[method_name] = {
            "total_pairs": len(graphs_by_pair),
            "pairs_distinguished_total": len(distinguished_pairs),
            "pairs_distinguished_vt": len(distinguished_pairs & vt_pairs),
            "pairs_distinguished_non_vt": len(distinguished_pairs & non_vt_pairs),
            "total_time_s": round(total_time, 6),
            "avg_time_per_graph_s": round(total_time / max(n_graphs, 1), 6),
        }
        logger.info(
            f"  {method_name}: {len(distinguished_pairs)}/{len(graphs_by_pair)} "
            f"distinguished (VT: {len(distinguished_pairs & vt_pairs)}, "
            f"non-VT: {len(distinguished_pairs & non_vt_pairs)}), "
            f"time={total_time:.3f}s"
        )

    return comparison_results, timing_results


# ============================================================================
# PART D (BONUS): PER-INIT FEATURE VARIATION ANALYSIS
# ============================================================================

@logger.catch
def run_part_d_feature_analysis(
    graphs_by_pair: dict,
    pair_metadata: dict,
    vt_cache: dict,
) -> list[dict]:
    """Analyze which scalar inits actually produce varying features on each graph."""
    logger.info("=== PART D: Feature Variation Analysis ===")
    analysis_results = []

    for pair_id, pair_graphs in graphs_by_pair.items():
        meta = pair_metadata[pair_id]
        is_vt = vt_cache.get(pair_id, True)
        for label in [0, 1]:
            g = pair_graphs[label]
            for init_name, init_fn in SCALAR_INITS.items():
                features = init_fn(g)
                std_val = float(np.std(features))
                min_val = float(np.min(features))
                max_val = float(np.max(features))
                n_unique = len(np.unique(np.round(features, decimals=10)))
                analysis_results.append({
                    "pair_id": pair_id,
                    "graph_label": label,
                    "graph_name": g["name"],
                    "dataset": meta["dataset"],
                    "is_vt": is_vt,
                    "init": init_name,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "n_unique_values": n_unique,
                    "n_nodes": g["n"],
                    "varies": std_val > 1e-10,
                })

    logger.info(f"Part D complete: {len(analysis_results)} feature analyses")
    return analysis_results


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def build_output(
    part_a_results: list[dict],
    delta_analysis: dict,
    spectral_results: list[dict],
    coupling_summary: dict,
    comparison_results: list[dict],
    timing_results: dict,
    feature_analysis: list[dict],
    pair_metadata: dict,
    vt_cache: dict,
    total_time: float,
) -> dict:
    """Build output in exp_gen_sol_out schema format."""
    vt_count = sum(1 for pid in pair_metadata if vt_cache.get(pid, True))
    non_vt_count = len(pair_metadata) - vt_count

    # Determine hypothesis verdict
    any_positive_delta = any(
        v["delta_non_vt"] > 0
        for v in delta_analysis.values()
    )
    any_coupling_increase = any(
        v["coupling_delta"] > 0.001
        for v in coupling_summary.values()
    )

    if any_positive_delta and any_coupling_increase:
        verdict = "HYPOTHESIS_SUPPORTED"
    elif any_positive_delta:
        verdict = "PARTIAL_SUPPORT_DISTINGUISH_BUT_NO_COUPLING"
    elif any_coupling_increase:
        verdict = "PARTIAL_SUPPORT_COUPLING_BUT_NO_DISTINGUISH_GAIN"
    else:
        verdict = "HYPOTHESIS_NOT_SUPPORTED"

    logger.info(f"Verdict: {verdict}")
    logger.info(f"  VT pairs: {vt_count}, Non-VT pairs: {non_vt_count}")
    logger.info(f"  Any positive delta (non-VT): {any_positive_delta}")
    logger.info(f"  Any coupling increase: {any_coupling_increase}")

    # Log key delta results
    for key, val in delta_analysis.items():
        if val["delta_non_vt"] != 0 or val["delta_vt"] != 0:
            logger.info(f"  Delta [{key}]: non-VT={val['delta_non_vt']}, VT={val['delta_vt']}")

    # Log coupling summary
    for key, val in coupling_summary.items():
        logger.info(f"  Coupling [{key}]: delta={val['coupling_delta']:.6f}")

    # Log baseline comparison
    for method, stats in timing_results.items():
        logger.info(
            f"  Baseline [{method}]: {stats['pairs_distinguished_total']}/{stats['total_pairs']} "
            f"distinguished, time={stats['total_time_s']:.4f}s"
        )

    # Build datasets
    # Dataset 1: Part A NDS ablation
    part_a_examples = []
    for r in part_a_results:
        part_a_examples.append({
            "input": json.dumps({
                "pair_id": r["pair_id"],
                "init": r["init"],
                "nonlinearity": r["nonlinearity"],
                "T": r["T"],
            }),
            "output": "non-isomorphic",
            "predict_nds_trajectory": "distinguished" if r["distinguished_trajectory"] else "indistinguishable",
            "predict_nds_single_step": "distinguished" if r["distinguished_single_step"] else "indistinguishable",
            "metadata_pair_id": r["pair_id"],
            "metadata_dataset": r["dataset"],
            "metadata_is_vt": r["is_vertex_transitive"],
            "metadata_init": r["init"],
            "metadata_nonlinearity": r["nonlinearity"],
            "metadata_T": r["T"],
            "metadata_frobenius_distance": r["frobenius_distance"],
            "metadata_step_frobenius": r["step_frobenius"],
            "metadata_init_varies_a": r["init_varies_a"],
            "metadata_init_varies_b": r["init_varies_b"],
        })

    # Dataset 2: Part B spectral coupling
    part_b_examples = []
    for r in spectral_results:
        part_b_examples.append({
            "input": json.dumps({
                "pair_id": r["pair_id"],
                "init": r["init"],
                "nonlinearity": r["nonlinearity"],
            }),
            "output": "non-isomorphic",
            "predict_coupling_detected": "coupling_detected" if r["avg_coupling_ratio"] > 0.01 else "no_coupling",
            "metadata_pair_id": r["pair_id"],
            "metadata_is_vt": r["is_vt"],
            "metadata_init": r["init"],
            "metadata_nonlinearity": r["nonlinearity"],
            "metadata_avg_coupling_ratio": r["avg_coupling_ratio"],
            "metadata_coupling_ratio_diff": r["coupling_ratio_diff"],
            "metadata_graph_a_coupling": r["graph_a_coupling_ratio"],
            "metadata_graph_b_coupling": r["graph_b_coupling_ratio"],
        })

    # Dataset 3: Part C baseline comparison
    part_c_examples = []
    for r in comparison_results:
        part_c_examples.append({
            "input": json.dumps({
                "pair_id": r["pair_id"],
                "method": r["method"],
            }),
            "output": "non-isomorphic",
            "predict_method": "distinguished" if r["distinguished"] else "indistinguishable",
            "metadata_pair_id": r["pair_id"],
            "metadata_method": r["method"],
            "metadata_dataset": r["dataset"],
            "metadata_is_vt": r["is_vt"],
            "metadata_frobenius_distance": r["frobenius_distance"],
            "metadata_time_s": r["preprocessing_time_s"],
        })

    # Dataset 4: Part D feature variation analysis
    part_d_examples = []
    for r in feature_analysis:
        part_d_examples.append({
            "input": json.dumps({
                "pair_id": r["pair_id"],
                "graph_label": r["graph_label"],
                "init": r["init"],
            }),
            "output": "feature_analysis",
            "predict_feature_varies": "varies" if r["varies"] else "constant",
            "metadata_pair_id": r["pair_id"],
            "metadata_graph_name": r["graph_name"],
            "metadata_dataset": r["dataset"],
            "metadata_is_vt": r["is_vt"],
            "metadata_init": r["init"],
            "metadata_std": r["std"],
            "metadata_min": r["min"],
            "metadata_max": r["max"],
            "metadata_n_unique": r["n_unique_values"],
            "metadata_n_nodes": r["n_nodes"],
        })

    output = {
        "metadata": {
            "method_name": "NDS_ModeCoupling_Spectral_Baseline_Comparison",
            "description": (
                "Three-part NDS analysis: (A) full init × nonlinearity × T ablation "
                "on 1-WL equivalent pairs including synthetic non-VT pairs; "
                "(B) Spectral decomposition measuring cross-frequency coupling energy; "
                "(C) Head-to-head comparison against RWSE, LapPE, substructure counting baselines; "
                "(D) Per-init feature variation analysis."
            ),
            "total_pairs": len(pair_metadata),
            "vt_pairs": vt_count,
            "non_vt_pairs": non_vt_count,
            "total_time_s": round(total_time, 2),
            "verdict": verdict,
            "delta_analysis": delta_analysis,
            "coupling_summary": coupling_summary,
            "baseline_timing": timing_results,
        },
        "datasets": [
            {"dataset": "part_a_nds_ablation", "examples": part_a_examples},
            {"dataset": "part_b_spectral_coupling", "examples": part_b_examples},
            {"dataset": "part_c_baseline_comparison", "examples": part_c_examples},
            {"dataset": "part_d_feature_analysis", "examples": part_d_examples},
        ],
    }
    return output


# ============================================================================
# MAIN
# ============================================================================

@logger.catch
def main() -> None:
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("NDS Mode-Coupling Spectral Baseline Experiment — START")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Load dependency data (original 9 pairs — all VT)
    # ------------------------------------------------------------------
    graphs_by_pair, pair_metadata = load_dependency_data(max_examples=MAX_EXAMPLES)

    # ------------------------------------------------------------------
    # Generate synthetic non-VT pairs (Fallback 3)
    # ------------------------------------------------------------------
    synth_graphs, synth_meta = generate_synthetic_non_vt_pairs()
    graphs_by_pair.update(synth_graphs)
    pair_metadata.update(synth_meta)
    logger.info(f"Total pairs (original + synthetic): {len(graphs_by_pair)}")

    # ------------------------------------------------------------------
    # Classify vertex transitivity
    # ------------------------------------------------------------------
    vt_cache: dict[str, bool] = {}
    for pair_id, pair_graphs in graphs_by_pair.items():
        vt_a = classify_vertex_transitivity(pair_graphs[0])
        vt_b = classify_vertex_transitivity(pair_graphs[1])
        vt_cache[pair_id] = vt_a and vt_b

    vt_count = sum(1 for v in vt_cache.values() if v)
    non_vt_count = sum(1 for v in vt_cache.values() if not v)
    logger.info(f"VT pairs: {vt_count}, Non-VT pairs: {non_vt_count}")
    for pid, is_vt in sorted(vt_cache.items()):
        logger.info(f"  {pid}: VT={is_vt}")

    # ------------------------------------------------------------------
    # Part A: NDS Distinguishability Ablation
    # ------------------------------------------------------------------
    part_a_results, delta_analysis = run_part_a(
        graphs_by_pair=graphs_by_pair,
        pair_metadata=pair_metadata,
        vt_cache=vt_cache,
    )

    # ------------------------------------------------------------------
    # Part B: Spectral Decomposition
    # ------------------------------------------------------------------
    spectral_results, coupling_summary = run_part_b(
        graphs_by_pair=graphs_by_pair,
        pair_metadata=pair_metadata,
        vt_cache=vt_cache,
    )

    # ------------------------------------------------------------------
    # Part C: Baseline Comparison
    # ------------------------------------------------------------------
    comparison_results, timing_results = run_part_c(
        graphs_by_pair=graphs_by_pair,
        pair_metadata=pair_metadata,
        vt_cache=vt_cache,
    )

    # ------------------------------------------------------------------
    # Part D: Feature Variation Analysis
    # ------------------------------------------------------------------
    feature_analysis = run_part_d_feature_analysis(
        graphs_by_pair=graphs_by_pair,
        pair_metadata=pair_metadata,
        vt_cache=vt_cache,
    )

    total_time = time.time() - t_start
    logger.info(f"All parts completed in {total_time:.1f}s")

    # ------------------------------------------------------------------
    # Build and save output
    # ------------------------------------------------------------------
    output = build_output(
        part_a_results=part_a_results,
        delta_analysis=delta_analysis,
        spectral_results=spectral_results,
        coupling_summary=coupling_summary,
        comparison_results=comparison_results,
        timing_results=timing_results,
        feature_analysis=feature_analysis,
        pair_metadata=pair_metadata,
        vt_cache=vt_cache,
        total_time=total_time,
    )

    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    logger.info(f"Output written to {OUTPUT_FILE}")

    # Count examples per dataset
    for ds in output["datasets"]:
        logger.info(f"  {ds['dataset']}: {len(ds['examples'])} examples")
    total_examples = sum(len(ds["examples"]) for ds in output["datasets"])
    logger.info(f"  TOTAL examples: {total_examples}")

    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
