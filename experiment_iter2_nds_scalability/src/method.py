#!/usr/bin/env python3
"""NDS Scalability & Spectral Heterodyning Comparison Experiment.

Two-part experiment:
  (A) Verify O(m*T) scaling of NDS preprocessing on synthetic ER graphs + real benchmarks.
  (B) Compare NDS (interleaved diffusion+ReLU/tanh) against spectral heterodyning
      (pairwise Hadamard products of polynomial filter outputs) and a linear-only
      baseline on 9 known 1-WL-equivalent graph pairs.
"""

import gc
import json
import resource
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import sparse
from scipy.sparse import csr_matrix

# ── Resource limits (14GB RAM, 1h CPU) ──────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:HH:mm:ss}|{level:<7}|{message}",
)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent

DEP_1WL = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/"
    "run__20260219_082247/3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
)
DEP_BENCH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/"
    "run__20260219_082247/3_invention_loop/iter_1/gen_art/data_id2_it1__opus"
)

# ── Config ───────────────────────────────────────────────────────────────────
# How many examples to process (None = all).  Set via MAX_EXAMPLES env or arg.
MAX_EXAMPLES: int | None = None


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE NDS IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_normalized_adjacency_from_edge_list(
    edge_list: list[list[int]],
    num_nodes: int,
) -> csr_matrix:
    """Build symmetric normalized adjacency D^{-1/2} A D^{-1/2} from edge list."""
    if not edge_list:
        return csr_matrix((num_nodes, num_nodes))
    rows, cols = [], []
    for u, v in edge_list:
        rows.extend([u, v])
        cols.extend([v, u])
    data = np.ones(len(rows), dtype=np.float64)
    A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    # De-duplicate: (A > 0) makes binary, ensures symmetry
    A = (A > 0).astype(np.float64)
    degrees = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros(num_nodes)
    nonzero = degrees > 0
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    D_inv_sqrt = sparse.diags(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def build_normalized_adjacency_from_dense(
    adj_matrix: list[list[int]],
) -> csr_matrix:
    """Build normalized adjacency from dense adjacency matrix (list-of-lists)."""
    A = csr_matrix(np.array(adj_matrix, dtype=np.float64))
    n = A.shape[0]
    degrees = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros(n)
    nonzero = degrees > 0
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    D_inv_sqrt = sparse.diags(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def build_normalized_adjacency_from_sparse(A: csr_matrix) -> csr_matrix:
    """Build normalized adjacency from an existing sparse adjacency matrix."""
    n = A.shape[0]
    degrees = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros(n)
    nonzero = degrees > 0
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    D_inv_sqrt = sparse.diags(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


# ─────────────────────────────────────────────────────────────────────────────
#  NDS — Nonlinear Diffusion Signatures
# ─────────────────────────────────────────────────────────────────────────────

def compute_nds(
    A_norm: csr_matrix,
    X_init: np.ndarray,
    T: int,
    nonlinearity: str = "relu",
) -> np.ndarray:
    """Compute Nonlinear Diffusion Signatures.

    Supports both 1-D init (n,) and multi-D init (n, d).
    Returns trajectory of shape (n, (T+1)*d) for multi-D or (n, T+1) for 1-D.
    """
    if X_init.ndim == 1:
        X_init = X_init[:, np.newaxis]  # (n, 1)
    n, d = X_init.shape
    trajectory_cols: list[np.ndarray] = [X_init.copy()]
    X = X_init.copy()
    for _t in range(T):
        X = A_norm @ X  # linear diffusion  (n, d)
        if nonlinearity == "relu":
            X = np.maximum(X, 0.0)
        elif nonlinearity == "tanh":
            X = np.tanh(X)
        elif nonlinearity == "abs":
            X = np.abs(X)
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
        trajectory_cols.append(X.copy())
    # Stack along columns: (n, (T+1)*d)
    return np.hstack(trajectory_cols)


# ─────────────────────────────────────────────────────────────────────────────
#  SPECTRAL HETERODYNING
# ─────────────────────────────────────────────────────────────────────────────

def compute_polynomial_filters(
    A_norm: csr_matrix,
    X_init: np.ndarray,
    K: int,
) -> list[np.ndarray]:
    """Compute polynomial filter outputs: A^k * X_init for k=1..K.

    Supports both 1-D (n,) and multi-D (n, d) initial features.
    Returns list of K arrays, each (n,) or (n, d).
    """
    outputs: list[np.ndarray] = []
    X = X_init.copy()
    for _k in range(K):
        X = A_norm @ X  # A^{k+1} * X_init
        outputs.append(X.copy())
    return outputs


def compute_heterodyning_features(
    poly_outputs: list[np.ndarray],
) -> np.ndarray:
    """Form all pairwise Hadamard products p_i (element-wise) p_j.

    Works for 1-D (n,) or multi-D (n, d) filter outputs.
    Returns (n, K*(K+1)/2) for 1-D or (n, K*(K+1)/2 * d) for multi-D.
    """
    K = len(poly_outputs)
    features: list[np.ndarray] = []
    for i in range(K):
        for j in range(i, K):
            prod = poly_outputs[i] * poly_outputs[j]
            if prod.ndim == 1:
                prod = prod[:, np.newaxis]
            features.append(prod)
    return np.hstack(features)


# ─────────────────────────────────────────────────────────────────────────────
#  LINEAR-ONLY BASELINE (SIGN-like)
# ─────────────────────────────────────────────────────────────────────────────

def compute_linear_diffusion(
    A_norm: csr_matrix,
    X_init: np.ndarray,
    T: int,
) -> np.ndarray:
    """Linear diffusion only (no nonlinearity) — SIGN baseline.

    Returns trajectory (n, (T+1)*d).
    """
    if X_init.ndim == 1:
        X_init = X_init[:, np.newaxis]
    trajectory_cols: list[np.ndarray] = [X_init.copy()]
    X = X_init.copy()
    for _t in range(T):
        X = A_norm @ X
        trajectory_cols.append(X.copy())
    return np.hstack(trajectory_cols)


# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH DISTINGUISHING
# ─────────────────────────────────────────────────────────────────────────────

def graphs_distinguished(
    feat1: np.ndarray,
    feat2: np.ndarray,
    tol: float = 1e-8,
) -> bool:
    """Check if two graphs produce different sorted feature multisets."""
    # Sort rows lexicographically
    s1 = feat1[np.lexsort(feat1.T[::-1])]
    s2 = feat2[np.lexsort(feat2.T[::-1])]
    return not np.allclose(s1, s2, atol=tol)


# ─────────────────────────────────────────────────────────────────────────────
#  INITIALIZATION STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def init_degree(A_norm: csr_matrix, n: int) -> np.ndarray:
    """Degree-based initialization (scalar per node)."""
    # Use raw adjacency degrees approximated from normalized adjacency:
    # Since A_norm = D^{-1/2} A D^{-1/2}, row-sum = D^{-1/2} A D^{-1/2} 1
    # For SRGs (regular), this is just the constant k/sqrt(k*k) = 1.
    # Better: use actual degree from A_norm structure.
    # Actually, for regular graphs the row sum of A_norm = k * (1/k) = 1 for all.
    # So degree init on normalized adj is useless for regular graphs.
    # We return the un-normalized degree instead.
    A_raw = A_norm.copy()
    # Recover raw adjacency: A_raw might lose info.
    # Instead, compute degree from nnz per row of the normalized matrix pattern
    # (the sparsity pattern is the same as original adjacency).
    degrees = np.array((A_norm != 0).sum(axis=1)).flatten().astype(np.float64)
    return degrees


def init_onehot(n: int) -> np.ndarray:
    """One-hot initialization (identity matrix) — most expressive."""
    return np.eye(n, dtype=np.float64)


def init_random(n: int, seed: int = 42) -> np.ndarray:
    """Random initialization for reference."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, 1)


def init_local_structure(A_norm: csr_matrix, n: int) -> np.ndarray:
    """Local structural init: [degree, triangle_count, clustering_coeff].

    For regular graphs, degree is constant but triangle counts differ
    between non-isomorphic graphs with the same SRG parameters.
    """
    # Get the binary adjacency pattern
    A_bin = (A_norm != 0).astype(np.float64)
    degrees = np.array(A_bin.sum(axis=1)).flatten()
    # Triangle count per node: (A^3)_ii / 2
    A2 = A_bin @ A_bin
    A3_diag = np.array((A2.multiply(A_bin)).sum(axis=1)).flatten()
    triangles = A3_diag / 2.0
    # Clustering coefficient
    clustering = np.zeros(n)
    nonzero_deg = degrees > 1
    clustering[nonzero_deg] = (
        2.0 * triangles[nonzero_deg]
        / (degrees[nonzero_deg] * (degrees[nonzero_deg] - 1))
    )
    return np.column_stack([degrees, triangles, clustering])


def init_spectral(A_norm: csr_matrix, n: int, k: int = 4) -> np.ndarray:
    """Spectral position encoding: top-k eigenvectors of normalized Laplacian.

    Sign-invariant: we use absolute values of eigenvectors.
    """
    from scipy.sparse.linalg import eigsh
    # Normalized Laplacian: L = I - A_norm
    L = sparse.eye(n) - A_norm
    # Get smallest k+1 eigenvalues (skip the constant eigenvector)
    num_eigs = min(k + 1, n - 1)
    try:
        eigenvalues, eigenvectors = eigsh(L, k=num_eigs, which="SM")
    except Exception:
        # Fallback: use dense eigendecomposition for small graphs
        L_dense = L.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        eigenvalues = eigenvalues[:num_eigs]
        eigenvectors = eigenvectors[:, :num_eigs]
    # Skip the trivial (all-ones) eigenvector (eigenvalue ≈ 0)
    # Use absolute value for sign invariance
    return np.abs(eigenvectors[:, 1:k + 1]) if eigenvectors.shape[1] > 1 else np.abs(eigenvectors)


def init_random_lowdim(n: int, d: int = 4, seed: int = 42) -> np.ndarray:
    """Random d-dimensional initialization."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, d)


# ═══════════════════════════════════════════════════════════════════════════════
#  PART A: SCALABILITY EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_scalability_experiment(
    sizes: list[int] | None = None,
    avg_degree: int = 10,
    T: int = 10,
    n_repeats: int = 5,
) -> tuple[list[dict], float, float]:
    """Measure NDS wall-clock time and peak memory vs graph size.

    Returns (results_list, r_squared, slope).
    """
    if sizes is None:
        sizes = [100, 500, 1000, 5000, 10000, 50000]

    results: list[dict] = []

    for n in sizes:
        logger.info(f"Scalability: n={n}")
        rng = np.random.RandomState(42)

        # Build ER graph as sparse adjacency
        m_target = int(n * avg_degree / 2)
        rows_raw = rng.randint(0, n, size=m_target * 3)
        cols_raw = rng.randint(0, n, size=m_target * 3)
        mask = rows_raw != cols_raw
        rows_raw, cols_raw = rows_raw[mask], cols_raw[mask]
        # Take first m_target edges
        rows_raw = rows_raw[:m_target]
        cols_raw = cols_raw[:m_target]
        data_raw = np.ones(len(rows_raw))
        A = csr_matrix((data_raw, (rows_raw, cols_raw)), shape=(n, n))
        A = ((A + A.T) > 0).astype(np.float64)

        actual_m = A.nnz // 2
        A_norm = build_normalized_adjacency_from_sparse(A)
        x_init = np.array(A.sum(axis=1)).flatten()  # degree

        # Warmup run (excluded from timing)
        _ = compute_nds(A_norm, x_init, T)

        # Timed runs
        times: list[float] = []
        peak_mems: list[float] = []
        for _rep in range(n_repeats):
            tracemalloc.start()
            t0 = time.perf_counter()
            _ = compute_nds(A_norm, x_init, T)
            t1 = time.perf_counter()
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            times.append(t1 - t0)
            peak_mems.append(peak / (1024 * 1024))

        median_time = float(np.median(times))
        median_mem = float(np.median(peak_mems))
        results.append({
            "n": n,
            "m": actual_m,
            "mT": actual_m * T,
            "time_sec": median_time,
            "peak_memory_MB": median_mem,
            "T": T,
            "all_times": [round(t, 6) for t in times],
        })
        logger.info(
            f"  n={n}, m={actual_m}, mT={actual_m*T}, "
            f"time={median_time:.5f}s, mem={median_mem:.1f}MB"
        )
        gc.collect()

    # Linear regression: time vs m*T
    mT_vals = np.array([r["mT"] for r in results], dtype=np.float64)
    time_vals = np.array([r["time_sec"] for r in results], dtype=np.float64)
    A_reg = np.vstack([mT_vals, np.ones(len(mT_vals))]).T
    coeffs, residuals, rank, sv = np.linalg.lstsq(A_reg, time_vals, rcond=None)
    slope, intercept = coeffs
    ss_res = float(np.sum((time_vals - (slope * mT_vals + intercept)) ** 2))
    ss_tot = float(np.sum((time_vals - np.mean(time_vals)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    logger.info(f"Scalability linear fit: R²={r_squared:.4f}, slope={slope:.2e}")
    return results, r_squared, slope


# ═══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK TIMING (real graphs from data_id2)
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark_timing(
    benchmark_data: dict,
    T: int = 10,
    max_graphs_per_ds: int = 20,
) -> dict[str, dict]:
    """Time NDS on real benchmark graphs (CSL, MUTAG, PROTEINS, IMDB-BINARY)."""
    results_per_dataset: dict[str, dict] = {}

    for ds in benchmark_data["datasets"]:
        ds_name: str = ds["dataset"]
        # Skip "_mini" variants to avoid duplication
        if ds_name.endswith("_mini"):
            continue
        times: list[float] = []
        n_nodes_list: list[int] = []
        n_edges_list: list[int] = []
        examples = ds["examples"][:max_graphs_per_ds]
        for ex in examples:
            graph = json.loads(ex["input"])
            edge_list = graph["edge_list"]
            num_nodes = graph["num_nodes"]
            A_norm = build_normalized_adjacency_from_edge_list(edge_list, num_nodes)
            x_init = init_degree(A_norm, num_nodes)

            t0 = time.perf_counter()
            _ = compute_nds(A_norm, x_init, T)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            n_nodes_list.append(num_nodes)
            n_edges_list.append(len(edge_list) // 2)

        results_per_dataset[ds_name] = {
            "mean_time": float(np.mean(times)),
            "max_time": float(np.max(times)),
            "std_time": float(np.std(times)),
            "num_graphs": len(times),
            "avg_nodes": float(np.mean(n_nodes_list)),
            "avg_edges": float(np.mean(n_edges_list)),
        }
        logger.info(
            f"Benchmark {ds_name}: {len(times)} graphs, "
            f"mean={np.mean(times):.5f}s, max={np.max(times):.5f}s"
        )
    return results_per_dataset


# ═══════════════════════════════════════════════════════════════════════════════
#  PART B: FORMULATION COMPARISON — NDS vs HETERODYNING on 1-WL pairs
# ═══════════════════════════════════════════════════════════════════════════════

def parse_graph_pair(example: dict) -> tuple[dict, dict, dict]:
    """Parse a 1-WL pair example into (pair_info, graph_a_data, graph_b_data)."""
    pair_info = json.loads(example["input"])
    graph_a = pair_info["graph_a"]
    graph_b = pair_info["graph_b"]
    return pair_info, graph_a, graph_b


def build_adjacency_and_init(
    graph_data: dict,
    init_strategy: str = "onehot",
) -> tuple[csr_matrix, np.ndarray]:
    """Build normalized adjacency and initial features from graph data."""
    n = graph_data["num_nodes"]
    if "adjacency_matrix" in graph_data:
        A_norm = build_normalized_adjacency_from_dense(graph_data["adjacency_matrix"])
    else:
        A_norm = build_normalized_adjacency_from_edge_list(
            graph_data["edge_list"], n
        )
    if init_strategy == "onehot":
        x_init = init_onehot(n)
    elif init_strategy == "degree":
        x_init = init_degree(A_norm, n)
    elif init_strategy == "random":
        x_init = init_random(n)
    elif init_strategy == "local_structure":
        x_init = init_local_structure(A_norm, n)
    elif init_strategy == "spectral":
        x_init = init_spectral(A_norm, n)
    elif init_strategy == "random_lowdim":
        x_init = init_random_lowdim(n)
    else:
        raise ValueError(f"Unknown init strategy: {init_strategy}")
    return A_norm, x_init


def run_formulation_comparison(
    pairs_data: list[dict],
    T_values: list[int] | None = None,
    K_values: list[int] | None = None,
) -> list[dict]:
    """Compare NDS vs spectral heterodyning on 1-WL-equivalent pairs.

    Tests with BOTH degree and one-hot initialization.
    For each pair, for each init strategy:
      - Linear baseline at each T
      - NDS-ReLU at each T
      - NDS-tanh at each T
      - NDS-abs at each T (additional)
      - Spectral heterodyning at each K
    """
    if T_values is None:
        T_values = [5, 10, 15, 20]
    if K_values is None:
        K_values = [5, 10]

    results: list[dict] = []

    for pair_idx, ex in enumerate(pairs_data):
        pair_info, graph_a, graph_b = parse_graph_pair(ex)
        pair_id = pair_info["pair_id"]
        category = pair_info["category"]
        n = graph_a["num_nodes"]
        logger.info(f"Pair {pair_idx}: {pair_id} (n={n}, cat={category})")

        pair_result: dict = {
            "pair_idx": pair_idx,
            "pair_id": pair_id,
            "category": category,
            "n": n,
        }

        # ── Test with ALL init strategies ──
        for init_strat in ["degree", "onehot", "local_structure", "spectral", "random_lowdim"]:
            A1_norm, x1 = build_adjacency_and_init(graph_a, init_strategy=init_strat)
            A2_norm, x2 = build_adjacency_and_init(graph_b, init_strategy=init_strat)
            prefix = init_strat

            # ── Linear baseline ──
            for T in T_values:
                t0 = time.perf_counter()
                lin1 = compute_linear_diffusion(A1_norm, x1, T)
                lin2 = compute_linear_diffusion(A2_norm, x2, T)
                t1 = time.perf_counter()
                dist = graphs_distinguished(lin1, lin2)
                pair_result[f"{prefix}_linear_T{T}_distinguished"] = dist
                pair_result[f"{prefix}_linear_T{T}_time"] = round(t1 - t0, 6)

            # ── NDS with ReLU ──
            for T in T_values:
                t0 = time.perf_counter()
                nds1 = compute_nds(A1_norm, x1, T, "relu")
                nds2 = compute_nds(A2_norm, x2, T, "relu")
                t1 = time.perf_counter()
                dist = graphs_distinguished(nds1, nds2)
                pair_result[f"{prefix}_nds_relu_T{T}_distinguished"] = dist
                pair_result[f"{prefix}_nds_relu_T{T}_time"] = round(t1 - t0, 6)

            # ── NDS with tanh ──
            for T in T_values:
                t0 = time.perf_counter()
                nds1 = compute_nds(A1_norm, x1, T, "tanh")
                nds2 = compute_nds(A2_norm, x2, T, "tanh")
                t1 = time.perf_counter()
                dist = graphs_distinguished(nds1, nds2)
                pair_result[f"{prefix}_nds_tanh_T{T}_distinguished"] = dist
                pair_result[f"{prefix}_nds_tanh_T{T}_time"] = round(t1 - t0, 6)

            # ── NDS with abs (additional nonlinearity) ──
            for T in T_values:
                t0 = time.perf_counter()
                nds1 = compute_nds(A1_norm, x1, T, "abs")
                nds2 = compute_nds(A2_norm, x2, T, "abs")
                t1 = time.perf_counter()
                dist = graphs_distinguished(nds1, nds2)
                pair_result[f"{prefix}_nds_abs_T{T}_distinguished"] = dist
                pair_result[f"{prefix}_nds_abs_T{T}_time"] = round(t1 - t0, 6)

            # ── Spectral heterodyning ──
            for K in K_values:
                t0 = time.perf_counter()
                poly1 = compute_polynomial_filters(A1_norm, x1, K)
                het1 = compute_heterodyning_features(poly1)
                poly2 = compute_polynomial_filters(A2_norm, x2, K)
                het2 = compute_heterodyning_features(poly2)
                t1 = time.perf_counter()
                dist = graphs_distinguished(het1, het2)
                pair_result[f"{prefix}_hetero_K{K}_distinguished"] = dist
                pair_result[f"{prefix}_hetero_K{K}_time"] = round(t1 - t0, 6)

        results.append(pair_result)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION & SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_distinguishing_power(
    comparison_results: list[dict],
    T_values: list[int] | None = None,
    K_values: list[int] | None = None,
) -> dict:
    """Count how many pairs each method distinguishes."""
    if T_values is None:
        T_values = [5, 10, 15, 20]
    if K_values is None:
        K_values = [5, 10]

    total = len(comparison_results)
    summary: dict = {"total_pairs": total}

    for init_strat in ["degree", "onehot", "local_structure", "spectral", "random_lowdim"]:
        prefix = init_strat
        for T in T_values:
            key = f"{prefix}_linear_T{T}"
            count = sum(
                1 for r in comparison_results if r.get(f"{key}_distinguished", False)
            )
            summary[f"{key}_count"] = count

        for nl in ["relu", "tanh", "abs"]:
            for T in T_values:
                key = f"{prefix}_nds_{nl}_T{T}"
                count = sum(
                    1 for r in comparison_results
                    if r.get(f"{key}_distinguished", False)
                )
                summary[f"{key}_count"] = count

        for K in K_values:
            key = f"{prefix}_hetero_K{K}"
            count = sum(
                1 for r in comparison_results if r.get(f"{key}_distinguished", False)
            )
            summary[f"{key}_count"] = count

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT BUILDER — exp_gen_sol_out schema
# ═══════════════════════════════════════════════════════════════════════════════

def build_method_output(
    scale_results: list[dict],
    r_squared: float,
    slope: float,
    bench_timing: dict[str, dict],
    comparison_results: list[dict],
    summary: dict,
    T_values: list[int],
    K_values: list[int],
) -> dict:
    """Build output conforming to exp_gen_sol_out schema."""
    output: dict = {
        "metadata": {
            "method_name": "NDS Scalability & Spectral Heterodyning Comparison",
            "T_values": T_values,
            "K_values": K_values,
            "nonlinearities": ["relu", "tanh", "abs"],
            "init_strategies": ["degree", "onehot", "local_structure", "spectral", "random_lowdim"],
            "r_squared_linear_fit": round(r_squared, 6),
            "scaling_slope": slope,
            "summary": summary,
        },
        "datasets": [],
    }

    # ── Dataset 1: scalability_synthetic ──
    scale_examples: list[dict] = []
    for r in scale_results:
        scale_examples.append({
            "input": json.dumps({
                "n": r["n"],
                "m": r["m"],
                "T": r["T"],
                "graph_type": "erdos_renyi",
            }),
            "output": str(round(r["time_sec"], 6)),
            "predict_nds_time_sec": str(round(r["time_sec"], 6)),
            "predict_nds_peak_memory_MB": str(round(r["peak_memory_MB"], 2)),
            "metadata_mT": r["mT"],
            "metadata_n": r["n"],
            "metadata_m": r["m"],
            "metadata_all_times": json.dumps(r["all_times"]),
        })
    output["datasets"].append({
        "dataset": "scalability_synthetic",
        "examples": scale_examples,
    })

    # ── Dataset 2: scalability_benchmarks ──
    bench_examples: list[dict] = []
    for ds_name, stats in bench_timing.items():
        bench_examples.append({
            "input": json.dumps({
                "dataset": ds_name,
                "num_graphs_sampled": stats["num_graphs"],
                "avg_nodes": round(stats["avg_nodes"], 1),
                "avg_edges": round(stats["avg_edges"], 1),
            }),
            "output": str(round(stats["mean_time"], 6)),
            "predict_nds_mean_time": str(round(stats["mean_time"], 6)),
            "predict_nds_max_time": str(round(stats["max_time"], 6)),
            "predict_nds_std_time": str(round(stats["std_time"], 6)),
            "metadata_dataset": ds_name,
            "metadata_num_graphs": stats["num_graphs"],
        })
    output["datasets"].append({
        "dataset": "scalability_benchmarks",
        "examples": bench_examples,
    })

    # ── Dataset 3: 1wl_pair_distinguishing ──
    pair_examples: list[dict] = []
    for r in comparison_results:
        ex: dict = {
            "input": json.dumps({
                "pair_idx": r["pair_idx"],
                "pair_id": r["pair_id"],
                "category": r["category"],
                "n": r["n"],
            }),
            # Main output: whether NDS-ReLU with onehot init at T=10 distinguishes
            "output": str(r.get("onehot_nds_relu_T10_distinguished", False)).lower(),
            "metadata_pair_id": r["pair_id"],
            "metadata_category": r["category"],
            "metadata_n": r["n"],
        }
        # Add all prediction fields
        for init_strat in ["degree", "onehot", "local_structure", "spectral", "random_lowdim"]:
            prefix = init_strat
            for T in T_values:
                k = f"{prefix}_linear_T{T}"
                ex[f"predict_{k}"] = str(r.get(f"{k}_distinguished", False)).lower()
            for nl in ["relu", "tanh", "abs"]:
                for T in T_values:
                    k = f"{prefix}_nds_{nl}_T{T}"
                    ex[f"predict_{k}"] = str(
                        r.get(f"{k}_distinguished", False)
                    ).lower()
            for K in K_values:
                k = f"{prefix}_hetero_K{K}"
                ex[f"predict_{k}"] = str(r.get(f"{k}_distinguished", False)).lower()

        # Add timing metadata for key methods (onehot init)
        for T in T_values:
            t_key = f"onehot_nds_relu_T{T}_time"
            if t_key in r:
                ex[f"metadata_nds_relu_T{T}_time"] = r[t_key]
        for K in K_values:
            t_key = f"onehot_hetero_K{K}_time"
            if t_key in r:
                ex[f"metadata_hetero_K{K}_time"] = r[t_key]

        pair_examples.append(ex)

    output["datasets"].append({
        "dataset": "1wl_pair_distinguishing",
        "examples": pair_examples,
    })

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dependency(dep_path: Path, max_examples: int | None = None) -> dict:
    """Load a dependency JSON file, optionally limiting examples."""
    full_path = dep_path / "full_data_out.json"
    mini_path = dep_path / "mini_data_out.json"

    # Use mini for quick testing, full otherwise
    if max_examples is not None and max_examples <= 3:
        path = mini_path
    else:
        path = full_path

    logger.info(f"Loading {path}")
    data = json.loads(path.read_text())

    if max_examples is not None:
        for ds in data["datasets"]:
            ds["examples"] = ds["examples"][:max_examples]
        logger.info(f"Limited to {max_examples} examples per dataset")

    total_ex = sum(len(ds["examples"]) for ds in data["datasets"])
    logger.info(f"Loaded {len(data['datasets'])} datasets, {total_ex} total examples")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main() -> None:
    t_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("NDS Scalability & Spectral Heterodyning Comparison")
    logger.info("=" * 60)

    # ── Parse optional max_examples from argv ──
    global MAX_EXAMPLES
    if len(sys.argv) > 1:
        MAX_EXAMPLES = int(sys.argv[1])
        logger.info(f"MAX_EXAMPLES set to {MAX_EXAMPLES}")

    # ── Load dependencies ──
    pairs_data = load_dependency(DEP_1WL, max_examples=MAX_EXAMPLES)
    benchmark_data = load_dependency(DEP_BENCH, max_examples=MAX_EXAMPLES)

    T_values = [5, 10, 15, 20]
    K_values = [5, 10]

    # ── PART A: Scalability ──
    logger.info("=" * 60)
    logger.info("PART A: Scalability Experiment")
    logger.info("=" * 60)
    scale_results, r_squared, slope = run_scalability_experiment()
    bench_timing = run_benchmark_timing(benchmark_data, T=10)

    # ── PART B: Formulation Comparison ──
    logger.info("=" * 60)
    logger.info("PART B: NDS vs Heterodyning on 1-WL pairs")
    logger.info("=" * 60)
    all_pairs = pairs_data["datasets"][0]["examples"]
    comparison_results = run_formulation_comparison(
        all_pairs,
        T_values=T_values,
        K_values=K_values,
    )

    # ── Aggregate ──
    summary = aggregate_distinguishing_power(
        comparison_results,
        T_values=T_values,
        K_values=K_values,
    )
    logger.info("Distinguishing power summary:")
    for k, v in sorted(summary.items()):
        logger.info(f"  {k}: {v}")

    # ── Build output ──
    output = build_method_output(
        scale_results=scale_results,
        r_squared=r_squared,
        slope=slope,
        bench_timing=bench_timing,
        comparison_results=comparison_results,
        summary=summary,
        T_values=T_values,
        K_values=K_values,
    )

    # ── Write output ──
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    t_end = time.perf_counter()
    logger.info(f"Total runtime: {t_end - t_start:.1f}s")
    logger.success("Done.")


if __name__ == "__main__":
    main()
