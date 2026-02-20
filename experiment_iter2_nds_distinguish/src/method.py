#!/usr/bin/env python3
"""
NDS Distinguishability on 1-WL-Equivalent Graph Pairs: Comprehensive Ablation Study.

Computes Nonlinear Diffusion Signatures (NDS) on 9 verified 1-WL-equivalent graph pairs
across 5 nonlinearities (linear/relu/tanh/abs/square), 3 initialization strategies
(degree/adjacency_row/random), and T=0..20 diffusion rounds. Tests whether interleaved
nonlinearity creates features that distinguish graphs where linear diffusion alone fails.

Output: method_out.json in exp_gen_sol_out schema with predict_ fields.
"""

import json
import hashlib
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse
from scipy.spatial.distance import cdist
from loguru import logger

# ---------------------------------------------------------------------------
# Resource limits (14 GB RAM, 1-hour CPU)
# ---------------------------------------------------------------------------
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
DATA_PATH = WORKSPACE / "dependencies" / "data_id3_it1__opus" / "full_data_out.json"

T_VALUES = list(range(0, 21))  # T=0..20 — full range

NONLINEARITIES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "linear": lambda x: x,
    "relu": lambda x: np.maximum(x, 0),
    "tanh": lambda x: np.tanh(x),
    "abs": lambda x: np.abs(x),
    "square": lambda x: x ** 2,
}

# Additional nonlinearities for extended analysis
EXTRA_NONLINEARITIES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
    "softplus": lambda x: np.log1p(np.exp(np.clip(x, -50, 50))),
}

# Additional diffusion operators for fallback/extended analysis
DIFFUSION_OPERATORS = ["symmetric_norm", "row_norm", "unnormalized", "laplacian"]


# =========================================================================
# STEP 1: Data Loading
# =========================================================================
def load_graph_pairs(data_path: Path) -> list[dict[str, Any]]:
    """Load all graph pairs from the dataset JSON."""
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    examples = raw["datasets"][0]["examples"]
    logger.info(f"Found {len(examples)} graph pairs")

    pairs = []
    for ex in examples:
        inp = json.loads(ex["input"])
        adj_a = np.array(inp["graph_a"]["adjacency_matrix"], dtype=np.float64)
        adj_b = np.array(inp["graph_b"]["adjacency_matrix"], dtype=np.float64)
        n = adj_a.shape[0]

        # Sanity checks
        assert adj_a.shape == (n, n), f"adj_a shape mismatch: {adj_a.shape}"
        assert adj_b.shape == (n, n), f"adj_b shape mismatch: {adj_b.shape}"
        assert np.allclose(adj_a, adj_a.T), f"adj_a not symmetric for {inp['pair_id']}"
        assert np.allclose(adj_b, adj_b.T), f"adj_b not symmetric for {inp['pair_id']}"
        assert np.allclose(np.diag(adj_a), 0), f"adj_a non-zero diagonal for {inp['pair_id']}"
        assert np.allclose(np.diag(adj_b), 0), f"adj_b non-zero diagonal for {inp['pair_id']}"

        pairs.append({
            "pair_id": inp["pair_id"],
            "category": inp["category"],
            "adj_a": adj_a,
            "adj_b": adj_b,
            "n": n,
            "num_edges_a": int(adj_a.sum()) // 2,
            "num_edges_b": int(adj_b.sum()) // 2,
            "example": ex,
            "graph_a_name": inp["graph_a"]["name"],
            "graph_b_name": inp["graph_b"]["name"],
        })
        logger.info(
            f"  Pair {inp['pair_id']}: n={n}, "
            f"edges_a={int(adj_a.sum())//2}, edges_b={int(adj_b.sum())//2}, "
            f"category={inp['category']}"
        )

    return pairs


# =========================================================================
# STEP 2: Core NDS Computation
# =========================================================================
def compute_diffusion_operator(
    adj: np.ndarray,
    operator_type: str = "symmetric_norm",
) -> np.ndarray:
    """
    Compute diffusion operator from adjacency matrix.

    Parameters
    ----------
    adj : np.ndarray
        (n, n) adjacency matrix.
    operator_type : str
        One of 'symmetric_norm', 'row_norm', 'unnormalized', 'laplacian'.

    Returns
    -------
    np.ndarray
        (n, n) dense diffusion operator matrix.
    """
    n = adj.shape[0]
    deg = adj.sum(axis=1)

    if operator_type == "symmetric_norm":
        # D^{-1/2} A D^{-1/2}
        d_inv_sqrt = np.zeros(n)
        nonzero = deg > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt

    elif operator_type == "row_norm":
        # D^{-1} A  (random walk normalization)
        d_inv = np.zeros(n)
        nonzero = deg > 0
        d_inv[nonzero] = 1.0 / deg[nonzero]
        D_inv = np.diag(d_inv)
        return D_inv @ adj

    elif operator_type == "unnormalized":
        return adj.copy()

    elif operator_type == "laplacian":
        # I - D^{-1/2} A D^{-1/2}
        d_inv_sqrt = np.zeros(n)
        nonzero = deg > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        A_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        return np.eye(n) - A_norm

    else:
        raise ValueError(f"Unknown operator type: {operator_type}")


def init_features(
    adj: np.ndarray,
    strategy: str,
) -> np.ndarray:
    """
    Initialize node features based on strategy.

    Parameters
    ----------
    adj : np.ndarray
        (n, n) adjacency matrix.
    strategy : str
        One of 'degree', 'adjacency_row', 'random'.

    Returns
    -------
    np.ndarray
        (n,) or (n, d) initial feature matrix.
    """
    n = adj.shape[0]

    if strategy == "degree":
        return adj.sum(axis=1).astype(np.float64)  # (n,)

    elif strategy == "adjacency_row":
        return adj.astype(np.float64)  # (n, n)

    elif strategy == "random":
        rng = np.random.RandomState(42)
        return rng.randn(n).astype(np.float64)  # (n,)

    else:
        raise ValueError(f"Unknown init strategy: {strategy}")


def _safe_normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize features to prevent numerical overflow.

    Divides by the max absolute value (row-wise for 2D, globally for 1D)
    when values exceed a threshold. This preserves relative differences
    between feature vectors while keeping values in a numerically stable range.
    """
    max_val = np.max(np.abs(x))
    if max_val > 1e30 or np.any(np.isinf(x)) or np.any(np.isnan(x)):
        # Replace inf/nan with 0, then normalize
        x = np.where(np.isfinite(x), x, 0.0)
        max_val = np.max(np.abs(x))
        if max_val > 0:
            x = x / max_val
    return x


def compute_nds(
    diffusion_op: np.ndarray,
    x0: np.ndarray,
    nonlinearity_fn: Callable[[np.ndarray], np.ndarray],
    T: int,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute NDS trajectory.

    Parameters
    ----------
    diffusion_op : np.ndarray
        (n, n) diffusion operator.
    x0 : np.ndarray
        (n,) or (n, d) initial features.
    nonlinearity_fn : callable
        Pointwise nonlinearity.
    T : int
        Number of diffusion-nonlinearity rounds.
    normalize : bool
        Whether to normalize features after each step to prevent overflow.

    Returns
    -------
    np.ndarray
        (n, cols) trajectory matrix.
        For 1D init: (n, T+1).
        For 2D init: (n, d*(T+1)).
    """
    trajectory = [x0.copy()]
    x = x0.copy()

    for _ in range(T):
        x = diffusion_op @ x  # diffusion step
        x = nonlinearity_fn(x)  # nonlinearity step
        if normalize:
            x = _safe_normalize(x)
        trajectory.append(x.copy())

    if x0.ndim == 1:
        return np.column_stack(trajectory)  # (n, T+1)
    else:
        return np.concatenate(trajectory, axis=1)  # (n, d*(T+1))


# =========================================================================
# STEP 3: Distinguishability Metrics
# =========================================================================
def _sanitize_matrix(X: np.ndarray) -> np.ndarray:
    """Replace inf/nan with 0 to prevent comparison failures."""
    return np.where(np.isfinite(X), X, 0.0)


def sort_rows_lexicographic(X: np.ndarray, decimals: int = 10) -> np.ndarray:
    """Sort rows of X lexicographically after rounding."""
    X_clean = _sanitize_matrix(X)
    X_rounded = np.round(X_clean, decimals)
    if X_rounded.ndim == 1:
        X_rounded = X_rounded.reshape(-1, 1)
    # Lexicographic sort: sort by last column first, then second last, etc.
    indices = np.lexsort(X_rounded.T[::-1])
    return X_rounded[indices]


def exact_multiset_match(nds_a: np.ndarray, nds_b: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if sorted NDS feature matrices are identical within tolerance."""
    sorted_a = sort_rows_lexicographic(nds_a)
    sorted_b = sort_rows_lexicographic(nds_b)
    return bool(np.allclose(sorted_a, sorted_b, atol=tol))


def frobenius_distance(nds_a: np.ndarray, nds_b: np.ndarray) -> float:
    """Frobenius norm of difference between sorted feature matrices."""
    sorted_a = sort_rows_lexicographic(nds_a)
    sorted_b = sort_rows_lexicographic(nds_b)
    return float(np.linalg.norm(sorted_a - sorted_b))


def compute_mmd_rbf(X: np.ndarray, Y: np.ndarray, bandwidth: float | None = None) -> float:
    """
    Compute MMD^2 with RBF kernel between two sets of row vectors.

    Parameters
    ----------
    X, Y : np.ndarray
        (n, d) feature matrices.
    bandwidth : float or None
        RBF bandwidth. If None, uses median heuristic.

    Returns
    -------
    float
        MMD^2 value.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Sanitize inputs
    X = _sanitize_matrix(X)
    Y = _sanitize_matrix(Y)

    if bandwidth is None:
        all_pts = np.vstack([X, Y])
        all_dists = cdist(all_pts, all_pts, "sqeuclidean")
        pos_dists = all_dists[all_dists > 0]
        if len(pos_dists) == 0:
            bandwidth = 1.0
        else:
            bandwidth = float(np.sqrt(np.median(pos_dists)))
        if bandwidth < 1e-12:
            bandwidth = 1.0

    gamma = 1.0 / (2.0 * bandwidth ** 2)

    # Clip exponent arguments to prevent underflow to exactly 0
    d_xx = cdist(X, X, "sqeuclidean")
    d_yy = cdist(Y, Y, "sqeuclidean")
    d_xy = cdist(X, Y, "sqeuclidean")

    K_xx = np.exp(-gamma * np.clip(d_xx, 0, 500 / max(gamma, 1e-12)))
    K_yy = np.exp(-gamma * np.clip(d_yy, 0, 500 / max(gamma, 1e-12)))
    K_xy = np.exp(-gamma * np.clip(d_xy, 0, 500 / max(gamma, 1e-12)))

    mmd2 = float(K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean())
    return mmd2


def graph_level_hash(nds: np.ndarray, decimals: int = 8) -> str:
    """Compute hash of sorted, rounded feature matrix (NaN/inf safe)."""
    sorted_m = sort_rows_lexicographic(nds, decimals=decimals)
    # sort_rows_lexicographic already sanitizes via _sanitize_matrix
    return hashlib.sha256(sorted_m.tobytes()).hexdigest()


# =========================================================================
# STEP 4: Self-Consistency Check
# =========================================================================
def self_consistency_check(pairs: list[dict]) -> None:
    """Verify NDS gives identical features when comparing a graph to itself."""
    pair = pairs[0]
    adj = pair["adj_a"]
    diff_op = compute_diffusion_operator(adj, operator_type="symmetric_norm")
    x0 = init_features(adj, strategy="adjacency_row")
    nds = compute_nds(
        diffusion_op=diff_op,
        x0=x0,
        nonlinearity_fn=NONLINEARITIES["relu"],
        T=10,
    )
    frob = frobenius_distance(nds, nds)
    is_match = exact_multiset_match(nds, nds)
    logger.info(f"Self-consistency check: frob={frob:.2e}, match={is_match}")
    assert frob < 1e-12, f"Self-consistency FAILED: frob={frob}"
    assert is_match, "Self-consistency FAILED: multiset mismatch"
    logger.info("Self-consistency check PASSED")


# =========================================================================
# STEP 5: Main Experiment Loop
# =========================================================================
def compute_nds_final_step(
    diffusion_op: np.ndarray,
    x0: np.ndarray,
    nonlinearity_fn: Callable[[np.ndarray], np.ndarray],
    T: int,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute NDS features at ONLY step T (not the full trajectory).

    This is useful for analyzing whether the nonlinearity helps at each
    individual diffusion step, separate from the trajectory accumulation.
    """
    x = x0.copy()
    for _ in range(T):
        x = diffusion_op @ x
        x = nonlinearity_fn(x)
        if normalize:
            x = _safe_normalize(x)
    return x


def run_single_config(
    pair: dict,
    nonlin_name: str,
    init_name: str,
    T: int,
    diffusion_type: str = "symmetric_norm",
) -> dict[str, Any]:
    """Run NDS for one (pair, nonlinearity, init, T, diffusion_type) config."""
    adj_a = pair["adj_a"]
    adj_b = pair["adj_b"]

    # Select nonlinearity
    all_nonlins = {**NONLINEARITIES, **EXTRA_NONLINEARITIES}
    nonlin_fn = all_nonlins[nonlin_name]

    # Compute diffusion operators
    diff_op_a = compute_diffusion_operator(adj_a, operator_type=diffusion_type)
    diff_op_b = compute_diffusion_operator(adj_b, operator_type=diffusion_type)

    # Initialize features
    x0_a = init_features(adj_a, strategy=init_name)
    x0_b = init_features(adj_b, strategy=init_name)

    # Compute full NDS trajectory
    nds_a = compute_nds(diff_op_a, x0_a, nonlin_fn, T)
    nds_b = compute_nds(diff_op_b, x0_b, nonlin_fn, T)

    # Trajectory-based metrics (all steps concatenated)
    is_identical = exact_multiset_match(nds_a, nds_b)
    distinguished = not is_identical
    frob = frobenius_distance(nds_a, nds_b)
    mmd = compute_mmd_rbf(nds_a, nds_b)
    hash_a = graph_level_hash(nds_a)
    hash_b = graph_level_hash(nds_b)
    hash_distinguished = hash_a != hash_b

    # Single-step metrics (features at ONLY step T, not trajectory)
    if T > 0:
        final_a = compute_nds_final_step(diff_op_a, x0_a, nonlin_fn, T)
        final_b = compute_nds_final_step(diff_op_b, x0_b, nonlin_fn, T)
        step_distinguished = not exact_multiset_match(final_a, final_b)
        step_frob = frobenius_distance(final_a, final_b)
    else:
        # T=0 means just the initial features
        step_distinguished = not exact_multiset_match(x0_a, x0_b)
        step_frob = frobenius_distance(x0_a, x0_b)

    return {
        "pair_id": pair["pair_id"],
        "category": pair["category"],
        "nonlinearity": nonlin_name,
        "init": init_name,
        "T": T,
        "diffusion_type": diffusion_type,
        "distinguished": distinguished,
        "frobenius_distance": frob,
        "mmd_rbf": mmd,
        "hash_distinguished": hash_distinguished,
        "step_distinguished": step_distinguished,
        "step_frobenius": step_frob,
        "num_nodes": pair["n"],
        "num_edges": pair["num_edges_a"],
    }


def run_experiment(pairs: list[dict]) -> list[dict[str, Any]]:
    """Run the full ablation experiment."""
    t0 = time.time()

    init_strategies = ["degree", "adjacency_row", "random"]
    nonlinearities_main = list(NONLINEARITIES.keys())  # 5 main

    total_configs = len(pairs) * len(nonlinearities_main) * len(init_strategies) * len(T_VALUES)
    logger.info(f"Main grid: {len(pairs)} pairs x {len(nonlinearities_main)} nonlins "
                f"x {len(init_strategies)} inits x {len(T_VALUES)} T values = {total_configs}")

    # -----------------------------------------------------------------------
    # Phase 1: Quick sanity — one pair, one config
    # -----------------------------------------------------------------------
    logger.info("--- Phase 1: Single-pair sanity check ---")
    test_result = run_single_config(
        pair=pairs[0],
        nonlin_name="relu",
        init_name="adjacency_row",
        T=5,
        diffusion_type="symmetric_norm",
    )
    logger.info(
        f"Sanity result: pair={test_result['pair_id']}, relu/adj_row/T=5 → "
        f"distinguished={test_result['distinguished']}, "
        f"frob={test_result['frobenius_distance']:.6e}, "
        f"mmd={test_result['mmd_rbf']:.6e}"
    )
    phase1_time = time.time() - t0
    logger.info(f"Phase 1 time: {phase1_time:.2f}s")

    # -----------------------------------------------------------------------
    # Phase 2: All pairs with relu + adjacency_row, T=5
    # -----------------------------------------------------------------------
    logger.info("--- Phase 2: All pairs, relu/adj_row/T=5 ---")
    phase2_results = []
    for pair in pairs:
        r = run_single_config(
            pair=pair,
            nonlin_name="relu",
            init_name="adjacency_row",
            T=5,
            diffusion_type="symmetric_norm",
        )
        phase2_results.append(r)
        logger.info(f"  {r['pair_id']}: distinguished={r['distinguished']}, frob={r['frobenius_distance']:.6e}")
    phase2_time = time.time() - t0
    n_dist = sum(1 for r in phase2_results if r["distinguished"])
    logger.info(f"Phase 2: {n_dist}/{len(pairs)} pairs distinguished. Time: {phase2_time:.2f}s")

    # -----------------------------------------------------------------------
    # Phase 3: Full main grid
    # -----------------------------------------------------------------------
    logger.info("--- Phase 3: Full main grid ---")
    results: list[dict[str, Any]] = []
    count = 0

    for pair in pairs:
        for init_name in init_strategies:
            for nonlin_name in nonlinearities_main:
                for T in T_VALUES:
                    r = run_single_config(
                        pair=pair,
                        nonlin_name=nonlin_name,
                        init_name=init_name,
                        T=T,
                        diffusion_type="symmetric_norm",
                    )
                    results.append(r)
                    count += 1

        logger.info(f"  Completed pair {pair['pair_id']} ({count}/{total_configs})")

    phase3_time = time.time() - t0
    logger.info(f"Phase 3 complete: {count} configs in {phase3_time:.2f}s")

    # -----------------------------------------------------------------------
    # Phase 4: Extended analysis — extra nonlinearities on best init
    # -----------------------------------------------------------------------
    logger.info("--- Phase 4: Extra nonlinearities (leaky_relu, softplus) ---")
    extra_nonlins = list(EXTRA_NONLINEARITIES.keys())
    t_subset = [0, 1, 2, 3, 5, 10, 15, 20]
    for pair in pairs:
        for nonlin_name in extra_nonlins:
            for T in t_subset:
                r = run_single_config(
                    pair=pair,
                    nonlin_name=nonlin_name,
                    init_name="adjacency_row",
                    T=T,
                    diffusion_type="symmetric_norm",
                )
                results.append(r)

    phase4_time = time.time() - t0
    logger.info(f"Phase 4 complete. Total results: {len(results)}. Time: {phase4_time:.2f}s")

    # -----------------------------------------------------------------------
    # Phase 5: Extended analysis — alternative diffusion operators
    # -----------------------------------------------------------------------
    logger.info("--- Phase 5: Alternative diffusion operators ---")
    alt_ops = ["row_norm", "unnormalized", "laplacian"]
    for pair in pairs:
        for diff_type in alt_ops:
            for T in t_subset:
                r = run_single_config(
                    pair=pair,
                    nonlin_name="relu",
                    init_name="adjacency_row",
                    T=T,
                    diffusion_type=diff_type,
                )
                results.append(r)

    phase5_time = time.time() - t0
    logger.info(f"Phase 5 complete. Total results: {len(results)}. Time: {phase5_time:.2f}s")

    # -----------------------------------------------------------------------
    # Timing measurements per pair
    # -----------------------------------------------------------------------
    logger.info("--- Timing measurements (relu, adj_row, T=10) ---")
    for pair in pairs:
        t_start = time.perf_counter()
        _ = run_single_config(
            pair=pair,
            nonlin_name="relu",
            init_name="adjacency_row",
            T=10,
            diffusion_type="symmetric_norm",
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        time_per_edge = elapsed_ms / max(pair["num_edges_a"], 1)
        logger.info(
            f"  {pair['pair_id']}: {elapsed_ms:.2f}ms total, "
            f"{time_per_edge:.4f}ms/edge (n={pair['n']}, m={pair['num_edges_a']})"
        )

    total_time = time.time() - t0
    logger.info(f"Total experiment time: {total_time:.2f}s")

    return results


# =========================================================================
# STEP 6: Summary Statistics & Analysis
# =========================================================================
def compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from experiment results."""

    # Filter to main grid only (symmetric_norm, main 5 nonlinearities)
    main_nonlins = set(NONLINEARITIES.keys())
    main_results = [
        r for r in results
        if r["diffusion_type"] == "symmetric_norm" and r["nonlinearity"] in main_nonlins
    ]

    # --- Per (nonlinearity, init) at T=10: how many of 9 pairs distinguished? ---
    summary_table: dict[str, dict[str, int]] = {}
    for init_name in ["degree", "adjacency_row", "random"]:
        for nonlin_name in main_nonlins:
            key = f"{nonlin_name}/{init_name}"
            count = sum(
                1 for r in main_results
                if r["nonlinearity"] == nonlin_name
                and r["init"] == init_name
                and r["T"] == 10
                and r["distinguished"]
            )
            summary_table[key] = {"pairs_distinguished_at_T10": count}

    # --- Best config ---
    best_config = max(summary_table.items(), key=lambda x: x[1]["pairs_distinguished_at_T10"])
    best_config_name = best_config[0]
    best_config_count = best_config[1]["pairs_distinguished_at_T10"]

    # --- Linear baseline count ---
    linear_degree_count = sum(
        1 for r in main_results
        if r["nonlinearity"] == "linear" and r["init"] == "degree"
        and r["T"] == 10 and r["distinguished"]
    )
    linear_adj_count = sum(
        1 for r in main_results
        if r["nonlinearity"] == "linear" and r["init"] == "adjacency_row"
        and r["T"] == 10 and r["distinguished"]
    )

    # --- Minimum T per pair for best nonlinear config ---
    min_T_per_pair: dict[str, int | None] = {}
    for r in main_results:
        if r["nonlinearity"] == "relu" and r["init"] == "adjacency_row" and r["diffusion_type"] == "symmetric_norm":
            pid = r["pair_id"]
            if r["distinguished"]:
                prev = min_T_per_pair.get(pid)
                if prev is None or r["T"] < prev:
                    min_T_per_pair[pid] = r["T"]
            else:
                if pid not in min_T_per_pair:
                    min_T_per_pair[pid] = None

    # --- Nonlinearity comparison at T=10, adjacency_row (TRAJECTORY-based) ---
    nonlin_comparison: dict[str, int] = {}
    for nonlin_name in main_nonlins:
        c = sum(
            1 for r in main_results
            if r["nonlinearity"] == nonlin_name
            and r["init"] == "adjacency_row"
            and r["T"] == 10
            and r["distinguished"]
        )
        nonlin_comparison[nonlin_name] = c

    # --- Single-step comparison at T=10, adjacency_row ---
    # This measures whether features at EXACTLY step T distinguish,
    # not the accumulated trajectory. More informative for the NDS hypothesis.
    step_nonlin_comparison: dict[str, int] = {}
    for nonlin_name in main_nonlins:
        c = sum(
            1 for r in main_results
            if r["nonlinearity"] == nonlin_name
            and r["init"] == "adjacency_row"
            and r["T"] == 10
            and r["step_distinguished"]
        )
        step_nonlin_comparison[nonlin_name] = c

    # --- Average frobenius distance per nonlinearity (adj_row, T=10) ---
    frob_by_nonlin: dict[str, float] = {}
    for nonlin_name in main_nonlins:
        frobs = [
            r["frobenius_distance"] for r in main_results
            if r["nonlinearity"] == nonlin_name
            and r["init"] == "adjacency_row"
            and r["T"] == 10
        ]
        frob_by_nonlin[nonlin_name] = float(np.mean(frobs)) if frobs else 0.0

    # --- Step-level Frobenius ---
    step_frob_by_nonlin: dict[str, float] = {}
    for nonlin_name in main_nonlins:
        frobs = [
            r["step_frobenius"] for r in main_results
            if r["nonlinearity"] == nonlin_name
            and r["init"] == "adjacency_row"
            and r["T"] == 10
        ]
        step_frob_by_nonlin[nonlin_name] = float(np.mean(frobs)) if frobs else 0.0

    return {
        "pairs_distinguished_by_best_nds_config": best_config_count,
        "best_nds_config": best_config_name,
        "pairs_distinguished_by_linear_baseline_degree": linear_degree_count,
        "pairs_distinguished_by_linear_baseline_adj_row": linear_adj_count,
        "nonlinearity_comparison_trajectory_T10_adj_row": nonlin_comparison,
        "nonlinearity_comparison_single_step_T10_adj_row": step_nonlin_comparison,
        "avg_frobenius_trajectory_by_nonlin_T10_adj_row": frob_by_nonlin,
        "avg_frobenius_single_step_by_nonlin_T10_adj_row": step_frob_by_nonlin,
        "min_T_relu_adj_row_per_pair": {k: v for k, v in min_T_per_pair.items()},
        "summary_table": summary_table,
    }


def print_summary_table(results: list[dict[str, Any]]) -> None:
    """Print a readable summary table."""
    main_nonlins = list(NONLINEARITIES.keys())
    init_strategies = ["degree", "adjacency_row", "random"]

    # Collect unique pair_ids preserving order
    seen: set[str] = set()
    pair_ids: list[str] = []
    for r in results:
        if r["pair_id"] not in seen:
            seen.add(r["pair_id"])
            pair_ids.append(r["pair_id"])

    logger.info("=== NDS ABLATION SUMMARY (T=10, symmetric_norm) ===")

    for init_name in init_strategies:
        header = f"Init: {init_name}"
        logger.info(f"\n{header}")
        logger.info(f"{'Pair':<35} " + " ".join(f"{nl:>8}" for nl in main_nonlins))
        logger.info("-" * (35 + 9 * len(main_nonlins)))

        for pid in pair_ids:
            row_parts = []
            for nl in main_nonlins:
                match = [
                    r for r in results
                    if r["pair_id"] == pid
                    and r["nonlinearity"] == nl
                    and r["init"] == init_name
                    and r["T"] == 10
                    and r["diffusion_type"] == "symmetric_norm"
                ]
                if match:
                    sym = "YES" if match[0]["distinguished"] else "no"
                else:
                    sym = "?"
                row_parts.append(f"{sym:>8}")
            # Truncate pair_id to fit
            pid_short = pid[:34]
            logger.info(f"{pid_short:<35} " + " ".join(row_parts))

    # Single-step table (features at exactly step T, not trajectory)
    logger.info("\n=== SINGLE-STEP ANALYSIS (T=10, adj_row, sym_norm) ===")
    logger.info("(Features at ONLY step T, not accumulated trajectory)")
    logger.info(f"{'Pair':<35} " + " ".join(f"{nl:>8}" for nl in main_nonlins))
    logger.info("-" * (35 + 9 * len(main_nonlins)))
    for pid in pair_ids:
        row_parts = []
        for nl in main_nonlins:
            match = [
                r for r in results
                if r["pair_id"] == pid
                and r["nonlinearity"] == nl
                and r["init"] == "adjacency_row"
                and r["T"] == 10
                and r["diffusion_type"] == "symmetric_norm"
            ]
            if match:
                sym = "YES" if match[0]["step_distinguished"] else "no"
            else:
                sym = "?"
            row_parts.append(f"{sym:>8}")
        pid_short = pid[:34]
        logger.info(f"{pid_short:<35} " + " ".join(row_parts))

    # Minimum T table (trajectory-based)
    logger.info("\n=== Min T for trajectory distinction (relu, adj_row) ===")
    logger.info(f"{'Pair':<35} {'Min T':>8}")
    logger.info("-" * 45)
    for pid in pair_ids:
        matches = [
            r for r in results
            if r["pair_id"] == pid
            and r["nonlinearity"] == "relu"
            and r["init"] == "adjacency_row"
            and r["diffusion_type"] == "symmetric_norm"
            and r["distinguished"]
        ]
        if matches:
            min_t = min(r["T"] for r in matches)
            logger.info(f"{pid:<35} {min_t:>8}")
        else:
            logger.info(f"{pid:<35} {'never':>8}")

    # Minimum T table (single-step)
    logger.info("\n=== Min T for SINGLE-STEP distinction (relu, adj_row) ===")
    logger.info(f"{'Pair':<35} {'Min T':>8}")
    logger.info("-" * 45)
    for pid in pair_ids:
        matches = [
            r for r in results
            if r["pair_id"] == pid
            and r["nonlinearity"] == "relu"
            and r["init"] == "adjacency_row"
            and r["diffusion_type"] == "symmetric_norm"
            and r["step_distinguished"]
            and r["T"] > 0  # Skip T=0 which is init-dependent
        ]
        if matches:
            min_t = min(r["T"] for r in matches)
            logger.info(f"{pid:<35} {min_t:>8}")
        else:
            logger.info(f"{pid:<35} {'never':>8}")


# =========================================================================
# STEP 7: Output Formatting (exp_gen_sol_out schema)
# =========================================================================
def format_output(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    total_time: float,
) -> dict[str, Any]:
    """Format results into exp_gen_sol_out schema."""

    # Build lookups for linear baseline results (same init, same T, same pair)
    linear_traj_lookup: dict[str, bool] = {}
    linear_step_lookup: dict[str, bool] = {}
    for r in results:
        if r["nonlinearity"] == "linear":
            key = f"{r['pair_id']}|{r['init']}|{r['T']}|{r['diffusion_type']}"
            linear_traj_lookup[key] = r["distinguished"]
            linear_step_lookup[key] = r["step_distinguished"]

    examples = []
    for r in results:
        # Look up corresponding linear baseline
        linear_key = f"{r['pair_id']}|{r['init']}|{r['T']}|{r['diffusion_type']}"
        linear_traj_distinguished = linear_traj_lookup.get(linear_key, False)
        linear_step_distinguished = linear_step_lookup.get(linear_key, False)

        input_data = {
            "pair_id": r["pair_id"],
            "category": r["category"],
            "nonlinearity": r["nonlinearity"],
            "init_strategy": r["init"],
            "T": r["T"],
            "diffusion_type": r["diffusion_type"],
            "num_nodes": r["num_nodes"],
            "num_edges": r["num_edges"],
        }

        example = {
            "input": json.dumps(input_data),
            "output": "non-isomorphic",
            "predict_nds_trajectory": "distinguished" if r["distinguished"] else "indistinguishable",
            "predict_nds_single_step": "distinguished" if r["step_distinguished"] else "indistinguishable",
            "predict_linear_baseline": "distinguished" if linear_traj_distinguished else "indistinguishable",
            "predict_linear_single_step": "distinguished" if linear_step_distinguished else "indistinguishable",
            "metadata_pair_id": r["pair_id"],
            "metadata_category": r["category"],
            "metadata_nonlinearity": r["nonlinearity"],
            "metadata_init": r["init"],
            "metadata_T": r["T"],
            "metadata_diffusion_type": r["diffusion_type"],
            "metadata_num_nodes": r["num_nodes"],
            "metadata_num_edges": r["num_edges"],
            "metadata_frobenius_distance": round(r["frobenius_distance"], 10),
            "metadata_mmd_rbf": round(r["mmd_rbf"], 10),
            "metadata_distinguished_trajectory": r["distinguished"],
            "metadata_distinguished_single_step": r["step_distinguished"],
            "metadata_step_frobenius": round(r["step_frobenius"], 10),
            "metadata_hash_distinguished": r["hash_distinguished"],
        }
        examples.append(example)

    output = {
        "metadata": {
            "method_name": "Nonlinear Diffusion Signatures (NDS)",
            "description": (
                "Comprehensive ablation study of NDS features on 9 verified "
                "1-WL-equivalent graph pairs. Tests whether interleaved nonlinearity "
                "between diffusion steps breaks spectral invariance."
            ),
            "hypothesis": (
                "Interleaved nonlinearity between diffusion steps creates "
                "cross-frequency coupling that breaks spectral invariance, "
                "enabling distinction of 1-WL-equivalent graphs."
            ),
            "nonlinearities_tested": list(NONLINEARITIES.keys()) + list(EXTRA_NONLINEARITIES.keys()),
            "init_strategies_tested": ["degree", "adjacency_row", "random"],
            "diffusion_operators_tested": DIFFUSION_OPERATORS,
            "T_values_tested": T_VALUES,
            "total_configurations": len(results),
            "total_time_seconds": round(total_time, 2),
            "summary": summary,
        },
        "datasets": [
            {
                "dataset": "nds_ablation_results",
                "examples": examples,
            }
        ],
    }

    return output


# =========================================================================
# Main
# =========================================================================
@logger.catch
def main() -> None:
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("NDS Ablation Study — Nonlinear Diffusion Signatures")
    logger.info("=" * 70)

    # Load data
    pairs = load_graph_pairs(DATA_PATH)

    # Self-consistency check
    self_consistency_check(pairs)

    # Run full experiment
    results = run_experiment(pairs)

    total_time = time.time() - t_start

    # Compute summary
    summary = compute_summary(results)

    # Print human-readable summary
    print_summary_table(results)

    # Log key findings
    logger.info("\n=== KEY FINDINGS ===")
    logger.info(f"Best NDS config (trajectory): {summary['best_nds_config']}")
    logger.info(f"Pairs distinguished by best NDS: {summary['pairs_distinguished_by_best_nds_config']}/9")
    logger.info(f"Pairs distinguished by linear (degree): {summary['pairs_distinguished_by_linear_baseline_degree']}/9")
    logger.info(f"Pairs distinguished by linear (adj_row): {summary['pairs_distinguished_by_linear_baseline_adj_row']}/9")
    logger.info(f"Trajectory nonlin comparison (adj_row, T=10): {summary['nonlinearity_comparison_trajectory_T10_adj_row']}")
    logger.info(f"Single-step nonlin comparison (adj_row, T=10): {summary['nonlinearity_comparison_single_step_T10_adj_row']}")
    logger.info(f"Avg trajectory Frobenius by nonlin (T=10): {summary['avg_frobenius_trajectory_by_nonlin_T10_adj_row']}")
    logger.info(f"Avg single-step Frobenius by nonlin (T=10): {summary['avg_frobenius_single_step_by_nonlin_T10_adj_row']}")
    logger.info(f"Min T (relu/adj_row): {summary['min_T_relu_adj_row_per_pair']}")
    logger.info(f"Total time: {total_time:.2f}s")

    # Format output
    output = format_output(
        results=results,
        summary=summary,
        total_time=total_time,
    )

    # Save
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {len(output['datasets'][0]['examples'])} examples to {out_path}")
    logger.info(f"File size: {out_path.stat().st_size / 1024:.1f} KB")

    logger.info("DONE")


if __name__ == "__main__":
    main()
