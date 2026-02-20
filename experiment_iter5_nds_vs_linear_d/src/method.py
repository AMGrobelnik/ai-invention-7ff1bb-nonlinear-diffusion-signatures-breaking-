#!/usr/bin/env python3
"""NDS vs Linear Diffusion: Definitive Discriminative Power Test on 1-WL-Equivalent Graph Pairs.

Compares Nonlinear Diffusion Signatures (NDS) with tanh/sin(pi*x)/x*tanh(x) nonlinearities
against linear diffusion across 5 initialization types and T=1..20 steps on 98 verified
1-WL-equivalent graph pairs (89 non-VT + 9 VT).

Core question: does interleaved nonlinearity EVER help beyond what multi-dimensional
linear diffusion already provides?

Output: method_out.json conforming to exp_gen_sol_out schema.
"""

import json
import resource
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from loguru import logger
from scipy import sparse
from scipy.sparse.linalg import eigsh

# ============================================================
# CONFIG
# ============================================================
WORKSPACE = Path(__file__).resolve().parent

DEP1_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260219_082247"
    "/3_invention_loop/iter_4/gen_art/data_id1_it4__opus"
)
DEP2_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260219_082247"
    "/3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
)

T_VALUES = [1, 2, 5, 10, 15, 20]
NONLINEARITIES = ["tanh", "sin_pi", "x_tanh_x"]
INIT_TYPES = ["degree", "multi_scalar", "onehot", "random", "laplacian_pe"]
NUM_RANDOM_SEEDS = 5
RANDOM_DIM = 8
LAPLACIAN_K = 8
ONEHOT_MAX_N = 100  # Skip one-hot for graphs with n > this

# Resource limits: 14GB RAM, 3600s CPU
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# Suppress scipy sparse warnings
warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)

# ============================================================
# LOGGING
# ============================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(WORKSPACE / "logs" / "run.log", rotation="30 MB", level="DEBUG")


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class GraphPair:
    pair_id: str
    graph_A_edges: list
    graph_B_edges: list
    n_nodes_A: int
    n_nodes_B: int
    is_vertex_transitive: bool
    is_regular: bool
    family: str
    source_dataset: str  # "non_VT" or "VT"
    raw_metadata: dict = field(default_factory=dict)


# ============================================================
# DATA PARSING
# ============================================================
def parse_dep1_example(example: dict) -> GraphPair:
    """Parse a dependency 1 example (BREC_Basic or Custom_CFI)."""
    inp = json.loads(example["input"])
    g1 = inp["graph_1"]
    g2 = inp["graph_2"]

    is_vt = (
        example.get("metadata_graph_1_vertex_transitive", False)
        and example.get("metadata_graph_2_vertex_transitive", False)
    )
    is_reg = (
        example.get("metadata_graph_1_is_regular", False)
        and example.get("metadata_graph_2_is_regular", False)
    )

    return GraphPair(
        pair_id=example.get("metadata_pair_id", inp.get("pair_id", "unknown")),
        graph_A_edges=[tuple(e) for e in g1["edge_list"]],
        graph_B_edges=[tuple(e) for e in g2["edge_list"]],
        n_nodes_A=g1["num_nodes"],
        n_nodes_B=g2["num_nodes"],
        is_vertex_transitive=is_vt,
        is_regular=is_reg,
        family=example.get("metadata_category", "unknown"),
        source_dataset="non_VT",
        raw_metadata={k: v for k, v in example.items() if k.startswith("metadata_")},
    )


def parse_dep2_example(example: dict) -> GraphPair:
    """Parse a dependency 2 example (SRG / CSL)."""
    inp = json.loads(example["input"])
    ga = inp["graph_a"]
    gb = inp["graph_b"]

    category = example.get("metadata_category", inp.get("category", "unknown"))

    # All dep2 graphs are strongly regular or CSL => vertex-transitive & regular
    is_vt = True
    is_reg = True

    return GraphPair(
        pair_id=example.get("metadata_pair_id", inp.get("pair_id", "unknown")),
        graph_A_edges=[tuple(e) for e in ga["edge_list"]],
        graph_B_edges=[tuple(e) for e in gb["edge_list"]],
        n_nodes_A=ga["num_nodes"],
        n_nodes_B=gb["num_nodes"],
        is_vertex_transitive=is_vt,
        is_regular=is_reg,
        family=category,
        source_dataset="VT",
        raw_metadata={k: v for k, v in example.items() if k.startswith("metadata_")},
    )


def load_all_pairs(
    dep1_file: str = "full_data_out.json",
    dep2_file: str = "full_data_out.json",
    max_pairs: Optional[int] = None,
) -> list[GraphPair]:
    """Load graph pairs from both dependencies."""
    pairs = []

    # Load dep1
    dep1_path = DEP1_DIR / dep1_file
    logger.info(f"Loading dep1 from {dep1_path}")
    dep1_data = json.loads(dep1_path.read_text())
    for ds in dep1_data["datasets"]:
        for ex in ds["examples"]:
            try:
                pairs.append(parse_dep1_example(ex))
            except Exception:
                logger.exception(f"Failed to parse dep1 example: {ex.get('metadata_pair_id', 'unknown')}")
                continue

    # Load dep2
    dep2_path = DEP2_DIR / dep2_file
    logger.info(f"Loading dep2 from {dep2_path}")
    dep2_data = json.loads(dep2_path.read_text())
    for ds in dep2_data["datasets"]:
        for ex in ds["examples"]:
            try:
                pairs.append(parse_dep2_example(ex))
            except Exception:
                logger.exception(f"Failed to parse dep2 example: {ex.get('metadata_pair_id', 'unknown')}")
                continue

    logger.info(f"Loaded {len(pairs)} total pairs")

    if max_pairs is not None and max_pairs < len(pairs):
        pairs = pairs[:max_pairs]
        logger.info(f"Truncated to {max_pairs} pairs")

    return pairs


# ============================================================
# CORE ENGINE: Adjacency, Initializations, Propagation
# ============================================================
def build_adjacency(edges: list[tuple[int, int]], n: int) -> sparse.csr_matrix:
    """Build symmetric adjacency matrix from edge list."""
    if len(edges) == 0:
        return sparse.csr_matrix((n, n))
    row, col = zip(*edges)
    row_full = list(row) + list(col)
    col_full = list(col) + list(row)
    data = np.ones(len(row_full), dtype=np.float64)
    A = sparse.csr_matrix((data, (row_full, col_full)), shape=(n, n))
    # Remove duplicates
    A.data[:] = 1.0
    return A


def build_normalized_adjacency(edges: list[tuple[int, int]], n: int) -> sparse.csr_matrix:
    """Build symmetric normalized adjacency: D^{-1/2} A D^{-1/2}."""
    A = build_adjacency(edges, n)
    deg = np.array(A.sum(axis=1)).flatten()
    with np.errstate(divide="ignore", invalid="ignore"):
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sparse.diags(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def compute_initializations(
    edges: list[tuple[int, int]],
    n: int,
    A_norm: sparse.csr_matrix,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    """Compute all initialization types for a graph."""
    A = build_adjacency(edges, n)

    # (a) degree: shape (n, 1)
    degree = np.array(A.sum(axis=1)).flatten().reshape(-1, 1)

    # (b) multi_scalar: degree + clustering + betweenness (n, 3)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    clustering_dict = nx.clustering(G)
    clustering = np.array([clustering_dict.get(v, 0.0) for v in range(n)]).reshape(-1, 1)
    betweenness_dict = nx.betweenness_centrality(G)
    betweenness = np.array([betweenness_dict.get(v, 0.0) for v in range(n)]).reshape(-1, 1)
    multi_scalar = np.hstack([degree, clustering, betweenness])

    # (c) one-hot: shape (n, n) — only for small graphs
    if n <= ONEHOT_MAX_N:
        onehot = np.eye(n)
    else:
        onehot = None  # Will be skipped

    # (d) random: 5 seeds, shape (n, RANDOM_DIM)
    randoms = [np.random.RandomState(seed).randn(n, RANDOM_DIM) for seed in range(NUM_RANDOM_SEEDS)]

    # (e) Laplacian PE: first-k eigenvectors with sign disambiguation
    L = sparse.diags(np.array(A.sum(axis=1)).flatten()) - A
    try:
        k_eig = min(LAPLACIAN_K + 1, n - 1)
        if k_eig < 2:
            eigvecs = np.zeros((n, LAPLACIAN_K))
        else:
            eigenvalues, eigenvectors = eigsh(L.astype(np.float64), k=k_eig, which="SM")
            # Sign disambiguation: ensure max-abs component is positive
            for j in range(eigenvectors.shape[1]):
                idx = np.argmax(np.abs(eigenvectors[:, j]))
                if eigenvectors[idx, j] < 0:
                    eigenvectors[:, j] *= -1
            # Skip constant (zero eigenvalue) eigenvectors
            mask = eigenvalues > 1e-8
            eigvecs = eigenvectors[:, mask][:, :LAPLACIAN_K]
            # Pad if fewer than LAPLACIAN_K
            if eigvecs.shape[1] < LAPLACIAN_K:
                eigvecs = np.hstack([eigvecs, np.zeros((n, LAPLACIAN_K - eigvecs.shape[1]))])
    except Exception:
        logger.debug(f"eigsh failed for n={n}, falling back to dense eigh")
        try:
            L_dense = L.toarray().astype(np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            for j in range(eigenvectors.shape[1]):
                idx = np.argmax(np.abs(eigenvectors[:, j]))
                if eigenvectors[idx, j] < 0:
                    eigenvectors[:, j] *= -1
            mask = eigenvalues > 1e-8
            eigvecs = eigenvectors[:, mask][:, :LAPLACIAN_K]
            if eigvecs.shape[1] < LAPLACIAN_K:
                eigvecs = np.hstack([eigvecs, np.zeros((n, LAPLACIAN_K - eigvecs.shape[1]))])
        except Exception:
            logger.exception(f"Dense eigh also failed for n={n}")
            eigvecs = np.zeros((n, LAPLACIAN_K))

    return {
        "degree": degree,
        "multi_scalar": multi_scalar,
        "onehot": onehot,
        "random": randoms,
        "laplacian_pe": eigvecs,
    }


# ============================================================
# PROPAGATION
# ============================================================
NL_FUNCS = {
    "tanh": np.tanh,
    "sin_pi": lambda x: np.sin(np.pi * x),
    "x_tanh_x": lambda x: x * np.tanh(x),
}


def propagate_linear(A_norm: sparse.csr_matrix, x0: np.ndarray, T: int) -> np.ndarray:
    """Linear diffusion: concatenate [x0, Ax0, A^2 x0, ..., A^T x0].
    Returns shape (n, d*(T+1))."""
    trajectory = [x0]
    x = x0.copy()
    for _ in range(T):
        x = A_norm @ x
        trajectory.append(x.copy())
    return np.hstack(trajectory)


def propagate_nds(
    A_norm: sparse.csr_matrix,
    x0: np.ndarray,
    T: int,
    nonlinearity: str = "tanh",
) -> np.ndarray:
    """NDS: x(t+1) = sigma(A_norm * x(t)), concatenate trajectory.
    Returns shape (n, d*(T+1))."""
    sigma = NL_FUNCS[nonlinearity]
    trajectory = [x0]
    x = x0.copy()
    for _ in range(T):
        x = sigma(A_norm @ x)
        trajectory.append(x.copy())
    return np.hstack(trajectory)


# ============================================================
# DISTINGUISHABILITY MEASUREMENT
# ============================================================
def measure_distinguishability(feat_A: np.ndarray, feat_B: np.ndarray, n: int) -> dict:
    """Compare two feature matrices for distinguishability using multiset comparison.

    Since graphs are non-isomorphic but node orderings differ, we compare SORTED
    multisets of row feature vectors.
    """
    d = feat_A.shape[1]

    # Sort rows lexicographically for canonical ordering
    sorted_A = feat_A[np.lexsort(feat_A.T[::-1])]
    sorted_B = feat_B[np.lexsort(feat_B.T[::-1])]

    # Frobenius distance (normalized)
    frob_dist = float(np.linalg.norm(sorted_A - sorted_B) / max(np.sqrt(n * d), 1e-12))

    # Multiset comparison: sorted tuple lists (handles multiplicity correctly)
    ms_A = sorted([tuple(np.round(row, 8)) for row in feat_A])
    ms_B = sorted([tuple(np.round(row, 8)) for row in feat_B])
    multiset_exact_8 = ms_A != ms_B

    # Also at 6 decimals for robustness
    ms_A_6 = sorted([tuple(np.round(row, 6)) for row in feat_A])
    ms_B_6 = sorted([tuple(np.round(row, 6)) for row in feat_B])
    multiset_exact_6 = ms_A_6 != ms_B_6

    return {
        "frobenius_distance": frob_dist,
        "multiset_distinguishable_8dec": bool(multiset_exact_8),
        "multiset_distinguishable_6dec": bool(multiset_exact_6),
        "feature_dim": d,
    }


# ============================================================
# PAIR COMPARISON ENGINE
# ============================================================
def compare_pair(pair: GraphPair) -> dict:
    """For one graph pair, compute all init x propagation x T combinations."""
    assert pair.n_nodes_A == pair.n_nodes_B, (
        f"Pair {pair.pair_id}: n_A={pair.n_nodes_A} != n_B={pair.n_nodes_B}"
    )
    n = pair.n_nodes_A

    A_norm_A = build_normalized_adjacency(pair.graph_A_edges, n)
    A_norm_B = build_normalized_adjacency(pair.graph_B_edges, n)

    inits_A = compute_initializations(pair.graph_A_edges, n, A_norm_A)
    inits_B = compute_initializations(pair.graph_B_edges, n, A_norm_B)

    configs = {}

    # Deterministic initializations
    for init_name in ["degree", "multi_scalar", "onehot", "laplacian_pe"]:
        x0_A = inits_A[init_name]
        x0_B = inits_B[init_name]

        # Skip onehot if too large
        if x0_A is None or x0_B is None:
            continue

        for T in T_VALUES:
            # Linear diffusion
            feat_A_lin = propagate_linear(A_norm_A, x0_A, T)
            feat_B_lin = propagate_linear(A_norm_B, x0_B, T)
            lin_key = f"{init_name}_linear_T{T}"
            configs[lin_key] = measure_distinguishability(feat_A_lin, feat_B_lin, n)

            # NDS with each nonlinearity
            for nl in NONLINEARITIES:
                feat_A_nds = propagate_nds(A_norm_A, x0_A, T, nl)
                feat_B_nds = propagate_nds(A_norm_B, x0_B, T, nl)
                nds_key = f"{init_name}_{nl}_T{T}"
                configs[nds_key] = measure_distinguishability(feat_A_nds, feat_B_nds, n)

    # Random initializations: average over seeds
    for seed_idx in range(NUM_RANDOM_SEEDS):
        x0_A = inits_A["random"][seed_idx]
        x0_B = inits_B["random"][seed_idx]

        for T in T_VALUES:
            # Linear
            feat_A_lin = propagate_linear(A_norm_A, x0_A, T)
            feat_B_lin = propagate_linear(A_norm_B, x0_B, T)
            lin_key = f"random_s{seed_idx}_linear_T{T}"
            configs[lin_key] = measure_distinguishability(feat_A_lin, feat_B_lin, n)

            # NDS
            for nl in NONLINEARITIES:
                feat_A_nds = propagate_nds(A_norm_A, x0_A, T, nl)
                feat_B_nds = propagate_nds(A_norm_B, x0_B, T, nl)
                nds_key = f"random_s{seed_idx}_{nl}_T{T}"
                configs[nds_key] = measure_distinguishability(feat_A_nds, feat_B_nds, n)

    return {
        "pair_id": pair.pair_id,
        "family": pair.family,
        "is_vt": pair.is_vertex_transitive,
        "is_regular": pair.is_regular,
        "n_nodes": n,
        "source_dataset": pair.source_dataset,
        "configs": configs,
    }


# ============================================================
# SPECTRAL COUPLING ANALYSIS
# ============================================================
def spectral_coupling_analysis(
    pair: GraphPair,
    T: int = 10,
    nonlinearity: str = "tanh",
) -> dict:
    """Compute spectral coupling matrices for a representative pair.

    Tracks how energy transfers between eigenmodes due to the nonlinearity.
    Only for small graphs (n <= 200).
    """
    n = pair.n_nodes_A
    if n > 200:
        return {"skipped": True, "reason": "too large for full eigendecomposition"}

    A = build_adjacency(pair.graph_A_edges, n)
    L_dense = (sparse.diags(np.array(A.sum(axis=1)).flatten()) - A).toarray().astype(np.float64)

    try:
        eigenvalues, U = np.linalg.eigh(L_dense)
    except np.linalg.LinAlgError:
        return {"skipped": True, "reason": "eigendecomposition failed"}

    A_norm = build_normalized_adjacency(pair.graph_A_edges, n)

    # Use multi_scalar init (3D: degree, clustering, betweenness) for meaningful coupling
    inits = compute_initializations(pair.graph_A_edges, n, A_norm)
    x = inits["multi_scalar"].copy()  # shape (n, 3)

    nl_func = NL_FUNCS.get(nonlinearity, np.tanh)

    coupling_energy_offdiag = []

    for _ in range(T):
        # Spectral coefficients before nonlinearity
        c_before = U.T @ x  # shape (n, d)

        # Apply diffusion + nonlinearity
        x_diffused = A_norm @ x
        x_after_nl = nl_func(x_diffused)

        # Spectral coefficients after
        c_after = U.T @ x_after_nl

        # Coupling matrix: for each feature channel, compute how spectral modes mix
        # c_before and c_after are (n, d). For each channel j, we have n spectral
        # coefficients. Coupling = how the n spectral coefficients remap.
        # We compute per-channel coupling: c_after_col_j ~= M_j * c_before_col_j
        # where M_j is (n, n) spectral coupling matrix for channel j
        # For tractability, just measure the off-diagonal energy in the
        # spectral domain: compare c_before vs c_after per channel
        try:
            offdiag_energy = 0.0
            for ch in range(c_before.shape[1]):
                cb = c_before[:, ch]
                ca = c_after[:, ch]
                # If linear, ca = diag(f(lambda)) * cb => ca and cb are proportional per mode
                # Nonlinearity mixes modes => deviation from proportionality
                safe_cb = np.where(np.abs(cb) > 1e-12, cb, 1e-12)
                ratios = ca / safe_cb
                # Off-diagonal coupling ~ variance of ratios (if linear, all ratios equal)
                if np.any(np.abs(cb) > 1e-12):
                    valid = np.abs(cb) > 1e-12
                    offdiag_energy += float(np.var(ratios[valid]))
        except Exception:
            offdiag_energy = 0.0

        coupling_energy_offdiag.append(float(offdiag_energy))
        x = x_after_nl

    return {
        "coupling_energy_offdiag": coupling_energy_offdiag,
        "total_offdiag_energy": sum(coupling_energy_offdiag),
        "n_steps": T,
        "nonlinearity": nonlinearity,
        "n_nodes": n,
        "pair_id": pair.pair_id,
    }


# ============================================================
# AGGREGATE STATISTICS
# ============================================================
def compute_aggregate_statistics(all_pair_results: list[dict]) -> dict:
    """Compute the KEY analysis: for each init x T, count NDS-only vs linear-only distinctions."""
    stats = {}

    # For each init and T, compare linear vs each nonlinearity
    det_inits = ["degree", "multi_scalar", "onehot", "laplacian_pe"]

    for init_name in det_inits:
        for T in T_VALUES:
            lin_key = f"{init_name}_linear_T{T}"

            for nl in NONLINEARITIES:
                nds_key = f"{init_name}_{nl}_T{T}"

                nonlinear_only = 0
                linear_only = 0
                both = 0
                neither = 0
                deltas = []
                total_valid = 0

                for pr in all_pair_results:
                    lin_cfg = pr["configs"].get(lin_key)
                    nds_cfg = pr["configs"].get(nds_key)

                    if lin_cfg is None or nds_cfg is None:
                        continue

                    total_valid += 1
                    lin_d = lin_cfg["multiset_distinguishable_8dec"]
                    nds_d = nds_cfg["multiset_distinguishable_8dec"]

                    if nds_d and not lin_d:
                        nonlinear_only += 1
                    elif lin_d and not nds_d:
                        linear_only += 1
                    elif nds_d and lin_d:
                        both += 1
                    else:
                        neither += 1

                    deltas.append(nds_cfg["frobenius_distance"] - lin_cfg["frobenius_distance"])

                config_key = f"{init_name}_{nl}_vs_linear_T{T}"
                stats[config_key] = {
                    "nonlinear_only": nonlinear_only,
                    "linear_only": linear_only,
                    "both": both,
                    "neither": neither,
                    "total_valid": total_valid,
                    "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
                    "median_delta": float(np.median(deltas)) if deltas else 0.0,
                    "frac_positive_delta": float(np.mean([d > 0 for d in deltas])) if deltas else 0.0,
                }

    # Random init: aggregate over seeds
    for T in T_VALUES:
        for nl in NONLINEARITIES:
            nonlinear_only = 0
            linear_only = 0
            both = 0
            neither = 0
            deltas = []
            total_valid = 0

            for pr in all_pair_results:
                # Average distinguishability across seeds
                lin_dists = []
                nds_dists = []
                lin_frobs = []
                nds_frobs = []
                for s in range(NUM_RANDOM_SEEDS):
                    lin_cfg = pr["configs"].get(f"random_s{s}_linear_T{T}")
                    nds_cfg = pr["configs"].get(f"random_s{s}_{nl}_T{T}")
                    if lin_cfg and nds_cfg:
                        lin_dists.append(lin_cfg["multiset_distinguishable_8dec"])
                        nds_dists.append(nds_cfg["multiset_distinguishable_8dec"])
                        lin_frobs.append(lin_cfg["frobenius_distance"])
                        nds_frobs.append(nds_cfg["frobenius_distance"])

                if not lin_dists:
                    continue

                total_valid += 1
                # Majority vote across seeds
                lin_d = sum(lin_dists) > NUM_RANDOM_SEEDS / 2
                nds_d = sum(nds_dists) > NUM_RANDOM_SEEDS / 2

                if nds_d and not lin_d:
                    nonlinear_only += 1
                elif lin_d and not nds_d:
                    linear_only += 1
                elif nds_d and lin_d:
                    both += 1
                else:
                    neither += 1

                deltas.append(np.mean(nds_frobs) - np.mean(lin_frobs))

            config_key = f"random_mean_{nl}_vs_linear_T{T}"
            stats[config_key] = {
                "nonlinear_only": nonlinear_only,
                "linear_only": linear_only,
                "both": both,
                "neither": neither,
                "total_valid": total_valid,
                "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
                "median_delta": float(np.median(deltas)) if deltas else 0.0,
                "frac_positive_delta": float(np.mean([d > 0 for d in deltas])) if deltas else 0.0,
            }

    # Stratified analysis
    strata = {
        "VT": lambda p: p["is_vt"],
        "non_VT": lambda p: not p["is_vt"],
        "regular": lambda p: p["is_regular"],
        "non_regular": lambda p: not p["is_regular"],
    }

    stratified = {}
    for stratum_name, filter_fn in strata.items():
        subset = [p for p in all_pair_results if filter_fn(p)]
        stratum_stats = {}

        for init_name in det_inits:
            for T in T_VALUES:
                lin_key = f"{init_name}_linear_T{T}"
                for nl in NONLINEARITIES:
                    nds_key = f"{init_name}_{nl}_T{T}"

                    nl_only = sum(
                        1 for pr in subset
                        if pr["configs"].get(nds_key, {}).get("multiset_distinguishable_8dec", False)
                        and not pr["configs"].get(lin_key, {}).get("multiset_distinguishable_8dec", False)
                    )
                    lin_only_ct = sum(
                        1 for pr in subset
                        if pr["configs"].get(lin_key, {}).get("multiset_distinguishable_8dec", False)
                        and not pr["configs"].get(nds_key, {}).get("multiset_distinguishable_8dec", False)
                    )
                    both_ct = sum(
                        1 for pr in subset
                        if pr["configs"].get(lin_key, {}).get("multiset_distinguishable_8dec", False)
                        and pr["configs"].get(nds_key, {}).get("multiset_distinguishable_8dec", False)
                    )
                    neither_ct = len(subset) - nl_only - lin_only_ct - both_ct

                    sk = f"{init_name}_{nl}_vs_linear_T{T}"
                    stratum_stats[sk] = {
                        "nonlinear_only": nl_only,
                        "linear_only": lin_only_ct,
                        "both": both_ct,
                        "neither": neither_ct,
                        "total": len(subset),
                    }

        stratified[stratum_name] = stratum_stats

    return {"overall": stats, "stratified": stratified}


# ============================================================
# OUTPUT FORMATTING
# ============================================================
def select_key_configs() -> list[str]:
    """Select ~12 key configurations for predict_ fields."""
    key_configs = []
    for init in ["degree", "multi_scalar", "laplacian_pe"]:
        for T in [10, 20]:
            key_configs.append(f"{init}_linear_T{T}")
            key_configs.append(f"{init}_tanh_T{T}")
    return key_configs


def format_example(pair_result: dict, pair: GraphPair) -> dict:
    """Format a single pair result as an exp_gen_sol_out example."""
    ex = {
        "input": json.dumps({
            "pair_id": pair_result["pair_id"],
            "family": pair_result["family"],
            "n_nodes": pair_result["n_nodes"],
            "is_vt": pair_result["is_vt"],
            "is_regular": pair_result["is_regular"],
            "source_dataset": pair_result["source_dataset"],
        }),
        "output": json.dumps({
            "ground_truth": "non-isomorphic, 1-WL equivalent",
            "pair_id": pair_result["pair_id"],
        }),
        "metadata_pair_id": pair_result["pair_id"],
        "metadata_family": pair_result["family"],
        "metadata_is_vt": pair_result["is_vt"],
        "metadata_is_regular": pair_result["is_regular"],
        "metadata_n_nodes": pair_result["n_nodes"],
        "metadata_source_dataset": pair_result["source_dataset"],
    }

    # Add predict_ fields for key configurations
    key_configs = select_key_configs()
    for kcfg in key_configs:
        cfg_data = pair_result["configs"].get(kcfg)
        if cfg_data is not None:
            label = "distinguished" if cfg_data["multiset_distinguishable_8dec"] else "not_distinguished"
            ex[f"predict_{kcfg}"] = label

            # Also add delta for NDS vs linear pairs
            if "tanh" in kcfg or "sin_pi" in kcfg or "x_tanh_x" in kcfg:
                # Find corresponding linear key
                parts = kcfg.rsplit("_", 1)  # e.g. "degree_tanh" + "T10"
                init_parts = kcfg.split("_")
                # Reconstruct linear key
                if init_parts[0] in ["degree", "multi", "onehot", "laplacian"]:
                    if init_parts[0] == "multi":
                        init_n = "multi_scalar"
                        nl_idx = 2
                    elif init_parts[0] == "laplacian":
                        init_n = "laplacian_pe"
                        nl_idx = 2
                    else:
                        init_n = init_parts[0]
                        nl_idx = 1
                    t_part = init_parts[-1]  # e.g. "T10"
                    lin_k = f"{init_n}_linear_{t_part}"
                    lin_data = pair_result["configs"].get(lin_k)
                    if lin_data is not None:
                        delta = cfg_data["frobenius_distance"] - lin_data["frobenius_distance"]
                        ex[f"predict_delta_{kcfg}"] = f"{delta:.6f}"

    # Store full configs as metadata (JSON string)
    # Only store summary to keep size manageable
    configs_summary = {}
    for k, v in pair_result["configs"].items():
        configs_summary[k] = {
            "frob": round(v["frobenius_distance"], 8),
            "dist_8": v["multiset_distinguishable_8dec"],
            "dist_6": v["multiset_distinguishable_6dec"],
        }
    ex["metadata_configs"] = json.dumps(configs_summary)

    return ex


def build_output(
    all_pair_results: list[dict],
    pairs: list[GraphPair],
    aggregate_stats: dict,
    spectral_results: list[dict],
) -> dict:
    """Build the final method_out.json conforming to exp_gen_sol_out schema."""
    # Split into VT and non-VT datasets
    non_vt_examples = []
    vt_examples = []

    pair_map = {p.pair_id: p for p in pairs}

    for pr in all_pair_results:
        pair = pair_map[pr["pair_id"]]
        ex = format_example(pr, pair)
        if pr["source_dataset"] == "VT":
            vt_examples.append(ex)
        else:
            non_vt_examples.append(ex)

    datasets = []
    if non_vt_examples:
        datasets.append({
            "dataset": "1WL_equiv_pairs_non_VT",
            "examples": non_vt_examples,
        })
    if vt_examples:
        datasets.append({
            "dataset": "1WL_equiv_pairs_VT",
            "examples": vt_examples,
        })

    output = {
        "metadata": {
            "method_name": "NDS_vs_Linear_Diffusion",
            "description": "Definitive test of nonlinear vs linear diffusion on 98 1-WL-equiv pairs",
            "T_values": T_VALUES,
            "nonlinearities": NONLINEARITIES,
            "initializations": INIT_TYPES,
            "num_random_seeds": NUM_RANDOM_SEEDS,
            "onehot_max_n": ONEHOT_MAX_N,
            "total_pairs": len(all_pair_results),
            "aggregate_statistics": aggregate_stats,
            "spectral_coupling_analysis": spectral_results,
        },
        "datasets": datasets,
    }

    return output


# ============================================================
# MAIN
# ============================================================
@logger.catch
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("NDS vs Linear Diffusion — Experiment Start")
    logger.info("=" * 60)

    # Load all pairs
    pairs = load_all_pairs()
    logger.info(f"Total pairs loaded: {len(pairs)}")

    non_vt = [p for p in pairs if not p.is_vertex_transitive]
    vt = [p for p in pairs if p.is_vertex_transitive]
    regular = [p for p in pairs if p.is_regular]
    non_regular = [p for p in pairs if not p.is_regular]
    logger.info(f"  Non-VT: {len(non_vt)}, VT: {len(vt)}")
    logger.info(f"  Regular: {len(regular)}, Non-regular: {len(non_regular)}")

    # Process all pairs
    all_pair_results = []
    for i, pair in enumerate(pairs):
        t0 = time.time()
        logger.info(f"Processing pair {i+1}/{len(pairs)}: {pair.pair_id} (n={pair.n_nodes_A}, family={pair.family})")
        try:
            result = compare_pair(pair)
            all_pair_results.append(result)

            # Quick summary
            n_dist_lin = sum(
                1 for k, v in result["configs"].items()
                if "linear" in k and v["multiset_distinguishable_8dec"]
            )
            n_dist_nds = sum(
                1 for k, v in result["configs"].items()
                if "linear" not in k and v["multiset_distinguishable_8dec"]
            )
            elapsed = time.time() - t0
            logger.info(f"  Done in {elapsed:.1f}s | linear_dist={n_dist_lin}, nds_dist={n_dist_nds}, total_configs={len(result['configs'])}")

        except Exception:
            logger.exception(f"Failed on pair {pair.pair_id}")
            continue

    logger.info(f"Processed {len(all_pair_results)}/{len(pairs)} pairs successfully")

    # Aggregate statistics
    logger.info("Computing aggregate statistics...")
    aggregate_stats = compute_aggregate_statistics(all_pair_results)

    # Log key findings
    logger.info("=== KEY FINDINGS ===")
    for cfg_key, cfg_val in aggregate_stats["overall"].items():
        if "tanh_vs_linear_T10" in cfg_key and cfg_key.startswith("degree"):
            logger.info(f"  {cfg_key}: NL_only={cfg_val['nonlinear_only']}, "
                       f"Lin_only={cfg_val['linear_only']}, "
                       f"both={cfg_val['both']}, "
                       f"neither={cfg_val['neither']}")

    # Spectral coupling analysis on 10 representative small pairs
    logger.info("Running spectral coupling analysis on representative pairs...")
    spectral_pairs = []
    # Pick pairs: prefer small graphs, mix of families
    vt_small = sorted([p for p in pairs if p.is_vertex_transitive and p.n_nodes_A <= 200],
                       key=lambda p: p.n_nodes_A)[:3]
    brec_small = sorted([p for p in pairs if p.family == "Basic" and p.n_nodes_A <= 50],
                         key=lambda p: p.n_nodes_A)[:4]
    cfi_small = sorted([p for p in pairs if p.family == "Custom_CFI" and p.n_nodes_A <= 100],
                        key=lambda p: p.n_nodes_A)[:3]
    spectral_pairs = vt_small + brec_small + cfi_small
    spectral_pairs = spectral_pairs[:10]

    spectral_results = []
    for sp in spectral_pairs:
        logger.info(f"  Spectral analysis on {sp.pair_id} (n={sp.n_nodes_A})")
        try:
            sr = spectral_coupling_analysis(sp, T=10, nonlinearity="tanh")
            spectral_results.append(sr)
        except Exception:
            logger.exception(f"  Spectral analysis failed for {sp.pair_id}")
            spectral_results.append({"skipped": True, "reason": "exception", "pair_id": sp.pair_id})

    # Build output
    logger.info("Building output JSON...")
    output = build_output(all_pair_results, pairs, aggregate_stats, spectral_results)

    # Write method_out.json
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
