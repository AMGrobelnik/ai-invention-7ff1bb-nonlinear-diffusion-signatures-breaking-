#!/usr/bin/env python3
"""NDS Ablation: Scalar Initializations x Signed Nonlinearities on 1-WL-Equivalent Graph Pairs.

Full ablation experiment testing 5 scalar initializations x 4 nonlinearities x T=1..20
on 9 provably 1-WL-equivalent graph pairs.

Key test: does tanh/leaky_relu NDS distinguish pairs that linear-diffusion-only
with the SAME init cannot?

Additional analysis:
- Multi-dimensional vector initialization fallback for vertex-transitive graphs
- Spectral analysis (eigenmode energy distribution)
- Heat kernel diagonal and subgraph centrality initializations
"""

import json
import resource
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from loguru import logger
from scipy.spatial.distance import cdist

# ============================================================
# Setup: Logging, resource limits, paths
# ============================================================

WORKSPACE = Path(__file__).resolve().parent

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:HH:mm:ss}|{level:<7}|{message}",
)
logger.add(
    str(WORKSPACE / "logs" / "run.log"),
    rotation="30 MB",
    level="DEBUG",
)

# Resource limits: 14GB RAM, 3600s CPU
try:
    resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
except ValueError:
    logger.warning("Could not set resource limits (may require root)")

T_MAX = 20
DIST_THRESHOLD_ABS = 1e-8  # Absolute threshold for Frobenius distance
DIST_THRESHOLD_REL = 1e-10  # Relative threshold (distance / feature norm)


# ============================================================
# STEP 1: Load graph pairs from dependency data
# ============================================================

def load_graph_pairs(data_path: Path) -> list[dict]:
    """Load graph pairs from the full_data_out.json dependency file.

    Returns list of dicts with keys:
        pair_id, category, num_nodes, graph_a_nx, graph_b_nx,
        graph_a_adj, graph_b_adj, graph_a_name, graph_b_name
    """
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    examples = raw["datasets"][0]["examples"]
    logger.info(f"Found {len(examples)} graph pairs")

    pairs = []
    for ex in examples:
        inp = json.loads(ex["input"])
        pair_id = inp["pair_id"]
        category = inp["category"]

        ga_data = inp["graph_a"]
        gb_data = inp["graph_b"]

        adj_a = np.array(ga_data["adjacency_matrix"], dtype=np.float64)
        adj_b = np.array(gb_data["adjacency_matrix"], dtype=np.float64)

        G_a = nx.from_numpy_array(adj_a)
        G_b = nx.from_numpy_array(adj_b)

        pairs.append({
            "pair_id": pair_id,
            "category": category,
            "num_nodes": ga_data["num_nodes"],
            "graph_a_nx": G_a,
            "graph_b_nx": G_b,
            "graph_a_adj": adj_a,
            "graph_b_adj": adj_b,
            "graph_a_name": ga_data["name"],
            "graph_b_name": gb_data["name"],
        })
        logger.debug(
            f"  Loaded pair {pair_id}: {ga_data['name']} vs {gb_data['name']} "
            f"(n={ga_data['num_nodes']}, category={category})"
        )

    categories = sorted(set(p["category"] for p in pairs))
    logger.info(f"Loaded {len(pairs)} graph pairs across categories: {categories}")
    return pairs


# ============================================================
# STEP 2: Initialization functions (5 scalar + 3 fallback)
# ============================================================

def init_degree(G: nx.Graph) -> np.ndarray:
    """Degree of each node."""
    return np.array([G.degree(v) for v in range(G.number_of_nodes())], dtype=np.float64)


def init_clustering_coefficient(G: nx.Graph) -> np.ndarray:
    """Local clustering coefficient of each node."""
    cc = nx.clustering(G)
    return np.array([cc[v] for v in range(G.number_of_nodes())], dtype=np.float64)


def init_pagerank(G: nx.Graph) -> np.ndarray:
    """PageRank centrality (alpha=0.85)."""
    pr = nx.pagerank(G, alpha=0.85)
    return np.array([pr[v] for v in range(G.number_of_nodes())], dtype=np.float64)


def init_betweenness_centrality(G: nx.Graph) -> np.ndarray:
    """Betweenness centrality."""
    bc = nx.betweenness_centrality(G)
    return np.array([bc[v] for v in range(G.number_of_nodes())], dtype=np.float64)


def init_eigenvector_centrality(G: nx.Graph) -> np.ndarray:
    """Eigenvector centrality with numpy fallback."""
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector centrality did not converge, using numpy fallback")
        A = nx.to_numpy_array(G)
        vals, vecs = np.linalg.eigh(A)
        ec_vec = np.abs(vecs[:, -1])
        ec = {v: float(ec_vec[v]) for v in range(G.number_of_nodes())}
    return np.array([ec[v] for v in range(G.number_of_nodes())], dtype=np.float64)


# Additional fallback inits for vertex-transitive graphs

def init_subgraph_centrality(G: nx.Graph) -> np.ndarray:
    """Subgraph centrality: diagonal of e^A. May be constant on vertex-transitive graphs."""
    A = nx.to_numpy_array(G, dtype=np.float64)
    expA = _matrix_exp(A)
    return np.diag(expA).copy()


def init_heat_kernel_diag(G: nx.Graph, t: float = 1.0) -> np.ndarray:
    """Heat kernel diagonal: diag(e^{-tL}). May be constant on vertex-transitive graphs."""
    L = nx.laplacian_matrix(G).toarray().astype(np.float64)
    expL = _matrix_exp(-t * L)
    return np.diag(expL).copy()


def init_multidim_powers(G: nx.Graph) -> np.ndarray:
    """Multi-dimensional init: [A^2_{vv}, A^3_{vv}, A^4_{vv}, A^5_{vv}] per node.

    Returns the L2 norm of this 4-vector as a scalar per node.
    This is a richer init that may break vertex-transitivity via higher powers.
    """
    A = nx.to_numpy_array(G, dtype=np.float64)
    A2 = A @ A
    A3 = A2 @ A
    A4 = A3 @ A
    A5 = A4 @ A
    features = np.stack([np.diag(A2), np.diag(A3), np.diag(A4), np.diag(A5)], axis=1)
    return np.linalg.norm(features, axis=1)


def _matrix_exp(M: np.ndarray) -> np.ndarray:
    """Compute matrix exponential via eigendecomposition (for symmetric matrices)."""
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    return eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T


INIT_FUNCTIONS = {
    "degree": init_degree,
    "clustering_coeff": init_clustering_coefficient,
    "pagerank": init_pagerank,
    "betweenness": init_betweenness_centrality,
    "eigvec_centrality": init_eigenvector_centrality,
    "subgraph_centrality": init_subgraph_centrality,
    "heat_kernel_diag": init_heat_kernel_diag,
    "multidim_powers": init_multidim_powers,
}

# Primary inits (from plan)
PRIMARY_INITS = ["degree", "clustering_coeff", "pagerank", "betweenness", "eigvec_centrality"]
# Fallback inits (for vertex-transitive graph analysis)
FALLBACK_INITS = ["subgraph_centrality", "heat_kernel_diag", "multidim_powers"]


# ============================================================
# STEP 3: Nonlinearity functions (4 nonlinearities)
# ============================================================

def nl_linear(x: np.ndarray) -> np.ndarray:
    """Identity (baseline - no nonlinearity)."""
    return x


def nl_tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent."""
    return np.tanh(x)


def nl_leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Leaky ReLU with slope alpha for negative values."""
    return np.where(x >= 0, x, alpha * x)


def nl_abs(x: np.ndarray) -> np.ndarray:
    """Absolute value."""
    return np.abs(x)


NONLINEARITIES = {
    "linear": nl_linear,
    "tanh": nl_tanh,
    "leaky_relu": nl_leaky_relu,
    "abs": nl_abs,
}


# ============================================================
# STEP 4: NDS feature computation
# ============================================================

def build_normalized_adjacency(G: nx.Graph) -> np.ndarray:
    """Build D^{-1/2} A D^{-1/2} normalized adjacency matrix."""
    A = nx.to_numpy_array(G, dtype=np.float64)
    degrees = A.sum(axis=1)
    D_inv_sqrt = np.diag(np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0))
    return D_inv_sqrt @ A @ D_inv_sqrt


def compute_nds_features(
    A_norm: np.ndarray,
    x0: np.ndarray,
    nonlinearity_fn,
    T_max: int = T_MAX,
) -> np.ndarray:
    """Compute Nonlinear Diffusion Signatures for a graph.

    Args:
        A_norm: Normalized adjacency matrix (precomputed)
        x0: Initial scalar features per node
        nonlinearity_fn: Pointwise nonlinearity to apply after each diffusion step
        T_max: Maximum number of diffusion steps

    Returns:
        trajectory: np.ndarray of shape (n, T_max+1)
    """
    n = A_norm.shape[0]
    trajectory = np.zeros((n, T_max + 1), dtype=np.float64)
    x = x0.copy()
    trajectory[:, 0] = x

    for t in range(1, T_max + 1):
        x = A_norm @ x
        x = nonlinearity_fn(x)
        trajectory[:, t] = x

    return trajectory


# ============================================================
# STEP 5: Distinguishability metrics
# ============================================================

def compute_multiset_distance(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    T: int,
) -> tuple[bool, float, float]:
    """Check if sorted NDS feature vectors at step T are different.

    Args:
        traj_a: Trajectory for graph A, shape (n_a, T_max+1)
        traj_b: Trajectory for graph B, shape (n_b, T_max+1)
        T: Diffusion step to compare at

    Returns:
        (is_distinguished, frobenius_distance, mmd)
    """
    feat_a = traj_a[:, :T + 1]
    feat_b = traj_b[:, :T + 1]

    # Sort rows lexicographically for multiset comparison
    sorted_a = feat_a[np.lexsort(feat_a.T[::-1])]
    sorted_b = feat_b[np.lexsort(feat_b.T[::-1])]

    frobenius = float(np.linalg.norm(sorted_a - sorted_b, "fro"))

    # Use both absolute and relative thresholds to handle numerical noise
    # For large-valued features (e.g. e^A diagonal ~ 6500), numerical noise
    # can produce Frobenius ~ 1e-9 which is NOT a real distinction
    feature_norm = max(
        float(np.linalg.norm(sorted_a, "fro")),
        float(np.linalg.norm(sorted_b, "fro")),
        1e-15,  # avoid division by zero
    )
    relative_dist = frobenius / feature_norm
    is_distinguished = (frobenius > DIST_THRESHOLD_ABS) and (relative_dist > DIST_THRESHOLD_REL)

    # MMD with RBF kernel
    mmd = _compute_mmd(feat_a, feat_b)

    # Flag numerically marginal cases (absolute > 1e-12 but relative < 1e-8)
    numerically_marginal = (frobenius > 1e-12) and (relative_dist < 1e-6)

    return is_distinguished, frobenius, mmd, numerically_marginal


def _compute_mmd(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
    """Maximum Mean Discrepancy with RBF kernel."""
    D_aa = cdist(feat_a, feat_a, "euclidean")
    D_bb = cdist(feat_b, feat_b, "euclidean")
    D_ab = cdist(feat_a, feat_b, "euclidean")

    all_dists = np.concatenate([D_aa.ravel(), D_bb.ravel(), D_ab.ravel()])
    sigma = float(np.median(all_dists))
    if sigma < 1e-12:
        sigma = 1.0

    K_aa = np.exp(-D_aa**2 / (2 * sigma**2))
    K_bb = np.exp(-D_bb**2 / (2 * sigma**2))
    K_ab = np.exp(-D_ab**2 / (2 * sigma**2))

    mmd_sq = K_aa.mean() + K_bb.mean() - 2 * K_ab.mean()
    return float(np.sqrt(max(0.0, mmd_sq)))


# ============================================================
# STEP 6: Run full ablation grid
# ============================================================

def run_ablation_grid(
    pairs: list[dict],
    init_names: list[str],
    T_max: int = T_MAX,
) -> list[dict]:
    """Run full ablation grid: inits x nonlinearities x T x pairs.

    Returns list of result dicts.
    """
    total_configs = len(init_names) * len(NONLINEARITIES) * T_max * len(pairs)
    logger.info(
        f"Running ablation grid: {len(init_names)} inits x "
        f"{len(NONLINEARITIES)} nonlinearities x T=1..{T_max} x "
        f"{len(pairs)} pairs = {total_configs} evaluations"
    )

    results = []
    config_count = 0

    for pair_idx, pair in enumerate(pairs):
        G_a = pair["graph_a_nx"]
        G_b = pair["graph_b_nx"]
        pair_id = pair["pair_id"]
        category = pair["category"]

        # Precompute normalized adjacencies
        A_norm_a = build_normalized_adjacency(G_a)
        A_norm_b = build_normalized_adjacency(G_b)

        for init_name in init_names:
            init_fn = INIT_FUNCTIONS[init_name]

            # Check: are init features identical for this pair?
            feat_a_init = init_fn(G_a)
            feat_b_init = init_fn(G_b)
            init_identical = bool(
                np.allclose(np.sort(feat_a_init), np.sort(feat_b_init), atol=1e-10)
            )

            # Check if init is constant per graph (vertex-transitive indicator)
            a_const = bool(np.allclose(feat_a_init, feat_a_init[0], atol=1e-10))
            b_const = bool(np.allclose(feat_b_init, feat_b_init[0], atol=1e-10))
            init_is_constant = a_const and b_const

            if init_is_constant:
                logger.debug(
                    f"  Init '{init_name}' is CONSTANT on pair {pair_id} "
                    f"(vertex-transitive behavior)"
                )

            for nl_name, nl_fn in NONLINEARITIES.items():
                t_start = time.time()

                traj_a = compute_nds_features(
                    A_norm=A_norm_a,
                    x0=feat_a_init,
                    nonlinearity_fn=nl_fn,
                    T_max=T_max,
                )
                traj_b = compute_nds_features(
                    A_norm=A_norm_b,
                    x0=feat_b_init,
                    nonlinearity_fn=nl_fn,
                    T_max=T_max,
                )

                wall_clock = time.time() - t_start

                for T in range(1, T_max + 1):
                    is_dist, frob, mmd, marginal = compute_multiset_distance(
                        traj_a, traj_b, T
                    )

                    results.append({
                        "pair_id": pair_id,
                        "category": category,
                        "num_nodes": pair["num_nodes"],
                        "init": init_name,
                        "nonlinearity": nl_name,
                        "T": T,
                        "distinguished": is_dist,
                        "frobenius_distance": frob,
                        "mmd": mmd,
                        "numerically_marginal": marginal,
                        "init_features_identical": init_identical,
                        "init_is_constant": init_is_constant,
                        "wall_clock_s": wall_clock,
                    })

                config_count += T_max

        logger.info(
            f"  Pair {pair_idx + 1}/{len(pairs)} done: {pair_id} "
            f"({config_count}/{total_configs} configs)"
        )

    logger.info(f"Ablation grid complete: {len(results)} results")
    return results


# ============================================================
# STEP 7: Compute Nonlinear vs Linear Gap analysis
# ============================================================

def compute_gap_analysis(
    results: list[dict],
    pairs: list[dict],
    init_names: list[str],
) -> dict:
    """Compute gap between nonlinear and linear distinguishability.

    For each (init, nonlinearity != linear), count how many pairs are
    distinguished by the nonlinear method but NOT by linear (positive gap).
    """
    gap_analysis = {}

    for init_name in init_names:
        for nl_name in [n for n in NONLINEARITIES if n != "linear"]:
            key = f"{init_name}__{nl_name}"
            gap_entry = {
                "pairs_distinguished_nonlinear": 0,
                "pairs_distinguished_linear": 0,
                "positive_gap_pairs": [],
                "negative_gap_pairs": [],
                "same_pairs": [],
            }

            for pair in pairs:
                pid = pair["pair_id"]

                # Best-T distinguishability for nonlinear
                nl_results = [
                    r for r in results
                    if r["init"] == init_name
                    and r["nonlinearity"] == nl_name
                    and r["pair_id"] == pid
                ]
                any_nl_dist = any(r["distinguished"] for r in nl_results)

                # Best-T distinguishability for linear
                lin_results = [
                    r for r in results
                    if r["init"] == init_name
                    and r["nonlinearity"] == "linear"
                    and r["pair_id"] == pid
                ]
                any_lin_dist = any(r["distinguished"] for r in lin_results)

                if any_nl_dist:
                    gap_entry["pairs_distinguished_nonlinear"] += 1
                if any_lin_dist:
                    gap_entry["pairs_distinguished_linear"] += 1
                if any_nl_dist and not any_lin_dist:
                    gap_entry["positive_gap_pairs"].append(pid)
                elif any_lin_dist and not any_nl_dist:
                    gap_entry["negative_gap_pairs"].append(pid)
                else:
                    gap_entry["same_pairs"].append(pid)

            gap_analysis[key] = gap_entry

    return gap_analysis


# ============================================================
# STEP 8: Spectral analysis
# ============================================================

def run_spectral_analysis(
    pairs: list[dict],
    init_names: list[str],
    T_max: int = T_MAX,
) -> dict:
    """Project NDS features onto Laplacian eigenvectors.

    For each pair, compute spectral energy distribution for each
    init x nonlinearity combination.
    """
    logger.info("Running spectral analysis...")
    spectral_results = {}

    for pair in pairs:
        G_a = pair["graph_a_nx"]
        pid = pair["pair_id"]

        L_a = nx.laplacian_matrix(G_a).toarray().astype(np.float64)
        eigenvalues_a, eigenvectors_a = np.linalg.eigh(L_a)

        A_norm_a = build_normalized_adjacency(G_a)

        for init_name in init_names:
            init_fn = INIT_FUNCTIONS[init_name]
            x0_a = init_fn(G_a)

            for nl_name, nl_fn in NONLINEARITIES.items():
                traj_a = compute_nds_features(
                    A_norm=A_norm_a,
                    x0=x0_a,
                    nonlinearity_fn=nl_fn,
                    T_max=T_max,
                )

                # Project each time step onto eigenvectors
                coefficients = eigenvectors_a.T @ traj_a  # shape (n, T+1)
                spectral_energy = coefficients**2

                # Summarize: total energy per eigenmode
                total_energy_per_mode = spectral_energy.sum(axis=1).tolist()
                # Energy at each time step per mode (truncated for JSON size)
                energy_at_T20 = spectral_energy[:, -1].tolist()

                spectral_results[f"{pid}__{init_name}__{nl_name}"] = {
                    "eigenvalues": eigenvalues_a.tolist(),
                    "total_energy_per_mode": total_energy_per_mode,
                    "energy_at_T20": energy_at_T20,
                    "num_nonzero_modes_T20": int(np.sum(np.abs(spectral_energy[:, -1]) > 1e-12)),
                }

    logger.info(f"Spectral analysis complete: {len(spectral_results)} entries")
    return spectral_results


# ============================================================
# STEP 9: Aggregate results and build summary tables
# ============================================================

def build_per_init_summary(
    results: list[dict],
    pairs: list[dict],
    init_names: list[str],
) -> dict:
    """Per-init summary: how many pairs distinguished with each nonlinearity."""
    summary = {}
    for init_name in init_names:
        init_summary = {}
        for nl_name in NONLINEARITIES:
            distinguished_pairs = []
            for pair in pairs:
                pid = pair["pair_id"]
                pair_results = [
                    r for r in results
                    if r["init"] == init_name
                    and r["nonlinearity"] == nl_name
                    and r["pair_id"] == pid
                ]
                if any(r["distinguished"] for r in pair_results):
                    # Find first T that distinguishes
                    first_T = min(
                        r["T"] for r in pair_results if r["distinguished"]
                    )
                    best_frob = max(r["frobenius_distance"] for r in pair_results)
                    distinguished_pairs.append({
                        "pair_id": pid,
                        "first_T": first_T,
                        "best_frobenius": best_frob,
                    })

            init_summary[nl_name] = {
                "num_distinguished": len(distinguished_pairs),
                "distinguished_pairs": distinguished_pairs,
            }
        summary[init_name] = init_summary
    return summary


def build_per_pair_summary(
    results: list[dict],
    pairs: list[dict],
    init_names: list[str],
) -> dict:
    """Per-pair summary: which init x nonlinearity combinations distinguish it."""
    summary = {}
    for pair in pairs:
        pid = pair["pair_id"]
        pair_summary = {
            "category": pair["category"],
            "num_nodes": pair["num_nodes"],
            "distinguished_by": [],
            "not_distinguished_by": [],
        }
        for init_name in init_names:
            for nl_name in NONLINEARITIES:
                pair_results = [
                    r for r in results
                    if r["init"] == init_name
                    and r["nonlinearity"] == nl_name
                    and r["pair_id"] == pid
                ]
                any_dist = any(r["distinguished"] for r in pair_results)
                config = f"{init_name}__{nl_name}"
                if any_dist:
                    first_T = min(r["T"] for r in pair_results if r["distinguished"])
                    best_frob = max(r["frobenius_distance"] for r in pair_results)
                    pair_summary["distinguished_by"].append({
                        "config": config,
                        "first_T": first_T,
                        "best_frobenius": best_frob,
                    })
                else:
                    pair_summary["not_distinguished_by"].append(config)
        summary[pid] = pair_summary
    return summary


def build_per_nonlinearity_summary(
    results: list[dict],
    pairs: list[dict],
    init_names: list[str],
) -> dict:
    """Per-nonlinearity summary: total pairs distinguished across all inits."""
    summary = {}
    for nl_name in NONLINEARITIES:
        total_dist = 0
        frob_values = []
        for init_name in init_names:
            for pair in pairs:
                pid = pair["pair_id"]
                pair_results = [
                    r for r in results
                    if r["init"] == init_name
                    and r["nonlinearity"] == nl_name
                    and r["pair_id"] == pid
                ]
                if any(r["distinguished"] for r in pair_results):
                    total_dist += 1
                    frob_values.append(
                        max(r["frobenius_distance"] for r in pair_results)
                    )

        summary[nl_name] = {
            "total_distinguished_init_pair_combos": total_dist,
            "mean_frobenius": float(np.mean(frob_values)) if frob_values else 0.0,
            "max_frobenius": float(np.max(frob_values)) if frob_values else 0.0,
        }
    return summary


# ============================================================
# STEP 10: Build output JSON conforming to exp_gen_sol_out schema
# ============================================================

def build_output(
    results: list[dict],
    pairs: list[dict],
    gap_analysis: dict,
    spectral_analysis: dict,
    per_init_summary: dict,
    per_pair_summary: dict,
    per_nonlinearity_summary: dict,
    init_names: list[str],
    total_wall_clock_s: float,
) -> dict:
    """Build output JSON conforming to exp_gen_sol_out schema.

    Schema requires:
      - datasets: array of {dataset: str, examples: [{input, output, metadata_*, predict_*}]}
      - metadata: optional top-level
    """
    # Determine hypothesis verdict
    any_positive_gap = any(
        len(g["positive_gap_pairs"]) > 0 for g in gap_analysis.values()
    )

    if any_positive_gap:
        verdict = "MODE_COUPLING_VALIDATED"
    else:
        # Check if ALL primary inits are constant (vertex-transitivity issue)
        all_constant = all(
            r["init_is_constant"]
            for r in results
            if r["init"] in PRIMARY_INITS
        )
        if all_constant:
            verdict = "HYPOTHESIS_FALSIFIED_VERTEX_TRANSITIVITY"
        else:
            verdict = "HYPOTHESIS_FALSIFIED"

    # Critical control: degree init should never distinguish on regular graphs
    degree_control = {
        "degree_linear_distinguished": 0,
        "degree_tanh_distinguished": 0,
        "degree_leaky_relu_distinguished": 0,
        "degree_abs_distinguished": 0,
    }
    for nl_name in NONLINEARITIES:
        key = f"degree_{nl_name}_distinguished"
        count = 0
        for pair in pairs:
            pid = pair["pair_id"]
            pair_results = [
                r for r in results
                if r["init"] == "degree"
                and r["nonlinearity"] == nl_name
                and r["pair_id"] == pid
            ]
            if any(r["distinguished"] for r in pair_results):
                count += 1
        degree_control[key] = count

    metadata = {
        "method_name": "NDS_Ablation",
        "description": (
            "Full ablation of Nonlinear Diffusion Signatures: "
            "scalar initializations x signed nonlinearities on "
            "1-WL-equivalent graph pairs. Tests whether pointwise "
            "nonlinearity between diffusion steps creates cross-frequency "
            "coupling that breaks spectral invariance."
        ),
        "inits": init_names,
        "nonlinearities": list(NONLINEARITIES.keys()),
        "T_range": [1, T_MAX],
        "total_pairs": len(pairs),
        "total_configs": len(init_names) * len(NONLINEARITIES) * T_MAX * len(pairs),
        "total_wall_clock_s": total_wall_clock_s,
        "summary": {
            "hypothesis_verdict": verdict,
            "any_positive_gap": any_positive_gap,
            "gap_analysis": gap_analysis,
            "per_init_summary": per_init_summary,
            "per_nonlinearity_summary": per_nonlinearity_summary,
            "per_pair_summary": per_pair_summary,
            "critical_control": degree_control,
        },
    }

    # Build examples: one per (pair, T) for T in [1,2,3,4,5,10,15,20]
    # This gives 9 pairs x 8 T values = 72 examples (> 50 minimum)
    T_SAMPLE = [1, 2, 3, 4, 5, 10, 15, 20]
    examples = []

    for pair in pairs:
        pid = pair["pair_id"]
        category = pair["category"]
        num_nodes = pair["num_nodes"]
        pair_sum = per_pair_summary.get(pid, {})

        for T_val in T_SAMPLE:
            input_dict = {
                "pair_id": pid,
                "category": category,
                "graph_a_name": pair["graph_a_name"],
                "graph_b_name": pair["graph_b_name"],
                "num_nodes": num_nodes,
                "diffusion_step_T": T_val,
            }

            example = {
                "input": json.dumps(input_dict),
                "output": "non_isomorphic",
                "metadata_pair_id": pid,
                "metadata_category": category,
                "metadata_num_nodes": num_nodes,
                "metadata_diffusion_step_T": T_val,
            }

            # Add predict_ fields for each init x nonlinearity combination
            for init_name in init_names:
                # Check if init features are identical for this pair
                init_ident_key = f"metadata_init_features_identical_{init_name}"
                pair_init_results = [
                    r for r in results
                    if r["init"] == init_name and r["pair_id"] == pid
                ]
                if pair_init_results:
                    example[init_ident_key] = pair_init_results[0]["init_features_identical"]
                    example[f"metadata_init_is_constant_{init_name}"] = pair_init_results[0]["init_is_constant"]

                for nl_name in NONLINEARITIES:
                    predict_key = f"predict_{nl_name}_{init_name}_T{T_val}"

                    # Find the result for this exact config
                    matching = [
                        r for r in results
                        if r["init"] == init_name
                        and r["nonlinearity"] == nl_name
                        and r["pair_id"] == pid
                        and r["T"] == T_val
                    ]

                    if matching:
                        r = matching[0]
                        if r["distinguished"]:
                            example[predict_key] = (
                                f"distinguished_T{T_val}_frob{r['frobenius_distance']:.6e}"
                            )
                        else:
                            example[predict_key] = "indistinguishable"
                    else:
                        example[predict_key] = "no_data"

            # Add gap summary for this pair
            gap_summaries = []
            for key, g in gap_analysis.items():
                if pid in g["positive_gap_pairs"]:
                    gap_summaries.append(f"positive_gap_{key}")
            if gap_summaries:
                example["metadata_gap_summary"] = "; ".join(gap_summaries)
            else:
                example["metadata_gap_summary"] = "no_positive_gap"

            # Add spectral summary
            spectral_keys_for_pair = [
                k for k in spectral_analysis if k.startswith(f"{pid}__")
            ]
            if spectral_keys_for_pair:
                max_modes = max(
                    spectral_analysis[k]["num_nonzero_modes_T20"]
                    for k in spectral_keys_for_pair
                )
                example["metadata_spectral_max_modes_T20"] = max_modes

            examples.append(example)

    output = {
        "metadata": metadata,
        "datasets": [
            {
                "dataset": "nds_ablation_results",
                "examples": examples,
            }
        ],
    }

    return output


# ============================================================
# STEP 11-12: Validate and save
# ============================================================

def save_outputs(output: dict) -> None:
    """Save full, mini, and preview output files."""
    full_path = WORKSPACE / "method_out.json"
    logger.info(f"Saving full output to {full_path}")
    full_path.write_text(json.dumps(output, indent=2))

    # Validate JSON is well-formed by re-parsing
    try:
        json.loads(full_path.read_text())
        logger.info("JSON validation: full output is well-formed")
    except json.JSONDecodeError:
        logger.exception("CRITICAL: full output JSON is malformed!")
        raise

    num_examples = len(output["datasets"][0]["examples"])
    logger.info(f"Total examples in output: {num_examples}")

    # Check all examples have at least one predict_ field
    for i, ex in enumerate(output["datasets"][0]["examples"]):
        predict_keys = [k for k in ex if k.startswith("predict_")]
        if not predict_keys:
            logger.warning(f"Example {i} has NO predict_ fields!")
        elif len(predict_keys) < 5:
            logger.warning(
                f"Example {i} has only {len(predict_keys)} predict_ fields"
            )


# ============================================================
# Main entry point
# ============================================================

@logger.catch
def main() -> None:
    """Run the full NDS ablation experiment."""
    overall_start = time.time()

    # === Load data ===
    data_path = WORKSPACE / "dependencies" / "data_id3_it1__opus" / "full_data_out.json"
    pairs = load_graph_pairs(data_path)

    # === Phase 1: Run primary ablation grid (5 primary inits) ===
    logger.info("=" * 60)
    logger.info("PHASE 1: Primary ablation grid (5 inits x 4 NLs x T=1..20)")
    logger.info("=" * 60)

    primary_results = run_ablation_grid(
        pairs=pairs,
        init_names=PRIMARY_INITS,
        T_max=T_MAX,
    )

    # === Phase 2: Check if primary inits are all constant (vertex-transitivity) ===
    all_primary_constant = all(
        r["init_is_constant"]
        for r in primary_results
        if r["init"] in PRIMARY_INITS and r["T"] == 1
    )

    if all_primary_constant:
        logger.warning(
            "ALL primary inits are constant on ALL pairs! "
            "This is expected for vertex-transitive SRG/CSL graphs."
        )
    else:
        # Check which inits have variation
        for init_name in PRIMARY_INITS:
            non_const_pairs = set()
            for r in primary_results:
                if r["init"] == init_name and r["T"] == 1 and not r["init_is_constant"]:
                    non_const_pairs.add(r["pair_id"])
            if non_const_pairs:
                logger.info(
                    f"Init '{init_name}' has NON-CONSTANT features on: "
                    f"{list(non_const_pairs)}"
                )

    # === Phase 3: Run fallback inits ===
    logger.info("=" * 60)
    logger.info("PHASE 3: Fallback init grid (3 fallback inits x 4 NLs x T=1..20)")
    logger.info("=" * 60)

    fallback_results = run_ablation_grid(
        pairs=pairs,
        init_names=FALLBACK_INITS,
        T_max=T_MAX,
    )

    # Combine all results
    all_results = primary_results + fallback_results
    all_init_names = PRIMARY_INITS + FALLBACK_INITS

    # === Phase 4: Gap analysis ===
    logger.info("=" * 60)
    logger.info("PHASE 4: Gap analysis (nonlinear vs linear)")
    logger.info("=" * 60)

    gap_analysis = compute_gap_analysis(
        results=all_results,
        pairs=pairs,
        init_names=all_init_names,
    )

    # Log gap analysis summary
    any_positive = False
    for key, g in gap_analysis.items():
        if g["positive_gap_pairs"]:
            any_positive = True
            logger.info(
                f"  POSITIVE GAP: {key} -> "
                f"nonlinear distinguishes {g['pairs_distinguished_nonlinear']}/9, "
                f"linear distinguishes {g['pairs_distinguished_linear']}/9, "
                f"gap pairs: {g['positive_gap_pairs']}"
            )

    if not any_positive:
        logger.warning("NO positive gap found for any init x nonlinearity combination!")

    # === Phase 5: Spectral analysis ===
    logger.info("=" * 60)
    logger.info("PHASE 5: Spectral analysis")
    logger.info("=" * 60)

    spectral_analysis = run_spectral_analysis(
        pairs=pairs,
        init_names=all_init_names,
        T_max=T_MAX,
    )

    # === Phase 6: Build summaries ===
    logger.info("=" * 60)
    logger.info("PHASE 6: Building summaries")
    logger.info("=" * 60)

    per_init_summary = build_per_init_summary(
        results=all_results,
        pairs=pairs,
        init_names=all_init_names,
    )
    per_pair_summary = build_per_pair_summary(
        results=all_results,
        pairs=pairs,
        init_names=all_init_names,
    )
    per_nonlinearity_summary = build_per_nonlinearity_summary(
        results=all_results,
        pairs=pairs,
        init_names=all_init_names,
    )

    total_wall_clock = time.time() - overall_start

    # === Phase 7: Build and save output ===
    logger.info("=" * 60)
    logger.info("PHASE 7: Building output JSON")
    logger.info("=" * 60)

    output = build_output(
        results=all_results,
        pairs=pairs,
        gap_analysis=gap_analysis,
        spectral_analysis=spectral_analysis,
        per_init_summary=per_init_summary,
        per_pair_summary=per_pair_summary,
        per_nonlinearity_summary=per_nonlinearity_summary,
        init_names=all_init_names,
        total_wall_clock_s=total_wall_clock,
    )

    save_outputs(output)

    # === Final summary ===
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    verdict = output["metadata"]["summary"]["hypothesis_verdict"]
    logger.info(f"Hypothesis verdict: {verdict}")
    logger.info(f"Total wall clock: {total_wall_clock:.1f}s")
    logger.info(f"Total results: {len(all_results)}")
    logger.info(f"Output examples: {len(output['datasets'][0]['examples'])}")

    # Print critical control results
    ctrl = output["metadata"]["summary"]["critical_control"]
    logger.info("Critical control (degree init, should all be 0 on regular graphs):")
    for key, val in ctrl.items():
        status = "PASS" if val == 0 else "FAIL"
        logger.info(f"  {key}: {val} ({status})")

    # Print per-init distinguishability counts
    logger.info("Per-init distinguishability (pairs distinguished at any T):")
    for init_name in all_init_names:
        for nl_name in NONLINEARITIES:
            count = per_init_summary[init_name][nl_name]["num_distinguished"]
            if count > 0:
                logger.info(f"  {init_name} + {nl_name}: {count}/9 pairs")

    logger.info("Done.")


if __name__ == "__main__":
    main()
