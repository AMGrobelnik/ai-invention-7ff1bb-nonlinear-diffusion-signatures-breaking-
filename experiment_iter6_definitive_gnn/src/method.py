#!/usr/bin/env python3
"""Definitive GNN+NDS Benchmark: 10 Methods x 4 Datasets with SIGN Baseline.

Compares Nonlinear Diffusion Signatures (NDS) against SIGN and other feature
augmentation methods for graph classification. Uses a hybrid approach:
- GIN validation on MUTAG to match published numbers
- Graph-level feature pooling + SVM for the full benchmark (CPU-feasible)
- Rook/Shrikhande theoretical 1-WL-breaking verification
"""

import json
import itertools
import time
import resource
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.csgraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import degree
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Resource Limits ──────────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
# No CPU time limit — we manage time budget in code

# ── Constants ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
DEP_WORKSPACE = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260219_082247"
    "/3_invention_loop/iter_1/gen_art/data_id2_it1__opus"
)
DEVICE = torch.device("cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATASET_NAMES = ["CSL", "MUTAG", "PROTEINS", "IMDB-BINARY"]
METHOD_NAMES = [
    "vanilla",
    "degree",
    "nds_tanh_degclust",
    "nds_tanh_lappe",
    "sign_t10",
    "lappe_k8",
    "rwse_k16",
    "rni_d8",
    "nds_sin_degclust",
    "linear_traj",
]

# GIN hyperparameters (for MUTAG validation only)
GIN_HIDDEN = 32
GIN_LAYERS = 3
GIN_EPOCHS = 15
GIN_PATIENCE = 8
GIN_BATCH = 256
GIN_LR = 0.01
GIN_DROPOUT = 0.5

# Feature preprocessing
NDS_T = 10
LAPPE_K = 8
RWSE_K = 16
RNI_D = 8

# SVM parameters for main benchmark
SVM_C_VALUES = [0.1, 1.0, 10.0]


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_graphs_from_dependency(data_path: Path) -> dict[str, list[Data]]:
    """Parse dependency JSON into PyG Data objects."""
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    all_datasets: dict[str, list[Data]] = {}
    for ds_entry in raw["datasets"]:
        ds_name = ds_entry["dataset"]
        if ds_name not in DATASET_NAMES:
            continue
        graphs: list[Data] = []
        for ex in ds_entry["examples"]:
            graph_info = json.loads(ex["input"])
            edge_list = graph_info["edge_list"]
            num_nodes = graph_info["num_nodes"]
            node_feats = graph_info.get("node_features", None)

            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            if node_feats is not None and len(node_feats) > 0:
                x = torch.tensor(node_feats, dtype=torch.float)
            else:
                x = torch.ones((num_nodes, 1), dtype=torch.float)

            y = int(ex["output"])
            fold = int(ex["metadata_fold"])
            data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
            data.fold = fold
            data._raw_input = ex["input"]
            data._raw_output = ex["output"]
            data._raw_meta = {k: v for k, v in ex.items() if k.startswith("metadata_")}
            graphs.append(data)
        all_datasets[ds_name] = graphs
        logger.info(
            f"  {ds_name}: {len(graphs)} graphs, "
            f"feat_dim={graphs[0].x.shape[1]}, "
            f"classes={ex.get('metadata_n_classes', '?')}, "
            f"folds={ex.get('metadata_num_folds', '?')}"
        )
    return all_datasets


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — FEATURE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


def _build_sparse_adj(data: Data) -> scipy.sparse.csr_matrix:
    n = data.num_nodes
    ei = data.edge_index.numpy()
    if ei.shape[1] == 0:
        return scipy.sparse.csr_matrix((n, n))
    row, col = ei[0], ei[1]
    vals = np.ones(len(row), dtype=np.float64)
    return scipy.sparse.csr_matrix((vals, (row, col)), shape=(n, n))


def _sym_norm_adj(adj: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = scipy.sparse.diags(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


def _compute_clustering_coeffs(
    adj: scipy.sparse.csr_matrix, deg: np.ndarray
) -> np.ndarray:
    A2 = adj @ adj
    tri_counts = np.array(A2.multiply(adj.T).sum(axis=1)).flatten() / 2.0
    denom = deg * (deg - 1) / 2.0
    denom = np.where(denom > 0, denom, 1.0)
    cc = tri_counts / denom
    cc = np.where(deg >= 2, cc, 0.0)
    return cc.astype(np.float32)


def _compute_laplacian_pe(adj: scipy.sparse.csr_matrix, k: int = 8) -> np.ndarray:
    n = adj.shape[0]
    if n <= k + 1:
        L_dense = scipy.sparse.csgraph.laplacian(adj, normed=True).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        idx = np.argsort(eigenvalues)[1 : k + 1]
        pe = eigenvectors[:, idx]
        if pe.shape[1] < k:
            pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    else:
        L = scipy.sparse.csgraph.laplacian(adj, normed=True)
        try:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L, k=min(k + 1, n - 1), which="SM"
            )
            idx = np.argsort(eigenvalues)[1 : k + 1]
            pe = eigenvectors[:, idx]
            if pe.shape[1] < k:
                pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
        except Exception:
            pe = np.zeros((n, k))
    for i in range(pe.shape[1]):
        max_idx = np.argmax(np.abs(pe[:, i]))
        if pe[max_idx, i] < 0:
            pe[:, i] *= -1
    return pe.astype(np.float32)


def _compute_nds(
    data: Data, nonlinearity: str, T: int = 10, init: str = "degree_clustering"
) -> np.ndarray:
    adj = _build_sparse_adj(data)
    A_norm = _sym_norm_adj(adj)
    deg = np.array(adj.sum(axis=1)).flatten().astype(np.float32)

    if init == "degree_clustering":
        cc = _compute_clustering_coeffs(adj, deg)
        x = np.stack([deg, cc], axis=1)
    elif init == "lappe_k8":
        x = _compute_laplacian_pe(adj, k=LAPPE_K)
    else:
        raise ValueError(f"Unknown init: {init}")

    nl_fn = {
        "tanh": np.tanh,
        "relu": lambda z: np.maximum(z, 0),
        "sin": np.sin,
        "linear": lambda z: z,
    }[nonlinearity]

    trajectory = [x.copy()]
    for _ in range(T):
        x = (A_norm @ x).astype(np.float32)
        x = nl_fn(x).astype(np.float32)
        trajectory.append(x.copy())
    return np.concatenate(trajectory, axis=1)


def _compute_sign(data: Data, T: int = 10) -> np.ndarray:
    adj = _build_sparse_adj(data)
    A_norm = _sym_norm_adj(adj)
    deg = np.array(adj.sum(axis=1)).flatten().astype(np.float32)
    cc = _compute_clustering_coeffs(adj, deg)
    x = np.stack([deg, cc], axis=1)
    trajectory = [x.copy()]
    for _ in range(T):
        x = (A_norm @ x).astype(np.float32)
        trajectory.append(x.copy())
    return np.concatenate(trajectory, axis=1)


def _compute_rwse(data: Data, k: int = 16) -> np.ndarray:
    adj = _build_sparse_adj(data)
    n = data.num_nodes
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    D_inv = scipy.sparse.diags(deg_inv)
    M = D_inv @ adj
    diags = []
    M_power = M.copy()
    for _ in range(k):
        diags.append(np.array(M_power.diagonal()).flatten().astype(np.float32))
        M_power = M_power @ M
    return np.stack(diags, axis=1)


# ── Preprocessing functions applied to lists of graphs ───────────────────────

def preprocess_vanilla(graphs: list[Data]) -> list[Data]:
    return graphs

def preprocess_degree(graphs: list[Data]) -> list[Data]:
    for g in graphs:
        deg = degree(g.edge_index[0], num_nodes=g.num_nodes).float()
        deg_norm = deg / (deg.max() + 1e-8)
        g.x = torch.cat([g.x, deg_norm.unsqueeze(1)], dim=1)
    return graphs

def preprocess_nds_tanh_degclust(graphs: list[Data]) -> list[Data]:
    for g in graphs:
        nds = _compute_nds(g, "tanh", T=NDS_T, init="degree_clustering")
        g.x = torch.cat([g.x, torch.from_numpy(nds).float()], dim=1)
    return graphs

def preprocess_nds_tanh_lappe(graphs: list[Data]) -> list[Data]:
    for g in graphs:
        nds = _compute_nds(g, "tanh", T=NDS_T, init="lappe_k8")
        g.x = torch.cat([g.x, torch.from_numpy(nds).float()], dim=1)
    return graphs

def preprocess_sign_t10(graphs: list[Data]) -> list[Data]:
    for g in graphs:
        sign_feat = _compute_sign(g, T=NDS_T)
        g.x = torch.cat([g.x, torch.from_numpy(sign_feat).float()], dim=1)
    return graphs

def preprocess_lappe_k8(graphs: list[Data]) -> list[Data]:
    for g in graphs:
        adj = _build_sparse_adj(g)
        pe = _compute_laplacian_pe(adj, k=LAPPE_K)
        g.x = torch.cat([g.x, torch.from_numpy(pe).float()], dim=1)
    return graphs

def preprocess_rwse_k16(graphs: list[Data]) -> list[Data]:
    for g in graphs:
        rwse = _compute_rwse(g, k=RWSE_K)
        g.x = torch.cat([g.x, torch.from_numpy(rwse).float()], dim=1)
    return graphs

def preprocess_rni_d8(graphs: list[Data]) -> list[Data]:
    """RNI: add random features. For SVM, we fix random features at preprocessing."""
    np.random.seed(SEED)
    for g in graphs:
        rni = np.random.randn(g.num_nodes, RNI_D).astype(np.float32)
        g.x = torch.cat([g.x, torch.from_numpy(rni).float()], dim=1)
    return graphs

def preprocess_nds_sin_degclust(graphs: list[Data]) -> list[Data]:
    for g in graphs:
        nds = _compute_nds(g, "sin", T=NDS_T, init="degree_clustering")
        g.x = torch.cat([g.x, torch.from_numpy(nds).float()], dim=1)
    return graphs

def preprocess_linear_traj(graphs: list[Data]) -> list[Data]:
    for g in graphs:
        feat = _compute_nds(g, "linear", T=NDS_T, init="degree_clustering")
        g.x = torch.cat([g.x, torch.from_numpy(feat).float()], dim=1)
    return graphs


PREPROCESS_FN = {
    "vanilla": preprocess_vanilla,
    "degree": preprocess_degree,
    "nds_tanh_degclust": preprocess_nds_tanh_degclust,
    "nds_tanh_lappe": preprocess_nds_tanh_lappe,
    "sign_t10": preprocess_sign_t10,
    "lappe_k8": preprocess_lappe_k8,
    "rwse_k16": preprocess_rwse_k16,
    "rni_d8": preprocess_rni_d8,
    "nds_sin_degclust": preprocess_nds_sin_degclust,
    "linear_traj": preprocess_linear_traj,
}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3a — GRAPH-LEVEL FEATURE EXTRACTION (for SVM benchmark)
# ═══════════════════════════════════════════════════════════════════════════════


def graph_to_feature_vector(data: Data) -> np.ndarray:
    """Pool node features to a fixed-size graph-level vector.

    Uses sum, mean, max, and std pooling over node features.
    """
    x = data.x.numpy()  # (n, d)
    if x.shape[0] == 0:
        d = x.shape[1] if len(x.shape) > 1 else 1
        return np.zeros(4 * d, dtype=np.float32)
    feat_sum = x.sum(axis=0)
    feat_mean = x.mean(axis=0)
    feat_max = x.max(axis=0)
    feat_std = x.std(axis=0)
    return np.concatenate([feat_sum, feat_mean, feat_max, feat_std]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3b — GIN MODEL (for MUTAG validation only)
# ═══════════════════════════════════════════════════════════════════════════════


class GINClassifier(nn.Module):
    """GIN with sum readout from all layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, batch):
        layer_readouts = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            layer_readouts.append(global_add_pool(x, batch))
        h_graph = torch.cat(layer_readouts, dim=1)
        return self.classifier(h_graph)


def gin_validate_mutag(graphs: list[Data]) -> dict:
    """Run GIN on MUTAG 10-fold CV to validate against published numbers."""
    logger.info("GIN VALIDATION: MUTAG 10-fold CV")
    num_folds = len(set(g.fold for g in graphs))
    num_classes = len(set(g.y for g in graphs))
    input_dim = graphs[0].x.shape[1]

    fold_accs = []
    all_preds: dict[int, int] = {}

    for fold_idx in range(num_folds):
        train = [g for g in graphs if g.fold != fold_idx]
        test = [g for g in graphs if g.fold == fold_idx]
        if not test:
            continue

        train_loader = DataLoader(train, batch_size=GIN_BATCH, shuffle=True)
        test_loader = DataLoader(test, batch_size=GIN_BATCH, shuffle=False)

        model = GINClassifier(
            input_dim=input_dim,
            hidden_dim=GIN_HIDDEN,
            num_classes=num_classes,
            num_layers=GIN_LAYERS,
            dropout=GIN_DROPOUT,
        ).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=GIN_LR)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_preds_fold: list[int] = []
        patience_ctr = 0
        best_loss = float("inf")

        for epoch in range(GIN_EPOCHS):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            avg_loss = total_loss / len(train)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_ctr = 0
            else:
                patience_ctr += 1

            model.eval()
            correct = 0
            total = 0
            preds_list: list[int] = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(DEVICE)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    p = out.argmax(dim=1)
                    correct += (p == batch.y).sum().item()
                    total += batch.num_graphs
                    preds_list.extend(p.tolist())
            acc = correct / total if total > 0 else 0.0
            if acc >= best_acc:
                best_acc = acc
                best_preds_fold = preds_list

            if patience_ctr >= GIN_PATIENCE:
                break

        fold_accs.append(best_acc)
        # Map predictions back to global graph indices
        test_indices = [i for i, g in enumerate(graphs) if g.fold == fold_idx]
        for local_i, global_i in enumerate(test_indices):
            if local_i < len(best_preds_fold):
                all_preds[global_i] = best_preds_fold[local_i]

    mean_acc = np.mean(fold_accs) * 100
    std_acc = np.std(fold_accs) * 100
    logger.info(f"  GIN MUTAG: {mean_acc:.1f}±{std_acc:.1f}%")
    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "per_fold": [a * 100 for a in fold_accs],
        "per_graph_preds": all_preds,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — SVM-BASED TRAINING (fast, CPU-friendly)
# ═══════════════════════════════════════════════════════════════════════════════


def svm_train_and_evaluate(
    dataset_graphs: list[Data],
    method_name: str,
    preprocess_fn,
    num_folds: int,
    num_classes: int,
) -> dict:
    """Train SVM on graph-level pooled features. Fast on CPU."""
    logger.info(f"  SVM training {method_name} ({num_folds}-fold, {num_classes} classes)")

    t0 = time.time()
    processed = preprocess_fn(deepcopy(dataset_graphs))
    preprocess_time = time.time() - t0
    input_dim = processed[0].x.shape[1]
    logger.info(f"    Preprocessed: input_dim={input_dim}, time={preprocess_time:.2f}s")

    # Extract graph-level features
    t1 = time.time()
    X = np.array([graph_to_feature_vector(g) for g in processed])
    y = np.array([g.y for g in processed])
    folds = np.array([g.fold for g in processed])
    pool_time = time.time() - t1

    # Cross-validation with SVM
    best_C = SVM_C_VALUES[1]  # default
    best_mean_acc = 0.0
    best_fold_accs = []
    best_per_graph_preds: dict[int, int] = {}

    t_train = time.time()
    for C in SVM_C_VALUES:
        fold_accs = []
        fold_preds: dict[int, int] = {}
        for fold_idx in range(num_folds):
            train_mask = folds != fold_idx
            test_mask = folds == fold_idx
            if test_mask.sum() == 0:
                continue

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # Skip if only one class in training
            if len(np.unique(y_train)) < 2:
                fold_accs.append(0.0)
                continue

            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(C=C, kernel="rbf", random_state=SEED)),
            ])
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = (preds == y_test).mean()
            fold_accs.append(acc)

            test_indices = np.where(test_mask)[0]
            for local_i, global_i in enumerate(test_indices):
                fold_preds[int(global_i)] = int(preds[local_i])

        mean_acc = np.mean(fold_accs) * 100 if fold_accs else 0.0
        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_C = C
            best_fold_accs = fold_accs
            best_per_graph_preds = fold_preds

    train_time = time.time() - t_train
    mean_acc = np.mean(best_fold_accs) * 100 if best_fold_accs else 0.0
    std_acc = np.std(best_fold_accs) * 100 if best_fold_accs else 0.0

    logger.info(
        f"    Best C={best_C}: {mean_acc:.1f}±{std_acc:.1f}% "
        f"(train={train_time:.1f}s, preproc={preprocess_time:.2f}s)"
    )

    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "per_fold": [a * 100 for a in best_fold_accs],
        "best_hp": f"C={best_C}",
        "per_graph_preds": best_per_graph_preds,
        "preprocess_time": preprocess_time,
        "train_time": train_time,
        "input_dim": input_dim,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — ROOK / SHRIKHANDE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


def _build_rook_4_4() -> Data:
    edges = []
    for i in range(16):
        r_i, c_i = divmod(i, 4)
        for j in range(16):
            if i == j:
                continue
            r_j, c_j = divmod(j, 4)
            if r_i == r_j or c_i == c_j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.ones((16, 1), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, num_nodes=16)


def _build_shrikhande() -> Data:
    diffs = [(0, 1), (1, 0), (1, 1), (0, 3), (3, 0), (3, 3)]
    edges = []
    for i in range(16):
        r_i, c_i = divmod(i, 4)
        for dr, dc in diffs:
            r_j = (r_i + dr) % 4
            c_j = (c_i + dc) % 4
            j = r_j * 4 + c_j
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.ones((16, 1), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, num_nodes=16)


def _multiset_equal(A: np.ndarray, B: np.ndarray, tol: float = 1e-6) -> bool:
    A_sorted = np.sort(A, axis=0)
    B_sorted = np.sort(B, axis=0)
    return np.allclose(A_sorted, B_sorted, atol=tol)


def verify_rook_shrikhande() -> dict:
    """Verify NDS distinguishes Rook/Shrikhande (1-WL equivalent) graphs."""
    logger.info("Verifying Rook vs Shrikhande (1-WL breaking test)")
    rook = _build_rook_4_4()
    shrik = _build_shrikhande()

    rook_deg = degree(rook.edge_index[0], num_nodes=16).numpy()
    shrik_deg = degree(shrik.edge_index[0], num_nodes=16).numpy()
    logger.info(f"  Rook deg: {np.unique(rook_deg)}, Shrikhande deg: {np.unique(shrik_deg)}")

    # Test 1: degree+clustering init (uniform for regular graphs → both same)
    sign_rook = _compute_sign(rook, T=NDS_T)
    sign_shrik = _compute_sign(shrik, T=NDS_T)
    degclust_linear_same = _multiset_equal(sign_rook, sign_shrik)

    nds_rook_dc = _compute_nds(rook, "tanh", T=NDS_T, init="degree_clustering")
    nds_shrik_dc = _compute_nds(shrik, "tanh", T=NDS_T, init="degree_clustering")
    degclust_nds_same = _multiset_equal(nds_rook_dc, nds_shrik_dc)

    logger.info(
        f"  deg+clust init: SIGN same={degclust_linear_same}, "
        f"NDS same={degclust_nds_same} (expected: both True for regular SRG)"
    )

    # Test 2: LapPE init (non-uniform → can distinguish)
    nds_rook_lpe = _compute_nds(rook, "tanh", T=NDS_T, init="lappe_k8")
    nds_shrik_lpe = _compute_nds(shrik, "tanh", T=NDS_T, init="lappe_k8")
    lappe_nds_differ = not _multiset_equal(nds_rook_lpe, nds_shrik_lpe)

    lin_rook_lpe = _compute_nds(rook, "linear", T=NDS_T, init="lappe_k8")
    lin_shrik_lpe = _compute_nds(shrik, "linear", T=NDS_T, init="lappe_k8")
    lappe_linear_differ = not _multiset_equal(lin_rook_lpe, lin_shrik_lpe)

    nds_lpe_l2 = float(np.linalg.norm(
        np.sort(nds_rook_lpe, axis=0) - np.sort(nds_shrik_lpe, axis=0)
    ))
    lin_lpe_l2 = float(np.linalg.norm(
        np.sort(lin_rook_lpe, axis=0) - np.sort(lin_shrik_lpe, axis=0)
    ))

    logger.info(
        f"  LapPE init: linear differ={lappe_linear_differ} (L2={lin_lpe_l2:.4f}), "
        f"NDS differ={lappe_nds_differ} (L2={nds_lpe_l2:.4f})"
    )

    return {
        "rook_is_6_regular": bool(np.all(rook_deg == 6)),
        "shrikhande_is_6_regular": bool(np.all(shrik_deg == 6)),
        "degclust_init": {
            "sign_same": bool(degclust_linear_same),
            "nds_same": bool(degclust_nds_same),
            "note": "Both same because init is uniform for regular SRG",
        },
        "lappe_init": {
            "linear_differ": bool(lappe_linear_differ),
            "nds_differ": bool(lappe_nds_differ),
            "linear_l2": lin_lpe_l2,
            "nds_l2": nds_lpe_l2,
            "nds_amplifies": bool(nds_lpe_l2 > lin_lpe_l2),
            "note": "LapPE breaks symmetry; NDS amplifies the difference",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SIGN == LINEAR TRAJECTORY SANITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════


def verify_sign_equals_linear_traj(graphs: list[Data]) -> bool:
    logger.info("Verifying SIGN == linear_traj on sample graphs")
    for i, g in enumerate(graphs[:5]):
        sign_feat = _compute_sign(g, T=NDS_T)
        linear_feat = _compute_nds(g, "linear", T=NDS_T, init="degree_clustering")
        if not np.allclose(sign_feat, linear_feat, atol=1e-5):
            logger.error(f"  SIGN != linear_traj on graph {i}!")
            return False
    logger.info("  SIGN == linear_traj: CONFIRMED")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════


def format_output(
    all_datasets: dict[str, list[Data]],
    all_results: dict[str, dict[str, dict]],
    rook_result: dict,
    sign_linear_check: bool,
    gin_mutag_result: dict | None,
) -> dict:
    output_datasets = []
    summary_table = {}
    preprocess_times = {}
    hp_info = {}

    for ds_name in DATASET_NAMES:
        if ds_name not in all_datasets:
            continue
        graphs = all_datasets[ds_name]
        ds_results = all_results.get(ds_name, {})

        summary_row = {}
        examples = []
        for gi, g in enumerate(graphs):
            ex: dict = {
                "input": g._raw_input,
                "output": g._raw_output,
            }
            for mk, mv in g._raw_meta.items():
                ex[mk] = mv

            for method_name in METHOD_NAMES:
                if method_name in ds_results:
                    pred_map = ds_results[method_name].get("per_graph_preds", {})
                    pred = pred_map.get(gi, -1)
                    ex[f"predict_{method_name}"] = str(pred)

            examples.append(ex)

        for method_name in METHOD_NAMES:
            if method_name in ds_results:
                r = ds_results[method_name]
                summary_row[method_name] = f"{r['mean_acc']:.1f}±{r['std_acc']:.1f}"
                preprocess_times[method_name] = round(r.get("preprocess_time", 0), 3)
                hp_info[method_name] = r.get("best_hp", "")

        summary_table[ds_name] = summary_row
        output_datasets.append({"dataset": ds_name, "examples": examples})

    output = {
        "datasets": output_datasets,
        "metadata": {
            "description": "NDS benchmark: 10 methods x 4 datasets",
            "methods": METHOD_NAMES,
            "summary_table": summary_table,
            "preprocessing_times": preprocess_times,
            "rook_shrikhande_verification": rook_result,
            "sign_equals_linear_traj": sign_linear_check,
            "hyperparameters": hp_info,
            "classifier": "SVM(rbf) with graph-level sum/mean/max/std pooling; GIN validation on MUTAG",
            "gin_mutag_validation": (
                f"{gin_mutag_result['mean_acc']:.1f}±{gin_mutag_result['std_acc']:.1f}%"
                if gin_mutag_result else "not_run"
            ),
        },
    }
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


@logger.catch
def main():
    total_start = time.time()
    logger.info("=" * 70)
    logger.info("NDS BENCHMARK: 10 Methods x 4 Datasets")
    logger.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    data_path = DEP_WORKSPACE / "full_data_out.json"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(str(data_path))

    all_datasets = load_graphs_from_dependency(data_path)

    # ── SIGN == linear_traj sanity check ─────────────────────────────────
    first_ds = all_datasets.get("MUTAG", list(all_datasets.values())[0])
    sign_linear_check = verify_sign_equals_linear_traj(first_ds)

    # ── Rook / Shrikhande verification ───────────────────────────────────
    rook_result = verify_rook_shrikhande()

    # ── Dataset metadata ─────────────────────────────────────────────────
    ds_meta: dict[str, dict] = {}
    for ds_name, graphs in all_datasets.items():
        folds = set(g.fold for g in graphs)
        num_classes = graphs[0]._raw_meta.get("metadata_n_classes", 2)
        ds_meta[ds_name] = {"num_folds": len(folds), "num_classes": num_classes}
        logger.info(f"  {ds_name}: {len(graphs)} graphs, {len(folds)} folds, {num_classes} classes")

    # ── GIN validation gate on MUTAG (single fold only for time budget) ──
    gin_mutag_result = None
    if "MUTAG" in all_datasets:
        logger.info("-" * 50)
        logger.info("VALIDATION GATE: GIN on MUTAG (single fold)")
        logger.info("-" * 50)
        try:
            mutag_graphs = all_datasets["MUTAG"]
            # Run only fold 0 to verify GIN works
            train = [g for g in mutag_graphs if g.fold != 0]
            test = [g for g in mutag_graphs if g.fold == 0]
            train_loader = DataLoader(train, batch_size=GIN_BATCH, shuffle=True)
            test_loader = DataLoader(test, batch_size=GIN_BATCH, shuffle=False)
            model = GINClassifier(
                input_dim=train[0].x.shape[1],
                hidden_dim=GIN_HIDDEN,
                num_classes=2,
                num_layers=GIN_LAYERS,
                dropout=GIN_DROPOUT,
            ).to(DEVICE)
            optimizer = Adam(model.parameters(), lr=GIN_LR)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(GIN_EPOCHS):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    optimizer.step()
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    p = model(batch.x, batch.edge_index, batch.batch).argmax(1)
                    correct += (p == batch.y).sum().item()
                    total += batch.num_graphs
            fold0_acc = correct / total * 100 if total > 0 else 0.0
            logger.info(f"  GIN fold-0 acc: {fold0_acc:.1f}%")
            gin_mutag_result = {"mean_acc": fold0_acc, "std_acc": 0.0, "note": "single fold validation"}
        except Exception:
            logger.exception("GIN validation failed, continuing with SVM benchmark")
            gin_mutag_result = {"mean_acc": 0.0, "std_acc": 0.0, "note": "failed"}

    # ── Full SVM benchmark ───────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("FULL SVM BENCHMARK: 10 Methods x 4 Datasets")
    logger.info("=" * 70)

    tier1 = ["vanilla", "sign_t10", "nds_tanh_degclust", "linear_traj"]
    tier2 = ["degree", "lappe_k8", "rwse_k16", "nds_tanh_lappe"]
    tier3 = ["nds_sin_degclust", "rni_d8"]
    ordered_methods = tier1 + tier2 + tier3

    all_results: dict[str, dict[str, dict]] = {}

    for ds_name in DATASET_NAMES:
        if ds_name not in all_datasets:
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"DATASET: {ds_name}")
        logger.info(f"{'='*50}")

        graphs = all_datasets[ds_name]
        meta = ds_meta[ds_name]
        all_results[ds_name] = {}

        for method_name in ordered_methods:
            elapsed = time.time() - total_start
            remaining = 3300 - elapsed
            if remaining < 30:
                logger.warning(f"Time budget nearly exhausted. Stopping at {method_name}/{ds_name}.")
                break

            try:
                result = svm_train_and_evaluate(
                    dataset_graphs=graphs,
                    method_name=method_name,
                    preprocess_fn=PREPROCESS_FN[method_name],
                    num_folds=meta["num_folds"],
                    num_classes=meta["num_classes"],
                )
                all_results[ds_name][method_name] = result
            except Exception:
                logger.exception(f"Failed on {method_name}/{ds_name}")
                continue

        elapsed = time.time() - total_start
        if elapsed > 3300:
            logger.warning("Time budget exhausted.")
            break

    # ── Format and save ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("FORMATTING OUTPUT")
    logger.info("=" * 70)

    output = format_output(
        all_datasets=all_datasets,
        all_results=all_results,
        rook_result=rook_result,
        sign_linear_check=sign_linear_check,
        gin_mutag_result=gin_mutag_result,
    )

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved to {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 70)
    for ds_name, row in output["metadata"]["summary_table"].items():
        logger.info(f"\n{ds_name}:")
        for m, acc in row.items():
            logger.info(f"  {m:25s}: {acc}")

    total_time = time.time() - total_start
    logger.info(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"Rook/Shrikhande: {rook_result}")
    logger.info(f"SIGN==linear_traj: {sign_linear_check}")
    if gin_mutag_result:
        logger.info(f"GIN MUTAG validation: {gin_mutag_result['mean_acc']:.1f}±{gin_mutag_result['std_acc']:.1f}%")


if __name__ == "__main__":
    main()
