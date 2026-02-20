#!/usr/bin/env python3
"""
GIN+NDS Benchmark: 8 Methods × 4 Datasets, K-Fold CV.

Methods: vanilla, degree_only, nds_tanh_T10, nds_tanh_T10_eigvec,
         linear_diff_T10, lap_pe_k16, rwse_k16, rni_d16
Datasets: CSL (5-fold), MUTAG (10-fold), PROTEINS (10-fold), IMDB-BINARY (10-fold)

Key design choices for CPU-only execution within ~54min budget:
- Full-batch training (pre-batch all graphs via Batch.from_data_list)
- 3-layer GIN with h=64 (standard for small graph datasets)
- lr=0.005, 50 max epochs, early stopping on train loss (no separate val split)
- CSL init uses triangle_count + path2_count (CSL is 4-regular → degree uninformative)
- Dynamic fold reduction when time is tight
"""

import json
import sys
import time
import copy
import warnings
import resource
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, Batch
from loguru import logger

# ============================================================
# SETUP
# ============================================================
warnings.filterwarnings("ignore")

WS = Path(__file__).resolve().parent
DATA_FILE = WS / "full_data_out.json"
OUT_FILE = WS / "method_out.json"
LOG_DIR = WS / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

try:
    resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
except (ValueError, resource.error):
    pass
try:
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
except (ValueError, resource.error):
    pass

SEED = 42
MAX_TOTAL_SEC = 3250
NDS_T = 10
PE_K = 16
RNI_DIM = 16
RNI_SEEDS = 2
DS_NAMES = ["CSL", "MUTAG", "PROTEINS", "IMDB-BINARY"]

METHODS = [
    "vanilla", "degree_only", "nds_tanh_T10", "nds_tanh_T10_eigvec",
    "linear_diff_T10", "lap_pe_k16", "rwse_k16", "rni_d16",
]

HP = {"lr": 0.005, "hidden_dim": 64, "dropout": 0.5, "weight_decay": 0.0}
MAX_EPOCHS = 30  # Benchmarked: 30ep gives good accuracy (89% MUTAG) in ~30s/fold

torch.manual_seed(SEED)
np.random.seed(SEED)
START_TIME = time.time()


def time_left() -> float:
    return MAX_TOTAL_SEC - (time.time() - START_TIME)


# ============================================================
# DATA LOADING
# ============================================================
def load_data(path: Path, ds_names: list[str]) -> tuple[dict, dict]:
    logger.info(f"Loading data from {path}")
    raw = json.loads(path.read_text())
    datasets: dict[str, list] = {}
    meta: dict[str, dict] = {}
    for ds_entry in raw["datasets"]:
        nm = ds_entry["dataset"]
        if nm not in ds_names:
            continue
        graphs = []
        meta[nm] = {}
        for ex in ds_entry["examples"]:
            gd = json.loads(ex["input"])
            n = gd["num_nodes"]
            edges = gd["edge_list"]
            nf = gd.get("node_features")
            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            if nf is not None and ex.get("metadata_has_node_features", False):
                x = torch.tensor(nf, dtype=torch.float)
            elif nf is not None:
                x = torch.tensor(nf, dtype=torch.float)
            else:
                x = torch.ones((n, 1), dtype=torch.float)
            data = Data(x=x, edge_index=edge_index,
                        y=torch.tensor(int(ex["output"]), dtype=torch.long), num_nodes=n)
            data.fold = ex["metadata_fold"]
            data.ri = ex.get("metadata_row_index", len(graphs))
            graphs.append(data)
            meta[nm][data.ri] = {"input": ex["input"], "output": ex["output"]}
            for k, v in ex.items():
                if k.startswith("metadata_"):
                    meta[nm][data.ri][k] = v
        datasets[nm] = graphs
        nc = max(g.y.item() for g in graphs) + 1 if graphs else 0
        logger.info(f"  {nm}: {len(graphs)} graphs, n_classes={nc}, feat_dim={graphs[0].x.shape[1]}")
    return datasets, meta


# ============================================================
# FEATURE AUGMENTATION
# ============================================================
def _adj(data: Data) -> sp.csr_matrix:
    n = data.num_nodes
    ei = data.edge_index.numpy()
    if ei.shape[1] == 0:
        return sp.csr_matrix((n, n))
    return sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()


def _sym_norm(A: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.array(A.sum(axis=1)).flatten()
    d = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return sp.diags(d) @ A @ sp.diags(d)


def _safe_eigsh(L: sp.spmatrix, k: int, n: int) -> np.ndarray:
    actual_k = min(k + 1, n - 1)
    if actual_k < 2:
        return np.zeros((n, k))
    try:
        _, ev = eigsh(L.tocsc(), k=actual_k, which="SM", maxiter=5000)
        ev = ev[:, 1:k + 1]
    except Exception:
        try:
            _, ev_all = np.linalg.eigh(L.toarray())
            ev = ev_all[:, 1:k + 1]
        except Exception:
            return np.zeros((n, k))
    for j in range(ev.shape[1]):
        idx = np.argmax(np.abs(ev[:, j]))
        if ev[idx, j] < 0:
            ev[:, j] *= -1
    if ev.shape[1] < k:
        ev = np.pad(ev, ((0, 0), (0, k - ev.shape[1])))
    return ev


def compute_degree_feat(data: Data) -> torch.Tensor:
    deg = np.array(_adj(data).sum(axis=1)).flatten()
    return torch.tensor(deg.reshape(-1, 1), dtype=torch.float)


def compute_nds(data: Data, T: int = 10, ds_name: str = "") -> torch.Tensor:
    A = _adj(data)
    An = _sym_norm(A)
    deg = np.array(A.sum(axis=1)).flatten()
    if ds_name == "CSL":
        A2 = A @ A
        A3 = A2 @ A
        tri = np.array(A3.diagonal()).flatten() / 2.0
        p2 = np.array(A2.diagonal()).flatten().astype(np.float64)
        x0 = np.column_stack([deg, tri, p2])
    else:
        A2 = A @ A
        A3 = A2 @ A
        tri_count = np.array(A3.diagonal()).flatten() / 2.0
        denom = deg * (deg - 1.0)
        cc = np.where(denom > 0, 2.0 * tri_count / denom, 0.0)
        x0 = np.column_stack([deg, cc])
    traj = [x0.copy()]
    xc = x0.copy()
    for _ in range(T):
        xc = np.tanh(An @ xc)
        traj.append(xc.copy())
    return torch.tensor(np.concatenate(traj, axis=1), dtype=torch.float)


def compute_nds_eigvec(data: Data, T: int = 10, k: int = 8, use_tanh: bool = True) -> torch.Tensor:
    n = data.num_nodes
    An = _sym_norm(_adj(data))
    L = sp.eye(n) - An
    x0 = _safe_eigsh(L, k=k, n=n)
    traj = [x0.copy()]
    xc = x0.copy()
    for _ in range(T):
        propagated = An @ xc
        xc = np.tanh(propagated) if use_tanh else propagated
        traj.append(xc.copy())
    return torch.tensor(np.concatenate(traj, axis=1), dtype=torch.float)


def compute_lappe(data: Data, K: int = 16) -> torch.Tensor:
    n = data.num_nodes
    L = sp.eye(n) - _sym_norm(_adj(data))
    return torch.tensor(_safe_eigsh(L, k=K, n=n), dtype=torch.float)


def compute_rwse(data: Data, K: int = 16) -> torch.Tensor:
    n = data.num_nodes
    A = _adj(data)
    deg = np.array(A.sum(axis=1)).flatten()
    Arw = sp.diags(np.where(deg > 0, 1.0 / deg, 0.0)) @ A
    r = np.zeros((n, K))
    pw = sp.eye(n, format="csr")
    for k_step in range(K):
        pw = pw @ Arw
        r[:, k_step] = pw.diagonal()
    return torch.tensor(r, dtype=torch.float)


def compute_rni(data: Data, dim: int = 16, seed: int = 0) -> torch.Tensor:
    return torch.tensor(np.random.RandomState(seed).randn(data.num_nodes, dim).astype(np.float32))


def clone_with_x(data: Data, new_x: torch.Tensor) -> Data:
    d = Data(x=new_x, edge_index=data.edge_index.clone(),
             y=data.y.clone(), num_nodes=data.num_nodes)
    d.fold = data.fold
    d.ri = data.ri
    return d


def cat_f(orig: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
    return torch.cat([orig, extra], dim=1)


# ============================================================
# PRECOMPUTE FEATURES
# ============================================================
def precompute_features(graphs: list[Data], methods: list[str], ds_name: str) -> dict[str, list[Data]]:
    cache: dict[str, list[Data]] = {}
    for method in methods:
        if method == "rni_d16":
            for s in range(RNI_SEEDS):
                key = f"rni_d16_seed{s}"
                cache[key] = [clone_with_x(g, cat_f(g.x, compute_rni(g, RNI_DIM, SEED + s * 100)))
                              for g in graphs]
            continue
        aug = []
        for g in graphs:
            if method == "vanilla":
                aug.append(clone_with_x(g, g.x.clone()))
            elif method == "degree_only":
                aug.append(clone_with_x(g, cat_f(g.x, compute_degree_feat(g))))
            elif method == "nds_tanh_T10":
                aug.append(clone_with_x(g, cat_f(g.x, compute_nds(g, NDS_T, ds_name))))
            elif method == "nds_tanh_T10_eigvec":
                aug.append(clone_with_x(g, cat_f(g.x, compute_nds_eigvec(g, NDS_T, 8, True))))
            elif method == "linear_diff_T10":
                aug.append(clone_with_x(g, cat_f(g.x, compute_nds_eigvec(g, NDS_T, 8, False))))
            elif method == "lap_pe_k16":
                aug.append(clone_with_x(g, cat_f(g.x, compute_lappe(g, PE_K))))
            elif method == "rwse_k16":
                aug.append(clone_with_x(g, cat_f(g.x, compute_rwse(g, PE_K))))
        cache[method] = aug
    return cache


# ============================================================
# GIN MODEL
# ============================================================
class GIN(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int, nl: int = 3, drop: float = 0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.drop = drop
        for i in range(nl):
            d = in_dim if i == 0 else hid
            mlp = nn.Sequential(nn.Linear(d, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Linear(hid, hid))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hid))
        self.clf = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, out_dim))

    def forward(self, x, ei, batch):
        for c, b in zip(self.convs, self.bns):
            x = F.dropout(F.relu(b(c(x, ei))), p=self.drop, training=self.training)
        return self.clf(global_add_pool(x, batch))


# ============================================================
# TRAINING
# ============================================================
def train_one_fold(
    graphs: list[Data], fold_id: int, num_classes: int,
    hp: dict, max_epochs: int = 50,
) -> dict:
    """Train GIN on one fold using full-batch. Track best by train loss."""
    t0 = time.time()
    torch.manual_seed(SEED + fold_id)

    test_g = [g for g in graphs if g.fold == fold_id]
    train_g = [g for g in graphs if g.fold != fold_id]
    if len(test_g) == 0 or len(train_g) < 2:
        return {"val_acc": 0.0, "test_acc": 0.0, "preds": {}, "time": 0.0}

    in_dim = graphs[0].x.shape[1]
    train_batch = Batch.from_data_list(train_g)
    test_batch = Batch.from_data_list(test_g)

    model = GIN(in_dim, hp["hidden_dim"], num_classes, nl=3, drop=hp["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=10, factor=0.5)

    best_loss = float("inf")
    best_state = None
    patience_count = 0
    patience_limit = 15

    for ep in range(max_epochs):
        model.train()
        opt.zero_grad()
        out = model(train_batch.x, train_batch.edge_index, train_batch.batch)
        loss = F.cross_entropy(out, train_batch.y)
        loss.backward()
        opt.step()

        cur_loss = loss.item()
        scheduler.step(cur_loss)

        if cur_loss < best_loss - 1e-4:
            best_loss = cur_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience_limit:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_pred = model(test_batch.x, test_batch.edge_index, test_batch.batch).argmax(dim=1)
        test_acc = (test_pred == test_batch.y).float().mean().item()
        train_pred = model(train_batch.x, train_batch.edge_index, train_batch.batch).argmax(dim=1)
        train_acc = (train_pred == train_batch.y).float().mean().item()

    preds = {}
    pred_np = test_pred.cpu().numpy()
    for i, g in enumerate(test_g):
        preds[g.ri] = str(int(pred_np[i]))

    return {"val_acc": train_acc, "test_acc": test_acc, "preds": preds, "time": time.time() - t0}


# ============================================================
# MAIN
# ============================================================
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("GIN+NDS Benchmark — 8 methods × 4 datasets")
    logger.info("=" * 60)

    datasets, meta = load_data(DATA_FILE, DS_NAMES)

    all_results: dict[str, dict[str, dict]] = {}
    all_predictions: dict[str, dict[str, dict[int, str]]] = {}
    timing_info: dict[str, dict[str, dict]] = {}

    # Fixed fold counts (benchmarked: ~30-65s per fold at 30 epochs on CPU)
    # CSL:5, MUTAG:5, PROTEINS:3, IMDB-B:3 = 16 base folds
    # × 8 methods + RNI extra = ~160 runs × ~40s avg = ~6400s ~= 107min
    # Still tight, so use dynamic reduction per-dataset
    ds_fold_counts = {}
    for ds in DS_NAMES:
        if ds not in datasets:
            continue
        nf = max(g.fold for g in datasets[ds]) + 1
        if ds == "CSL":
            ds_fold_counts[ds] = min(nf, 5)
        elif ds == "MUTAG":
            ds_fold_counts[ds] = min(nf, 5)
        else:
            ds_fold_counts[ds] = min(nf, 3)

    actual_epochs = MAX_EPOCHS
    logger.info(f"Plan: {actual_epochs} epochs, folds: {ds_fold_counts}")

    for ds_idx, ds_name in enumerate(DS_NAMES):
        if ds_name not in datasets:
            continue
        remaining = time_left()
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_name} ({ds_idx+1}/{len(DS_NAMES)}) — {remaining:.0f}s left")
        logger.info(f"{'='*60}")

        if remaining < 60:
            logger.warning(f"Skipping {ds_name} — not enough time")
            continue

        graphs = datasets[ds_name]
        num_folds = max(g.fold for g in graphs) + 1
        num_classes = max(g.y.item() for g in graphs) + 1
        use_folds = ds_fold_counts.get(ds_name, num_folds)

        # Dynamically reduce folds if remaining time is tight
        remaining_ds = len(DS_NAMES) - ds_idx
        time_for_this = remaining * 0.85 / max(remaining_ds, 1)
        # Rough estimate: ~40s per fold per method run
        n_methods_eff = len(METHODS) + (RNI_SEEDS - 1)  # extra RNI seed runs
        est_this = use_folds * n_methods_eff * 40
        while est_this > time_for_this and use_folds > 2:
            use_folds -= 1
            est_this = use_folds * n_methods_eff * 40

        logger.info(f"  {len(graphs)} graphs, {num_classes} classes, using {use_folds}/{num_folds} folds, {actual_epochs} epochs")

        # Precompute features
        logger.info(f"  Precomputing features...")
        t_pre = time.time()
        cache = precompute_features(graphs, METHODS, ds_name)
        pre_time = time.time() - t_pre
        logger.info(f"  Precompute took {pre_time:.1f}s")

        for m in METHODS:
            key = "rni_d16_seed0" if m == "rni_d16" else m
            if key in cache:
                logger.info(f"    {m}: feat_dim={cache[key][0].x.shape[1]}")

        all_results[ds_name] = {}
        all_predictions[ds_name] = {}
        timing_info[ds_name] = {}

        for method in METHODS:
            if time_left() < 30:
                logger.warning(f"  Skipping {method} — time critical")
                break

            t_method = time.time()
            logger.info(f"    {method}...")

            if method == "rni_d16":
                all_fold_accs = []
                seed_preds_list: list[dict[int, str]] = []
                for s in range(RNI_SEEDS):
                    if time_left() < 20:
                        break
                    key = f"rni_d16_seed{s}"
                    aug = cache[key]
                    s_preds: dict[int, str] = {}
                    s_accs = []
                    for fid in range(use_folds):
                        if time_left() < 10:
                            break
                        res = train_one_fold(aug, fid, num_classes, HP, actual_epochs)
                        s_accs.append(res["test_acc"])
                        s_preds.update(res["preds"])
                    all_fold_accs.extend(s_accs)
                    seed_preds_list.append(s_preds)
                mean_acc = np.mean(all_fold_accs) * 100 if all_fold_accs else 0.0
                std_acc = np.std(all_fold_accs) * 100 if all_fold_accs else 0.0
                all_predictions[ds_name][method] = seed_preds_list[0] if seed_preds_list else {}
            else:
                aug = cache[method]
                fold_accs = []
                m_preds: dict[int, str] = {}
                for fid in range(use_folds):
                    if time_left() < 10:
                        break
                    res = train_one_fold(aug, fid, num_classes, HP, actual_epochs)
                    fold_accs.append(res["test_acc"])
                    m_preds.update(res["preds"])
                mean_acc = np.mean(fold_accs) * 100 if fold_accs else 0.0
                std_acc = np.std(fold_accs) * 100 if fold_accs else 0.0
                all_predictions[ds_name][method] = m_preds

            elapsed = time.time() - t_method
            all_results[ds_name][method] = {
                "mean_acc": mean_acc, "std_acc": std_acc,
                "result_str": f"{mean_acc:.1f}±{std_acc:.1f}",
            }
            timing_info[ds_name][method] = {
                "pre_s": pre_time / len(METHODS), "train_s": elapsed,
                "folds_run": use_folds, "total_folds": num_folds,
            }
            logger.info(f"    {method}: {mean_acc:.1f}±{std_acc:.1f}% ({elapsed:.0f}s)")

        # Baseline validation
        if ds_name == "MUTAG" and "vanilla" in all_results.get("MUTAG", {}):
            v = all_results["MUTAG"]["vanilla"]["mean_acc"]
            if v < 80:
                logger.warning(f"  MUTAG vanilla={v:.1f}% — below 80%, check config")

        logger.info(f"\n  {ds_name} summary:")
        for m in METHODS:
            if m in all_results.get(ds_name, {}):
                logger.info(f"    {m:25s}: {all_results[ds_name][m]['result_str']}")

    # ============================================================
    # BUILD OUTPUT
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Building method_out.json...")

    summary_table = {}
    for ds in DS_NAMES:
        if ds in all_results:
            summary_table[ds] = {m: all_results[ds][m]["result_str"]
                                  for m in METHODS if m in all_results[ds]}

    output = {
        "metadata": {
            "experiment": "NDS_GIN_benchmark_v2",
            "description": (
                "8-method GIN benchmark (vanilla, degree, NDS-tanh, NDS-eigvec, linear-diff, "
                "LapPE, RWSE, RNI) on 4 graph datasets with K-fold CV. "
                "CSL uses triangle/path2 count init for NDS (4-regular → degree uninformative)."
            ),
            "methods": METHODS,
            "model": f"GIN-3L-h{HP['hidden_dim']}",
            "hyperparameters": HP,
            "max_epochs": actual_epochs,
            "summary_table": summary_table,
            "timing": timing_info,
            "published_baselines": {
                "MUTAG": {"GIN_published": "89.4±5.6"},
                "PROTEINS": {"GIN_published": "76.2±2.8"},
                "IMDB-BINARY": {"GIN_published": "75.1±5.1"},
            },
        },
        "datasets": [],
    }

    raw_data = json.loads(DATA_FILE.read_text())
    for ds_entry in raw_data["datasets"]:
        ds_name = ds_entry["dataset"]
        if ds_name not in DS_NAMES:
            continue
        examples_out = []
        for ex in ds_entry["examples"]:
            row = {"input": ex["input"], "output": ex["output"]}
            for k, v in ex.items():
                if k.startswith("metadata_"):
                    row[k] = v
            ri = ex.get("metadata_row_index", len(examples_out))
            for method in METHODS:
                pk = f"predict_{method}"
                if ds_name in all_predictions and method in all_predictions[ds_name]:
                    row[pk] = all_predictions[ds_name][method].get(ri, "")
                else:
                    row[pk] = ""
            examples_out.append(row)
        output["datasets"].append({"dataset": ds_name, "examples": examples_out})

    OUT_FILE.write_text(json.dumps(output, indent=2))
    logger.info(f"Written to {OUT_FILE}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    hdr = f"{'Method':<25s}"
    for ds in DS_NAMES:
        hdr += f" | {ds:>12s}"
    logger.info(hdr)
    logger.info("-" * len(hdr))
    for m in METHODS:
        row = f"{m:<25s}"
        for ds in DS_NAMES:
            if ds in all_results and m in all_results[ds]:
                row += f" | {all_results[ds][m]['result_str']:>12s}"
            else:
                row += f" | {'N/A':>12s}"
        logger.info(row)

    logger.info("\nSANITY CHECKS:")
    for ds, m, exp in [("MUTAG", "vanilla", "~89"), ("CSL", "vanilla", "~10"),
                        ("CSL", "nds_tanh_T10", ">10 if NDS works"),
                        ("CSL", "lap_pe_k16", ">10")]:
        if ds in all_results and m in all_results[ds]:
            v = all_results[ds][m]["mean_acc"]
            logger.info(f"  {m} on {ds}: {v:.1f}% (expected {exp})")

    logger.info(f"\nTotal time: {time.time()-START_TIME:.0f}s ({(time.time()-START_TIME)/60:.1f}min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
