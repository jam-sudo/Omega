#!/usr/bin/env python3
"""L2 Transfer Learning: Morgan Fingerprint + MLP for PK parameter prediction.

Instead of training a GNN from scratch on 21K molecules (prone to mode collapse),
use pre-computed molecular fingerprints and train only a lightweight MLP head.

Morgan fingerprints (2048-bit ECFP4) capture molecular substructure information
and are competitive with GNNs on property prediction benchmarks, especially
with limited data (<50K molecules).

Usage:
    python scripts/train_l2_transfer.py
    python scripts/train_l2_transfer.py --epochs 200 --hidden 512
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("l2_transfer")

LABELS_CSV = repo_root / "data" / "ml" / "gnn_labels_large.csv"
MODEL_SAVE_PATH = repo_root / "models" / "level2" / "transfer_model.pt"
PARAM_NAMES = ["logP", "fup", "clint_L_h", "mw", "rbp", "peff"]
# Params to predict in log-space (positive, wide range)
LOG_PARAMS = {"clint_L_h", "mw", "peff"}
FP_RADIUS = 2  # ECFP4
FP_NBITS = 2048


# ---------------------------------------------------------------------------
# Fingerprint generation
# ---------------------------------------------------------------------------
def smiles_to_fingerprint(smiles: str) -> np.ndarray | None:
    """Convert SMILES to Morgan fingerprint (2048-bit ECFP4)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    return np.array(fp, dtype=np.float32)


def generate_fingerprints(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    """Generate fingerprints for all SMILES, return array + valid indices."""
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    return np.stack(fps), valid_idx


# ---------------------------------------------------------------------------
# MLP Model
# ---------------------------------------------------------------------------
class TransferMLP(nn.Module):
    """MLP: Morgan FP (2048) -> PK parameters (6).

    Architecture: 2048 -> hidden -> hidden/2 -> 6
    With batch norm, ReLU, dropout.
    """

    def __init__(
        self, n_input: int = FP_NBITS, n_output: int = 6, hidden: int = 512, dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_output),
        )
        self.param_names = PARAM_NAMES
        self.log_params_mask = torch.tensor([1.0 if p in LOG_PARAMS else 0.0 for p in PARAM_NAMES])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(csv_path: Path) -> tuple[list[str], np.ndarray]:
    """Load SMILES and labels from CSV."""
    smiles = []
    params = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles.append(row["smiles"])
            params.append([float(row[p]) for p in PARAM_NAMES])
    return smiles, np.array(params, dtype=np.float32)


def transform_labels(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Transform labels: log-transform wide-range params, then standardize."""
    transformed = labels.copy()
    stats = {}

    for i, name in enumerate(PARAM_NAMES):
        if name in LOG_PARAMS:
            transformed[:, i] = np.log1p(np.clip(labels[:, i], 0, None))

        mean = transformed[:, i].mean()
        std = transformed[:, i].std() + 1e-8
        transformed[:, i] = (transformed[:, i] - mean) / std
        stats[name] = {"mean": float(mean), "std": float(std), "log": name in LOG_PARAMS}

    return transformed, stats


def inverse_transform(predictions: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse transform predictions back to original scale."""
    result = predictions.copy()
    for i, name in enumerate(PARAM_NAMES):
        s = stats[name]
        result[:, i] = result[:, i] * s["std"] + s["mean"]
        if s["log"]:
            result[:, i] = np.expm1(result[:, i])
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def scaffold_split_simple(
    smiles_list: list[str], val_frac: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Simple random split (faster than scaffold split)."""
    rng = np.random.default_rng(seed)
    n = len(smiles_list)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    return list(idx[n_val:]), list(idx[:n_val])


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden: int = 512,
    lr: float = 1e-3,
    n_epochs: int = 200,
    batch_size: int = 256,
    patience: int = 30,
    dropout: float = 0.1,
) -> tuple[TransferMLP, float]:
    """Train the MLP model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_output = y_train.shape[1]

    model = TransferMLP(
        n_input=X_train.shape[1], n_output=n_output, hidden=hidden, dropout=dropout
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params, device=%s", n_params, device)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 100
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    stale = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = loss_fn(val_pred, y_v).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if (epoch + 1) % 20 == 0 or stale == patience:
            logger.info(
                "Epoch %3d/%d: train=%.6f val=%.6f best=%.6f stale=%d",
                epoch + 1,
                n_epochs,
                epoch_loss / n_batch,
                val_loss,
                best_val,
                stale,
            )

        if patience > 0 and stale >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    if best_state:
        model.load_state_dict(best_state)
    model.cpu().eval()
    return model, best_val


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
def save_model(
    model: TransferMLP, stats: dict, save_path: Path, n_train: int, val_loss: float
) -> None:
    """Save model + metadata."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_stats": stats,
            "param_names": PARAM_NAMES,
            "fp_radius": FP_RADIUS,
            "fp_nbits": FP_NBITS,
            "n_train": n_train,
            "val_loss": val_loss,
            "model_type": "transfer_mlp",
        },
        str(save_path),
    )
    logger.info("Saved to %s", save_path)


# ---------------------------------------------------------------------------
# Evaluation (quick inline)
# ---------------------------------------------------------------------------
def quick_eval(model: TransferMLP, stats: dict, device: str = "cpu") -> None:
    """Quick evaluation on benchmark drugs."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from eval_l2_model import BENCHMARK_DRUGS, fold_error, load_observed_pk

    logger.info("=== Quick Evaluation on 20 Benchmark Drugs ===")
    cmax_fes, auc_fes = [], []
    latencies = []

    for drug_name, info in BENCHMARK_DRUGS.items():
        smiles = info["smiles"]
        dose_mg = info["dose_mg"]

        t0 = time.perf_counter()

        # Predict params
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            logger.warning("Invalid SMILES: %s", drug_name)
            continue

        model.eval()
        with torch.no_grad():
            x = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)
            pred_norm = model(x).numpy()

        pred_params = inverse_transform(pred_norm, stats)[0]
        params = {name: float(pred_params[i]) for i, name in enumerate(PARAM_NAMES)}

        # Run real ODE
        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.drugs.drug import Drug

        drug = Drug(
            name=drug_name,
            mw=max(params["mw"], 50),
            logP=params["logP"],
            fup=max(min(params["fup"], 1.0), 0.001),
            rbp=max(params["rbp"], 0.3),
            peff=max(params["peff"], 0.01),
            clint_hepatic_L_per_h=max(params["clint_L_h"], 0.001),
            clint={"CYP3A4": 1.0},
            smiles=smiles,
            dose_mg=dose_mg,
            route="oral",
        )

        try:
            pbpk = WholeBodyPBPK(drug, body_weight=70.0)
            pbpk.setup_oral(dose_mg)
            sim = pbpk.simulate(t_end_h=24.0)
            pk = sim.pk_summary()
            pred_cmax = pk.get("Cmax_mg_L", 0)
            pred_auc = pk.get("AUC_mg_h_L", 0)
        except Exception as e:
            logger.warning("%s ODE failed: %s", drug_name, e)
            continue

        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        observed = load_observed_pk(drug_name)
        if observed:
            fe_c = fold_error(pred_cmax, observed["cmax"])
            fe_a = fold_error(pred_auc, observed["auc"])
            cmax_fes.append(fe_c)
            auc_fes.append(fe_a)
            mark = "✓" if fe_c <= 2.0 and fe_a <= 2.0 else "✗"
            logger.info(
                "%s: Cmax=%.4f (obs=%.4f, FE=%.1fx) AUC=%.4f (obs=%.4f, FE=%.1fx) %s [%dms]",
                drug_name,
                pred_cmax,
                observed["cmax"],
                fe_c,
                pred_auc,
                observed["auc"],
                fe_a,
                mark,
                latency,
            )

    if cmax_fes:
        aafe_c = float(10 ** np.mean(np.log10(cmax_fes)))
        aafe_a = float(10 ** np.mean(np.log10(auc_fes)))
        pct_c = sum(1 for fe in cmax_fes if fe <= 2.0) / len(cmax_fes) * 100
        pct_a = sum(1 for fe in auc_fes if fe <= 2.0) / len(auc_fes) * 100
        avg_lat = np.mean(latencies)

        logger.info("=" * 60)
        logger.info("TRANSFER MODEL RESULTS")
        logger.info("  Cmax: AAFE=%.2f, %%2-fold=%.0f%%", aafe_c, pct_c)
        logger.info("  AUC:  AAFE=%.2f, %%2-fold=%.0f%%", aafe_a, pct_a)
        logger.info("  Latency: mean=%.0fms", avg_lat)
        logger.info(
            "  L2 EXIT: AAFE<2.0=%s, %%2fold≥70%%=%s, speed<500ms=%s",
            "PASS" if aafe_c < 2.0 and aafe_a < 2.0 else "FAIL",
            "PASS" if pct_c >= 70 and pct_a >= 70 else "FAIL",
            "PASS" if avg_lat < 500 else "FAIL",
        )
        logger.info("  vs L1: Cmax AAFE 2.16→%.2f, AUC AAFE 1.66→%.2f", aafe_c, aafe_a)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    # Step 1: Load data
    logger.info("Loading labels from %s", LABELS_CSV)
    smiles_list, labels = load_data(LABELS_CSV)
    logger.info("Loaded %d compounds, %d params", len(smiles_list), labels.shape[1])

    # Step 2: Generate fingerprints
    logger.info("Generating Morgan fingerprints (radius=%d, bits=%d)...", FP_RADIUS, FP_NBITS)
    t0 = time.time()
    fps, valid_idx = generate_fingerprints(smiles_list)
    labels = labels[valid_idx]
    logger.info(
        "Generated %d FPs in %.1fs (%.0f/s)",
        len(fps),
        time.time() - t0,
        len(fps) / (time.time() - t0),
    )

    # Step 3: Transform labels
    transformed, stats = transform_labels(labels)

    # Step 4: Split
    train_idx, val_idx = scaffold_split_simple([smiles_list[i] for i in valid_idx], val_frac=0.2)
    X_train, y_train = fps[train_idx], transformed[train_idx]
    X_val, y_val = fps[val_idx], transformed[val_idx]
    logger.info("Split: %d train, %d val", len(train_idx), len(val_idx))

    # Step 5: Train
    logger.info(
        "=== Training MLP: hidden=%d, lr=%.1e, epochs=%d ===", args.hidden, args.lr, args.epochs
    )
    model, val_loss = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        hidden=args.hidden,
        lr=args.lr,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        dropout=args.dropout,
    )
    logger.info("Training done. Best val loss: %.6f", val_loss)

    # Step 6: Save
    save_model(model, stats, MODEL_SAVE_PATH, n_train=len(train_idx), val_loss=val_loss)

    # Step 7: Quick eval
    if not args.no_eval:
        quick_eval(model, stats)


if __name__ == "__main__":
    main()
