#!/usr/bin/env python3
"""Train UDE multi-task PK model (Phase 1: 1-compartment analytical).

Multi-task: Cmax (MMPK 1,098) + CLint (TDC ~1,213) + fup (TDC ~1,614).
Quality-weighted loss, scaffold-split validation, early stopping.

Usage:
    python scripts/train_ude.py
    python scripts/train_ude.py --epochs 200 --lr 5e-4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.ml.models.ude.data import (  # noqa: E402
    load_mmpk_training,
    load_tdc_clint,
    load_tdc_fup,
    scaffold_split,
)
from omega_pbpk.ml.models.ude.model import MultiTaskPKModel  # noqa: E402


def compute_aafe(log_errors: torch.Tensor) -> float:
    """AAFE from log10 prediction errors."""
    return float(10 ** log_errors.abs().mean())


def to_tensors(data_list: list[dict], keys: list[str], device: torch.device) -> dict:
    """Convert list of dicts to dict of tensors."""
    result = {}
    for key in keys:
        if key == "features":
            result[key] = torch.tensor(
                np.stack([d[key] for d in data_list]), dtype=torch.float32
            ).to(device)
        else:
            result[key] = torch.tensor([d[key] for d in data_list], dtype=torch.float32).to(device)
    return result


def train(
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_clint: float = 0.1,
    lambda_fup: float = 0.1,
    batch_size: int = 64,
    patience: int = 15,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("Loading data...")
    t0 = time.perf_counter()

    mmpk_data = load_mmpk_training()
    clint_data = load_tdc_clint()
    fup_data = load_tdc_fup()

    print(f"  MMPK Cmax: {len(mmpk_data)} drugs")
    print(f"  TDC CLint: {len(clint_data)} drugs")
    print(f"  TDC fup:   {len(fup_data)} drugs")
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Scaffold split MMPK
    train_mmpk, val_mmpk = scaffold_split(mmpk_data, val_frac=0.2, seed=seed)
    print(f"  MMPK split: {len(train_mmpk)} train / {len(val_mmpk)} val")

    # Tensors
    train_t = to_tensors(train_mmpk, ["features", "dose_mg", "cmax_obs", "quality_weight"], device)
    val_t = to_tensors(val_mmpk, ["features", "dose_mg", "cmax_obs", "quality_weight"], device)
    clint_t = to_tensors(clint_data, ["features", "clint"], device) if clint_data else None
    fup_t = to_tensors(fup_data, ["features", "fup"], device) if fup_data else None

    # --- Model ---
    model = MultiTaskPKModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} parameters")

    # --- Training loop ---
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    n_train = len(train_mmpk)

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_cmax_loss = 0.0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            x = train_t["features"][idx]
            dose = train_t["dose_mg"][idx]
            cmax_obs = train_t["cmax_obs"][idx]
            weights = train_t["quality_weight"][idx]

            # Primary: Cmax MSE in log10 space
            cmax_pred, _ = model.predict_cmax(x, dose)
            log_err = torch.log10(cmax_pred.clamp(min=1e-10)) - torch.log10(
                cmax_obs.clamp(min=1e-10)
            )
            loss_cmax = (weights * log_err**2).mean()
            loss = loss_cmax

            # Auxiliary: CLint
            if clint_t is not None and lambda_clint > 0:
                n_aux = min(batch_size, len(clint_data))
                aux_idx = torch.randint(0, len(clint_data), (n_aux,), device=device)
                clint_pred = model.predict_clint(clint_t["features"][aux_idx])
                clint_obs = clint_t["clint"][aux_idx]
                loss_clint = (
                    (
                        torch.log10(clint_pred.clamp(min=1e-6))
                        - torch.log10(clint_obs.clamp(min=1e-6))
                    )
                    ** 2
                ).mean()
                loss = loss + lambda_clint * loss_clint

            # Auxiliary: fup
            if fup_t is not None and lambda_fup > 0:
                n_aux = min(batch_size, len(fup_data))
                aux_idx = torch.randint(0, len(fup_data), (n_aux,), device=device)
                fup_pred = model.predict_fup(fup_t["features"][aux_idx])
                fup_obs = fup_t["fup"][aux_idx]
                loss_fup = (
                    (torch.log10(fup_pred.clamp(min=1e-6)) - torch.log10(fup_obs.clamp(min=1e-6)))
                    ** 2
                ).mean()
                loss = loss + lambda_fup * loss_fup

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_cmax_loss += loss_cmax.item() * (end - start)

        epoch_cmax_loss /= n_train

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_cmax_pred, _ = model.predict_cmax(val_t["features"], val_t["dose_mg"])
            val_log_err = torch.log10(val_cmax_pred.clamp(min=1e-10)) - torch.log10(
                val_t["cmax_obs"].clamp(min=1e-10)
            )
            val_loss = (val_t["quality_weight"] * val_log_err**2).mean().item()
            val_aafe = compute_aafe(val_log_err)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch <= 3 or epoch == epochs:
            print(
                f"  Epoch {epoch:3d}: "
                f"train_loss={epoch_cmax_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_AAFE={val_aafe:.3f}  "
                f"lr={current_lr:.1e}"
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(epoch_cmax_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_aafe": round(val_aafe, 4),
            }
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"\n  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    # --- Save ---
    model.load_state_dict(best_state)

    model_dir = repo_root / "models" / "ude"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_dir / "multitask_pk_phase1.pt")

    best_val_aafe = history[best_epoch - 1]["val_aafe"]

    meta = {
        "phase": 1,
        "model": "1-compartment analytical multi-task",
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_aafe": best_val_aafe,
        "n_train_mmpk": len(train_mmpk),
        "n_val_mmpk": len(val_mmpk),
        "n_tdc_clint": len(clint_data),
        "n_tdc_fup": len(fup_data),
        "n_params": n_params,
        "hyperparams": {
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "lambda_clint": lambda_clint,
            "lambda_fup": lambda_fup,
            "batch_size": batch_size,
            "patience": patience,
            "seed": seed,
        },
    }
    with open(model_dir / "meta_phase1.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val AAFE: {best_val_aafe:.3f}")
    print(f"  Model saved: {model_dir / 'multitask_pk_phase1.pt'}")
    print(f"{'=' * 70}")

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UDE Phase 1 model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-clint", type=float, default=0.1)
    parser.add_argument("--lambda-fup", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_clint=args.lambda_clint,
        lambda_fup=args.lambda_fup,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
