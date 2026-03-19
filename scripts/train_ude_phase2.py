#!/usr/bin/env python3
"""Train UDE Phase 2: MLP encoder → ADME → differentiable PBPK ODE → Cmax.

Multi-task: Cmax (MMPK 1,098) + CLint (TDC ~1,213) + fup (TDC ~1,614).
Quality-weighted loss, scaffold-split validation, early stopping.

Usage:
    python scripts/train_ude_phase2.py
    python scripts/train_ude_phase2.py --epochs 80 --lr 3e-4 --batch-size 16
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
from omega_pbpk.ml.models.ude.model_phase2 import UDEPhase2Model  # noqa: E402


def compute_aafe(log_errors: torch.Tensor) -> float:
    return float(10 ** log_errors.abs().mean())


def to_tensors(data_list, keys, device):
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
    epochs: int = 80,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    lambda_clint: float = 0.05,
    lambda_fup: float = 0.05,
    batch_size: int = 16,
    patience: int = 20,
    seed: int = 42,
):
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
    print(f"  MMPK: {len(mmpk_data)}, CLint: {len(clint_data)}, fup: {len(fup_data)}")
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    train_mmpk, val_mmpk = scaffold_split(mmpk_data, val_frac=0.2, seed=seed)
    print(f"  Split: {len(train_mmpk)} train / {len(val_mmpk)} val")

    train_t = to_tensors(train_mmpk, ["features", "dose_mg", "cmax_obs", "quality_weight"], device)
    val_t = to_tensors(val_mmpk, ["features", "dose_mg", "cmax_obs", "quality_weight"], device)
    clint_t = to_tensors(clint_data, ["features", "clint"], device) if clint_data else None
    fup_t = to_tensors(fup_data, ["features", "fup"], device) if fup_data else None

    # --- Model ---
    model = UDEPhase2Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPhase 2 model: {n_params:,} parameters")
    print("ODE: 13-state PBPK via torchdiffeq (dopri5)")

    # --- Training ---
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    n_train = len(train_mmpk)
    n_ode_fails = 0

    print(f"\nTraining for up to {epochs} epochs (batch={batch_size}, patience={patience})")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        epoch_count = 0
        epoch_fails = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            x = train_t["features"][idx]
            dose = train_t["dose_mg"][idx]
            cmax_obs = train_t["cmax_obs"][idx]
            weights = train_t["quality_weight"][idx]

            try:
                cmax_pred, _ = model.predict_cmax(x, dose)

                # Primary: Cmax MSE in log10 space
                log_err = torch.log10(cmax_pred.clamp(min=1e-10)) - torch.log10(
                    cmax_obs.clamp(min=1e-10)
                )
                loss_cmax = (weights * log_err**2).mean()
                loss = loss_cmax

                # Auxiliary: CLint
                if clint_t is not None and lambda_clint > 0:
                    n_aux = min(batch_size, len(clint_data))
                    aux_idx = torch.randint(0, len(clint_data), (n_aux,), device=device)
                    cp = model.predict_clint(clint_t["features"][aux_idx])
                    co = clint_t["clint"][aux_idx]
                    loss = (
                        loss
                        + lambda_clint
                        * (
                            (torch.log10(cp.clamp(min=1e-6)) - torch.log10(co.clamp(min=1e-6))) ** 2
                        ).mean()
                    )

                # Auxiliary: fup
                if fup_t is not None and lambda_fup > 0:
                    n_aux = min(batch_size, len(fup_data))
                    aux_idx = torch.randint(0, len(fup_data), (n_aux,), device=device)
                    fp = model.predict_fup(fup_t["features"][aux_idx])
                    fo = fup_t["fup"][aux_idx]
                    loss = (
                        loss
                        + lambda_fup
                        * (
                            (torch.log10(fp.clamp(min=1e-6)) - torch.log10(fo.clamp(min=1e-6))) ** 2
                        ).mean()
                    )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss_cmax.item() * (end - start)
                epoch_count += end - start

            except RuntimeError as e:
                if "underflow" in str(e).lower() or "nan" in str(e).lower():
                    epoch_fails += 1
                    n_ode_fails += 1
                    continue
                raise

        scheduler.step()

        if epoch_count == 0:
            print(f"  Epoch {epoch}: ALL BATCHES FAILED — ODE divergence")
            continue

        epoch_loss /= epoch_count

        # --- Validation ---
        model.eval()
        val_log_errs = []

        with torch.no_grad():
            # Process val in small batches (ODE memory)
            for vs in range(0, len(val_mmpk), batch_size):
                ve = min(vs + batch_size, len(val_mmpk))
                try:
                    vp, _ = model.predict_cmax(val_t["features"][vs:ve], val_t["dose_mg"][vs:ve])
                    vle = torch.log10(vp.clamp(min=1e-10)) - torch.log10(
                        val_t["cmax_obs"][vs:ve].clamp(min=1e-10)
                    )
                    val_log_errs.append(vle)
                except RuntimeError:
                    continue

        if not val_log_errs:
            print(f"  Epoch {epoch}: Val failed — ODE divergence")
            continue

        val_all_errs = torch.cat(val_log_errs)
        val_loss = (val_all_errs**2).mean().item()
        val_aafe = compute_aafe(val_all_errs)

        if epoch % 5 == 0 or epoch <= 3 or epoch == epochs:
            print(
                f"  Epoch {epoch:3d}: "
                f"train={epoch_loss:.4f}  "
                f"val={val_loss:.4f}  "
                f"AAFE={val_aafe:.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.1e}"
                + (f"  ({epoch_fails} ODE fails)" if epoch_fails else "")
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
    if best_state is None:
        print("ERROR: No valid epoch — training failed")
        sys.exit(1)

    model.load_state_dict(best_state)

    model_dir = repo_root / "models" / "ude"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_dir / "phase2_pbpk.pt")

    meta = {
        "phase": 2,
        "model": "13-state PBPK ODE via torchdiffeq",
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "n_train": len(train_mmpk),
        "n_val": len(val_mmpk),
        "n_params": n_params,
        "n_ode_fails": n_ode_fails,
        "hyperparams": {
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "lambda_clint": lambda_clint,
            "lambda_fup": lambda_fup,
            "batch_size": batch_size,
            "patience": patience,
        },
    }
    with open(model_dir / "meta_phase2.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Phase 2 training complete!")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  ODE failures: {n_ode_fails}")
    print(f"  Model: {model_dir / 'phase2_pbpk.pt'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UDE Phase 2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-clint", type=float, default=0.05)
    parser.add_argument("--lambda-fup", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(**vars(args))
