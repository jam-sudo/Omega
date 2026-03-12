#!/usr/bin/env python3
"""Retrain the differentiable ODE surrogate with more data.

Steps:
  1. Generate N PBPK ODE profiles via LHS sampling (CPU-parallel)
  2. Train surrogate (Linear MLP, log1p space)
  3. Validate against 20 benchmark drugs
  4. Save model + report

The original surrogate was trained on ~450 profiles → AAFE 7.24 on real drugs.
Goal: 10K+ profiles → AAFE < 1.5 on real drugs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("retrain_surrogate")


def generate_data(n_samples: int, n_workers: int, seed: int) -> tuple:
    """Generate PBPK ODE training data."""
    from omega_pbpk.ml.data.synthetic import generate_pbpk_data

    logger.info("Generating %d PBPK ODE profiles with %d workers...", n_samples, n_workers)
    ds = generate_pbpk_data(
        n_samples=n_samples,
        seed=seed,
        n_workers=n_workers,
    )
    logger.info(
        "Generated %d valid profiles (%.1f%% success rate)",
        ds.n_samples,
        100.0 * ds.n_samples / n_samples,
    )
    return ds.params, ds.curves, ds


def train(
    params: np.ndarray,
    curves: np.ndarray,
    n_epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    save_dir: str,
) -> dict:
    """Train surrogate with early stopping."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from omega_pbpk.ml.models.surrogate.differentiable_ode import DifferentiableODESurrogate

    n_samples = params.shape[0]
    n_input = params.shape[1]
    n_output = curves.shape[1]

    # Train/val split (90/10)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_samples)
    n_val = max(1, int(n_samples * 0.1))
    train_idx = indices[: n_samples - n_val]
    val_idx = indices[n_samples - n_val :]

    X_train = torch.tensor(params[train_idx], dtype=torch.float32)
    X_val = torch.tensor(params[val_idx], dtype=torch.float32)
    y_train = torch.log1p(torch.clamp(torch.tensor(curves[train_idx], dtype=torch.float32), min=0))
    y_val = torch.log1p(torch.clamp(torch.tensor(curves[val_idx], dtype=torch.float32), min=0))

    # Model — wider network for more data
    hidden_dim = 512 if n_samples >= 5000 else 256
    model = DifferentiableODESurrogate(n_input=n_input, n_output=n_output, hidden_dim=hidden_dim)
    model.param_mean = X_train.mean(dim=0)
    model.param_std = X_train.std(dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 100
    )
    mse_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    log_space_weight = 0.5

    logger.info(
        "Training: %d samples, hidden=%d, device=%s, epochs=%d, patience=%d",
        n_samples,
        hidden_dim,
        device,
        n_epochs,
        patience,
    )

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)

            # Combined loss: log1p MSE + linear MSE
            loss_log = mse_fn(y_pred, y_batch)
            y_pred_lin = torch.expm1(torch.clamp(y_pred, max=20.0))
            y_true_lin = torch.expm1(y_batch)
            loss_lin = mse_fn(y_pred_lin, y_true_lin)
            loss = log_space_weight * loss_log + (1.0 - log_space_weight) * loss_lin

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            y_vp = model(X_val)
            val_loss = mse_fn(y_vp, y_val).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 20 == 0 or epochs_no_improve == patience:
            logger.info(
                "Epoch %d/%d: train=%.6f val=%.6f best=%.6f stale=%d lr=%.2e",
                epoch + 1,
                n_epochs,
                avg_train,
                val_loss,
                best_val_loss,
                epochs_no_improve,
                optimizer.param_groups[0]["lr"],
            )

        if patience > 0 and epochs_no_improve >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()
    model.eval()

    # Compute AAFE on validation set
    with torch.no_grad():
        y_vp_log = model(X_val.cpu())
        y_vp_lin = torch.expm1(torch.clamp(y_vp_log, max=20.0)).clamp(min=0).numpy()

    true_np = curves[val_idx]
    eps = 1e-12
    pred_cmax = np.maximum(y_vp_lin.max(axis=1), eps)
    true_cmax = np.maximum(true_np.max(axis=1), eps)
    fold_errors = np.maximum(pred_cmax / true_cmax, true_cmax / pred_cmax)
    aafe = float(np.exp(np.mean(np.log(fold_errors))))
    pct_2fold = float(np.mean(fold_errors <= 2.0) * 100)

    logger.info("Validation AAFE (Cmax): %.3f, %%2-fold: %.1f%%", aafe, pct_2fold)

    # Save
    import torch as torch_save

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model_file = save_path / "surrogate_model.pt"
    torch_save.save(
        {
            "model_state_dict": model.state_dict(),
            "n_input": n_input,
            "n_output": n_output,
            "hidden_dim": hidden_dim,
            "param_mean": model.param_mean,
            "param_std": model.param_std,
            "aafe": aafe,
            "pct_2fold": pct_2fold,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        },
        str(model_file),
    )
    logger.info("Saved model to %s", model_file)

    return {"aafe": aafe, "pct_2fold": pct_2fold, "model": model, "hidden_dim": hidden_dim}


def validate_against_benchmark(model_path: str) -> dict:
    """Run validate_surrogate logic against 20 benchmark drugs."""
    import torch

    from omega_pbpk.ml.models.surrogate.differentiable_ode import load_surrogate

    benchmark_file = Path("outputs/l1_benchmark_results.json")
    if not benchmark_file.exists():
        logger.warning("No benchmark file found, skipping validation")
        return {}

    with open(benchmark_file) as f:
        data = json.load(f)

    drugs = data["per_drug"]
    surrogate = load_surrogate(model_path)

    IVIVE = 40.0 * 45.0 * 1800.0 / 1e6 / 60.0

    cmax_fes = []
    auc_fes = []
    corrs = []

    for drug in drugs:
        adme = drug["adme"]
        clint_L_h = (adme.get("clint_3a4", 0.1) + adme.get("clint_2d6", 0.0)) * IVIVE
        params_6d = np.array(
            [adme["logP"], adme["fup"], clint_L_h, adme["mw"], adme["rbp"], adme.get("peff", 1.0)],
            dtype=np.float32,
        )

        surr_curve = surrogate.predict(params_6d)

        # Run real ODE
        try:
            from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

            pipeline = OmegaPipeline()
            req = SimulationRequest(
                smiles=drug["smiles"], dose_mg=drug["dose_mg"], route="oral", duration_h=24.0
            )
            result = pipeline.simulate(req)
            ode_times = np.array(result.time_h)
            ode_cp = np.array(result.cp_mg_L)
        except Exception:
            continue

        # Compare
        surr_times = np.arange(len(surr_curve)) * 0.1
        ode_interp = np.interp(surr_times, ode_times, ode_cp)

        eps = 1e-12
        surr_cmax = max(float(surr_curve.max()), eps)
        ode_cmax = max(float(ode_interp.max()), eps)
        cmax_fe = max(surr_cmax / ode_cmax, ode_cmax / surr_cmax)
        cmax_fes.append(cmax_fe)

        surr_auc = max(float(np.trapz(surr_curve, surr_times)), eps)
        ode_auc = max(float(np.trapz(ode_interp, surr_times)), eps)
        auc_fe = max(surr_auc / ode_auc, ode_auc / surr_auc)
        auc_fes.append(auc_fe)

        if np.std(surr_curve) > eps and np.std(ode_interp) > eps:
            corrs.append(float(np.corrcoef(surr_curve, ode_interp)[0, 1]))

    if not cmax_fes:
        return {}

    aafe_cmax = float(np.exp(np.mean(np.log(cmax_fes))))
    aafe_auc = float(np.exp(np.mean(np.log(auc_fes))))
    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    pct_2fold = float(np.mean(np.array(cmax_fes) <= 2.0) * 100)

    logger.info("=" * 60)
    logger.info("BENCHMARK VALIDATION (%d drugs)", len(cmax_fes))
    logger.info("  Cmax AAFE:          %.3f", aafe_cmax)
    logger.info("  AUC  AAFE:          %.3f", aafe_auc)
    logger.info("  Cmax %%2-fold:       %.0f%%", pct_2fold)
    logger.info("  Mean correlation:   %.3f", mean_corr)
    logger.info("=" * 60)

    if aafe_cmax < 1.5 and mean_corr > 0.9:
        logger.info("VERDICT: ACCURATE — surrogate is ready for L2 training")
    elif aafe_cmax < 2.0 and mean_corr > 0.7:
        logger.info("VERDICT: ACCEPTABLE — minor error source")
    else:
        logger.info("VERDICT: INACCURATE — need more data or architecture changes")

    return {
        "aafe_cmax": aafe_cmax,
        "aafe_auc": aafe_auc,
        "pct_2fold_cmax": pct_2fold,
        "mean_correlation": mean_corr,
        "n_drugs": len(cmax_fes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain ODE surrogate with more data")
    parser.add_argument("--n-samples", type=int, default=10000, help="ODE profiles to generate")
    parser.add_argument("--n-workers", type=int, default=None, help="CPU workers (default: all)")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/pbpk_surrogate/6param",
        help="Model save directory",
    )
    parser.add_argument(
        "--skip-generate", action="store_true", help="Skip data generation, use existing HDF5"
    )
    parser.add_argument(
        "--data-file", type=str, default="data/ml/pbpk_curves.h5", help="HDF5 data file"
    )
    parser.add_argument("--skip-validate", action="store_true", help="Skip benchmark validation")
    args = parser.parse_args()

    import os

    n_workers = args.n_workers or os.cpu_count() or 1

    # Step 1: Generate or load data
    if args.skip_generate:
        from omega_pbpk.ml.data.synthetic import CurveDataset

        logger.info("Loading existing data from %s", args.data_file)
        ds = CurveDataset.load_hdf5(args.data_file)
        params, curves = ds.params, ds.curves
    else:
        params, curves, ds = generate_data(args.n_samples, n_workers, args.seed)
        # Save HDF5 for reuse
        data_path = Path(args.data_file)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_hdf5(data_path)
        logger.info("Saved training data to %s", data_path)

    # Step 2: Train
    result = train(
        params=params,
        curves=curves,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        save_dir=args.save_dir,
    )

    # Step 3: Validate against benchmark drugs
    if not args.skip_validate:
        model_path = str(Path(args.save_dir) / "surrogate_model.pt")
        bench = validate_against_benchmark(model_path)
        if bench:
            # Save combined results
            output = {**result, **bench}
            output.pop("model", None)
            output_path = Path("outputs/surrogate_retrain_results.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
