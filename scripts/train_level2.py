#!/usr/bin/env python3
"""Train the Level 2 end-to-end SMILES-to-PK model.

Usage:
    # Stage 1 only (1-cpt pre-training)
    python scripts/train_level2.py --stage1-data data/ml/1cpt_100k.h5

    # Full curriculum (stages 1 + 2)
    python scripts/train_level2.py \
        --stage1-data data/ml/1cpt_100k.h5 \
        --stage2-data data/ml/pbpk_50k.h5

    # Resume from checkpoint
    python scripts/train_level2.py --stage2-data data/ml/pbpk_50k.h5 --resume
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_level2")


def load_hdf5_as_tensordataset(path: str, val_fraction: float = 0.1) -> tuple:
    """Load HDF5 CurveDataset and split into train/val TensorDatasets.

    Returns (train_dataset, val_dataset, n_train, n_val).
    """
    from omega_pbpk.ml.data.synthetic import CurveDataset

    ds = CurveDataset.load_hdf5(path)
    logger.info(
        "Loaded %d samples from %s (params: %s, curves: %s)",
        ds.n_samples,
        path,
        ds.params.shape,
        ds.curves.shape,
    )

    n_val = max(1, int(ds.n_samples * val_fraction))
    n_train = ds.n_samples - n_val

    rng = np.random.default_rng(42)
    indices = rng.permutation(ds.n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    params_train = torch.tensor(ds.params[train_idx], dtype=torch.float32)
    curves_train = torch.tensor(ds.curves[train_idx], dtype=torch.float32)
    metrics_train = torch.tensor(ds.metrics[train_idx], dtype=torch.float32)

    params_val = torch.tensor(ds.params[val_idx], dtype=torch.float32)
    curves_val = torch.tensor(ds.curves[val_idx], dtype=torch.float32)
    metrics_val = torch.tensor(ds.metrics[val_idx], dtype=torch.float32)

    train_ds = TensorDataset(params_train, curves_train, metrics_train)
    val_ds = TensorDataset(params_val, curves_val, metrics_val)

    logger.info("Split: %d train, %d val", n_train, n_val)
    return train_ds, val_ds, n_train, n_val


def train_surrogate_first(data_path: str) -> None:
    """Pre-train the surrogate on the provided data if needed."""
    from pathlib import Path

    from omega_pbpk.ml.data.synthetic import CurveDataset
    from omega_pbpk.ml.models.surrogate.differentiable_ode import train_surrogate

    surrogate_path = Path("models/pbpk_surrogate/1cpt/surrogate_model.pt")
    if surrogate_path.exists():
        logger.info("Surrogate model already exists at %s, skipping", surrogate_path)
        return

    logger.info("Training surrogate on %s...", data_path)
    ds = CurveDataset.load_hdf5(data_path)
    result = train_surrogate(
        params=ds.params,
        curves=ds.curves,
        n_epochs=200,
        batch_size=512,
        lr=1e-3,
        val_fraction=0.1,
        save_dir="models/pbpk_surrogate/1cpt",
        verbose=True,
    )
    logger.info("Surrogate AAFE: %.3f", result["aafe"])


def main():
    parser = argparse.ArgumentParser(description="Train Level 2 SMILES-to-PK model")
    parser.add_argument("--stage1-data", type=str, help="Path to 1-cpt HDF5 data")
    parser.add_argument("--stage2-data", type=str, help="Path to PBPK HDF5 data")
    parser.add_argument("--stage3-data", type=str, help="Path to clinical HDF5 data")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="models/level2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--base-lr", type=float, default=1e-3)
    args = parser.parse_args()

    if not args.stage1_data and not args.stage2_data:
        logger.error("At least --stage1-data or --stage2-data must be provided")
        sys.exit(1)

    # Step 1: Ensure surrogate is trained
    if args.stage1_data:
        train_surrogate_first(args.stage1_data)

    # Step 2: Create end-to-end model
    from omega_pbpk.ml.models.foundation.end_to_end import SMILESToPKModel
    from omega_pbpk.ml.training.curriculum import MultiFidelityCurriculum

    model = SMILESToPKModel()

    # Load pre-trained surrogate weights into the model's surrogate
    from pathlib import Path

    from omega_pbpk.ml.models.surrogate.differentiable_ode import load_surrogate

    surrogate_path = Path("models/pbpk_surrogate/1cpt/surrogate_model.pt")
    if surrogate_path.exists():
        trained_surrogate = load_surrogate(surrogate_path)
        model.surrogate.load_state_dict(trained_surrogate.state_dict())
        logger.info("Loaded pre-trained surrogate into end-to-end model")

    # Step 3: Prepare data
    stage1_train, stage1_val = None, None
    stage2_train, stage2_val = None, None
    stage3_train, stage3_val = None, None

    if args.stage1_data:
        stage1_train, stage1_val, _, _ = load_hdf5_as_tensordataset(args.stage1_data)

    if args.stage2_data:
        stage2_train, stage2_val, _, _ = load_hdf5_as_tensordataset(args.stage2_data)

    if args.stage3_data:
        stage3_train, stage3_val, _, _ = load_hdf5_as_tensordataset(args.stage3_data)

    # Step 4: Resume if requested
    if args.resume:
        final_path = Path(args.checkpoint_dir) / "final.pt"
        if final_path.exists():
            state = torch.load(str(final_path), map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            logger.info("Resumed from %s", final_path)

    # Step 5: Run curriculum
    curriculum = MultiFidelityCurriculum(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        base_lr=args.base_lr,
    )

    result = curriculum.run_curriculum(
        stage1_train=stage1_train,
        stage1_val=stage1_val,
        stage2_train=stage2_train,
        stage2_val=stage2_val,
        stage3_train=stage3_train,
        stage3_val=stage3_val,
    )

    # Step 6: Report
    logger.info("=== Curriculum Training Complete ===")
    logger.info("Stages completed: %s", result.stages_completed)
    for stage_name, history in result.stage_histories.items():
        logger.info(
            "  %s: best_val=%.6f @ epoch %d (%.1fs)",
            stage_name,
            history.best_val_loss,
            history.best_epoch + 1,
            history.elapsed_seconds,
        )
    if result.final_metrics:
        logger.info("Final metrics: %s", result.final_metrics)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d total params, %d trainable", total_params, trainable)


if __name__ == "__main__":
    main()
