#!/usr/bin/env python3
"""Train L2 GNN encoder + param_head end-to-end.

Pipeline:
  Step 1 – Generate SMILES→ADME labels using EnsembleADMEPredictor
  Step 2 – Train GNN encoder + param_head supervised (MSE in log-space)
  Step 3 – Fine-tune end-to-end through the frozen 6-param surrogate
  Step 4 – Save checkpoint to models/level2/final.pt

Usage:
    python scripts/train_l2_gnn_encoder.py
    python scripts/train_l2_gnn_encoder.py --skip-data-gen  # if labels already cached
    python scripts/train_l2_gnn_encoder.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_l2_gnn")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IVIVE_FACTOR = 40.0 * 45.0 * 1800.0 / 1e6 / 60.0  # uL/min/pmol -> L/h (~0.054)
LABELS_CACHE = Path("data/ml/gnn_labels.csv")
SURROGATE_PATH = Path("models/pbpk_surrogate/6param/surrogate_model.pt")
CHECKPOINT_PATH = Path("models/level2/final.pt")

# 6-param order matching surrogate training data
PARAM_NAMES = ["logP", "fup", "clint_L_h", "mw", "rbp", "peff"]

# Log-space params: MSE in log-space for these
LOG_PARAMS = {"fup", "clint_L_h", "mw", "peff"}  # all positive, log-scale makes sense
# Linear params: logP (can be negative), rbp (bounded)
LINEAR_PARAMS = {"logP", "rbp"}


# ---------------------------------------------------------------------------
# Step 1: Generate SMILES → ADME labels
# ---------------------------------------------------------------------------


def collect_smiles() -> list[str]:
    """Collect unique drug-like SMILES from TDC data files."""
    smiles_set: set[str] = set()

    # clearance_hepatocyte_az.tab: columns ID, X (SMILES), Y
    clr_path = Path("data/clearance_hepatocyte_az.tab")
    if clr_path.exists():
        with open(clr_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                smi = row.get("X", "").strip().strip('"')
                if smi and len(smi) > 3:
                    smiles_set.add(smi)
        logger.info("clearance_hepatocyte: %d unique SMILES", len(smiles_set))

    # ppbr_az.tab: columns Drug_ID, Drug (SMILES), Y, Species
    ppbr_path = Path("data/ppbr_az.tab")
    if ppbr_path.exists():
        before = len(smiles_set)
        with open(ppbr_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                smi = row.get("Drug", "").strip().strip('"')
                if smi and len(smi) > 3:
                    smiles_set.add(smi)
        logger.info("ppbr_az: +%d new SMILES (total %d)", len(smiles_set) - before, len(smiles_set))

    smiles = sorted(smiles_set)
    logger.info("Total unique SMILES collected: %d", len(smiles))
    return smiles


def generate_labels(smiles_list: list[str], cache_path: Path, batch_size: int = 500) -> Path:
    """Run ADMET-AI batch prediction + XGBoost on SMILES list, cache as CSV.

    Uses ADMETModel.predict(list) for batch GPU inference (much faster than
    calling predict() one-at-a-time through EnsembleADMEPredictor).
    """
    from admet_ai import ADMETModel
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    from omega_pbpk.ml.models.adme.admet_ai_wrapper import (
        CLINT_2D6_NON_SUBSTRATE,
        CLINT_2D6_SUBSTRATE,
        HEPATOCYTE_TO_PMOL_CYP3A4,
    )
    from omega_pbpk.ml.models.adme.xgboost_adme import XGBoostRBPPredictor
    from omega_pbpk.ml.models.adme.xgboost_fup import XGBoostFupPredictor

    admet_model = ADMETModel()
    try:
        xgb_rbp = XGBoostRBPPredictor()
        logger.info("XGBoost RBP predictor loaded")
    except Exception as exc:
        logger.warning("XGBoost RBP unavailable (%s), using default rbp=1.0", exc)
        xgb_rbp = None
    try:
        xgb_fup = XGBoostFupPredictor()
        logger.info("XGBoost fup predictor loaded")
    except Exception as exc:
        logger.warning("XGBoost fup unavailable (%s), using ADMET-AI fup", exc)
        xgb_fup = None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["smiles", "logP", "fup", "clint_L_h", "mw", "rbp", "peff"]
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    with open(cache_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for batch_start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[batch_start : batch_start + batch_size]
            elapsed = time.time() - t0
            logger.info(
                "Batch %d/%d (smiles %d-%d)  ok=%d  fail=%d  elapsed=%.0fs",
                batch_start // batch_size + 1,
                (len(smiles_list) + batch_size - 1) // batch_size,
                batch_start,
                min(batch_start + batch_size, len(smiles_list)),
                n_ok,
                n_fail,
                elapsed,
            )

            # Filter to valid SMILES first
            valid = []
            for smi in batch:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid.append(smi)
                else:
                    n_fail += 1

            if not valid:
                continue

            try:
                df = admet_model.predict(smiles=valid)
            except Exception as exc:
                logger.warning("ADMET-AI batch failed: %s — skipping %d compounds", exc, len(valid))
                n_fail += len(valid)
                continue

            for i, smi in enumerate(valid):
                try:
                    row_data = df.iloc[i].to_dict() if hasattr(df, "iloc") else df

                    # MW from RDKit
                    mol = Chem.MolFromSmiles(smi)
                    mw = float(Descriptors.MolWt(mol))

                    # logP
                    logp = float(row_data.get("Lipophilicity", 2.0))

                    # fup from PPBR: fup = 1 - ppbr/100
                    ppbr = float(row_data.get("PPBR_AZ", row_data.get("PPBR", 80.0)))
                    fup_admet = max(0.001, min(1.0, 1.0 - ppbr / 100.0))

                    # Ensemble fup with XGBoost if available
                    if xgb_fup is not None:
                        try:
                            xgb_fup_val, _, _ = xgb_fup.predict_fup_with_interval(smi)
                            import math

                            fup = math.sqrt(max(fup_admet, 0.001) * max(xgb_fup_val, 0.001))
                        except Exception:
                            fup = fup_admet
                    else:
                        fup = fup_admet
                    fup = max(0.001, min(1.0, fup))

                    # clint_3a4: hepatocyte clearance -> pmol CYP3A4
                    clint_hep = float(
                        row_data.get(
                            "Clearance_Hepatocyte_AZ", row_data.get("Clearance_Hepatocyte", 10.0)
                        )
                    )
                    clint_3a4 = max(0.01, clint_hep / HEPATOCYTE_TO_PMOL_CYP3A4)

                    # clint_2d6: categorical from CYP2D6_Substrate
                    cyp2d6 = float(row_data.get("CYP2D6_Substrate", 0.5))
                    clint_2d6 = CLINT_2D6_SUBSTRATE if cyp2d6 >= 0.5 else CLINT_2D6_NON_SUBSTRATE

                    # clint_L_h via IVIVE
                    clint_L_h = (clint_3a4 + clint_2d6) * IVIVE_FACTOR
                    if clint_L_h <= 0:
                        n_fail += 1
                        continue

                    # rbp: XGBoost primary, default 1.0
                    if xgb_rbp is not None:
                        try:
                            rbp = xgb_rbp.predict_rbp(smi)
                        except Exception:
                            rbp = 1.0
                    else:
                        rbp = 1.0
                    rbp = max(0.5, min(3.0, rbp))

                    # peff: Caco2 * 100 (convert 10^-6 cm/s -> 10^-4 cm/s)
                    caco2 = float(row_data.get("Caco2_Wang", 0.03))
                    peff = max(0.01, caco2 * 100.0)

                    writer.writerow(
                        {
                            "smiles": smi,
                            "logP": round(logp, 3),
                            "fup": round(fup, 5),
                            "clint_L_h": round(clint_L_h, 5),
                            "mw": round(mw, 2),
                            "rbp": round(rbp, 3),
                            "peff": round(peff, 5),
                        }
                    )
                    n_ok += 1
                except Exception as exc:
                    logger.debug("Row processing failed for %s: %s", smi[:40], exc)
                    n_fail += 1

    elapsed = time.time() - t0
    logger.info("Label generation done: %d ok, %d failed (%.1fs total)", n_ok, n_fail, elapsed)
    return cache_path


def load_labels(cache_path: Path) -> tuple[list[str], np.ndarray]:
    """Load SMILES and 6D param labels from CSV cache."""
    smiles = []
    params = []
    with open(cache_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles.append(row["smiles"])
            params.append([float(row[p]) for p in PARAM_NAMES])
    logger.info("Loaded %d labeled compounds from %s", len(smiles), cache_path)
    return smiles, np.array(params, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MolecularDataset(Dataset):
    """Dataset of (molecular graph, 6D param vector) pairs."""

    def __init__(self, smiles: list[str], params: np.ndarray) -> None:
        from omega_pbpk.ml.features.graphs import smiles_to_graph

        self.graphs = []
        self.params = []
        n_fail = 0

        for smi, par in zip(smiles, params):
            g = smiles_to_graph(smi)
            if g is None:
                n_fail += 1
                continue
            self.graphs.append(g)
            self.params.append(par)

        self.params_tensor = torch.tensor(self.params, dtype=torch.float32)
        if n_fail:
            logger.warning("Skipped %d SMILES that failed graph conversion", n_fail)
        logger.info("MolecularDataset: %d valid samples", len(self.graphs))

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple:
        return self.graphs[idx], self.params_tensor[idx]


def collate_graphs(batch: list[tuple]) -> tuple:
    """Collate a list of (graph, params) into a batched graph + params tensor."""
    try:
        from torch_geometric.data import Batch

        graphs, params = zip(*batch)
        batched = Batch.from_data_list(list(graphs))
        params_t = torch.stack(params)
        return batched, params_t
    except ImportError:
        # Pure-PyTorch fallback: process individually
        graphs, params = zip(*batch)
        params_t = torch.stack(params)
        return list(graphs), params_t


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def param_loss(pred: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """MSE loss in log-space for positive params, linear for logP/rbp.

    Args:
        pred: dict from PKParameterHead.forward() — keys include clint_3a4, clint_2d6
        target: (batch, 6) ground-truth params in PARAM_NAMES order:
                [logP, fup, clint_L_h, mw, rbp, peff]

    Returns:
        Scalar loss tensor.
    """
    eps = 1e-6
    losses = []

    # PKParameterHead outputs clint_3a4 + clint_2d6 separately; combine via IVIVE
    pred_clint_L_h = (pred["clint_3a4"] + pred["clint_2d6"]) * IVIVE_FACTOR

    # Build pred tensor aligned to PARAM_NAMES
    pred_aligned = {
        "logP": pred["logP"],
        "fup": pred["fup"],
        "clint_L_h": pred_clint_L_h,
        "mw": pred["mw"],
        "rbp": pred["rbp"],
        "peff": pred["peff"],
    }

    for i, name in enumerate(PARAM_NAMES):
        p = pred_aligned[name]  # (batch,)
        t = target[:, i]  # (batch,)
        if name in LOG_PARAMS:
            losses.append(nn.functional.mse_loss(torch.log(p + eps), torch.log(t.abs() + eps)))
        else:
            losses.append(nn.functional.mse_loss(p, t))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Scaffold split (fallback to random if RDKit not available)
# ---------------------------------------------------------------------------


def scaffold_split(smiles: list[str], val_frac: float = 0.2) -> tuple[list[int], list[int]]:
    """Split indices by Bemis-Murcko scaffold. Falls back to random split."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffold_to_idx: dict[str, list[int]] = {}
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                sca = ""
            else:
                sca = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffold_to_idx.setdefault(sca, []).append(i)

        scaffold_groups = sorted(scaffold_to_idx.values(), key=len, reverse=True)
        n_val = int(len(smiles) * val_frac)
        val_idx, train_idx = [], []
        for group in scaffold_groups:
            if len(val_idx) < n_val:
                val_idx.extend(group)
            else:
                train_idx.extend(group)

        logger.info(
            "Scaffold split: %d train, %d val (%d unique scaffolds)",
            len(train_idx),
            len(val_idx),
            len(scaffold_groups),
        )
        return train_idx, val_idx
    except Exception as exc:
        logger.warning("Scaffold split failed (%s), using random split", exc)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(smiles))
        n_val = int(len(smiles) * val_frac)
        return idx[n_val:].tolist(), idx[:n_val].tolist()


# ---------------------------------------------------------------------------
# Step 2: Supervised GNN training
# ---------------------------------------------------------------------------


def train_supervised(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 100,
    lr: float = 3e-4,
) -> tuple:
    """Train GNN encoder + PKParameterHead supervised on ADME labels."""
    from omega_pbpk.ml.models.foundation.gnn_encoder import MolecularEncoder
    from omega_pbpk.ml.models.foundation.param_head import PKParameterHead

    encoder = MolecularEncoder().to(device)
    param_head = PKParameterHead(embedding_dim=encoder.embedding_dim).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(param_head.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float("inf")
    best_encoder_state = None
    best_head_state = None

    for epoch in range(n_epochs):
        # Train
        encoder.train()
        param_head.train()
        train_loss_sum = 0.0
        n_batches = 0

        for graphs, targets in train_loader:
            targets = targets.to(device)
            if isinstance(graphs, list):
                # Pure-PyTorch fallback: process one at a time, stack embeddings
                embeddings = []
                for g in graphs:
                    g.x = g.x.to(device)
                    g.edge_index = g.edge_index.to(device)
                    g.edge_attr = g.edge_attr.to(device)
                    emb = encoder(g)
                    embeddings.append(emb)
                embedding = torch.cat(embeddings, dim=0)
            else:
                graphs.x = graphs.x.to(device)
                graphs.edge_index = graphs.edge_index.to(device)
                graphs.edge_attr = graphs.edge_attr.to(device)
                if hasattr(graphs, "batch") and graphs.batch is not None:
                    graphs.batch = graphs.batch.to(device)
                embedding = encoder(graphs)

            pred_params = param_head(embedding)
            loss = param_loss(pred_params, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(param_head.parameters()), 1.0
            )
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        encoder.eval()
        param_head.eval()
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for graphs, targets in val_loader:
                targets = targets.to(device)
                if isinstance(graphs, list):
                    embeddings = []
                    for g in graphs:
                        g.x = g.x.to(device)
                        g.edge_index = g.edge_index.to(device)
                        g.edge_attr = g.edge_attr.to(device)
                        emb = encoder(g)
                        embeddings.append(emb)
                    embedding = torch.cat(embeddings, dim=0)
                else:
                    graphs.x = graphs.x.to(device)
                    graphs.edge_index = graphs.edge_index.to(device)
                    graphs.edge_attr = graphs.edge_attr.to(device)
                    if hasattr(graphs, "batch") and graphs.batch is not None:
                        graphs.batch = graphs.batch.to(device)
                    embedding = encoder(graphs)

                pred_params = param_head(embedding)
                loss = param_loss(pred_params, targets)
                val_loss_sum += loss.item()
                n_val_batches += 1

        avg_train = train_loss_sum / max(n_batches, 1)
        avg_val = val_loss_sum / max(n_val_batches, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_head_state = {k: v.cpu().clone() for k, v in param_head.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Epoch %d/%d  train=%.4f  val=%.4f  best=%.4f",
                epoch + 1,
                n_epochs,
                avg_train,
                avg_val,
                best_val_loss,
            )

    logger.info("Supervised training done. Best val loss: %.4f", best_val_loss)
    encoder.load_state_dict(best_encoder_state)
    param_head.load_state_dict(best_head_state)
    return encoder, param_head, best_val_loss


# ---------------------------------------------------------------------------
# Step 3: Fine-tune through frozen surrogate
# ---------------------------------------------------------------------------


def finetune_through_surrogate(
    encoder: nn.Module,
    param_head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 30,
    lr: float = 1e-4,
) -> tuple[nn.Module, nn.Module, float]:
    """Fine-tune GNN+param_head with curve-level loss through frozen surrogate."""
    from omega_pbpk.ml.models.surrogate.differentiable_ode import load_surrogate

    surrogate = load_surrogate(SURROGATE_PATH).to(device)
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    encoder = encoder.to(device)
    param_head = param_head.to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(param_head.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    mse = nn.MSELoss()

    best_val_loss = float("inf")
    best_encoder_state = None
    best_head_state = None

    def _build_surrogate_input(pred_params: dict) -> torch.Tensor:
        """Build 6D surrogate input from PKParameterHead output."""
        clint_total = pred_params["clint_3a4"] + pred_params["clint_2d6"]
        clint_L_h = clint_total * IVIVE_FACTOR
        return torch.stack(
            [
                pred_params["logP"],
                pred_params["fup"],
                clint_L_h,
                pred_params["mw"],
                pred_params["rbp"],
                pred_params["peff"],
            ],
            dim=-1,
        )

    def _forward_batch(graphs, device):
        if isinstance(graphs, list):
            embeddings = []
            for g in graphs:
                g.x = g.x.to(device)
                g.edge_index = g.edge_index.to(device)
                g.edge_attr = g.edge_attr.to(device)
                embeddings.append(encoder(g))
            return torch.cat(embeddings, dim=0)
        else:
            graphs.x = graphs.x.to(device)
            graphs.edge_index = graphs.edge_index.to(device)
            graphs.edge_attr = graphs.edge_attr.to(device)
            if hasattr(graphs, "batch") and graphs.batch is not None:
                graphs.batch = graphs.batch.to(device)
            return encoder(graphs)

    for epoch in range(n_epochs):
        encoder.train()
        param_head.train()
        train_loss_sum = 0.0
        n_batches = 0

        for graphs, target_params_6d in train_loader:
            target_params_6d = target_params_6d.to(device)

            # Target curves: run L1 labels through frozen surrogate
            with torch.no_grad():
                target_curves_log = surrogate(target_params_6d)  # (batch, 241) in log1p space

            # Predicted params → curves
            embedding = _forward_batch(graphs, device)
            pred_params = param_head(embedding)
            pred_params_6d = _build_surrogate_input(pred_params)
            pred_curves_log = surrogate(pred_params_6d)

            # Curve-level loss in log1p space (both are log1p outputs)
            loss = mse(pred_curves_log, target_curves_log)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(param_head.parameters()), 1.0
            )
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        encoder.eval()
        param_head.eval()
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for graphs, target_params_6d in val_loader:
                target_params_6d = target_params_6d.to(device)
                target_curves_log = surrogate(target_params_6d)

                embedding = _forward_batch(graphs, device)
                pred_params = param_head(embedding)
                pred_params_6d = _build_surrogate_input(pred_params)
                pred_curves_log = surrogate(pred_params_6d)

                loss = mse(pred_curves_log, target_curves_log)
                val_loss_sum += loss.item()
                n_val_batches += 1

        avg_train = train_loss_sum / max(n_batches, 1)
        avg_val = val_loss_sum / max(n_val_batches, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_head_state = {k: v.cpu().clone() for k, v in param_head.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            logger.info(
                "Finetune epoch %d/%d  train=%.6f  val=%.6f  best=%.6f",
                epoch + 1,
                n_epochs,
                avg_train,
                avg_val,
                best_val_loss,
            )

    logger.info("Surrogate fine-tuning done. Best val loss: %.6f", best_val_loss)
    encoder.load_state_dict(best_encoder_state)
    param_head.load_state_dict(best_head_state)
    return encoder, param_head, best_val_loss


# ---------------------------------------------------------------------------
# Step 4: Save checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(
    encoder: nn.Module,
    param_head: nn.Module,
    surrogate_path: Path,
    n_compounds: int,
    val_loss: float,
) -> None:
    """Save final checkpoint with all components and metadata."""
    from omega_pbpk.ml.models.surrogate.differentiable_ode import load_surrogate

    surrogate = load_surrogate(surrogate_path)
    surrogate_ck = torch.load(str(surrogate_path), map_location="cpu", weights_only=False)

    # Build combined state dict prefixed by component
    combined_state: dict = {}
    for k, v in encoder.state_dict().items():
        combined_state[f"encoder.{k}"] = v
    for k, v in param_head.state_dict().items():
        combined_state[f"param_head.{k}"] = v
    for k, v in surrogate.state_dict().items():
        combined_state[f"surrogate.{k}"] = v

    checkpoint = {
        "model_state_dict": combined_state,
        "embedding_dim": encoder.embedding_dim,
        "surrogate_n_output": surrogate_ck["n_output"],
        "surrogate_hidden": surrogate_ck["hidden_dim"],
        "dt_h": 0.1,
        "trained_components": ["encoder", "param_head", "surrogate"],
        "training_info": {
            "n_compounds": n_compounds,
            "val_loss": val_loss,
            "date": str(date.today()),
            "surrogate_aafe": surrogate_ck.get("aafe", None),
        },
    }

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(CHECKPOINT_PATH))
    logger.info("Saved checkpoint to %s", CHECKPOINT_PATH)
    logger.info("  trained_components: %s", checkpoint["trained_components"])
    logger.info("  n_compounds: %d, val_loss: %.6f", n_compounds, val_loss)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train L2 GNN encoder end-to-end")
    parser.add_argument(
        "--skip-data-gen",
        action="store_true",
        help="Skip label generation if cache already exists",
    )
    parser.add_argument("--device", default="auto", help="Device: auto/cpu/cuda")
    parser.add_argument("--supervised-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    if device.type == "cuda":
        logger.info(
            "GPU: %s (%.1f GB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # -----------------------------------------------------------------------
    # Step 1: Generate / load labels
    # -----------------------------------------------------------------------
    if args.skip_data_gen and LABELS_CACHE.exists():
        logger.info("Skipping data gen, loading from %s", LABELS_CACHE)
    else:
        smiles_list = collect_smiles()
        if not smiles_list:
            logger.error("No SMILES found. Check data/ directory.")
            sys.exit(1)
        logger.info("Generating ADME labels for %d compounds...", len(smiles_list))
        generate_labels(smiles_list, LABELS_CACHE)

    all_smiles, all_params = load_labels(LABELS_CACHE)
    if len(all_smiles) < 50:
        logger.error("Too few labeled compounds (%d). Aborting.", len(all_smiles))
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: Build datasets with scaffold split
    # -----------------------------------------------------------------------
    train_idx, val_idx = scaffold_split(all_smiles, val_frac=0.2)
    train_smiles = [all_smiles[i] for i in train_idx]
    train_params = all_params[train_idx]
    val_smiles = [all_smiles[i] for i in val_idx]
    val_params = all_params[val_idx]

    logger.info("Building molecular datasets (graph conversion)...")
    train_ds = MolecularDataset(train_smiles, train_params)
    val_ds = MolecularDataset(val_smiles, val_params)

    if len(train_ds) < 20:
        logger.error("Training set too small (%d). Aborting.", len(train_ds))
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0,
    )
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    # -----------------------------------------------------------------------
    # Step 3: Supervised training
    # -----------------------------------------------------------------------
    logger.info("=== Step 2: Supervised GNN training (%d epochs) ===", args.supervised_epochs)
    encoder, param_head, sup_val_loss = train_supervised(
        train_loader,
        val_loader,
        device=device,
        n_epochs=args.supervised_epochs,
        lr=args.lr,
    )
    logger.info("Supervised val loss: %.4f", sup_val_loss)

    # -----------------------------------------------------------------------
    # Step 4: Fine-tune through frozen surrogate
    # -----------------------------------------------------------------------
    if not SURROGATE_PATH.exists():
        logger.warning("Surrogate not found at %s — skipping fine-tune step", SURROGATE_PATH)
        final_val_loss = sup_val_loss
    else:
        logger.info("=== Step 3: Surrogate fine-tuning (%d epochs) ===", args.finetune_epochs)
        encoder, param_head, finetune_val_loss = finetune_through_surrogate(
            encoder,
            param_head,
            train_loader,
            val_loader,
            device=device,
            n_epochs=args.finetune_epochs,
            lr=args.finetune_lr,
        )
        logger.info("Fine-tune val loss: %.6f", finetune_val_loss)
        final_val_loss = finetune_val_loss

    # -----------------------------------------------------------------------
    # Step 5: Save checkpoint
    # -----------------------------------------------------------------------
    logger.info("=== Step 4: Saving checkpoint ===")
    save_checkpoint(
        encoder=encoder,
        param_head=param_head,
        surrogate_path=SURROGATE_PATH,
        n_compounds=len(train_ds) + len(val_ds),
        val_loss=final_val_loss,
    )

    logger.info("=== Training complete ===")


if __name__ == "__main__":
    main()
