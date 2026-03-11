#!/usr/bin/env python3
"""Scale-up L2 GNN encoder training: 10K-50K compounds, SMILES augmentation, Optuna sweep.

Pipeline:
  1. Download ~20K drug-like SMILES from ZINC15 (filter: MW 150-600, logP -2..5, RB ≤10)
  2. Merge with existing TDC labels (data/ml/gnn_labels.csv)
  3. Batch ADMET-AI labeling → data/ml/gnn_labels_large.csv
  4. SMILES augmentation ×4 per molecule on training set
  5. Train with batch_size=128, 300 epochs, cosine-annealing + warmup
  6. Surrogate fine-tuning (frozen)
  7. Optional Optuna sweep (--optuna)
  8. Save checkpoint to models/level2/final.pt

Usage:
    python scripts/train_l2_gnn_large.py
    python scripts/train_l2_gnn_large.py --skip-download  # reuse ZINC cache
    python scripts/train_l2_gnn_large.py --skip-labels    # reuse label cache
    python scripts/train_l2_gnn_large.py --optuna         # run HP sweep first
    python scripts/train_l2_gnn_large.py --zinc-count 50000 --augment 5
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
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
logger = logging.getLogger("train_l2_large")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IVIVE_FACTOR = 40.0 * 45.0 * 1800.0 / 1e6 / 60.0  # ~0.054 uL/min/pmol -> L/h

ZINC_CACHE = Path("data/ml/zinc_drug_like.smi")
LABELS_CACHE = Path("data/ml/gnn_labels_large.csv")
SURROGATE_PATH = Path("models/pbpk_surrogate/6param/surrogate_model.pt")
CHECKPOINT_PATH = Path("models/level2/final.pt")

PARAM_NAMES = ["logP", "fup", "clint_L_h", "mw", "rbp", "peff"]
LOG_PARAMS = {"fup", "clint_L_h", "mw", "peff"}

# Drug-likeness filters (applied to ZINC download)
MW_MIN, MW_MAX = 150.0, 600.0
LOGP_MIN, LOGP_MAX = -2.0, 5.0
MAX_ROT_BONDS = 10


# ---------------------------------------------------------------------------
# Step 1: Download ZINC drug-like SMILES (with BRICS fallback)
# ---------------------------------------------------------------------------


def _generate_brics_smiles(target_count: int, cache_path: Path) -> Path:
    """Generate drug-like SMILES via BRICS fragmentation + recombination as fallback."""
    from rdkit import Chem
    from rdkit.Chem import BRICS, Descriptors, rdMolDescriptors

    logger.info("BRICS fallback: generating %d drug-like SMILES from TDC data...", target_count)

    # Collect source molecules from TDC files
    source_mols = []
    for tab_file, smi_col in [
        (Path("data/clearance_hepatocyte_az.tab"), "X"),
        (Path("data/ppbr_az.tab"), "Drug"),
    ]:
        if not tab_file.exists():
            continue
        with open(tab_file) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                smi = row.get(smi_col, "").strip().strip('"')
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    source_mols.append(mol)

    logger.info("Source molecules for BRICS: %d", len(source_mols))

    # Fragment all molecules and collect unique fragments
    all_frags: set[str] = set()
    for mol in source_mols:
        try:
            frags = BRICS.BRICSDecompose(mol)
            all_frags.update(frags)
        except Exception:
            pass

    frag_list = [Chem.MolFromSmiles(f) for f in all_frags if Chem.MolFromSmiles(f) is not None]
    logger.info("Unique BRICS fragments: %d", len(frag_list))

    # Recombine fragments to make new molecules
    generated: list[str] = []
    seen: set[str] = set()
    rng = np.random.default_rng(42)
    attempts = 0
    max_attempts = target_count * 20

    while len(generated) < target_count and attempts < max_attempts:
        attempts += 1
        # Pick 2-4 random fragments
        n_frags = rng.integers(2, 5)
        if len(frag_list) < n_frags:
            break
        chosen = [frag_list[i] for i in rng.choice(len(frag_list), n_frags, replace=False)]
        try:
            products = list(BRICS.BRICSBuild(chosen))
            if not products:
                continue
            mol = products[rng.integers(len(products))]
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
            if not smi or smi in seen:
                continue

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            rb = rdMolDescriptors.CalcNumRotatableBonds(mol)

            if MW_MIN <= mw <= MW_MAX and LOGP_MIN <= logp <= LOGP_MAX and rb <= MAX_ROT_BONDS:
                seen.add(smi)
                generated.append(smi)
        except Exception:
            pass

    logger.info("BRICS generated %d drug-like SMILES (%d attempts)", len(generated), attempts)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for smi in generated:
            f.write(smi + "\n")
    return cache_path


def download_zinc(target_count: int, cache_path: Path) -> Path:
    """Download drug-like SMILES from ChEMBL API (paginated) with BRICS fallback.

    ChEMBL has 2M+ drug-like compounds accessible via REST with proper pagination.
    Falls back to BRICS generation from TDC source molecules if API fails.
    """
    import json
    import subprocess

    from rdkit import Chem

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    chunk = 1000
    base_url = (
        "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
        "?molecule_properties__mw_freebase__lte=600"
        "&molecule_properties__mw_freebase__gte=150"
        "&molecule_properties__alogp__gte=-2"
        "&molecule_properties__alogp__lte=5"
        f"&limit={chunk}"
    )

    accepted: list[str] = []
    seen: set[str] = set()
    offset = 0
    t0 = time.time()

    logger.info("Downloading %d drug-like SMILES from ChEMBL...", target_count)
    while len(accepted) < target_count:
        url = f"{base_url}&offset={offset}"
        result = subprocess.run(
            ["curl", "-sL", "--max-time", "30", url],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            logger.warning("ChEMBL request failed at offset %d (rc=%d)", offset, result.returncode)
            break

        try:
            data = json.loads(result.stdout)
        except Exception as exc:
            logger.warning("ChEMBL JSON parse error at offset %d: %s", offset, exc)
            break

        molecules = data.get("molecules", [])
        if not molecules:
            break

        for mol_data in molecules:
            structs = mol_data.get("molecule_structures") or {}
            smi = structs.get("canonical_smiles")
            if not smi or smi in seen:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol)
            if canon in seen:
                continue
            seen.add(canon)
            accepted.append(canon)
            if len(accepted) >= target_count:
                break

        elapsed = time.time() - t0
        logger.info("ChEMBL offset=%d: total=%d (%.0fs)", offset, len(accepted), elapsed)
        offset += chunk
        time.sleep(0.3)

    if len(accepted) == 0:
        logger.error("ChEMBL download failed — falling back to BRICS generation")
        return _generate_brics_smiles(target_count, cache_path)

    with open(cache_path, "w") as f:
        for smi in accepted:
            f.write(smi + "\n")

    logger.info("Saved %d ChEMBL SMILES to %s (%.0fs)", len(accepted), cache_path, time.time() - t0)
    return cache_path


def load_zinc_smiles(cache_path: Path) -> list[str]:
    """Load ZINC SMILES from cache file."""
    with open(cache_path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Step 2: Label generation (batch ADMET-AI)
# ---------------------------------------------------------------------------


def generate_labels_batch(
    smiles_list: list[str],
    cache_path: Path,
    existing_smiles: set[str] | None = None,
    batch_size: int = 500,
) -> Path:
    """Run batch ADMET-AI + XGBoost on SMILES, skip already-labeled ones."""
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
    except Exception as exc:
        logger.warning("XGBoost RBP unavailable: %s", exc)
        xgb_rbp = None
    try:
        xgb_fup = XGBoostFupPredictor()
    except Exception as exc:
        logger.warning("XGBoost fup unavailable: %s", exc)
        xgb_fup = None

    # Skip SMILES already labeled
    to_label = [s for s in smiles_list if existing_smiles is None or s not in existing_smiles]
    logger.info(
        "Labeling %d new SMILES (%d already cached)",
        len(to_label),
        len(smiles_list) - len(to_label),
    )

    header = ["smiles", "logP", "fup", "clint_L_h", "mw", "rbp", "peff"]
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    # Append to existing file
    file_mode = "a" if cache_path.exists() else "w"
    with open(cache_path, file_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if file_mode == "w":
            writer.writeheader()

        for batch_start in range(0, len(to_label), batch_size):
            batch = to_label[batch_start : batch_start + batch_size]
            elapsed = time.time() - t0
            logger.info(
                "Batch %d/%d  ok=%d  fail=%d  elapsed=%.0fs",
                batch_start // batch_size + 1,
                (len(to_label) + batch_size - 1) // batch_size,
                n_ok,
                n_fail,
                elapsed,
            )

            # Filter to valid SMILES
            valid = [s for s in batch if Chem.MolFromSmiles(s) is not None]
            n_fail += len(batch) - len(valid)
            if not valid:
                continue

            try:
                df = admet_model.predict(smiles=valid)
            except Exception as exc:
                logger.warning("ADMET-AI batch failed: %s — skipping %d", exc, len(valid))
                n_fail += len(valid)
                continue

            for i, smi in enumerate(valid):
                try:
                    row = df.iloc[i].to_dict()
                    mol = Chem.MolFromSmiles(smi)
                    mw = float(Descriptors.MolWt(mol))
                    logp = float(row.get("Lipophilicity", 2.0))
                    ppbr = float(row.get("PPBR_AZ", row.get("PPBR", 80.0)))
                    fup_admet = max(0.001, min(1.0, 1.0 - ppbr / 100.0))

                    if xgb_fup is not None:
                        try:
                            xf, _, _ = xgb_fup.predict_fup_with_interval(smi)
                            fup = math.sqrt(max(fup_admet, 0.001) * max(xf, 0.001))
                        except Exception:
                            fup = fup_admet
                    else:
                        fup = fup_admet
                    fup = max(0.001, min(1.0, fup))

                    clint_hep = float(
                        row.get("Clearance_Hepatocyte_AZ", row.get("Clearance_Hepatocyte", 10.0))
                    )
                    clint_3a4 = max(0.01, clint_hep / HEPATOCYTE_TO_PMOL_CYP3A4)
                    cyp2d6 = float(row.get("CYP2D6_Substrate", 0.5))
                    clint_2d6 = CLINT_2D6_SUBSTRATE if cyp2d6 >= 0.5 else CLINT_2D6_NON_SUBSTRATE
                    clint_L_h = (clint_3a4 + clint_2d6) * IVIVE_FACTOR
                    if clint_L_h <= 0:
                        n_fail += 1
                        continue

                    if xgb_rbp is not None:
                        try:
                            rbp = max(0.5, min(3.0, xgb_rbp.predict_rbp(smi)))
                        except Exception:
                            rbp = 1.0
                    else:
                        rbp = 1.0

                    caco2 = float(row.get("Caco2_Wang", 0.03))
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
                    logger.debug("Row failed for %s: %s", smi[:40], exc)
                    n_fail += 1

    elapsed = time.time() - t0
    logger.info("Labeling done: %d ok, %d failed (%.0fs)", n_ok, n_fail, elapsed)
    return cache_path


def load_labels(cache_path: Path) -> tuple[list[str], np.ndarray]:
    """Load SMILES and 6D param labels from CSV."""
    smiles, params = [], []
    with open(cache_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                smiles.append(row["smiles"])
                params.append([float(row[p]) for p in PARAM_NAMES])
            except (KeyError, ValueError, TypeError):
                continue
    logger.info("Loaded %d labeled compounds", len(smiles))
    return smiles, np.array(params, dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 3: SMILES augmentation
# ---------------------------------------------------------------------------


def augment_smiles(smi: str, n_variants: int = 4) -> list[str]:
    """Generate N random SMILES representations of the same molecule."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    variants: set[str] = set()
    for _ in range(n_variants * 3):  # over-sample and deduplicate
        rand_smi = Chem.MolToSmiles(mol, doRandom=True)
        if rand_smi and rand_smi != smi:
            variants.add(rand_smi)
        if len(variants) >= n_variants:
            break
    return list(variants)[:n_variants]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MolecularDataset(Dataset):
    """Dataset of (graph, 6D param vector) pairs with optional SMILES augmentation."""

    def __init__(self, smiles: list[str], params: np.ndarray, augment: int = 0) -> None:
        from omega_pbpk.ml.features.graphs import smiles_to_graph

        self.graphs = []
        self.params_list = []
        n_fail = 0
        n_aug = 0

        for smi, par in zip(smiles, params):
            g = smiles_to_graph(smi)
            if g is None:
                n_fail += 1
                continue
            self.graphs.append(g)
            self.params_list.append(par)

            # SMILES augmentation: add random variants with same label
            if augment > 0:
                for aug_smi in augment_smiles(smi, augment):
                    g2 = smiles_to_graph(aug_smi)
                    if g2 is not None:
                        self.graphs.append(g2)
                        self.params_list.append(par)
                        n_aug += 1

        self.params_tensor = torch.tensor(np.array(self.params_list), dtype=torch.float32)
        if n_fail:
            logger.warning("Skipped %d invalid SMILES", n_fail)
        if augment:
            logger.info("Augmented: %d aug samples added (total %d)", n_aug, len(self.graphs))
        else:
            logger.info("Dataset: %d samples", len(self.graphs))

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple:
        return self.graphs[idx], self.params_tensor[idx]


def collate_graphs(batch: list[tuple]) -> tuple:
    """Batch graphs using PyG Batch."""
    from torch_geometric.data import Batch

    graphs, params = zip(*batch)
    batched = Batch.from_data_list(list(graphs))
    return batched, torch.stack(params)


# ---------------------------------------------------------------------------
# Scaffold split
# ---------------------------------------------------------------------------


def scaffold_split(smiles: list[str], val_frac: float = 0.2) -> tuple[list[int], list[int]]:
    """Bemis-Murcko scaffold split."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffold_to_idx: dict[str, list[int]] = {}
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            sca = (
                MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
            )
            scaffold_to_idx.setdefault(sca, []).append(i)

        groups = sorted(scaffold_to_idx.values(), key=len, reverse=True)
        n_val = int(len(smiles) * val_frac)
        val_idx, train_idx = [], []
        for g in groups:
            if len(val_idx) < n_val:
                val_idx.extend(g)
            else:
                train_idx.extend(g)

        logger.info("Scaffold split: %d train, %d val", len(train_idx), len(val_idx))
        return train_idx, val_idx
    except Exception as exc:
        logger.warning("Scaffold split failed (%s), using random", exc)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(smiles))
        n_val = int(len(smiles) * val_frac)
        return idx[n_val:].tolist(), idx[:n_val].tolist()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def param_loss(pred: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Per-param MSE in log-space (positive params) or linear (logP, rbp)."""
    eps = 1e-6
    pred_clint_L_h = (pred["clint_3a4"] + pred["clint_2d6"]) * IVIVE_FACTOR
    pred_aligned = {
        "logP": pred["logP"],
        "fup": pred["fup"],
        "clint_L_h": pred_clint_L_h,
        "mw": pred["mw"],
        "rbp": pred["rbp"],
        "peff": pred["peff"],
    }
    losses = []
    for i, name in enumerate(PARAM_NAMES):
        p = pred_aligned[name]
        t = target[:, i]
        if name in LOG_PARAMS:
            losses.append(nn.functional.mse_loss(torch.log(p + eps), torch.log(t.abs() + eps)))
        else:
            losses.append(nn.functional.mse_loss(p, t))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay."""

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def forward_batch(encoder: nn.Module, graphs, device: torch.device) -> torch.Tensor:
    """Move batched PyG graph to device and run encoder."""
    graphs.x = graphs.x.to(device)
    graphs.edge_index = graphs.edge_index.to(device)
    graphs.edge_attr = graphs.edge_attr.to(device)
    if hasattr(graphs, "batch") and graphs.batch is not None:
        graphs.batch = graphs.batch.to(device)
    return encoder(graphs)


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------


def train_supervised(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    hidden_dim: int = 256,
    n_layers: int = 3,
    lr: float = 3e-4,
    n_epochs: int = 300,
    warmup_epochs: int = 20,
) -> tuple[nn.Module, nn.Module, float]:
    """Train GNN encoder + param_head supervised."""
    from omega_pbpk.ml.models.foundation.gnn_encoder import MolecularEncoder
    from omega_pbpk.ml.models.foundation.param_head import PKParameterHead

    hidden_dims = tuple([hidden_dim] * n_layers)
    encoder = MolecularEncoder(hidden_dims=hidden_dims, embedding_dim=hidden_dim).to(device)
    param_head = PKParameterHead(embedding_dim=hidden_dim).to(device)

    all_params = list(encoder.parameters()) + list(param_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, n_epochs)

    best_val = float("inf")
    best_enc_state = None
    best_head_state = None

    for epoch in range(n_epochs):
        encoder.train()
        param_head.train()
        train_loss = 0.0
        n_batches = 0

        for graphs, targets in train_loader:
            targets = targets.to(device)
            emb = forward_batch(encoder, graphs, device)
            pred = param_head(emb)
            loss = param_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        encoder.eval()
        param_head.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for graphs, targets in val_loader:
                targets = targets.to(device)
                emb = forward_batch(encoder, graphs, device)
                pred = param_head(emb)
                val_loss += param_loss(pred, targets).item()
                n_val += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(n_val, 1)

        if avg_val < best_val:
            best_val = avg_val
            best_enc_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_head_state = {k: v.cpu().clone() for k, v in param_head.state_dict().items()}

        if (epoch + 1) % 30 == 0:
            current_lr = scheduler.get_last_lr()[0] * lr
            logger.info(
                "Epoch %d/%d  train=%.4f  val=%.4f  best=%.4f  lr=%.2e",
                epoch + 1,
                n_epochs,
                avg_train,
                avg_val,
                best_val,
                current_lr,
            )

    logger.info("Supervised done. Best val: %.4f", best_val)
    encoder.load_state_dict(best_enc_state)
    param_head.load_state_dict(best_head_state)
    return encoder, param_head, best_val


# ---------------------------------------------------------------------------
# Surrogate fine-tuning
# ---------------------------------------------------------------------------


def finetune_surrogate(
    encoder: nn.Module,
    param_head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 60,
    lr: float = 5e-5,
) -> tuple[nn.Module, nn.Module, float]:
    """Fine-tune GNN+param_head through frozen surrogate (curve-level loss)."""
    from omega_pbpk.ml.models.surrogate.differentiable_ode import load_surrogate

    surrogate = load_surrogate(SURROGATE_PATH).to(device)
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    encoder = encoder.to(device)
    param_head = param_head.to(device)

    all_params = list(encoder.parameters()) + list(param_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    scheduler = get_warmup_cosine_scheduler(optimizer, 5, n_epochs)
    mse = nn.MSELoss()

    best_val = float("inf")
    best_enc_state = None
    best_head_state = None

    for epoch in range(n_epochs):
        encoder.train()
        param_head.train()
        train_loss = 0.0
        n_batches = 0

        for graphs, target_params in train_loader:
            target_params = target_params.to(device)
            with torch.no_grad():
                target_curves = surrogate(target_params)

            emb = forward_batch(encoder, graphs, device)
            pred_p = param_head(emb)
            clint_L_h = (pred_p["clint_3a4"] + pred_p["clint_2d6"]) * IVIVE_FACTOR
            pred_params_6d = torch.stack(
                [
                    pred_p["logP"],
                    pred_p["fup"],
                    clint_L_h,
                    pred_p["mw"],
                    pred_p["rbp"],
                    pred_p["peff"],
                ],
                dim=-1,
            )
            pred_curves = surrogate(pred_params_6d)
            loss = mse(pred_curves, target_curves)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        encoder.eval()
        param_head.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for graphs, target_params in val_loader:
                target_params = target_params.to(device)
                target_curves = surrogate(target_params)
                emb = forward_batch(encoder, graphs, device)
                pred_p = param_head(emb)
                clint_L_h = (pred_p["clint_3a4"] + pred_p["clint_2d6"]) * IVIVE_FACTOR
                pred_params_6d = torch.stack(
                    [
                        pred_p["logP"],
                        pred_p["fup"],
                        clint_L_h,
                        pred_p["mw"],
                        pred_p["rbp"],
                        pred_p["peff"],
                    ],
                    dim=-1,
                )
                val_loss += mse(surrogate(pred_params_6d), target_curves).item()
                n_val += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(n_val, 1)

        if avg_val < best_val:
            best_val = avg_val
            best_enc_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_head_state = {k: v.cpu().clone() for k, v in param_head.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Finetune %d/%d  train=%.6f  val=%.6f  best=%.6f",
                epoch + 1,
                n_epochs,
                avg_train,
                avg_val,
                best_val,
            )

    logger.info("Finetune done. Best val: %.6f", best_val)
    encoder.load_state_dict(best_enc_state)
    param_head.load_state_dict(best_head_state)
    return encoder, param_head, best_val


# ---------------------------------------------------------------------------
# Optuna HP sweep
# ---------------------------------------------------------------------------


def optuna_sweep(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_trials: int = 20,
) -> dict:
    """Run Optuna sweep over lr, hidden_dim, n_layers. Returns best HP dict."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        n_layers = trial.suggest_int("n_layers", 2, 4)

        _, _, val_loss = train_supervised(
            train_loader,
            val_loader,
            device=device,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            lr=lr,
            n_epochs=40,  # short sweep
            warmup_epochs=5,
        )
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.info(
        "Optuna best: lr=%.2e  hidden_dim=%d  n_layers=%d  val=%.4f",
        best["lr"],
        best["hidden_dim"],
        best["n_layers"],
        study.best_value,
    )
    return best


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------


def save_checkpoint(
    encoder: nn.Module,
    param_head: nn.Module,
    n_compounds: int,
    val_loss: float,
) -> None:
    """Save full model checkpoint with metadata."""
    from omega_pbpk.ml.models.surrogate.differentiable_ode import load_surrogate

    surrogate = load_surrogate(SURROGATE_PATH)
    surr_ck = torch.load(str(SURROGATE_PATH), map_location="cpu", weights_only=False)

    state: dict = {}
    for k, v in encoder.state_dict().items():
        state[f"encoder.{k}"] = v
    for k, v in param_head.state_dict().items():
        state[f"param_head.{k}"] = v
    for k, v in surrogate.state_dict().items():
        state[f"surrogate.{k}"] = v

    checkpoint = {
        "model_state_dict": state,
        "embedding_dim": encoder.embedding_dim,
        "surrogate_n_output": surr_ck["n_output"],
        "surrogate_hidden": surr_ck["hidden_dim"],
        "dt_h": 0.1,
        "trained_components": ["encoder", "param_head", "surrogate"],
        "training_info": {
            "n_compounds": n_compounds,
            "val_loss": val_loss,
            "date": str(date.today()),
            "surrogate_aafe": surr_ck.get("aafe"),
            "script": "train_l2_gnn_large.py",
        },
    }

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(CHECKPOINT_PATH))
    n_params = sum(v.numel() for v in state.values()) / 1e6
    logger.info("Saved checkpoint to %s (%.1fM params)", CHECKPOINT_PATH, n_params)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale-up L2 GNN training")
    parser.add_argument("--zinc-count", type=int, default=20000, help="Target ZINC compounds")
    parser.add_argument("--skip-download", action="store_true", help="Skip ZINC download")
    parser.add_argument("--skip-labels", action="store_true", help="Skip label generation")
    parser.add_argument("--augment", type=int, default=4, help="SMILES augmentations per compound")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--finetune-epochs", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--optuna", action="store_true", help="Run Optuna sweep first")
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s (%.1f GB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # -----------------------------------------------------------------------
    # Step 1: Get SMILES
    # -----------------------------------------------------------------------
    if not args.skip_download or not ZINC_CACHE.exists():
        logger.info("=== Downloading ZINC drug-like SMILES (target %d) ===", args.zinc_count)
        download_zinc(args.zinc_count, ZINC_CACHE)
    else:
        logger.info("Using cached ZINC SMILES: %s", ZINC_CACHE)

    zinc_smiles = load_zinc_smiles(ZINC_CACHE)
    logger.info("ZINC SMILES loaded: %d", len(zinc_smiles))

    # Load existing TDC labels and get their SMILES
    existing_smiles: set[str] = set()
    existing_labels_path = Path("data/ml/gnn_labels.csv")

    if LABELS_CACHE.exists() and args.skip_labels:
        logger.info("Skipping label generation, loading %s", LABELS_CACHE)
    else:
        # Start fresh cache; copy existing TDC labels if available
        if existing_labels_path.exists() and not LABELS_CACHE.exists():
            import shutil

            shutil.copy(existing_labels_path, LABELS_CACHE)
            logger.info("Copied TDC labels as base: %s", LABELS_CACHE)

        # Load existing SMILES to skip re-labeling
        if LABELS_CACHE.exists():
            with open(LABELS_CACHE) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_smiles.add(row.get("smiles", ""))
            logger.info("Already labeled: %d compounds", len(existing_smiles))

        logger.info("=== Generating ADME labels for ZINC compounds ===")
        generate_labels_batch(zinc_smiles, LABELS_CACHE, existing_smiles=existing_smiles)

    all_smiles, all_params = load_labels(LABELS_CACHE)
    logger.info("Total labeled compounds: %d", len(all_smiles))

    if len(all_smiles) < 100:
        logger.error("Too few compounds (%d), aborting", len(all_smiles))
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: Split
    # -----------------------------------------------------------------------
    train_idx, val_idx = scaffold_split(all_smiles, val_frac=0.2)
    train_smi = [all_smiles[i] for i in train_idx]
    train_par = all_params[train_idx]
    val_smi = [all_smiles[i] for i in val_idx]
    val_par = all_params[val_idx]

    # -----------------------------------------------------------------------
    # Step 3: Build datasets
    # -----------------------------------------------------------------------
    logger.info("Building datasets (train with augment=%d)...", args.augment)
    train_ds = MolecularDataset(train_smi, train_par, augment=args.augment)
    val_ds = MolecularDataset(val_smi, val_par, augment=0)  # no augmentation on val

    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=4,
        persistent_workers=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=4,
        persistent_workers=True,
        pin_memory=(device.type == "cuda"),
    )

    # -----------------------------------------------------------------------
    # Step 4: Optional Optuna sweep
    # -----------------------------------------------------------------------
    hp = {"lr": args.lr, "hidden_dim": args.hidden_dim, "n_layers": args.n_layers}
    if args.optuna:
        logger.info("=== Optuna HP sweep (%d trials) ===", args.optuna_trials)
        hp = optuna_sweep(train_loader, val_loader, device, n_trials=args.optuna_trials)
        logger.info("Best HPs from Optuna: %s", hp)

    # -----------------------------------------------------------------------
    # Step 5: Full supervised training
    # -----------------------------------------------------------------------
    logger.info(
        "=== Supervised training: %d epochs, batch=%d, lr=%.2e, hidden=%d, layers=%d ===",
        args.epochs,
        args.batch_size,
        hp["lr"],
        hp["hidden_dim"],
        hp["n_layers"],
    )
    encoder, param_head, sup_val = train_supervised(
        train_loader,
        val_loader,
        device=device,
        hidden_dim=hp["hidden_dim"],
        n_layers=hp["n_layers"],
        lr=hp["lr"],
        n_epochs=args.epochs,
        warmup_epochs=args.warmup,
    )
    logger.info("Supervised val loss: %.4f", sup_val)

    # -----------------------------------------------------------------------
    # Step 6: Surrogate fine-tuning
    # -----------------------------------------------------------------------
    if SURROGATE_PATH.exists():
        logger.info("=== Surrogate fine-tuning: %d epochs ===", args.finetune_epochs)
        encoder, param_head, ft_val = finetune_surrogate(
            encoder,
            param_head,
            train_loader,
            val_loader,
            device=device,
            n_epochs=args.finetune_epochs,
            lr=hp["lr"] * 0.1,
        )
        final_val = ft_val
        logger.info("Finetune val loss: %.6f", ft_val)
    else:
        logger.warning("Surrogate not found, skipping fine-tuning")
        final_val = sup_val

    # -----------------------------------------------------------------------
    # Step 7: Save
    # -----------------------------------------------------------------------
    save_checkpoint(
        encoder, param_head, n_compounds=len(train_ds) + len(val_ds), val_loss=final_val
    )
    logger.info("=== Training complete ===")


if __name__ == "__main__":
    main()
