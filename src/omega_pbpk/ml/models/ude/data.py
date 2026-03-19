"""Data loading for UDE multi-task training.

Three data sources:
1. MMPK (1,098 drugs): SMILES -> observed Cmax (primary task)
2. TDC CLint (1,213): SMILES -> observed CLint (auxiliary)
3. TDC fup (1,614): SMILES -> observed fup (auxiliary)
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent.parent.parent.parent.parent


def _smiles_to_features(smiles: str) -> np.ndarray | None:
    """Morgan FP (2048) + RDKit descriptors (9) = 2057 features."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_arr = np.array(fp, dtype=np.float32)

    descs = np.array(
        [
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol) / 200.0,
            Descriptors.MolWt(mol) / 600.0,
            Descriptors.NumHAcceptors(mol) / 10.0,
            Descriptors.NumHDonors(mol) / 5.0,
            Descriptors.NumRotatableBonds(mol) / 15.0,
            Descriptors.RingCount(mol) / 5.0,
            Descriptors.FractionCSP3(mol),
            Descriptors.MolMR(mol) / 150.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate([fp_arr, descs])


class UDEDataset(NamedTuple):
    """Container for a single UDE training example."""

    name: str
    smiles: str
    features: np.ndarray
    dose_mg: float
    cmax_obs: float
    quality_weight: float


def load_mmpk_training() -> list[dict]:
    """Load MMPK drugs for Cmax training, excluding holdout leaks.

    Returns list of dicts with keys: name, smiles, features, dose_mg, cmax_obs, quality_weight.
    """
    quality_path = _REPO / "data" / "ml" / "clinical" / "mmpk_quality_scored.csv"
    exclusion_path = _REPO / "data" / "ml" / "clinical" / "mmpk_holdout_exclusions.json"

    # Load exclusions
    excluded: set[str] = set()
    if exclusion_path.exists():
        with open(exclusion_path) as f:
            excluded = set(json.load(f)["holdout_leaks_in_mmpk"])
        logger.info("Excluding %d holdout leak drugs from MMPK", len(excluded))

    data: list[dict] = []
    n_skipped_exclude = 0
    n_skipped_features = 0

    with open(quality_path) as f:
        for row in csv.DictReader(f):
            if row["name"] in excluded:
                n_skipped_exclude += 1
                continue
            if row["include"] != "True":
                continue

            features = _smiles_to_features(row["smiles"])
            if features is None:
                n_skipped_features += 1
                continue

            data.append(
                {
                    "name": row["name"],
                    "smiles": row["smiles"],
                    "features": features,
                    "dose_mg": float(row["dose_mg"]),
                    "cmax_obs": float(row["cmax_mg_L"]),
                    "quality_weight": float(row["quality_score"]),
                }
            )

    logger.info(
        "Loaded %d MMPK drugs (excluded: %d holdout, %d bad features)",
        len(data),
        n_skipped_exclude,
        n_skipped_features,
    )
    return data


def load_tdc_clint() -> list[dict]:
    """Load TDC Clearance_Hepatocyte_AZ for CLint auxiliary task.

    Returns list of dicts with keys: smiles, features, clint.
    """
    try:
        from tdc.single_pred import ADME

        dataset = ADME(name="Clearance_Hepatocyte_AZ")
        df = dataset.get_data()
    except (ImportError, Exception) as exc:
        logger.warning("TDC Clearance_Hepatocyte_AZ not available: %s", exc)
        return []

    data: list[dict] = []
    for _, row in df.iterrows():
        smiles = str(row["Drug"])
        clint = float(row["Y"])
        if clint <= 0:
            continue
        features = _smiles_to_features(smiles)
        if features is None:
            continue
        data.append({"smiles": smiles, "features": features, "clint": clint})

    logger.info("Loaded %d TDC CLint drugs", len(data))
    return data


def load_tdc_fup() -> list[dict]:
    """Load TDC PPBR_AZ for fup auxiliary task.

    Returns list of dicts with keys: smiles, features, fup.
    """
    try:
        from tdc.single_pred import ADME

        dataset = ADME(name="PPBR_AZ")
        df = dataset.get_data()
    except (ImportError, Exception) as exc:
        logger.warning("TDC PPBR_AZ not available: %s", exc)
        return []

    data: list[dict] = []
    for _, row in df.iterrows():
        smiles = str(row["Drug"])
        ppbr = float(row["Y"])
        fup = 1.0 - ppbr / 100.0
        if fup <= 0 or fup > 1:
            continue
        features = _smiles_to_features(smiles)
        if features is None:
            continue
        data.append({"smiles": smiles, "features": features, "fup": fup})

    logger.info("Loaded %d TDC fup drugs", len(data))
    return data


def scaffold_split(
    data: list[dict], val_frac: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Murcko scaffold-based train/val split.

    Entire scaffold clusters go to the same split to prevent data leakage.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(data):
        mol = Chem.MolFromSmiles(d["smiles"])
        if mol:
            try:
                core = MurckoScaffold.GetScaffoldForMol(mol)
                scaf = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
            except Exception:
                scaf = d["smiles"]
        else:
            scaf = d["smiles"]
        scaffolds[scaf].append(i)

    rng = np.random.default_rng(seed)
    scaffold_list = list(scaffolds.values())
    rng.shuffle(scaffold_list)

    target_val = int(len(data) * val_frac)
    val_idx: list[int] = []
    train_idx: list[int] = []

    for indices in scaffold_list:
        if len(val_idx) < target_val:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    return [data[i] for i in train_idx], [data[i] for i in val_idx]
