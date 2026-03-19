"""Tests for UDE data preprocessing pipeline."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def test_ude_data_module_exists():
    from omega_pbpk.ml.models.ude.data import UDEDataset  # noqa: F401


def test_mmpk_dataset_loads():
    from omega_pbpk.ml.models.ude.data import load_mmpk_training

    data = load_mmpk_training()
    assert len(data) >= 800, f"Only {len(data)} drugs loaded"
    d = data[0]
    assert "features" in d
    assert "cmax_obs" in d
    assert "quality_weight" in d
    assert d["features"].shape == (2057,)
    assert d["cmax_obs"] > 0
    assert 0 < d["quality_weight"] <= 1.0


def test_holdout_excluded():
    from omega_pbpk.ml.models.ude.data import load_mmpk_training

    exclusion_path = REPO / "data" / "ml" / "clinical" / "mmpk_holdout_exclusions.json"
    with open(exclusion_path) as f:
        exclusions = json.load(f)
    excluded_names = set(exclusions["holdout_leaks_in_mmpk"])

    data = load_mmpk_training()
    train_names = {d["name"] for d in data}
    overlap = train_names & excluded_names
    assert len(overlap) == 0, f"Holdout leaks in training: {overlap}"


def test_scaffold_split():
    from omega_pbpk.ml.models.ude.data import load_mmpk_training, scaffold_split

    data = load_mmpk_training()
    train, val = scaffold_split(data, val_frac=0.2, seed=42)
    assert len(train) + len(val) == len(data)
    assert len(val) >= len(data) * 0.15, "Val set too small"
    # No drug should appear in both
    train_names = {d["name"] for d in train}
    val_names = {d["name"] for d in val}
    assert len(train_names & val_names) == 0


def test_feature_computation():
    from omega_pbpk.ml.models.ude.data import _smiles_to_features

    # Caffeine
    feat = _smiles_to_features("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
    assert feat is not None
    assert feat.shape == (2057,)
    assert feat[:2048].sum() > 0  # Some FP bits should be on
    # Invalid SMILES
    assert _smiles_to_features("NOT_A_SMILES") is None
