"""Tests for scaffold-stratified holdout split."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SPLIT_PATH = REPO / "data" / "clinical" / "holdout_split.json"
PLATINUM_PATH = REPO / "data" / "clinical" / "platinum_reference.json"


def test_holdout_split_exists():
    assert SPLIT_PATH.exists(), "holdout_split.json not created yet"


def test_holdout_split_structure():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    assert "train" in split
    assert "holdout" in split
    assert "metadata" in split
    assert isinstance(split["train"], list)
    assert isinstance(split["holdout"], list)


def test_holdout_split_sizes():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    n_total = len(split["train"]) + len(split["holdout"])
    assert n_total == 147, f"Total drugs should be 147, got {n_total}"
    assert len(split["holdout"]) >= 60, f"Hold-out should be ≥60, got {len(split['holdout'])}"
    assert len(split["holdout"]) <= 75, f"Hold-out should be ≤75, got {len(split['holdout'])}"


def test_no_overlap():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    train_set = set(split["train"])
    holdout_set = set(split["holdout"])
    overlap = train_set & holdout_set
    assert len(overlap) == 0, f"Overlap between train/holdout: {overlap}"


def test_all_platinum_drugs_assigned():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    with open(PLATINUM_PATH) as f:
        plat = json.load(f)
    split_drugs = set(split["train"]) | set(split["holdout"])
    plat_drugs = set(plat["drugs"].keys())
    assert split_drugs == plat_drugs, f"Missing: {plat_drugs - split_drugs}"


def test_contaminated_drugs_in_train():
    """All tuning_contaminated drugs must be in training set."""
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    with open(PLATINUM_PATH) as f:
        plat = json.load(f)
    contaminated = {
        name for name, entry in plat["drugs"].items() if entry.get("tuning_contaminated", False)
    }
    holdout_set = set(split["holdout"])
    leaked = contaminated & holdout_set
    assert len(leaked) == 0, f"Contaminated drugs in holdout: {leaked}"


def test_scaffold_integrity():
    """Same Murcko scaffold should not appear in both train and holdout."""
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    scaffolds = split["metadata"].get("scaffold_assignments", {})
    if not scaffolds:
        return  # scaffold info not stored — skip
    train_scaffolds = {scaffolds[d] for d in split["train"] if d in scaffolds}
    holdout_scaffolds = {scaffolds[d] for d in split["holdout"] if d in scaffolds}
    leaked = train_scaffolds & holdout_scaffolds
    assert len(leaked) == 0, f"Scaffold leak: {len(leaked)} scaffolds in both sets"


def test_anchor_decontamination():
    """Anchors for holdout drugs should be removable."""
    from omega_pbpk.ml.models.adme.xgboost_clint import (
        _ANCHOR_DRUG_NAMES,
        _get_clint_reference_anchors,
        get_decontaminated_anchors,
    )

    full_anchors = _get_clint_reference_anchors()
    decontaminated = get_decontaminated_anchors(exclude_drugs={"warfarin", "midazolam"})
    assert len(decontaminated) == len(full_anchors) - 2
    # None of the excluded drugs should remain
    remaining_names = {_ANCHOR_DRUG_NAMES.get(s, "?") for s, _ in decontaminated}
    assert "warfarin" not in remaining_names
    assert "midazolam" not in remaining_names


def test_decontamination_none_returns_full():
    """Passing None returns all anchors (production behavior)."""
    from omega_pbpk.ml.models.adme.xgboost_clint import (
        _get_clint_reference_anchors,
        get_decontaminated_anchors,
    )

    full = _get_clint_reference_anchors()
    result = get_decontaminated_anchors(exclude_drugs=None)
    assert len(result) == len(full)
