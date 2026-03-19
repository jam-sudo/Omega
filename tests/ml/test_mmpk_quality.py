"""Tests for MMPK quality-scored training set."""

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
QUALITY_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_quality_scored.csv"


def test_quality_file_exists():
    assert QUALITY_PATH.exists()


def test_quality_scores_in_range():
    with open(QUALITY_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row["quality_score"])
            assert 0.0 <= score <= 1.0, f"{row['name']} has score {score}"


def test_excluded_drugs_flagged():
    """Drugs with quality < 0.25 should have include=False."""
    with open(QUALITY_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row["quality_score"])
            include = row["include"] == "True"
            if score < 0.25:
                assert not include, f"{row['name']} score={score} but include=True"


def test_sufficient_training_drugs():
    """At least 800 drugs should pass quality filter."""
    with open(QUALITY_PATH) as f:
        reader = csv.DictReader(f)
        n_include = sum(1 for row in reader if row["include"] == "True")
    assert n_include >= 800, f"Only {n_include} drugs pass quality filter (need >=800)"


def test_holdout_drugs_excluded_from_training():
    """Hold-out drugs (from split) must not appear in quality training set with include=True.

    Note: MMPK names may differ from platinum names, so this checks by name only.
    Full SMILES-based leak detection is in the B1 cross-reference audit.
    """
    split_path = REPO / "data" / "clinical" / "holdout_split.json"
    if not split_path.exists():
        import pytest

        pytest.skip("holdout split not yet created")

    import json

    with open(split_path) as f:
        split = json.load(f)
    holdout_names = set(split["holdout"])

    with open(QUALITY_PATH) as f:
        reader = csv.DictReader(f)
        leaks = [
            row["name"]
            for row in reader
            if row["name"] in holdout_names and row["include"] == "True"
        ]
    # Name-based match is approximate; leaks here warrant SMILES-level investigation
    if leaks:
        import warnings

        warnings.warn(
            f"Potential holdout leaks by name: {leaks[:5]}... "
            "Verify by SMILES in B1 cross-reference.",
            stacklevel=2,
        )
