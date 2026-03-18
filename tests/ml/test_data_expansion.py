"""Data expansion pipeline tests."""

import pytest


def test_pkdb_expansion_script_runs():
    """PK-DB expansion script should run without crashing in dry-run mode."""
    import subprocess

    result = subprocess.run(
        ["python", "scripts/expand_pkdb_cmax.py", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"


def test_build_ml_dataset_script_runs():
    """Build ML dataset script should run without crashing."""
    import subprocess

    result = subprocess.run(
        ["python", "scripts/build_ml_dataset.py"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"


def test_fda_expansion_script_runs():
    """FDA expansion script should run without crashing in dry-run mode."""
    import subprocess

    result = subprocess.run(
        ["python", "scripts/expand_fda_cmax.py", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"


def test_ml_dataset_has_strict_split():
    """ML dataset must have non-overlapping train/val/test splits."""
    import json
    from pathlib import Path

    split_path = Path("data/ml/clinical/dataset_split.json")
    if not split_path.exists():
        pytest.skip("Dataset not yet built")

    split = json.loads(split_path.read_text())
    train = set(split["train"])
    val = set(split["validation"])
    test = set(split["test"])

    assert len(train & val) == 0, "Train/val overlap"
    assert len(train & test) == 0, "Train/test overlap"
    assert len(val & test) == 0, "Val/test overlap"
    assert len(train) >= 50, f"Train too small: {len(train)}"
    assert len(val) >= 10, f"Val too small: {len(val)}"
    assert len(test) >= 10, f"Test too small: {len(test)}"


def test_ml_dataset_csv_matches_split():
    """Expanded CSV must contain exactly the drugs in the split."""
    import csv
    import json
    from pathlib import Path

    split_path = Path("data/ml/clinical/dataset_split.json")
    csv_path = Path("data/ml/clinical/expanded_cmax.csv")
    if not split_path.exists() or not csv_path.exists():
        pytest.skip("Dataset not yet built")

    split = json.loads(split_path.read_text())
    all_split_drugs = set(split["train"] + split["validation"] + split["test"])

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        csv_drugs = {row["drug"] for row in reader}

    assert csv_drugs == all_split_drugs, (
        f"CSV drugs ({len(csv_drugs)}) != split drugs ({len(all_split_drugs)}). "
        f"In CSV only: {csv_drugs - all_split_drugs}, "
        f"In split only: {all_split_drugs - csv_drugs}"
    )


def test_ml_dataset_temporal_in_test():
    """Temporal holdout drugs with Cmax data must be in test split."""
    import csv
    import json
    from pathlib import Path

    split_path = Path("data/ml/clinical/dataset_split.json")
    temporal_path = Path("data/ml/clinical/temporal_holdout.csv")
    gold_path = Path("data/ml/clinical/cmax_training_set.csv")
    if not split_path.exists() or not temporal_path.exists():
        pytest.skip("Dataset not yet built")

    split = json.loads(split_path.read_text())

    # Get gold drug names (these override temporal for train)
    gold_names = set()
    if gold_path.exists():
        with open(gold_path) as f:
            for row in csv.DictReader(f):
                gold_names.add(row["drug"].strip().lower())

    # Get temporal drugs that have Cmax and are NOT in gold
    temporal_with_cmax = set()
    with open(temporal_path) as f:
        for row in csv.DictReader(f):
            name = row["drug_name"].strip()
            cmax = row.get("obs_cmax_mg_L", "").strip()
            if cmax and name.lower() not in gold_names:
                temporal_with_cmax.add(name.lower())

    # All test drugs (lowercased)
    test_lower = {d.lower() for d in split["test"]}

    for drug in temporal_with_cmax:
        assert drug in test_lower, f"Temporal drug '{drug}' not in test split"
