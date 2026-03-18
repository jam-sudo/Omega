#!/usr/bin/env python3
"""Build unified ML Cmax dataset with strict train/val/test split.

Merges all Cmax data sources, applies quality filters, deduplicates,
and creates a 60/20/20 train/val/test split with zero overlap.

Sources:
  1. data/ml/clinical/cmax_training_set.csv  (gold-tier benchmark)
  2. data/ml/clinical/pkdb_expanded_cmax.csv  (PK-DB expanded)
  3. data/ml/clinical/fda_expanded_cmax.csv   (FDA expanded)
  4. data/ml/clinical/temporal_holdout.csv     (temporal holdout → test)

Output:
  - data/ml/clinical/expanded_cmax.csv     (all drugs, merged)
  - data/ml/clinical/dataset_split.json    (drug lists per split)
"""

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data/ml/clinical")
GOLD_PATH = DATA_DIR / "cmax_training_set.csv"
PKDB_PATH = DATA_DIR / "pkdb_expanded_cmax.csv"
FDA_PATH = DATA_DIR / "fda_expanded_cmax.csv"
TEMPORAL_PATH = DATA_DIR / "temporal_holdout.csv"

OUT_CSV = DATA_DIR / "expanded_cmax.csv"
OUT_SPLIT = DATA_DIR / "dataset_split.json"

# ── Quality filter constants ───────────────────────────────────────────
MIN_DOSE_MG = 0.0
MAX_DOSE_MG = 5000.0
MIN_CMAX = 0.0  # exclusive
MIN_CMAX_PER_DOSE = 0.00001  # mg/L per mg
MAX_CMAX_PER_DOSE = 1.0  # mg/L per mg

# Known bad entries (drug names, lowercased) — dose extraction errors
KNOWN_BAD = {"colchicine", "mefenamic acid"}

# ── Source priority for dedup (higher = preferred) ─────────────────────
SOURCE_PRIORITY = {"gold": 4, "platinum": 4, "pkdb_expanded": 3, "fda_expanded": 2, "temporal": 1}

# ── Reproducible split ─────────────────────────────────────────────────
SPLIT_SEED = 42


def validate_smiles(smiles: str) -> bool:
    """Check if SMILES parses in RDKit."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def normalize_drug_name(name: str) -> str:
    """Lowercase, strip whitespace for matching."""
    return name.strip().lower()


def load_gold(path: Path) -> list[dict]:
    """Load gold-tier benchmark CSV."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug = row["drug"].strip()
            smiles = row["smiles"].strip()
            try:
                dose = float(row["dose_mg"])
                cmax = float(row["obs_cmax"])
            except (ValueError, TypeError):
                continue
            tier = row.get("tier", "gold").strip()
            source = "gold" if tier == "gold" else tier
            rows.append(
                {
                    "drug": drug,
                    "smiles": smiles,
                    "dose_mg": dose,
                    "cmax_mg_L": cmax,
                    "source": source,
                    "tier": tier,
                }
            )
    return rows


def load_pkdb(path: Path) -> list[dict]:
    """Load PK-DB expanded CSV."""
    rows = []
    if not path.exists():
        log.warning("PK-DB file not found: %s", path)
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug = row["drug"].strip()
            smiles = row["smiles"].strip()
            if not drug or not smiles:
                continue
            try:
                dose = float(row["dose_mg"])
                cmax = float(row["obs_cmax"])
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "drug": drug,
                    "smiles": smiles,
                    "dose_mg": dose,
                    "cmax_mg_L": cmax,
                    "source": "pkdb_expanded",
                    "tier": "pkdb_expanded",
                }
            )
    return rows


def load_fda(path: Path) -> list[dict]:
    """Load FDA expanded CSV."""
    rows = []
    if not path.exists():
        log.warning("FDA file not found: %s", path)
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug = row["drug"].strip()
            smiles = row["smiles"].strip()
            if not drug or not smiles:
                continue
            try:
                dose = float(row["dose_mg"])
                cmax = float(row["obs_cmax"])
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "drug": drug,
                    "smiles": smiles,
                    "dose_mg": dose,
                    "cmax_mg_L": cmax,
                    "source": "fda_expanded",
                    "tier": "fda_expanded",
                }
            )
    return rows


def load_temporal(path: Path) -> list[dict]:
    """Load temporal holdout CSV (different column names)."""
    rows = []
    if not path.exists():
        log.warning("Temporal file not found: %s", path)
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug = row["drug_name"].strip()
            smiles = row["smiles"].strip()
            if not drug or not smiles:
                continue
            # Some temporal drugs have no Cmax (only AUC / t_half)
            cmax_str = row.get("obs_cmax_mg_L", "").strip()
            if not cmax_str:
                continue
            try:
                dose = float(row["dose_mg"])
                cmax = float(cmax_str)
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "drug": drug,
                    "smiles": smiles,
                    "dose_mg": dose,
                    "cmax_mg_L": cmax,
                    "source": "temporal",
                    "tier": "temporal",
                }
            )
    return rows


def quality_filter(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply quality filters. Returns (passed, rejected) with rejection reasons."""
    passed = []
    rejected = []

    for row in rows:
        drug_lower = normalize_drug_name(row["drug"])

        # Known bad entries
        if drug_lower in KNOWN_BAD:
            rejected.append({**row, "reject_reason": "known_bad_entry"})
            continue

        # SMILES validation
        if not validate_smiles(row["smiles"]):
            rejected.append({**row, "reject_reason": "invalid_smiles"})
            continue

        # Dose range
        dose = row["dose_mg"]
        if dose <= MIN_DOSE_MG or dose >= MAX_DOSE_MG:
            rejected.append({**row, "reject_reason": f"dose_out_of_range ({dose})"})
            continue

        # Cmax positive
        cmax = row["cmax_mg_L"]
        if cmax <= MIN_CMAX:
            rejected.append({**row, "reject_reason": f"cmax_not_positive ({cmax})"})
            continue

        # Cmax/dose ratio sanity check
        ratio = cmax / dose
        if ratio < MIN_CMAX_PER_DOSE or ratio > MAX_CMAX_PER_DOSE:
            rejected.append({**row, "reject_reason": f"cmax_dose_ratio_out_of_range ({ratio:.6f})"})
            continue

        passed.append(row)

    return passed, rejected


def deduplicate(rows: list[dict]) -> list[dict]:
    """Deduplicate by normalized drug name, keeping highest-priority source."""
    by_drug: dict[str, dict] = {}
    for row in rows:
        key = normalize_drug_name(row["drug"])
        priority = SOURCE_PRIORITY.get(row["source"], 0)
        if key not in by_drug:
            by_drug[key] = row
        else:
            existing_priority = SOURCE_PRIORITY.get(by_drug[key]["source"], 0)
            if priority > existing_priority:
                by_drug[key] = row
    return list(by_drug.values())


def create_split(
    drugs: list[dict],
    gold_names: set[str],
    temporal_names: set[str],
    seed: int = SPLIT_SEED,
) -> dict[str, list[str]]:
    """Create train/val/test split.

    Rules:
      - Gold-tier benchmark drugs → train
      - Temporal holdout drugs → test
      - Remaining: random 60/20/20 stratified split
    """
    rng = np.random.RandomState(seed)

    train_drugs = []
    test_drugs = []
    remaining_drugs = []

    for d in drugs:
        name_lower = normalize_drug_name(d["drug"])
        if name_lower in temporal_names:
            test_drugs.append(d["drug"])
        elif name_lower in gold_names:
            train_drugs.append(d["drug"])
        else:
            remaining_drugs.append(d["drug"])

    # Shuffle remaining
    rng.shuffle(remaining_drugs)

    # Split remaining into val (20% of total) and test (fill), plus top-up train
    total = len(drugs)
    target_val = max(1, int(round(total * 0.20)))
    target_test = max(1, int(round(total * 0.20)))

    # How many more test drugs do we need beyond temporal?
    test_need = max(0, target_test - len(test_drugs))
    # How many val drugs do we need?
    val_need = min(target_val, len(remaining_drugs) - test_need)

    val_drugs = remaining_drugs[:val_need]
    extra_test = remaining_drugs[val_need : val_need + test_need]
    extra_train = remaining_drugs[val_need + test_need :]

    test_drugs.extend(extra_test)
    train_drugs.extend(extra_train)

    return {
        "train": sorted(train_drugs),
        "validation": sorted(val_drugs),
        "test": sorted(test_drugs),
    }


def main():
    log.info("=" * 60)
    log.info("Building unified ML Cmax dataset")
    log.info("=" * 60)

    # ── Load all sources ──────────────────────────────────────────
    gold_rows = load_gold(GOLD_PATH)
    pkdb_rows = load_pkdb(PKDB_PATH)
    fda_rows = load_fda(FDA_PATH)
    temporal_rows = load_temporal(TEMPORAL_PATH)

    log.info(
        "Loaded: gold=%d, pkdb=%d, fda=%d, temporal=%d",
        len(gold_rows),
        len(pkdb_rows),
        len(fda_rows),
        len(temporal_rows),
    )

    all_rows = gold_rows + pkdb_rows + fda_rows + temporal_rows
    log.info("Total raw rows: %d", len(all_rows))

    # ── Quality filter ────────────────────────────────────────────
    passed, rejected = quality_filter(all_rows)
    log.info("Quality filter: %d passed, %d rejected", len(passed), len(rejected))
    for r in rejected:
        log.info("  REJECTED: %s [%s] — %s", r["drug"], r["source"], r["reject_reason"])

    # ── Deduplicate ───────────────────────────────────────────────
    deduped = deduplicate(passed)
    log.info("After dedup: %d unique drugs (from %d rows)", len(deduped), len(passed))

    # Show dedup choices
    passed_names = {}
    for row in passed:
        key = normalize_drug_name(row["drug"])
        if key not in passed_names:
            passed_names[key] = []
        passed_names[key].append(row["source"])
    dupes = {k: v for k, v in passed_names.items() if len(v) > 1}
    if dupes:
        log.info("Deduplicated drugs (kept highest priority):")
        for drug, sources in sorted(dupes.items()):
            kept = next(d for d in deduped if normalize_drug_name(d["drug"]) == drug)
            log.info("  %s: sources=%s → kept=%s", drug, sources, kept["source"])

    # ── Create split ──────────────────────────────────────────────
    # Get gold benchmark names and temporal names for routing
    gold_names = {normalize_drug_name(r["drug"]) for r in gold_rows}
    temporal_names_from_file = set()
    # Re-read temporal to get ALL drug names (even those without Cmax)
    if TEMPORAL_PATH.exists():
        with open(TEMPORAL_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["drug_name"].strip()
                if name:
                    temporal_names_from_file.add(normalize_drug_name(name))

    # Remove temporal drugs that are also in gold (gold wins for training)
    # But if a drug is in both gold AND temporal, keep it in train (gold priority)
    pure_temporal = temporal_names_from_file - gold_names
    log.info("Gold benchmark drugs: %d", len(gold_names))
    log.info(
        "Pure temporal holdout drugs: %d (excluded from gold: %d)",
        len(pure_temporal),
        len(temporal_names_from_file & gold_names),
    )

    split = create_split(deduped, gold_names, pure_temporal)
    total_split = len(split["train"]) + len(split["validation"]) + len(split["test"])
    log.info(
        "Split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%)",
        len(split["train"]),
        100 * len(split["train"]) / total_split,
        len(split["validation"]),
        100 * len(split["validation"]) / total_split,
        len(split["test"]),
        100 * len(split["test"]) / total_split,
    )

    # Verify zero overlap
    train_set = set(split["train"])
    val_set = set(split["validation"])
    test_set = set(split["test"])
    assert len(train_set & val_set) == 0, f"Train/val overlap: {train_set & val_set}"
    assert len(train_set & test_set) == 0, f"Train/test overlap: {train_set & test_set}"
    assert len(val_set & test_set) == 0, f"Val/test overlap: {val_set & test_set}"
    assert total_split == len(deduped), f"Split total {total_split} != dedup {len(deduped)}"
    log.info("Zero overlap verified across all splits.")

    # ── Save outputs ──────────────────────────────────────────────
    # Save merged CSV
    fieldnames = ["drug", "smiles", "dose_mg", "cmax_mg_L", "source", "tier"]
    deduped_sorted = sorted(deduped, key=lambda d: d["drug"].lower())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in deduped_sorted:
            writer.writerow({k: row[k] for k in fieldnames})
    log.info("Saved merged CSV: %s (%d drugs)", OUT_CSV, len(deduped_sorted))

    # Save split JSON
    split_meta = {
        "train": split["train"],
        "validation": split["validation"],
        "test": split["test"],
        "metadata": {
            "total_drugs": len(deduped),
            "seed": SPLIT_SEED,
            "sources": {
                "gold": len(gold_rows),
                "pkdb_expanded": len(pkdb_rows),
                "fda_expanded": len(fda_rows),
                "temporal": len(temporal_rows),
            },
            "after_quality_filter": len(passed),
            "after_dedup": len(deduped),
            "rejected_count": len(rejected),
            "rejected_drugs": [
                {"drug": r["drug"], "source": r["source"], "reason": r["reject_reason"]}
                for r in rejected
            ],
        },
    }
    with open(OUT_SPLIT, "w") as f:
        json.dump(split_meta, f, indent=2)
    log.info("Saved split JSON: %s", OUT_SPLIT)

    # ── Summary ───────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)

    # Per-source counts in final dataset
    source_counts = {}
    for row in deduped:
        src = row["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in sorted(source_counts.items()):
        log.info("  Source %-15s: %d drugs", src, count)

    # Per-split details
    log.info("")
    log.info("  Train (%d):", len(split["train"]))
    for d in split["train"]:
        log.info("    %s", d)
    log.info("  Validation (%d):", len(split["validation"]))
    for d in split["validation"]:
        log.info("    %s", d)
    log.info("  Test (%d):", len(split["test"]))
    for d in split["test"]:
        log.info("    %s", d)

    log.info("")
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
