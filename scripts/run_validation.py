"""Runner script for T8/T9/T10 validation tasks.

T8: Confidence calibration — scaffold-split 20% holdout from adme_reference.csv
T9: Novel molecule validation Tier 2 (structural analogs)
T10: Novel molecule validation Tier 3 (de novo molecules)

Usage:
    python scripts/run_validation.py [--t8] [--t9] [--t10] [--all] [--verbose]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_validation")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Scaffold split helper
# ---------------------------------------------------------------------------

def scaffold_split(smiles_list: list[str], test_frac: float = 0.20, seed: int = 42):
    """Scaffold-based train/test split. Returns (train_indices, test_indices)."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        logger.warning("RDKit not available; falling back to random split")
        import random
        rng = random.Random(seed)
        idx = list(range(len(smiles_list)))
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_frac))
        return idx[n_test:], idx[:n_test]

    scaffold_to_indices: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold = "__invalid__"
        else:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except Exception:
                scaffold = smi
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    # Sort scaffolds by size desc (put large groups into train)
    scaffold_groups = sorted(scaffold_to_indices.values(), key=len, reverse=True)
    n_test_target = int(len(smiles_list) * test_frac)

    test_idx: list[int] = []
    train_idx: list[int] = []
    for group in scaffold_groups:
        if len(test_idx) < n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    return train_idx, test_idx


# ---------------------------------------------------------------------------
# T8: Confidence calibration
# ---------------------------------------------------------------------------

def run_t8(verbose: bool = False) -> dict:
    """Run confidence calibration on scaffold holdout from adme_reference.csv."""
    from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor
    from omega_pbpk.ml.evaluation.metrics import calibrate_uncertainty

    data_path = REPO_ROOT / "data" / "adme_reference.csv"
    rows: list[dict] = []
    with open(data_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    # Build SMILES list (use smiles col if present, else name lookup)
    from omega_pbpk.ml.models.adme.xgboost_adme import _name_to_smiles

    smiles_list: list[str] = []
    valid_rows: list[dict] = []
    for row in rows:
        smi = row.get("smiles", row.get("SMILES", ""))
        if not smi:
            smi = _name_to_smiles(row.get("name", ""))
        if smi:
            smiles_list.append(smi)
            row["_smiles"] = smi
            valid_rows.append(row)

    logger.info("Loaded %d compounds with SMILES from %s", len(valid_rows), data_path.name)

    # Scaffold split — use 20% as calibration holdout
    train_idx, test_idx = scaffold_split(smiles_list, test_frac=0.20)
    holdout_rows = [valid_rows[i] for i in test_idx]
    logger.info("Holdout set: %d compounds (%.0f%%)", len(holdout_rows), 100 * len(holdout_rows) / len(valid_rows))

    # Write holdout to a temp file that calibrate_uncertainty can read
    import tempfile, os
    tmp_path = Path(tempfile.mktemp(suffix=".csv"))
    fieldnames = list(holdout_rows[0].keys())
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(holdout_rows)

    try:
        predictor = EnsembleADMEPredictor()
        report = calibrate_uncertainty(predictor, data_path=tmp_path, target_coverage=0.90)

        print("\n" + str(report))

        # --- Monotonicity check ---
        # Group holdout predictions by confidence label, compute AAFE per group
        mono_results = _check_confidence_monotonicity(predictor, holdout_rows, verbose=verbose)

        return {"calibration_report": report, "monotonicity": mono_results}
    finally:
        if tmp_path.exists():
            os.unlink(tmp_path)


def _check_confidence_monotonicity(predictor, holdout_rows: list[dict], verbose: bool = False) -> dict:
    """Check that high-confidence predictions have lower AAFE than low-confidence ones."""
    import math
    from omega_pbpk.ml.models.adme.xgboost_adme import _name_to_smiles

    groups: dict[str, list[float]] = {"high": [], "medium": [], "low": []}
    # Use fup as the test property (most entries have ground truth)
    for row in holdout_rows:
        smi = row.get("_smiles", "")
        if not smi:
            continue
        val_str = row.get("fup")
        if not val_str:
            continue
        try:
            true_val = float(val_str)
        except ValueError:
            continue
        try:
            adme = predictor.predict(smi)
            pred = adme.fup
            if pred > 0 and true_val > 0:
                fe = max(pred / true_val, true_val / pred)
                conf = adme.confidence
                if conf in groups:
                    groups[conf].append(fe)
        except Exception as exc:
            logger.debug("Monotonicity check failed for %s: %s", smi, exc)
            continue

    aafe_by_conf: dict[str, float] = {}
    for conf, fes in groups.items():
        if fes:
            aafe = math.exp(sum(math.log(fe) for fe in fes) / len(fes))
            aafe_by_conf[conf] = round(aafe, 3)
            logger.info("  AAFE[fup, %s confidence] = %.3f (n=%d)", conf, aafe, len(fes))
        else:
            logger.info("  No samples for %s confidence", conf)

    # Check monotonicity: high < medium < low (where present)
    h = aafe_by_conf.get("high")
    m = aafe_by_conf.get("medium")
    lo = aafe_by_conf.get("low")
    checks = []
    if h and m:
        ok = h <= m
        checks.append(("high_le_medium", ok, f"high={h:.3f} medium={m:.3f}"))
    if m and lo:
        ok = m <= lo
        checks.append(("medium_le_low", ok, f"medium={m:.3f} low={lo:.3f}"))
    if h and lo:
        ok = h <= lo
        checks.append(("high_le_low", ok, f"high={h:.3f} low={lo:.3f}"))

    monotonic = all(c[1] for c in checks) if checks else True
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        print(f"  Monotonicity [{name}]: {detail} [{status}]")
    print(f"  Confidence monotonicity overall: {'MONOTONIC' if monotonic else 'NOT MONOTONIC'}")

    return {"aafe_by_confidence": aafe_by_conf, "monotonic": monotonic, "checks": checks}


# ---------------------------------------------------------------------------
# T9: Tier 2 structural analog validation
# ---------------------------------------------------------------------------

def run_t9(verbose: bool = False) -> dict:
    """Run Tier 2 structural analog validation."""
    from omega_pbpk.ml.evaluation.novel_validator import NovelMoleculeValidator

    validator = NovelMoleculeValidator()
    logger.info("Running Tier 2 (structural analog) validation...")
    result = validator.validate_analogs(verbose=verbose)

    print(f"\n{result}")
    if verbose and result.details:
        print("\nDetails:")
        for d in result.details:
            print(f"  {d}")

    return {"tier2": result}


# ---------------------------------------------------------------------------
# T10: Tier 3 de novo validation
# ---------------------------------------------------------------------------

def run_t10(n_molecules: int = 1000, verbose: bool = False) -> dict:
    """Run Tier 3 de novo molecule validation."""
    from omega_pbpk.ml.evaluation.novel_validator import NovelMoleculeValidator

    validator = NovelMoleculeValidator()
    logger.info("Running Tier 3 (de novo, n=%d) validation...", n_molecules)
    result = validator.validate_de_novo(n_molecules=n_molecules, verbose=verbose)

    print(f"\n{result}")
    if verbose and result.details:
        # Print a sample of failures
        failures = [d for d in result.details if d.get("status") != "PASS"]
        if failures:
            print(f"\nSample failures (first 10):")
            for d in failures[:10]:
                print(f"  {d}")

    return {"tier3": result}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run validation tasks T8/T9/T10")
    parser.add_argument("--t8", action="store_true", help="Run T8 confidence calibration")
    parser.add_argument("--t9", action="store_true", help="Run T9 structural analog validation")
    parser.add_argument("--t10", action="store_true", help="Run T10 de novo validation")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--n-molecules", type=int, default=1000, help="De novo molecule count (T10)")
    args = parser.parse_args()

    if not (args.t8 or args.t9 or args.t10 or getattr(args, "all")):
        parser.print_help()
        sys.exit(1)

    run_all = getattr(args, "all")

    if args.t8 or run_all:
        print("\n" + "=" * 60)
        print("T8: CONFIDENCE CALIBRATION")
        print("=" * 60)
        t8_result = run_t8(verbose=args.verbose)

    if args.t9 or run_all:
        print("\n" + "=" * 60)
        print("T9: TIER 2 STRUCTURAL ANALOG VALIDATION")
        print("=" * 60)
        t9_result = run_t9(verbose=args.verbose)

    if args.t10 or run_all:
        print("\n" + "=" * 60)
        print(f"T10: TIER 3 DE NOVO VALIDATION (n={args.n_molecules})")
        print("=" * 60)
        t10_result = run_t10(n_molecules=args.n_molecules, verbose=args.verbose)


if __name__ == "__main__":
    main()
