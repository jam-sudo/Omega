#!/usr/bin/env python
"""Generate PBPK features for MMPK drugs.

Task 3 of the MMPK Meta-Learner plan.
Runs each MMPK drug through the OmegaPipeline and extracts intermediate
features (cmax_pbpk, fup, clint, logP, TPSA, MW, compound_type, pgp_flag).
Output: data/ml/clinical/mmpk_pbpk_features.csv
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

from omega_pbpk.ml.applicability import check_applicability  # noqa: E402
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402
from omega_pbpk.prediction.pka_predictor import predict_pka  # noqa: E402


def main() -> None:
    input_path = REPO_ROOT / "data" / "ml" / "clinical" / "mmpk_clean.csv"
    output_path = REPO_ROOT / "data" / "ml" / "clinical" / "mmpk_pbpk_features.csv"

    # Read input data
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        drugs = list(reader)
    total = len(drugs)
    print(f"Loaded {total} drugs from {input_path.name}")

    # Initialize pipeline once
    pipeline = OmegaPipeline()
    pipeline._ensure_initialized()
    print("Pipeline initialized.")

    # Output columns
    fieldnames = [
        "name",
        "smiles",
        "dose_mg",
        "cmax_obs",
        "cmax_pbpk",
        "fup",
        "clint",
        "logP",
        "TPSA_norm",
        "MW_norm",
        "is_acid",
        "is_base",
        "pgp_flag",
        "in_platinum",
    ]

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_start = time.time()

    for i, drug in enumerate(drugs):
        name = drug["name"]
        smiles = drug["smiles"]
        dose_mg = float(drug["dose_mg"])
        cmax_obs = float(drug["cmax_mg_L"])
        in_platinum = drug["in_platinum"]

        try:
            # Run PBPK simulation
            req = SimulationRequest(smiles=smiles, dose_mg=dose_mg, route="oral")
            result = pipeline.simulate(req)

            # Extract PBPK features
            cmax_pbpk = result.cmax_mg_L
            adme = result.adme_properties
            fup = float(adme.get("fup", 0.1))
            clint = float(adme.get("clint_3a4", 5.0))

            # RDKit descriptors
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"RDKit cannot parse SMILES: {smiles}")

            logP_val = Descriptors.MolLogP(mol)
            tpsa_norm = Descriptors.TPSA(mol) / 200.0
            mw_norm = Descriptors.MolWt(mol) / 600.0

            # Compound type via pKa predictor
            is_acid = 0.0
            is_base = 0.0
            try:
                pka_result = predict_pka(smiles, molecule_type="neutral")
                if pka_result and pka_result.detected_group:
                    group = pka_result.detected_group
                    if group in ("carboxylic_acid", "phenol", "sulfonamide", "enol_lactone"):
                        is_acid = 1.0
                    elif group in (
                        "amine_primary",
                        "amine_secondary",
                        "pyridine",
                        "imidazole",
                    ):
                        is_base = 1.0
            except Exception:
                pass

            # P-gp flag via applicability check
            pgp_flag = 0.0
            try:
                app = check_applicability(smiles)
                if "pgp_substrate" in app.flags:
                    pgp_flag = 1.0
            except Exception:
                pass

            # Validate numeric values are finite
            for val_name, val in [
                ("cmax_pbpk", cmax_pbpk),
                ("fup", fup),
                ("clint", clint),
                ("logP", logP_val),
                ("TPSA_norm", tpsa_norm),
                ("MW_norm", mw_norm),
            ]:
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite value for {val_name}: {val}")

            rows.append(
                {
                    "name": name,
                    "smiles": smiles,
                    "dose_mg": dose_mg,
                    "cmax_obs": cmax_obs,
                    "cmax_pbpk": cmax_pbpk,
                    "fup": fup,
                    "clint": clint,
                    "logP": logP_val,
                    "TPSA_norm": tpsa_norm,
                    "MW_norm": mw_norm,
                    "is_acid": is_acid,
                    "is_base": is_base,
                    "pgp_flag": pgp_flag,
                    "in_platinum": in_platinum,
                }
            )
        except Exception as exc:
            failures.append((name, str(exc)))

        # Progress reporting
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{i + 1}/{total}] {len(rows)} ok, {len(failures)} fail "
                f"({elapsed:.1f}s, {rate:.1f} drugs/s)"
            )

    # Write output CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed_total = time.time() - t_start

    # Summary
    print(f"\n{'=' * 60}")
    print("PBPK feature generation complete")
    print(f"  Drugs processed: {len(rows)}/{total}")
    print(f"  Failures: {len(failures)}/{total} ({100 * len(failures) / total:.1f}%)")
    print(f"  Output: {output_path}")
    print(f"  Runtime: {elapsed_total:.1f}s ({elapsed_total / total:.3f}s/drug)")

    if rows:
        cmax_vals = [r["cmax_pbpk"] for r in rows]
        print(f"  cmax_pbpk range: [{min(cmax_vals):.4f}, {max(cmax_vals):.4f}]")
        print(f"  cmax_pbpk median: {np.median(cmax_vals):.4f}")

    if failures:
        print(f"\nFailure details ({len(failures)}):")
        for name, err in failures[:20]:
            print(f"  {name}: {err}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
