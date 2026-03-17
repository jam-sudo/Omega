"""
Error Cancellation Analysis for Omega PBPK gold tier drugs.

Computes the Error Cancellation Index (CI) per drug:
    CI = |log(FE_cmax)| / sum(|log(FE_param_i)|)

CI < 0.5  → strong cancellation (fragile)
CI ≈ 1.0  → no cancellation (direct error propagation)

Usage:
    source .venv/bin/activate && python3 scripts/error_cancellation_analysis.py
"""

import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402

# ── Reference data ────────────────────────────────────────────────────────────

DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "clinical", "reference_database.json"
)
ADME_REF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "adme_reference.csv")
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "docs", "team", "findings_cancellation.json"
)

with open(DB_PATH) as f:
    db = json.load(f)

drugs_db = db["drugs"]

# Build ADME reference lookup: name → {fup, clint_3a4}
adme_ref: dict[str, dict] = {}
if os.path.exists(ADME_REF_PATH):
    adme_df = pd.read_csv(ADME_REF_PATH)
    for _, row in adme_df.iterrows():
        key = str(row["name"]).lower().strip()
        adme_ref[key] = {
            "fup": float(row["fup"]) if pd.notna(row["fup"]) else None,
            "clint_3a4": float(row["clint_3a4_uL_min_pmol"])
            if pd.notna(row["clint_3a4_uL_min_pmol"])
            else None,
        }

# ── Pipeline ──────────────────────────────────────────────────────────────────

pipeline = OmegaPipeline()

# ── Helper ────────────────────────────────────────────────────────────────────


def safe_fe(predicted: float | None, measured: float | None, label: str = "") -> float | None:
    """Return fold error = predicted/measured, or None if either is missing/invalid."""
    if predicted is None or measured is None:
        return None
    if measured <= 0 or predicted <= 0:
        return None
    return predicted / measured


def cancellation_index(fe_cmax: float, param_fes: list[float]) -> float | None:
    """CI = |log(FE_cmax)| / sum(|log(FE_param_i)|)."""
    log_cmax = abs(math.log(fe_cmax))
    log_params = sum(abs(math.log(fe)) for fe in param_fes if fe is not None)
    if log_params == 0:
        return None
    return log_cmax / log_params


def ref_cmax(drug_entry: dict) -> float | None:
    """Extract reference Cmax (mg/L) from pk_params or ct_curve max."""
    cmax = drug_entry.get("pk_params", {}).get("cmax_mg_L")
    if cmax and cmax > 0:
        return float(cmax)
    ct = drug_entry.get("ct_curve", {})
    conc = ct.get("conc_mg_L") or ct.get("concentration_mg_L")
    if conc:
        try:
            return float(max(conc))
        except (TypeError, ValueError):
            pass
    return None


# ── Main loop ─────────────────────────────────────────────────────────────────

gold_drugs = [
    (name, d)
    for name, d in drugs_db.items()
    if d.get("tier") == "gold" and d.get("smiles") and d.get("dose_mg") is not None
]

rows: list[dict] = []

print("=== Error Cancellation Analysis ===")
print(f"Gold tier drugs available: {len(gold_drugs)}")
print()
header = f"{'Drug':<20} | {'FE_Cmax':>7} | {'FE_fup':>7} | {'FE_CLint':>8} | {'CI':>6} | Status"
print(header)
print("-" * len(header))

for name, drug in sorted(gold_drugs):
    smiles = drug["smiles"]
    dose_mg = float(drug["dose_mg"])
    route = drug.get("route", "oral")

    cmax_ref = ref_cmax(drug)
    if cmax_ref is None:
        continue  # can't compute FE_cmax without reference

    # Run pipeline
    try:
        req = SimulationRequest(smiles=smiles, dose_mg=dose_mg, route=route)
        result = pipeline.simulate(req)
        cmax_pred = result.cmax_mg_L
        adme = result.adme_properties
    except Exception as exc:
        print(f"{'  ' + name:<20} | SIMULATION ERROR: {exc}")
        continue

    fe_cmax = safe_fe(cmax_pred, cmax_ref, "cmax")
    if fe_cmax is None:
        continue

    # ADME fold errors (predicted vs measured)
    adme_lookup = adme_ref.get(name.lower(), {})

    fe_fup = safe_fe(adme.get("fup"), adme_lookup.get("fup"), "fup")
    fe_clint = safe_fe(adme.get("clint_3a4"), adme_lookup.get("clint_3a4"), "clint")

    # Gather non-None param fold errors for CI
    param_fes = [fe for fe in [fe_fup, fe_clint] if fe is not None]
    ci_val = cancellation_index(fe_cmax, param_fes) if param_fes else None

    if ci_val is not None:
        status = "HIGH_CANCEL" if ci_val < 0.5 else "LOW_CANCEL"
    else:
        status = "NO_PARAMS"

    row = {
        "drug": name,
        "fe_cmax": round(fe_cmax, 3),
        "fe_fup": round(fe_fup, 3) if fe_fup else None,
        "fe_clint": round(fe_clint, 3) if fe_clint else None,
        "ci": round(ci_val, 3) if ci_val is not None else None,
        "status": status,
    }
    rows.append(row)

    fe_fup_str = f"{fe_fup:>7.2f}" if fe_fup else "    N/A"
    fe_clint_str = f"{fe_clint:>8.2f}" if fe_clint else "     N/A"
    ci_str = f"{ci_val:>6.3f}" if ci_val is not None else "   N/A"
    print(f"{name:<20} | {fe_cmax:>7.2f} | {fe_fup_str} | {fe_clint_str} | {ci_str} | {status}")

# ── Summary ───────────────────────────────────────────────────────────────────

print()
print("Summary:")
n_analyzed = len(rows)
n_with_ci = sum(1 for r in rows if r["ci"] is not None)
n_high_cancel = sum(1 for r in rows if r["status"] == "HIGH_CANCEL")
mean_ci = (
    sum(r["ci"] for r in rows if r["ci"] is not None) / n_with_ci if n_with_ci else float("nan")
)
mean_aafe = (
    sum(max(r["fe_cmax"], 1 / r["fe_cmax"]) for r in rows) / n_analyzed
    if n_analyzed
    else float("nan")
)

print(f"  Drugs analyzed:                  {n_analyzed}")
print(f"  Drugs with CI (has ADME ref):    {n_with_ci}")
print(f"  Strong cancellation (CI < 0.5):  {n_high_cancel} drugs")
print(f"  Mean CI (drugs with ADME ref):   {mean_ci:.3f}")
print(f"  Mean Cmax AAFE:                  {mean_aafe:.3f}")

# ── Save output ───────────────────────────────────────────────────────────────

output = {
    "analysis": "error_cancellation",
    "date": "2026-03-17",
    "n_drugs_analyzed": n_analyzed,
    "n_with_ci": n_with_ci,
    "n_high_cancellation": n_high_cancel,
    "mean_ci": round(mean_ci, 4) if not math.isnan(mean_ci) else None,
    "mean_cmax_aafe": round(mean_aafe, 4) if not math.isnan(mean_aafe) else None,
    "results": rows,
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {OUTPUT_PATH}")
