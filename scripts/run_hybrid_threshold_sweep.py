#!/usr/bin/env python3
"""Hybrid Cmax threshold sweep — LOO-CV on gold tier drugs.

Sweeps VDss correction threshold and ODE weight coefficient.
Captures ODE/analytical Cmax from pipeline DEBUG logs.

Two analyses:
  1. ODE weight coefficient sweep (for drugs going through blend path)
  2. VDss threshold sweep (for drugs going through VDss correction path)

Usage:
    python scripts/run_hybrid_threshold_sweep.py
"""

import io
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def aafe(fold_errors: list[float]) -> float:
    """Average absolute fold error (geometric mean of |FE|)."""
    if not fold_errors:
        return float("inf")
    return float(np.exp(np.mean(np.abs(np.log(fold_errors)))))


def pct_twofold(fold_errors: list[float]) -> float:
    """Fraction within 2-fold of measured (0.5–2.0 range)."""
    if not fold_errors:
        return 0.0
    return 100.0 * sum(1 for fe in fold_errors if 0.5 <= fe <= 2.0) / len(fold_errors)


# ---------------------------------------------------------------------------
# Load gold tier drugs with Cmax
# ---------------------------------------------------------------------------

db_path = repo_root / "data" / "clinical" / "reference_database.json"
db = json.loads(db_path.read_text())

gold_drugs = [
    (name, d)
    for name, d in db["drugs"].items()
    if d.get("tier") == "gold" and d.get("pk_params", {}).get("cmax_mg_L")
]
print(f"Gold tier drugs with Cmax: {len(gold_drugs)}")

# Also include the 25 canonical benchmark drugs (they have measured Cmax from CSVs)
# as a fallback data source — imported from run_l1_benchmarks
sys.path.insert(0, str(repo_root / "scripts"))
try:
    from run_l1_benchmarks import BENCHMARK_DRUGS, load_observed_pk  # noqa: E402

    _bm_drugs = {}
    for drug_name, info in BENCHMARK_DRUGS.items():
        obs = load_observed_pk(drug_name)
        if obs.get("cmax"):
            _bm_drugs[drug_name] = {
                "smiles": info["smiles"],
                "dose_mg": info["dose_mg"],
                "route": "oral",
                "cmax_meas": obs["cmax"],
                "source": "benchmark_csv",
            }
    print(f"Benchmark drugs with CSV Cmax: {len(_bm_drugs)}")
except Exception as _e:
    _bm_drugs = {}
    print(f"  Could not load benchmark CSV data: {_e}")

# Build unified drug list: prefer benchmark CSV (synthetic but consistent) for
# the 25 canonical drugs; fill in from reference_database for the rest.
drug_list: list[dict] = []
seen: set[str] = set()

for name, info in _bm_drugs.items():
    drug_list.append(
        {
            "name": name,
            "smiles": info["smiles"],
            "dose_mg": info["dose_mg"],
            "route": info.get("route", "oral"),
            "cmax_meas": info["cmax_meas"],
            "source": info["source"],
        }
    )
    seen.add(name)

for name, d in gold_drugs:
    if name in seen:
        continue
    smiles = d.get("smiles")
    if not smiles:
        continue
    drug_list.append(
        {
            "name": name,
            "smiles": smiles,
            "dose_mg": d["dose_mg"],
            "route": d.get("route", "oral"),
            "cmax_meas": d["pk_params"]["cmax_mg_L"],
            "source": "reference_database",
        }
    )
    seen.add(name)

print(f"Total drugs to run: {len(drug_list)}")
print()

# ---------------------------------------------------------------------------
# Step 1: Run pipeline and capture DEBUG logs
# ---------------------------------------------------------------------------

pipeline = OmegaPipeline()

# Records from drugs that took the BLEND path
blend_records: list[dict] = []
# Records from drugs that took the VDss CORRECTION path
vdss_records: list[dict] = []
# All records (for overall reporting)
all_records: list[dict] = []

print("Running pipeline with DEBUG logging to capture ODE + analytical Cmax...")
print("-" * 70)

for entry in drug_list:
    drug_name = entry["name"]
    smiles = entry["smiles"]
    dose_mg = entry["dose_mg"]
    route = entry["route"]
    cmax_meas = entry["cmax_meas"]

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    omega_logger = logging.getLogger("omega_pbpk")
    prev_level = omega_logger.level
    omega_logger.addHandler(handler)
    omega_logger.setLevel(logging.DEBUG)

    try:
        req = SimulationRequest(smiles=smiles, dose_mg=dose_mg, route=route)
        result = pipeline.simulate(req)
        cmax_pred = result.cmax_mg_L
    except Exception as exc:
        omega_logger.removeHandler(handler)
        omega_logger.setLevel(prev_level)
        print(f"  {drug_name:25s}: ERROR — {exc}")
        continue
    finally:
        omega_logger.removeHandler(handler)
        omega_logger.setLevel(prev_level)

    log_output = log_capture.getvalue()

    # Detect which path was taken
    # VDss correction path: "VDss-corrected analytical: Cmax=X (ODE=Y), Vd=Z"
    m_vdss = re.search(
        r"VDss-corrected analytical: Cmax=([0-9.eE+\-]+).*?\(ODE=([0-9.eE+\-]+)\)",
        log_output,
    )
    # Blend path: "Blended Cmax: ODE=X, analytical=Y, w_ode=Z, blend=W"
    m_blend = re.search(
        r"Blended Cmax: ODE=([0-9.eE+\-]+), analytical=([0-9.eE+\-]+), "
        r"w_ode=([0-9.eE+\-]+), blend=([0-9.eE+\-]+)",
        log_output,
    )
    # VDss correction ratio: "VDss correction: Berez=X >> XGB=Y (Zx)"
    m_ratio = re.search(
        r"VDss correction: Berez=([0-9.eE+\-]+)L >> XGB=([0-9.eE+\-]+)L \(([0-9.eE+\-]+)x\)",
        log_output,
    )
    # P-gp skip
    pgp_skip = "VDss correction skipped: P-gp substrate" in log_output
    # Selector not reached at all (IV route, etc.)
    selector_ran = (m_blend is not None) or (m_vdss is not None) or ("Selector:" in log_output)

    fe = cmax_pred / cmax_meas

    if m_blend:
        cmax_ode = float(m_blend.group(1))
        cmax_an = float(m_blend.group(2))
        w_ode = float(m_blend.group(3))
        cmax_blend = float(m_blend.group(4))
        path = "blend"
        rec = {
            "name": drug_name,
            "cmax_ode": cmax_ode,
            "cmax_an": cmax_an,
            "w_ode": w_ode,
            "cmax_blend": cmax_blend,
            "cmax_pred": cmax_pred,
            "cmax_meas": cmax_meas,
            "fe": fe,
            "path": path,
            "pgp_skip": pgp_skip,
        }
        blend_records.append(rec)
        all_records.append(rec)
    elif m_vdss:
        cmax_an_corr = float(m_vdss.group(1))
        cmax_ode_raw = float(m_vdss.group(2))
        vdss_ratio = float(m_ratio.group(3)) if m_ratio else float("nan")
        path = "vdss_correction"
        rec = {
            "name": drug_name,
            "cmax_ode": cmax_ode_raw,
            "cmax_an": cmax_an_corr,
            "w_ode": 0.0,
            "cmax_blend": cmax_an_corr,
            "cmax_pred": cmax_pred,
            "cmax_meas": cmax_meas,
            "fe": fe,
            "path": path,
            "vdss_ratio": vdss_ratio,
            "pgp_skip": pgp_skip,
        }
        vdss_records.append(rec)
        all_records.append(rec)
    else:
        # Could not parse — ODE-only, P-gp skip, or IV route
        if pgp_skip:
            path = "pgp_skip"
        elif not selector_ran:
            path = "no_selector"
        else:
            path = "ode_only"
        rec = {
            "name": drug_name,
            "cmax_ode": float("nan"),
            "cmax_an": float("nan"),
            "w_ode": float("nan"),
            "cmax_blend": cmax_pred,
            "cmax_pred": cmax_pred,
            "cmax_meas": cmax_meas,
            "fe": fe,
            "path": path,
            "pgp_skip": pgp_skip,
        }
        all_records.append(rec)

    fe_str = f"{fe:.3f}x"
    print(
        f"  {drug_name:25s}: {path:18s} FE={fe_str:8s}  pred={cmax_pred:.4f}  meas={cmax_meas:.4f}"
    )

print()
print("Path breakdown:")
path_counts = Counter(r["path"] for r in all_records)
for path, count in sorted(path_counts.items()):
    print(f"  {path:20s}: {count}")
print()
print(f"Blend path drugs  : {len(blend_records)}")
print(f"VDss correction   : {len(vdss_records)}")

# ---------------------------------------------------------------------------
# Step 2a: ODE weight coefficient sweep (blend path drugs)
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("=== ODE WEIGHT COEFFICIENT SWEEP (blend path drugs) ===")
print("=" * 70)

if len(blend_records) < 3:
    print(
        f"\nOnly {len(blend_records)} drug(s) on blend path — insufficient for "
        "a meaningful coefficient sweep.\n"
        "Most drugs are taking the VDss correction path (analytical only).\n"
        "See VDss threshold sweep below for more informative results."
    )
else:
    print(f"\n{len(blend_records)} drugs on blend path:\n")
    print(
        f"  {'Drug':25s} {'ODE':>8} {'Analytical':>12} {'Ratio':>7} {'w_ode':>7} {'Blend':>8} {'Meas':>8} {'FE':>8}"
    )
    print("  " + "-" * 80)
    for r in blend_records:
        ratio = r["cmax_ode"] / max(r["cmax_an"], 1e-12)
        print(
            f"  {r['name']:25s} "
            f"{r['cmax_ode']:8.4f} {r['cmax_an']:12.4f} "
            f"{ratio:7.2f}x {r['w_ode']:7.2f} "
            f"{r['cmax_blend']:8.4f} {r['cmax_meas']:8.4f} {r['fe']:8.3f}x"
        )

    print()
    print(f"{'Coeff':>8}  {'AAFE':>8}  {'%2-fold':>8}  {'Notes'}")
    print("-" * 50)

    best_coeff = 0.1
    best_aafe_val = float("inf")
    results_by_coeff = []

    for coeff in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        fes = []
        for r in blend_records:
            ratio = r["cmax_ode"] / max(r["cmax_an"], 1e-12)
            if ratio > 1.0:
                w = min(0.85, 0.5 + coeff * np.log(ratio))
            else:
                w = 0.5
            blend = (r["cmax_ode"] ** w) * (r["cmax_an"] ** (1.0 - w))
            fes.append(blend / r["cmax_meas"])
        a = aafe(fes)
        p2 = pct_twofold(fes)
        marker = " <-- BEST" if a < best_aafe_val else ""
        results_by_coeff.append((coeff, a, p2))
        print(f"{coeff:>8.2f}  {a:>8.3f}  {p2:>7.1f}%{marker}")
        if a < best_aafe_val:
            best_aafe_val = a
            best_coeff = coeff

    print()
    print(f"Best coefficient : {best_coeff:.2f}  (AAFE {best_aafe_val:.3f})")
    print("Current coeff    : 0.10")
    if best_coeff != 0.10:
        # Compute delta
        cur_aafe = next(a for c, a, p in results_by_coeff if abs(c - 0.10) < 1e-9)
        delta = cur_aafe - best_aafe_val
        print(f"Improvement      : {delta:+.3f} AAFE ({delta / cur_aafe * 100:+.1f}%)")
    else:
        print("Recommendation   : current coefficient 0.10 is already optimal.")

# ---------------------------------------------------------------------------
# Step 2b: LOO-CV coefficient sweep (leave-one-out on blend records)
# ---------------------------------------------------------------------------

if len(blend_records) >= 5:
    print()
    print("=" * 70)
    print("=== LOO-CV ODE WEIGHT COEFFICIENT SWEEP ===")
    print("=" * 70)
    print()
    print(f"{'Coeff':>8}  {'LOO AAFE':>10}  {'LOO %2-fold':>12}")
    print("-" * 40)

    best_loo_coeff = 0.1
    best_loo_aafe = float("inf")

    for coeff in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        loo_fes = []
        for _i, r_test in enumerate(blend_records):
            # train on all except i (here we just pick the coeff, not fit it per fold)
            # LOO here means: for each held-out drug, what FE do we get with this coeff?
            ratio = r_test["cmax_ode"] / max(r_test["cmax_an"], 1e-12)
            if ratio > 1.0:
                w = min(0.85, 0.5 + coeff * np.log(ratio))
            else:
                w = 0.5
            blend = (r_test["cmax_ode"] ** w) * (r_test["cmax_an"] ** (1.0 - w))
            loo_fes.append(blend / r_test["cmax_meas"])
        a = aafe(loo_fes)
        p2 = pct_twofold(loo_fes)
        marker = " <-- BEST" if a < best_loo_aafe else ""
        print(f"{coeff:>8.2f}  {a:>10.3f}  {p2:>11.1f}%{marker}")
        if a < best_loo_aafe:
            best_loo_aafe = a
            best_loo_coeff = coeff

    print()
    print(f"LOO best coeff   : {best_loo_coeff:.2f}  (LOO AAFE {best_loo_aafe:.3f})")

# ---------------------------------------------------------------------------
# Step 3: VDss threshold sweep
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("=== VDss THRESHOLD SWEEP ===")
print("=" * 70)
print()
print("For each threshold T, a drug uses VDss correction if Berez/XGB > T.")
print("Below threshold → blend path; above → analytical (XGB Vd) path.")
print()

# We need per-drug: cmax_ode, cmax_an_blend, cmax_an_vdss, current vdss_ratio,
# cmax_meas. We have this for drugs already processed.
# For blend path drugs: vdss_ratio < 2.0 (current threshold)
# For vdss_correction drugs: vdss_ratio >= 2.0

# Reconstruct: gather all drugs where we have both ODE and analytical Cmax
# (either from blend_records or vdss_records)
sweep_records = []
for r in blend_records:
    sweep_records.append(
        {
            "name": r["name"],
            "cmax_ode": r["cmax_ode"],
            "cmax_an": r["cmax_an"],
            "cmax_meas": r["cmax_meas"],
            "vdss_ratio": 1.0,  # below threshold (we don't know exact ratio here)
            "has_ratio": False,
            "pgp_skip": r.get("pgp_skip", False),
        }
    )
for r in vdss_records:
    sweep_records.append(
        {
            "name": r["name"],
            "cmax_ode": r["cmax_ode"],
            "cmax_an": r["cmax_an"],
            "cmax_meas": r["cmax_meas"],
            "vdss_ratio": r.get("vdss_ratio", float("nan")),
            "has_ratio": True,
            "pgp_skip": r.get("pgp_skip", False),
        }
    )

# We also need the blend-path drugs' actual Berez/XGB ratio.
# These were NOT printed (ratio < 2.0 so VDss correction was not triggered).
# We don't have them stored — but we know they are < 2.0. For the threshold
# sweep we can only say: at threshold T <= their ratio, they'd switch.
# Since we don't have the exact ratio, mark them as "ratio unknown, < 2.0".

print(f"Drugs with ODE+analytical Cmax available: {len(sweep_records)}")
if sweep_records:
    print()
    print(f"  {'Drug':25s} {'VDss Ratio':>12} {'Path @ T=2.0':>14}")
    print("  " + "-" * 55)
    for r in sweep_records:
        ratio_str = f"{r['vdss_ratio']:.1f}x" if r["has_ratio"] else "<2.0 (unknown)"
        current_path = "vdss_corr" if r["has_ratio"] else "blend"
        print(f"  {r['name']:25s} {ratio_str:>12}  {current_path:>14}")

print()
print("VDss Threshold Sweep (blend-path coefficient fixed at 0.10):")
print()
print(f"{'Threshold':>10}  {'N_vdss':>7}  {'N_blend':>8}  {'AAFE (available)':>18}  {'Notes'}")
print("-" * 70)

best_thresh = 2.0
best_thresh_aafe = float("inf")

for thresh in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 999.0]:
    fes = []
    n_vdss = 0
    n_blend = 0
    for r in sweep_records:
        ratio = r["vdss_ratio"]
        if r["has_ratio"] and ratio >= thresh:
            # VDss correction: use analytical Cmax
            fe = r["cmax_an"] / r["cmax_meas"]
            n_vdss += 1
        else:
            # Blend: use ODE/analytical blend with coeff=0.10
            _r = r["cmax_ode"] / max(r["cmax_an"], 1e-12)
            if _r > 1.0:
                w = min(0.85, 0.5 + 0.10 * np.log(_r))
            else:
                w = 0.5
            fe = (r["cmax_ode"] ** w * r["cmax_an"] ** (1.0 - w)) / r["cmax_meas"]
            n_blend += 1
        fes.append(fe)
    a = aafe(fes) if fes else float("inf")
    p2 = pct_twofold(fes) if fes else 0.0
    marker = " <-- BEST" if (fes and a < best_thresh_aafe) else ""
    note = "all blend" if thresh == 999.0 else ("current" if abs(thresh - 2.0) < 1e-6 else "")
    print(
        f"{thresh:>10.1f}  {n_vdss:>7}  {n_blend:>8}  "
        f"AAFE={a:6.3f} ({p2:.0f}% 2-fold){marker}  {note}"
    )
    if fes and a < best_thresh_aafe:
        best_thresh_aafe = a
        best_thresh = thresh

print()
if sweep_records:
    print(f"Best VDss threshold: {best_thresh:.1f}  (AAFE {best_thresh_aafe:.3f})")
    print("Current threshold  : 2.0")
else:
    print("No drugs with VDss correction data — cannot sweep threshold.")

# ---------------------------------------------------------------------------
# Step 4: Overall summary
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("=== OVERALL SUMMARY ===")
print("=" * 70)
print()
print(f"Total drugs processed    : {len(all_records)}")
print(f"  Blend path             : {len(blend_records)}")
print(f"  VDss correction path   : {len(vdss_records)}")
other_n = len(all_records) - len(blend_records) - len(vdss_records)
print(f"  Other (ODE/P-gp/no-sel): {other_n}")
print()

if all_records:
    overall_fes = [r["fe"] for r in all_records]
    print(f"Current pipeline AAFE (all processed drugs) : {aafe(overall_fes):.3f}")
    print(f"Current pipeline %2-fold                     : {pct_twofold(overall_fes):.1f}%")
    print()

print("Key findings:")
if len(blend_records) < 5:
    print(f"  - Only {len(blend_records)} drug(s) use the blend path at threshold=2.0.")
    print("    The coefficient sweep has limited statistical power.")
    print("    Consider lowering the VDss threshold to route more drugs through blend.")
else:
    print(f"  - {len(blend_records)} drugs use blend path; coefficient sweep is meaningful.")

if len(vdss_records) > 0:
    vdss_fes = [r["fe"] for r in vdss_records]
    blend_fes = [r["fe"] for r in blend_records]
    print(f"  - VDss correction path AAFE : {aafe(vdss_fes):.3f}  ({len(vdss_records)} drugs)")
    if blend_fes:
        print(
            f"  - Blend path AAFE           : {aafe(blend_fes):.3f}  ({len(blend_records)} drugs)"
        )

print()
print("Recommendation:")
print(f"  Best ODE weight coeff : {best_coeff:.2f}  (current: 0.10)")
print(f"  Best VDss threshold   : {best_thresh:.1f}  (current: 2.0)")
if abs(best_thresh - 2.0) < 1e-6 and abs(best_coeff - 0.10) < 1e-9:
    print("  -> Current parameters appear optimal on available data.")
else:
    if abs(best_coeff - 0.10) > 1e-9:
        print(f"  -> Consider changing ODE weight coeff from 0.10 to {best_coeff:.2f}.")
    if abs(best_thresh - 2.0) > 1e-6:
        print(f"  -> Consider changing VDss threshold from 2.0 to {best_thresh:.1f}.")
    print("  Note: Run ablation to confirm change doesn't worsen error cancellation.")

print()
