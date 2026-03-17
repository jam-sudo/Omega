#!/usr/bin/env python3
"""Compare analytical 1-compartment model vs full 35-state ODE for Cmax prediction."""

import io
import logging
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts"))

from run_l1_benchmarks import BENCHMARK_DRUGS, compute_fold_error, load_observed_pk  # noqa: E402

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402

# Capture debug logs
log_capture = io.StringIO()
handler = logging.StreamHandler(log_capture)
handler.setLevel(logging.DEBUG)
logging.getLogger("omega_pbpk.pipeline").addHandler(handler)
logging.getLogger("omega_pbpk.pipeline").setLevel(logging.DEBUG)

pipeline = OmegaPipeline()

results = []
for drug_name, info in BENCHMARK_DRUGS.items():
    log_capture.truncate(0)
    log_capture.seek(0)

    try:
        result = pipeline.simulate(
            SimulationRequest(
                smiles=info["smiles"],
                dose_mg=info["dose_mg"],
                route="oral",
                duration_h=24.0,
            )
        )
    except Exception as e:
        print(f"  ERROR simulating {drug_name}: {e}")
        continue

    logs = log_capture.getvalue()

    # Parse ODE and analytical from logs
    ode_cmax = None
    analytical_cmax = None
    for line in logs.split("\n"):
        if "Blended Cmax: ODE=" in line:
            # "Blended Cmax: ODE=%.4f, analytical=%.4f, w_ode=%.2f, blend=%.4f"
            parts = line.split("ODE=")[1]
            ode_cmax = float(parts.split(",")[0])
            analytical_cmax = float(parts.split("analytical=")[1].split(",")[0])
        elif "VDss-corrected analytical" in line:
            # "VDss-corrected analytical: Cmax=%.4f (ODE=%.4f), Vd=%.0fL"
            analytical_cmax = float(line.split("Cmax=")[1].split(" ")[0].rstrip(","))
            ode_cmax = float(line.split("ODE=")[1].split(")")[0].rstrip(","))

    obs = load_observed_pk(drug_name)
    obs_cmax = obs.get("cmax", 0)

    results.append(
        {
            "drug": drug_name,
            "dose": info["dose_mg"],
            "obs": obs_cmax,
            "ode": ode_cmax,
            "analytical": analytical_cmax,
            "final": result.cmax_mg_L,
        }
    )

# Print comparison table
print(
    f"\n{'Drug':20s} {'Dose':>5s} {'Obs':>10s} {'ODE':>10s} {'Analyt':>10s} {'Final':>10s} {'FE_ODE':>8s} {'FE_Ana':>8s} {'FE_Fin':>8s}"
)
print("-" * 100)
ode_fes = []
ana_fes = []
fin_fes = []
for r in sorted(
    results,
    key=lambda x: (
        -(compute_fold_error(x.get("ode") or 0, x["obs"]) if x.get("ode") and x["obs"] > 0 else 0)
    ),
):
    obs = r["obs"]
    fe_ode = compute_fold_error(r["ode"], obs) if r["ode"] and obs > 0 else 0
    fe_ana = compute_fold_error(r["analytical"], obs) if r["analytical"] and obs > 0 else 0
    fe_fin = compute_fold_error(r["final"], obs) if obs > 0 else 0
    if fe_ode > 0 and not np.isnan(fe_ode):
        ode_fes.append(fe_ode)
    if fe_ana > 0 and not np.isnan(fe_ana):
        ana_fes.append(fe_ana)
    if fe_fin > 0 and not np.isnan(fe_fin):
        fin_fes.append(fe_fin)

    ode_str = f"{r['ode']:10.4f}" if r["ode"] is not None else "      None"
    ana_str = f"{r['analytical']:10.4f}" if r["analytical"] is not None else "      None"
    fe_ode_str = f"{fe_ode:7.2f}x" if fe_ode > 0 else "     N/A"
    fe_ana_str = f"{fe_ana:7.2f}x" if fe_ana > 0 else "     N/A"
    fe_fin_str = f"{fe_fin:7.2f}x" if fe_fin > 0 else "     N/A"

    print(
        f"{r['drug']:20s} {r['dose']:5.0f} {obs:10.4f} {ode_str} {ana_str} {r['final']:10.4f} {fe_ode_str} {fe_ana_str} {fe_fin_str}"
    )

print("\n" + "=" * 60)


def aafe(fes):
    return float(np.exp(np.mean(np.log([f for f in fes if f > 0])))) if fes else 0


def pct2(fes):
    return 100 * sum(1 for f in fes if f <= 2.0) / len(fes) if fes else 0


def median_fe(fes):
    return float(np.median(fes)) if fes else 0


print(f"{'':20s} {'ODE Raw':>15s} {'Analytical':>15s} {'Full Pipeline':>15s}")
print(f"{'N drugs':20s} {len(ode_fes):15d} {len(ana_fes):15d} {len(fin_fes):15d}")
print(f"{'AAFE':20s} {aafe(ode_fes):15.3f} {aafe(ana_fes):15.3f} {aafe(fin_fes):15.3f}")
print(
    f"{'Median FE':20s} {median_fe(ode_fes):15.3f} {median_fe(ana_fes):15.3f} {median_fe(fin_fes):15.3f}"
)
print(f"{'%2-fold':20s} {pct2(ode_fes):14.0f}% {pct2(ana_fes):14.0f}% {pct2(fin_fes):14.0f}%")

# Show which model wins per drug
print("\n" + "=" * 60)
print("Per-drug winner (ODE vs Analytical):")
ode_wins = 0
ana_wins = 0
ties = 0
for r in results:
    obs = r["obs"]
    if r["ode"] and r["analytical"] and obs > 0:
        fe_ode = compute_fold_error(r["ode"], obs)
        fe_ana = compute_fold_error(r["analytical"], obs)
        if not np.isnan(fe_ode) and not np.isnan(fe_ana):
            if abs(fe_ode - fe_ana) < 0.05:
                ties += 1
                winner = "TIE"
            elif fe_ode < fe_ana:
                ode_wins += 1
                winner = "ODE"
            else:
                ana_wins += 1
                winner = "ANA"
            print(f"  {r['drug']:20s} ODE={fe_ode:.2f}x  ANA={fe_ana:.2f}x  -> {winner}")
print(f"\nODE wins: {ode_wins}, Analytical wins: {ana_wins}, Ties: {ties}")
