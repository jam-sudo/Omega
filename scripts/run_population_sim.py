"""Phase 5.4 — Population Pharmacokinetic Simulation.

Simulates N=100 virtual subjects with varying physiological and genetic
parameters to generate PK variability distributions for CYP substrate drugs.

Population variability sources:
  - CYP3A4 activity: log-normal(mean=1.0, CV=60%) for CYP3A4 substrates
  - CYP2D6 activity: mixture model (PM 20%, EM 70%, UM 10%) for CYP2D6 substrates
  - Body weight: normal(70 kg, SD=15) clipped [40, 120] kg

Injection approach:
  Monkey-patching OmegaPipeline:
    - _adme_predictor → FixedADMEPredictor returning pre-computed base props
    - _clint_predictor → ScaledCLintPredictor applying (CYP_multiplier × BW_factor)
    - subject_weight_kg in SimulationRequest → allometric Vd scaling via ODE

Usage::

    python scripts/run_population_sim.py --drug midazolam --n-subjects 100
    python scripts/run_population_sim.py --drug tramadol --n-subjects 100 --seed 99

Output JSON:
  results/population_sim_<drug>_N<n>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Ensure src/ is on path when running from project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("population_sim")

# ---------------------------------------------------------------------------
# Known CYP substrate drugs — SMILES and dosing from reference_database.json
# ---------------------------------------------------------------------------
DRUG_REGISTRY: dict[str, dict[str, Any]] = {
    "midazolam": {
        "smiles": "CC1=NC=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4F",
        "dose_mg": 7.5,
        "route": "oral",
        "primary_cyp": "CYP3A4",
        "description": "CYP3A4 substrate (benzodiazepine)",
        "ref_cmax_cv_pct": 80.0,  # published population CV% (Gorski 2003)
    },
    "tramadol": {
        "smiles": "CN(C)CC1CCCCC1(C2=CC(=CC=C2)OC)O",
        "dose_mg": 100.0,
        "route": "oral",
        "primary_cyp": "CYP2D6",
        "description": "CYP2D6 substrate (opioid analgesic)",
        "ref_cmax_cv_pct": 50.0,  # published approx
    },
    "metoprolol": {
        "smiles": "CC(C)NCC(COC1=CC=C(C=C1)CCOC)O",
        "dose_mg": 200.0,
        "route": "oral",
        "primary_cyp": "CYP2D6",
        "description": "CYP2D6 substrate (beta-blocker)",
        "ref_cmax_cv_pct": 60.0,
    },
}


# ---------------------------------------------------------------------------
# Stub predictor classes for monkey-patching the pipeline
# ---------------------------------------------------------------------------


class FixedADMEPredictor:
    """Returns a pre-computed, fixed ADMEProperties (ignores SMILES input).

    Used to bypass the ensemble predictor and inject scaled ADME properties
    for each virtual subject.
    """

    def __init__(self, props: Any) -> None:
        """
        Args:
            props: ADMEProperties object to return for any SMILES.
        """
        self._props = props

    def predict(self, smiles: str) -> Any:  # noqa: ARG002
        return self._props

    def _ensure_initialized(self) -> None:
        pass


class ScaledCLintPredictor:
    """Wraps XGBoostCLintPredictor and applies a scalar multiplier.

    Used to inject per-subject CYP activity variation into the IVIVE chain.
    The multiplier incorporates both enzyme activity and body-weight allometric
    scaling: effective_multiplier = cyp_activity × (BW/70)^0.75.
    """

    def __init__(self, base_predictor: Any, multiplier: float) -> None:
        """
        Args:
            base_predictor: Original XGBoostCLintPredictor instance.
            multiplier: Scalar to apply to the base CLint prediction.
        """
        self._base = base_predictor
        self._multiplier = multiplier

    def predict_clint(self, smiles: str) -> float:
        """Return base CLint × multiplier, clipped to [0.1, 3000]."""
        base = self._base.predict_clint(smiles)
        return float(np.clip(base * self._multiplier, 0.1, 3000.0))


# ---------------------------------------------------------------------------
# Population parameter sampling
# ---------------------------------------------------------------------------


@dataclass
class VirtualSubject:
    """Parameters for a single virtual subject."""

    subject_id: int
    body_weight_kg: float
    cyp3a4_activity: float  # multiplier relative to EM (1.0 = EM)
    cyp2d6_activity: float  # multiplier relative to EM (1.0 = EM)
    cyp2d6_phenotype: str  # PM / EM / UM


def sample_population(n_subjects: int, seed: int = 42) -> list[VirtualSubject]:
    """Sample N virtual subjects from physiologically plausible distributions.

    CYP3A4:
      log-normal with geometric mean 1.0 and CV=60%.
      CV=60% → log-normal sigma_ln = sqrt(ln(1 + 0.6^2)) ≈ 0.472.
      This reflects the ~100-fold range from poor to ultra-rapid CYP3A4 activity
      observed in vivo (Thummel & Wilkinson, Annu Rev Pharmacol 1998).

    CYP2D6 mixture model:
      PM (poor metabolizer):  20% of population, activity = 0.05 (near-zero)
      EM (extensive):         70% of population, activity = 1.0
      UM (ultra-rapid):       10% of population, activity = 2.5
      Frequencies from Gaedigk et al. (Clin Pharmacol Ther 2008).

    Body weight:
      Normal(mean=70 kg, SD=15 kg) clipped to [40, 120] kg.
      SD=15 reflects North American adult population (NHANES).

    Args:
        n_subjects: Number of virtual subjects.
        seed: Random seed for reproducibility.

    Returns:
        List of VirtualSubject dataclass instances.
    """
    rng = np.random.default_rng(seed)

    # Body weight
    bw = rng.normal(loc=70.0, scale=15.0, size=n_subjects)
    bw = np.clip(bw, 40.0, 120.0)

    # CYP3A4: log-normal, CV=60%
    cv3a4 = 0.60
    sigma_ln = np.sqrt(np.log(1.0 + cv3a4**2))
    # Parametrise so geometric mean = 1.0 (i.e., log-normal mean_ln = -sigma_ln^2/2)
    mu_ln = -0.5 * sigma_ln**2
    cyp3a4 = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n_subjects)
    cyp3a4 = np.clip(cyp3a4, 0.02, 8.0)  # realistic physiological range

    # CYP2D6: mixture model
    # Assign phenotype then activity
    u = rng.uniform(size=n_subjects)
    cyp2d6_activity = np.empty(n_subjects)
    cyp2d6_phenotype: list[str] = []
    for i, ui in enumerate(u):
        if ui < 0.20:
            cyp2d6_activity[i] = 0.05  # PM: near-zero activity
            cyp2d6_phenotype.append("PM")
        elif ui < 0.90:
            cyp2d6_activity[i] = 1.0  # EM: normal
            cyp2d6_phenotype.append("EM")
        else:
            cyp2d6_activity[i] = 2.5  # UM: ultra-rapid
            cyp2d6_phenotype.append("UM")

    subjects = []
    for i in range(n_subjects):
        subjects.append(
            VirtualSubject(
                subject_id=i + 1,
                body_weight_kg=float(bw[i]),
                cyp3a4_activity=float(cyp3a4[i]),
                cyp2d6_activity=float(cyp2d6_activity[i]),
                cyp2d6_phenotype=cyp2d6_phenotype[i],
            )
        )
    return subjects


# ---------------------------------------------------------------------------
# Core simulation logic
# ---------------------------------------------------------------------------


def _build_pipeline() -> Any:
    """Build and initialize OmegaPipeline, returning it ready-to-use."""
    from omega_pbpk.pipeline import OmegaPipeline

    pipe = OmegaPipeline()
    pipe._ensure_initialized()
    return pipe


def _get_base_adme(pipe: Any, smiles: str) -> Any:
    """Run the ADME predictor once to get baseline properties."""
    warnings: list = []
    return pipe._predict_adme(smiles, warnings)


def simulate_subject(
    base_pipe: Any,
    base_adme_props: Any,
    base_clint_predictor: Any,
    subject: VirtualSubject,
    drug_info: dict[str, Any],
) -> dict[str, Any]:
    """Run a single virtual subject simulation.

    Injection approach:
      1. FixedADMEPredictor: returns the base ADMEProperties for any SMILES call,
         bypassing re-prediction (only CLint is subject-specific via _clint_predictor).
      2. ScaledCLintPredictor: wraps the base XGBoostCLintPredictor and applies
         effective_multiplier = cyp_activity × (BW/70)^0.75.
         This scales the hepatocyte CLint before IVIVE, reflecting that:
         - CYP activity directly scales metabolic capacity (linear)
         - Body weight scales enzyme mass allometrically (exponent 0.75)
      3. subject_weight_kg in SimulationRequest: the ODE engine's WholeBodyPBPK
         uses this to scale organ volumes, cardiac output, and blood flow,
         providing proper Vd allometric scaling (Vd ∝ BW^1.0 implicitly via
         organ volume scaling in the 35-compartment model).

    The combination captures the three dominant sources of PK variability:
    CYP activity (→ CLint → CL), body composition (→ Vd), and gut/hepatic
    extraction balance (the hybrid Cmax selector re-evaluates for each subject).

    Args:
        base_pipe: Initialized OmegaPipeline instance.
        base_adme_props: Pre-computed base ADMEProperties dict from _predict_adme.
        base_clint_predictor: Original XGBoostCLintPredictor (unscaled).
        subject: VirtualSubject parameters.
        drug_info: Drug registry entry with SMILES, dose, primary_cyp.

    Returns:
        Dict with subject_id, cmax_mg_L, auc_mg_h_L, t_half_h, body_weight_kg,
        cyp3a4_activity, cyp2d6_activity, cyp2d6_phenotype.
    """
    from omega_pbpk.pipeline import SimulationRequest
    from omega_pbpk.prediction.adme_predictor import ADMEProperties

    bw = subject.body_weight_kg
    bw_factor = (bw / 70.0) ** 0.75  # allometric exponent for CL

    # Select enzyme multiplier based on drug's primary CYP substrate
    primary_cyp = drug_info["primary_cyp"]
    if primary_cyp == "CYP3A4":
        cyp_activity = subject.cyp3a4_activity
    else:
        # CYP2D6 substrate
        cyp_activity = subject.cyp2d6_activity

    effective_multiplier = cyp_activity * bw_factor

    # --- Build subject-specific ADMEProperties from the base dict ---
    # Only CLint varies (via ScaledCLintPredictor); fup/rbp/logP are population mean
    base_fup = float(base_adme_props.get("fup", 0.1))
    base_rbp = float(base_adme_props.get("rbp", 1.0))
    base_logp = float(base_adme_props.get("logP", 2.0))
    base_logs = float(base_adme_props.get("logS", -3.0))
    base_peff = float(base_adme_props.get("peff", 1.0))
    base_clint3a4 = float(base_adme_props.get("clint_3a4", 5.0))
    base_clint2d6 = float(base_adme_props.get("clint_2d6", 0.5))

    fixed_props = ADMEProperties(
        mw=float(base_adme_props.get("mw", 300.0)),
        logP=base_logp,
        logS=base_logs,
        peff=base_peff,
        fup=base_fup,
        rbp=base_rbp,
        clint_3a4=base_clint3a4,
        clint_2d6=base_clint2d6,
        herg_ic50_uM=float(base_adme_props.get("herg_ic50_uM", 10.0)),
        confidence=base_adme_props.get("confidence", "medium"),
        fup_lo=float(base_adme_props.get("fup_lo", base_fup * 0.5)),
        fup_hi=float(base_adme_props.get("fup_hi", min(base_fup * 2.0, 1.0))),
        clint_3a4_lo=float(base_adme_props.get("clint_3a4_lo", base_clint3a4 / 11.24)),
        clint_3a4_hi=float(base_adme_props.get("clint_3a4_hi", base_clint3a4 * 11.24)),
        peff_lo=float(base_adme_props.get("peff_lo", base_peff * 0.5)),
        peff_hi=float(base_adme_props.get("peff_hi", base_peff * 2.0)),
        rbp_lo=float(base_adme_props.get("rbp_lo", max(0.5, base_rbp * 0.8))),
        rbp_hi=float(base_adme_props.get("rbp_hi", min(3.0, base_rbp * 1.2))),
    )

    # Monkey-patch: inject fixed ADME + scaled CLint
    original_adme = base_pipe._adme_predictor
    original_clint = base_pipe._clint_predictor

    base_pipe._adme_predictor = FixedADMEPredictor(fixed_props)
    if base_clint_predictor is not None:
        base_pipe._clint_predictor = ScaledCLintPredictor(
            base_clint_predictor, effective_multiplier
        )

    try:
        req = SimulationRequest(
            smiles=drug_info["smiles"],
            dose_mg=drug_info["dose_mg"],
            route=drug_info["route"],
            duration_h=24.0,
            subject_weight_kg=bw,
        )
        result = base_pipe.simulate(req)
        cmax = float(result.cmax_mg_L)
        auc = float(result.auc0t_mg_h_L)
        t_half = float(result.t_half_h) if not np.isnan(result.t_half_h) else float("nan")
    finally:
        # Always restore original predictors
        base_pipe._adme_predictor = original_adme
        base_pipe._clint_predictor = original_clint

    return {
        "subject_id": subject.subject_id,
        "body_weight_kg": round(bw, 1),
        "cyp3a4_activity": round(subject.cyp3a4_activity, 4),
        "cyp2d6_activity": round(subject.cyp2d6_activity, 4),
        "cyp2d6_phenotype": subject.cyp2d6_phenotype,
        "effective_cl_multiplier": round(effective_multiplier, 4),
        "cmax_mg_L": round(cmax, 6),
        "auc_mg_h_L": round(auc, 6),
        "t_half_h": round(t_half, 3) if not np.isnan(t_half) else None,
    }


def run_population_simulation(
    drug_name: str,
    n_subjects: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run population PK simulation for a given drug.

    Args:
        drug_name: Key in DRUG_REGISTRY.
        n_subjects: Number of virtual subjects.
        seed: Random seed.
        verbose: Print progress and summary.

    Returns:
        Summary dict with population statistics and per-subject results.

    Raises:
        KeyError: If drug_name not found in DRUG_REGISTRY.
    """
    if drug_name not in DRUG_REGISTRY:
        available = ", ".join(sorted(DRUG_REGISTRY.keys()))
        raise KeyError(f"Unknown drug '{drug_name}'. Available: {available}")

    drug_info = DRUG_REGISTRY[drug_name]
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Population PK Simulation: {drug_name.upper()}")
        print(f"  {drug_info['description']}")
        print(f"  Primary CYP: {drug_info['primary_cyp']}")
        print(f"  Dose: {drug_info['dose_mg']} mg {drug_info['route']}")
        print(f"  N = {n_subjects} virtual subjects | seed = {seed}")
        print(f"{'=' * 60}")

    t0 = time.time()

    # Build pipeline once (expensive — loads XGBoost models)
    if verbose:
        print("Initializing pipeline...", flush=True)
    pipe = _build_pipeline()

    # Get base ADME prediction once
    base_adme = _get_base_adme(pipe, drug_info["smiles"])
    if verbose:
        print(
            f"Base ADME: fup={base_adme.get('fup'):.3f}, "
            f"clint_3a4={base_adme.get('clint_3a4'):.2f} µL/min/pmol, "
            f"logP={base_adme.get('logP'):.2f}"
        )

    # Keep reference to the real CLint predictor before we patch
    base_clint_predictor = pipe._clint_predictor

    # Sample virtual population
    subjects = sample_population(n_subjects, seed=seed)

    # Count phenotype breakdown
    phenotypes = [s.cyp2d6_phenotype for s in subjects]
    pm_count = phenotypes.count("PM")
    em_count = phenotypes.count("EM")
    um_count = phenotypes.count("UM")
    if verbose:
        print(
            f"Population: BW {np.mean([s.body_weight_kg for s in subjects]):.1f} "
            f"± {np.std([s.body_weight_kg for s in subjects]):.1f} kg | "
            f"CYP2D6: {pm_count} PM / {em_count} EM / {um_count} UM"
        )
        print(f"Simulating {n_subjects} subjects...", flush=True)

    # Run simulations
    results = []
    n_failed = 0
    for i, subj in enumerate(subjects):
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_subjects - i - 1) / rate
            print(
                f"  {i + 1}/{n_subjects} [{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]",
                flush=True,
            )
        try:
            res = simulate_subject(pipe, base_adme, base_clint_predictor, subj, drug_info)
            results.append(res)
        except Exception as exc:
            logger.warning("Subject %d failed: %s", subj.subject_id, exc)
            n_failed += 1

    total_time = time.time() - t0
    n_ok = len(results)

    if n_ok == 0:
        raise RuntimeError("All subject simulations failed. Check pipeline setup.")

    # Compute summary statistics
    cmax_vals = np.array([r["cmax_mg_L"] for r in results])
    auc_vals = np.array([r["auc_mg_h_L"] for r in results])
    thalf_vals = np.array([r["t_half_h"] for r in results if r["t_half_h"] is not None])

    def _cv(arr: np.ndarray) -> float:
        return float(100.0 * np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else float("nan")

    def _pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q))

    cmax_stats = {
        "mean": float(np.mean(cmax_vals)),
        "median": float(np.median(cmax_vals)),
        "sd": float(np.std(cmax_vals)),
        "cv_pct": _cv(cmax_vals),
        "p5": _pct(cmax_vals, 5),
        "p25": _pct(cmax_vals, 25),
        "p75": _pct(cmax_vals, 75),
        "p95": _pct(cmax_vals, 95),
        "gm": float(np.exp(np.mean(np.log(cmax_vals[cmax_vals > 0])))),
        "gsd": float(np.exp(np.std(np.log(cmax_vals[cmax_vals > 0])))),
    }
    auc_stats = {
        "mean": float(np.mean(auc_vals)),
        "median": float(np.median(auc_vals)),
        "sd": float(np.std(auc_vals)),
        "cv_pct": _cv(auc_vals),
        "p5": _pct(auc_vals, 5),
        "p25": _pct(auc_vals, 25),
        "p75": _pct(auc_vals, 75),
        "p95": _pct(auc_vals, 95),
    }

    # Per-phenotype Cmax for CYP2D6 drugs
    phenotype_stats: dict[str, Any] = {}
    if drug_info["primary_cyp"] == "CYP2D6":
        for phen in ("PM", "EM", "UM"):
            mask = [r["cyp2d6_phenotype"] == phen for r in results]
            sub_cmax = cmax_vals[[i for i, m in enumerate(mask) if m]]
            if len(sub_cmax) > 0:
                phenotype_stats[phen] = {
                    "n": len(sub_cmax),
                    "mean_cmax": float(np.mean(sub_cmax)),
                    "cv_pct": _cv(sub_cmax) if len(sub_cmax) > 1 else 0.0,
                }

    # Published reference CV for comparison
    ref_cv = drug_info.get("ref_cmax_cv_pct")
    ratio_to_ref = cmax_stats["cv_pct"] / ref_cv if ref_cv else None

    summary = {
        "drug": drug_name,
        "description": drug_info["description"],
        "primary_cyp": drug_info["primary_cyp"],
        "dose_mg": drug_info["dose_mg"],
        "n_subjects": n_ok,
        "n_failed": n_failed,
        "simulation_time_s": round(total_time, 1),
        "seed": seed,
        "cmax_stats": cmax_stats,
        "auc_stats": auc_stats,
        "thalf_stats": {
            "mean": float(np.mean(thalf_vals)) if len(thalf_vals) > 0 else None,
            "cv_pct": _cv(thalf_vals) if len(thalf_vals) > 0 else None,
        },
        "phenotype_cmax_stats": phenotype_stats,
        "ref_cmax_cv_pct": ref_cv,
        "cv_vs_ref_ratio": round(ratio_to_ref, 2) if ratio_to_ref else None,
        "subjects": results,
    }

    if verbose:
        print(f"\nResults for {drug_name.upper()} (N={n_ok}, {total_time:.0f}s):")
        print(f"  Cmax: {cmax_stats['mean']:.4f} ± {cmax_stats['sd']:.4f} mg/L")
        print(
            f"  Cmax CV%: {cmax_stats['cv_pct']:.1f}%  (ref: {ref_cv}%  ratio: {ratio_to_ref:.2f}x)"
            if ref_cv
            else f"  Cmax CV%: {cmax_stats['cv_pct']:.1f}%"
        )
        print(
            f"  Cmax GeoSD: {cmax_stats['gsd']:.2f}x (p5–p95: {cmax_stats['p5']:.4f}–{cmax_stats['p95']:.4f} mg/L)"
        )
        print(f"  AUC CV%:  {auc_stats['cv_pct']:.1f}%")
        if thalf_vals.size > 0:
            print(f"  t½ CV%:   {_cv(thalf_vals):.1f}%")
        if phenotype_stats:
            print("  CYP2D6 phenotype breakdown:")
            for phen, ps in phenotype_stats.items():
                print(
                    f"    {phen}: N={ps['n']:2d}  Cmax={ps['mean_cmax']:.4f} mg/L  CV%={ps['cv_pct']:.1f}%"
                )
        if n_failed > 0:
            print(f"  WARNING: {n_failed} subjects failed simulation")
        print()

    return summary


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_results(summary: dict[str, Any], output_dir: Path) -> Path:
    """Save population simulation results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    drug = summary["drug"]
    n = summary["n_subjects"]
    seed = summary["seed"]
    out_path = output_dir / f"population_sim_{drug}_N{n}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Population PK simulation (Phase 5.4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--drug",
        type=str,
        default="midazolam",
        choices=list(DRUG_REGISTRY.keys()),
        help="Drug to simulate (default: midazolam)",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=100,
        help="Number of virtual subjects (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--all-drugs",
        action="store_true",
        help="Simulate all drugs in registry",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Output directory for JSON results (default: results/)",
    )
    parser.add_argument(
        "--save-figure",
        action="store_true",
        help="Generate population figure after simulation",
    )
    args = parser.parse_args()

    drugs_to_run = list(DRUG_REGISTRY.keys()) if args.all_drugs else [args.drug]
    all_summaries = []

    for drug_name in drugs_to_run:
        summary = run_population_simulation(
            drug_name=drug_name,
            n_subjects=args.n_subjects,
            seed=args.seed,
            verbose=True,
        )
        out_path = save_results(summary, args.output_dir)
        print(f"Results saved to: {out_path}")
        all_summaries.append(summary)

    if args.save_figure:
        try:
            fig_script = ROOT / "docs" / "paper" / "figures" / "fig_population.py"
            if fig_script.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location("fig_population", fig_script)
                fig_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                spec.loader.exec_module(fig_mod)  # type: ignore[union-attr]
                fig_mod.generate_figure(all_summaries)
        except Exception as exc:
            logger.warning("Figure generation failed: %s", exc)


if __name__ == "__main__":
    main()
