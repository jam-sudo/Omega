"""Clinical trial simulator — Phase 15.

Simulates multi-dose PK studies in virtual populations, computing
steady-state PK endpoints (Cmax_ss, AUC_tau, Cmin, t½, AI) and
performing parallel-group bioequivalence analysis.

Multi-dose approach:
  - Single-dose PBPK run → plasma Cp(t)
  - Superposition: Cp_md(t) = Σ_{k=0}^{N} Cp(t − k·τ)  [linear PK]
  - Steady state attained after ≥ 5 doses (or user-specified N)

Bioequivalence (EMA/FDA guideline):
  - ln-transform AUC and Cmax
  - 90 % CI for Test/Reference GMR using Welch 2-sample t-test
  - BE criterion: 80.00 – 125.00 % for both endpoints

Usage::

    from omega_pbpk.clinical.trial import DoseRegimen, TrialArm, run_clinical_trial
    from omega_pbpk.drugs.drug import Drug

    drug_a = Drug(name="DrugA", mw=350, logP=2.0, fup=0.2, ...)
    arm_a  = TrialArm("A", drug_a, DoseRegimen(dose_mg=100, n_doses=10, interval_h=24))
    arm_b  = TrialArm("B", drug_a, DoseRegimen(dose_mg=50,  n_doses=10, interval_h=12))

    result = run_clinical_trial([arm_a, arm_b], n_subjects=24, seed=42)
    print(result.arm_summaries[0].cmax["median"])
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as t_dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trial design types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoseRegimen:
    """Multi-dose administration schedule.

    Attributes:
        dose_mg:     Single-dose amount (mg).
        n_doses:     Number of administered doses (default 10 → ~SS for t½ < 50 h).
        interval_h:  Dosing interval (h). Common: 24 = QD, 12 = BID, 8 = TID.
        route:       "oral" or "iv".
    """

    dose_mg: float
    n_doses: int = 10
    interval_h: float = 24.0
    route: str = "oral"

    @property
    def total_duration_h(self) -> float:
        """Total study period covered by all doses (h)."""
        return self.n_doses * self.interval_h


@dataclass(frozen=True)
class TrialArm:
    """One arm in a clinical trial.

    Attributes:
        name:      Arm label (e.g. "Test", "Reference", "Low dose").
        drug:      Drug object defining PK parameters.
        regimen:   Dosing schedule.
    """

    name: str
    drug: Any  # omega_pbpk.drugs.drug.Drug
    regimen: DoseRegimen


# ---------------------------------------------------------------------------
# Per-subject PK result
# ---------------------------------------------------------------------------


@dataclass
class SubjectPKResult:
    """Single-subject steady-state PK endpoints.

    All concentration units: mg/L.  Time units: h.
    """

    subject_id: int
    body_weight_kg: float
    cyp3a4_activity: float

    # --- Steady-state (last dosing interval) ---
    cmax_ss: float  # Peak plasma conc at steady state (mg/L)
    cmin_ss: float  # Trough plasma conc at steady state (mg/L)
    auc_tau: float  # AUC over one dosing interval at SS (mg·h/L)

    # --- Single-dose ---
    cmax_1: float  # Peak after first dose
    auc_inf: float  # AUC from 0 to ∞ (single dose)
    t_half_h: float  # Terminal half-life (h)
    tmax_h: float  # Time of Cmax (first dose, h)

    # --- Derived ---
    @property
    def accumulation_index(self) -> float:
        """Cmax_ss / Cmax_1 (accumulation ratio)."""
        return self.cmax_ss / max(self.cmax_1, 1e-12)

    @property
    def peak_trough_ratio(self) -> float:
        """Cmax_ss / Cmin_ss."""
        return self.cmax_ss / max(self.cmin_ss, 1e-12)


# ---------------------------------------------------------------------------
# Arm-level summary statistics
# ---------------------------------------------------------------------------


@dataclass
class ArmPKSummary:
    """Population PK summary for one trial arm."""

    arm_name: str
    n: int

    cmax_ss: dict[str, float]  # mean, median, cv_pct, p5, p95
    cmin_ss: dict[str, float]
    auc_tau: dict[str, float]
    t_half: dict[str, float]
    accumulation_index: dict[str, float]

    @staticmethod
    def _stats(values: NDArray[np.float64]) -> dict[str, float]:
        if len(values) == 0:
            return {"mean": 0.0, "median": 0.0, "cv_pct": 0.0, "p5": 0.0, "p95": 0.0}
        mean_ = float(np.mean(values))
        return {
            "mean": round(mean_, 4),
            "median": round(float(np.median(values)), 4),
            "cv_pct": round(float(np.std(values) / max(mean_, 1e-12) * 100), 2),
            "p5": round(float(np.percentile(values, 5)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
        }

    @classmethod
    def from_subjects(cls, arm_name: str, results: list[SubjectPKResult]) -> ArmPKSummary:
        arr = np.array
        return cls(
            arm_name=arm_name,
            n=len(results),
            cmax_ss=cls._stats(arr([r.cmax_ss for r in results])),
            cmin_ss=cls._stats(arr([r.cmin_ss for r in results])),
            auc_tau=cls._stats(arr([r.auc_tau for r in results])),
            t_half=cls._stats(arr([r.t_half_h for r in results])),
            accumulation_index=cls._stats(arr([r.accumulation_index for r in results])),
        )


# ---------------------------------------------------------------------------
# Bioequivalence analysis
# ---------------------------------------------------------------------------


@dataclass
class BioequivalenceResult:
    """Parallel-group bioequivalence analysis (EMA / FDA 2022 guideline).

    Uses ln-transformed AUC_tau and Cmax_ss with Welch 2-sample 90 % CI.
    BE window: 80.00 – 125.00 %.
    """

    test_arm: str
    reference_arm: str

    auc_gmr: float  # AUC geometric mean ratio (test/ref)
    auc_ci_90_lower: float  # 90 % CI lower bound
    auc_ci_90_upper: float  # 90 % CI upper bound

    cmax_gmr: float
    cmax_ci_90_lower: float
    cmax_ci_90_upper: float

    @property
    def auc_be(self) -> bool:
        """AUC 90 % CI falls within 80.00 – 125.00 %."""
        return 0.80 <= self.auc_ci_90_lower and self.auc_ci_90_upper <= 1.25

    @property
    def cmax_be(self) -> bool:
        """Cmax 90 % CI falls within 80.00 – 125.00 %."""
        return 0.80 <= self.cmax_ci_90_lower and self.cmax_ci_90_upper <= 1.25

    @property
    def is_bioequivalent(self) -> bool:
        """Both AUC and Cmax meet BE criteria."""
        return self.auc_be and self.cmax_be


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Complete clinical trial simulation output."""

    arm_results: list[list[SubjectPKResult]]
    arm_summaries: list[ArmPKSummary]
    bioequivalence: BioequivalenceResult | None
    design: dict[str, Any]

    def summary_table(self) -> list[dict[str, Any]]:
        """Return list of arm-summary dicts for tabular display."""
        rows = []
        for s in self.arm_summaries:
            rows.append(
                {
                    "arm": s.arm_name,
                    "n": s.n,
                    "Cmax_ss_mean_mg_L": s.cmax_ss["mean"],
                    "Cmax_ss_cv%": s.cmax_ss["cv_pct"],
                    "AUCtau_mean_mgh_L": s.auc_tau["mean"],
                    "AUCtau_cv%": s.auc_tau["cv_pct"],
                    "Cmin_ss_mean_mg_L": s.cmin_ss["mean"],
                    "t_half_mean_h": s.t_half["mean"],
                    "AI_mean": s.accumulation_index["mean"],
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Core simulation functions
# ---------------------------------------------------------------------------


def _simulate_single_dose(
    drug: Any,
    dose_mg: float,
    route: str,
    body_weight: float,
    hepatic_scale: float,
    cyp3a4: float,
    n_points: int = 500,
) -> tuple[NDArray, NDArray]:
    """Run a single-dose PBPK simulation.

    Returns (time_h, cp_mg_L) arrays of length n_points.
    """
    from omega_pbpk.core.body import WholeBodyPBPK
    from omega_pbpk.drugs.drug import Drug

    # Scale drug clearances
    scaled_drug = Drug(
        name=getattr(drug, "name", "compound"),
        mw=drug.mw,
        logP=drug.logP,
        fup=drug.fup,
        rbp=drug.rbp,
        pka=list(drug.pka) if drug.pka else [7.0],
        clint_hepatic_L_per_h=drug.clint_hepatic_L_per_h * hepatic_scale * cyp3a4,
        clr_L_per_h=drug.clr_L_per_h,
        # carry optional fields
        **{
            attr: getattr(drug, attr)
            for attr in (
                "drug_type",
                "clint_gut_L_per_h",
                "peff",
                "solubility_mg_mL",
                "fm",
                "kp",
                "partition_method",
                "compound_type",
            )
            if hasattr(drug, attr) and getattr(drug, attr) is not None
        },
    )

    # Determine simulation length: cover ~7 half-lives
    # Quick estimate: t_half ~ (Vd * fup) / CL; use 24 h as floor
    t_end = 72.0  # h — safe default covering most t½ ≤ 48 h

    model = WholeBodyPBPK(drug=scaled_drug, body_weight=body_weight)

    if route == "iv":
        model.setup_iv(dose_mg=dose_mg)
    else:
        model.setup_oral(dose_mg=dose_mg)

    result = model.simulate(t_end_h=t_end, dt_h=t_end / (n_points - 1))
    cp_raw = result.plasma_concentration()
    t_raw = result.time_h

    # Uniform time grid
    t_out = np.linspace(0.0, t_end, n_points)
    cp_out = np.interp(t_out, t_raw, cp_raw, left=0.0, right=0.0)
    return t_out, cp_out


def _superpose_doses(
    cp_single: NDArray, t_single: NDArray, interval_h: float, n_doses: int
) -> tuple[NDArray, NDArray]:
    """Superimpose n_doses single-dose profiles to get multi-dose curve.

    The superposition principle is valid for linear (first-order) PK.
    Returns (t_md, cp_md) over the full study window.
    """
    t_end_single = float(t_single[-1])
    t_end_md = interval_h * n_doses  # include one extra interval after last dose
    n_pts_md = int(len(t_single) * n_doses)  # proportionally more points
    n_pts_md = max(n_pts_md, 500)

    t_md = np.linspace(0.0, t_end_md, n_pts_md)
    cp_md = np.zeros(n_pts_md)

    for k in range(n_doses):
        offset = k * interval_h
        t_shifted = t_md - offset
        # Contributions only where t_shifted ∈ [0, t_end_single]
        mask = (t_shifted >= 0) & (t_shifted <= t_end_single)
        cp_md[mask] += np.interp(t_shifted[mask], t_single, cp_single)

    return t_md, cp_md


def _extract_ss_pk(
    t_md: NDArray, cp_md: NDArray, interval_h: float, n_doses: int
) -> tuple[float, float, float]:
    """Extract SS endpoints from the last dosing interval.

    Returns (cmax_ss, cmin_ss, auc_tau).
    """
    # Last dosing interval: [(n_doses-1)*tau, n_doses*tau]
    t_start = (n_doses - 1) * interval_h
    t_end = n_doses * interval_h

    mask = (t_md >= t_start) & (t_md <= t_end)
    if not np.any(mask):
        return 0.0, 0.0, 0.0

    cp_interval = cp_md[mask]
    t_interval = t_md[mask]

    cmax_ss = float(np.max(cp_interval))
    cmin_ss = float(cp_interval[-1])  # trough at end of interval
    auc_tau = float(np.trapezoid(cp_interval, t_interval))
    return cmax_ss, cmin_ss, auc_tau


def _terminal_half_life(t: NDArray, cp: NDArray) -> float:
    """Estimate terminal half-life by log-linear regression on tail."""
    n = len(t)
    if n < 4 or np.all(cp < 1e-12):
        return 99.0
    # Use last 30 % of time points after Cmax
    i_max = int(np.argmax(cp))
    tail_start = max(i_max + 1, int(0.6 * n))
    if tail_start >= n - 2:
        return 99.0
    t_tail = t[tail_start:]
    cp_tail = cp[tail_start:]
    pos = cp_tail > 1e-12
    if np.sum(pos) < 2:
        return 99.0
    slope, _ = np.polyfit(t_tail[pos], np.log(cp_tail[pos]), 1)
    if slope >= -1e-9:
        return 99.0
    return float(min(-math.log(2) / slope, 200.0))


def _simulate_one_subject(
    subj_id: int,
    drug: Any,
    regimen: DoseRegimen,
    body_weight: float,
    hepatic_scale: float,
    cyp3a4: float,
) -> SubjectPKResult | None:
    """Full single-subject multi-dose simulation."""
    try:
        t_single, cp_single = _simulate_single_dose(
            drug=drug,
            dose_mg=regimen.dose_mg,
            route=regimen.route,
            body_weight=body_weight,
            hepatic_scale=hepatic_scale,
            cyp3a4=cyp3a4,
        )

        # Single-dose PK
        i_max = int(np.argmax(cp_single))
        cmax_1 = float(cp_single[i_max])
        tmax_h = float(t_single[i_max])
        auc_inf = float(np.trapezoid(cp_single, t_single))
        t_half = _terminal_half_life(t_single, cp_single)

        # Multi-dose superposition
        t_md, cp_md = _superpose_doses(
            cp_single,
            t_single,
            interval_h=regimen.interval_h,
            n_doses=regimen.n_doses,
        )

        cmax_ss, cmin_ss, auc_tau = _extract_ss_pk(t_md, cp_md, regimen.interval_h, regimen.n_doses)

        return SubjectPKResult(
            subject_id=subj_id,
            body_weight_kg=round(body_weight, 2),
            cyp3a4_activity=round(cyp3a4, 4),
            cmax_ss=round(cmax_ss, 6),
            cmin_ss=round(cmin_ss, 6),
            auc_tau=round(auc_tau, 4),
            cmax_1=round(cmax_1, 6),
            auc_inf=round(auc_inf, 4),
            t_half_h=round(t_half, 2),
            tmax_h=round(tmax_h, 2),
        )
    except Exception as exc:
        logger.debug("Subject %d failed: %s", subj_id, exc)
        return None


def simulate_arm(
    arm: TrialArm,
    n_subjects: int = 24,
    seed: int = 42,
) -> list[SubjectPKResult]:
    """Simulate PK for all subjects in a trial arm.

    Args:
        arm:        Trial arm (drug + regimen).
        n_subjects: Number of virtual subjects.
        seed:       Random seed for population sampling.

    Returns:
        List of SubjectPKResult (only successful simulations included).
    """
    from omega_pbpk.population.physiology import VirtualPopulation

    vp = VirtualPopulation(n=n_subjects, seed=seed)
    subjects = vp.sample(n_subjects, seed=seed)

    _CP_SCALE = {"normal": 1.0, "A": 0.75, "B": 0.50, "C": 0.25}

    results: list[SubjectPKResult] = []
    for i, subj in enumerate(subjects):
        hepatic = _CP_SCALE.get(subj.child_pugh, 1.0)
        res = _simulate_one_subject(
            subj_id=i,
            drug=arm.drug,
            regimen=arm.regimen,
            body_weight=subj.body_weight_kg,
            hepatic_scale=hepatic,
            cyp3a4=subj.cyp3a4_activity,
        )
        if res is not None and res.cmax_ss > 0:
            results.append(res)

    logger.info("Arm '%s': %d/%d subjects succeeded.", arm.name, len(results), n_subjects)
    return results


# ---------------------------------------------------------------------------
# Bioequivalence
# ---------------------------------------------------------------------------


def _welch_90ci_ratio(
    test_values: NDArray[np.float64],
    ref_values: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Compute GMR and 90 % CI for test/reference ratio.

    Uses ln-transform and Welch 2-sample t-test (α = 0.05, one-tailed each).

    Returns:
        (gmr, ci_lower, ci_upper) — all as fractions (not percentages).
    """
    ln_test = np.log(np.maximum(test_values, 1e-12))
    ln_ref = np.log(np.maximum(ref_values, 1e-12))

    n1, n2 = len(ln_test), len(ln_ref)
    if n1 < 2 or n2 < 2:
        gmr = float(np.exp(np.mean(ln_test) - np.mean(ln_ref)))
        return gmr, 0.0, float("inf")

    mean_diff = float(np.mean(ln_test) - np.mean(ln_ref))
    var1 = float(np.var(ln_test, ddof=1))
    var2 = float(np.var(ln_ref, ddof=1))
    se = math.sqrt(var1 / n1 + var2 / n2)

    # Welch-Satterthwaite degrees of freedom
    if se == 0.0:
        nu = n1 + n2 - 2.0
    else:
        nu = (var1 / n1 + var2 / n2) ** 2 / (
            (var1 / n1) ** 2 / max(n1 - 1, 1) + (var2 / n2) ** 2 / max(n2 - 1, 1)
        )

    t_crit = float(t_dist.ppf(0.95, nu))  # 90 % two-sided = 5 % one-sided each

    ci_low_ln = mean_diff - t_crit * se
    ci_high_ln = mean_diff + t_crit * se

    gmr = math.exp(mean_diff)
    ci_lo = math.exp(ci_low_ln)
    ci_hi = math.exp(ci_high_ln)
    return round(gmr, 4), round(ci_lo, 4), round(ci_hi, 4)


def compute_bioequivalence(
    test_results: list[SubjectPKResult],
    ref_results: list[SubjectPKResult],
    test_arm_name: str = "Test",
    ref_arm_name: str = "Reference",
) -> BioequivalenceResult:
    """Parallel-group bioequivalence analysis.

    Applies EMA/FDA standard: ln-transform + 90 % CI from Welch t-test.
    BE window: 80.00 – 125.00 % for both AUC_tau and Cmax_ss.
    """
    auc_test = np.array([r.auc_tau for r in test_results], dtype=float)
    auc_ref = np.array([r.auc_tau for r in ref_results], dtype=float)
    cmax_test = np.array([r.cmax_ss for r in test_results], dtype=float)
    cmax_ref = np.array([r.cmax_ss for r in ref_results], dtype=float)

    auc_gmr, auc_lo, auc_hi = _welch_90ci_ratio(auc_test, auc_ref)
    cmax_gmr, cmax_lo, cmax_hi = _welch_90ci_ratio(cmax_test, cmax_ref)

    return BioequivalenceResult(
        test_arm=test_arm_name,
        reference_arm=ref_arm_name,
        auc_gmr=auc_gmr,
        auc_ci_90_lower=auc_lo,
        auc_ci_90_upper=auc_hi,
        cmax_gmr=cmax_gmr,
        cmax_ci_90_lower=cmax_lo,
        cmax_ci_90_upper=cmax_hi,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_clinical_trial(
    arms: list[TrialArm],
    n_subjects: int = 24,
    seed: int = 42,
) -> TrialResult:
    """Simulate a complete clinical trial across all arms.

    Args:
        arms:        List of TrialArm objects (1 = single-arm, 2 = parallel).
        n_subjects:  Number of virtual subjects per arm.
        seed:        Base random seed (arm k uses seed + k).

    Returns:
        TrialResult with per-arm summaries and, if 2 arms, BE analysis.
    """
    all_results: list[list[SubjectPKResult]] = []
    summaries: list[ArmPKSummary] = []

    for k, arm in enumerate(arms):
        arm_seed = seed + k
        res = simulate_arm(arm, n_subjects=n_subjects, seed=arm_seed)
        all_results.append(res)
        summaries.append(ArmPKSummary.from_subjects(arm.name, res))

    be: BioequivalenceResult | None = None
    if len(arms) == 2 and all_results[0] and all_results[1]:
        be = compute_bioequivalence(
            test_results=all_results[0],
            ref_results=all_results[1],
            test_arm_name=arms[0].name,
            ref_arm_name=arms[1].name,
        )

    design: dict[str, Any] = {
        "arms": [a.name for a in arms],
        "n_subjects": n_subjects,
        "seed": seed,
        "regimens": [
            {
                "arm": a.name,
                "dose_mg": a.regimen.dose_mg,
                "n_doses": a.regimen.n_doses,
                "interval_h": a.regimen.interval_h,
                "route": a.regimen.route,
            }
            for a in arms
        ],
    }

    return TrialResult(
        arm_results=all_results,
        arm_summaries=summaries,
        bioequivalence=be,
        design=design,
    )
