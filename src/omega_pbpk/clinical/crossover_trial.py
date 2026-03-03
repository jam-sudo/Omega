"""Virtual crossover trial simulation with TOST bioequivalence analysis.

Simulates 2×2 crossover BE studies using a 1-compartment analytical PK model
with between-subject and within-subject variability, then performs TOST
analysis on log-transformed PK parameters (AUC and Cmax).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import t as t_dist

from omega_pbpk.clinical.be_study_design import BE_LOWER, BE_UPPER, _tost_power
from omega_pbpk.drugs.drug import Drug

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossoverArm:
    """One arm (formulation) in a crossover trial."""

    name: str
    drug: Drug
    dose_mg: float
    route: str = "oral"


@dataclass(frozen=True)
class CrossoverDesign:
    """Design specification for a virtual crossover trial."""

    arms: tuple[CrossoverArm, ...]
    design: str = "2x2"
    n_subjects: int = 24
    washout_sufficient: bool = True
    wsv_cv_cl: float = 0.15
    wsv_cv_vd: float = 0.10
    wsv_cv_ka: float = 0.20
    period_effect: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class SubjectPeriodResult:
    """PK result for one subject in one period."""

    subject_id: int
    period: int
    sequence: int
    arm_name: str
    cmax_mg_L: float
    tmax_h: float
    auc0t_mg_h_L: float
    auc0inf_mg_h_L: float
    t_half_h: float
    cl_L_per_h: float


@dataclass(frozen=True)
class CrossoverTrialResult:
    """Complete result of a virtual crossover trial."""

    design: str
    n_subjects: int
    n_completed: int
    subject_results: list[SubjectPeriodResult]
    gmr_auc: float
    gmr_cmax: float
    ci90_auc: tuple[float, float]
    ci90_cmax: tuple[float, float]
    tost_p_auc: float
    tost_p_cmax: float
    be_conclusion_auc: str
    be_conclusion_cmax: str
    be_conclusion_overall: str
    be_limits: tuple[float, float]
    power_auc: float
    power_cmax: float
    cv_within_auc: float
    cv_within_cmax: float
    anova_summary: dict[str, float]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assign_sequences(n_subjects: int, design: str, rng: np.random.Generator) -> list[list[int]]:
    """Assign treatment sequences for a crossover design.

    For 2x2: half get [0,1] (Test-first), half get [1,0] (Ref-first).
    """
    if design != "2x2":
        raise ValueError(f"Unsupported design: {design}")

    n_half = n_subjects // 2
    seqs: list[list[int]] = [[0, 1]] * n_half + [[1, 0]] * (n_subjects - n_half)
    # Shuffle so assignment is randomized
    indices = rng.permutation(len(seqs)).tolist()
    return [seqs[i] for i in indices]


def _apply_wsv(
    base_cl: float,
    base_vd: float,
    base_ka: float,
    wsv_cv_cl: float,
    wsv_cv_vd: float,
    wsv_cv_ka: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Apply within-subject variability as log-normal multiplicative factors."""

    def _sample(cv: float) -> float:
        if cv <= 0:
            return 1.0
        sigma = math.sqrt(math.log(1 + cv**2))
        return float(rng.lognormal(mean=0.0, sigma=sigma))

    cl = base_cl * _sample(wsv_cv_cl)
    vd = base_vd * _sample(wsv_cv_vd)
    ka = base_ka * _sample(wsv_cv_ka)
    return cl, vd, ka


def _simulate_subject_period(
    drug: Drug,
    dose_mg: float,
    route: str,
    cl: float,
    vd: float,
    ka: float,
    period_factor: float,
    t_end_h: float,
) -> SubjectPeriodResult:
    """Simulate one subject-period using the 1-compartment analytical model.

    Oral: C(t) = (F*D*ka)/(Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
    IV:   C(t) = (D/Vd) * exp(-ke*t)
    """
    ke = cl / vd
    f_bio = 1.0  # bioavailability assumed 1.0 for analytical model

    if route == "iv":
        # IV bolus
        cmax = (dose_mg / vd) * period_factor
        tmax = 0.0
        auc_inf = (dose_mg / cl) * period_factor
        t_half = math.log(2) / ke
    else:
        # Oral first-order absorption
        if abs(ka - ke) < 1e-8:
            # Special case: ka ≈ ke — use limit form
            tmax = 1.0 / ke
            cmax = (f_bio * dose_mg / vd) * math.exp(-1.0) * period_factor
        else:
            tmax = math.log(ka / ke) / (ka - ke)
            cmax = (
                (f_bio * dose_mg * ka)
                / (vd * (ka - ke))
                * (math.exp(-ke * tmax) - math.exp(-ka * tmax))
            ) * period_factor

        auc_inf = (f_bio * dose_mg / cl) * period_factor
        t_half = math.log(2) / ke

    # AUC0-t: for analytical model, approximate as fraction of AUC_inf
    frac_t = 1.0 - math.exp(-ke * t_end_h)
    auc_0t = auc_inf * frac_t

    return SubjectPeriodResult(
        subject_id=0,  # filled by caller
        period=0,
        sequence=0,
        arm_name="",
        cmax_mg_L=round(cmax, 6),
        tmax_h=round(tmax, 4),
        auc0t_mg_h_L=round(auc_0t, 6),
        auc0inf_mg_h_L=round(auc_inf, 6),
        t_half_h=round(t_half, 4),
        cl_L_per_h=round(cl, 4),
    )


# ---------------------------------------------------------------------------
# TOST analysis
# ---------------------------------------------------------------------------


def _estimate_within_subject_cv(mse_log: float) -> float:
    """Estimate within-subject CV from MSE on log scale.

    CV = sqrt(exp(MSE) - 1)
    """
    return math.sqrt(math.exp(mse_log) - 1) if mse_log > 0 else 0.0


def _tost_analysis(
    log_ratios_auc: np.ndarray,
    log_ratios_cmax: np.ndarray,
    n: int,
    design: str,
    alpha: float = 0.05,
    be_limits: tuple[float, float] = (BE_LOWER, BE_UPPER),
) -> dict:
    """Perform TOST analysis on log-transformed within-subject ratios.

    log_ratios = log(Test) - log(Ref) for each subject (paired differences).
    90% CI = exp(mean_diff ± t(0.95, n-1) * SE)
    TOST p-values from one-sided t-tests against log(limits).
    """
    df = n - 1
    t_crit = float(t_dist.ppf(1 - alpha, df))

    results: dict = {}
    for label, log_diffs in [("auc", log_ratios_auc), ("cmax", log_ratios_cmax)]:
        mean_diff = float(np.mean(log_diffs))
        sd_diff = float(np.std(log_diffs, ddof=1))
        se = sd_diff / math.sqrt(n)

        # 90% CI
        ci_lower = math.exp(mean_diff - t_crit * se) if se > 0 else math.exp(mean_diff)
        ci_upper = math.exp(mean_diff + t_crit * se) if se > 0 else math.exp(mean_diff)

        # GMR
        gmr = math.exp(mean_diff)

        # TOST p-values (one-sided tests)
        log_lower = math.log(be_limits[0])
        log_upper = math.log(be_limits[1])

        if se > 0:
            t_lower = (mean_diff - log_lower) / se
            t_upper = (log_upper - mean_diff) / se
            p_lower = 1.0 - float(t_dist.cdf(t_lower, df))
            p_upper = 1.0 - float(t_dist.cdf(t_upper, df))
        else:
            p_lower = 0.0
            p_upper = 0.0

        tost_p = max(p_lower, p_upper)

        # BE conclusion
        be_pass = ci_lower >= be_limits[0] and ci_upper <= be_limits[1]

        # MSE from paired differences
        mse_log = sd_diff**2 if sd_diff > 0 else 0.0
        cv_within = _estimate_within_subject_cv(mse_log)

        results[label] = {
            "gmr": round(gmr, 6),
            "ci90": (round(ci_lower, 6), round(ci_upper, 6)),
            "tost_p": round(tost_p, 6),
            "be_pass": be_pass,
            "mean_diff": round(mean_diff, 6),
            "sd_diff": round(sd_diff, 6),
            "se": round(se, 6),
            "mse_log": round(mse_log, 6),
            "cv_within": round(cv_within, 6),
        }

    return results


# ---------------------------------------------------------------------------
# Population parameter generation
# ---------------------------------------------------------------------------


def _derive_pop_params(drug: Drug) -> tuple[float, float, float]:
    """Derive population CL, Vd, ka from Drug object.

    Uses simplified relationships:
    - CL from clint_hepatic_L_per_h if available, else scaled CLint, else default
    - Vd from mw heuristic (mw * 0.15), clamped to reasonable range
    - ka from peff or default
    """
    # CL
    if drug.clint_hepatic_L_per_h > 0:
        cl = drug.clint_hepatic_L_per_h
    elif drug.clint_scaled_L_per_h > 0:
        cl = drug.clint_scaled_L_per_h
    else:
        cl = 5.0  # default

    # Vd
    vd = drug.mw * 0.15
    vd = max(10.0, min(vd, 200.0))  # clamp to reasonable range

    # ka
    if drug.peff > 0:
        ka = drug.peff * 10.0
        ka = max(0.1, min(ka, 5.0))
    else:
        ka = 1.0

    return cl, vd, ka


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_crossover_trial(
    design: CrossoverDesign,
    t_end_h: float = 48.0,
) -> CrossoverTrialResult:
    """Run a complete virtual crossover trial simulation.

    1. Generate N subjects with between-subject variability (BSV)
    2. Assign sequences randomly
    3. For each subject × period: apply WSV, period effect, simulate PK
    4. Collect paired Test/Ref data
    5. Run TOST analysis
    6. Estimate power from observed within-subject CV
    """
    if len(design.arms) != 2:
        raise ValueError("Crossover trial requires exactly 2 arms (Test and Reference)")

    rng = np.random.default_rng(design.seed)
    n = design.n_subjects
    warnings: list[str] = []

    if n < 12:
        warnings.append(f"Small sample size (N={n}); results may have limited reliability")

    # Derive population PK parameters for each arm
    arm_pop: list[tuple[float, float, float]] = []
    for arm in design.arms:
        cl, vd, ka = _derive_pop_params(arm.drug)
        arm_pop.append((cl, vd, ka))

    # Generate BSV — sample individual base parameters from log-normal
    bsv_cv_cl = 0.30
    bsv_cv_vd = 0.25
    bsv_cv_ka = 0.40

    sigma_cl = math.sqrt(math.log(1 + bsv_cv_cl**2))
    sigma_vd = math.sqrt(math.log(1 + bsv_cv_vd**2))
    sigma_ka = math.sqrt(math.log(1 + bsv_cv_ka**2))

    # Per-subject random effects (same for both arms — crossover eliminates BSV)
    eta_cl = rng.normal(0, sigma_cl, n)
    eta_vd = rng.normal(0, sigma_vd, n)
    eta_ka = rng.normal(0, sigma_ka, n)

    # Assign sequences
    sequences = _assign_sequences(n, design.design, rng)

    # Simulate all subject-period combinations
    all_results: list[SubjectPeriodResult] = []
    # Paired data: per subject, store log(metric) for each arm
    test_log_auc = np.zeros(n)
    ref_log_auc = np.zeros(n)
    test_log_cmax = np.zeros(n)
    ref_log_cmax = np.zeros(n)

    n_periods = 2  # 2x2 design

    for subj_idx in range(n):
        seq = sequences[subj_idx]

        for period in range(n_periods):
            arm_idx = seq[period]
            arm = design.arms[arm_idx]
            pop_cl, pop_vd, pop_ka = arm_pop[arm_idx]

            # Apply BSV: individual base = population * exp(eta)
            base_cl = pop_cl * math.exp(eta_cl[subj_idx])
            base_vd = pop_vd * math.exp(eta_vd[subj_idx])
            base_ka = pop_ka * math.exp(eta_ka[subj_idx])

            # Apply WSV per occasion
            cl, vd, ka = _apply_wsv(
                base_cl,
                base_vd,
                base_ka,
                design.wsv_cv_cl,
                design.wsv_cv_vd,
                design.wsv_cv_ka,
                rng,
            )

            # Period effect (multiplicative on concentration)
            period_factor = 1.0 + design.period_effect * period

            # Simulate
            res = _simulate_subject_period(
                arm.drug,
                arm.dose_mg,
                arm.route,
                cl,
                vd,
                ka,
                period_factor,
                t_end_h,
            )

            # Create full result with metadata
            spr = SubjectPeriodResult(
                subject_id=subj_idx + 1,
                period=period + 1,
                sequence=subj_idx % 2,
                arm_name=arm.name,
                cmax_mg_L=res.cmax_mg_L,
                tmax_h=res.tmax_h,
                auc0t_mg_h_L=res.auc0t_mg_h_L,
                auc0inf_mg_h_L=res.auc0inf_mg_h_L,
                t_half_h=res.t_half_h,
                cl_L_per_h=res.cl_L_per_h,
            )
            all_results.append(spr)

            # Store for TOST — arm 0 = Test, arm 1 = Reference
            if arm_idx == 0:
                test_log_auc[subj_idx] = math.log(max(res.auc0inf_mg_h_L, 1e-12))
                test_log_cmax[subj_idx] = math.log(max(res.cmax_mg_L, 1e-12))
            else:
                ref_log_auc[subj_idx] = math.log(max(res.auc0inf_mg_h_L, 1e-12))
                ref_log_cmax[subj_idx] = math.log(max(res.cmax_mg_L, 1e-12))

    # Paired log-differences (Test - Ref)
    log_ratios_auc = test_log_auc - ref_log_auc
    log_ratios_cmax = test_log_cmax - ref_log_cmax

    # TOST analysis
    be_limits = (BE_LOWER, BE_UPPER)
    tost = _tost_analysis(log_ratios_auc, log_ratios_cmax, n, design.design, be_limits=be_limits)

    auc_res = tost["auc"]
    cmax_res = tost["cmax"]

    # Overall BE conclusion
    overall = "BE" if auc_res["be_pass"] and cmax_res["be_pass"] else "NOT BE"

    # Estimate power from observed within-subject CV
    power_auc = _tost_power(
        n,
        auc_res["cv_within"],
        auc_res["gmr"],
        be_limits[0],
        be_limits[1],
        0.05,
        design.design,
    )
    power_cmax = _tost_power(
        n,
        cmax_res["cv_within"],
        cmax_res["gmr"],
        be_limits[0],
        be_limits[1],
        0.05,
        design.design,
    )

    # ANOVA summary
    anova_summary = {
        "mse_auc": auc_res["mse_log"],
        "mse_cmax": cmax_res["mse_log"],
        "mean_diff_auc": auc_res["mean_diff"],
        "mean_diff_cmax": cmax_res["mean_diff"],
        "sd_diff_auc": auc_res["sd_diff"],
        "sd_diff_cmax": cmax_res["sd_diff"],
    }

    return CrossoverTrialResult(
        design=design.design,
        n_subjects=n,
        n_completed=n,
        subject_results=all_results,
        gmr_auc=auc_res["gmr"],
        gmr_cmax=cmax_res["gmr"],
        ci90_auc=auc_res["ci90"],
        ci90_cmax=cmax_res["ci90"],
        tost_p_auc=auc_res["tost_p"],
        tost_p_cmax=cmax_res["tost_p"],
        be_conclusion_auc="BE" if auc_res["be_pass"] else "NOT BE",
        be_conclusion_cmax="BE" if cmax_res["be_pass"] else "NOT BE",
        be_conclusion_overall=overall,
        be_limits=be_limits,
        power_auc=round(power_auc, 6),
        power_cmax=round(power_cmax, 6),
        cv_within_auc=auc_res["cv_within"],
        cv_within_cmax=cmax_res["cv_within"],
        anova_summary=anova_summary,
        warnings=warnings,
    )
