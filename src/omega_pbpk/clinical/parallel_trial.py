"""Phase 47 — Parallel-Group Clinical Trial Simulator.

Simulates a multi-arm, parallel-group phase 2/3 clinical trial with:
  - 2–5 dose groups (including optional placebo)
  - 1-compartment PK with between-subject variability (log-normal BSV)
  - Emax / Hill PD model linking AUC or Cmax to therapeutic effect
  - Statistical analysis: one-way ANOVA + Bonferroni pairwise t-tests
  - Dose-response curve fitting (Hill equation via scipy.optimize.curve_fit)
  - Responder analysis (% subjects exceeding effect threshold)
  - Interim futility analysis at 50% enrolment (conditional power)

Distinct from Phase 33 (crossover_trial.py) which handles 2×2 BE trials.
This module targets phase 2 dose-finding and phase 3 efficacy confirmation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.stats import f_oneway, ttest_ind

__all__ = [
    "DoseGroup",
    "ParallelTrialDesign",
    "SubjectResult",
    "ArmSummary",
    "PairwiseResult",
    "DoseResponseFit",
    "InterimAnalysis",
    "ParallelTrialResult",
    "run_parallel_trial",
]

# ---------------------------------------------------------------------------
# Default BSV (between-subject variability) coefficients of variation
# ---------------------------------------------------------------------------

_DEFAULT_BSV_CV_CL: float = 0.30   # CL CV (30%)
_DEFAULT_BSV_CV_VD: float = 0.20   # Vd CV (20%)
_DEFAULT_BSV_CV_KA: float = 0.40   # ka CV (40%)

_RESPONSE_THRESHOLD: float = 0.50  # fraction of Emax considered "response"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoseGroup:
    """One arm (dose level) in a parallel trial.

    Attributes:
        name: Human-readable label (e.g. 'Placebo', '10 mg', '50 mg').
        dose_mg: Administered dose in mg. Use 0 for placebo.
        route: 'oral' or 'iv'.
        n_subjects: Number of subjects randomised to this arm.
    """

    name: str
    dose_mg: float
    route: str = "oral"
    n_subjects: int = 30


@dataclass(frozen=True)
class ParallelTrialDesign:
    """Full specification of a parallel-group trial.

    Attributes:
        dose_groups: Tuple of DoseGroup (2–5 arms; include placebo as dose=0).
        cl_L_per_h: Population mean clearance (L/h).
        vd_L: Population mean volume of distribution (L).
        ka_per_h: Population mean absorption rate constant (1/h; oral).
        f_bioavail: Oral bioavailability (0–1).
        bsv_cv_cl: BSV CV for CL (log-normal).
        bsv_cv_vd: BSV CV for Vd.
        bsv_cv_ka: BSV CV for ka.
        emax: Maximum drug effect (e.g. 1.0 = 100% response, or absolute units).
        ec50_mg_L: Concentration eliciting 50% of Emax (mg/L).
        hill: Hill coefficient (steepness of the Emax curve).
        pd_input: PD driver — 'cmax' or 'auc' (default 'cmax').
        t_end_h: PK simulation horizon (h).
        alpha: Two-sided α for statistical tests.
        interim_fraction: Enrolment fraction at interim analysis (0 = no interim).
        seed: Random seed.
    """

    dose_groups: tuple[DoseGroup, ...]
    cl_L_per_h: float = 5.0
    vd_L: float = 70.0
    ka_per_h: float = 1.2
    f_bioavail: float = 0.8
    bsv_cv_cl: float = _DEFAULT_BSV_CV_CL
    bsv_cv_vd: float = _DEFAULT_BSV_CV_VD
    bsv_cv_ka: float = _DEFAULT_BSV_CV_KA
    emax: float = 1.0
    ec50_mg_L: float = 1.0
    hill: float = 1.0
    pd_input: str = "cmax"
    t_end_h: float = 24.0
    alpha: float = 0.05
    interim_fraction: float = 0.5
    seed: int = 42


@dataclass(frozen=True)
class SubjectResult:
    """PK/PD outcome for one simulated subject.

    Attributes:
        subject_id: Sequential ID within the arm.
        arm_name: Name of the dose group.
        dose_mg: Nominal dose.
        cl_L_per_h: Individual CL.
        vd_L: Individual Vd.
        cmax_mg_L: Peak plasma concentration.
        tmax_h: Time to Cmax.
        auc_mg_h_L: AUC₀₋∞.
        t_half_h: Elimination half-life.
        effect: Emax-model PD response (0–Emax range).
        responder: True if effect ≥ _RESPONSE_THRESHOLD × Emax.
    """

    subject_id: int
    arm_name: str
    dose_mg: float
    cl_L_per_h: float
    vd_L: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_L: float
    t_half_h: float
    effect: float
    responder: bool


@dataclass
class ArmSummary:
    """Aggregate PK/PD statistics for one dose group.

    Attributes:
        name: Arm label.
        dose_mg: Nominal dose (mg).
        n: Number of subjects.
        cmax_mean: Geometric mean Cmax (mg/L).
        cmax_gsd: Geometric SD of Cmax.
        auc_mean: Geometric mean AUC (mg·h/L).
        auc_gsd: Geometric SD of AUC.
        effect_mean: Arithmetic mean effect.
        effect_sd: SD of effect.
        responder_pct: Percentage of subjects who are responders.
    """

    name: str
    dose_mg: float
    n: int
    cmax_mean: float = 0.0
    cmax_gsd: float = 0.0
    auc_mean: float = 0.0
    auc_gsd: float = 0.0
    effect_mean: float = 0.0
    effect_sd: float = 0.0
    responder_pct: float = 0.0


@dataclass(frozen=True)
class PairwiseResult:
    """Pairwise comparison of a dose arm vs. the reference (lowest-dose) arm.

    Attributes:
        arm_name: Dose arm label.
        dose_mg: Dose (mg).
        endpoint: 'auc', 'cmax', or 'effect'.
        t_stat: t-statistic.
        p_value: Raw two-sided p-value from Welch t-test.
        p_adjusted: Bonferroni-adjusted p-value.
        significant: True if p_adjusted < alpha.
        mean_difference: Arithmetic mean difference (dose arm − reference).
        ci_lower: Lower 95% CI for mean difference.
        ci_upper: Upper 95% CI for mean difference.
    """

    arm_name: str
    dose_mg: float
    endpoint: str
    t_stat: float
    p_value: float
    p_adjusted: float
    significant: bool
    mean_difference: float
    ci_lower: float
    ci_upper: float


@dataclass
class DoseResponseFit:
    """Hill-equation fit to the dose-response relationship.

    Attributes:
        emax_fit: Fitted maximum effect.
        ec50_dose_mg: Effective dose giving 50% of Emax (mg).
        hill_fit: Fitted Hill coefficient.
        r_squared: Goodness of fit R².
        ed80_mg: Dose achieving 80% of Emax (mg).
        ed90_mg: Dose achieving 90% of Emax (mg).
        fit_success: Whether curve_fit converged.
    """

    emax_fit: float = 0.0
    ec50_dose_mg: float = 0.0
    hill_fit: float = 1.0
    r_squared: float = 0.0
    ed80_mg: float = 0.0
    ed90_mg: float = 0.0
    fit_success: bool = False


@dataclass
class InterimAnalysis:
    """Result of the interim futility analysis.

    Attributes:
        conducted: Whether an interim was run.
        fraction_enrolled: Fraction of subjects enrolled at interim.
        conditional_power: Estimated conditional power at interim.
        futility_stop: True if conditional power < 20% (standard futility boundary).
        n_interim: Number of subjects available at interim.
    """

    conducted: bool = False
    fraction_enrolled: float = 0.0
    conditional_power: float = 0.0
    futility_stop: bool = False
    n_interim: int = 0


@dataclass
class ParallelTrialResult:
    """Full result of a parallel-group clinical trial simulation.

    Attributes:
        arms: Per-arm PK/PD summaries.
        subjects: All individual subject results.
        anova_p_auc: One-way ANOVA p-value across arms for AUC.
        anova_p_cmax: One-way ANOVA p-value across arms for Cmax.
        anova_p_effect: One-way ANOVA p-value across arms for effect.
        pairwise: Pairwise comparisons vs. reference arm (Bonferroni-corrected).
        dose_response: Hill-equation fit to arm mean effects vs. dose.
        interim: Interim futility analysis.
        trial_success: True if at least one dose arm shows significant effect.
        optimal_dose_mg: Dose achieving >= 80% Emax by dose-response fit.
        total_subjects: Total randomised.
    """

    arms: list[ArmSummary] = field(default_factory=list)
    subjects: list[SubjectResult] = field(default_factory=list)
    anova_p_auc: float = 1.0
    anova_p_cmax: float = 1.0
    anova_p_effect: float = 1.0
    pairwise: list[PairwiseResult] = field(default_factory=list)
    dose_response: DoseResponseFit = field(default_factory=DoseResponseFit)
    interim: InterimAnalysis = field(default_factory=InterimAnalysis)
    trial_success: bool = False
    optimal_dose_mg: float = 0.0
    total_subjects: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lognormal_bsv(mean: float, cv: float, rng: np.random.Generator) -> float:
    """Sample from a log-normal distribution with given mean and CV.

    CV = σ/μ for the underlying normal,  approximated as σ_ln ≈ CV for small CV.
    Exact: μ_ln = log(mean) - σ_ln²/2,  σ_ln = sqrt(log(1 + CV²)).
    """
    sigma_ln = math.sqrt(math.log(1.0 + cv**2))
    mu_ln = math.log(max(mean, 1e-12)) - 0.5 * sigma_ln**2
    return float(np.exp(mu_ln + sigma_ln * rng.standard_normal()))


def _simulate_subject_pk(
    dose_mg: float,
    route: str,
    cl_pop: float,
    vd_pop: float,
    ka_pop: float,
    f_bio: float,
    bsv_cv_cl: float,
    bsv_cv_vd: float,
    bsv_cv_ka: float,
    t_end_h: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float]:
    """Simulate one subject's PK using log-normal BSV.

    Returns:
        (cl_i, vd_i, cmax, tmax, auc_inf, t_half)
    """
    if dose_mg <= 0.0:
        # Placebo: zero PK
        return 0.0, vd_pop, 0.0, 0.0, 0.0, math.log(2) / (cl_pop / vd_pop)

    cl_i = _lognormal_bsv(cl_pop, bsv_cv_cl, rng)
    vd_i = _lognormal_bsv(vd_pop, bsv_cv_vd, rng)
    ka_i = _lognormal_bsv(ka_pop, bsv_cv_ka, rng)
    ke = cl_i / vd_i
    t_half = math.log(2) / ke

    if route == "iv":
        cmax = dose_mg / vd_i
        tmax = 0.0
        auc_inf = dose_mg / cl_i
    else:
        if abs(ka_i - ke) < 1e-8:
            tmax = 1.0 / ke
            cmax = f_bio * dose_mg / vd_i * math.exp(-1.0)
        else:
            tmax = math.log(ka_i / ke) / (ka_i - ke)
            cmax = (
                f_bio * dose_mg * ka_i / (vd_i * (ka_i - ke))
                * (math.exp(-ke * tmax) - math.exp(-ka_i * tmax))
            )
        auc_inf = f_bio * dose_mg / cl_i

    # Clamp negatives (numerical edge cases)
    cmax = max(cmax, 0.0)
    auc_inf = max(auc_inf, 0.0)

    return cl_i, vd_i, cmax, tmax, auc_inf, t_half


def _emax_effect(
    c: float | NDArray[np.float64],
    emax: float,
    ec50: float,
    hill: float,
) -> float | NDArray[np.float64]:
    """Hill / Emax equation: effect = Emax × C^n / (EC50^n + C^n).

    Accepts both scalar and array inputs.
    """
    c_arr = np.asarray(c, dtype=np.float64)
    ec50_n = max(ec50, 1e-12) ** hill
    c_n = np.power(np.maximum(c_arr, 0.0), hill)
    result = emax * c_n / (ec50_n + c_n)
    return float(result) if c_arr.ndim == 0 else result


def _one_way_anova(groups: list[NDArray[np.float64]]) -> float:
    """Return p-value from one-way ANOVA (scipy.stats.f_oneway)."""
    if len(groups) < 2:
        return 1.0
    valid = [g for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return 1.0
    try:
        _, p = f_oneway(*valid)
        return float(p) if math.isfinite(p) else 1.0
    except Exception:
        return 1.0


def _pairwise_tests(
    reference: NDArray[np.float64],
    arms: list[tuple[str, float, NDArray[np.float64]]],
    endpoint: str,
    alpha: float,
) -> list[PairwiseResult]:
    """Welch t-test of each arm vs. reference with Bonferroni correction.

    Args:
        reference: Reference arm (placebo or lowest dose) data.
        arms: List of (arm_name, dose_mg, data_array) for non-reference arms.
        endpoint: Endpoint label ('auc', 'cmax', 'effect').
        alpha: Family-wise α (Bonferroni divisor = len(arms)).

    Returns:
        List of PairwiseResult, one per arm.
    """
    n_comparisons = len(arms)
    results: list[PairwiseResult] = []

    for arm_name, dose_mg, data in arms:
        if len(data) < 2 or len(reference) < 2:
            results.append(PairwiseResult(
                arm_name=arm_name, dose_mg=dose_mg, endpoint=endpoint,
                t_stat=0.0, p_value=1.0, p_adjusted=1.0, significant=False,
                mean_difference=0.0, ci_lower=0.0, ci_upper=0.0,
            ))
            continue

        try:
            stat, p = ttest_ind(data, reference, equal_var=False)
        except Exception:
            stat, p = 0.0, 1.0

        p_adj = min(float(p) * n_comparisons, 1.0)
        mean_diff = float(np.mean(data) - np.mean(reference))

        # 95% CI for mean difference (Welch approximation)
        from scipy.stats import t as t_dist  # noqa: PLC0415
        n1, n2 = len(data), len(reference)
        se1 = float(np.var(data, ddof=1)) / n1
        se2 = float(np.var(reference, ddof=1)) / n2
        se_diff = math.sqrt(se1 + se2)
        # Welch–Satterthwaite df
        if se_diff > 0:
            df = (se1 + se2)**2 / (se1**2 / (n1 - 1) + se2**2 / (n2 - 1))
            t_crit = t_dist.ppf(0.975, df)
            ci_lo = mean_diff - t_crit * se_diff
            ci_hi = mean_diff + t_crit * se_diff
        else:
            ci_lo = ci_hi = mean_diff

        results.append(PairwiseResult(
            arm_name=arm_name,
            dose_mg=dose_mg,
            endpoint=endpoint,
            t_stat=round(float(stat), 4),
            p_value=round(float(p), 6),
            p_adjusted=round(p_adj, 6),
            significant=p_adj < alpha,
            mean_difference=round(mean_diff, 4),
            ci_lower=round(ci_lo, 4),
            ci_upper=round(ci_hi, 4),
        ))

    return results


def _hill_equation(dose: NDArray, emax: float, ed50: float, hill: float) -> NDArray:
    """Hill equation: effect = Emax × dose^n / (ED50^n + dose^n)."""
    d_n = np.power(np.maximum(dose, 0.0), hill)
    ed50_n = max(ed50, 1e-12)**hill
    return emax * d_n / (ed50_n + d_n)


def _fit_dose_response(
    doses: NDArray[np.float64], effects: NDArray[np.float64]
) -> DoseResponseFit:
    """Fit Hill equation to (dose, effect) pairs.

    Excludes placebo (dose=0) from the fit; uses it as the baseline intercept.
    """
    mask = doses > 0
    d_fit = doses[mask]
    e_fit = effects[mask]

    if len(d_fit) < 2:
        return DoseResponseFit(fit_success=False)

    emax_guess = float(np.max(e_fit))
    ed50_guess = float(np.median(d_fit))

    try:
        bounds = ([0.0, 0.0, 0.1], [emax_guess * 5 + 1e-6, d_fit.max() * 100, 5.0])
        popt, _ = curve_fit(
            _hill_equation, d_fit, e_fit,
            p0=[emax_guess, ed50_guess, 1.0],
            bounds=bounds,
            maxfev=2000,
        )
        emax_f, ed50_f, hill_f = float(popt[0]), float(popt[1]), float(popt[2])

        # R²
        e_pred = _hill_equation(d_fit, emax_f, ed50_f, hill_f)
        ss_res = float(np.sum((e_fit - e_pred) ** 2))
        ss_tot = float(np.sum((e_fit - e_fit.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)

        # ED80, ED90
        def _inv_hill(frac: float) -> float:
            return ed50_f * (frac / (1.0 - frac)) ** (1.0 / hill_f)

        ed80 = _inv_hill(0.8)
        ed90 = _inv_hill(0.9)

        return DoseResponseFit(
            emax_fit=round(emax_f, 4),
            ec50_dose_mg=round(ed50_f, 4),
            hill_fit=round(hill_f, 4),
            r_squared=round(r2, 4),
            ed80_mg=round(ed80, 2),
            ed90_mg=round(ed90, 2),
            fit_success=True,
        )
    except Exception:
        return DoseResponseFit(fit_success=False)


def _conditional_power(
    effects_interim: NDArray[np.float64],
    reference_interim: NDArray[np.float64],
    n_total: int,
    alpha: float = 0.05,
) -> float:
    """Approximate conditional power at interim.

    Assumes equal information fraction and uses observed effect size
    to project power at full enrolment via the Lan-DeMets boundary.

    Returns:
        Estimated conditional power ∈ [0, 1].
    """
    from scipy.stats import norm  # noqa: PLC0415

    n_int = len(effects_interim) + len(reference_interim)
    if n_int < 4 or n_total < n_int:
        return 0.5

    eff_diff = float(np.mean(effects_interim) - np.mean(reference_interim))
    pooled_sd = float(np.std(
        np.concatenate([effects_interim, reference_interim]), ddof=1
    ))
    if pooled_sd < 1e-12:
        return 0.5

    n_per = n_int // 2
    z_obs = eff_diff / (pooled_sd * math.sqrt(2.0 / n_per))
    info_frac = n_int / n_total

    # Expected Z at full analysis given current trend
    z_final_expected = z_obs * math.sqrt(info_frac) + math.sqrt(1.0 - info_frac) * 0.0
    z_alpha = norm.ppf(1.0 - alpha / 2)
    cp = float(norm.cdf(z_final_expected - z_alpha))
    return round(max(0.0, min(1.0, cp)), 4)


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def run_parallel_trial(design: ParallelTrialDesign) -> ParallelTrialResult:
    """Simulate a parallel-group clinical trial.

    For each dose group:
    1. Simulate n_subjects with log-normal BSV around population PK params.
    2. Compute Cmax, AUC, t½ from 1-compartment analytical model.
    3. Apply Emax/Hill PD model to derive individual effects.
    4. Accumulate subject-level results.

    Statistical analysis:
    - One-way ANOVA across all arms for AUC, Cmax, effect.
    - Bonferroni pairwise t-tests vs. lowest-dose (or placebo) arm.
    - Hill-equation dose-response fit to arm mean effects.
    - Interim futility check at interim_fraction enrolment.

    Args:
        design: ParallelTrialDesign specification.

    Returns:
        ParallelTrialResult with all PK/PD/stats outputs.
    """
    rng = np.random.default_rng(design.seed)
    result = ParallelTrialResult()

    all_subjects: list[SubjectResult] = []
    arm_data: dict[str, dict[str, list[float]]] = {}

    for grp in design.dose_groups:
        arm_data[grp.name] = {"cmax": [], "auc": [], "effect": [], "dose": grp.dose_mg}
        for subj_id in range(grp.n_subjects):
            cl_i, vd_i, cmax, tmax, auc_inf, t_half = _simulate_subject_pk(
                dose_mg=grp.dose_mg,
                route=grp.route,
                cl_pop=design.cl_L_per_h,
                vd_pop=design.vd_L,
                ka_pop=design.ka_per_h,
                f_bio=design.f_bioavail,
                bsv_cv_cl=design.bsv_cv_cl,
                bsv_cv_vd=design.bsv_cv_vd,
                bsv_cv_ka=design.bsv_cv_ka,
                t_end_h=design.t_end_h,
                rng=rng,
            )
            pd_input_val = cmax if design.pd_input == "cmax" else auc_inf
            effect = _emax_effect(pd_input_val, design.emax, design.ec50_mg_L, design.hill)
            responder = effect >= _RESPONSE_THRESHOLD * design.emax

            subj = SubjectResult(
                subject_id=subj_id,
                arm_name=grp.name,
                dose_mg=grp.dose_mg,
                cl_L_per_h=round(cl_i, 3),
                vd_L=round(vd_i, 2),
                cmax_mg_L=round(cmax, 4),
                tmax_h=round(tmax, 3),
                auc_mg_h_L=round(auc_inf, 4),
                t_half_h=round(t_half, 3),
                effect=round(effect, 4),
                responder=responder,
            )
            all_subjects.append(subj)
            arm_data[grp.name]["cmax"].append(cmax)
            arm_data[grp.name]["auc"].append(auc_inf)
            arm_data[grp.name]["effect"].append(effect)

    result.subjects = all_subjects
    result.total_subjects = len(all_subjects)

    # Build arm summaries
    arm_summaries: list[ArmSummary] = []
    for grp in design.dose_groups:
        d = arm_data[grp.name]
        cmax_arr = np.array(d["cmax"]) + 1e-12
        auc_arr = np.array(d["auc"]) + 1e-12
        eff_arr = np.array(d["effect"])

        cmax_gmean = float(np.exp(np.mean(np.log(cmax_arr))))
        cmax_gsd = float(np.exp(np.std(np.log(cmax_arr))))
        auc_gmean = float(np.exp(np.mean(np.log(auc_arr))))
        auc_gsd = float(np.exp(np.std(np.log(auc_arr))))

        arm_summaries.append(ArmSummary(
            name=grp.name,
            dose_mg=grp.dose_mg,
            n=grp.n_subjects,
            cmax_mean=round(cmax_gmean, 4),
            cmax_gsd=round(cmax_gsd, 4),
            auc_mean=round(auc_gmean, 4),
            auc_gsd=round(auc_gsd, 4),
            effect_mean=round(float(np.mean(eff_arr)), 4),
            effect_sd=round(float(np.std(eff_arr, ddof=1)), 4),
            responder_pct=round(float(np.mean([s.responder for s in all_subjects
                                               if s.arm_name == grp.name])) * 100, 1),
        ))
    result.arms = arm_summaries

    # One-way ANOVA
    cmax_groups = [np.array(arm_data[g.name]["cmax"]) for g in design.dose_groups]
    auc_groups = [np.array(arm_data[g.name]["auc"]) for g in design.dose_groups]
    eff_groups = [np.array(arm_data[g.name]["effect"]) for g in design.dose_groups]

    result.anova_p_cmax = round(_one_way_anova(cmax_groups), 6)
    result.anova_p_auc = round(_one_way_anova(auc_groups), 6)
    result.anova_p_effect = round(_one_way_anova(eff_groups), 6)

    # Pairwise Bonferroni vs. reference arm (first / lowest-dose group)
    ref_grp = design.dose_groups[0]
    ref_effect = np.array(arm_data[ref_grp.name]["effect"])
    ref_auc = np.array(arm_data[ref_grp.name]["auc"])
    ref_cmax = np.array(arm_data[ref_grp.name]["cmax"])

    non_ref = [(g.name, g.dose_mg) for g in design.dose_groups[1:]]
    pw_effect = _pairwise_tests(
        ref_effect,
        [(n, d, np.array(arm_data[n]["effect"])) for n, d in non_ref],
        "effect", design.alpha,
    )
    pw_auc = _pairwise_tests(
        ref_auc,
        [(n, d, np.array(arm_data[n]["auc"])) for n, d in non_ref],
        "auc", design.alpha,
    )
    pw_cmax = _pairwise_tests(
        ref_cmax,
        [(n, d, np.array(arm_data[n]["cmax"])) for n, d in non_ref],
        "cmax", design.alpha,
    )
    result.pairwise = pw_effect + pw_auc + pw_cmax

    # Dose-response fit
    doses_arr = np.array([s.dose_mg for s in arm_summaries])
    effects_arr = np.array([s.effect_mean for s in arm_summaries])
    result.dose_response = _fit_dose_response(doses_arr, effects_arr)

    # Optimal dose (ED80)
    if result.dose_response.fit_success:
        result.optimal_dose_mg = result.dose_response.ed80_mg
    else:
        # Fallback: lowest dose arm with effect_mean > 80% of max
        max_eff = max(s.effect_mean for s in arm_summaries)
        for s in arm_summaries:
            if s.dose_mg > 0 and s.effect_mean >= 0.8 * max_eff:
                result.optimal_dose_mg = s.dose_mg
                break

    # Interim analysis
    if 0 < design.interim_fraction < 1.0 and len(design.dose_groups) >= 2:
        n_interim = int(design.dose_groups[-1].n_subjects * design.interim_fraction)
        n_total = design.dose_groups[-1].n_subjects
        eff_int = np.array(arm_data[design.dose_groups[-1].name]["effect"])[:n_interim]
        ref_int = ref_effect[:n_interim]
        cp = _conditional_power(eff_int, ref_int, n_total, design.alpha)
        result.interim = InterimAnalysis(
            conducted=True,
            fraction_enrolled=design.interim_fraction,
            conditional_power=cp,
            futility_stop=cp < 0.20,
            n_interim=n_interim * 2,
        )

    # Trial success: at least one pairwise effect comparison significant
    effect_pairs = [p for p in result.pairwise if p.endpoint == "effect"]
    result.trial_success = any(p.significant for p in effect_pairs)

    return result
