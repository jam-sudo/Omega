"""Bioequivalence statistical analysis — TOST procedure, GMR, 90% CI."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "BEResult",
    "compute_gmr_and_ci",
    "tost_test",
    "simulate_be_study",
]


@dataclass(frozen=True)
class BEResult:
    n_test: int
    n_ref: int
    gmr_auc: float
    ci_lower_auc: float
    ci_upper_auc: float
    is_bioequivalent: bool
    delta_log: float
    se_log: float
    t_critical: float
    df: int
    notes: str


class LCG:
    def __init__(self, seed: int) -> None:
        self.state = seed

    def next_float(self) -> float:
        self.state = (1664525 * self.state + 1013904223) % (2**32)
        return self.state / (2**32)

    def normal(self) -> float:
        u1 = max(1e-10, self.next_float())
        u2 = self.next_float()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def _t_critical(df: int, alpha: float = 0.05) -> float:
    """One-sided t critical value for 90% CI (alpha = 0.05)."""
    if df >= 30:
        return 1.645
    elif df >= 20:
        return 1.725
    elif df >= 12:
        return 1.782
    elif df >= 8:
        return 1.860
    else:
        return 2.015


def compute_gmr_and_ci(
    test_aucs: list[float],
    ref_aucs: list[float],
    alpha: float = 0.05,
) -> BEResult:
    """Compute geometric mean ratio and 90% CI for bioequivalence assessment."""
    if len(test_aucs) < 2:
        raise ValueError("test_aucs must have at least 2 values")
    if len(ref_aucs) < 2:
        raise ValueError("ref_aucs must have at least 2 values")
    if any(a <= 0 for a in test_aucs):
        raise ValueError("All test AUC values must be > 0")
    if any(a <= 0 for a in ref_aucs):
        raise ValueError("All ref AUC values must be > 0")

    n_t = len(test_aucs)
    n_r = len(ref_aucs)

    log_test = [math.log(a) for a in test_aucs]
    log_ref = [math.log(a) for a in ref_aucs]

    mean_log_test = _mean(log_test)
    mean_log_ref = _mean(log_ref)
    delta = mean_log_test - mean_log_ref

    var_t = _variance(log_test)
    var_r = _variance(log_ref)

    df = n_t + n_r - 2
    if df < 1:
        sp2 = 0.0
    else:
        sp2 = ((n_t - 1) * var_t + (n_r - 1) * var_r) / df

    se = math.sqrt(sp2 * (1.0 / n_t + 1.0 / n_r)) if sp2 >= 0 else 0.0

    t_crit = _t_critical(df, alpha)

    gmr = math.exp(delta)
    ci_lower = math.exp(delta - t_crit * se)
    ci_upper = math.exp(delta + t_crit * se)

    is_be = (ci_lower >= 0.80) and (ci_upper <= 1.25)

    notes = (
        f"GMR={gmr:.4f}, 90% CI=[{ci_lower:.4f}, {ci_upper:.4f}], "
        f"df={df}, BE={'yes' if is_be else 'no'}"
    )

    return BEResult(
        n_test=n_t,
        n_ref=n_r,
        gmr_auc=gmr,
        ci_lower_auc=ci_lower,
        ci_upper_auc=ci_upper,
        is_bioequivalent=is_be,
        delta_log=delta,
        se_log=se,
        t_critical=t_crit,
        df=df,
        notes=notes,
    )


def tost_test(
    test_values: list[float],
    ref_values: list[float],
    lower_limit: float = 0.80,
    upper_limit: float = 1.25,
) -> dict:
    """Two One-Sided Tests (TOST) for bioequivalence."""
    if len(test_values) < 2:
        raise ValueError("test_values must have at least 2 values")
    if len(ref_values) < 2:
        raise ValueError("ref_values must have at least 2 values")
    if any(a <= 0 for a in test_values):
        raise ValueError("All test values must be > 0")
    if any(a <= 0 for a in ref_values):
        raise ValueError("All ref values must be > 0")
    if not (0 < lower_limit < upper_limit):
        raise ValueError("Must have 0 < lower_limit < upper_limit")

    n_t = len(test_values)
    n_r = len(ref_values)

    log_test = [math.log(a) for a in test_values]
    log_ref = [math.log(a) for a in ref_values]

    delta = _mean(log_test) - _mean(log_ref)

    var_t = _variance(log_test)
    var_r = _variance(log_ref)

    df = n_t + n_r - 2
    if df < 1:
        sp2 = 0.0
    else:
        sp2 = ((n_t - 1) * var_t + (n_r - 1) * var_r) / df

    se = math.sqrt(sp2 * (1.0 / n_t + 1.0 / n_r)) if sp2 > 0 else 1e-10

    t_crit = _t_critical(df)

    t_lower = (delta - math.log(lower_limit)) / se
    t_upper = (math.log(upper_limit) - delta) / se

    be_demonstrated = (t_lower > t_crit) and (t_upper > t_crit)

    notes = (
        f"TOST: t_lower={t_lower:.3f}, t_upper={t_upper:.3f}, t_crit={t_crit:.3f}. "
        f"BE {'demonstrated' if be_demonstrated else 'not demonstrated'}."
    )

    return {
        "t_lower": t_lower,
        "t_upper": t_upper,
        "t_crit": t_crit,
        "df": df,
        "be_demonstrated": be_demonstrated,
        "delta_log": delta,
        "se_log": se,
        "notes": notes,
    }


def _simulate_1cpt_auc(
    cl: float,
    vd: float,
    ka: float,
    dose: float,
    f_oral: float,
    t_end: float = 48.0,
    dt: float = 0.1,
) -> float:
    """Simulate 1-compartment oral PK with Forward Euler, return AUC via trapezoid."""
    a_depot = dose * f_oral
    a_central = 0.0
    auc = 0.0
    prev_c = 0.0
    t = 0.0

    while t <= t_end + 1e-9:
        c = a_central / vd if vd > 0 else 0.0
        if t > 0:
            auc += 0.5 * (prev_c + c) * dt
        prev_c = c

        d_depot = -ka * a_depot
        d_central = ka * a_depot - (cl / vd) * a_central

        a_depot += d_depot * dt
        a_central += d_central * dt
        t += dt

    return auc


def simulate_be_study(
    reference_pk_params: dict,
    test_pk_params: dict,
    n_subjects: int = 24,
    seed: int = 42,
) -> dict:
    """Simulate a crossover BE study with LCG-based inter-individual variability.

    reference_pk_params and test_pk_params: dicts with keys cl, vd, ka, dose, f_oral.
    Returns: dict with be_result (BEResult), test_aucs, ref_aucs, summary.
    """
    rng = LCG(seed)

    omega_cl = 0.3
    omega_vd = 0.3

    ref_cl = reference_pk_params["cl"]
    ref_vd = reference_pk_params["vd"]
    ref_ka = reference_pk_params.get("ka", 1.0)
    ref_dose = reference_pk_params["dose"]
    ref_f = reference_pk_params.get("f_oral", 0.8)

    tst_cl = test_pk_params["cl"]
    tst_vd = test_pk_params["vd"]
    tst_ka = test_pk_params.get("ka", 1.0)
    tst_dose = test_pk_params["dose"]
    tst_f = test_pk_params.get("f_oral", 0.8)

    ref_aucs = []
    test_aucs = []

    for _ in range(n_subjects):
        # LCG-based log-normal IIV
        eta_cl = rng.normal()
        eta_vd = rng.normal()

        subj_ref_cl = ref_cl * math.exp(omega_cl * eta_cl)
        subj_ref_vd = ref_vd * math.exp(omega_vd * eta_vd)

        auc_ref = _simulate_1cpt_auc(
            cl=subj_ref_cl,
            vd=subj_ref_vd,
            ka=ref_ka,
            dose=ref_dose,
            f_oral=ref_f,
        )
        ref_aucs.append(auc_ref)

        # Independent IIV for test period (crossover: same subject but different eta)
        eta_cl2 = rng.normal()
        eta_vd2 = rng.normal()

        subj_tst_cl = tst_cl * math.exp(omega_cl * eta_cl2)
        subj_tst_vd = tst_vd * math.exp(omega_vd * eta_vd2)

        auc_tst = _simulate_1cpt_auc(
            cl=subj_tst_cl,
            vd=subj_tst_vd,
            ka=tst_ka,
            dose=tst_dose,
            f_oral=tst_f,
        )
        test_aucs.append(auc_tst)

    be_result = compute_gmr_and_ci(test_aucs, ref_aucs)

    summary = {
        "n_subjects": n_subjects,
        "mean_ref_auc": _mean(ref_aucs),
        "mean_test_auc": _mean(test_aucs),
        "gmr": be_result.gmr_auc,
        "ci_lower": be_result.ci_lower_auc,
        "ci_upper": be_result.ci_upper_auc,
        "be_demonstrated": be_result.is_bioequivalent,
    }

    return {
        "be_result": be_result,
        "test_aucs": test_aucs,
        "ref_aucs": ref_aucs,
        "summary": summary,
    }
