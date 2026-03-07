"""Phase 360 — PK/PD Efficacy Score Calculator.

Computes a composite PK/PD efficacy score for a drug candidate based on
steady-state exposure and pharmacodynamic target attainment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PKPDEfficacyResult:
    """Result of PK/PD efficacy scoring."""

    drug_name: str
    dose_mg: float
    pd_model: str
    auc_ss_mg_h_per_L: float
    cmax_ss_mg_L: float
    trough_ss_mg_L: float
    efficacy_score: float
    safety_score: float
    composite_score: float
    time_above_mec_pct: float
    time_above_mtc_pct: float
    therapeutic_window_adequate: bool
    recommended_dose_adjustment: str
    notes: str


_VALID_PD_MODELS = {"emax", "time_above_ec50", "auc_ec50"}


def _simulate_ss_pk(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float,
    tau_h: float,
    n_doses: int = 10,
    dt_h: float = 0.01,
) -> tuple[list, list]:
    """Simulate oral PK for n_doses and return last-interval time/conc arrays."""
    ke = cl_L_per_h / vd_L
    c = 0.0
    a_gut = 0.0

    all_times: list = []
    all_conc: list = []

    t_global = 0.0
    steps_per_interval = int(round(tau_h / dt_h))

    for dose_num in range(n_doses):
        # Administer dose
        a_gut += f_oral * dose_mg

        for _step in range(steps_per_interval):
            absorption_flux = ka_per_h * a_gut
            da_gut = -absorption_flux
            dc = absorption_flux / vd_L - ke * c

            a_gut = max(0.0, a_gut + da_gut * dt_h)
            c = max(0.0, c + dc * dt_h)
            t_global += dt_h

            if dose_num == n_doses - 1:
                all_times.append(t_global - (n_doses - 1) * tau_h)
                all_conc.append(c)

    return all_times, all_conc


def _hill(c: float, emax: float, ec50: float, hill_n: float) -> float:
    """Hill equation for Emax PD model."""
    if c <= 0.0:
        return 0.0
    cn = c**hill_n
    ec50n = ec50**hill_n
    return emax * cn / (ec50n + cn)


def compute_pkpd_efficacy_score(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float,
    tau_h: float,
    ec50_mg_L: float,
    emax: float,
    hill_n: float = 1.0,
    pd_model: str = "emax",
    mec_mg_L: float = 0.1,
    mtc_mg_L: float = 10.0,
    target_auc_mg_h_per_L: float = 100.0,
) -> PKPDEfficacyResult:
    """Compute PK/PD efficacy score from steady-state simulation.

    Parameters
    ----------
    drug_name:
        Name of the drug candidate.
    dose_mg:
        Dose in mg.
    cl_L_per_h:
        Clearance (L/h).
    vd_L:
        Volume of distribution (L).
    ka_per_h:
        Oral absorption rate constant (h^-1).
    f_oral:
        Oral bioavailability fraction (0-1).
    tau_h:
        Dosing interval (h).
    ec50_mg_L:
        EC50 in mg/L.
    emax:
        Maximum effect (0-1].
    hill_n:
        Hill coefficient.
    pd_model:
        PD model: "emax", "time_above_ec50", or "auc_ec50".
    mec_mg_L:
        Minimum effective concentration (mg/L).
    mtc_mg_L:
        Minimum toxic concentration (mg/L).
    target_auc_mg_h_per_L:
        Target AUC for reference.

    Returns
    -------
    PKPDEfficacyResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if ec50_mg_L <= 0:
        raise ValueError(f"ec50_mg_L must be > 0, got {ec50_mg_L}")
    if not (0.0 < emax <= 1.0):
        raise ValueError(f"emax must be in (0, 1], got {emax}")
    if pd_model not in _VALID_PD_MODELS:
        raise ValueError(f"pd_model must be one of {_VALID_PD_MODELS}, got '{pd_model}'")

    dt_h = 0.01
    times, conc = _simulate_ss_pk(
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ka_per_h=ka_per_h,
        f_oral=f_oral,
        tau_h=tau_h,
        n_doses=10,
        dt_h=dt_h,
    )

    if not conc:
        raise RuntimeError("Simulation produced no data points.")

    # AUC (trapezoidal)
    auc_ss = sum(
        0.5 * (conc[i] + conc[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )

    cmax_ss = max(conc)
    trough_ss = min(conc)
    css_avg = auc_ss / tau_h if tau_h > 0 else 0.0

    # Time above MEC and MTC
    n_above_mec = sum(1 for c in conc if c > mec_mg_L)
    n_above_mtc = sum(1 for c in conc if c > mtc_mg_L)
    total_pts = len(conc)

    time_above_mec_pct = (n_above_mec / total_pts * 100.0) if total_pts > 0 else 0.0
    time_above_mtc_pct = (n_above_mtc / total_pts * 100.0) if total_pts > 0 else 0.0

    # PD / Efficacy score
    if pd_model == "emax":
        e_at_avg = _hill(css_avg, emax, ec50_mg_L, hill_n)
        efficacy_score = e_at_avg * 100.0
    elif pd_model == "time_above_ec50":
        n_above_ec50 = sum(1 for c in conc if c > ec50_mg_L)
        ft_above = n_above_ec50 / total_pts if total_pts > 0 else 0.0
        efficacy_score = ft_above * 100.0
    else:  # auc_ec50
        efficacy_index = auc_ss / ec50_mg_L
        efficacy_score = min(100.0, efficacy_index * 10.0)

    # Safety score
    therapeutic_window_score = 100.0 - time_above_mtc_pct
    safety_score = max(0.0, therapeutic_window_score)

    # Composite score
    composite_score = 0.7 * efficacy_score + 0.3 * safety_score
    composite_score = max(0.0, min(100.0, composite_score))

    # Therapeutic window adequate
    therapeutic_window_adequate = composite_score > 50.0 and time_above_mtc_pct < 10.0

    # Dose adjustment recommendation
    if time_above_mtc_pct >= 10.0:
        recommended_dose_adjustment = "decrease"
    elif composite_score > 50.0:
        recommended_dose_adjustment = "maintain"
    else:
        recommended_dose_adjustment = "increase"

    notes = (
        f"PK/PD efficacy scoring for {drug_name} ({pd_model} model). "
        f"Dose: {dose_mg} mg q{tau_h}h. "
        f"Css_avg: {css_avg:.3f} mg/L, AUC_ss: {auc_ss:.2f} mg.h/L. "
        f"Efficacy: {efficacy_score:.1f}/100, Safety: {safety_score:.1f}/100, "
        f"Composite: {composite_score:.1f}/100."
    )

    return PKPDEfficacyResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        pd_model=pd_model,
        auc_ss_mg_h_per_L=auc_ss,
        cmax_ss_mg_L=cmax_ss,
        trough_ss_mg_L=trough_ss,
        efficacy_score=efficacy_score,
        safety_score=safety_score,
        composite_score=composite_score,
        time_above_mec_pct=time_above_mec_pct,
        time_above_mtc_pct=time_above_mtc_pct,
        therapeutic_window_adequate=therapeutic_window_adequate,
        recommended_dose_adjustment=recommended_dose_adjustment,
        notes=notes,
    )


def optimize_dose(
    drug_name: str,
    doses_mg: list,
    cl: float,
    vd: float,
    ka: float,
    f_oral: float,
    tau_h: float,
    ec50: float,
    emax: float,
    hill_n: float = 1.0,
    pd_model: str = "emax",
    mec: float = 0.1,
    mtc: float = 10.0,
    target_auc: float = 100.0,
) -> list:
    """Evaluate multiple doses and return results sorted by composite_score (descending).

    Parameters
    ----------
    drug_name:
        Drug name.
    doses_mg:
        List of doses (mg) to evaluate.
    cl, vd, ka, f_oral, tau_h:
        PK parameters.
    ec50, emax, hill_n:
        PD parameters.
    pd_model:
        PD model type.
    mec, mtc:
        MEC and MTC thresholds.
    target_auc:
        Target AUC reference.

    Returns
    -------
    list of PKPDEfficacyResult sorted by composite_score descending.
    """
    results = []
    for dose in doses_mg:
        try:
            result = compute_pkpd_efficacy_score(
                drug_name=drug_name,
                dose_mg=dose,
                cl_L_per_h=cl,
                vd_L=vd,
                ka_per_h=ka,
                f_oral=f_oral,
                tau_h=tau_h,
                ec50_mg_L=ec50,
                emax=emax,
                hill_n=hill_n,
                pd_model=pd_model,
                mec_mg_L=mec,
                mtc_mg_L=mtc,
                target_auc_mg_h_per_L=target_auc,
            )
            results.append(result)
        except ValueError:
            continue

    results.sort(key=lambda r: r.composite_score, reverse=True)
    return results
