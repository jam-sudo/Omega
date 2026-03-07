"""Two-site target-mediated drug disposition (TMDD) model — Phase 693.

Models drug binding to two distinct target sites (e.g., soluble + membrane-bound
receptors, or two receptor subtypes) using forward Euler ODE integration.

All concentrations are in nM. Drug mg/L is converted to nM using MW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "TwoSiteBindingResult",
    "simulate_two_site_binding",
    "compare_affinity_scenarios",
]


@dataclass
class TwoSiteBindingResult:
    """Result of two-site TMDD simulation.

    Fields
    ------
    drug_name : Name of the drug.
    dose_mg : Dose administered (mg IV bolus).
    mw_kDa : Molecular weight (kDa).
    times_h : Time points (h).
    c_free_mg_L : Free drug concentration (mg/L) over time.
    c_t1_mg_L : Drug-target1 complex (mg/L) over time.
    c_t2_mg_L : Drug-target2 complex (mg/L) over time.
    r1_free_nM : Free target 1 (nM) over time.
    r2_free_nM : Free target 2 (nM) over time.
    cmax_free_mg_L : Maximum free drug concentration (mg/L).
    tmax_h : Time of Cmax (h).
    auc_free : AUC of free drug (mg/L * h), trapezoidal.
    t_half_apparent_h : Apparent terminal half-life (h).
    target1_occupancy_at_cmax : Fraction of target 1 occupied at Cmax.
    target2_occupancy_at_cmax : Fraction of target 2 occupied at Cmax.
    notes : Summary notes.
    """

    drug_name: str
    dose_mg: float
    mw_kDa: float
    times_h: list
    c_free_mg_L: list
    c_t1_mg_L: list
    c_t2_mg_L: list
    r1_free_nM: list
    r2_free_nM: list
    cmax_free_mg_L: float
    tmax_h: float
    auc_free: float
    t_half_apparent_h: float
    target1_occupancy_at_cmax: float
    target2_occupancy_at_cmax: float
    notes: str


def _mg_per_L_to_nM(c_mg_per_L: float, mw_kDa: float) -> float:
    """Convert mg/L to nM using molecular weight in kDa."""
    # c_mg_per_L / (mw_g_per_mol) * 1e6 (to convert g/L to mol/L then to nmol/L)
    # mw_g_per_mol = mw_kDa * 1000
    return c_mg_per_L * 1e6 / (mw_kDa * 1000.0)


def _nM_to_mg_per_L(c_nM: float, mw_kDa: float) -> float:
    """Convert nM to mg/L using molecular weight in kDa."""
    return c_nM * mw_kDa * 1000.0 / 1e6


def _trapezoidal_auc(times: list, values: list) -> float:
    """Compute AUC using trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return auc


def _terminal_half_life(times: list, values: list) -> float:
    """Estimate terminal half-life from last 30% of non-zero time points.

    Uses linear regression on log(concentration) vs time.
    Returns t_half = ln(2) / lambda_z.
    """
    n = len(times)
    start_idx = max(1, int(0.7 * n))
    # Filter positive values in terminal phase
    t_term = []
    c_term = []
    for i in range(start_idx, n):
        if values[i] > 0:
            t_term.append(times[i])
            c_term.append(values[i])

    if len(t_term) < 2:
        return float("nan")

    # Linear regression: log(c) = log(c0) - lambda_z * t
    log_c = [math.log(c) for c in c_term]
    n_t = len(t_term)
    mean_t = sum(t_term) / n_t
    mean_lc = sum(log_c) / n_t
    ss_tt = sum((t_term[i] - mean_t) ** 2 for i in range(n_t))
    if ss_tt == 0:
        return float("nan")
    slope = sum((t_term[i] - mean_t) * (log_c[i] - mean_lc) for i in range(n_t)) / ss_tt
    if slope >= 0:
        return float("nan")
    return math.log(2.0) / (-slope)


def simulate_two_site_binding(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float = 150.0,
    vd_L: float = 5.0,
    ke_per_h: float = 0.01,
    kon1_nM_per_h: float = 0.1,
    koff1_per_h: float = 0.01,
    kint1_per_h: float = 0.02,
    R1_0_nM: float = 10.0,
    kdeg1_per_h: float = 0.1,
    kon2_nM_per_h: float = 0.05,
    koff2_per_h: float = 0.05,
    kint2_per_h: float = 0.01,
    R2_0_nM: float = 5.0,
    kdeg2_per_h: float = 0.05,
    t_end_h: float = 720.0,
    dt_h: float = 0.5,
) -> TwoSiteBindingResult:
    """Simulate two-site target-mediated drug disposition (TMDD).

    IV bolus administration. Free drug, two drug-target complexes, and two
    free receptor concentrations are integrated using forward Euler.

    Parameters
    ----------
    drug_name : Identifier for the drug compound.
    dose_mg : IV bolus dose in mg.
    mw_kDa : Molecular weight in kDa (for unit conversion).
    vd_L : Volume of distribution / plasma volume (L).
    ke_per_h : Linear elimination rate constant (1/h).
    kon1_nM_per_h : Association rate for target 1 (1/(nM·h)).
    koff1_per_h : Dissociation rate for target 1 (1/h).
    kint1_per_h : Internalization rate of drug-target1 complex (1/h).
    R1_0_nM : Baseline target 1 concentration (nM).
    kdeg1_per_h : Degradation rate of free target 1 (1/h).
    kon2_nM_per_h : Association rate for target 2 (1/(nM·h)).
    koff2_per_h : Dissociation rate for target 2 (1/h).
    kint2_per_h : Internalization rate of drug-target2 complex (1/h).
    R2_0_nM : Baseline target 2 concentration (nM).
    kdeg2_per_h : Degradation rate of free target 2 (1/h).
    t_end_h : Simulation duration (h).
    dt_h : Time step for forward Euler (h).

    Returns
    -------
    TwoSiteBindingResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if ke_per_h <= 0:
        raise ValueError("ke_per_h must be > 0")
    if kon1_nM_per_h <= 0:
        raise ValueError("kon1_nM_per_h must be > 0")
    if koff1_per_h <= 0:
        raise ValueError("koff1_per_h must be > 0")
    if kint1_per_h <= 0:
        raise ValueError("kint1_per_h must be > 0")
    if kdeg1_per_h <= 0:
        raise ValueError("kdeg1_per_h must be > 0")
    if kon2_nM_per_h <= 0:
        raise ValueError("kon2_nM_per_h must be > 0")
    if koff2_per_h <= 0:
        raise ValueError("koff2_per_h must be > 0")
    if kint2_per_h <= 0:
        raise ValueError("kint2_per_h must be > 0")
    if kdeg2_per_h <= 0:
        raise ValueError("kdeg2_per_h must be > 0")
    if R1_0_nM < 0:
        raise ValueError("R1_0_nM must be >= 0")
    if R2_0_nM < 0:
        raise ValueError("R2_0_nM must be >= 0")

    # Synthesis rates for baseline steady-state
    ksyn1 = kdeg1_per_h * R1_0_nM
    ksyn2 = kdeg2_per_h * R2_0_nM

    # Initial conditions in nM
    # IV bolus: C_free(0) = dose_mg / vd_L, then convert to nM
    c_free_0_mg_L = dose_mg / vd_L
    c_free_nM = _mg_per_L_to_nM(c_free_0_mg_L, mw_kDa)
    ct1_nM = 0.0
    ct2_nM = 0.0
    r1_nM = R1_0_nM
    r2_nM = R2_0_nM

    # Storage
    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_free_mg_L_list = []
    c_t1_mg_L_list = []
    c_t2_mg_L_list = []
    r1_free_nM_list = []
    r2_free_nM_list = []

    t = 0.0
    for _ in range(n_steps):
        # Record current state (in mg/L for drug, nM for receptors)
        times_h.append(t)
        cf_mg_L = _nM_to_mg_per_L(c_free_nM, mw_kDa)
        ct1_mg_L = _nM_to_mg_per_L(ct1_nM, mw_kDa)
        ct2_mg_L = _nM_to_mg_per_L(ct2_nM, mw_kDa)
        c_free_mg_L_list.append(cf_mg_L)
        c_t1_mg_L_list.append(ct1_mg_L)
        c_t2_mg_L_list.append(ct2_mg_L)
        r1_free_nM_list.append(r1_nM)
        r2_free_nM_list.append(r2_nM)

        # ODEs in nM units
        # Binding terms
        bind1 = kon1_nM_per_h * c_free_nM * r1_nM
        unbind1 = koff1_per_h * ct1_nM
        bind2 = kon2_nM_per_h * c_free_nM * r2_nM
        unbind2 = koff2_per_h * ct2_nM

        d_cfree = -ke_per_h * c_free_nM - bind1 + unbind1 - bind2 + unbind2
        d_ct1 = bind1 - unbind1 - kint1_per_h * ct1_nM
        d_ct2 = bind2 - unbind2 - kint2_per_h * ct2_nM
        d_r1 = ksyn1 - kdeg1_per_h * r1_nM - bind1 + unbind1
        d_r2 = ksyn2 - kdeg2_per_h * r2_nM - bind2 + unbind2

        # Forward Euler
        c_free_nM = max(0.0, c_free_nM + dt_h * d_cfree)
        ct1_nM = max(0.0, ct1_nM + dt_h * d_ct1)
        ct2_nM = max(0.0, ct2_nM + dt_h * d_ct2)
        r1_nM = max(0.0, r1_nM + dt_h * d_r1)
        r2_nM = max(0.0, r2_nM + dt_h * d_r2)

        t += dt_h

    # Compute summary metrics
    cmax_free = max(c_free_mg_L_list)
    tmax_idx = c_free_mg_L_list.index(cmax_free)
    tmax_h = times_h[tmax_idx]

    auc_free = _trapezoidal_auc(times_h, c_free_mg_L_list)

    t_half = _terminal_half_life(times_h, c_free_mg_L_list)
    if math.isnan(t_half) or t_half <= 0:
        # Fallback: use linear elimination estimate
        t_half = math.log(2.0) / ke_per_h if ke_per_h > 0 else float("nan")

    # Target occupancy at Cmax index
    ct1_at_cmax = c_t1_mg_L_list[tmax_idx]
    r1_at_cmax_mg_L = _nM_to_mg_per_L(r1_free_nM_list[tmax_idx], mw_kDa)
    total1_at_cmax = ct1_at_cmax + r1_at_cmax_mg_L
    occ1 = ct1_at_cmax / total1_at_cmax if total1_at_cmax > 0 else 0.0

    ct2_at_cmax = c_t2_mg_L_list[tmax_idx]
    r2_at_cmax_mg_L = _nM_to_mg_per_L(r2_free_nM_list[tmax_idx], mw_kDa)
    total2_at_cmax = ct2_at_cmax + r2_at_cmax_mg_L
    occ2 = ct2_at_cmax / total2_at_cmax if total2_at_cmax > 0 else 0.0

    kd1_nM = koff1_per_h / kon1_nM_per_h
    kd2_nM = koff2_per_h / kon2_nM_per_h

    notes = (
        f"{drug_name}: dose={dose_mg} mg IV, MW={mw_kDa} kDa, "
        f"Cmax_free={cmax_free:.3f} mg/L, AUC_free={auc_free:.2f} mg/L*h, "
        f"t1/2={t_half:.1f} h; "
        f"Target1 Kd={kd1_nM:.3f} nM (occ@Cmax={occ1:.2%}), "
        f"Target2 Kd={kd2_nM:.3f} nM (occ@Cmax={occ2:.2%})"
    )

    return TwoSiteBindingResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_kDa=mw_kDa,
        times_h=times_h,
        c_free_mg_L=c_free_mg_L_list,
        c_t1_mg_L=c_t1_mg_L_list,
        c_t2_mg_L=c_t2_mg_L_list,
        r1_free_nM=r1_free_nM_list,
        r2_free_nM=r2_free_nM_list,
        cmax_free_mg_L=cmax_free,
        tmax_h=tmax_h,
        auc_free=auc_free,
        t_half_apparent_h=t_half,
        target1_occupancy_at_cmax=occ1,
        target2_occupancy_at_cmax=occ2,
        notes=notes,
    )


def compare_affinity_scenarios(
    drug_name: str,
    dose_mg: float,
    scenarios: list,
) -> list:
    """Compare multiple binding affinity scenarios for the same drug.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : IV bolus dose (mg).
    scenarios : List of dicts, each with keys:
        - "label" (str): scenario name
        - "kon1_nM_per_h" (float): association rate for target 1
        - "koff1_per_h" (float): dissociation rate for target 1

    Returns
    -------
    list[TwoSiteBindingResult]
    """
    results = []
    for sc in scenarios:
        kon1 = sc.get("kon1_nM_per_h", 0.1)
        koff1 = sc.get("koff1_per_h", 0.01)
        label = sc.get("label", drug_name)
        res = simulate_two_site_binding(
            drug_name=label,
            dose_mg=dose_mg,
            kon1_nM_per_h=kon1,
            koff1_per_h=koff1,
        )
        results.append(res)
    return results
