"""
Phase 212 — Eye Drop PK Model
Ocular drug delivery: 3-compartment Forward Euler ODE
(tear film → cornea → aqueous humor).
Pure Python (stdlib only: math, dataclasses).
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Physiological constants
# ---------------------------------------------------------------------------

_V_TEAR_ML = 0.007  # mL  — tear film volume
_V_CORNEA_ML = 0.2  # mL  — corneal tissue volume (approximate)
_V_AQUEOUS_ML = 0.25  # mL  — aqueous humor volume

_K_DRAIN = 30.0  # /h  — tear drainage (0.5/min → 30/h)
_K_TC_BASE = 0.1  # /h  — tear→cornea transfer (baseline)
_K_CT = 0.05  # /h  — cornea→tear back-diffusion
_K_CA = 0.08  # /h  — cornea→aqueous
_K_AQ_ELIM = 0.03  # /h  — aqueous humor turnover / elimination

_DT_H = 1.0 / 3600.0  # default time step: 1 second in hours
_T_END_H = 6.0  # default simulation end: 6 hours


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EyeDropPKResult:
    """Plain dataclass (has list fields)."""

    drug_name: str
    dose_mg: float
    corneal_perm_factor: float
    times_h: list[float]
    c_tear_mg_mL: list[float]
    c_cornea_mg_g: list[float]
    c_aqueous_mg_mL: list[float]
    cmax_aqueous_mg_mL: float
    tmax_aqueous_h: float
    auc_aqueous_mg_h_per_mL: float
    bioavailability_corneal_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1])
    return auc


def _validate_inputs(dose_mg: float, corneal_perm_factor: float) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0.1 <= corneal_perm_factor <= 10.0):
        raise ValueError(f"corneal_perm_factor must be in [0.1, 10.0], got {corneal_perm_factor}")


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_eye_drop_pk(
    drug_name: str,
    dose_mg: float,
    corneal_perm_factor: float = 1.0,
    t_end_h: float = _T_END_H,
    dt_h: float = _DT_H,
) -> EyeDropPKResult:
    """
    Simulate ocular PK from a single eye drop.

    Parameters
    ----------
    drug_name : str
    dose_mg : float — dose in mg
    corneal_perm_factor : float — multiplier on k_tc (0.1–10.0)
    t_end_h : float — simulation duration (hours)
    dt_h : float — forward Euler time step (hours)

    Returns
    -------
    EyeDropPKResult
    """
    _validate_inputs(dose_mg, corneal_perm_factor)

    # Effective rate constants
    k_tc = _K_TC_BASE * corneal_perm_factor

    # Initial conditions
    c_tear = dose_mg / _V_TEAR_ML  # mg/mL
    c_cornea = 0.0  # mg/g  (density ~1 g/mL assumed)
    c_aqueous = 0.0  # mg/mL

    times: list[float] = [0.0]
    ct_list: list[float] = [c_tear]
    cc_list: list[float] = [c_cornea]
    ca_list: list[float] = [c_aqueous]

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps):
        # ODEs
        dc_tear = (
            -_K_DRAIN * c_tear
            - k_tc * c_tear * _V_TEAR_ML
            + _K_CT * c_cornea * _V_CORNEA_ML / _V_TEAR_ML
        )
        dc_cornea = k_tc * c_tear * _V_TEAR_ML / _V_CORNEA_ML - _K_CT * c_cornea - _K_CA * c_cornea
        dc_aqueous = _K_CA * c_cornea * _V_CORNEA_ML / _V_AQUEOUS_ML - _K_AQ_ELIM * c_aqueous

        c_tear = max(c_tear + dc_tear * dt_h, 0.0)
        c_cornea = max(c_cornea + dc_cornea * dt_h, 0.0)
        c_aqueous = max(c_aqueous + dc_aqueous * dt_h, 0.0)

        t += dt_h
        times.append(t)
        ct_list.append(c_tear)
        cc_list.append(c_cornea)
        ca_list.append(c_aqueous)

    # PK metrics for aqueous
    cmax_aqueous = max(ca_list)
    tmax_idx = ca_list.index(cmax_aqueous)
    tmax_aqueous = times[tmax_idx]
    auc_aqueous = _trapz_auc(times, ca_list)

    # Corneal bioavailability: fraction of dose that entered cornea
    # Estimate via AUC of corneal concentration × k_ca × V_cornea
    auc_cornea = _trapz_auc(times, cc_list)
    amount_to_aqueous = _K_CA * auc_cornea * _V_CORNEA_ML  # mg
    bioavailability_corneal_pct = min(100.0 * amount_to_aqueous / dose_mg, 100.0)

    notes = (
        f"3-cpt eye drop PK: tear→cornea→aqueous. "
        f"corneal_perm_factor={corneal_perm_factor:.2f}. "
        f"Tmax_aqueous={tmax_aqueous:.3f} h, Cmax_aqueous={cmax_aqueous:.4f} mg/mL."
    )

    return EyeDropPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        corneal_perm_factor=corneal_perm_factor,
        times_h=times,
        c_tear_mg_mL=ct_list,
        c_cornea_mg_g=cc_list,
        c_aqueous_mg_mL=ca_list,
        cmax_aqueous_mg_mL=cmax_aqueous,
        tmax_aqueous_h=tmax_aqueous,
        auc_aqueous_mg_h_per_mL=auc_aqueous,
        bioavailability_corneal_pct=bioavailability_corneal_pct,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Compare drop volumes
# ---------------------------------------------------------------------------


def compare_drop_volumes(
    drug_name: str,
    conc_mg_mL: float,
    volumes_uL: list[float],
    corneal_perm_factor: float = 1.0,
    t_end_h: float = _T_END_H,
    dt_h: float = _DT_H,
) -> list[EyeDropPKResult]:
    """
    Compare eye drop volumes at fixed drug concentration.

    Parameters
    ----------
    drug_name : str
    conc_mg_mL : float — drug concentration in mg/mL
    volumes_uL : list of float — drop volumes in µL
    corneal_perm_factor : float
    t_end_h : float
    dt_h : float

    Returns
    -------
    List of EyeDropPKResult sorted by auc_aqueous_mg_h_per_mL ascending.
    """
    results = []
    for vol_uL in volumes_uL:
        dose_mg = conc_mg_mL * vol_uL / 1000.0  # µL → mL
        result = simulate_eye_drop_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            corneal_perm_factor=corneal_perm_factor,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    return sorted(results, key=lambda r: r.auc_aqueous_mg_h_per_mL)
