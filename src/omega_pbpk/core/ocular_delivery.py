"""Sustained-release ocular drug delivery: implants and intravitreal injections.

Focus: Long-acting ocular drug delivery systems for posterior segment diseases
(e.g., wet AMD, DME, uveitis) using biodegradable/non-biodegradable implants
and repeated intravitreal (IVT) injections.

Compartment structure
---------------------
- Implant reservoir (mg drug remaining)
- Vitreous humor (V = 4.0 mL default)

PD: VEGF inhibition modelled as Emax with EC50 = 0.05 mg/mL (ranibizumab-like).

References
----------
- Kang-Mieler JJ et al., Eye (Lond). 2020;34:1371-1379 (ocular drug delivery systems)
- Xu J et al., Invest Ophthalmol Vis Sci. 2013;54:1576-1585 (IVT PK)
- Shah M et al., Expert Opin Drug Deliv. 2010;7(9):1159-74 (implant systems)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

_EC50_DEFAULT_MG_ML = 0.05  # ranibizumab-like VEGF inhibition EC50
_EMAX_DEFAULT = 100.0  # % effect at saturation


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OcularImplantResult:
    """Results from sustained-release ocular implant simulation.

    Attributes
    ----------
    drug_name : Drug name.
    drug_load_mg : Total drug loaded in implant (mg).
    implant_type : "biodegradable" or "non_biodegradable".
    times_days : Simulation time points (days).
    c_vitreous_mg_mL : Vitreous humor drug concentration (mg/mL).
    effect_pct : VEGF inhibition effect (%, 0-100).
    days_above_ec50 : Number of days with C_vit > EC50.
    auc_vitreous : AUC in vitreous (mg*day/mL).
    peak_c : Peak vitreous concentration (mg/mL).
    notes : Summary text.
    """

    drug_name: str
    drug_load_mg: float
    implant_type: str
    times_days: list[float]
    c_vitreous_mg_mL: list[float]
    effect_pct: list[float]
    days_above_ec50: float
    auc_vitreous: float
    peak_c: float
    notes: str


@dataclass(frozen=True)
class IntravitreInjResult:
    """Results from intravitreal injection simulation.

    Attributes
    ----------
    drug_name : Drug name.
    dose_mg : Injected dose (mg).
    times_days : Simulation time points (days).
    c_vitreous_mg_mL : Vitreous humor drug concentration (mg/mL).
    effect_pct : VEGF inhibition effect (%, 0-100).
    days_above_ec50 : Number of days with C_vit > EC50.
    auc_vitreous : AUC in vitreous (mg*day/mL).
    peak_c : Peak vitreous concentration (mg/mL).
    notes : Summary text.
    """

    drug_name: str
    dose_mg: float
    times_days: list[float]
    c_vitreous_mg_mL: list[float]
    effect_pct: list[float]
    days_above_ec50: float
    auc_vitreous: float
    peak_c: float
    notes: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_common(drug_load_or_dose: float, k_release: float, ke: float, vd: float) -> None:
    """Shared parameter validation for implant and injection simulations."""
    if drug_load_or_dose <= 0:
        raise ValueError("drug_load_mg / dose_mg must be > 0")
    if k_release <= 0:
        raise ValueError("k_release_per_day must be > 0")
    if ke <= 0:
        raise ValueError("ke_vitreous_per_day must be > 0")
    if vd <= 0:
        raise ValueError("vd_vitreous_mL must be > 0")


# ---------------------------------------------------------------------------
# Implant simulation
# ---------------------------------------------------------------------------


def simulate_ocular_implant(
    drug_name: str,
    drug_load_mg: float,
    implant_type: str = "biodegradable",
    k_release_per_day: float = 0.02,
    ke_vitreous_per_day: float = 0.3,
    vd_vitreous_mL: float = 4.0,
    ec50_mg_mL: float = _EC50_DEFAULT_MG_ML,
    emax_pct: float = _EMAX_DEFAULT,
    t_end_days: float = 180.0,
    dt_days: float = 0.5,
) -> OcularImplantResult:
    """Simulate sustained-release ocular implant drug delivery.

    Parameters
    ----------
    drug_name : Drug name.
    drug_load_mg : Total drug loaded in implant (mg). Must be > 0.
    implant_type : "biodegradable" (first-order release) or
        "non_biodegradable" (zero-order release). Default "biodegradable".
    k_release_per_day : Release rate constant (day^-1 for biodegradable;
        fractional rate for non-biodegradable). Must be > 0.
    ke_vitreous_per_day : Vitreous elimination rate constant (day^-1). Must be > 0.
    vd_vitreous_mL : Volume of vitreous humor (mL). Must be > 0.
    ec50_mg_mL : EC50 for pharmacodynamic effect (mg/mL). Default 0.05.
    emax_pct : Maximum effect (%). Default 100.
    t_end_days : Simulation duration (days). Default 180.
    dt_days : Forward Euler time step (days). Default 0.5.

    Returns
    -------
    OcularImplantResult

    Raises
    ------
    ValueError : If any required parameter is invalid or implant_type unknown.
    """
    if implant_type not in ("biodegradable", "non_biodegradable"):
        raise ValueError(
            f"implant_type must be 'biodegradable' or 'non_biodegradable', got '{implant_type}'"
        )
    _validate_common(drug_load_mg, k_release_per_day, ke_vitreous_per_day, vd_vitreous_mL)
    if ec50_mg_mL <= 0:
        raise ValueError("ec50_mg_mL must be > 0")

    n_steps = max(int(t_end_days / dt_days), 1)
    t = np.linspace(0.0, t_end_days, n_steps + 1)

    a_implant = np.zeros(n_steps + 1)  # mg remaining in implant
    c_vit = np.zeros(n_steps + 1)  # mg/mL in vitreous
    a_implant[0] = drug_load_mg

    # Non-biodegradable: constant release rate until implant depleted
    # Modelled as zero-order release = k_release_per_day * drug_load_mg (mg/day)
    # until a_implant exhausted.
    zero_order_rate = k_release_per_day * drug_load_mg  # mg/day

    for i in range(n_steps):
        a_imp = a_implant[i]
        c = c_vit[i]

        if implant_type == "biodegradable":
            release_rate = k_release_per_day * a_imp  # mg/day (first-order)
        else:
            # Zero-order until depleted
            release_rate = zero_order_rate if a_imp > 0.0 else 0.0

        da_imp = -release_rate * dt_days
        dc_vit = (release_rate / vd_vitreous_mL - ke_vitreous_per_day * c) * dt_days

        a_implant[i + 1] = max(0.0, a_imp + da_imp)
        c_vit[i + 1] = max(0.0, c + dc_vit)

    # Pharmacodynamics: Emax model
    effect = emax_pct * c_vit / (ec50_mg_mL + c_vit)

    # Metrics
    dt_arr = np.diff(t)
    above_ec50 = float(np.sum(dt_arr[c_vit[:-1] > ec50_mg_mL]))
    auc_vit = float(np_trapz(c_vit, t))
    peak_c = float(np.max(c_vit))

    notes = (
        f"Ocular implant: {drug_name}, load={drug_load_mg}mg ({implant_type}). "
        f"k_release={k_release_per_day}/day, ke={ke_vitreous_per_day}/day. "
        f"Peak C_vit={peak_c:.4f} mg/mL. "
        f"Days above EC50={above_ec50:.1f}. AUC={auc_vit:.4f} mg*day/mL."
    )

    return OcularImplantResult(
        drug_name=drug_name,
        drug_load_mg=drug_load_mg,
        implant_type=implant_type,
        times_days=t.tolist(),
        c_vitreous_mg_mL=c_vit.tolist(),
        effect_pct=effect.tolist(),
        days_above_ec50=above_ec50,
        auc_vitreous=auc_vit,
        peak_c=peak_c,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Intravitreal injection simulation
# ---------------------------------------------------------------------------


def simulate_intravitreal_injection(
    drug_name: str,
    dose_mg: float,
    ke_vitreous_per_day: float = 0.3,
    vd_vitreous_mL: float = 4.0,
    ec50_mg_mL: float = _EC50_DEFAULT_MG_ML,
    emax_pct: float = _EMAX_DEFAULT,
    t_end_days: float = 90.0,
    dt_days: float = 0.1,
) -> IntravitreInjResult:
    """Simulate single intravitreal injection with monoexponential vitreous PK.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Injected dose (mg). Must be > 0.
    ke_vitreous_per_day : Elimination rate constant from vitreous (day^-1). Must be > 0.
    vd_vitreous_mL : Vitreous volume (mL). Must be > 0.
    ec50_mg_mL : EC50 for pharmacodynamic effect (mg/mL). Default 0.05.
    emax_pct : Maximum effect (%). Default 100.
    t_end_days : Simulation duration (days). Default 90.
    dt_days : Forward Euler time step (days). Default 0.1.

    Returns
    -------
    IntravitreInjResult

    Raises
    ------
    ValueError : If any required parameter is invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ke_vitreous_per_day <= 0:
        raise ValueError("ke_vitreous_per_day must be > 0")
    if vd_vitreous_mL <= 0:
        raise ValueError("vd_vitreous_mL must be > 0")
    if ec50_mg_mL <= 0:
        raise ValueError("ec50_mg_mL must be > 0")

    n_steps = max(int(t_end_days / dt_days), 1)
    t = np.linspace(0.0, t_end_days, n_steps + 1)

    c_vit = np.zeros(n_steps + 1)
    c_vit[0] = dose_mg / vd_vitreous_mL  # bolus IVT: C[0] = dose / Vd

    for i in range(n_steps):
        dc = -ke_vitreous_per_day * c_vit[i] * dt_days
        c_vit[i + 1] = max(0.0, c_vit[i] + dc)

    # Pharmacodynamics
    effect = emax_pct * c_vit / (ec50_mg_mL + c_vit)

    # Metrics
    dt_arr = np.diff(t)
    above_ec50 = float(np.sum(dt_arr[c_vit[:-1] > ec50_mg_mL]))
    auc_vit = float(np_trapz(c_vit, t))
    peak_c = float(c_vit[0])

    notes = (
        f"IVT injection: {drug_name}, dose={dose_mg}mg. "
        f"C0={peak_c:.4f} mg/mL, ke={ke_vitreous_per_day}/day. "
        f"Days above EC50={above_ec50:.1f}. AUC={auc_vit:.4f} mg*day/mL."
    )

    return IntravitreInjResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_days=t.tolist(),
        c_vitreous_mg_mL=c_vit.tolist(),
        effect_pct=effect.tolist(),
        days_above_ec50=above_ec50,
        auc_vitreous=auc_vit,
        peak_c=peak_c,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Delivery system comparison
# ---------------------------------------------------------------------------


def compare_delivery_systems(
    drug_name: str,
    dose_mg: float,
    drug_load_implant_mg: float,
    ke_vitreous_per_day: float = 0.3,
    vd_vitreous_mL: float = 4.0,
    ec50_mg_mL: float = _EC50_DEFAULT_MG_ML,
    t_end_days: float = 180.0,
) -> dict:
    """Compare three ocular drug delivery systems.

    Systems compared:
    1. Single IVT injection (dose_mg over t_end_days)
    2. Monthly IVT injections (dose_mg every 28 days)
    3. Biodegradable implant (drug_load_implant_mg over t_end_days)

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose for each IVT injection (mg). Must be > 0.
    drug_load_implant_mg : Total drug loaded in biodegradable implant (mg). Must be > 0.
    ke_vitreous_per_day : Vitreous elimination rate (day^-1). Default 0.3.
    vd_vitreous_mL : Vitreous volume (mL). Default 4.0.
    ec50_mg_mL : EC50 (mg/mL). Default 0.05.
    t_end_days : Simulation duration (days). Default 180.

    Returns
    -------
    dict with keys:
        "single_ivt": IntravitreInjResult
        "monthly_ivt": dict with keys "auc_vitreous", "days_above_ec50", "n_injections"
        "biodegradable_implant": OcularImplantResult
        "summary": dict comparing AUC and days above EC50
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if drug_load_implant_mg <= 0:
        raise ValueError("drug_load_implant_mg must be > 0")

    dt = 0.1

    # 1. Single IVT injection
    single = simulate_intravitreal_injection(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ke_vitreous_per_day=ke_vitreous_per_day,
        vd_vitreous_mL=vd_vitreous_mL,
        ec50_mg_mL=ec50_mg_mL,
        t_end_days=t_end_days,
        dt_days=dt,
    )

    # 2. Monthly IVT injections (superposition every 28 days)
    n_injections = max(1, int(t_end_days / 28.0))
    n_steps = max(int(t_end_days / dt), 1)
    t_arr = np.linspace(0.0, t_end_days, n_steps + 1)
    c_monthly = np.zeros(n_steps + 1)

    for inj_idx in range(n_injections):
        t_dose = inj_idx * 28.0
        c0 = dose_mg / vd_vitreous_mL
        # Add monoexponential contribution from this injection
        c_monthly += c0 * np.exp(-ke_vitreous_per_day * np.maximum(t_arr - t_dose, 0.0)) * (
            t_arr >= t_dose
        )

    dt_arr = np.diff(t_arr)
    monthly_auc = float(np_trapz(c_monthly, t_arr))
    monthly_above = float(np.sum(dt_arr[c_monthly[:-1] > ec50_mg_mL]))
    monthly_result = {
        "auc_vitreous": monthly_auc,
        "days_above_ec50": monthly_above,
        "n_injections": n_injections,
    }

    # 3. Biodegradable implant
    implant = simulate_ocular_implant(
        drug_name=drug_name,
        drug_load_mg=drug_load_implant_mg,
        implant_type="biodegradable",
        ke_vitreous_per_day=ke_vitreous_per_day,
        vd_vitreous_mL=vd_vitreous_mL,
        ec50_mg_mL=ec50_mg_mL,
        t_end_days=t_end_days,
        dt_days=dt,
    )

    summary = {
        "single_ivt_auc": single.auc_vitreous,
        "single_ivt_days_above_ec50": single.days_above_ec50,
        "monthly_ivt_auc": monthly_auc,
        "monthly_ivt_days_above_ec50": monthly_above,
        "monthly_ivt_n_injections": n_injections,
        "implant_auc": implant.auc_vitreous,
        "implant_days_above_ec50": implant.days_above_ec50,
        "best_days_above_ec50": max(
            single.days_above_ec50, monthly_above, implant.days_above_ec50
        ),
    }

    return {
        "single_ivt": single,
        "monthly_ivt": monthly_result,
        "biodegradable_implant": implant,
        "summary": summary,
    }


__all__ = [
    "OcularImplantResult",
    "IntravitreInjResult",
    "simulate_ocular_implant",
    "simulate_intravitreal_injection",
    "compare_delivery_systems",
]
