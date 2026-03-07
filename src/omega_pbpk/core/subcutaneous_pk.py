"""Subcutaneous (SC) drug absorption pharmacokinetics.

Models drug absorption after SC injection through two parallel pathways:
  1. Direct capillary absorption (predominant for small molecules)
  2. Lymphatic absorption (important for large molecules/biologics)

Reference: Supersaxo et al. (1990), Kagan et al. (2007).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "SCPKResult",
    "simulate_sc_pk",
    "sc_bioavailability",
    "compare_sc_iv",
    # Phase 392 additions
    "SCPKSimResult",
    "simulate_sc_depot",
    "compare_sc_formulations",
]


@dataclass(frozen=True)
class SCPKResult:
    """Result of a subcutaneous PK simulation.

    Attributes
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Administered SC dose (mg).
    mw_kDa : float
        Drug molecular weight (kDa).
    f_lymph : float
        Fraction of dose absorbed via lymphatic route (0-1).
    times_h : list[float]
        Time points (h).
    c_plasma : list[float]
        Plasma concentration at each time point (ng/mL or consistent units).
    cmax : float
        Maximum plasma concentration.
    tmax_h : float
        Time to maximum plasma concentration (h).
    auc : float
        AUC from 0 to t_end using trapezoidal rule.
    f_via_lymph : float
        Fraction of absorbed drug delivered via lymphatic route.
    f_via_capillary : float
        Fraction of absorbed drug delivered via capillary route.
    notes : str
        Simulation notes.
    """

    drug_name: str
    dose_mg: float
    mw_kDa: float
    f_lymph: float
    times_h: list[float]
    c_plasma: list[float]
    cmax: float
    tmax_h: float
    auc: float
    f_via_lymph: float
    f_via_capillary: float
    notes: str


def _auto_f_lymph(mw_kDa: float) -> float:
    """Auto-calculate lymphatic absorption fraction based on molecular weight.

    Sigmoid relationship: f_lymph = 1 / (1 + exp(-(MW_kDa - 3) / 1))
    - Small molecules (<1 kDa): ~0% lymphatic
    - Transition zone (~3 kDa)
    - Large molecules (>10 kDa): ~50% lymphatic

    Parameters
    ----------
    mw_kDa : float
        Molecular weight in kDa.

    Returns
    -------
    float
        Fraction absorbed via lymphatic route (0 to ~0.5).
    """
    f = 1.0 / (1.0 + math.exp(-(mw_kDa - 3.0) / 1.0))
    # Scale to [0, 0.5]: f_lymph = sigmoid * 0.5
    return f * 0.5


def simulate_sc_pk(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float = 0.5,
    ka_capillary_per_h: float = 0.5,
    ka_lymph_per_h: float = 0.02,
    f_lymph: float | None = None,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> SCPKResult:
    """Simulate subcutaneous drug absorption pharmacokinetics.

    SC depot splits into two parallel absorption pathways:
      - Capillary: A_cap[0] = dose * (1 - f_lymph), ka_cap per h
      - Lymph:     A_lymph[0] = dose * f_lymph, ka_lymph per h

    Both pathways deliver drug to central plasma compartment.

    ODE system (Forward Euler):
        dA_cap/dt   = -ka_cap * A_cap
        dA_lymph/dt = -ka_lymph * A_lymph
        dC/dt       = ka_cap*A_cap/Vd + ka_lymph*A_lymph/Vd - (CL/Vd)*C

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        SC dose (mg, must be > 0).
    mw_kDa : float
        Molecular weight (kDa, must be > 0). Used to auto-calculate f_lymph.
    ka_capillary_per_h : float
        Capillary absorption rate constant (1/h, must be > 0).
    ka_lymph_per_h : float
        Lymphatic absorption rate constant (1/h, must be > 0).
    f_lymph : float | None
        Fraction absorbed via lymphatics (0-1). If None, auto-calculated
        from MW using sigmoid: f_lymph = sigmoid(mw_kDa, center=3, scale=1) * 0.5
    cl_L_per_h : float
        Total body clearance (L/h, must be > 0).
    vd_L : float
        Volume of distribution (L, must be > 0).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step for Forward Euler integration (h).

    Returns
    -------
    SCPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if ka_capillary_per_h <= 0:
        raise ValueError("ka_capillary_per_h must be > 0")
    if ka_lymph_per_h <= 0:
        raise ValueError("ka_lymph_per_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if f_lymph is not None and not (0.0 <= f_lymph <= 1.0):
        raise ValueError("f_lymph must be in [0, 1]")

    # Auto-calculate f_lymph if not provided
    if f_lymph is None:
        f_lymph_val = _auto_f_lymph(mw_kDa)
    else:
        f_lymph_val = float(f_lymph)

    f_cap = 1.0 - f_lymph_val

    # Initial conditions
    A_cap = dose_mg * f_cap    # mg in capillary depot
    A_lymph = dose_mg * f_lymph_val  # mg in lymphatic depot
    C = 0.0                    # plasma concentration (mg/L)

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    n_steps = max(int(round(t_end_h / dt_h)), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)
    c_arr = np.empty(n_steps + 1)
    c_arr[0] = C

    # Track absorbed amounts for route fractions
    absorbed_cap = 0.0
    absorbed_lymph = 0.0

    # Forward Euler integration
    for i in range(n_steps):
        dA_cap = -ka_capillary_per_h * A_cap
        dA_lymph = -ka_lymph_per_h * A_lymph
        dC = (
            ka_capillary_per_h * A_cap / vd_L
            + ka_lymph_per_h * A_lymph / vd_L
            - ke * C
        )

        # Absorbed in this step
        absorbed_cap += -dA_cap * dt_h
        absorbed_lymph += -dA_lymph * dt_h

        A_cap = max(0.0, A_cap + dA_cap * dt_h)
        A_lymph = max(0.0, A_lymph + dA_lymph * dt_h)
        C = max(0.0, C + dC * dt_h)
        c_arr[i + 1] = C

    # Compute PK metrics
    cmax = float(np.max(c_arr))
    tmax_h = float(times[int(np.argmax(c_arr))])
    auc = float(np_trapz(c_arr, times))

    total_absorbed = absorbed_cap + absorbed_lymph
    if total_absorbed > 0:
        f_via_cap = absorbed_cap / total_absorbed
        f_via_lymph_final = absorbed_lymph / total_absorbed
    else:
        f_via_cap = f_cap
        f_via_lymph_final = f_lymph_val

    notes = (
        f"Forward Euler SC PK. MW={mw_kDa} kDa, f_lymph={f_lymph_val:.3f}. "
        f"ka_cap={ka_capillary_per_h}/h, ka_lymph={ka_lymph_per_h}/h. "
        f"CL={cl_L_per_h} L/h, Vd={vd_L} L, ke={ke:.4f}/h. dt={dt_h} h."
    )

    return SCPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_kDa=mw_kDa,
        f_lymph=f_lymph_val,
        times_h=times.tolist(),
        c_plasma=c_arr.tolist(),
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        f_via_lymph=f_via_lymph_final,
        f_via_capillary=f_via_cap,
        notes=notes,
    )


def sc_bioavailability(
    mw_kDa: float,
    logP: float,
    solubility_mg_mL: float,
) -> float:
    """Estimate SC bioavailability of a drug.

    SC bioavailability is generally high for small hydrophilic drugs,
    and decreases for:
      - Lipophilic drugs (local depot formation at injection site)
      - Large molecules (proteolytic degradation in SC tissue)

    Formula:
        f_sc = 0.9 * (1 - 0.3 * max(0, logP - 2) / 5) * min(1.0, 1 / (1 + 0.01*mw_kDa))
    Clamped to [0.1, 0.99].

    Parameters
    ----------
    mw_kDa : float
        Molecular weight (kDa).
    logP : float
        Lipophilicity.
    solubility_mg_mL : float
        Aqueous solubility (mg/mL). Not directly used in formula but validated.

    Returns
    -------
    float
        Estimated SC bioavailability fraction (0.1 to 0.99).
    """
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if solubility_mg_mL < 0:
        raise ValueError("solubility_mg_mL must be >= 0")

    # Lipophilicity penalty: applies when logP > 2
    lipo_factor = 1.0 - 0.3 * max(0.0, logP - 2.0) / 5.0

    # MW penalty: large molecules have lower SC bioavailability due to
    # proteolytic degradation
    mw_factor = min(1.0, 1.0 / (1.0 + 0.01 * mw_kDa))

    f_sc = 0.9 * lipo_factor * mw_factor

    # Clamp to [0.1, 0.99]
    return max(0.1, min(0.99, f_sc))


def compare_sc_iv(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float,
    cl_L_per_h: float,
    vd_L: float,
) -> dict:
    """Compare SC vs IV pharmacokinetics for the same drug.

    For IV: instantaneous bolus, analytical 1-cpt solution.
    For SC: simulate_sc_pk with default parameters.

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Dose (mg, same for both routes).
    mw_kDa : float
        Molecular weight (kDa).
    cl_L_per_h : float
        Total body clearance (L/h).
    vd_L : float
        Volume of distribution (L).

    Returns
    -------
    dict
        Keys: ``auc_sc``, ``auc_iv``, ``auc_ratio``, ``cmax_sc``, ``cmax_iv``,
        ``cmax_ratio``, ``tmax_sc_h``, ``f_lymph``.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    t_end_h = max(48.0, 5.0 * vd_L / cl_L_per_h)

    sc_result = simulate_sc_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_kDa=mw_kDa,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
    )

    # IV: C(t) = (dose/Vd) * exp(-ke*t), AUC = dose/CL
    ke = cl_L_per_h / vd_L
    auc_iv = dose_mg / cl_L_per_h
    cmax_iv = dose_mg / vd_L  # at t=0

    auc_ratio = sc_result.auc / auc_iv if auc_iv > 0 else float("nan")
    cmax_ratio = sc_result.cmax / cmax_iv if cmax_iv > 0 else float("nan")

    return {
        "drug_name": drug_name,
        "auc_sc": sc_result.auc,
        "auc_iv": auc_iv,
        "auc_ratio": auc_ratio,
        "cmax_sc": sc_result.cmax,
        "cmax_iv": cmax_iv,
        "cmax_ratio": cmax_ratio,
        "tmax_sc_h": sc_result.tmax_h,
        "f_lymph": sc_result.f_lymph,
        "ke_per_h": ke,
    }


# ---------------------------------------------------------------------------
# Phase 392: Simplified SC PK with lag time and explicit f_sc bioavailability
# ---------------------------------------------------------------------------


@dataclass
class SCPKSimResult:
    """Result of simulate_sc_depot simulation.

    Attributes
    ----------
    drug_name : str
        Drug identifier.
    dose_mg : float
        SC dose (mg).
    ka_sc_per_h : float
        SC absorption rate constant (1/h).
    f_sc : float
        SC bioavailability fraction.
    lag_time_h : float
        Lag time before absorption begins (h).
    times_h : list[float]
        Simulation time points (h).
    c_sc_mg_L : list[float]
        SC plasma concentration profile (mg/L).
    c_iv_mg_L : list[float]
        IV reference concentration profile (mg/L).
    cmax_sc : float
        SC peak concentration (mg/L).
    tmax_sc_h : float
        Time to SC peak (h).
    auc_sc : float
        SC AUC 0→t_end (mg·h/L), trapezoidal.
    auc_iv : float
        IV reference AUC 0→t_end (mg·h/L), trapezoidal.
    bioavailability_ratio : float
        auc_sc / auc_iv ≈ f_sc (linear PK).
    t_half_h : float
        Elimination half-life (h) = ln(2) / ke.
    notes : str
        Simulation summary.
    """

    drug_name: str
    dose_mg: float
    ka_sc_per_h: float
    f_sc: float
    lag_time_h: float
    times_h: list[float]
    c_sc_mg_L: list[float]
    c_iv_mg_L: list[float]
    cmax_sc: float
    tmax_sc_h: float
    auc_sc: float
    auc_iv: float
    bioavailability_ratio: float
    t_half_h: float
    notes: str


def simulate_sc_depot(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_sc_per_h: float,
    f_sc: float = 1.0,
    lag_time_h: float = 0.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> SCPKSimResult:
    """Simulate SC PK with explicit f_sc, lag time, and IV reference comparison.

    SC depot model:
      - Effective absorbed dose = f_sc * dose_mg
      - Depot (A_sc) absorbs via first-order ka_sc_per_h after lag_time_h
      - Plasma 1-compartment: dC/dt = ka_sc * A_sc / Vd - ke * C  (t > lag_time_h)
      - Before lag_time_h: dC/dt = -ke * C (no absorption)

    IV reference (same dose, instantaneous bolus):
      - c_iv(t) = (dose_mg / Vd) * exp(-ke * t)

    AUC computed via trapezoidal rule.

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Total SC dose (mg, must be > 0).
    cl_L_per_h : float
        Total body clearance (L/h, must be > 0).
    vd_L : float
        Volume of distribution (L, must be > 0).
    ka_sc_per_h : float
        SC absorption rate constant (1/h, must be > 0).
    f_sc : float
        SC bioavailability (0 < f_sc <= 1.0).
    lag_time_h : float
        Lag time before absorption starts (h, >= 0).
    t_end_h : float
        Simulation end time (h, must be > 0).
    dt_h : float
        Forward Euler time step (h, must be > 0).

    Returns
    -------
    SCPKSimResult
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if ka_sc_per_h <= 0.0:
        raise ValueError(f"ka_sc_per_h must be > 0, got {ka_sc_per_h}")
    if not (0.0 < f_sc <= 1.0):
        raise ValueError(f"f_sc must be in (0, 1], got {f_sc}")
    if lag_time_h < 0.0:
        raise ValueError(f"lag_time_h must be >= 0, got {lag_time_h}")
    if t_end_h <= 0.0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0.0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    ke = cl_L_per_h / vd_L
    t_half_h = math.log(2.0) / ke

    # Effective SC depot amount (mg)
    a_sc = f_sc * dose_mg
    c_sc = 0.0  # plasma SC concentration (mg/L)

    n_steps = max(int(round(t_end_h / dt_h)), 1)
    dt_actual = t_end_h / n_steps

    times: list[float] = []
    c_sc_arr: list[float] = []
    c_iv_arr: list[float] = []

    t = 0.0
    for i in range(n_steps + 1):
        t = i * dt_actual
        times.append(t)

        # IV reference: analytical
        c_iv = (dose_mg / vd_L) * math.exp(-ke * t)
        c_iv_arr.append(max(0.0, c_iv))
        c_sc_arr.append(max(0.0, c_sc))

        if i < n_steps:
            # Forward Euler step
            absorbing = t >= lag_time_h
            if absorbing:
                d_a_sc = -ka_sc_per_h * a_sc
                flux = ka_sc_per_h * a_sc / vd_L
            else:
                d_a_sc = 0.0
                flux = 0.0

            d_c_sc = flux - ke * c_sc

            a_sc = max(0.0, a_sc + d_a_sc * dt_actual)
            c_sc = max(0.0, c_sc + d_c_sc * dt_actual)

    # PK metrics for SC
    cmax_sc = max(c_sc_arr)
    tmax_sc_h = times[c_sc_arr.index(cmax_sc)]

    # AUC via trapezoidal rule (manual)
    auc_sc = _trapz(times, c_sc_arr)
    auc_iv = _trapz(times, c_iv_arr)

    bioavailability_ratio = auc_sc / auc_iv if auc_iv > 0.0 else float("nan")

    notes = (
        f"SC depot model for '{drug_name}'. f_sc={f_sc:.3f}, lag={lag_time_h} h, "
        f"ka={ka_sc_per_h:.3f}/h, ke={ke:.4f}/h, t½={t_half_h:.2f} h. "
        f"Cmax_SC={cmax_sc:.4g} mg/L at {tmax_sc_h:.2f} h. "
        f"AUC_SC={auc_sc:.4g}, AUC_IV={auc_iv:.4g} mg·h/L. "
        f"F_ratio={bioavailability_ratio:.3f}."
    )

    return SCPKSimResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_sc_per_h=ka_sc_per_h,
        f_sc=f_sc,
        lag_time_h=lag_time_h,
        times_h=times,
        c_sc_mg_L=c_sc_arr,
        c_iv_mg_L=c_iv_arr,
        cmax_sc=cmax_sc,
        tmax_sc_h=tmax_sc_h,
        auc_sc=auc_sc,
        auc_iv=auc_iv,
        bioavailability_ratio=bioavailability_ratio,
        t_half_h=t_half_h,
        notes=notes,
    )


def _trapz(xs: list[float], ys: list[float]) -> float:
    """Manual trapezoidal integration."""
    if len(xs) < 2:
        return 0.0
    total = 0.0
    for i in range(len(xs) - 1):
        total += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    return total


def compare_sc_formulations(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    formulations: list[dict],
) -> list[dict]:
    """Compare multiple SC formulations and rank by AUC descending.

    Each formulation dict must contain:
      - ``name`` (str): formulation label
      - ``ka_sc`` (float): SC absorption rate constant (1/h)
      - ``f_sc`` (float): SC bioavailability fraction
      - ``lag_h`` (float): lag time (h)

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        SC dose (mg, must be > 0).
    cl_L_per_h : float
        Total body clearance (L/h, must be > 0).
    vd_L : float
        Volume of distribution (L, must be > 0).
    formulations : list[dict]
        List of formulation parameter dicts.

    Returns
    -------
    list[dict]
        Formulation results sorted by AUC descending. Each dict contains:
        ``name``, ``ka_sc``, ``f_sc``, ``lag_h``, ``auc_sc``, ``auc_iv``,
        ``cmax_sc``, ``tmax_sc_h``, ``bioavailability_ratio``, ``t_half_h``.
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not formulations:
        raise ValueError("formulations list must not be empty")

    required_keys = {"name", "ka_sc", "f_sc", "lag_h"}
    for idx, form in enumerate(formulations):
        missing = required_keys - set(form.keys())
        if missing:
            raise ValueError(
                f"Formulation {idx} is missing required keys: {missing}"
            )

    results: list[dict] = []
    for form in formulations:
        name = str(form["name"])
        ka_sc = float(form["ka_sc"])
        f_sc_val = float(form["f_sc"])
        lag_h = float(form["lag_h"])

        sim = simulate_sc_depot(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            ka_sc_per_h=ka_sc,
            f_sc=f_sc_val,
            lag_time_h=lag_h,
        )
        results.append({
            "name": name,
            "ka_sc": ka_sc,
            "f_sc": f_sc_val,
            "lag_h": lag_h,
            "auc_sc": sim.auc_sc,
            "auc_iv": sim.auc_iv,
            "cmax_sc": sim.cmax_sc,
            "tmax_sc_h": sim.tmax_sc_h,
            "bioavailability_ratio": sim.bioavailability_ratio,
            "t_half_h": sim.t_half_h,
        })

    # Sort by AUC descending
    results.sort(key=lambda d: d["auc_sc"], reverse=True)
    return results
