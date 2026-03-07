"""Physiological Flow-Limited PBPK model.

Implements a simplified multi-organ PBPK using flow-limited (instantaneous
equilibrium) distribution. Drug is assumed to equilibrate instantly between
blood and each tissue according to the tissue-to-plasma partition coefficient
(Kp). The blood compartment is the central driver; tissue concentrations are
derived algebraically from C_blood × Kp.

Reference organs and volumes (Rodgers et al. 2005, Table 1):
    Liver:  1.7 L
    Kidney: 0.3 L
    Muscle: 35  L
    Fat:    15  L
    Blood:   5  L (central compartment)

Effective total volume of distribution:
    Vd_eff = V_blood + sum(Kp_i × V_i)

Clearance pathways:
    CL_total = CL_hepatic + CL_renal

For IV dosing:
    dC_blood/dt = -CL_total / Vd_eff * C_blood

For oral dosing (first-order absorption):
    dA_gut/dt  = -ka * A_gut
    dC_blood/dt = ka * A_gut * F / Vd_eff - CL_total / Vd_eff * C_blood
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz

# ---------------------------------------------------------------------------
# Physiological constants (reference human, Rodgers et al.)
# ---------------------------------------------------------------------------

_V_LIVER_L = 1.7
_V_KIDNEY_L = 0.3
_V_MUSCLE_L = 35.0
_V_FAT_L = 15.0
_V_BLOOD_L = 5.0

_ORGAN_VOLUMES = {
    "liver": _V_LIVER_L,
    "kidney": _V_KIDNEY_L,
    "muscle": _V_MUSCLE_L,
    "fat": _V_FAT_L,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowLimitedPBPKResult:
    """Result of a flow-limited PBPK simulation."""

    drug_name: str
    dose_mg: float
    route: str  # "iv" or "oral"

    # Tissue partition coefficients used
    kp_liver: float
    kp_kidney: float
    kp_muscle: float
    kp_fat: float

    # Derived volumes
    vd_effective_L: float  # effective Vd = V_blood + sum(Kp_i * V_i)

    # Simulation output
    times_h: list[float]
    c_blood_mg_L: list[float]

    # Derived PK metrics (blood)
    cmax_blood: float  # mg/L
    tmax_h: float  # h
    auc_blood: float  # mg·h/L

    # Tissue concentrations at landmark times
    c_liver_at_1h: float  # mg/L
    c_kidney_at_1h: float  # mg/L
    c_fat_at_24h: float  # mg/L

    # Terminal half-life from Vd/CL
    t_half_h: float  # h

    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulation function
# ---------------------------------------------------------------------------


def simulate_flow_limited_pbpk(
    drug_name: str,
    dose_mg: float,
    kp_liver: float = 3.0,
    kp_kidney: float = 2.0,
    kp_muscle: float = 0.5,
    kp_fat: float = 10.0,
    cl_hepatic_L_per_h: float = 5.0,
    cl_renal_L_per_h: float = 1.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> FlowLimitedPBPKResult:
    """Run a flow-limited PBPK simulation.

    Parameters
    ----------
    drug_name:
        Identifier for the drug (used in result labelling only).
    dose_mg:
        Dose in milligrams; must be > 0.
    kp_liver, kp_kidney, kp_muscle, kp_fat:
        Tissue-to-blood partition coefficients; all must be > 0.
    cl_hepatic_L_per_h:
        Hepatic clearance (L/h); must be > 0.
    cl_renal_L_per_h:
        Renal clearance (L/h); must be > 0.
    route:
        Dosing route: ``"iv"`` (bolus) or ``"oral"`` (first-order absorption).
    ka_per_h:
        Oral absorption rate constant (h⁻¹); only used for oral route.
    f_oral:
        Oral bioavailability fraction (0–1); only used for oral route.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step for forward Euler integration (h).

    Returns
    -------
    FlowLimitedPBPKResult
        Full time-course and summary PK metrics.
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    for name, kp in [
        ("kp_liver", kp_liver),
        ("kp_kidney", kp_kidney),
        ("kp_muscle", kp_muscle),
        ("kp_fat", kp_fat),
    ]:
        if kp <= 0:
            raise ValueError(f"{name} must be > 0, got {kp}")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError(f"cl_hepatic_L_per_h must be > 0, got {cl_hepatic_L_per_h}")
    if cl_renal_L_per_h <= 0:
        raise ValueError(f"cl_renal_L_per_h must be > 0, got {cl_renal_L_per_h}")
    route = route.lower()
    if route not in {"iv", "oral"}:
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")

    # --- Derived parameters ---
    kp_map = {
        "liver": kp_liver,
        "kidney": kp_kidney,
        "muscle": kp_muscle,
        "fat": kp_fat,
    }
    vd_effective = _V_BLOOD_L + sum(kp_map[org] * _ORGAN_VOLUMES[org] for org in kp_map)
    cl_total = cl_hepatic_L_per_h + cl_renal_L_per_h
    k_elim = cl_total / vd_effective  # overall elimination rate constant (h⁻¹)
    t_half = math.log(2.0) / k_elim

    # --- Forward Euler integration ---
    n_steps = max(int(round(t_end_h / dt_h)), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    # State: C_blood (mg/L); for oral route also A_gut (mg)
    c_blood = np.zeros(len(times))
    a_gut = np.zeros(len(times))

    if route == "iv":
        c_blood[0] = dose_mg / vd_effective
    else:
        # Oral: absorbed dose enters gut compartment
        a_gut[0] = dose_mg * f_oral
        c_blood[0] = 0.0

    for i in range(len(times) - 1):
        cb = c_blood[i]
        ag = a_gut[i]

        if route == "oral":
            d_ag = -ka_per_h * ag * dt_h
            absorption_rate = ka_per_h * ag  # mg/h
            d_cb = (absorption_rate / vd_effective - k_elim * cb) * dt_h
            a_gut[i + 1] = max(0.0, ag + d_ag)
        else:
            d_cb = -k_elim * cb * dt_h

        c_blood[i + 1] = max(0.0, cb + d_cb)

    # --- PK metrics ---
    cmax_blood = float(np.max(c_blood))
    tmax_idx = int(np.argmax(c_blood))
    tmax_h = float(times[tmax_idx])
    auc_blood = float(np_trapz(c_blood, times))

    # Tissue concentrations at landmark times
    def _c_tissue_at(tissue: str, t_h: float) -> float:
        idx = int(round(t_h / dt_h))
        idx = min(idx, len(times) - 1)
        return kp_map[tissue] * float(c_blood[idx])

    c_liver_at_1h = _c_tissue_at("liver", 1.0)
    c_kidney_at_1h = _c_tissue_at("kidney", 1.0)
    c_fat_at_24h = _c_tissue_at("fat", min(24.0, t_end_h))

    notes: list[str] = [
        f"Vd_eff = {vd_effective:.2f} L; CL_total = {cl_total:.2f} L/h; t½ = {t_half:.2f} h.",
        f"Route: {route}.",
    ]

    return FlowLimitedPBPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        kp_liver=kp_liver,
        kp_kidney=kp_kidney,
        kp_muscle=kp_muscle,
        kp_fat=kp_fat,
        vd_effective_L=vd_effective,
        times_h=times.tolist(),
        c_blood_mg_L=c_blood.tolist(),
        cmax_blood=cmax_blood,
        tmax_h=tmax_h,
        auc_blood=auc_blood,
        c_liver_at_1h=c_liver_at_1h,
        c_kidney_at_1h=c_kidney_at_1h,
        c_fat_at_24h=c_fat_at_24h,
        t_half_h=t_half,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Tissue concentration query
# ---------------------------------------------------------------------------


def tissue_concentration_at_t(
    result: FlowLimitedPBPKResult,
    tissue: str,
    t_h: float,
) -> float:
    """Return estimated tissue concentration at a given time.

    Applies flow-limited equilibrium: C_tissue(t) = Kp_tissue × C_blood(t).

    Parameters
    ----------
    result:
        A :class:`FlowLimitedPBPKResult` from :func:`simulate_flow_limited_pbpk`.
    tissue:
        One of ``"liver"``, ``"kidney"``, ``"muscle"``, ``"fat"``.
    t_h:
        Time of interest (h). Clamped to [0, t_end].

    Returns
    -------
    float
        Tissue concentration (mg/L).
    """
    valid_tissues = {"liver", "kidney", "muscle", "fat"}
    if tissue not in valid_tissues:
        raise ValueError(f"tissue must be one of {sorted(valid_tissues)}, got '{tissue}'")

    kp_map = {
        "liver": result.kp_liver,
        "kidney": result.kp_kidney,
        "muscle": result.kp_muscle,
        "fat": result.kp_fat,
    }
    kp = kp_map[tissue]

    # Find nearest time index via linear interpolation
    times = result.times_h
    c_blood = result.c_blood_mg_L

    if t_h <= times[0]:
        return kp * c_blood[0]
    if t_h >= times[-1]:
        return kp * c_blood[-1]

    # Linear interpolation between bracketing time points
    # Use numpy for efficiency
    t_arr = np.asarray(times)
    c_arr = np.asarray(c_blood)
    c_at_t = float(np.interp(t_h, t_arr, c_arr))
    return kp * c_at_t
