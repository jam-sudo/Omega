"""Retinal permeation model through the blood-retinal barrier (BRB).

Models drug permeation through the retinal blood barrier and distribution
within retinal layers (vitreous, retina, choroid) for intravitreal and
systemic administration routes.

References
----------
- Cunha-Vaz J, Prog Retin Eye Res. 2004;23:541-559 (BRB structure)
- Del Amo EM et al., Prog Retin Eye Res. 2017;57:134-185 (ocular drug delivery)
- Pitkanen L et al., Eur J Pharm Sci. 2005;25:355-360 (BRB permeability)
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_V_VITREOUS_ML = 4.0  # mL, default vitreous volume
_V_RETINA_ML = 0.5  # mL, retinal volume
_V_CHOROID_ML = 0.3  # mL, choroid volume
_SA_RETINA_CM2 = 0.01  # cm², effective retinal surface area factor
_CL_CHOROID_FAST = 0.1  # /h, choroid elimination rate constant

# BRB permeability bounds [cm/s]
_P_BRB_MIN = 1e-8
_P_BRB_MAX = 1e-4


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RetinalPermeationResult:
    """Result of retinal permeation simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list
    c_vitreous_mg_mL: list
    c_retina_mg_mL: list
    c_choroid_mg_mL: list
    cmax_retina_mg_mL: float
    tmax_retina_h: float
    auc_retina_mg_h_per_mL: float
    t_half_vitreous_h: float
    brb_permeability_cm_s: float
    retinal_penetration_ratio: float
    notes: str


# ---------------------------------------------------------------------------
# Helper: trapezoidal AUC
# ---------------------------------------------------------------------------
def _trapz_auc(x: list, y: list) -> float:
    """Manual trapezoidal integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def simulate_retinal_permeation(
    drug_name: str,
    dose_mg: float,
    route: str,
    logP: float,
    mw_Da: float,
    cl_vitreous_per_h: float = 0.02,
    vd_vitreous_mL: float = 4.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> RetinalPermeationResult:
    """Simulate drug permeation through the blood-retinal barrier.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Administered dose in mg.
    route:
        Administration route: ``"intravitreal"`` or ``"systemic"``.
    logP:
        Octanol-water partition coefficient (log scale).
    mw_Da:
        Molecular weight in Daltons.
    cl_vitreous_per_h:
        Vitreous clearance rate constant [/h]. Default 0.02 /h.
    vd_vitreous_mL:
        Vitreous volume [mL]. Default 4.0 mL.
    t_end_h:
        Simulation end time [h]. Default 168 h.
    dt_h:
        Integration time step [h]. Default 0.5 h.

    Returns
    -------
    RetinalPermeationResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if route not in ("intravitreal", "systemic"):
        raise ValueError(f"route must be 'intravitreal' or 'systemic', got {route!r}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if cl_vitreous_per_h <= 0:
        raise ValueError(f"cl_vitreous_per_h must be > 0, got {cl_vitreous_per_h}")

    # BRB permeability [cm/s]
    p_brb = 1e-6 * (1.0 + logP) * (300.0 / max(200.0, mw_Da))
    p_brb = max(_P_BRB_MIN, min(_P_BRB_MAX, p_brb))

    # Convert permeability to rate constant [/h]
    # k_brb = P_brb [cm/s] * 3600 [s/h] * 0.01 [cm² surface area factor]
    k_brb = p_brb * 3600.0 * _SA_RETINA_CM2

    # Volume ratios for mass-conserving transfer
    vol_vit_to_ret = vd_vitreous_mL / _V_RETINA_ML  # = 4.0/0.5 = 8
    vol_vit_to_cho = vd_vitreous_mL / _V_CHOROID_ML  # = 4.0/0.3 ≈ 13.33

    # Initial conditions
    if route == "intravitreal":
        c_vit = dose_mg / vd_vitreous_mL
        c_ret = 0.0
        c_cho = 0.0
    else:  # systemic
        c_vit = 0.0
        c_ret = 0.0
        c_cho = dose_mg / 50.0  # diluted in systemic volume equivalent

    # Storage
    n_steps = int(t_end_h / dt_h) + 1
    times: list = []
    c_vit_list: list = []
    c_ret_list: list = []
    c_cho_list: list = []

    times.append(0.0)
    c_vit_list.append(c_vit)
    c_ret_list.append(c_ret)
    c_cho_list.append(c_cho)

    # Forward Euler integration
    t = 0.0
    for _ in range(n_steps - 1):
        if route == "intravitreal":
            dc_vit = -k_brb * c_vit - cl_vitreous_per_h * c_vit
            dc_ret = k_brb * c_vit * vol_vit_to_ret - k_brb * c_ret
            dc_cho = k_brb * c_vit * vol_vit_to_cho * 0.1 - _CL_CHOROID_FAST * c_cho
        else:  # systemic: choroid feeds retina
            dc_cho = -0.05 * c_cho
            dc_ret = k_brb * c_cho - k_brb * c_ret
            dc_vit = k_brb * c_ret * 0.1 - cl_vitreous_per_h * c_vit

        c_vit = max(0.0, c_vit + dc_vit * dt_h)
        c_ret = max(0.0, c_ret + dc_ret * dt_h)
        c_cho = max(0.0, c_cho + dc_cho * dt_h)
        t += dt_h

        times.append(t)
        c_vit_list.append(c_vit)
        c_ret_list.append(c_ret)
        c_cho_list.append(c_cho)

    # Derived metrics — retina
    cmax_retina = max(c_ret_list)
    tmax_idx = c_ret_list.index(cmax_retina)
    tmax_retina = times[tmax_idx]
    auc_retina = _trapz_auc(times, c_ret_list)
    auc_vitreous = _trapz_auc(times, c_vit_list)

    # Half-life in vitreous: first time concentration drops to 50% of initial
    c_vit_initial = c_vit_list[0]
    t_half_vit = t_end_h  # default if never reached
    if c_vit_initial > 0:
        half_val = 0.5 * c_vit_initial
        for i in range(1, len(times)):
            if c_vit_list[i] <= half_val:
                # Linear interpolation
                dt_seg = times[i] - times[i - 1]
                dc_seg = c_vit_list[i] - c_vit_list[i - 1]
                if abs(dc_seg) > 1e-15:
                    frac = (half_val - c_vit_list[i - 1]) / dc_seg
                    t_half_vit = times[i - 1] + frac * dt_seg
                else:
                    t_half_vit = times[i]
                break

    penetration_ratio = auc_retina / auc_vitreous if auc_vitreous > 0 else 0.0

    notes = f"BRB permeability {p_brb:.2e} cm/s; k_brb={k_brb:.4f} /h; route={route}"

    return RetinalPermeationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_vitreous_mg_mL=c_vit_list,
        c_retina_mg_mL=c_ret_list,
        c_choroid_mg_mL=c_cho_list,
        cmax_retina_mg_mL=cmax_retina,
        tmax_retina_h=tmax_retina,
        auc_retina_mg_h_per_mL=auc_retina,
        t_half_vitreous_h=t_half_vit,
        brb_permeability_cm_s=p_brb,
        retinal_penetration_ratio=penetration_ratio,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Route comparison
# ---------------------------------------------------------------------------
def compare_routes_retinal(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
) -> list:
    """Compare intravitreal vs systemic routes for retinal drug delivery.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Dose in mg.
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons.

    Returns
    -------
    list
        [intravitreal_result, systemic_result]
    """
    iv_result = simulate_retinal_permeation(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="intravitreal",
        logP=logP,
        mw_Da=mw_Da,
    )
    sys_result = simulate_retinal_permeation(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="systemic",
        logP=logP,
        mw_Da=mw_Da,
    )
    return [iv_result, sys_result]
