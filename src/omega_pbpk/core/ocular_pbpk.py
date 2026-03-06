"""Ocular PBPK model for topical and intravitreal drug pharmacokinetics.

Models ocular drug delivery through 5 compartments plus systemic circulation
for topical (eye drop) and intravitreal injection routes.

Model structure
---------------
Five ocular compartments:

  1. Tear film (V = 0.007 mL) — topical dose entry, rapid drainage
  2. Cornea (V = 0.2 mL) — transcorneal permeation barrier
  3. Aqueous humor (V = 0.25 mL) — anterior chamber
  4. Vitreous humor (V = 4.0 mL) — posterior segment, intravitreal dose entry
  5. Retina/Choroid (V = 0.5 mL) — target tissue for posterior disease

Routes:
  - topical: dose into tear film, absorbed through cornea
  - intravitreal: dose directly into vitreous humor

References
----------
- Worakul N & Robinson JR, Eur J Pharm Biopharm. 1997;44:71-83 (ocular PK)
- Del Amo EM et al., Prog Retin Eye Res. 2017;57:134-185 (ocular barriers)
- Ranta VP & Bhatt DK, Adv Drug Deliv Rev. 2006;58:1131-1135 (tear turnover)
- Hutton-Smith LA et al., J Pharmacokinet Pharmacodyn. 2016;43:541-561 (IVT PK)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

# ---------------------------------------------------------------------------
# Compartment volumes (mL)
# ---------------------------------------------------------------------------
_V_TEAR_ML = 0.007
_V_CORNEA_ML = 0.2
_V_AQUEOUS_ML = 0.25
_V_VITREOUS_ML = 4.0
_V_RETINA_ML = 0.5

# Convert to L for concentration calculations
_V_TEAR_L = _V_TEAR_ML * 0.001
_V_CORNEA_L = _V_CORNEA_ML * 0.001
_V_AQUEOUS_L = _V_AQUEOUS_ML * 0.001
_V_VITREOUS_L = _V_VITREOUS_ML * 0.001
_V_RETINA_L = _V_RETINA_ML * 0.001

# ---------------------------------------------------------------------------
# Default transfer rates (per hour)
# ---------------------------------------------------------------------------
_KD = 30.0          # tear drainage (/h)
_K_TC_DEFAULT = 0.5  # tear → cornea default (/h) at corneal_perm = 1.5e-3 cm/h
_K_CA = 0.4          # cornea → aqueous (/h)
_K_AV = 0.05         # aqueous → vitreous (/h)
_K_AS = 0.1          # aqueous → systemic (trabecular meshwork) (/h)
_K_VR = 0.02         # vitreous → retina (/h)
_K_VS = 0.005        # vitreous → systemic (posterior path) (/h)
_K_RE = 0.1          # retina elimination (/h)

_VALID_ROUTES = {"topical", "intravitreal"}


@dataclass(frozen=True)
class OcularPKResult:
    """Results from ocular PBPK simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    route : Administration route ('topical' or 'intravitreal').
    dose_mg : Administered dose (mg).
    times_h : Simulation time points (h).
    c_tear_mg_L : Tear film concentration (mg/L).
    c_cornea_mg_L : Cornea concentration (mg/L).
    c_aqueous_mg_L : Aqueous humor concentration (mg/L).
    c_vitreous_mg_L : Vitreous humor concentration (mg/L).
    c_retina_mg_L : Retina/choroid concentration (mg/L).
    cp_mg_L : Systemic plasma concentration (mg/L).
    cmax_aqueous : Peak aqueous humor concentration (mg/L).
    tmax_aqueous_h : Time of peak aqueous concentration (h).
    auc_aqueous : AUC of aqueous humor (mg*h/L).
    cmax_vitreous : Peak vitreous humor concentration (mg/L).
    tmax_vitreous_h : Time of peak vitreous concentration (h).
    auc_vitreous : AUC of vitreous humor (mg*h/L).
    cmax_systemic : Peak systemic concentration (mg/L).
    notes : Summary text.
    """

    drug_name: str
    route: str
    dose_mg: float
    times_h: list[float]
    c_tear_mg_L: list[float]
    c_cornea_mg_L: list[float]
    c_aqueous_mg_L: list[float]
    c_vitreous_mg_L: list[float]
    c_retina_mg_L: list[float]
    cp_mg_L: list[float]
    cmax_aqueous: float
    tmax_aqueous_h: float
    auc_aqueous: float
    cmax_vitreous: float
    tmax_vitreous_h: float
    auc_vitreous: float
    cmax_systemic: float
    notes: str


def simulate_ocular_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "topical",
    corneal_perm_cm_h: float = 1.5e-3,
    cl_sys_L_per_h: float = 1.0,
    vd_sys_L: float = 50.0,
    kp_melanin: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.02,
) -> OcularPKResult:
    """Simulate ocular drug pharmacokinetics via topical or intravitreal route.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Administered dose (mg).
    route : 'topical' (eye drop) or 'intravitreal' (injection).
    corneal_perm_cm_h : Corneal permeability (cm/h). Default 1.5e-3.
        Scales tear-to-cornea rate proportionally.
    cl_sys_L_per_h : Systemic clearance (L/h).
    vd_sys_L : Systemic volume of distribution (L).
    kp_melanin : Melanin binding factor (>=1.0). Higher values increase
        effective vitreous volume, lowering free drug concentration.
    t_end_h : Simulation duration (h).
    dt_h : Integration time step (h).

    Returns
    -------
    OcularPKResult with full time-course and PK metrics.

    Raises
    ------
    ValueError : If any parameter is invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if route not in _VALID_ROUTES:
        raise ValueError(
            f"Unknown route '{route}'. Choose from: {sorted(_VALID_ROUTES)}"
        )

    # Transfer rates
    k_tc = (corneal_perm_cm_h / 1.5e-3) * _K_TC_DEFAULT
    ke_sys = cl_sys_L_per_h / vd_sys_L

    # Effective vitreous volume (melanin binding increases apparent Vd)
    v_vitreous_eff_L = _V_VITREOUS_L * kp_melanin

    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    # State: amounts in mg
    a_tear = np.zeros(n_steps + 1)
    a_cornea = np.zeros(n_steps + 1)
    a_aqueous = np.zeros(n_steps + 1)
    a_vitreous = np.zeros(n_steps + 1)
    a_retina = np.zeros(n_steps + 1)
    cp = np.zeros(n_steps + 1)  # systemic concentration (mg/L)

    # Initial conditions
    if route == "topical":
        a_tear[0] = dose_mg
    else:  # intravitreal
        a_vitreous[0] = dose_mg

    # Forward Euler integration
    for i in range(n_steps):
        # Fluxes (mg/h)
        drain_tear = _KD * a_tear[i]
        flux_tc = k_tc * a_tear[i]
        flux_ca = _K_CA * a_cornea[i]
        flux_av = _K_AV * a_aqueous[i]
        flux_as = _K_AS * a_aqueous[i]
        flux_vr = _K_VR * a_vitreous[i]
        flux_vs = _K_VS * a_vitreous[i]
        flux_re = _K_RE * a_retina[i]

        # ODEs
        da_tear = (-drain_tear - flux_tc) * dt_h
        da_cornea = (flux_tc - flux_ca) * dt_h
        da_aqueous = (flux_ca - flux_av - flux_as) * dt_h
        da_vitreous = (flux_av - flux_vr - flux_vs) * dt_h
        da_retina = (flux_vr - flux_re) * dt_h
        dcp = (
            (flux_as + flux_vs + flux_re * 0.5) / vd_sys_L - ke_sys * cp[i]
        ) * dt_h

        a_tear[i + 1] = max(0.0, a_tear[i] + da_tear)
        a_cornea[i + 1] = max(0.0, a_cornea[i] + da_cornea)
        a_aqueous[i + 1] = max(0.0, a_aqueous[i] + da_aqueous)
        a_vitreous[i + 1] = max(0.0, a_vitreous[i] + da_vitreous)
        a_retina[i + 1] = max(0.0, a_retina[i] + da_retina)
        cp[i + 1] = max(0.0, cp[i] + dcp)

    # Concentrations (mg/L) = amount (mg) / volume (L)
    c_tear = a_tear / _V_TEAR_L
    c_cornea = a_cornea / _V_CORNEA_L
    c_aqueous = a_aqueous / _V_AQUEOUS_L
    c_vitreous = a_vitreous / v_vitreous_eff_L
    c_retina = a_retina / _V_RETINA_L

    # PK metrics
    cmax_aq = float(np.max(c_aqueous))
    tmax_aq = float(t[int(np.argmax(c_aqueous))])
    auc_aq = float(np_trapz(c_aqueous, t))

    cmax_vit = float(np.max(c_vitreous))
    tmax_vit = float(t[int(np.argmax(c_vitreous))])
    auc_vit = float(np_trapz(c_vitreous, t))

    cmax_sys = float(np.max(cp))

    notes = (
        f"Ocular PBPK: {drug_name}, {dose_mg}mg via {route}. "
        f"Aqueous Cmax={cmax_aq:.4f} mg/L at {tmax_aq:.2f}h. "
        f"Vitreous Cmax={cmax_vit:.4f} mg/L at {tmax_vit:.2f}h. "
        f"Systemic Cmax={cmax_sys:.6f} mg/L."
    )

    return OcularPKResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        times_h=t.tolist(),
        c_tear_mg_L=c_tear.tolist(),
        c_cornea_mg_L=c_cornea.tolist(),
        c_aqueous_mg_L=c_aqueous.tolist(),
        c_vitreous_mg_L=c_vitreous.tolist(),
        c_retina_mg_L=c_retina.tolist(),
        cp_mg_L=cp.tolist(),
        cmax_aqueous=cmax_aq,
        tmax_aqueous_h=tmax_aq,
        auc_aqueous=auc_aq,
        cmax_vitreous=cmax_vit,
        tmax_vitreous_h=tmax_vit,
        auc_vitreous=auc_vit,
        cmax_systemic=cmax_sys,
        notes=notes,
    )


def compare_routes(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float = 1.0,
    vd_sys_L: float = 50.0,
    **kwargs,
) -> list[OcularPKResult]:
    """Compare topical vs intravitreal routes for the same dose.

    Returns
    -------
    List of [topical_result, intravitreal_result].
    """
    return [
        simulate_ocular_pk(
            drug_name, dose_mg, route="topical",
            cl_sys_L_per_h=cl_sys_L_per_h, vd_sys_L=vd_sys_L, **kwargs,
        ),
        simulate_ocular_pk(
            drug_name, dose_mg, route="intravitreal",
            cl_sys_L_per_h=cl_sys_L_per_h, vd_sys_L=vd_sys_L, **kwargs,
        ),
    ]


__all__ = [
    "OcularPKResult",
    "simulate_ocular_pk",
    "compare_routes",
]
