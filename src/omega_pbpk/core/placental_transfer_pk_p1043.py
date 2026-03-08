"""Drug placental transfer PK model (Phase 1043).

Models transporter-mediated + passive drug transfer across the placenta.
Distinct from placental_transfer_pk.py (Phase 906) — this module adds
gestational-age-dependent blood flow, amniotic fluid compartment,
P-gp efflux effects, and fetal exposure categorization.

Key safety metric: fetal:maternal AUC ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PlacentalTransferResult:
    """Result of placental transfer PK simulation (Phase 1043)."""

    drug_name: str
    dose_mg: float
    gestational_age_weeks: int
    times_h: list
    c_maternal_mg_L: list
    c_umbilical_mg_L: list
    c_amniotic_mg_L: list
    auc_maternal: float
    auc_fetal: float
    fetal_to_maternal_ratio: float
    placental_clearance_L_h: float
    transporter_effect: str
    fetal_exposure_category: str
    notes: str


def _trapz(times: list, values: list) -> float:
    """Compute AUC via trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i - 1] + values[i]) * dt
    return auc


# Gestational age lookup: (relative_surface_area, umbilical_blood_flow_L_h)
_GA_TABLE: dict[int, tuple[float, float]] = {
    12: (1.0, 0.05),
    20: (2.5, 0.15),
    28: (5.0, 0.35),
    36: (8.0, 0.55),
    40: (10.0, 0.70),
}

_VALID_GA = frozenset(_GA_TABLE.keys())
_VALID_ROUTES = frozenset({"iv", "oral"})


def simulate_placental_transfer(
    drug_name: str,
    dose_mg: float,
    gestational_age_weeks: int = 36,
    mw_Da: float = 300.0,
    logp: float = 2.0,
    fu_maternal: float = 0.3,
    cl_maternal_L_per_h: float = 5.0,
    vd_maternal_L: float = 30.0,
    pgp_substrate: bool = False,
    route: str = "iv",
    f_oral: float = 0.8,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PlacentalTransferResult:
    """Simulate drug placental transfer with gestational-age-dependent kinetics.

    Three compartments: maternal plasma, fetal (umbilical), amniotic fluid.
    Forward Euler integration.

    Args:
        drug_name: Drug identifier.
        dose_mg: Dose in mg.
        gestational_age_weeks: Gestational age in weeks (12, 20, 28, 36, or 40).
        mw_Da: Molecular weight in Da.
        logp: Octanol-water partition coefficient.
        fu_maternal: Maternal unbound fraction (0, 1].
        cl_maternal_L_per_h: Maternal systemic clearance in L/h.
        vd_maternal_L: Maternal volume of distribution in L.
        pgp_substrate: Whether drug is a P-gp substrate.
        route: Dosing route ("iv" or "oral").
        f_oral: Oral bioavailability fraction.
        t_end_h: Simulation end time in hours.
        dt_h: Forward Euler step size in hours.

    Returns:
        PlacentalTransferResult with time profiles and safety metrics.

    Raises:
        ValueError: If inputs are out of valid range.
    """
    # Validation
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if gestational_age_weeks not in _VALID_GA:
        raise ValueError(
            f"gestational_age_weeks must be one of {sorted(_VALID_GA)}, got {gestational_age_weeks}"
        )
    if not (0.0 < fu_maternal <= 1.0):
        raise ValueError(f"fu_maternal must be in (0, 1], got {fu_maternal}")
    if cl_maternal_L_per_h <= 0.0:
        raise ValueError(f"cl_maternal_L_per_h must be > 0, got {cl_maternal_L_per_h}")
    if vd_maternal_L <= 0.0:
        raise ValueError(f"vd_maternal_L must be > 0, got {vd_maternal_L}")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got {route!r}")

    # Gestational age effects
    _s, q_umbilical = _GA_TABLE[gestational_age_weeks]

    # Passive permeability
    perm = 10.0 ** (0.3 * logp - 0.002 * mw_Da - 1.5)
    perm = max(1e-5, min(0.1, perm))

    # Placental transfer rate constants (L/h, membrane-limited)
    k_m2f = q_umbilical * perm / (perm + q_umbilical / 10.0)
    k_f2m = k_m2f / 2.0  # asymmetric — fetal clearance lower

    # P-gp efflux increases back-transfer to maternal
    if pgp_substrate:
        k_f2m *= 3.0

    placental_clearance_L_h = k_m2f

    # Fetal compartment sizing
    vd_fetal = vd_maternal_L * 0.15
    ke_fetal = cl_maternal_L_per_h * 0.1 / vd_fetal  # 10% of maternal CL
    k_fetal_to_amniotic = 0.05  # /h

    # Amniotic fluid volume
    v_amniotic = 0.5  # L

    # Maternal elimination
    ke_mat = cl_maternal_L_per_h / vd_maternal_L

    # Initial conditions
    if route == "iv":
        bioavail_dose = dose_mg
        a_gut = 0.0
        ka = 0.0
        c_mat = bioavail_dose / vd_maternal_L
    else:
        bioavail_dose = dose_mg * f_oral
        a_gut = bioavail_dose
        ka = 1.0  # /h oral absorption
        c_mat = 0.0

    c_fetal = 0.0
    c_amniotic = 0.0

    times_h: list = []
    c_maternal_list: list = []
    c_umbilical_list: list = []
    c_amniotic_list: list = []

    n_steps = int(round(t_end_h / dt_h)) + 1
    t = 0.0

    # Ratio of volumes for ODE scaling
    vol_ratio = vd_maternal_L / vd_fetal

    for _i in range(n_steps):
        times_h.append(t)
        c_maternal_list.append(c_mat)
        c_umbilical_list.append(c_fetal)
        c_amniotic_list.append(c_amniotic)

        # ODEs
        if route == "oral":
            da_gut = -ka * a_gut * dt_h
            absorption_flux = ka * a_gut / vd_maternal_L
        else:
            da_gut = 0.0
            absorption_flux = 0.0

        dc_mat = (
            absorption_flux
            - ke_mat * c_mat
            - k_m2f * vol_ratio * c_mat * fu_maternal
            + k_f2m * c_fetal
        )
        dc_fetal = (
            k_m2f * vol_ratio * c_mat * fu_maternal
            - k_f2m * c_fetal
            - ke_fetal * c_fetal
            - k_fetal_to_amniotic * c_fetal
        )
        dc_amniotic = k_fetal_to_amniotic * c_fetal * (vd_fetal / v_amniotic) - 0.01 * c_amniotic

        # Forward Euler update
        if route == "oral":
            a_gut = max(0.0, a_gut + da_gut)
        c_mat = max(0.0, c_mat + dc_mat * dt_h)
        c_fetal = max(0.0, c_fetal + dc_fetal * dt_h)
        c_amniotic = max(0.0, c_amniotic + dc_amniotic * dt_h)

        t = round(t + dt_h, 10)

    auc_maternal = _trapz(times_h, c_maternal_list)
    auc_fetal = _trapz(times_h, c_umbilical_list)
    fetal_to_maternal_ratio = auc_fetal / auc_maternal if auc_maternal > 0.0 else 0.0

    # Transporter effect classification
    if pgp_substrate:
        transporter_effect = "P-gp efflux"
    elif perm < 0.01:
        transporter_effect = "passive"
    else:
        transporter_effect = "bidirectional"

    # Fetal exposure category
    r = fetal_to_maternal_ratio
    if r < 0.1:
        fetal_exposure_category = "minimal"
    elif r < 0.3:
        fetal_exposure_category = "low"
    elif r < 0.6:
        fetal_exposure_category = "moderate"
    else:
        fetal_exposure_category = "high"

    # Notes
    ga_pct = gestational_age_weeks
    _r_pct = round(fetal_to_maternal_ratio * 100.0, 1)
    notes = (
        f"GA {ga_pct} weeks; fetal/maternal AUC ratio = {_r_pct:.1f}%; "
        f"transporter: {transporter_effect}; "
        f"fetal exposure: {fetal_exposure_category}."
    )

    return PlacentalTransferResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        gestational_age_weeks=gestational_age_weeks,
        times_h=times_h,
        c_maternal_mg_L=c_maternal_list,
        c_umbilical_mg_L=c_umbilical_list,
        c_amniotic_mg_L=c_amniotic_list,
        auc_maternal=auc_maternal,
        auc_fetal=auc_fetal,
        fetal_to_maternal_ratio=fetal_to_maternal_ratio,
        placental_clearance_L_h=placental_clearance_L_h,
        transporter_effect=transporter_effect,
        fetal_exposure_category=fetal_exposure_category,
        notes=notes,
    )


# Suppress unused import of math (used for potential future use)
_ = math.log10  # keep import active for documentation purposes
