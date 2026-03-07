"""Phase 877 — Inhalation PBPK model for inhaled drug delivery.

Compartments:
  - Central airway (lung): deposition site, mucociliary clearance
  - Alveolar region: gas exchange / systemic absorption
  - Systemic plasma: absorbed drug

Routes supported: DPI, nebulizer, MDI
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InhalationPKResult:
    """Result of an inhalation PBPK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    a_central_airway_mg: list = field(default_factory=list)
    a_alveolar_mg: list = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    tmax_plasma_h: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    f_pulmonary: float = 0.0
    lung_deposition_central_frac: float = 0.0
    lung_deposition_alveolar_frac: float = 0.0
    notes: str = ""


def simulate_inhalation_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "DPI",
    f_lung: float = 0.4,
    f_alveolar_frac: float = 0.7,
    k_mcc_per_h: float = 0.5,
    k_abs_alv_per_h: float = 2.0,
    cl_L_per_h: float = 10.0,
    vd_L: float = 100.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> InhalationPKResult:
    """Simulate inhalation PK using a 3-compartment PBPK model.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Total inhaled dose in mg.
    route:
        Delivery device: ``"DPI"``, ``"nebulizer"``, or ``"MDI"``.
    f_lung:
        Fraction of dose deposited in the lungs (0-1).
    f_alveolar_frac:
        Fraction of lung-deposited dose reaching the alveolar region (DPI base).
    k_mcc_per_h:
        Mucociliary clearance rate constant from central airways (1/h).
    k_abs_alv_per_h:
        Absorption rate constant from the alveolar region (1/h).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    InhalationPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    valid_routes = {"DPI", "nebulizer", "MDI"}
    if route not in valid_routes:
        raise ValueError(f"route must be one of {valid_routes}, got '{route}'")

    # Route-dependent alveolar fraction adjustment
    if route == "nebulizer":
        eff_alveolar_frac = f_alveolar_frac * 0.4
    elif route == "MDI":
        eff_alveolar_frac = f_alveolar_frac * 0.6
    else:  # DPI
        eff_alveolar_frac = f_alveolar_frac

    # Clamp to [0, 1]
    eff_alveolar_frac = max(0.0, min(1.0, eff_alveolar_frac))

    # Deposition fractions stored for the result
    lung_dep_alv = eff_alveolar_frac
    lung_dep_central = 1.0 - eff_alveolar_frac

    # Rate constants
    k_abs_central = k_abs_alv_per_h * 0.3  # central airways absorb more slowly
    ke = cl_L_per_h / vd_L

    # Initial conditions
    a_central = dose_mg * f_lung * lung_dep_central
    a_alveolar = dose_mg * f_lung * lung_dep_alv
    c_plasma = 0.0

    # Storage
    times: list = []
    c_plasma_arr: list = []
    a_central_arr: list = []
    a_alveolar_arr: list = []

    t = 0.0
    n_steps = max(1, round(t_end_h / dt_h))

    for _i in range(n_steps + 1):
        times.append(t)
        c_plasma_arr.append(c_plasma)
        a_central_arr.append(a_central)
        a_alveolar_arr.append(a_alveolar)

        if _i == n_steps:
            break

        # ODEs
        da_central = -(k_mcc_per_h + k_abs_central) * a_central
        da_alveolar = -k_abs_alv_per_h * a_alveolar
        dc_plasma = (
            k_abs_central * a_central + k_abs_alv_per_h * a_alveolar
        ) / vd_L - ke * c_plasma

        # Forward Euler
        a_central = max(0.0, a_central + da_central * dt_h)
        a_alveolar = max(0.0, a_alveolar + da_alveolar * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        t += dt_h

    # Derived metrics
    cmax = max(c_plasma_arr)
    tmax = times[c_plasma_arr.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (c_plasma_arr[i - 1] + c_plasma_arr[i]) * (times[i] - times[i - 1])

    f_pulmonary = auc * cl_L_per_h / dose_mg

    # Notes
    if f_pulmonary > 0.3:
        notes = "Good pulmonary bioavailability"
    elif f_pulmonary > 0.1:
        notes = "Moderate pulmonary bioavailability"
    else:
        notes = "Poor pulmonary bioavailability"

    return InhalationPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma_arr,
        a_central_airway_mg=a_central_arr,
        a_alveolar_mg=a_alveolar_arr,
        cmax_plasma_mg_L=cmax,
        tmax_plasma_h=tmax,
        auc_plasma_mg_h_per_L=auc,
        f_pulmonary=f_pulmonary,
        lung_deposition_central_frac=lung_dep_central,
        lung_deposition_alveolar_frac=lung_dep_alv,
        notes=notes,
    )


def compare_inhalation_routes(
    drug_name: str,
    dose_mg: float,
    f_lung: float = 0.4,
    f_alveolar_frac: float = 0.7,
    k_mcc_per_h: float = 0.5,
    k_abs_alv_per_h: float = 2.0,
    cl_L_per_h: float = 10.0,
    vd_L: float = 100.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list:
    """Simulate inhalation PK for DPI, nebulizer, and MDI, sorted by AUC descending.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Total inhaled dose in mg.

    Other parameters are passed through to :func:`simulate_inhalation_pk`.

    Returns
    -------
    list[InhalationPKResult]
        Three results sorted by ``auc_plasma_mg_h_per_L`` in descending order.
    """
    results = []
    for route in ("DPI", "nebulizer", "MDI"):
        res = simulate_inhalation_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            f_lung=f_lung,
            f_alveolar_frac=f_alveolar_frac,
            k_mcc_per_h=k_mcc_per_h,
            k_abs_alv_per_h=k_abs_alv_per_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(res)

    results.sort(key=lambda r: r.auc_plasma_mg_h_per_L, reverse=True)
    return results
