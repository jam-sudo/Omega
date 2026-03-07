"""Absorption lag time models for oral drug formulations.

Models sources of absorption lag time:
  - Gastric emptying (fasted vs fed)
  - Tablet/capsule disintegration
  - Enteric coating dissolution (pH-dependent)

The lag time is applied to a 1-compartment oral PK model via forward Euler
simulation to produce plasma concentration-time profiles.

References:
    Dressman JB et al., Pharm Res 1990.
    Amidon GL et al., Pharm Res 1995 (BCS framework).
    Galia E et al., Pharm Res 1998 (biorelevant media).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "LagTimeResult",
    "calculate_lag_times",
    "simulate_absorption_with_lag",
    "compare_formulations",
]

_VALID_FORMULATIONS = {"immediate_release", "enteric_coated", "tablet", "capsule"}


@dataclass(frozen=True)
class LagTimeResult:
    """Result of absorption lag time simulation."""

    drug_name: str
    formulation_type: str

    tlag_gastric_emptying_h: float
    tlag_disintegration_h: float
    tlag_coating_h: float
    tlag_total_h: float

    # PK impact
    tmax_without_lag_h: float
    tmax_with_lag_h: float

    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_plasma_no_lag_mg_L: list[float]

    cmax_mg_L: float
    cmax_no_lag_mg_L: float
    auc_mg_h_per_L: float

    time_to_cmax_pct_of_ref: float
    cmax_reduction_pct: float

    notes: str


def calculate_lag_times(
    formulation_type: str,
    fasted_state: bool = True,
    gastric_ph: float = 1.5,  # noqa: ARG001 — pH used for future coating models
    tablet_size_mg: float = 500.0,
    coating_thickness_um: float = 50.0,
) -> dict[str, float]:
    """Calculate component lag times for a given formulation.

    Parameters
    ----------
    formulation_type:
        One of ``"immediate_release"``, ``"enteric_coated"``, ``"tablet"``,
        ``"capsule"``.
    fasted_state:
        If True, use fasted gastric emptying rate (faster). Default True.
    gastric_ph:
        Gastric pH — retained for API extensibility; not used in current model.
    tablet_size_mg:
        Total tablet mass (mg). Larger tablets take longer to disintegrate.
    coating_thickness_um:
        Enteric coating thickness (µm). Thicker coatings dissolve more slowly.

    Returns
    -------
    dict with keys:
        ``gastric_emptying_h``, ``disintegration_h``, ``coating_dissolution_h``,
        ``total_h``.
    """
    if formulation_type not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation_type must be one of {_VALID_FORMULATIONS}; got '{formulation_type}'"
        )

    # Gastric emptying
    gastric_emptying_h = 0.25 if fasted_state else 1.0

    # Disintegration
    if formulation_type == "immediate_release":
        disintegration_h = 0.08
    elif formulation_type == "tablet":
        disintegration_h = 0.17 + 0.0001 * tablet_size_mg
    elif formulation_type == "capsule":
        disintegration_h = 0.05
    else:  # enteric_coated — no disintegration in stomach
        disintegration_h = 0.0

    # Enteric coating dissolution (intestinal pH ~6.5-7)
    if formulation_type == "enteric_coated":
        coating_dissolution_h = coating_thickness_um / 100.0 * 0.5
    else:
        coating_dissolution_h = 0.0

    total_h = gastric_emptying_h + disintegration_h + coating_dissolution_h

    return {
        "gastric_emptying_h": gastric_emptying_h,
        "disintegration_h": disintegration_h,
        "coating_dissolution_h": coating_dissolution_h,
        "total_h": total_h,
    }


def simulate_absorption_with_lag(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.5,
    f_oral: float = 0.9,
    formulation_type: str = "immediate_release",
    fasted_state: bool = True,
    tablet_size_mg: float = 500.0,
    coating_thickness_um: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> LagTimeResult:
    """Simulate 1-compartment oral PK with formulation-specific lag time.

    Forward Euler integration. Before ``tlag``, no absorption occurs; at
    ``tlag`` the bioavailable dose is loaded into the gut depot.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Oral dose in mg.
    cl_L_per_h:
        Total body clearance (L/h).
    vd_L:
        Volume of distribution (L).
    ka_per_h:
        First-order absorption rate constant (h⁻¹).
    f_oral:
        Oral bioavailability (0, 1].
    formulation_type:
        One of ``"immediate_release"``, ``"enteric_coated"``, ``"tablet"``,
        ``"capsule"``.
    fasted_state:
        Fasted (True) or fed (False) gastric emptying.
    tablet_size_mg:
        Tablet mass for disintegration calculation (mg).
    coating_thickness_um:
        Coating thickness for enteric-coated formulations (µm).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration step (h).

    Returns
    -------
    LagTimeResult
    """
    # Input validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if not (0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")

    lag_dict = calculate_lag_times(
        formulation_type=formulation_type,
        fasted_state=fasted_state,
        tablet_size_mg=tablet_size_mg,
        coating_thickness_um=coating_thickness_um,
    )
    tlag = lag_dict["total_h"]

    ke = cl_L_per_h / vd_L
    bioavailable_dose = dose_mg * f_oral  # mg

    # --- Analytical tmax (without lag) ---
    if abs(ka_per_h - ke) < 1e-9:
        # degenerate case
        tmax_no_lag = 1.0 / ke
    else:
        tmax_no_lag = math.log(ka_per_h / ke) / (ka_per_h - ke)
    tmax_no_lag = max(tmax_no_lag, 0.0)
    tmax_with_lag = tmax_no_lag + tlag

    # --- Simulation ---
    n_steps = max(2, int(round(t_end_h / dt_h)) + 1)
    times = [i * dt_h for i in range(n_steps)]

    def _run_simulation(use_lag: bool) -> list[float]:
        """Forward Euler 1-cpt oral absorption."""
        gut_amount = 0.0  # mg in gut (absorbed via ka)
        central_conc = 0.0  # mg/L
        loaded = False
        concentrations = []

        for t in times:
            concentrations.append(central_conc)
            # Load dose at t=tlag (with lag) or t=0 (no lag)
            threshold = tlag if use_lag else 0.0
            if not loaded and t >= threshold:
                gut_amount = bioavailable_dose
                loaded = True

            # Forward Euler
            dA_gut = -ka_per_h * gut_amount
            dC_central = (ka_per_h * gut_amount / vd_L) - ke * central_conc

            gut_amount += dA_gut * dt_h
            gut_amount = max(0.0, gut_amount)
            central_conc += dC_central * dt_h
            central_conc = max(0.0, central_conc)

        return concentrations

    c_lag = _run_simulation(use_lag=True)
    c_no_lag = _run_simulation(use_lag=False)

    cmax = max(c_lag)
    cmax_no_lag = max(c_no_lag)

    # AUC via trapezoidal rule
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (c_lag[i - 1] + c_lag[i]) * dt_h

    # Therapeutic implications
    tmax_ref = tmax_no_lag if tmax_no_lag > 0 else 1e-9
    time_to_cmax_pct = tmax_with_lag / tmax_ref * 100.0

    if cmax_no_lag > 1e-12:
        cmax_reduction_pct = max(0.0, (cmax_no_lag - cmax) / cmax_no_lag * 100.0)
    else:
        cmax_reduction_pct = 0.0

    notes_parts = [
        f"tlag={tlag:.3f} h",
        f"ke={ke:.4f} h-1",
        f"tmax_analytical={tmax_no_lag:.2f} h",
        formulation_type,
        "fasted" if fasted_state else "fed",
    ]

    return LagTimeResult(
        drug_name=drug_name,
        formulation_type=formulation_type,
        tlag_gastric_emptying_h=lag_dict["gastric_emptying_h"],
        tlag_disintegration_h=lag_dict["disintegration_h"],
        tlag_coating_h=lag_dict["coating_dissolution_h"],
        tlag_total_h=tlag,
        tmax_without_lag_h=tmax_no_lag,
        tmax_with_lag_h=tmax_with_lag,
        times_h=times,
        c_plasma_mg_L=c_lag,
        c_plasma_no_lag_mg_L=c_no_lag,
        cmax_mg_L=cmax,
        cmax_no_lag_mg_L=cmax_no_lag,
        auc_mg_h_per_L=auc,
        time_to_cmax_pct_of_ref=time_to_cmax_pct,
        cmax_reduction_pct=cmax_reduction_pct,
        notes="; ".join(notes_parts),
    )


def compare_formulations(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    **kwargs,
) -> list[LagTimeResult]:
    """Simulate all 4 formulation types and return sorted by lag time.

    Parameters
    ----------
    drug_name, dose_mg, cl_L_per_h, vd_L:
        Drug parameters passed to :func:`simulate_absorption_with_lag`.
    **kwargs:
        Additional keyword arguments forwarded to
        :func:`simulate_absorption_with_lag` (e.g., ``ka_per_h``,
        ``f_oral``, ``t_end_h``).

    Returns
    -------
    list[LagTimeResult] sorted by ``tlag_total_h`` ascending.
    """
    results = []
    for formulation in sorted(_VALID_FORMULATIONS):
        res = simulate_absorption_with_lag(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            formulation_type=formulation,
            **kwargs,
        )
        results.append(res)

    results.sort(key=lambda r: r.tlag_total_h)
    return results
