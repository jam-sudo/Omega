"""Comprehensive pregnancy PK model covering all trimesters.

Physiological changes during pregnancy alter drug disposition through:
- Increased GFR (renal clearance enhancement)
- Increased cardiac output (tissue perfusion changes)
- Expanded plasma volume (apparent volume of distribution increase)
- Reduced albumin (altered protein binding and free fraction)

The model uses a one-compartment forward Euler approach with trimester-specific
scaling of CL and Vd compared to the non-pregnant state.

References:
    Pariente et al. Drug Metab Rev 2016; Anderson Clin Pharmacokinet 2005;
    Feghali & Mattison Clin Ther 2011.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Trimester physiological scaling factors
# ---------------------------------------------------------------------------

_TRIMESTER_FACTORS: dict[int, dict[str, float]] = {
    1: {
        "gfr_factor": 1.25,  # GFR +25%
        "cardiac_output_factor": 1.15,  # CO +15%
        "plasma_volume_factor": 1.10,  # plasma vol +10%
        "albumin_factor": 0.95,  # albumin -5%
    },
    2: {
        "gfr_factor": 1.50,  # GFR +50%
        "cardiac_output_factor": 1.35,  # CO +35%
        "plasma_volume_factor": 1.40,  # plasma vol +40%
        "albumin_factor": 0.85,  # albumin -15%
    },
    3: {
        "gfr_factor": 1.50,  # GFR +50%
        "cardiac_output_factor": 1.45,  # CO +45%
        "plasma_volume_factor": 1.50,  # plasma vol +50%
        "albumin_factor": 0.75,  # albumin -25%
    },
}

_VALID_ROUTES = {"oral", "iv"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PregnancyPKResult:
    """Result of a pregnancy PK simulation."""

    drug_name: str
    dose_mg: float
    trimester: int
    route: str
    cl_adjusted_L_per_h: float
    vd_adjusted_L: float
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    auc_pregnant: float = 0.0
    cmax_pregnant: float = 0.0
    auc_nonpregnant_estimate: float = 0.0
    auc_ratio: float = 0.0  # pregnant / non-pregnant
    dose_adjustment_recommendation: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapezoidal_auc(times: list[float], concentrations: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concentrations[i - 1] + concentrations[i]) * dt
    return auc


def _simulate_1cpt_oral(
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    ka_per_h: float,
    f_oral: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Forward Euler 1-compartment oral absorption PK."""
    ke = cl_L_per_h / vd_L
    times: list[float] = []
    concs: list[float] = []

    # State: gut depot (amount, mg) + central (amount, mg)
    a_gut = dose_mg * f_oral
    a_central = 0.0
    t = 0.0

    while t <= t_end_h + 1e-9:
        times.append(t)
        concs.append(a_central / vd_L)

        d_gut = -ka_per_h * a_gut * dt_h
        d_central = (ka_per_h * a_gut - ke * a_central) * dt_h
        a_gut += d_gut
        a_central += d_central
        t += dt_h

    return times, concs


def _simulate_1cpt_iv(
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Forward Euler 1-compartment IV bolus PK."""
    ke = cl_L_per_h / vd_L
    times: list[float] = []
    concs: list[float] = []

    a_central = dose_mg  # instantaneous IV bolus
    t = 0.0

    while t <= t_end_h + 1e-9:
        times.append(t)
        concs.append(a_central / vd_L)
        a_central -= ke * a_central * dt_h
        t += dt_h

    return times, concs


def _nonpregnant_auc_estimate(
    dose_mg: float,
    cl_normal_L_per_h: float,
    route: str,
    f_oral: float,
    vd_normal_L: float,
    ka_per_h: float,
    t_end_h: float,
    dt_h: float,
) -> float:
    """Estimate AUC in the non-pregnant state using the same simulation approach."""
    if route == "iv":
        times, concs = _simulate_1cpt_iv(dose_mg, vd_normal_L, cl_normal_L_per_h, t_end_h, dt_h)
    else:
        times, concs = _simulate_1cpt_oral(
            dose_mg, vd_normal_L, cl_normal_L_per_h, ka_per_h, f_oral, t_end_h, dt_h
        )
    return _trapezoidal_auc(times, concs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_pregnancy_pk(
    drug_name: str,
    dose_mg: float,
    cl_normal_L_per_h: float,
    vd_normal_L: float,
    trimester: int,
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PregnancyPKResult:
    """Simulate 1-compartment PK during pregnancy with trimester-specific scaling.

    Args:
        drug_name: Identifier for the compound.
        dose_mg: Administered dose in mg.
        cl_normal_L_per_h: Non-pregnant total clearance (L/h).
        vd_normal_L: Non-pregnant volume of distribution (L).
        trimester: Pregnancy trimester (1, 2, or 3).
        route: Administration route — "oral" or "iv".
        ka_per_h: First-order oral absorption rate constant (h⁻¹).
        f_oral: Oral bioavailability fraction (0–1).
        t_end_h: Simulation duration (h).
        dt_h: Integration time step (h).

    Returns:
        PregnancyPKResult with simulated PK and dose recommendation.

    Raises:
        ValueError: On invalid inputs.
    """
    # Input validation
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string.")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive; got {dose_mg}.")
    if cl_normal_L_per_h <= 0:
        raise ValueError(f"cl_normal_L_per_h must be positive; got {cl_normal_L_per_h}.")
    if vd_normal_L <= 0:
        raise ValueError(f"vd_normal_L must be positive; got {vd_normal_L}.")
    if trimester not in (1, 2, 3):
        raise ValueError(f"trimester must be 1, 2, or 3; got {trimester}.")
    route = route.lower()
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}; got '{route}'.")
    if route == "oral" and ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be positive for oral route; got {ka_per_h}.")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError(f"f_oral must be in (0, 1]; got {f_oral}.")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be positive; got {t_end_h}.")
    if dt_h <= 0 or dt_h >= t_end_h:
        raise ValueError(f"dt_h must be positive and < t_end_h; got {dt_h}.")

    factors = _TRIMESTER_FACTORS[trimester]
    gfr_factor = factors["gfr_factor"]
    plasma_volume_factor = factors["plasma_volume_factor"]

    # CL scaling: half renal (GFR-dependent), half hepatic (unchanged)
    cl_adjusted = cl_normal_L_per_h * gfr_factor * 0.5 + cl_normal_L_per_h * 0.5

    # Vd scaling by plasma volume expansion
    vd_adjusted = vd_normal_L * plasma_volume_factor

    # Simulate pregnant PK
    if route == "iv":
        times, concs = _simulate_1cpt_iv(dose_mg, vd_adjusted, cl_adjusted, t_end_h, dt_h)
    else:
        times, concs = _simulate_1cpt_oral(
            dose_mg, vd_adjusted, cl_adjusted, ka_per_h, f_oral, t_end_h, dt_h
        )

    auc_preg = _trapezoidal_auc(times, concs)
    cmax_preg = max(concs) if concs else 0.0

    # Non-pregnant AUC estimate
    auc_nonpreg = _nonpregnant_auc_estimate(
        dose_mg, cl_normal_L_per_h, route, f_oral, vd_normal_L, ka_per_h, t_end_h, dt_h
    )

    auc_ratio = auc_preg / auc_nonpreg if auc_nonpreg > 0 else float("nan")

    # Dose adjustment recommendation
    auc_change_pct = (auc_ratio - 1.0) * 100.0
    if abs(auc_change_pct) <= 20.0:
        recommendation = "No dose adjustment required; AUC change within 20%."
    elif auc_change_pct < -20.0:
        increase_pct = round(-auc_change_pct)
        recommendation = (
            f"Consider dose increase (~{increase_pct}%) to compensate for "
            f"reduced AUC in T{trimester} (AUC ratio={auc_ratio:.2f})."
        )
    else:
        decrease_pct = round(auc_change_pct)
        recommendation = (
            f"Consider dose reduction (~{decrease_pct}%) to avoid elevated "
            f"exposure in T{trimester} (AUC ratio={auc_ratio:.2f})."
        )

    notes = (
        f"T{trimester} physiology: GFR ×{gfr_factor}, "
        f"plasma vol ×{plasma_volume_factor}, "
        f"albumin ×{factors['albumin_factor']}. "
        f"CL scaled to {cl_adjusted:.3f} L/h; Vd scaled to {vd_adjusted:.1f} L."
    )

    return PregnancyPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        trimester=trimester,
        route=route,
        cl_adjusted_L_per_h=cl_adjusted,
        vd_adjusted_L=vd_adjusted,
        times_h=times,
        c_plasma_mg_L=concs,
        auc_pregnant=auc_preg,
        cmax_pregnant=cmax_preg,
        auc_nonpregnant_estimate=auc_nonpreg,
        auc_ratio=auc_ratio,
        dose_adjustment_recommendation=recommendation,
        notes=notes,
    )
