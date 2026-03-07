"""
Phase 939 — Neonatal and infant dosing model.

Highly immature organ function, weight-based PK scaling.
1-compartment forward Euler simulation with age-group-specific
CL maturation and Vd scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NeonatalDosingResult", "simulate_neonatal_dosing", "compare_age_groups"]

# ---------------------------------------------------------------------------
# Age-group reference data
# ---------------------------------------------------------------------------

_AGE_GROUPS = {
    "premature": {"cl_mult": 0.05, "weight_kg": 1.0, "vd_mult": 1.5},
    "term_neonate": {"cl_mult": 0.10, "weight_kg": 3.5, "vd_mult": 1.3},
    "infant_1_6m": {"cl_mult": 0.30, "weight_kg": 6.0, "vd_mult": 1.2},
    "infant_6_12m": {"cl_mult": 0.50, "weight_kg": 9.0, "vd_mult": 1.1},
    "toddler_1_2y": {"cl_mult": 0.70, "weight_kg": 12.0, "vd_mult": 1.0},
}

_ADULT_WEIGHT_KG = 70.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class NeonatalDosingResult:
    """Results from a neonatal/infant dosing simulation."""

    drug_name: str
    age_group: str
    dose_mg: float
    patient_weight_kg: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_h: float
    cl_effective_L_per_h: float
    vd_effective_L: float
    adult_t_half_h: float
    t_half_fold_increase: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_effective_params(
    age_group: str,
    cl_adult_L_per_h: float,
    vd_adult_L: float,
) -> tuple:
    """Return (cl_eff, vd_eff, weight_kg) for the given age group."""
    info = _AGE_GROUPS[age_group]
    weight_kg = info["weight_kg"]
    # Vd scales linearly with weight (allometric exponent 1.0) + extra body water mult
    vd_eff = vd_adult_L * (weight_kg / _ADULT_WEIGHT_KG) * info["vd_mult"]
    # CL scales with maturation multiplier * weight fraction
    cl_eff = cl_adult_L_per_h * info["cl_mult"] * (weight_kg / _ADULT_WEIGHT_KG)
    return cl_eff, vd_eff, weight_kg


def _trapezoidal_auc(times: list, conc: list) -> float:
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (conc[i - 1] + conc[i]) * (times[i] - times[i - 1])
    return auc


def _simulate_1cpt(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    route: str,
    n_doses: int,
    dosing_interval_h: float,
    t_end_h: float,
    dt_h: float,
) -> tuple:
    """Forward Euler 1-cpt simulation, returns (times, concentrations)."""
    ke = cl_L_per_h / vd_L

    n_steps = int(round(t_end_h / dt_h)) + 1
    times = [i * dt_h for i in range(n_steps)]

    c = 0.0
    a_gut = 0.0

    dose_times = [i * dosing_interval_h for i in range(n_doses)]
    dose_applied = [False] * n_doses

    # Apply first dose at t=0
    if route == "iv":
        c = dose_mg / vd_L
    else:
        a_gut = dose_mg
    dose_applied[0] = True

    concentrations = []

    for step_i in range(n_steps):
        t = times[step_i]
        concentrations.append(c)

        # Apply subsequent doses
        for d_idx in range(1, n_doses):
            if not dose_applied[d_idx]:
                if t >= dose_times[d_idx] - dt_h * 0.5:
                    if route == "iv":
                        c += dose_mg / vd_L
                    else:
                        a_gut += dose_mg
                    dose_applied[d_idx] = True

        # Forward Euler update
        if route == "iv":
            dc_dt = -ke * c
            c = max(0.0, c + dc_dt * dt_h)
        else:
            da_gut_dt = -ka_per_h * a_gut
            dc_dt = (ka_per_h * a_gut) / vd_L - ke * c
            a_gut = max(0.0, a_gut + da_gut_dt * dt_h)
            c = max(0.0, c + dc_dt * dt_h)

    return times, concentrations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_neonatal_dosing(
    drug_name: str,
    dose_mg_per_kg: float,
    age_group: str,
    cl_adult_L_per_h: float = 10.0,
    vd_adult_L: float = 50.0,
    ka_per_h: float = 1.0,
    route: str = "iv",
    n_doses: int = 1,
    dosing_interval_h: float = 12.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> NeonatalDosingResult:
    """Simulate neonatal/infant dosing for a single age group."""
    # Validation
    if dose_mg_per_kg <= 0:
        raise ValueError("dose_mg_per_kg must be positive")
    if cl_adult_L_per_h <= 0:
        raise ValueError("cl_adult_L_per_h must be positive")
    if vd_adult_L <= 0:
        raise ValueError("vd_adult_L must be positive")
    if age_group not in _AGE_GROUPS:
        raise ValueError(f"age_group must be one of {list(_AGE_GROUPS.keys())}")
    if route not in {"iv", "oral"}:
        raise ValueError("route must be 'iv' or 'oral'")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")

    cl_eff, vd_eff, weight_kg = _compute_effective_params(age_group, cl_adult_L_per_h, vd_adult_L)
    dose_mg = dose_mg_per_kg * weight_kg

    times, concentrations = _simulate_1cpt(
        dose_mg=dose_mg,
        cl_L_per_h=cl_eff,
        vd_L=vd_eff,
        ka_per_h=ka_per_h,
        route=route,
        n_doses=n_doses,
        dosing_interval_h=dosing_interval_h,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    cmax = max(concentrations)
    tmax = times[concentrations.index(cmax)]
    auc = _trapezoidal_auc(times, concentrations)

    ke_eff = cl_eff / vd_eff
    t_half = 0.693147 / ke_eff if ke_eff > 0 else float("inf")

    ke_adult = cl_adult_L_per_h / vd_adult_L
    adult_t_half = 0.693147 / ke_adult if ke_adult > 0 else float("inf")

    fold_increase = t_half / adult_t_half if adult_t_half > 0 else 0.0

    info = _AGE_GROUPS[age_group]
    notes = (
        f"Age group: {age_group} ({weight_kg} kg). "
        f"CL maturation: {info['cl_mult'] * 100:.0f}% of adult (weight-adjusted). "
        f"Vd body-water multiplier: {info['vd_mult']}. "
        f"t_half is {fold_increase:.1f}x longer than adult reference."
    )

    return NeonatalDosingResult(
        drug_name=drug_name,
        age_group=age_group,
        dose_mg=dose_mg,
        patient_weight_kg=weight_kg,
        times_h=times,
        c_plasma_mg_L=concentrations,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_h=t_half,
        cl_effective_L_per_h=cl_eff,
        vd_effective_L=vd_eff,
        adult_t_half_h=adult_t_half,
        t_half_fold_increase=fold_increase,
        notes=notes,
    )


def compare_age_groups(
    drug_name: str,
    dose_mg_per_kg: float,
    cl_adult_L_per_h: float = 10.0,
    vd_adult_L: float = 50.0,
    ka_per_h: float = 1.0,
    route: str = "iv",
    n_doses: int = 1,
    dosing_interval_h: float = 12.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> list:
    """Simulate all age groups and return results sorted by t_half_h descending."""
    results = []
    for ag in _AGE_GROUPS:
        r = simulate_neonatal_dosing(
            drug_name=drug_name,
            dose_mg_per_kg=dose_mg_per_kg,
            age_group=ag,
            cl_adult_L_per_h=cl_adult_L_per_h,
            vd_adult_L=vd_adult_L,
            ka_per_h=ka_per_h,
            route=route,
            n_doses=n_doses,
            dosing_interval_h=dosing_interval_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(r)
    results.sort(key=lambda x: x.t_half_h, reverse=True)
    return results
