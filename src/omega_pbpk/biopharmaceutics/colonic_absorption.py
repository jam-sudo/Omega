"""Colonic absorption model for modified-release formulations (Phase 431).

Models drug absorption in the colon, which is relevant for extended/modified-release
dosage forms where drug release continues beyond the small intestine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "ColonicAbsorptionResult",
    "simulate_colonic_absorption",
    "modified_release_colonic_pk",
]

# Colonic physiological constants
_V_COLON_ML: float = 40.0  # mL — low fluid volume in colon
_R_COLON_CM: float = 0.5  # cm — colon radius used for ka_colon calculation


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ColonicAbsorptionResult:
    """Result of a colonic absorption simulation."""

    drug_name: str
    dose_mg: float
    solubility_mg_mL: float
    t_transit_h: float
    times_h: list[float]
    a_colon_mg: list[float]  # drug amount in colonic depot (mg)
    c_plasma_mg_L: list[float]  # plasma concentration (mg/L)
    cmax: float  # peak plasma concentration (mg/L)
    auc: float  # AUC0-t via trapezoidal rule (mg*h/L)
    f_absorbed_colon: float  # fraction of dose absorbed from colon (0-1)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_colonic_inputs(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    peff_colon_cm_s: float,
    cl_L_per_h: float,
    vd_L: float,
    t_transit_h: float,
    t_end_h: float,
    dt_h: float,
) -> None:
    """Raise ValueError for invalid inputs."""
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string.")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0.")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0.")
    if peff_colon_cm_s <= 0:
        raise ValueError("peff_colon_cm_s must be > 0.")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0.")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0.")
    if t_transit_h <= 0:
        raise ValueError("t_transit_h must be > 0.")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0.")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0.")
    if dt_h >= t_end_h:
        raise ValueError("dt_h must be less than t_end_h.")


# ---------------------------------------------------------------------------
# Trapezoidal AUC (manual, no scipy)
# ---------------------------------------------------------------------------


def _trapz_auc(times: list[float], values: list[float]) -> float:
    """Compute AUC via the linear trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_colonic_absorption(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    peff_colon_cm_s: float,
    cl_L_per_h: float,
    vd_L: float,
    t_transit_h: float = 20.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> ColonicAbsorptionResult:
    """Simulate drug absorption in the colon via a 2-compartment ODE system.

    The colon depot (dissolved fraction) drains into the central plasma compartment
    using forward Euler integration.  Absorption ceases after the colonic transit
    time elapses (drug leaves the colon and is excreted).

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Amount of drug entering the colon (mg).
    solubility_mg_mL:
        Aqueous solubility in colon fluid (mg/mL).
    peff_colon_cm_s:
        Effective colonic permeability (cm/s).  Typically ~0.3 × Peff_SI.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_transit_h:
        Duration of colonic transit (h).  Typical range 8–36 h.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    ColonicAbsorptionResult
    """
    _validate_colonic_inputs(
        drug_name, dose_mg, solubility_mg_mL, peff_colon_cm_s,
        cl_L_per_h, vd_L, t_transit_h, t_end_h, dt_h,
    )

    # Derived PK parameters
    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # ka_colon: 2P/R, convert Peff from cm/s to cm/h first
    peff_colon_cm_h = peff_colon_cm_s * 3600.0
    ka_colon = 2.0 * peff_colon_cm_h / _R_COLON_CM  # 1/h

    # Dissolved fraction: limited by solubility and colonic fluid volume
    dissolved_frac = min(1.0, dose_mg / (solubility_mg_mL * _V_COLON_ML))

    notes: list[str] = []
    if dissolved_frac < 1.0:
        pct_dissolved = dissolved_frac * 100.0
        notes.append(
            f"Only {pct_dissolved:.1f}% of dose is dissolved in colonic fluid "
            f"(solubility-limited; V_colon={_V_COLON_ML} mL)."
        )
    if t_transit_h < 8.0:
        notes.append("t_transit_h < 8 h is shorter than typical colonic transit.")
    if t_transit_h > 36.0:
        notes.append("t_transit_h > 36 h is longer than typical colonic transit.")

    # Forward Euler integration
    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)

    times: list[float] = [0.0]
    a_colon: list[float] = [dose_mg]
    c_plasma: list[float] = [0.0]

    a_col = dose_mg
    c_pla = 0.0
    t = 0.0

    for _ in range(n_steps):
        # Determine effective ka: zero after transit ends
        ka_eff = ka_colon if t < t_transit_h else 0.0

        da_colon = -ka_eff * a_col * dissolved_frac
        dc_plasma = ka_eff * a_col * dissolved_frac / vd_L - ke * c_pla

        a_col = max(a_col + da_colon * dt_h, 0.0)
        c_pla = max(c_pla + dc_plasma * dt_h, 0.0)
        t += dt_h

        times.append(t)
        a_colon.append(a_col)
        c_plasma.append(c_pla)

    cmax = max(c_plasma)
    auc = _trapz_auc(times, c_plasma)

    # Fraction absorbed = drug that left the colon depot / initial dose
    f_absorbed_colon = (dose_mg - a_colon[-1]) / dose_mg

    return ColonicAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        solubility_mg_mL=solubility_mg_mL,
        t_transit_h=t_transit_h,
        times_h=times,
        a_colon_mg=a_colon,
        c_plasma_mg_L=c_plasma,
        cmax=cmax,
        auc=auc,
        f_absorbed_colon=f_absorbed_colon,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Modified-release 2-region model
# ---------------------------------------------------------------------------


def modified_release_colonic_pk(
    drug_name: str,
    dose_mg: float,
    si_release_pct: float,
    cl_L_per_h: float,
    vd_L: float,
    solubility_mg_mL: float = 1.0,
    peff_si_cm_s: float = 2.0e-4,
    peff_colon_cm_s: float | None = None,
    t_si_transit_h: float = 3.0,
    t_colon_transit_h: float = 20.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> dict:
    """Two-region (SI + colon) PK model for a modified-release formulation.

    Fraction ``si_release_pct`` of the dose is released and absorbed in the
    small intestine (fast first-order, ka_si = 1.5 /h).  The remaining
    fraction is delivered to the colon where it is absorbed with lower
    permeability.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Total dose administered (mg).
    si_release_pct:
        Fraction (0–1) of dose released in the small intestine.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    solubility_mg_mL:
        Solubility in GI fluid (mg/mL).
    peff_si_cm_s:
        Effective SI permeability (cm/s); used for reference.
    peff_colon_cm_s:
        Effective colon permeability (cm/s).  Defaults to 0.3 × peff_si_cm_s.
    t_si_transit_h:
        Transit time through the small intestine (h).
    t_colon_transit_h:
        Transit time through the colon (h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    dict with keys:
        drug_name, dose_mg, si_release_pct, colon_dose_mg,
        si_cmax, si_auc, colon_result (ColonicAbsorptionResult),
        combined_cmax, combined_auc, combined_times_h, combined_c_plasma,
        f_si, f_colon, f_total, notes
    """
    # --- input validation ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string.")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0.")
    if not (0.0 <= si_release_pct <= 1.0):
        raise ValueError("si_release_pct must be between 0 and 1.")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0.")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0.")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0.")
    if peff_si_cm_s <= 0:
        raise ValueError("peff_si_cm_s must be > 0.")
    if t_si_transit_h <= 0:
        raise ValueError("t_si_transit_h must be > 0.")
    if t_colon_transit_h <= 0:
        raise ValueError("t_colon_transit_h must be > 0.")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0.")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0.")

    if peff_colon_cm_s is None:
        peff_colon_cm_s = 0.3 * peff_si_cm_s

    if peff_colon_cm_s <= 0:
        raise ValueError("peff_colon_cm_s must be > 0.")

    ke = cl_L_per_h / vd_L
    ka_si = 1.5  # 1/h — fast SI absorption rate

    si_dose_mg = si_release_pct * dose_mg
    colon_dose_mg = (1.0 - si_release_pct) * dose_mg

    # --- SI absorption (forward Euler, 1-compartment) ---
    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)

    times: list[float] = [0.0]
    c_si: list[float] = [0.0]

    a_si = si_dose_mg
    c_val = 0.0

    for i in range(n_steps):
        t_cur = i * dt_h
        ka_eff = ka_si if t_cur < t_si_transit_h else 0.0
        dc = ka_eff * a_si / vd_L - ke * c_val
        da = -ka_eff * a_si
        a_si = max(a_si + da * dt_h, 0.0)
        c_val = max(c_val + dc * dt_h, 0.0)
        times.append((i + 1) * dt_h)
        c_si.append(c_val)

    si_cmax = max(c_si)
    si_auc = _trapz_auc(times, c_si)
    f_si = (si_dose_mg - a_si) / dose_mg if dose_mg > 0 else 0.0

    # --- Colonic absorption ---
    if colon_dose_mg > 0:
        colon_result = simulate_colonic_absorption(
            drug_name=drug_name,
            dose_mg=colon_dose_mg,
            solubility_mg_mL=solubility_mg_mL,
            peff_colon_cm_s=peff_colon_cm_s,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_transit_h=t_colon_transit_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        f_colon = colon_result.f_absorbed_colon * (colon_dose_mg / dose_mg)
    else:
        # No colonic dose — build a zero result
        colon_result = simulate_colonic_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg * 1e-9,  # near-zero placeholder
            solubility_mg_mL=solubility_mg_mL,
            peff_colon_cm_s=peff_colon_cm_s,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_transit_h=t_colon_transit_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        f_colon = 0.0

    # --- Combine SI + colon plasma profiles (superposition) ---
    n = len(times)
    c_colon_aligned = colon_result.c_plasma_mg_L
    # Both arrays have same length (same dt_h, t_end_h)
    combined_c: list[float] = [c_si[i] + c_colon_aligned[i] for i in range(n)]

    combined_cmax = max(combined_c)
    combined_auc = _trapz_auc(times, combined_c)
    f_total = min(f_si + f_colon, 1.0)

    notes: list[str] = []
    if si_release_pct > 0.9:
        notes.append("Most drug released in SI; colonic contribution is minor.")
    if si_release_pct < 0.1:
        notes.append("Most drug delivered to colon; SI contribution is minor.")

    return {
        "drug_name": drug_name,
        "dose_mg": dose_mg,
        "si_release_pct": si_release_pct,
        "colon_dose_mg": colon_dose_mg,
        "si_cmax": si_cmax,
        "si_auc": si_auc,
        "colon_result": colon_result,
        "combined_cmax": combined_cmax,
        "combined_auc": combined_auc,
        "combined_times_h": times,
        "combined_c_plasma": combined_c,
        "f_si": f_si,
        "f_colon": f_colon,
        "f_total": f_total,
        "notes": notes,
    }
