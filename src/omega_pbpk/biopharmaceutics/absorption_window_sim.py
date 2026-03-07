"""Oral absorption window simulation.

Simulates how GI transit time and regional absorption windows affect
bioavailability for drugs with site-specific absorption.  Models drug
movement through stomach, upper SI, lower SI, and colon with region-specific
permeability and transit rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_VALID_WINDOWS = {"upper_si", "full_si", "colon_included", "stomach_only"}

# GI segment geometry
_STOMACH_RADIUS_CM = 3.5
_SI_RADIUS_CM = 1.75
_COLON_RADIUS_CM = 3.0


@dataclass(frozen=True)
class AbsorptionWindowResult:
    """Results from an absorption window simulation."""

    drug_name: str
    dose_mg: float
    absorption_window: str  # "upper_si", "full_si", "colon_included", "stomach_only"

    # Regional absorption contributions (fraction of dose)
    f_absorbed_stomach: float
    f_absorbed_upper_si: float  # duodenum + proximal jejunum
    f_absorbed_lower_si: float  # distal jejunum + ileum
    f_absorbed_colon: float
    f_absorbed_total: float  # sum of above

    # Transit and timing
    stomach_emptying_h: float
    small_intestine_transit_h: float  # total SI transit time
    colon_transit_h: float

    # Optimal formulation recommendation
    recommended_formulation: str  # "immediate_release", "modified_release_4h", "modified_release_8h"

    times_h: list[float]
    absorption_rate_mg_per_h: list[float]  # total absorption rate over time

    notes: str


def simulate_absorption_window(
    drug_name: str,
    dose_mg: float,
    peff_cm_per_s: float,
    solubility_mg_mL: float,
    dose_volume_mL: float = 250.0,
    stomach_emptying_h: float = 0.5,
    si_transit_h: float = 4.0,
    colon_transit_h: float = 20.0,
    dose_number: float | None = None,
    absorption_window: str = "full_si",
    f_degraded_colon: float = 0.8,
    t_end_h: float = 28.0,
    dt_h: float = 0.1,
) -> AbsorptionWindowResult:
    """Simulate oral drug absorption considering GI transit and window constraints.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose (mg).
    peff_cm_per_s : float
        Effective intestinal permeability (cm/s), must be > 0.
    solubility_mg_mL : float
        Aqueous solubility (mg/mL), must be > 0.
    dose_volume_mL : float
        Volume of water co-administered (mL).
    stomach_emptying_h : float
        Gastric emptying time (h), must be > 0.
    si_transit_h : float
        Total small intestinal transit time (h), must be > 0.
    colon_transit_h : float
        Colonic transit time (h), must be > 0.
    dose_number : float or None
        Dose number (Dn = dose / (solubility * dose_volume)); computed if None.
    absorption_window : str
        One of "upper_si", "full_si", "colon_included", "stomach_only".
    f_degraded_colon : float
        Fraction of drug degraded/metabolised in colon (reduces colonic absorption).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration time step (h).

    Returns
    -------
    AbsorptionWindowResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if peff_cm_per_s <= 0:
        raise ValueError("peff_cm_per_s must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if stomach_emptying_h <= 0:
        raise ValueError("stomach_emptying_h must be > 0")
    if si_transit_h <= 0:
        raise ValueError("si_transit_h must be > 0")
    if colon_transit_h <= 0:
        raise ValueError("colon_transit_h must be > 0")
    if absorption_window not in _VALID_WINDOWS:
        raise ValueError(
            f"absorption_window must be one of {sorted(_VALID_WINDOWS)}, "
            f"got '{absorption_window}'"
        )

    # --- Dose number ---
    if dose_number is None:
        dose_number = dose_mg / (solubility_mg_mL * dose_volume_mL)

    # --- Dissolution limitation factor ---
    # If Dn > 1 drug is poorly soluble; scale absorption by dissolved fraction
    diss_factor = min(1.0, 1.0 / dose_number) if dose_number > 0 else 1.0

    # --- Regional absorption rate constants (h⁻¹) ---
    # ka = 2 * Peff / radius [s⁻¹] * 3600 [h⁻¹]
    peff_h = peff_cm_per_s * 3600.0  # cm/h

    # Base ka values
    ka_stom_base = 2.0 * peff_h / _STOMACH_RADIUS_CM
    ka_usi_base = 2.0 * peff_h / _SI_RADIUS_CM
    ka_lsi_base = 2.0 * peff_h / _SI_RADIUS_CM * 0.7  # reduced permeability in lower SI
    ka_col_base = 2.0 * peff_h / _COLON_RADIUS_CM * (1.0 - f_degraded_colon)

    # Apply dissolution limitation and absorption window constraints
    if absorption_window == "stomach_only":
        ka_stom = ka_stom_base * diss_factor
        ka_usi = 0.0
        ka_lsi = 0.0
        ka_col = 0.0
    elif absorption_window == "upper_si":
        ka_stom = ka_stom_base * diss_factor
        ka_usi = ka_usi_base * diss_factor
        ka_lsi = 0.0
        ka_col = 0.0
    elif absorption_window == "full_si":
        ka_stom = ka_stom_base * diss_factor
        ka_usi = ka_usi_base * diss_factor
        ka_lsi = ka_lsi_base * diss_factor
        ka_col = 0.0
    else:  # "colon_included"
        ka_stom = ka_stom_base * diss_factor
        ka_usi = ka_usi_base * diss_factor
        ka_lsi = ka_lsi_base * diss_factor
        ka_col = ka_col_base * diss_factor

    # --- Transit rate constants (h⁻¹) ---
    k_stom_exit = 1.0 / stomach_emptying_h
    k_si_exit = 1.0 / si_transit_h  # split equally between upper/lower SI
    k_col_exit = 1.0 / colon_transit_h

    # --- Time grid ---
    n_steps = max(int(t_end_h / dt_h) + 1, 2)
    times = np.linspace(0.0, t_end_h, n_steps)

    # --- State variables ---
    m_stomach = dose_mg
    m_upper_si = 0.0
    m_lower_si = 0.0
    m_colon = 0.0

    # Accumulated absorbed mass per region
    abs_stomach = 0.0
    abs_usi = 0.0
    abs_lsi = 0.0
    abs_col = 0.0

    abs_rate_arr = np.zeros(n_steps)

    for i in range(1, n_steps):
        # Absorption fluxes (mg/h) from each compartment
        flux_abs_stom = ka_stom * m_stomach
        flux_abs_usi = ka_usi * m_upper_si
        flux_abs_lsi = ka_lsi * m_lower_si
        flux_abs_col = ka_col * m_colon

        # Transit fluxes (mg/h)
        flux_stom_to_usi = k_stom_exit * m_stomach
        flux_usi_to_lsi = (k_si_exit * 0.5) * m_upper_si
        flux_lsi_to_col = (k_si_exit * 0.5) * m_lower_si
        flux_col_exit = k_col_exit * m_colon

        # Total rate at this step for tracking
        total_abs_rate = flux_abs_stom + flux_abs_usi + flux_abs_lsi + flux_abs_col
        abs_rate_arr[i] = total_abs_rate

        # Update masses (forward Euler)
        m_stomach = max(
            m_stomach + (-(flux_stom_to_usi + flux_abs_stom)) * dt_h, 0.0
        )
        m_upper_si = max(
            m_upper_si + (flux_stom_to_usi - flux_usi_to_lsi - flux_abs_usi) * dt_h,
            0.0,
        )
        m_lower_si = max(
            m_lower_si + (flux_usi_to_lsi - flux_lsi_to_col - flux_abs_lsi) * dt_h,
            0.0,
        )
        m_colon = max(
            m_colon + (flux_lsi_to_col - flux_col_exit - flux_abs_col) * dt_h, 0.0
        )

        # Accumulate absorbed mass per region
        abs_stomach += flux_abs_stom * dt_h
        abs_usi += flux_abs_usi * dt_h
        abs_lsi += flux_abs_lsi * dt_h
        abs_col += flux_abs_col * dt_h

    # Normalise by dose
    f_abs_stom = abs_stomach / dose_mg
    f_abs_usi = abs_usi / dose_mg
    f_abs_lsi = abs_lsi / dose_mg
    f_abs_col = abs_col / dose_mg
    f_abs_total = f_abs_stom + f_abs_usi + f_abs_lsi + f_abs_col

    # --- Formulation recommendation ---
    if f_abs_total > 0.8:
        recommended_formulation = "immediate_release"
    elif (f_abs_lsi + f_abs_col) < 0.1:
        # Absorption mostly in upper SI → MR could extend absorption
        recommended_formulation = "modified_release_4h"
    else:
        recommended_formulation = "modified_release_8h"

    # --- Notes ---
    notes_parts = [f"Dose number Dn={dose_number:.2f}."]
    if dose_number > 1:
        notes_parts.append("BCS II/IV: dissolution-limited absorption.")
    if absorption_window == "stomach_only":
        notes_parts.append("Absorption restricted to stomach — limited bioavailability expected.")
    elif absorption_window == "upper_si":
        notes_parts.append("Absorption restricted to upper small intestine.")
    notes = " ".join(notes_parts)

    return AbsorptionWindowResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        absorption_window=absorption_window,
        f_absorbed_stomach=f_abs_stom,
        f_absorbed_upper_si=f_abs_usi,
        f_absorbed_lower_si=f_abs_lsi,
        f_absorbed_colon=f_abs_col,
        f_absorbed_total=f_abs_total,
        stomach_emptying_h=stomach_emptying_h,
        small_intestine_transit_h=si_transit_h,
        colon_transit_h=colon_transit_h,
        recommended_formulation=recommended_formulation,
        times_h=times.tolist(),
        absorption_rate_mg_per_h=abs_rate_arr.tolist(),
        notes=notes,
    )


def compare_absorption_windows(
    drug_name: str,
    dose_mg: float,
    peff_cm_per_s: float,
    solubility_mg_mL: float,
) -> dict:
    """Compare all four absorption windows for a given drug.

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Administered dose (mg).
    peff_cm_per_s : float
        Effective permeability (cm/s).
    solubility_mg_mL : float
        Aqueous solubility (mg/mL).

    Returns
    -------
    dict with keys: "upper_si", "full_si", "colon_included", "stomach_only",
    "highest_f_absorbed" (window name with highest total absorption fraction).
    """
    results: dict[str, AbsorptionWindowResult] = {}
    for window in _VALID_WINDOWS:
        results[window] = simulate_absorption_window(
            drug_name=drug_name,
            dose_mg=dose_mg,
            peff_cm_per_s=peff_cm_per_s,
            solubility_mg_mL=solubility_mg_mL,
            absorption_window=window,
        )

    best_window = max(results, key=lambda w: results[w].f_absorbed_total)
    return {**results, "highest_f_absorbed": best_window}
