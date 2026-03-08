"""Phase 475 — Cutaneous Drug Permeation.

Models drug permeation through skin layers (SC → systemic) using the Potts-Guy
equation for skin permeability and forward-Euler ODE integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Layer geometry constants (per cm² of skin area)
# ---------------------------------------------------------------------------
_V_SC_PER_CM2 = 0.0015  # cm³/cm² (15 µm thick)
_V_EPI_PER_CM2 = 0.01  # cm³/cm² (100 µm thick)
_V_DERMIS_PER_CM2 = 0.1  # cm³/cm² (1000 µm thick)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class CutaneousPermeationResult:
    """Results from a cutaneous permeation simulation."""

    drug_name: str
    dose_mg: float
    mw_Da: float
    logP: float
    area_cm2: float
    times_h: list = field(default_factory=list)
    c_sc_mg_per_cm2: list = field(default_factory=list)
    flux_mg_per_cm2_per_h: list = field(default_factory=list)
    cumulative_absorbed_mg: list = field(default_factory=list)
    kp_cm_per_h: float = 0.0
    total_absorbed_mg: float = 0.0
    bioavailability_pct: float = 0.0
    t_lag_h: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Potts-Guy permeability
# ---------------------------------------------------------------------------
def _compute_kp(logP: float, mw_Da: float) -> float:
    """Return Kp in cm/h using the Potts-Guy equation.

    logKp = -2.7 + 0.71*logP - 0.0061*MW
    """
    log_kp = -2.7 + 0.71 * logP - 0.0061 * mw_Da
    return 10.0**log_kp


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def simulate_cutaneous_permeation(
    drug_name: str,
    dose_mg: float,
    mw_Da: float,
    logP: float,
    area_cm2: float = 100.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> CutaneousPermeationResult:
    """Simulate transdermal drug permeation from patch to systemic circulation.

    Uses a simplified single-SC-compartment model where drug in the stratum
    corneum diffuses into systemic circulation via Fickian permeation.

    Parameters
    ----------
    drug_name:  Identifier string for the drug.
    dose_mg:    Applied dose onto SC surface (mg).
    mw_Da:      Molecular weight (Da).
    logP:       Lipophilicity (octanol-water partition coefficient, log scale).
    area_cm2:   Patch application area (cm²); default 100 cm².
    t_end_h:    Simulation end time (h); default 24 h.
    dt_h:       Forward-Euler time step (h); default 0.1 h.

    Returns
    -------
    CutaneousPermeationResult with time-series and summary statistics.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if area_cm2 <= 0:
        raise ValueError("area_cm2 must be > 0")

    # --- Permeability ---
    kp = _compute_kp(logP, mw_Da)

    # Total SC volume for the patch area (mL = cm³ since density ≈ 1)
    v_sc_total = _V_SC_PER_CM2 * area_cm2  # cm³

    # Rate constant for SC → systemic (h⁻¹)
    # dA_sc/dt = -kp * area * C_sc  where C_sc = A_sc / v_sc_total
    # => dA_sc/dt = -(kp * area / v_sc_total) * A_sc = -k_elim * A_sc
    k_elim = kp * area_cm2 / v_sc_total  # = kp / _V_SC_PER_CM2  (h⁻¹)

    # --- Time-course ---
    n_steps = max(1, int(round(t_end_h / dt_h)))
    times: list[float] = []
    c_sc_list: list[float] = []
    flux_list: list[float] = []
    cumulative: list[float] = []

    a_sc = dose_mg  # amount in SC at t=0 (mg)
    cum_absorbed = 0.0
    max_flux = 0.0

    for i in range(n_steps + 1):
        t = i * dt_h
        c_sc = a_sc / v_sc_total  # mg/cm³ = mg/mL
        c_sc_per_cm2 = a_sc / area_cm2  # mg/cm²
        flux = kp * c_sc  # mg/(cm²·h)  [= kp * C_sc]
        # Equivalently flux_total = kp * area * c_sc  (mg/h), per cm² = flux

        times.append(t)
        c_sc_list.append(c_sc_per_cm2)
        flux_list.append(flux)
        cumulative.append(cum_absorbed)

        if flux > max_flux:
            max_flux = flux

        # Forward Euler — amount absorbed this step
        absorbed_this_step = k_elim * a_sc * dt_h
        cum_absorbed += absorbed_this_step
        a_sc -= absorbed_this_step
        if a_sc < 0.0:
            a_sc = 0.0

    total_absorbed = cumulative[-1]
    bioavailability_pct = (total_absorbed / dose_mg) * 100.0 if dose_mg > 0 else 0.0

    # --- Lag time: first time flux reaches 50% of max ---
    half_max = 0.5 * max_flux
    t_lag = 0.0
    for t_val, f_val in zip(times, flux_list):
        if f_val >= half_max:
            t_lag = t_val
            break

    notes = (
        f"Potts-Guy Kp={kp:.4e} cm/h; k_elim={k_elim:.4f} h⁻¹; "
        f"patch area={area_cm2} cm²; simplified single-SC-compartment model."
    )

    return CutaneousPermeationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_Da=mw_Da,
        logP=logP,
        area_cm2=area_cm2,
        times_h=times,
        c_sc_mg_per_cm2=c_sc_list,
        flux_mg_per_cm2_per_h=flux_list,
        cumulative_absorbed_mg=cumulative,
        kp_cm_per_h=kp,
        total_absorbed_mg=total_absorbed,
        bioavailability_pct=bioavailability_pct,
        t_lag_h=t_lag,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Multi-area comparison
# ---------------------------------------------------------------------------
def compare_formulation_sites(
    drug_name: str,
    dose_mg: float,
    mw_Da: float,
    logP: float,
    areas_cm2: list,
) -> list:
    """Simulate the same drug at different patch areas and rank by absorption.

    Parameters
    ----------
    drug_name:  Drug identifier.
    dose_mg:    Applied dose (mg).
    mw_Da:      Molecular weight (Da).
    logP:       Lipophilicity.
    areas_cm2:  List of patch areas (cm²) to evaluate.

    Returns
    -------
    List of CutaneousPermeationResult sorted by total_absorbed_mg descending.
    """
    results = [
        simulate_cutaneous_permeation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            mw_Da=mw_Da,
            logP=logP,
            area_cm2=area,
        )
        for area in areas_cm2
    ]
    results.sort(key=lambda r: r.total_absorbed_mg, reverse=True)
    return results
