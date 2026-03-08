"""Phase 1094 — Drug Synovial Joint PK Extended (Arthritis Therapeutics).

Comprehensive 3-compartment synovial joint PK model for arthritis drugs
(methotrexate, corticosteroids, biologics).  Adds joint inflammation effects,
drug binding to synovial macrophages, and dose persistence.

Compartments:
  Plasma ↔ Synovial fluid ↔ Cartilage/subchondral bone
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SynovialJointPKResult:
    """Result of a synovial joint PK simulation."""

    drug_name: str
    dose_mg: float
    inflammation_factor: float
    times_h: list
    c_plasma_mg_L: list
    c_synovial_mg_L: list
    c_cartilage_mg_L: list
    cmax_synovial: float
    tmax_synovial_h: float
    auc_synovial: float
    auc_cartilage: float
    synovial_to_plasma_ratio: float
    t_above_mic_synovial_h: float
    joint_penetration_efficacy: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Default compartment volumes (L)
_VP = 5.0  # plasma
_V_SYN = 0.004  # synovial fluid (4 mL)
_V_CART = 0.01  # cartilage

# Default rate constants (per hour)
_K_PS = 0.05  # plasma → synovial permeability
_K_SC = 0.02  # synovial → cartilage
_K_SYN_ELIM = 0.1  # synovial elimination

# Default partition coefficients
_KP_SYN = 0.6
_KP_CART = 0.3


def _trapz(times: list, values: list) -> float:
    """Trapezoidal integration."""
    result = 0.0
    for i in range(1, len(times)):
        result += 0.5 * (values[i - 1] + values[i]) * (times[i] - times[i - 1])
    return result


def _classify_penetration(ratio: float) -> str:
    """Classify joint penetration efficacy from AUC_syn/AUC_plasma."""
    if ratio < 0.2:
        return "poor"
    elif ratio <= 0.5:
        return "moderate"
    else:
        return "good"


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_synovial_joint_pk(
    drug_name: str,
    dose_mg: float,
    inflammation_factor: float = 1.0,
    cl_L_per_h: float = 5.0,
    vd_plasma_L: float = 5.0,
    mic_threshold_mg_L: float = 0.1,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> SynovialJointPKResult:
    """Simulate 3-compartment synovial joint PK via Forward Euler.

    IV bolus administration: C_plasma(0) = dose_mg / vd_plasma_L.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    inflammation_factor:
        Multiplier on k_ps for inflamed joint (1–5).  Higher value →
        increased synovial permeability.
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_plasma_L:
        Plasma volume of distribution in L.
    mic_threshold_mg_L:
        MIC threshold in mg/L; used to compute t_above_mic_synovial_h.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward-Euler step size in hours.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (1.0 <= inflammation_factor <= 5.0):
        raise ValueError("inflammation_factor must be in [1, 5]")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")

    ke = cl_L_per_h / vd_plasma_L
    k_ps_eff = _K_PS * inflammation_factor  # inflamed joint more permeable

    # Initial conditions
    c_plasma = dose_mg / vd_plasma_L
    c_syn = 0.0
    c_cart = 0.0

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_syn_list: list[float] = []
    c_cart_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_syn_list.append(c_syn)
        c_cart_list.append(c_cart)

        if _ < n_steps:
            # Driving concentrations
            gradient_ps = c_plasma - c_syn / _KP_SYN
            gradient_sc = c_syn - c_cart / _KP_CART

            # ODEs – Forward Euler
            dc_plasma = (-ke * c_plasma - k_ps_eff * gradient_ps) * dt_h

            dc_syn = (
                k_ps_eff * (_VP / _V_SYN) * gradient_ps - _K_SC * gradient_sc - _K_SYN_ELIM * c_syn
            ) * dt_h

            dc_cart = (_K_SC * (_V_SYN / _V_CART) * gradient_sc) * dt_h

            c_plasma = max(0.0, c_plasma + dc_plasma)
            c_syn = max(0.0, c_syn + dc_syn)
            c_cart = max(0.0, c_cart + dc_cart)

            t += dt_h

    # --- Derived PK metrics ---
    auc_plasma = _trapz(times, c_plasma_list)
    auc_synovial = _trapz(times, c_syn_list)
    auc_cartilage = _trapz(times, c_cart_list)

    cmax_synovial = max(c_syn_list)
    tmax_synovial_h = times[c_syn_list.index(cmax_synovial)]

    if auc_plasma > 0:
        synovial_to_plasma_ratio = auc_synovial / auc_plasma
    else:
        synovial_to_plasma_ratio = 0.0

    # Time above MIC in synovial fluid
    t_above_mic = sum(dt_h for i, c in enumerate(c_syn_list[:-1]) if c > mic_threshold_mg_L)

    joint_penetration_efficacy = _classify_penetration(synovial_to_plasma_ratio)

    notes = (
        f"Inflammation factor {inflammation_factor:.1f}x "
        f"(k_ps_eff={k_ps_eff:.3f}/h). "
        f"Synovial penetration: {joint_penetration_efficacy}. "
        f"t>MIC: {t_above_mic:.1f} h."
    )

    return SynovialJointPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        inflammation_factor=inflammation_factor,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_synovial_mg_L=c_syn_list,
        c_cartilage_mg_L=c_cart_list,
        cmax_synovial=cmax_synovial,
        tmax_synovial_h=tmax_synovial_h,
        auc_synovial=auc_synovial,
        auc_cartilage=auc_cartilage,
        synovial_to_plasma_ratio=synovial_to_plasma_ratio,
        t_above_mic_synovial_h=t_above_mic,
        joint_penetration_efficacy=joint_penetration_efficacy,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Multi-state comparison helper
# ---------------------------------------------------------------------------


def compare_inflammation_states(
    drug_name: str,
    dose_mg: float,
    inflammation_factors: list,
) -> list:
    """Simulate multiple inflammation states and return results sorted by auc_synovial descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    inflammation_factors:
        List of inflammation factor values (each must be in [1, 5]).
    """
    results = [
        simulate_synovial_joint_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            inflammation_factor=f,
        )
        for f in inflammation_factors
    ]
    results.sort(key=lambda r: r.auc_synovial, reverse=True)
    return results
