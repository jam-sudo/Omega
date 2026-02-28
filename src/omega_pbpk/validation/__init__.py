"""Validation utilities — mass balance and physiological sanity checks.

These checks ensure model integrity by verifying conservation of mass
and plausibility of physiological parameters.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray


def mass_balance_check(
    amounts: NDArray[np.floating[Any]],
    dose_mg: float,
    time_h: NDArray[np.floating[Any]] | None = None,
    tolerance_frac: float = 0.005,
) -> list[str]:
    """Verify mass conservation across all compartments.

    Works for both IV and oral routes.  For oral dosing the ``amounts`` array
    must include **all** state variables — including GI-tract compartments and
    any sink states for metabolised/excreted drug — so that the row sum equals
    the conserved mass at every timepoint:

        sum(all states) ≈ dose  (IV, at all times)
        sum(all states including GI residual) ≈ dose  (oral, at all times)

    For oral dosing the row sum naturally accounts for:
        absorbed_into_systemic + GI_remaining + metabolised + excreted ≈ dose

    If GI absorption is still in progress the deviation will be within
    tolerance as long as the GI compartment is part of the state vector.
    Use :func:`oral_mass_balance_check` for a post-absorption point check
    that does not require sink states to be tracked.

    Args:
        amounts: Array of shape (n_time, n_states) from SimulationResult.amounts.
        dose_mg: Administered dose (mg).
        time_h: Optional time array for pinpointing deviations.
        tolerance_frac: Maximum allowed fractional deviation from dose (default 0.5%).

    Returns:
        List of warning messages (empty if mass is conserved).
    """
    warnings_out: list[str] = []
    if dose_mg <= 0:
        return warnings_out

    total_mass = np.sum(amounts, axis=1)
    deviation_frac = np.abs(total_mass - dose_mg) / dose_mg
    max_dev_idx = int(np.argmax(deviation_frac))
    max_dev = float(deviation_frac[max_dev_idx])

    if max_dev > tolerance_frac:
        t_info = ""
        if time_h is not None and max_dev_idx < len(time_h):
            t_info = f" at t={float(time_h[max_dev_idx]):.2f}h"
        msg = (
            f"Mass balance deviation {max_dev:.4%} exceeds tolerance "
            f"{tolerance_frac:.4%}{t_info}: "
            f"total_mass={float(total_mass[max_dev_idx]):.6f} mg vs dose={dose_mg:.6f} mg"
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        warnings_out.append(msg)

    return warnings_out


def physiologic_sanity_check(
    organs: dict[str, Any],
    cardiac_output: float,
    fup: float,
    flow_tolerance_frac: float = 0.05,
) -> list[str]:
    """Run non-fatal physiological plausibility checks on model parameters.

    Args:
        organs: Dict of Organ objects from WholeBodyPBPK.organs.
        cardiac_output: Cardiac output (L/h).
        fup: Fraction unbound in plasma.
        flow_tolerance_frac: Tolerance for blood flow conservation check.

    Returns:
        List of warning messages (empty if all checks pass).
    """
    warnings_out: list[str] = []

    # 1. Blood flow conservation: non-lung organ flows ≈ cardiac output
    excluded = {"venous_blood", "arterial_blood", "lung", "portal_vein"}
    total_flow = sum(org.blood_flow_L_per_h for name, org in organs.items() if name not in excluded)
    if cardiac_output > 0:
        flow_err = abs(total_flow - cardiac_output) / cardiac_output
        if flow_err > flow_tolerance_frac:
            msg = (
                f"Blood flow imbalance: organ flows sum to {total_flow:.2f} L/h "
                f"vs cardiac output {cardiac_output:.2f} L/h "
                f"(relative error {flow_err:.2%})"
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            warnings_out.append(msg)

    # 2. Volume non-negativity
    neg_vols = [name for name, org in organs.items() if org.volume_L <= 0]
    if neg_vols:
        msg = f"Non-positive organ volumes: {', '.join(neg_vols)}"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        warnings_out.append(msg)

    # 3. Kp plausibility (should be > 0 for all organs)
    bad_kp = [(name, org.kp) for name, org in organs.items() if hasattr(org, "kp") and org.kp <= 0]
    if bad_kp:
        msg = f"Non-positive Kp values: {bad_kp}"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        warnings_out.append(msg)

    # 4. Fraction unbound plausibility
    if not (0.0 < fup <= 1.0):
        msg = f"fup should be in (0, 1]; got {fup:.4g}"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        warnings_out.append(msg)

    return warnings_out


def oral_mass_balance_check(
    amounts: NDArray[np.floating[Any]],
    dose_mg: float,
    time_h: NDArray[np.floating[Any]],
    gi_states: NDArray[np.floating[Any]] | None = None,
    tolerance_frac: float = 0.02,
    min_absorption_frac: float = 0.99,
) -> list[str]:
    """Verify mass conservation for oral dosing at simulation end.

    For oral dosing the simple sum-of-states check is only valid after GI
    absorption is complete (residual GI mass < 1%). This function checks
    conservation only at the final timepoint when GI residual is negligible.

    The mass balance identity at t_end is:
        dose ≈ absorbed_mass + metabolised + excreted + GI_residual

    In practice this reduces to: sum(all 35 states at t_end) ≈ dose,
    because metabolised/excreted mass is included in the amounts array as
    sink compartments (states 30-33 in the 35-state ODE).

    Args:
        amounts: Array of shape (n_time, n_states) from SimulationResult.amounts.
        dose_mg: Administered oral dose (mg).
        time_h: Time array (h).
        gi_states: Optional sub-array (n_time, n_gi_states) of GI compartment
            amounts only. If provided, used to compute GI residual fraction.
            If None, the check is performed unconditionally at the last timepoint.
        tolerance_frac: Maximum allowed fractional deviation from dose (default 2%).
        min_absorption_frac: Fraction of dose that must have left GI before checking.
            Only used when gi_states is provided (default 99%).

    Returns:
        List of warning messages (empty if mass is conserved or GI absorption
        not yet complete).
    """
    warnings_out: list[str] = []
    if dose_mg <= 0:
        return warnings_out

    # Determine the timepoint at which to evaluate mass balance
    if gi_states is not None and len(gi_states) > 0:
        gi_total = np.sum(gi_states, axis=1)
        gi_frac_remaining = gi_total / max(dose_mg, 1e-12)
        # Find first timepoint where GI fraction absorbed >= min_absorption_frac
        absorbed_frac = 1.0 - gi_frac_remaining
        check_idx_arr = np.where(absorbed_frac >= min_absorption_frac)[0]
        if len(check_idx_arr) == 0:
            # Absorption not complete; skip check
            return warnings_out
        check_idx = int(check_idx_arr[0])
    else:
        # Use final timepoint
        check_idx = -1

    total_mass = float(np.sum(amounts[check_idx]))
    deviation_frac = abs(total_mass - dose_mg) / dose_mg
    t_val = float(time_h[check_idx]) if time_h is not None and len(time_h) > abs(check_idx) else None

    if deviation_frac > tolerance_frac:
        t_info = f" at t={t_val:.2f}h" if t_val is not None else ""
        msg = (
            f"Oral mass balance deviation {deviation_frac:.4%} exceeds tolerance "
            f"{tolerance_frac:.4%}{t_info}: "
            f"total_mass={total_mass:.6f} mg vs dose={dose_mg:.6f} mg"
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        warnings_out.append(msg)

    return warnings_out


__all__ = ["mass_balance_check", "oral_mass_balance_check", "physiologic_sanity_check"]
