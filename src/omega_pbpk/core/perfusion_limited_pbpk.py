"""Perfusion-limited PBPK model — 4 tissue groups (liver, kidney, muscle, fat)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Plasma volume (L) — fixed for a 70 kg reference human
_VD_PLASMA_L = 3.0

#: Default organ volume fractions (unitless, fraction of body weight)
_VOLUME_FRACTIONS: dict[str, float] = {
    "liver": 0.026,
    "kidney": 0.004,
    "muscle": 0.40,
    "fat": 0.21,
}

#: Default tissue blood flows (L/h), cardiac output ~ 310 L/h reference
_DEFAULT_TISSUE_FLOWS: dict[str, float] = {
    "liver": 54.0,
    "kidney": 72.0,
    "muscle": 15.0,
    "fat": 5.0,
}

_VALID_TISSUES = frozenset(_VOLUME_FRACTIONS.keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PerfusionLimitedPBPKResult:
    """Simulation output from the perfusion-limited PBPK model."""

    drug_name: str
    dose_mg: float
    cl_hep_L_per_h: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    tissue_concs: dict[str, list[float]]  # tissue name → concentration time course
    cmax_plasma: float
    auc_plasma: float  # trapezoidal AUC over simulation period (mg·h/L)
    notes: str


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_perfusion_limited_pbpk(
    drug_name: str,
    dose_mg: float,
    cl_hep_L_per_h: float,
    qh_L_per_h: float,
    vd_total_L: float,
    kp_tissues: dict[str, float],
    tissue_flows: dict[str, float],
    t_end_h: float,
    dt_h: float,
) -> PerfusionLimitedPBPKResult:
    """Simulate perfusion-limited PBPK with 4 tissue compartments.

    The model uses IV bolus administration and forward Euler integration.

    Tissue ODEs (mass, mg):
        dA_i/dt = Q_i * (C_plasma - A_i / V_i / kp_i)

    Plasma ODE (concentration, mg/L):
        dC_plasma/dt = -cl_hep * C_plasma / Vd_plasma
                       + sum_i(Q_i * (A_i / V_i / kp_i - C_plasma)) / Vd_plasma

    where V_i = vd_total * volume_fraction[i] * kp_i.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    dose_mg : float
        IV bolus dose (mg).
    cl_hep_L_per_h : float
        Hepatic elimination clearance (L/h).
    qh_L_per_h : float
        Total hepatic blood flow (L/h). Included in notes for reference; hepatic
        elimination is handled by ``cl_hep``.
    vd_total_L : float
        Total volume of distribution (L), used to distribute tissue volumes.
    kp_tissues : dict[str, float]
        Tissue-to-plasma partition coefficients for each tissue. Must contain
        keys for all four tissues: liver, kidney, muscle, fat.
    tissue_flows : dict[str, float]
        Blood flow to each tissue (L/h). Defaults are used for any missing key.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler step size (h).

    Returns
    -------
    PerfusionLimitedPBPKResult
    """
    # --- Validate inputs ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_hep_L_per_h < 0:
        raise ValueError("cl_hep_L_per_h must be >= 0")
    if qh_L_per_h < 0:
        raise ValueError("qh_L_per_h must be >= 0")
    if vd_total_L <= 0:
        raise ValueError("vd_total_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    missing_kp = _VALID_TISSUES - set(kp_tissues.keys())
    if missing_kp:
        raise ValueError(f"kp_tissues missing entries for: {sorted(missing_kp)}")
    for tissue, kp in kp_tissues.items():
        if tissue not in _VALID_TISSUES:
            raise ValueError(f"Unknown tissue '{tissue}'; valid: {sorted(_VALID_TISSUES)}")
        if kp <= 0:
            raise ValueError(f"kp_tissues['{tissue}'] must be > 0; got {kp}")

    # Merge user-provided flows with defaults
    flows: dict[str, float] = {**_DEFAULT_TISSUE_FLOWS, **tissue_flows}
    for tissue in _VALID_TISSUES:
        if flows[tissue] <= 0:
            raise ValueError(f"tissue_flows['{tissue}'] must be > 0; got {flows[tissue]}")

    # --- Tissue volumes ---
    # V_i = vd_total * vf_i * kp_i
    tissue_volumes: dict[str, float] = {
        t: vd_total_L * _VOLUME_FRACTIONS[t] * kp_tissues[t] for t in _VALID_TISSUES
    }

    # --- Compute stable internal dt via CFL condition ---
    # For tissue i: dA_i/dt = Q_i*(C_plasma - A_i/V_i/kp_i)
    # The stiff rate constant is lambda_i = Q_i / V_i.
    # Forward Euler stability: dt * lambda_i < 1  → dt < V_i / Q_i
    # We also need dt for plasma ODE: sum(Q_i)/Vd_plasma + cl_hep/Vd_plasma < 1/dt
    max_lambda = max(flows[t] / tissue_volumes[t] for t in _VALID_TISSUES)
    plasma_lambda = (sum(flows[t] for t in _VALID_TISSUES) + cl_hep_L_per_h) / _VD_PLASMA_L
    overall_lambda = max(max_lambda, plasma_lambda)
    # Stability limit: dt_stable < 1 / lambda; use 0.5 safety factor
    dt_stable = 0.5 / overall_lambda if overall_lambda > 0 else dt_h
    # Number of sub-steps per user-facing step
    n_sub = max(1, int(dt_h / dt_stable) + 1)
    dt_inner = dt_h / n_sub

    # --- Initial conditions ---
    # IV bolus: all drug in plasma
    c_plasma = dose_mg / _VD_PLASMA_L
    tissue_amounts: dict[str, float] = {t: 0.0 for t in _VALID_TISSUES}

    # --- Storage ---
    n_steps = max(int(t_end_h / dt_h), 1)
    times = [0.0] * (n_steps + 1)
    c_plasma_arr = [0.0] * (n_steps + 1)
    tissue_arr: dict[str, list[float]] = {t: [0.0] * (n_steps + 1) for t in _VALID_TISSUES}

    times[0] = 0.0
    c_plasma_arr[0] = c_plasma
    for t_name in _VALID_TISSUES:
        tissue_arr[t_name][0] = 0.0  # initial tissue concentration

    # --- Forward Euler integration (with automatic sub-stepping for stability) ---
    for step in range(n_steps):
        # Inner sub-steps within each output interval
        for _ in range(n_sub):
            # Hepatic elimination flux (mg/h from plasma, removed from plasma)
            hep_elim = cl_hep_L_per_h * c_plasma  # mg/h

            # Tissue exchange fluxes
            # dA_i/dt = Q_i * (C_plasma - C_tissue_unbound)
            # C_tissue_unbound = (A_i / V_i) / kp_i
            # V_i = vd_total * vf_i * kp_i
            net_tissue_plasma_flux = 0.0
            d_tissue: dict[str, float] = {}
            for t_name in _VALID_TISSUES:
                q = flows[t_name]
                v_i = tissue_volumes[t_name]
                kp_i = kp_tissues[t_name]
                a_i = tissue_amounts[t_name]
                c_tissue_unbound = (a_i / v_i) / kp_i if v_i > 0 else 0.0
                # Mass ODE for tissue (mg/h)
                da_i = q * (c_plasma - c_tissue_unbound)
                d_tissue[t_name] = da_i
                # Return flux to plasma: Q*(c_tissue_unbound - c_plasma)
                net_tissue_plasma_flux += q * (c_tissue_unbound - c_plasma)

            # Plasma ODE:
            # Vd_plasma * dC_plasma/dt = -cl_hep*C_plasma + net_tissue_plasma_flux
            dc_plasma = (-hep_elim + net_tissue_plasma_flux) / _VD_PLASMA_L

            # Forward Euler update
            c_plasma = max(c_plasma + dc_plasma * dt_inner, 0.0)
            for t_name in _VALID_TISSUES:
                tissue_amounts[t_name] = max(
                    tissue_amounts[t_name] + d_tissue[t_name] * dt_inner, 0.0
                )

        # Record output at end of each user-facing step
        times[step + 1] = (step + 1) * dt_h
        c_plasma_arr[step + 1] = c_plasma
        for t_name in _VALID_TISSUES:
            v_i = tissue_volumes[t_name]
            tissue_arr[t_name][step + 1] = tissue_amounts[t_name] / v_i if v_i > 0 else 0.0

    # --- PK metrics ---
    c_arr = np.array(c_plasma_arr, dtype=float)
    t_arr = np.array(times, dtype=float)
    cmax_plasma = float(np.max(c_arr))
    auc_plasma = float(np_trapz(c_arr, t_arr))

    notes = (
        f"Perfusion-limited PBPK: IV bolus {dose_mg} mg, "
        f"cl_hep={cl_hep_L_per_h} L/h, qh={qh_L_per_h} L/h, "
        f"Vd_total={vd_total_L} L, Vd_plasma={_VD_PLASMA_L} L (fixed). "
        f"Volume fractions: liver={_VOLUME_FRACTIONS['liver']}, "
        f"kidney={_VOLUME_FRACTIONS['kidney']}, muscle={_VOLUME_FRACTIONS['muscle']}, "
        f"fat={_VOLUME_FRACTIONS['fat']}."
    )

    return PerfusionLimitedPBPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_hep_L_per_h=cl_hep_L_per_h,
        times_h=times,
        c_plasma_mg_L=c_plasma_arr,
        tissue_concs={t: tissue_arr[t] for t in _VALID_TISSUES},
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        notes=notes,
    )


__all__ = [
    "PerfusionLimitedPBPKResult",
    "simulate_perfusion_limited_pbpk",
]
