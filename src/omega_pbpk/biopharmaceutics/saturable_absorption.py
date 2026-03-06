"""Saturable (nonlinear) intestinal absorption model.

Models dose-dependent absorption via Michaelis-Menten transporter kinetics.
At low doses (C_gi << Km), absorption is approximately linear; at high doses
(C_gi >> Km), transporter saturation limits the absorption rate.

ODE system:
  GI compartment (mass in mg, volume V_gi mL):
    C_gi = A_gi / V_gi  (mg/mL)
    dA_gi/dt = -Jabs - k_transit * A_gi
    Jabs = Jmax * C_gi / (Km + C_gi)  [mg/h]

  Plasma compartment (concentration in mg/L, volume Vd L):
    dC_plasma/dt = Jabs / Vd - (CL / Vd) * C_plasma

References
----------
- Sun D et al. (2004) Clin Pharmacokinet 43:233-252
- Benet LZ (2010) AAPS J 12:357-363 (nonlinear absorption in drug development)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "SaturableAbsorptionResult",
    "simulate_saturable_absorption",
    "dose_escalation_sat",
]


@dataclass
class SaturableAbsorptionResult:
    """Results from a saturable absorption simulation."""

    drug_name: str
    dose_mg: float
    jmax_mg_per_h: float
    km_mg_mL: float
    times_h: np.ndarray
    c_gi_mg_mL: np.ndarray
    c_plasma_mg_L: np.ndarray
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_absorbed: float
    dose_proportional: bool
    saturation_fraction: float
    notes: list[str] = field(default_factory=list)


def simulate_saturable_absorption(
    drug_name: str,
    dose_mg: float,
    jmax_mg_per_h: float,
    km_mg_mL: float,
    cl_L_per_h: float,
    vd_L: float,
    v_gi_mL: float = 250.0,
    k_transit_per_h: float = 0.05,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
    _ref_f_absorbed: float | None = None,
) -> SaturableAbsorptionResult:
    """Simulate saturable intestinal absorption with Michaelis-Menten kinetics.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Oral dose in mg. Must be > 0.
    jmax_mg_per_h:
        Maximum absorption rate (mg/h). Must be > 0.
    km_mg_mL:
        Michaelis constant (mg/mL). Must be > 0.
    cl_L_per_h:
        Plasma clearance (L/h). Must be > 0.
    vd_L:
        Volume of distribution (L). Must be > 0.
    v_gi_mL:
        GI compartment volume (mL). Default 250 mL.
    k_transit_per_h:
        First-order non-absorption GI transit/loss rate (h^-1). Default 0.05 /h.
    t_end_h:
        Simulation end time (h). Default 12 h.
    dt_h:
        Forward Euler time step (h). Default 0.05 h.
    _ref_f_absorbed:
        Internal parameter used by dose_escalation_sat to pass the reference
        f_absorbed for dose-proportionality assessment. Not intended for
        direct use.

    Returns
    -------
    SaturableAbsorptionResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if jmax_mg_per_h <= 0:
        raise ValueError(f"jmax_mg_per_h must be > 0, got {jmax_mg_per_h}")
    if km_mg_mL <= 0:
        raise ValueError(f"km_mg_mL must be > 0, got {km_mg_mL}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    notes: list[str] = []
    ke = cl_L_per_h / vd_L  # elimination rate constant (h^-1)

    n_steps = int(round(t_end_h / dt_h)) + 1
    times = np.linspace(0.0, t_end_h, n_steps)

    c_gi = np.zeros(n_steps)  # mg/mL
    c_plasma = np.zeros(n_steps)  # mg/L

    # Initial conditions: dose deposited in GI compartment
    a_gi = dose_mg  # mass in GI (mg)
    c_gi[0] = a_gi / v_gi_mL
    c_plasma[0] = 0.0

    # Track total absorbed mass
    total_absorbed = 0.0

    for i in range(n_steps - 1):
        cgi = a_gi / v_gi_mL  # current GI concentration (mg/mL)

        # Michaelis-Menten absorption rate (mg/h)
        j_abs = jmax_mg_per_h * cgi / (km_mg_mL + cgi)

        # GI mass change
        da_gi = -(j_abs + k_transit_per_h * a_gi) * dt_h
        a_gi = max(0.0, a_gi + da_gi)

        # Track absorbed mass for this step
        total_absorbed += j_abs * dt_h

        # Plasma concentration change
        cp = c_plasma[i]
        dcp = j_abs / vd_L - ke * cp
        c_plasma[i + 1] = max(0.0, cp + dcp * dt_h)
        c_gi[i + 1] = max(0.0, a_gi / v_gi_mL)

    # --- NCA metrics ---
    cmax = float(np.max(c_plasma))
    tmax = float(times[int(np.argmax(c_plasma))])
    auc = float(np_trapz(c_plasma, times))
    f_absorbed = float(np.clip(total_absorbed / dose_mg, 0.0, 1.0))

    # Saturation fraction: peak GI concentration relative to Km
    cmax_gi = float(np.max(c_gi))
    saturation_fraction = float(cmax_gi / (km_mg_mL + cmax_gi))

    # Dose-proportionality: compare f_absorbed to reference (lowest dose)
    if _ref_f_absorbed is not None:
        ref = _ref_f_absorbed
        if ref > 0:
            # Within 20% of reference f_absorbed
            dose_proportional = abs(f_absorbed - ref) / ref <= 0.20
        else:
            dose_proportional = f_absorbed <= 0.20
    else:
        # Default: always proportional at first call (self-referential)
        dose_proportional = True

    # Notes
    c0_gi = dose_mg / v_gi_mL
    if c0_gi > 5.0 * km_mg_mL:
        notes.append(
            f"Initial GI concentration ({c0_gi:.3f} mg/mL) >> Km ({km_mg_mL:.3f} mg/mL): "
            "transporter saturated at dose onset."
        )
    elif c0_gi < 0.2 * km_mg_mL:
        notes.append(
            f"Initial GI concentration ({c0_gi:.3f} mg/mL) << Km ({km_mg_mL:.3f} mg/mL): "
            "near-linear absorption."
        )

    return SaturableAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        jmax_mg_per_h=jmax_mg_per_h,
        km_mg_mL=km_mg_mL,
        times_h=times,
        c_gi_mg_mL=c_gi,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_absorbed=f_absorbed,
        dose_proportional=dose_proportional,
        saturation_fraction=saturation_fraction,
        notes=notes,
    )


def dose_escalation_sat(
    drug_name: str,
    doses_mg: list[float],
    jmax_mg_per_h: float,
    km_mg_mL: float,
    cl_L_per_h: float,
    vd_L: float,
    **kwargs: object,
) -> list[SaturableAbsorptionResult]:
    """Simulate saturable absorption across a dose escalation series.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    doses_mg:
        List of doses (mg) to simulate. Results sorted by dose_mg ascending.
    jmax_mg_per_h:
        Maximum absorption rate (mg/h).
    km_mg_mL:
        Michaelis constant (mg/mL).
    cl_L_per_h:
        Plasma clearance (L/h).
    vd_L:
        Volume of distribution (L).
    **kwargs:
        Additional keyword arguments passed to simulate_saturable_absorption
        (v_gi_mL, k_transit_per_h, t_end_h, dt_h).

    Returns
    -------
    list[SaturableAbsorptionResult]
        Results sorted by dose_mg (ascending).
    """
    sorted_doses = sorted(doses_mg)

    # Use the lowest dose as reference for dose-proportionality
    ref_result = simulate_saturable_absorption(
        drug_name=drug_name,
        dose_mg=sorted_doses[0],
        jmax_mg_per_h=jmax_mg_per_h,
        km_mg_mL=km_mg_mL,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        **kwargs,
    )
    ref_f = ref_result.f_absorbed

    results: list[SaturableAbsorptionResult] = []
    for dose in sorted_doses:
        r = simulate_saturable_absorption(
            drug_name=drug_name,
            dose_mg=dose,
            jmax_mg_per_h=jmax_mg_per_h,
            km_mg_mL=km_mg_mL,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            _ref_f_absorbed=ref_f,
            **kwargs,
        )
        results.append(r)

    return results
