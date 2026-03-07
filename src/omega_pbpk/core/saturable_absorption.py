"""Saturable absorption kinetics — carrier-mediated transporter saturation at high doses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class SaturableAbsorptionResult:
    """Result from a single-dose saturable absorption simulation."""

    dose_mg: float
    times_h: list[float]
    c_plasma: list[float]
    f_absorbed: float  # fraction of dose absorbed into plasma
    cmax: float
    tmax_h: float
    auc: float
    saturation_index: float  # Jmax/(Jmax + passive_rate*Km) evaluated at Cmax
    notes: str


@dataclass(frozen=True)
class DoseProportionalityResult:
    """Result from dose-proportionality analysis with saturable absorption."""

    doses_mg: list[float]
    aucs: list[float]
    cmaxes: list[float]
    dose_normalized_aucs: list[float]  # AUC / dose
    power_law_exponent: float  # b in log(AUC) = a + b*log(dose)
    proportionality_classification: str  # 'linear' / 'slightly_nonlinear' / 'nonlinear'
    notes: str


def saturable_absorption_rate(
    c_lumen_mg_L: float,
    jmax_mg_per_h_per_cm2: float,
    km_mg_L: float,
    area_cm2: float = 200.0,
    passive_perm: float | None = None,
) -> float:
    """Carrier-mediated plus passive absorption rate.

    Rate = Jmax * area * C / (Km + C)  +  passive_perm * area * C

    Parameters
    ----------
    c_lumen_mg_L : float
        Drug concentration in gut lumen (mg/L).
    jmax_mg_per_h_per_cm2 : float
        Maximum carrier flux (mg/h/cm²).
    km_mg_L : float
        Michaelis constant for carrier absorption (mg/L).
    area_cm2 : float
        Absorptive surface area (cm²); default 200 cm².
    passive_perm : float or None
        Passive permeability (cm/h per cm²); default = 0.01 * Jmax/Km.

    Returns
    -------
    float
        Total absorption rate (mg/h).
    """
    if c_lumen_mg_L <= 0.0:
        return 0.0
    if passive_perm is None:
        passive_perm = 0.01 * jmax_mg_per_h_per_cm2 / km_mg_L
    carrier_rate = jmax_mg_per_h_per_cm2 * area_cm2 * c_lumen_mg_L / (km_mg_L + c_lumen_mg_L)
    passive_rate = passive_perm * area_cm2 * c_lumen_mg_L
    return carrier_rate + passive_rate


def simulate_saturable_absorption_pk(
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    jmax_mg_per_h_per_cm2: float,
    km_mg_L: float,
    passive_fraction: float = 0.1,
    gut_volume_L: float = 0.5,
    area_cm2: float = 200.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> SaturableAbsorptionResult:
    """Simulate PK with saturable carrier-mediated absorption.

    ODEs (Forward Euler):
      dA_lumen/dt  = -(sat_rate + passive_rate)
      dC_plasma/dt = absorbed_rate / vd - (cl/vd) * C_plasma

    The passive_fraction parameter scales the passive permeability:
      passive_perm = passive_fraction * Jmax / Km
    so passive contribution = passive_fraction * Jmax * area * C / Km (linear in C).

    Parameters
    ----------
    passive_fraction : float
        Scale factor for passive permeability relative to Jmax/Km; default 0.1.
    gut_volume_L : float
        Volume of gut lumen (L) for converting mass→concentration; default 0.5.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if jmax_mg_per_h_per_cm2 <= 0:
        raise ValueError("jmax_mg_per_h_per_cm2 must be > 0")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be > 0")
    if area_cm2 <= 0:
        raise ValueError("area_cm2 must be > 0")

    passive_perm = passive_fraction * jmax_mg_per_h_per_cm2 / km_mg_L  # cm/h (per cm²)

    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)
    c_plasma = np.zeros(n_steps + 1)

    a_lumen = dose_mg  # mg
    c_p = 0.0
    ke = cl_L_per_h / vd_L

    total_absorbed = 0.0  # mg

    for i in range(n_steps):
        c_lumen = a_lumen / gut_volume_L  # mg/L
        abs_rate = saturable_absorption_rate(
            c_lumen, jmax_mg_per_h_per_cm2, km_mg_L, area_cm2, passive_perm
        )  # mg/h

        # Mass absorbed this step
        mass_abs_step = min(abs_rate * dt_h, a_lumen)
        a_lumen = max(a_lumen - mass_abs_step, 0.0)
        total_absorbed += mass_abs_step

        # Plasma ODE
        actual_abs_rate = mass_abs_step / dt_h  # mg/h (clamped)
        dc_p = (actual_abs_rate / vd_L - ke * c_p) * dt_h
        c_p = max(c_p + dc_p, 0.0)
        c_plasma[i + 1] = c_p

    auc = float(np_trapz(c_plasma, times))
    cmax = float(np.max(c_plasma))
    tmax_h = float(times[int(np.argmax(c_plasma))])
    f_absorbed = total_absorbed / dose_mg

    # Saturation index: carrier fraction of total flux at Cmax
    c_lumen_at_cmax = cmax * vd_L / gut_volume_L  # rough back-estimate
    carrier_at_cmax = (
        jmax_mg_per_h_per_cm2 * area_cm2 * c_lumen_at_cmax / (km_mg_L + c_lumen_at_cmax)
    )
    passive_at_cmax = passive_perm * area_cm2 * c_lumen_at_cmax
    total_at_cmax = carrier_at_cmax + passive_at_cmax
    saturation_index = carrier_at_cmax / total_at_cmax if total_at_cmax > 0 else 0.0

    notes = (
        f"dose={dose_mg:.1f} mg; f_absorbed={f_absorbed:.3f}; "
        f"saturation_index={saturation_index:.3f}"
    )

    return SaturableAbsorptionResult(
        dose_mg=dose_mg,
        times_h=times.tolist(),
        c_plasma=c_plasma.tolist(),
        f_absorbed=f_absorbed,
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        saturation_index=saturation_index,
        notes=notes,
    )


def dose_proportionality_analysis(
    doses_mg: list[float],
    vd_L: float,
    cl_L_per_h: float,
    jmax: float,
    km: float,
    passive_fraction: float = 0.1,
    **kwargs,
) -> DoseProportionalityResult:
    """Simulate each dose and assess dose proportionality.

    Fits power model: log(AUC) = a + b * log(dose)
    - b ≈ 1.0  → linear
    - b < 1.0  → saturable absorption (less than proportional)
    - b > 1.0  → supra-proportional (rare with saturable absorption)

    Parameters
    ----------
    doses_mg : list[float]
        List of doses to evaluate.
    jmax : float
        Maximum carrier flux per unit area (mg/h/cm²).
    km : float
        Michaelis constant (mg/L).
    passive_fraction : float
        Passive permeability scale (default 0.1).
    """
    if len(doses_mg) < 2:
        raise ValueError("At least 2 doses required for proportionality analysis")
    for d in doses_mg:
        if d <= 0:
            raise ValueError("All doses must be > 0")

    aucs = []
    cmaxes = []
    for d in doses_mg:
        r = simulate_saturable_absorption_pk(
            d, vd_L, cl_L_per_h, jmax, km, passive_fraction=passive_fraction, **kwargs
        )
        aucs.append(r.auc)
        cmaxes.append(r.cmax)

    dose_arr = np.array(doses_mg, dtype=float)
    auc_arr = np.array(aucs, dtype=float)

    # Dose-normalised AUC
    dn_aucs = (auc_arr / dose_arr).tolist()

    # Power law fit: log(AUC) = a + b*log(dose)
    log_d = np.log(dose_arr)
    log_a = np.log(np.maximum(auc_arr, 1e-15))
    coeffs = np.polyfit(log_d, log_a, 1)
    b = float(coeffs[0])

    if abs(b - 1.0) < 0.10:
        classification = "linear"
    elif abs(b - 1.0) < 0.25:
        classification = "slightly_nonlinear"
    else:
        classification = "nonlinear"

    notes = f"power_law_exponent={b:.3f}; classification={classification}; doses={doses_mg}"

    return DoseProportionalityResult(
        doses_mg=list(doses_mg),
        aucs=aucs,
        cmaxes=cmaxes,
        dose_normalized_aucs=dn_aucs,
        power_law_exponent=b,
        proportionality_classification=classification,
        notes=notes,
    )


__all__ = [
    "SaturableAbsorptionResult",
    "DoseProportionalityResult",
    "saturable_absorption_rate",
    "simulate_saturable_absorption_pk",
    "dose_proportionality_analysis",
]
