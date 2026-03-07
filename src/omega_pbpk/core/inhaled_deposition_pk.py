"""Phase 751 — Inhaled Drug Deposition PK.

Models aerosol particle deposition in lung regions based on MMAD,
with regional absorption kinetics feeding into a 1-compartment plasma model.
Reference: ICRP 1994 simplified deposition model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class InhaledDepositionResult:
    """Result of inhaled drug deposition PK simulation."""

    drug_name: str
    dose_mg: float
    mmad_um: float
    times_h: list
    c_plasma_mg_L: list
    f_alveolar: float
    f_tracheobronchial: float
    f_oropharynx: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    systemic_f: float
    fine_particle_dose_mg: float
    t_half_h: float
    notes: str


def _compute_deposition_fractions(mmad_um: float) -> tuple[float, float, float]:
    """Compute regional deposition fractions from MMAD (ICRP 1994 simplified).

    Returns (f_oropharynx, f_tracheobronchial, f_alveolar).
    """
    # Oropharyngeal deposition increases with particle size
    f_oro = 0.15 + 0.15 * (mmad_um - 1.0) / 9.0
    f_oro = max(0.0, min(1.0, f_oro))

    # Tracheobronchial — Gaussian peak near MMAD=1 µm
    f_tb = max(0.05, 0.20 * math.exp(-((mmad_um - 1.0) ** 2) / 8.0))
    f_tb = max(0.0, min(1.0, f_tb))

    # Alveolar: remainder minus ~10% mucociliary loss, clamped ≥ 0
    f_alv = max(0.0, 1.0 - f_oro - f_tb - 0.10)
    f_alv = max(0.0, min(1.0, f_alv))

    # Normalize so total ≤ 1
    total = f_oro + f_tb + f_alv
    if total > 1.0:
        scale = 1.0 / total
        f_oro *= scale
        f_tb *= scale
        f_alv *= scale

    return f_oro, f_tb, f_alv


def simulate_inhaled_deposition_pk(
    drug_name: str,
    dose_mg: float,
    mmad_um: float = 2.0,
    ka_alveolar_per_h: float = 3.0,
    ka_oral_per_h: float = 0.5,
    f_oral: float = 0.3,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> InhaledDepositionResult:
    """Simulate inhaled drug deposition PK.

    Deposition fractions are computed from MMAD via ICRP 1994 simplified model.
    Alveolar fraction → fast systemic absorption (ka_alveolar_per_h).
    Oropharyngeal + tracheobronchial fraction → swallowed → oral absorption
    with bioavailability f_oral.

    Parameters
    ----------
    drug_name: str
        Drug identifier.
    dose_mg: float
        Emitted (inhaled) dose in mg.
    mmad_um: float
        Mass median aerodynamic diameter in µm.
    ka_alveolar_per_h: float
        First-order alveolar absorption rate constant (1/h).
    ka_oral_per_h: float
        First-order oral/GI absorption rate constant for swallowed fraction (1/h).
    f_oral: float
        Oral bioavailability of the swallowed fraction (0–1).
    cl_L_per_h: float
        Systemic clearance (L/h).
    vd_L: float
        Volume of distribution (L).
    t_end_h: float
        Simulation end time (h).
    dt_h: float
        Forward Euler time step (h).

    Returns
    -------
    InhaledDepositionResult
    """
    # --- Validation ---
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if mmad_um <= 0.0:
        raise ValueError(f"mmad_um must be > 0, got {mmad_um}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    # --- Deposition fractions ---
    f_oro, f_tb, f_alv = _compute_deposition_fractions(mmad_um)

    # --- Initial conditions ---
    # Alveolar depot (mg)
    A_alv = dose_mg * f_alv
    # Swallowed = oropharyngeal + tracheobronchial
    A_oral = dose_mg * (f_oro + f_tb)
    # Plasma concentration (mg/L)
    C_plasma = 0.0

    ke = cl_L_per_h / vd_L

    # --- Forward Euler ODE integration ---
    n_steps = max(1, int(math.ceil(t_end_h / dt_h)))
    times: list[float] = []
    c_plasma_vals: list[float] = []

    t = 0.0
    for _ in range(n_steps + 1):
        times.append(t)
        c_plasma_vals.append(C_plasma)

        # Derivatives
        dA_alv = -ka_alveolar_per_h * A_alv
        dA_oral = -ka_oral_per_h * A_oral
        # Input to plasma: alveolar (100% bioavailable) + oral × f_oral
        input_rate = (ka_alveolar_per_h * A_alv + ka_oral_per_h * A_oral * f_oral) / vd_L
        dC = input_rate - ke * C_plasma

        # Update state
        A_alv = max(0.0, A_alv + dA_alv * dt_h)
        A_oral = max(0.0, A_oral + dA_oral * dt_h)
        C_plasma = max(0.0, C_plasma + dC * dt_h)
        t += dt_h

    # --- Derived PK metrics ---
    cmax = max(c_plasma_vals)
    tmax = times[c_plasma_vals.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (c_plasma_vals[i - 1] + c_plasma_vals[i]) * (times[i] - times[i - 1])

    t_half = math.log(2.0) / ke if ke > 0.0 else float("inf")

    systemic_f = f_alv + (f_oro + f_tb) * f_oral
    fine_particle_dose_mg = dose_mg * f_alv

    notes = (
        f"MMAD={mmad_um:.1f}µm: f_alveolar={f_alv:.3f}, "
        f"f_tracheobronchial={f_tb:.3f}, f_oropharynx={f_oro:.3f}. "
        f"Systemic F={systemic_f:.3f}. "
        f"Cmax={cmax:.4f} mg/L at t={tmax:.2f}h, AUC={auc:.4f} mg·h/L. "
        f"t½={t_half:.2f}h."
    )

    return InhaledDepositionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mmad_um=mmad_um,
        times_h=times,
        c_plasma_mg_L=c_plasma_vals,
        f_alveolar=f_alv,
        f_tracheobronchial=f_tb,
        f_oropharynx=f_oro,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        systemic_f=systemic_f,
        fine_particle_dose_mg=fine_particle_dose_mg,
        t_half_h=t_half,
        notes=notes,
    )


def compare_particle_sizes(
    drug_name: str,
    dose_mg: float,
    mmad_values: list[float],
    ka_alveolar_per_h: float = 3.0,
    ka_oral_per_h: float = 0.5,
    f_oral: float = 0.3,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> list[InhaledDepositionResult]:
    """Simulate inhaled PK for multiple particle sizes (MMAD values).

    Parameters
    ----------
    drug_name: str
        Drug identifier.
    dose_mg: float
        Emitted dose in mg (same for all MMAD).
    mmad_values: list[float]
        List of MMAD values in µm to compare.
    (remaining kwargs same as simulate_inhaled_deposition_pk)

    Returns
    -------
    list[InhaledDepositionResult], one per MMAD value.
    """
    results = []
    for mmad in mmad_values:
        res = simulate_inhaled_deposition_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            mmad_um=mmad,
            ka_alveolar_per_h=ka_alveolar_per_h,
            ka_oral_per_h=ka_oral_per_h,
            f_oral=f_oral,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(res)
    return results
