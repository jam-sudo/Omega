"""Enterohepatic circulation (EHC) PK model — Phase 256.

Simulates drug recycling through bile secretion and intestinal reabsorption.
EHC causes characteristic secondary peaks in plasma PK profiles and
significantly extends apparent half-life.

References
----------
- Roberts MS et al., Clin Pharmacokinet. 2002;41(10):751-90
- Tse FLS & Jaffe JM, Presystemic Elimination. 1991
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EHCResult:
    """Result from enterohepatic circulation simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float]
    c_plasma_mg_L: list[float]  # Plasma concentration
    a_bile_mg: list[float]  # Drug in bile reservoir
    a_gi_ehc_mg: list[float]  # Drug pending GI reabsorption from bile
    cmax: float
    tmax_h: float
    auc: float
    secondary_peak: float  # 2nd peak height (0 if no EHC)
    secondary_peak_time_h: float
    n_secondary_peaks: int
    t_half_apparent_h: float  # Apparent t½ with EHC
    notes: str


def simulate_ehc(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_biliary: float = 0.2,
    k_bile_secr_per_h: float = 0.3,
    k_ehc_abs_per_h: float = 1.0,
    ehc_cycle_delay_h: float = 4.0,
    route: str = "iv",
    ka_per_h: float = 1.5,
    f_oral: float = 0.8,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> EHCResult:
    """Simulate PK with enterohepatic circulation.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg). Must be > 0.
    cl_L_per_h : Systemic clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    f_biliary : Fraction of CL that is biliary. Must be in [0, 1).
    k_bile_secr_per_h : Rate of bile secretion from plasma (h^-1).
    k_ehc_abs_per_h : Rate of GI reabsorption after bile dumping (h^-1).
    ehc_cycle_delay_h : Delay before bile dumps into GI (h).
    route : 'iv' or 'oral'.
    ka_per_h : Oral absorption rate constant (h^-1).
    f_oral : Oral bioavailability (0, 1].
    t_end_h : Simulation end time (h).
    dt_h : Time step (h).

    Returns
    -------
    EHCResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 <= f_biliary < 1):
        raise ValueError("f_biliary must be in [0, 1)")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")

    ke = cl_L_per_h / vd_L

    n_steps = int(t_end_h / dt_h) + 1
    t = [i * dt_h for i in range(n_steps)]

    c_plasma = [0.0] * n_steps
    a_bile = [0.0] * n_steps  # Drug accumulating in bile
    a_gi = [0.0] * n_steps  # Drug in GI awaiting reabsorption
    a_gut = 0.0  # oral absorption depot

    if route == "iv":
        c_plasma[0] = dose_mg / vd_L
    else:
        a_gut = dose_mg * f_oral

    # Bile dump schedule: every ehc_cycle_delay_h
    bile_dump_interval_steps = max(1, int(ehc_cycle_delay_h / dt_h))

    for i in range(1, n_steps):
        cp = c_plasma[i - 1]
        ab = a_bile[i - 1]
        ag = a_gi[i - 1]

        # Oral absorption
        if route == "oral":
            abs_rate = ka_per_h * a_gut
            a_gut = max(0.0, a_gut - abs_rate * dt_h)
        else:
            abs_rate = 0.0

        # Bile secretion from plasma
        bile_secr = k_bile_secr_per_h * f_biliary * cp * vd_L  # mg/h

        # EHC reabsorption from GI
        ehc_reabs = k_ehc_abs_per_h * ag  # mg/h

        # Plasma
        elim = ke * cp * vd_L  # mg/h (total elimination)
        dc = (abs_rate + ehc_reabs - elim - bile_secr) / vd_L * dt_h
        c_plasma[i] = max(0.0, cp + dc)

        # Bile accumulates, then dumps periodically
        da_bile = bile_secr * dt_h
        a_bile[i] = max(0.0, ab + da_bile)

        # Bile dump to GI
        if i % bile_dump_interval_steps == 0:
            dump_amount = a_bile[i]
            a_bile[i] = 0.0
            a_gi[i] = max(0.0, ag - ehc_reabs * dt_h + dump_amount)
        else:
            a_gi[i] = max(0.0, ag - ehc_reabs * dt_h)

    # Summary metrics
    cmax = max(c_plasma)
    tmax = t[c_plasma.index(cmax)]

    # AUC via trapezoidal rule
    auc = sum(0.5 * (c_plasma[i - 1] + c_plasma[i]) * dt_h for i in range(1, n_steps))

    # Detect secondary peaks: local maxima after first peak
    first_peak_idx = c_plasma.index(cmax)
    secondary_peaks = []
    for i in range(first_peak_idx + 2, n_steps - 1):
        if c_plasma[i] > c_plasma[i - 1] and c_plasma[i] > c_plasma[i + 1]:
            secondary_peaks.append((t[i], c_plasma[i]))

    secondary_peak = secondary_peaks[0][1] if secondary_peaks else 0.0
    secondary_peak_time = secondary_peaks[0][0] if secondary_peaks else 0.0
    n_secondary = len(secondary_peaks)

    # Apparent t½: time for Cmax to halve in the terminal phase
    half_cmax = cmax * 0.5
    t_half = t_end_h  # default
    for i in range(first_peak_idx, n_steps):
        if c_plasma[i] <= half_cmax:
            t_half = t[i] - tmax
            break

    notes = (
        f"{drug_name} EHC: f_biliary={f_biliary:.2f}, cycle={ehc_cycle_delay_h:.1f}h. "
        f"Cmax={cmax:.4f} mg/L, secondary peaks={n_secondary}. "
        f"AUC={auc:.2f} mg*h/L, apparent t\u00bd={t_half:.1f}h."
    )

    return EHCResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=t,
        c_plasma_mg_L=c_plasma,
        a_bile_mg=a_bile,
        a_gi_ehc_mg=a_gi,
        cmax=cmax,
        tmax_h=tmax,
        auc=auc,
        secondary_peak=secondary_peak,
        secondary_peak_time_h=secondary_peak_time,
        n_secondary_peaks=n_secondary,
        t_half_apparent_h=t_half,
        notes=notes,
    )


__all__ = ["EHCResult", "simulate_ehc"]
