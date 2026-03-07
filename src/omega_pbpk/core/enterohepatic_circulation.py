"""
Phase 1078 -- Drug Biliary Excretion and Enterohepatic Circulation

Models drug with biliary excretion and enterohepatic recirculation (EHC),
producing the characteristic secondary peak in plasma concentration-time profiles.

Gallbladder emptying is modeled as periodic bolus releases (every gall_cycle_h hours),
which is physiologically realistic and produces the characteristic secondary peaks.
"""

from __future__ import annotations

from dataclasses import dataclass


def _trapz(times: list[float], values: list[float]) -> float:
    """Trapezoidal integration."""
    area = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        area += 0.5 * (values[i] + values[i - 1]) * dt
    return area


def _validate_inputs(
    dose_mg: float,
    fraction_biliary: float,
    cl_total: float,
    vd: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 <= fraction_biliary <= 1):
        raise ValueError(f"fraction_biliary must be in [0, 1], got {fraction_biliary}")
    if cl_total <= 0:
        raise ValueError(f"cl_total_L_per_h must be > 0, got {cl_total}")
    if vd <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd}")


@dataclass
class EHCResult:
    drug_name: str
    dose_mg: float
    fraction_biliary: float
    times_h: list
    c_plasma_mg_L: list
    a_bile_mg: list
    a_gut_mg: list
    cmax_first_peak: float
    tmax_first_peak_h: float
    cmax_second_peak: float
    tmax_second_peak_h: float
    auc: float
    f_fecal_eliminated: float
    has_secondary_peak: bool
    notes: str


def simulate_ehc(
    drug_name: str,
    dose_mg: float,
    fraction_biliary: float = 0.3,
    cl_total_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    k_gall_per_h: float = 0.3,
    ka_ehc_per_h: float = 0.5,
    k_fecal_per_h: float = 0.05,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
    gall_cycle_h: float = 4.0,
    gall_empty_fraction: float = 0.8,
) -> EHCResult:
    """
    Simulate drug biliary excretion and enterohepatic circulation via Forward Euler ODE.

    Compartments:
      C_plasma (mg/L): systemic plasma
      A_bile (mg):     bile duct accumulation pool
      A_gut (mg):      gut lumen (available for reabsorption)

    ODEs between gallbladder emptying events:
      ke         = (1 - fraction_biliary) * cl_total / vd
      k_bile_rate = fraction_biliary * cl_total / vd   [on C_plasma, units: 1/h]

      dC_plasma/dt = -(ke + k_bile_rate) * C_plasma + ka_ehc * A_gut / vd
      dA_bile/dt   =  k_bile_rate * C_plasma * vd       (biliary secretion into bile pool)
      dA_gut/dt    =  - (ka_ehc + k_fecal) * A_gut      (gut absorption / fecal loss)

    Gallbladder empties periodically (every gall_cycle_h hours):
      bolus of gall_empty_fraction * A_bile is added to A_gut instantaneously.
    """
    _validate_inputs(dose_mg, fraction_biliary, cl_total_L_per_h, vd_L)

    # Rate constants
    ke = (1.0 - fraction_biliary) * cl_total_L_per_h / vd_L
    k_bile_rate = fraction_biliary * cl_total_L_per_h / vd_L  # per hour (on C_plasma)

    # Initial conditions
    c_plasma = dose_mg / vd_L
    a_bile = 0.0
    a_gut = 0.0

    n_steps = int(round(t_end_h / dt_h)) + 1
    times: list[float] = []
    c_plasma_list: list[float] = []
    a_bile_list: list[float] = []
    a_gut_list: list[float] = []

    fecal_mass = 0.0
    last_gall_empty_t = 0.0  # track last gallbladder emptying time

    for i in range(n_steps):
        t = i * dt_h

        # Gallbladder periodic emptying: bolus release into gut
        if fraction_biliary > 0 and gall_cycle_h > 0:
            if t > 0 and (t - last_gall_empty_t) >= gall_cycle_h - 1e-9:
                released = gall_empty_fraction * a_bile
                a_bile = a_bile - released
                a_gut = a_gut + released
                last_gall_empty_t = t

        times.append(t)
        c_plasma_list.append(c_plasma)
        a_bile_list.append(a_bile)
        a_gut_list.append(a_gut)

        # Derivatives
        dc_plasma = -(ke + k_bile_rate) * c_plasma + ka_ehc_per_h * a_gut / vd_L
        da_bile = k_bile_rate * c_plasma * vd_L
        da_gut = -(ka_ehc_per_h + k_fecal_per_h) * a_gut

        # Track fecal elimination
        fecal_rate = k_fecal_per_h * a_gut
        fecal_mass += fecal_rate * dt_h

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        a_bile = max(0.0, a_bile + da_bile * dt_h)
        a_gut = max(0.0, a_gut + da_gut * dt_h)

    # AUC via trapezoidal rule
    auc = _trapz(times, c_plasma_list)

    # First peak: maximum of c_plasma (for IV bolus this is at t=0)
    cmax_first = max(c_plasma_list)
    tmax_first_idx = c_plasma_list.index(cmax_first)
    tmax_first = times[tmax_first_idx]

    # Secondary peak detection: local maximum after tmax_first_idx
    # Must be > 10% of cmax_first
    threshold = 0.1 * cmax_first
    cmax_second = 0.0
    tmax_second = 0.0

    search_start = tmax_first_idx + 1
    for i in range(search_start + 1, len(c_plasma_list) - 1):
        v_prev = c_plasma_list[i - 1]
        v_curr = c_plasma_list[i]
        v_next = c_plasma_list[i + 1]
        if v_curr >= v_prev and v_curr >= v_next and v_curr > threshold:
            if v_curr > cmax_second:
                cmax_second = v_curr
                tmax_second = times[i]

    has_secondary_peak = cmax_second > threshold

    # Fraction eliminated via feces
    f_fecal = min(1.0, fecal_mass / dose_mg)

    notes = (
        f"EHC simulation for {drug_name}. "
        f"f_biliary={fraction_biliary:.2f}. "
        f"Gallbladder empties every {gall_cycle_h:.1f}h. "
        f"Secondary peak: {'detected' if has_secondary_peak else 'not detected'}. "
        f"AUC={auc:.2f} mg*h/L."
    )

    return EHCResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fraction_biliary=fraction_biliary,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        a_bile_mg=a_bile_list,
        a_gut_mg=a_gut_list,
        cmax_first_peak=cmax_first,
        tmax_first_peak_h=tmax_first,
        cmax_second_peak=cmax_second,
        tmax_second_peak_h=tmax_second,
        auc=auc,
        f_fecal_eliminated=f_fecal,
        has_secondary_peak=has_secondary_peak,
        notes=notes,
    )


def compare_biliary_fractions(
    drug_name: str,
    dose_mg: float,
    fractions: list[float],
) -> list[EHCResult]:
    """Run EHC simulation for multiple biliary fractions, sorted by cmax_second_peak descending."""
    results = []
    for f in fractions:
        result = simulate_ehc(
            drug_name=drug_name,
            dose_mg=dose_mg,
            fraction_biliary=f,
        )
        results.append(result)
    results.sort(key=lambda r: r.cmax_second_peak, reverse=True)
    return results
