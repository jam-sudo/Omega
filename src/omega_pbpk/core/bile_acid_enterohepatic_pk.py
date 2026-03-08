"""Bile acid enterohepatic circulation (EHC) model for drug PK.

Models drugs that undergo significant EHC via bile, producing multiple
peaks in the plasma concentration-time profile.
"""

from dataclasses import dataclass, field


@dataclass
class BileAcidEnterohepticResult:
    """Result of enterohepatic circulation simulation."""

    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_liver_mg_L: list = field(default_factory=list)
    a_bile_mg: list = field(default_factory=list)
    c_total_mg_L: list = field(default_factory=list)
    cmax_first_peak_mg_L: float = 0.0
    cmax_second_peak_mg_L: float = 0.0
    tmax_first_h: float = 0.0
    tmax_second_h: float = 0.0
    auc_total_mg_h_per_L: float = 0.0
    f_ehc: float = 0.0
    number_of_cycles: int = 0
    notes: str = ""


def simulate_enterohepatic_circulation(
    drug_name: str,
    dose_mg: float,
    f_ehc: float = 0.3,
    k_biliary_per_h: float = 0.5,
    bile_lag_h: float = 4.0,
    k_reabsorption_per_h: float = 0.8,
    v_plasma_L: float = 5.0,
    v_liver_L: float = 1.5,
    cl_plasma_L_per_h: float = 3.0,
    route: str = "oral",
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> BileAcidEnterohepticResult:
    """Simulate enterohepatic circulation of a drug.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    f_ehc:
        Fraction of drug undergoing EHC per cycle (0-1).
    k_biliary_per_h:
        Biliary excretion rate constant (1/h).
    bile_lag_h:
        Bile duct lag time before intestinal release (h).
    k_reabsorption_per_h:
        Intestinal reabsorption rate constant (1/h).
    v_plasma_L:
        Plasma volume (L).
    v_liver_L:
        Liver volume (L) — reserved for future use.
    cl_plasma_L_per_h:
        Plasma clearance (L/h).
    route:
        Dosing route: "oral" or "iv".
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    BileAcidEnterohepticResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0.0 <= f_ehc <= 1.0):
        raise ValueError(f"f_ehc must be in [0, 1], got {f_ehc}")
    if bile_lag_h < 0:
        raise ValueError(f"bile_lag_h must be >= 0, got {bile_lag_h}")
    if v_plasma_L <= 0:
        raise ValueError(f"v_plasma_L must be > 0, got {v_plasma_L}")
    if cl_plasma_L_per_h <= 0:
        raise ValueError(f"cl_plasma_L_per_h must be > 0, got {cl_plasma_L_per_h}")
    if route not in {"oral", "iv"}:
        raise ValueError(f"route must be 'oral' or 'iv', got {route!r}")

    # Derived rate constants
    ke = cl_plasma_L_per_h / v_plasma_L
    ka = 1.0  # oral absorption rate constant (1/h)

    # Bile release: exponential lag approximation
    # Avoid division by zero if bile_lag_h == 0
    k_release = 1.0 / bile_lag_h if bile_lag_h > 0 else 10.0

    # Initial conditions
    a_gut_oral = dose_mg if route == "oral" else 0.0
    c_plasma = dose_mg / v_plasma_L if route == "iv" else 0.0
    a_bile = 0.0

    n_steps = int(t_end_h / dt_h) + 1
    times = []
    c_plasma_list = []
    c_liver_list = []
    a_bile_list = []
    c_total_list = []

    for step in range(n_steps):
        t = step * dt_h

        # Record state
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_liver_list.append(2.0 * c_plasma)
        a_bile_list.append(a_bile)
        # total body burden: plasma + bile (converted to mg/L equivalent using v_plasma)
        c_total_list.append(c_plasma + a_bile / v_plasma_L)

        # ODE right-hand sides
        # Gut: oral absorption source; bile-reabsorbed drug re-enters gut
        d_a_gut = -ka * a_gut_oral

        # Bile: receives fraction of plasma drug via biliary excretion; releases with lag
        d_a_bile = f_ehc * k_biliary_per_h * c_plasma * v_plasma_L - k_release * a_bile

        # Plasma: absorbs from gut, loses via elimination and biliary excretion,
        # gains from bile release (reabsorbed)
        d_c_plasma = (
            ka * a_gut_oral / v_plasma_L
            - ke * c_plasma
            - f_ehc * k_biliary_per_h * c_plasma
            + k_release * a_bile / v_plasma_L
        )

        # Forward Euler update
        a_gut_oral = max(0.0, a_gut_oral + d_a_gut * dt_h)
        a_bile = max(0.0, a_bile + d_a_bile * dt_h)
        c_plasma = max(0.0, c_plasma + d_c_plasma * dt_h)

    # --- Peak detection ---
    # Find first peak: highest concentration before it starts declining
    first_peak_val = 0.0
    first_peak_idx = 0

    second_peak_val = 0.0
    second_peak_idx = 0

    # Scan through time series to find local maxima
    # First peak: first local maximum after initial rise
    # Second peak: any subsequent local maximum (EHC cycle)
    n = len(c_plasma_list)

    local_maxima = []
    for i in range(1, n - 1):
        if c_plasma_list[i] > c_plasma_list[i - 1] and c_plasma_list[i] >= c_plasma_list[i + 1]:
            local_maxima.append((i, c_plasma_list[i]))

    if local_maxima:
        first_peak_idx, first_peak_val = local_maxima[0]

        if len(local_maxima) >= 2:
            second_peak_idx, second_peak_val = local_maxima[1]
        else:
            # No distinct second peak found; use the global max after first tmax as fallback
            search_start = first_peak_idx + 1
            best_val = 0.0
            best_idx = first_peak_idx + 1
            for i in range(search_start, n):
                if c_plasma_list[i] > best_val:
                    best_val = c_plasma_list[i]
                    best_idx = i
            second_peak_val = best_val
            second_peak_idx = best_idx
    else:
        # No local maximum found; use global max
        first_peak_val = max(c_plasma_list)
        first_peak_idx = c_plasma_list.index(first_peak_val)
        second_peak_val = 0.0
        second_peak_idx = min(first_peak_idx + 1, n - 1)

    tmax_first = times[first_peak_idx]
    tmax_second = times[second_peak_idx]

    # Ensure tmax_second > tmax_first
    if tmax_second <= tmax_first and second_peak_idx < n - 1:
        tmax_second = times[min(second_peak_idx + 1, n - 1)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, n):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * dt_h

    # EHC cycle count
    number_of_cycles = max(1, int(t_end_h / bile_lag_h)) if bile_lag_h > 0 else 1

    notes = (
        f"EHC simulation: route={route}, f_ehc={f_ehc:.2f}, "
        f"bile_lag={bile_lag_h}h, k_biliary={k_biliary_per_h}/h"
    )

    return BileAcidEnterohepticResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_liver_mg_L=c_liver_list,
        a_bile_mg=a_bile_list,
        c_total_mg_L=c_total_list,
        cmax_first_peak_mg_L=first_peak_val,
        cmax_second_peak_mg_L=second_peak_val,
        tmax_first_h=tmax_first,
        tmax_second_h=tmax_second,
        auc_total_mg_h_per_L=auc,
        f_ehc=f_ehc,
        number_of_cycles=number_of_cycles,
        notes=notes,
    )
