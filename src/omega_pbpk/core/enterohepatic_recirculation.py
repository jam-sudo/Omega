"""Phase 606 — Enterohepatic Recirculation (EHC) Model.

Models hepatic bile secretion → gallbladder accumulation → intestinal
reabsorption producing secondary plasma peaks.
"""

from dataclasses import dataclass


@dataclass
class EHCResult:
    """Result of enterohepatic recirculation simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_bile_mg: list  # drug in bile/gallbladder
    c_gut_ehc_mg: list  # drug in gut for reabsorption (EHC compartment)
    cmax_mg_L: float
    tmax_h: float
    secondary_peak_time_h: float | None  # time of secondary plasma peak
    secondary_peak_conc_mg_L: float | None
    auc_mg_h_L: float
    f_recycled_pct: float  # fraction of dose recycled via EHC
    notes: str


def detect_secondary_peak(
    times_h: list,
    c_plasma: list,
    primary_peak_time_h: float,
) -> tuple:
    """Find secondary peak after primary peak.

    A secondary peak exists if there's a local maximum after primary peak with
    conc > 10% of primary Cmax and a real dip (minimum before candidate < 90%
    of candidate value).

    Args:
        times_h: Time points (hours)
        c_plasma: Plasma concentration at each time point
        primary_peak_time_h: Time of primary peak (hours)

    Returns:
        (secondary_peak_time, secondary_peak_conc) or (None, None) if not found
    """
    if not c_plasma or not times_h:
        return (None, None)

    primary_cmax = max(c_plasma)

    # Find index after primary peak
    start_idx = 0
    for i, t in enumerate(times_h):
        if t > primary_peak_time_h:
            start_idx = i
            break

    if start_idx == 0 or start_idx >= len(c_plasma) - 2:
        return (None, None)

    n = len(c_plasma)

    # Scan for local maxima after primary peak
    best_time = None
    best_conc = None

    for i in range(start_idx + 1, n - 1):
        c_curr = c_plasma[i]
        c_prev = c_plasma[i - 1]
        c_next = c_plasma[i + 1]

        # Local maximum check
        if c_curr <= c_prev or c_curr <= c_next:
            continue

        # Must be > 10% of primary Cmax
        if c_curr < 0.10 * primary_cmax:
            continue

        # Check for real dip: minimum between primary peak and candidate < 90% of candidate
        primary_idx = 0
        for j, t in enumerate(times_h):
            if t >= primary_peak_time_h:
                primary_idx = j
                break

        # Find minimum between primary peak and this candidate
        min_val = min(c_plasma[primary_idx : i + 1])
        if min_val >= 0.90 * c_curr:
            # No real dip — not a genuine secondary peak
            continue

        # Take the most prominent one
        if best_conc is None or c_curr > best_conc:
            best_time = times_h[i]
            best_conc = c_curr

    return (best_time, best_conc)


def simulate_ehc(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    cl_biliary_L_per_h: float = 0.5,
    kgb_per_h: float = 0.1,
    f_reabs: float = 0.6,
    lag_biliary_h: float = 4.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> EHCResult:
    """Simulate enterohepatic recirculation (EHC) PK model.

    5-compartment ODE system:
      - A_gut: drug in gut absorption compartment
      - C_plasma: plasma concentration (mg/L)
      - A_bile: drug in bile/gallbladder (accumulated)
      - A_gut_ehc: drug released from bile into gut (for reabsorption)
      - A_eliminated: total drug eliminated (tracking)

    Args:
        drug_name: Name of the drug
        dose_mg: Oral dose (mg)
        cl_L_per_h: Total systemic clearance (L/h)
        vd_L: Volume of distribution (L)
        ka_per_h: Oral absorption rate constant (1/h)
        f_oral: Oral bioavailability fraction (0-1]
        cl_biliary_L_per_h: Biliary clearance rate (L/h)
        kgb_per_h: Gallbladder emptying rate constant (1/h)
        f_reabs: Fraction of bile drug reabsorbed from gut
        lag_biliary_h: Lag time before gallbladder empties (h)
        t_end_h: Simulation end time (h)
        dt_h: ODE integration step size (h)

    Returns:
        EHCResult with full PK profiles and secondary peak detection

    Raises:
        ValueError: If parameters are out of valid range
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0.0 <= f_reabs <= 1.0):
        raise ValueError(f"f_reabs must be in [0, 1], got {f_reabs}")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")

    # Derived parameters
    ka_reabs = ka_per_h * 0.5  # slower reabsorption rate

    # Initial conditions
    a_gut = dose_mg * f_oral  # drug in gut (mg)
    c_plasma = 0.0  # plasma concentration (mg/L)
    a_bile = 0.0  # drug in bile/gallbladder (mg)
    a_gut_ehc = 0.0  # drug in gut from biliary emptying (mg)
    a_eliminated = 0.0  # tracking eliminated drug (mg)

    # Output arrays
    n_steps = int(t_end_h / dt_h) + 1
    times_h_list = []
    c_plasma_list = []
    c_bile_list = []
    c_gut_ehc_list = []

    # Tracking for f_recycled calculation
    total_reabsorbed = 0.0  # integral of f_reabs * ka_reabs * A_gut_ehc

    t = 0.0
    for _step in range(n_steps):
        # Record state
        times_h_list.append(t)
        c_plasma_list.append(max(0.0, c_plasma))
        c_bile_list.append(max(0.0, a_bile))
        c_gut_ehc_list.append(max(0.0, a_gut_ehc))

        # Gallbladder emptying rate (active only after lag period)
        k_empty = kgb_per_h if t >= lag_biliary_h else 0.0

        # ODE system (Forward Euler)
        # dA_gut/dt = -ka * A_gut
        d_a_gut = -ka_per_h * a_gut

        # dC_plasma/dt = ka*A_gut/Vd + f_reabs*ka_reabs*A_gut_ehc/Vd
        #                - (cl/Vd + cl_biliary/Vd) * C_plasma
        reabs_flux = f_reabs * ka_reabs * a_gut_ehc
        d_c_plasma = (
            ka_per_h * a_gut / vd_L
            + reabs_flux / vd_L
            - (cl_L_per_h / vd_L + cl_biliary_L_per_h / vd_L) * c_plasma
        )

        # dA_bile/dt = cl_biliary * C_plasma - k_empty * A_bile
        d_a_bile = cl_biliary_L_per_h * c_plasma - k_empty * a_bile

        # dA_gut_ehc/dt = k_empty * A_bile - ka_reabs * A_gut_ehc
        d_a_gut_ehc = k_empty * a_bile - ka_reabs * a_gut_ehc

        # Tracking eliminated drug
        d_a_eliminated = cl_L_per_h * c_plasma

        # Accumulate reabsorption for f_recycled calculation
        total_reabsorbed += reabs_flux * dt_h

        # Update states (Forward Euler)
        a_gut = max(0.0, a_gut + d_a_gut * dt_h)
        c_plasma = max(0.0, c_plasma + d_c_plasma * dt_h)
        a_bile = max(0.0, a_bile + d_a_bile * dt_h)
        a_gut_ehc = max(0.0, a_gut_ehc + d_a_gut_ehc * dt_h)
        a_eliminated = a_eliminated + d_a_eliminated * dt_h

        t += dt_h

    # Compute summary statistics
    cmax_mg_L = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_mg_L)
    tmax_h = times_h_list[tmax_idx]

    # AUC via trapezoidal rule
    auc = 0.0
    for i in range(1, len(times_h_list)):
        dt = times_h_list[i] - times_h_list[i - 1]
        auc += 0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * dt

    # f_recycled = fraction of administered dose that was reabsorbed via EHC
    dose_absorbed = dose_mg * f_oral
    f_recycled_pct = 100.0 * total_reabsorbed / dose_absorbed if dose_absorbed > 0 else 0.0
    f_recycled_pct = min(100.0, max(0.0, f_recycled_pct))

    # Detect secondary peak
    secondary_time, secondary_conc = detect_secondary_peak(times_h_list, c_plasma_list, tmax_h)

    # Build notes
    notes_parts = [
        f"EHC simulation for '{drug_name}'.",
        f"Biliary clearance: {cl_biliary_L_per_h:.2f} L/h.",
        f"Gallbladder emptying starts at t={lag_biliary_h:.1f} h.",
        f"Reabsorption fraction: {f_reabs:.0%}.",
    ]
    if secondary_time is not None:
        notes_parts.append(
            f"Secondary peak detected at t={secondary_time:.1f} h "
            f"(Cmax2={secondary_conc:.3f} mg/L)."
        )
    else:
        notes_parts.append("No secondary peak detected.")

    return EHCResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h_list,
        c_plasma_mg_L=c_plasma_list,
        c_bile_mg=c_bile_list,
        c_gut_ehc_mg=c_gut_ehc_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        secondary_peak_time_h=secondary_time,
        secondary_peak_conc_mg_L=secondary_conc,
        auc_mg_h_L=auc,
        f_recycled_pct=f_recycled_pct,
        notes=" ".join(notes_parts),
    )
