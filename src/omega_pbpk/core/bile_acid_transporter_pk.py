"""Phase 477 — Bile Acid Transporter PK (Enterohepatic Recirculation).

Models drug enterohepatic recirculation (EHC) via bile acid transporter-mediated
biliary excretion: BSEP/MRP2 exports drug into bile, gallbladder stores it,
meal-triggered release back to intestine creates secondary plasma peak.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fraction of bile pool emptied per meal trigger
_MEAL_BILE_FRACTION = 0.8


@dataclass
class BileAcidTransporterResult:
    """Result of bile acid transporter PK simulation."""

    drug_name: str
    dose_mg: float
    f_biliary: float
    times_h: list
    c_systemic_mg_L: list
    cmax_primary_mg_L: float
    tmax_primary_h: float
    cmax_secondary_mg_L: float
    tmax_secondary_h: float
    ehc_detected: bool
    auc_total_mg_h_per_L: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def _find_peaks(values: list) -> list:
    """Return indices of local maxima in values list."""
    peaks = []
    n = len(values)
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            peaks.append(i)
    return peaks


def simulate_bile_acid_transporter_pk(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float = 1.0,
    f_oral: float = 0.7,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 50.0,
    f_biliary: float = 0.3,
    k_bile_release_per_h: float = 0.5,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
    meal_times_h: list | None = None,
) -> BileAcidTransporterResult:
    """Simulate drug PK with enterohepatic recirculation via bile acid transporters.

    Compartments:
      M_gi   : GI tract (mg) — absorption site
      C_sys  : Systemic concentration (mg/L)
      A_bile : Bile pool (mg) — gallbladder storage

    ODEs (Forward Euler):
      dM_gi/dt  = -ka * M_gi + k_bile_release * A_bile
      dC_sys/dt = (ka * M_gi * f_oral / vd_sys)
                  - (cl_sys / vd_sys) * C_sys
                  - (f_biliary * cl_sys / vd_sys) * C_sys
      dA_bile/dt = f_biliary * cl_sys * C_sys - k_bile_release * A_bile

    Meal events (meal_times_h): instantaneous release of 80% of bile pool to GI,
    simulating meal-triggered gallbladder contraction.  Defaults to [4.0] h.
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if not (0.0 <= f_biliary <= 1.0):
        raise ValueError("f_biliary must be in [0, 1]")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")

    if meal_times_h is None:
        meal_times_h = [4.0]

    # convert meal times to set of step indices for fast lookup
    meal_steps: set = set()
    for mt in meal_times_h:
        step_idx = int(round(mt / dt_h))
        if 0 < step_idx <= int(t_end_h / dt_h):
            meal_steps.add(step_idx)

    # --- initial conditions ---
    m_gi = dose_mg
    c_sys = 0.0
    a_bile = 0.0

    n_steps = int(t_end_h / dt_h) + 1
    times: list = []
    c_systemic: list = []

    for step in range(n_steps):
        t = step * dt_h
        times.append(t)
        c_systemic.append(c_sys)

        # meal-triggered bile bolus: empty bile into GI before ODE step
        if step in meal_steps and f_biliary > 0.0:
            bile_bolus = _MEAL_BILE_FRACTION * a_bile
            m_gi += bile_bolus
            a_bile -= bile_bolus

        # derivatives
        dm_gi = -ka_per_h * m_gi + k_bile_release_per_h * a_bile
        dc_sys = (
            (ka_per_h * m_gi * f_oral / vd_sys_L)
            - (cl_sys_L_per_h / vd_sys_L) * c_sys
            - (f_biliary * cl_sys_L_per_h / vd_sys_L) * c_sys
        )
        da_bile = f_biliary * cl_sys_L_per_h * c_sys - k_bile_release_per_h * a_bile

        # forward Euler update
        m_gi = max(0.0, m_gi + dm_gi * dt_h)
        c_sys = max(0.0, c_sys + dc_sys * dt_h)
        a_bile = max(0.0, a_bile + da_bile * dt_h)

    # --- find peaks ---
    peaks = _find_peaks(c_systemic)

    cmax_primary = 0.0
    tmax_primary = 0.0
    cmax_secondary = 0.0
    tmax_secondary = 0.0
    ehc_detected = False

    if peaks:
        # primary peak = first peak
        idx_primary = peaks[0]
        cmax_primary = c_systemic[idx_primary]
        tmax_primary = times[idx_primary]

        # secondary peak: any subsequent peak
        if len(peaks) >= 2:
            # find highest secondary peak
            best_sec_idx = max(peaks[1:], key=lambda i: c_systemic[i])
            cmax_secondary = c_systemic[best_sec_idx]
            tmax_secondary = times[best_sec_idx]
            ehc_detected = True
    else:
        # no local maxima — use overall maximum as primary
        idx_max = max(range(len(c_systemic)), key=lambda i: c_systemic[i])
        cmax_primary = c_systemic[idx_max]
        tmax_primary = times[idx_max]

    auc = _trapezoidal_auc(times, c_systemic)

    notes_parts = [f"EHC simulation for {drug_name}"]
    if ehc_detected:
        notes_parts.append(
            f"Secondary EHC peak detected at t={tmax_secondary:.1f}h "
            f"(Cmax={cmax_secondary:.4f} mg/L)"
        )
    else:
        notes_parts.append("No secondary EHC peak detected")
    notes = "; ".join(notes_parts)

    return BileAcidTransporterResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_biliary=f_biliary,
        times_h=times,
        c_systemic_mg_L=c_systemic,
        cmax_primary_mg_L=cmax_primary,
        tmax_primary_h=tmax_primary,
        cmax_secondary_mg_L=cmax_secondary,
        tmax_secondary_h=tmax_secondary,
        ehc_detected=ehc_detected,
        auc_total_mg_h_per_L=auc,
        notes=notes,
    )


def compare_biliary_fractions(
    drug_name: str,
    dose_mg: float,
    fractions: list | None = None,
) -> list:
    """Compare EHC profiles for different f_biliary values.

    Returns list of BileAcidTransporterResult sorted by auc_total descending.
    """
    if fractions is None:
        fractions = [0.0, 0.1, 0.3, 0.5]

    results = []
    for f_bil in fractions:
        result = simulate_bile_acid_transporter_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            f_biliary=f_bil,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_total_mg_h_per_L, reverse=True)
    return results
