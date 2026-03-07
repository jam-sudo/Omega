"""
Phase 589 — Antimicrobial PK/PD Target Attainment Modeling

Implements PK/PD index computation, target attainment analysis,
MBC/MIC ratio classification, and multiple-dose PTA simulation.
"""



# ---------------------------------------------------------------------------
# PK/PD index mapping
# ---------------------------------------------------------------------------

_PKPD_INDEX_MAP = {
    "beta_lactam": "fT_MIC",
    "aminoglycoside": "Cmax_MIC",
    "fluoroquinolone": "AUC_MIC",
    "vancomycin": "AUC_MIC",
    "azithromycin": "AUC_MIC",
}

_TARGET_MAP = {
    "beta_lactam": 50.0,  # % fT>MIC
    "aminoglycoside": 10.0,  # Cmax/MIC ratio
    "fluoroquinolone": 125.0,  # AUC/MIC (gram-negative)
    "vancomycin": 400.0,  # AUC/MIC
    "azithromycin": 125.0,  # AUC/MIC
}

_VALID_CLASSES = set(_PKPD_INDEX_MAP.keys())


def _validate_class(antibiotic_class: str) -> None:
    if antibiotic_class not in _VALID_CLASSES:
        raise ValueError(
            f"Unknown antibiotic_class '{antibiotic_class}'. "
            f"Valid options: {sorted(_VALID_CLASSES)}"
        )


def pk_pd_index(antibiotic_class: str) -> str:
    """Return the primary PK/PD index for the given antibiotic class."""
    _validate_class(antibiotic_class)
    return _PKPD_INDEX_MAP[antibiotic_class]


# ---------------------------------------------------------------------------
# Target attainment calculation
# ---------------------------------------------------------------------------


def target_attainment(
    auc_mg_h_L: float,
    cmax_mg_L: float,
    mic_mg_L: float,
    t_above_mic_h: float,
    tau_h: float,
    antibiotic_class: str,
) -> dict:
    """
    Compute PK/PD index value and compare to target.

    Parameters
    ----------
    auc_mg_h_L    : AUC over dosing interval (mg·h/L)
    cmax_mg_L     : Peak concentration (mg/L)
    mic_mg_L      : Minimum inhibitory concentration (mg/L)
    t_above_mic_h : Hours above MIC in the dosing interval
    tau_h         : Dosing interval (h)
    antibiotic_class : One of the valid antibiotic classes

    Returns
    -------
    dict with pk_pd_index_name, index_value, target_value,
              attainment_achieved, attainment_pct
    """
    _validate_class(antibiotic_class)

    index_name = _PKPD_INDEX_MAP[antibiotic_class]
    target = _TARGET_MAP[antibiotic_class]

    if antibiotic_class == "beta_lactam":
        # fT>MIC expressed as percentage
        index_value = (t_above_mic_h / tau_h) * 100.0 if tau_h > 0 else 0.0
    elif antibiotic_class == "aminoglycoside":
        index_value = cmax_mg_L / mic_mg_L if mic_mg_L > 0 else 0.0
    else:
        # AUC/MIC — fluoroquinolone, vancomycin, azithromycin
        index_value = auc_mg_h_L / mic_mg_L if mic_mg_L > 0 else 0.0

    attainment_achieved = index_value >= target
    attainment_pct = (index_value / target * 100.0) if target > 0 else 0.0

    return {
        "pk_pd_index_name": index_name,
        "index_value": index_value,
        "target_value": target,
        "attainment_achieved": attainment_achieved,
        "attainment_pct": attainment_pct,
    }


# ---------------------------------------------------------------------------
# MBC/MIC ratio classification
# ---------------------------------------------------------------------------


def mbc_to_mic_ratio(mbc: float, mic: float) -> dict:
    """
    Classify antibacterial kill type based on MBC/MIC ratio.

    MBC/MIC < 4   → bactericidal
    MBC/MIC 4-16  → intermediate
    MBC/MIC > 16  → bacteriostatic
    """
    if mic <= 0:
        raise ValueError("MIC must be positive")
    ratio = mbc / mic

    if ratio < 4:
        kill_type = "bactericidal"
        notes = "MBC/MIC < 4: drug is bactericidal at the MIC"
    elif ratio <= 16:
        kill_type = "intermediate"
        notes = "MBC/MIC 4-16: intermediate bactericidal/bacteriostatic activity"
    else:
        kill_type = "bacteriostatic"
        notes = "MBC/MIC > 16: drug is primarily bacteriostatic"

    return {
        "ratio": ratio,
        "kill_type": kill_type,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Multiple-dose PTA simulation
# ---------------------------------------------------------------------------


def antibiotic_pta_simulation(
    drug_name: str,
    dose_mg: float,
    interval_h: float,
    cl_L_per_h: float,
    vd_L: float,
    f_unbound: float,
    mic_mg_L: float,
    antibiotic_class: str,
    n_doses: int = 5,
    dt_h: float = 0.1,
) -> dict:
    """
    Simulate multiple-dose IV bolus PK (1-compartment) and compute PTA
    for the last dose interval.

    Returns
    -------
    dict with:
        times_h           : list of time points (h)
        c_total           : list of total drug concentrations (mg/L)
        c_free            : list of free drug concentrations (mg/L)
        t_above_mic_pct   : % of last dosing interval where c_free > MIC
        auc_free_mg_h_L   : AUC of free drug over last dosing interval
        cmax_free         : peak free concentration in last interval
        pk_pd_attainment  : dict from target_attainment()
    """
    _validate_class(antibiotic_class)

    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")
    if not (0 < f_unbound <= 1):
        raise ValueError("f_unbound must be in (0, 1]")

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # Forward Euler simulation of multiple IV bolus doses
    total_time = n_doses * interval_h
    n_steps = int(round(total_time / dt_h))

    times_h: list[float] = []
    c_total: list[float] = []
    c_free: list[float] = []

    c = 0.0
    t = 0.0

    for step in range(n_steps + 1):
        # Apply bolus at each dose time (t = 0, interval_h, 2*interval_h, ...)
        if step > 0 and abs(t % interval_h) < dt_h * 0.5:
            c += dose_mg / vd_L

        times_h.append(t)
        c_total.append(max(c, 0.0))
        c_free.append(max(c * f_unbound, 0.0))

        # Forward Euler for elimination
        dcdt = -ke * c
        c += dcdt * dt_h
        t += dt_h

    # Apply initial bolus at t=0
    # Re-run with bolus at t=0 included
    times_h = []
    c_total = []
    c_free = []

    c = dose_mg / vd_L  # first bolus at t=0
    t = 0.0

    for _step in range(n_steps + 1):
        times_h.append(t)
        c_total.append(max(c, 0.0))
        c_free.append(max(c * f_unbound, 0.0))

        dcdt = -ke * c
        c += dcdt * dt_h
        t += dt_h

        # Apply bolus at next dose times (after advancing time)
        next_t = t
        for dose_idx in range(1, n_doses):
            dose_t = dose_idx * interval_h
            # Check if we just crossed a dose time
            if abs(next_t - dose_t) < dt_h * 0.5:
                c += dose_mg / vd_L
                break

    # Identify the last dosing interval
    last_dose_start = (n_doses - 1) * interval_h
    last_dose_end = n_doses * interval_h

    # Extract data for last interval
    last_times = []
    last_c_free = []
    for i, ti in enumerate(times_h):
        if last_dose_start - dt_h * 0.5 <= ti <= last_dose_end + dt_h * 0.5:
            last_times.append(ti)
            last_c_free.append(c_free[i])

    # Compute metrics for last interval
    if not last_times:
        last_times = times_h
        last_c_free = c_free

    tau_h = interval_h
    cmax_free = max(last_c_free) if last_c_free else 0.0

    # t > MIC percentage
    n_above = sum(1 for cf in last_c_free if cf > mic_mg_L)
    t_above_mic_pct = (n_above / len(last_c_free) * 100.0) if last_c_free else 0.0

    # AUC by trapezoidal rule
    auc_free = 0.0
    for i in range(1, len(last_times)):
        dt = last_times[i] - last_times[i - 1]
        auc_free += 0.5 * (last_c_free[i - 1] + last_c_free[i]) * dt

    # t_above_mic_h for target_attainment
    t_above_mic_h = t_above_mic_pct / 100.0 * tau_h

    pk_pd = target_attainment(
        auc_mg_h_L=auc_free,
        cmax_mg_L=cmax_free,
        mic_mg_L=mic_mg_L,
        t_above_mic_h=t_above_mic_h,
        tau_h=tau_h,
        antibiotic_class=antibiotic_class,
    )

    return {
        "times_h": times_h,
        "c_total": c_total,
        "c_free": c_free,
        "t_above_mic_pct": t_above_mic_pct,
        "auc_free_mg_h_L": auc_free,
        "cmax_free": cmax_free,
        "pk_pd_attainment": pk_pd,
    }
