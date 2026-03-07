"""Antibiotic PK/PD — time-dependent, concentration-dependent, and AUC-dependent killing."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AntibioticPKResult:
    """PK/PD result for antibiotic dosing simulation."""

    drug_name: str
    antibiotic_class: str
    dose_mg: float
    interval_h: float
    n_doses: int
    mic_mg_L: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    cmax: float
    auc_last_interval: float
    pkpd_index_value: float
    pkpd_index_name: str
    target_attained: bool
    notes: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_CLASSES = ("beta_lactam", "aminoglycoside", "fluoroquinolone")

# Free fraction (fu) used per class
_FU: dict[str, float] = {
    "beta_lactam": 0.4,
    "aminoglycoside": 0.9,
    "fluoroquinolone": 1.0,  # AUC/MIC uses total AUC; fu built into target
}

# PK/PD index names per class
_INDEX_NAME: dict[str, str] = {
    "beta_lactam": "fT>MIC (%)",
    "aminoglycoside": "Cmax/MIC",
    "fluoroquinolone": "AUC24/MIC",
}

# Target thresholds
_TARGET: dict[str, float] = {
    "beta_lactam": 40.0,        # fT>MIC > 40%
    "aminoglycoside": 10.0,     # Cmax/MIC > 10
    "fluoroquinolone": 125.0,   # AUC24/MIC > 125
}


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_antibiotic_pk(
    drug_name: str,
    antibiotic_class: str,
    dose_mg: float,
    interval_h: float,
    n_doses: int,
    cl_L_per_h: float,
    vd_L: float,
    mic_mg_L: float,
    ka_per_h: float = 1.5,
    f_oral: float = 1.0,
    dt_h: float = 0.05,
) -> AntibioticPKResult:
    """Simulate multi-dose antibiotic PK and compute PK/PD indices.

    Parameters
    ----------
    drug_name :
        Drug name.
    antibiotic_class :
        One of ``"beta_lactam"``, ``"aminoglycoside"``, ``"fluoroquinolone"``.
    dose_mg :
        Dose per administration (mg).
    interval_h :
        Dosing interval (h).
    n_doses :
        Number of doses to simulate.
    cl_L_per_h :
        Systemic clearance (L/h).
    vd_L :
        Volume of distribution (L).
    mic_mg_L :
        Minimum inhibitory concentration (mg/L).
    ka_per_h :
        First-order absorption rate (1/h). Set 0 for IV bolus.
    f_oral :
        Oral bioavailability fraction (0–1).
    dt_h :
        Forward-Euler time step (h).

    Returns
    -------
    AntibioticPKResult
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if antibiotic_class not in _VALID_CLASSES:
        raise ValueError(
            f"antibiotic_class must be one of {_VALID_CLASSES}, got '{antibiotic_class}'"
        )
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if interval_h <= 0:
        raise ValueError("interval_h must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if mic_mg_L <= 0:
        raise ValueError("mic_mg_L must be > 0")
    if ka_per_h < 0:
        raise ValueError("ka_per_h must be >= 0")
    if not 0.0 < f_oral <= 1.0:
        raise ValueError("f_oral must be in (0, 1]")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # ------------------------------------------------------------------
    # Forward-Euler 1-compartment oral PK
    # ------------------------------------------------------------------
    ke = cl_L_per_h / vd_L  # 1/h
    t_end = n_doses * interval_h
    n_steps = max(int(t_end / dt_h), 1)

    times_h: list[float] = []
    c_plasma_mg_L: list[float] = []

    a_gut = 0.0
    a_central = 0.0

    # Schedule dose times
    dose_times = [i * interval_h for i in range(n_doses)]
    next_dose_idx = 0

    for step in range(n_steps + 1):
        t = step * dt_h
        times_h.append(t)
        c_plasma_mg_L.append(a_central / vd_L)

        if step < n_steps:
            # Administer dose at scheduled time (within half dt tolerance)
            if next_dose_idx < len(dose_times):
                if t >= dose_times[next_dose_idx] - dt_h / 2:
                    bioavailable_dose = dose_mg * f_oral
                    if ka_per_h > 0:
                        a_gut += bioavailable_dose
                    else:
                        # IV bolus: straight into central compartment
                        a_central += bioavailable_dose
                    next_dose_idx += 1

            # Absorption (oral)
            if ka_per_h > 0 and a_gut > 0:
                absorbed = ka_per_h * a_gut * dt_h
                a_gut -= absorbed
                a_gut = max(a_gut, 0.0)
                a_central += absorbed

            # Elimination
            eliminated = ke * a_central * dt_h
            a_central = max(a_central - eliminated, 0.0)

    # ------------------------------------------------------------------
    # PK/PD index computation using last dosing interval
    # ------------------------------------------------------------------
    # Extract last interval time points
    last_dose_t = dose_times[-1]
    last_interval_times: list[float] = []
    last_interval_conc: list[float] = []
    for t_val, c_val in zip(times_h, c_plasma_mg_L, strict=True):
        if t_val >= last_dose_t - dt_h / 2:
            last_interval_times.append(t_val)
            last_interval_conc.append(c_val)

    # AUC of last interval (trapezoidal)
    auc_last = 0.0
    for j in range(len(last_interval_times) - 1):
        auc_last += (
            0.5
            * (last_interval_conc[j] + last_interval_conc[j + 1])
            * (last_interval_times[j + 1] - last_interval_times[j])
        )

    cmax = max(c_plasma_mg_L) if c_plasma_mg_L else 0.0
    fu = _FU[antibiotic_class]

    # Compute class-specific PK/PD index
    index_name = _INDEX_NAME[antibiotic_class]
    target = _TARGET[antibiotic_class]

    if antibiotic_class == "beta_lactam":
        # fT>MIC: fraction of last dosing interval with free conc > MIC
        free_conc_last = [c * fu for c in last_interval_conc]
        n_above = sum(1 for c in free_conc_last if c > mic_mg_L)
        f_t_mic_pct = 100.0 * n_above / max(len(free_conc_last), 1)
        index_value = f_t_mic_pct

    elif antibiotic_class == "aminoglycoside":
        # Cmax/MIC (using overall Cmax * fu)
        free_cmax = cmax * fu
        index_value = free_cmax / mic_mg_L

    else:  # fluoroquinolone
        # AUC24/MIC — scale last interval AUC to 24 h
        auc24 = auc_last * (24.0 / interval_h)
        index_value = auc24 / mic_mg_L

    target_attained = index_value >= target

    notes = (
        f"Class: {antibiotic_class}; fu={fu}; "
        f"target={target} {index_name}; attained={target_attained}"
    )

    return AntibioticPKResult(
        drug_name=drug_name,
        antibiotic_class=antibiotic_class,
        dose_mg=dose_mg,
        interval_h=interval_h,
        n_doses=n_doses,
        mic_mg_L=mic_mg_L,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        cmax=cmax,
        auc_last_interval=auc_last,
        pkpd_index_value=index_value,
        pkpd_index_name=index_name,
        target_attained=target_attained,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# MIC breakpoint analysis
# ---------------------------------------------------------------------------


def mic_breakpoint_analysis(
    drug_name: str,
    antibiotic_class: str,
    dose_mg: float,
    interval_h: float,
    cl_L_per_h: float,
    vd_L: float,
    mic_values: list[float],
    n_doses: int = 5,
    ka_per_h: float = 1.5,
    f_oral: float = 1.0,
    dt_h: float = 0.05,
) -> list[dict]:
    """Compute PK/PD index and target attainment for a series of MIC values.

    Parameters
    ----------
    mic_values :
        List of MIC values (mg/L) to evaluate.

    Returns
    -------
    list[dict]
        Each dict has keys: ``mic_mg_L``, ``pkpd_index_value``, ``pkpd_index_name``,
        ``target_attained``.
    """
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if not mic_values:
        raise ValueError("mic_values must be a non-empty list")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if interval_h <= 0:
        raise ValueError("interval_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if antibiotic_class not in _VALID_CLASSES:
        raise ValueError(
            f"antibiotic_class must be one of {_VALID_CLASSES}, got '{antibiotic_class}'"
        )

    results = []
    for mic in mic_values:
        if mic <= 0:
            raise ValueError(f"All mic_values must be > 0, got {mic}")
        res = simulate_antibiotic_pk(
            drug_name=drug_name,
            antibiotic_class=antibiotic_class,
            dose_mg=dose_mg,
            interval_h=interval_h,
            n_doses=n_doses,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            mic_mg_L=mic,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
            dt_h=dt_h,
        )
        results.append(
            {
                "mic_mg_L": mic,
                "pkpd_index_value": res.pkpd_index_value,
                "pkpd_index_name": res.pkpd_index_name,
                "target_attained": res.target_attained,
            }
        )
    return results


__all__ = [
    "AntibioticPKResult",
    "simulate_antibiotic_pk",
    "mic_breakpoint_analysis",
]
