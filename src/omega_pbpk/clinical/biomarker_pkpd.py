"""
Phase 234 — Biomarker PK/PD Model

Models drug effect on a biomarker (e.g., LDL-C, HbA1c, viral load) using
indirect response models. Biomarkers have natural turnover and drug alters
production or degradation.

Indirect response models:
  Model I   — Inhibit production:   dB/dt = kin*(1 - Imax*C/(IC50+C)) - kout*B
  Model II  — Stimulate production: dB/dt = kin*(1 + Smax*C/(SC50+C)) - kout*B
  Model III — Inhibit degradation:  dB/dt = kin - kout*B*(1 - Imax*C/(IC50+C))
  Model IV  — Stimulate degradation:dB/dt = kin - kout*B*(1 + Smax*C/(SC50+C))

Baseline: B0 = kin/kout
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_VALID_MODELS = {"I", "II", "III", "IV"}
_INHIBIT_MODELS = {"I", "III"}
_STIMULATE_MODELS = {"II", "IV"}


@dataclass
class BiomarkerResult:
    """Result of a biomarker PK/PD simulation."""

    drug_name: str
    biomarker_name: str
    dose_mg: float
    model: str
    times_h: np.ndarray
    c_plasma_mg_L: np.ndarray
    biomarker_values: np.ndarray
    b0_baseline: float
    b_min: float
    b_max: float
    pct_change_at_end: float
    time_to_nadir_h: float
    time_to_peak_h: float
    notes: list[str] = field(default_factory=list)


def simulate_biomarker_pkpd(
    drug_name: str,
    biomarker_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    kin_per_h: float,
    kout_per_h: float,
    imax_or_smax: float = 0.8,
    ic50_or_sc50_mg_L: float = 1.0,
    model: str = "I",
    route: str = "oral",
    ka_per_h: float = 1.0,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> BiomarkerResult:
    """Simulate indirect response biomarker PK/PD.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    biomarker_name:
        Name of the biomarker (e.g., "LDL-C", "HbA1c").
    dose_mg:
        Dose in milligrams (must be > 0).
    cl_L_per_h:
        Clearance in L/h (must be > 0).
    vd_L:
        Volume of distribution in L (must be > 0).
    kin_per_h:
        Biomarker zero-order production rate (must be > 0).
    kout_per_h:
        Biomarker first-order elimination rate (must be > 0).
    imax_or_smax:
        Maximum inhibition (Imax, in (0, 1]) or stimulation (Smax, > 0).
        For inhibition models (I, III): must be in (0, 1].
    ic50_or_sc50_mg_L:
        Drug concentration at half-maximal effect in mg/L (must be > 0).
    model:
        Indirect response model type: "I", "II", "III", or "IV".
    route:
        Administration route: "oral" or "iv".
    ka_per_h:
        Oral absorption rate constant (must be > 0 for oral route).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Integration time step in hours.

    Returns
    -------
    BiomarkerResult
        Simulation result with time-course arrays and summary metrics.
    """
    # --- Input validation ---
    if model not in _VALID_MODELS:
        raise ValueError(f"model must be one of {_VALID_MODELS}, got '{model}'")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if kin_per_h <= 0:
        raise ValueError("kin_per_h must be > 0")
    if kout_per_h <= 0:
        raise ValueError("kout_per_h must be > 0")
    if model in _INHIBIT_MODELS:
        if not (0 < imax_or_smax <= 1.0):
            raise ValueError("Imax must be in (0, 1] for inhibition models I and III")
    else:
        if imax_or_smax <= 0:
            raise ValueError("Smax must be > 0 for stimulation models II and IV")
    if ic50_or_sc50_mg_L <= 0:
        raise ValueError("ic50_or_sc50_mg_L must be > 0")
    if route not in {"oral", "iv"}:
        raise ValueError("route must be 'oral' or 'iv'")
    if route == "oral" and ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0 for oral route")

    ke = cl_L_per_h / vd_L
    b0 = kin_per_h / kout_per_h  # baseline biomarker level

    n_steps = int(t_end_h / dt_h) + 1
    times = np.linspace(0.0, t_end_h, n_steps)

    c_plasma = np.zeros(n_steps)
    biomarker = np.zeros(n_steps)

    # Initial conditions
    if route == "oral":
        gut_remaining = dose_mg
        c_plasma[0] = 0.0
    else:
        gut_remaining = 0.0
        c_plasma[0] = dose_mg / vd_L

    biomarker[0] = b0

    eff_param = imax_or_smax
    ec50 = ic50_or_sc50_mg_L

    for i in range(1, n_steps):
        c = c_plasma[i - 1]
        b = biomarker[i - 1]

        # PK: 1-compartment forward Euler
        if route == "oral":
            absorbed_rate = ka_per_h * gut_remaining  # mg/h
            gut_remaining = max(0.0, gut_remaining - absorbed_rate * dt_h)
            dc = (absorbed_rate / vd_L) - ke * c
        else:
            dc = -ke * c

        c_new = max(0.0, c + dc * dt_h)
        c_plasma[i] = c_new

        # Use midpoint concentration for biomarker step
        c_mid = 0.5 * (c + c_new)

        # Indirect response ODE
        if model == "I":
            # Inhibit production
            dB = kin_per_h * (1.0 - eff_param * c_mid / (ec50 + c_mid)) - kout_per_h * b
        elif model == "II":
            # Stimulate production
            dB = kin_per_h * (1.0 + eff_param * c_mid / (ec50 + c_mid)) - kout_per_h * b
        elif model == "III":
            # Inhibit degradation → biomarker increases
            dB = kin_per_h - kout_per_h * b * (1.0 - eff_param * c_mid / (ec50 + c_mid))
        else:  # model == "IV"
            # Stimulate degradation → biomarker decreases
            dB = kin_per_h - kout_per_h * b * (1.0 + eff_param * c_mid / (ec50 + c_mid))

        biomarker[i] = max(0.0, b + dB * dt_h)

    b_end = float(biomarker[-1])
    b_min_val = float(np.min(biomarker))
    b_max_val = float(np.max(biomarker))
    pct_change_at_end = (b_end - b0) / b0 * 100.0 if b0 > 0 else 0.0

    # Time to nadir (minimum) — relevant for Models I and IV (biomarker decreases)
    idx_min = int(np.argmin(biomarker))
    time_to_nadir = float(times[idx_min])

    # Time to peak (maximum) — relevant for Models II and III (biomarker increases)
    idx_max = int(np.argmax(biomarker))
    time_to_peak = float(times[idx_max])

    notes: list[str] = []
    model_desc = {
        "I": "Inhibit production — biomarker decreases",
        "II": "Stimulate production — biomarker increases",
        "III": "Inhibit degradation — biomarker increases",
        "IV": "Stimulate degradation — biomarker decreases",
    }
    notes.append(model_desc[model])
    notes.append(f"B0 = {b0:.4g} (kin/kout = {kin_per_h}/{kout_per_h})")

    return BiomarkerResult(
        drug_name=drug_name,
        biomarker_name=biomarker_name,
        dose_mg=dose_mg,
        model=model,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        biomarker_values=biomarker,
        b0_baseline=b0,
        b_min=b_min_val,
        b_max=b_max_val,
        pct_change_at_end=pct_change_at_end,
        time_to_nadir_h=time_to_nadir,
        time_to_peak_h=time_to_peak,
        notes=notes,
    )


def time_to_effect(
    result: BiomarkerResult,
    target_pct_change: float = 50.0,
) -> float:
    """Find the time to achieve a target percent change in biomarker from baseline.

    Parameters
    ----------
    result:
        A :class:`BiomarkerResult` from :func:`simulate_biomarker_pkpd`.
    target_pct_change:
        Target percent change from baseline (positive = increase, negative = decrease).

    Returns
    -------
    float
        Time in hours to first reach the target, or ``np.inf`` if never achieved.
    """
    b0 = result.b0_baseline
    if b0 <= 0:
        return np.inf

    target_b = b0 * (1.0 + target_pct_change / 100.0)
    times = result.times_h
    b_values = result.biomarker_values

    # Search for first crossing
    if target_pct_change > 0:
        # Looking for biomarker to rise above target
        for i in range(len(b_values)):
            if b_values[i] >= target_b:
                if i == 0:
                    return float(times[0])
                # Linear interpolation
                frac = (target_b - b_values[i - 1]) / (b_values[i] - b_values[i - 1])
                return float(times[i - 1] + frac * (times[i] - times[i - 1]))
    else:
        # Looking for biomarker to fall below target
        for i in range(len(b_values)):
            if b_values[i] <= target_b:
                if i == 0:
                    return float(times[0])
                frac = (target_b - b_values[i - 1]) / (b_values[i] - b_values[i - 1])
                return float(times[i - 1] + frac * (times[i] - times[i - 1]))

    return np.inf
