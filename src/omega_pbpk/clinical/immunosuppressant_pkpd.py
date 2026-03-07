"""Immunosuppressant PK/PD Modeling — Phase 254.

Models PK/PD of immunosuppressant drugs (cyclosporine, tacrolimus, mycophenolate).
Used in transplant patients — requires narrow therapeutic window monitoring.

For cyclosporine (prototypical immunosuppressant):
- Cyclosporine binds cyclophilin -> inhibits calcineurin -> reduces IL-2 production
- PD model: Inhibition of IL-2 = Imax * C / (IC50 + C)

References
----------
- Kahan BD. N Engl J Med. 1989;321(25):1725-38 (cyclosporine pharmacology)
- Staatz CE, Tett SE. Clin Pharmacokinet. 2004;43(10):623-53 (tacrolimus PK)
- Shipkova M et al. Ther Drug Monit. 2016;38(S1):S1-5 (MPA TDM)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

# ---------------------------------------------------------------------------
# Therapeutic ranges (ng/mL) for supported drugs
# ---------------------------------------------------------------------------

_THERAPEUTIC_RANGES: dict[str, dict[str, float]] = {
    "cyclosporine": {"low": 100.0, "high": 300.0},
    "tacrolimus": {"low": 5.0, "high": 15.0},
    "mycophenolate": {"low": 1000.0, "high": 3500.0},  # ng/mL (MPA AUC proxy)
}

_VALID_DRUGS = set(_THERAPEUTIC_RANGES.keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImmunoSuppResult:
    """Result from immunosuppressant PK/PD simulation.

    Attributes
    ----------
    drug_name : Name of the immunosuppressant.
    dose_mg : Administered dose (mg).
    patient_weight_kg : Patient body weight (kg).
    route : Administration route ('oral' or 'iv').
    times_h : Simulation time points (h).
    c_plasma_ng_mL : Plasma concentration time-course (ng/mL).
    il2_inhibition_pct : IL-2 inhibition percentage time-course (%).
    cmax_ng_mL : Peak plasma concentration (ng/mL).
    trough_12h_ng_mL : Trough concentration at 12 h (C0, ng/mL).
    auc_0_12h_ng_mL_h : AUC from 0 to 12 h (ng·h/mL).
    therapeutic_status : Trough-based status ('subtherapeutic', 'therapeutic',
                         'supratherapeutic').
    notes : Simulation summary notes.
    """

    drug_name: str
    dose_mg: float
    patient_weight_kg: float
    route: str
    times_h: list[float]
    c_plasma_ng_mL: list[float]
    il2_inhibition_pct: list[float]
    cmax_ng_mL: float
    trough_12h_ng_mL: float
    auc_0_12h_ng_mL_h: float
    therapeutic_status: str
    notes: str


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def simulate_immunosuppressant_pk(
    drug_name: str,
    dose_mg: float,
    patient_weight_kg: float = 70.0,
    cl_L_per_h: float = 25.0,
    vd_L: float = 600.0,
    ka_per_h: float = 1.5,
    f_oral: float = 0.3,
    route: str = "oral",
    imax: float = 0.9,
    ic50_ng_mL: float = 200.0,
    tac_dose: bool = False,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> ImmunoSuppResult:
    """Simulate immunosuppressant PK and IL-2 inhibition PD.

    Uses a one-compartment model:
    - Oral: first-order absorption with bioavailability
    - IV: bolus into central compartment

    PD model (inhibitory Emax):
        IL-2_inhibition_pct = 100 * Imax * C / (IC50 + C)

    Therapeutic monitoring based on 12-h trough (C0):
    - Cyclosporine: 100-300 ng/mL
    - Tacrolimus: 5-15 ng/mL
    - Mycophenolate: 1000-3500 ng/mL

    Parameters
    ----------
    drug_name : Drug name. Must be one of 'cyclosporine', 'tacrolimus',
                'mycophenolate'.
    dose_mg : Oral dose (mg). Must be > 0.
    patient_weight_kg : Patient body weight (kg). Must be > 0.
    cl_L_per_h : Total clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    ka_per_h : First-order absorption rate constant (1/h). Used for oral route.
    f_oral : Oral bioavailability fraction [0, 1].
    route : 'oral' or 'iv'.
    imax : Maximum fractional inhibition of IL-2 (0, 1].
    ic50_ng_mL : Drug concentration producing 50% of Imax (ng/mL). Must be > 0.
    tac_dose : Unused legacy flag (kept for API compatibility).
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    ImmunoSuppResult

    Raises
    ------
    ValueError
        If input validation fails.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if patient_weight_kg <= 0:
        raise ValueError(f"patient_weight_kg must be > 0, got {patient_weight_kg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if imax <= 0 or imax > 1:
        raise ValueError(f"imax must be in (0, 1], got {imax}")
    if ic50_ng_mL <= 0:
        raise ValueError(f"ic50_ng_mL must be > 0, got {ic50_ng_mL}")
    if drug_name.lower() not in _VALID_DRUGS:
        raise ValueError(f"drug_name must be one of {sorted(_VALID_DRUGS)}, got {drug_name!r}")

    drug_key = drug_name.lower()
    route_lower = route.lower()

    # --- PK simulation ---
    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    times = np.arange(0.0, t_end_h + dt_h / 2.0, dt_h)
    n = len(times)

    # Concentration in mg/L (plasma)
    c_mg_L = np.zeros(n)

    if route_lower == "iv":
        # Bolus: C(t) = (Dose / Vd) * exp(-ke * t)
        c0_mg_L = dose_mg / vd_L  # mg/L
        c_mg_L = c0_mg_L * np.exp(-ke * times)
    else:
        # Oral: C(t) = (F * Dose * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))
        # Avoid numerical issue if ka == ke
        absorbed_dose_mg = f_oral * dose_mg
        if abs(ka_per_h - ke) < 1e-6:
            # Degenerate case: use small offset
            ka_eff = ka_per_h + 1e-6
        else:
            ka_eff = ka_per_h

        coeff = (absorbed_dose_mg * ka_eff) / (vd_L * (ka_eff - ke))
        c_mg_L = coeff * (np.exp(-ke * times) - np.exp(-ka_eff * times))
        c_mg_L = np.maximum(c_mg_L, 0.0)

    # Convert mg/L -> ng/mL: 1 mg/L = 1000 ng/mL
    c_ng_mL = c_mg_L * 1000.0

    # --- PD: IL-2 inhibition ---
    # Inhibitory Emax: inhibition_frac = Imax * C / (IC50 + C)
    il2_inhibition = 100.0 * imax * c_ng_mL / (ic50_ng_mL + c_ng_mL)

    # --- Derived PK metrics ---
    cmax_ng_mL = float(np.max(c_ng_mL))

    # Trough at 12 h (or closest time point)
    idx_12h = int(round(12.0 / dt_h))
    idx_12h = min(idx_12h, n - 1)
    trough_12h_ng_mL = float(c_ng_mL[idx_12h])

    # AUC 0 -> 12 h
    times_12 = times[: idx_12h + 1]
    c_12 = c_ng_mL[: idx_12h + 1]
    auc_0_12h = float(np_trapz(c_12, times_12))

    # --- Therapeutic monitoring ---
    tm_status = therapeutic_monitoring_assessment(
        trough_ng_mL=trough_12h_ng_mL,
        drug_name=drug_key,
    )
    therapeutic_status = tm_status["status"]

    notes = (
        f"{drug_name} {dose_mg} mg {route}: "
        f"Cmax={cmax_ng_mL:.1f} ng/mL, "
        f"C_trough(12h)={trough_12h_ng_mL:.1f} ng/mL, "
        f"AUC(0-12h)={auc_0_12h:.0f} ng·h/mL, "
        f"status={therapeutic_status}."
    )

    return ImmunoSuppResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        patient_weight_kg=patient_weight_kg,
        route=route,
        times_h=times.tolist(),
        c_plasma_ng_mL=c_ng_mL.tolist(),
        il2_inhibition_pct=il2_inhibition.tolist(),
        cmax_ng_mL=cmax_ng_mL,
        trough_12h_ng_mL=trough_12h_ng_mL,
        auc_0_12h_ng_mL_h=auc_0_12h,
        therapeutic_status=therapeutic_status,
        notes=notes,
    )


def therapeutic_monitoring_assessment(
    trough_ng_mL: float,
    drug_name: str = "cyclosporine",
) -> dict[str, str]:
    """Assess therapeutic drug monitoring status from trough concentration.

    Parameters
    ----------
    trough_ng_mL : Measured trough concentration (ng/mL).
    drug_name : Drug name (case-insensitive). Must be one of 'cyclosporine',
                'tacrolimus', 'mycophenolate'.

    Returns
    -------
    dict with keys:
        - 'status': 'subtherapeutic', 'therapeutic', or 'supratherapeutic'
        - 'recommendation': dosing recommendation string

    Raises
    ------
    ValueError
        If drug_name is not recognised.
    """
    drug_key = drug_name.lower()
    if drug_key not in _VALID_DRUGS:
        raise ValueError(f"drug_name must be one of {sorted(_VALID_DRUGS)}, got {drug_name!r}")

    ranges = _THERAPEUTIC_RANGES[drug_key]
    low = ranges["low"]
    high = ranges["high"]

    if trough_ng_mL < low:
        status = "subtherapeutic"
        recommendation = (
            f"Trough {trough_ng_mL:.1f} ng/mL is below the target range "
            f"({low:.0f}-{high:.0f} ng/mL). Consider dose increase."
        )
    elif trough_ng_mL <= high:
        status = "therapeutic"
        recommendation = (
            f"Trough {trough_ng_mL:.1f} ng/mL is within the target range "
            f"({low:.0f}-{high:.0f} ng/mL). Continue current dose."
        )
    else:
        status = "supratherapeutic"
        recommendation = (
            f"Trough {trough_ng_mL:.1f} ng/mL exceeds the target range "
            f"({low:.0f}-{high:.0f} ng/mL). Consider dose reduction and "
            "monitor for toxicity."
        )

    return {"status": status, "recommendation": recommendation}


def dose_adjustment_by_trough(
    current_dose_mg: float,
    current_trough_ng_mL: float,
    target_trough_ng_mL: float,
    drug_name: str = "cyclosporine",
) -> float:
    """Compute a proportional dose adjustment to reach a target trough.

    Assumes linear (first-order) PK:
        new_dose = current_dose * target_trough / current_trough

    Parameters
    ----------
    current_dose_mg : Current dose (mg). Must be > 0.
    current_trough_ng_mL : Measured trough under current dose (ng/mL).
                           Must be > 0.
    target_trough_ng_mL : Desired trough concentration (ng/mL). Must be > 0.
    drug_name : Drug name (case-insensitive).

    Returns
    -------
    float
        Recommended new dose (mg).

    Raises
    ------
    ValueError
        If any parameter fails validation.
    """
    if current_dose_mg <= 0:
        raise ValueError(f"current_dose_mg must be > 0, got {current_dose_mg}")
    if current_trough_ng_mL <= 0:
        raise ValueError(f"current_trough_ng_mL must be > 0, got {current_trough_ng_mL}")
    if target_trough_ng_mL <= 0:
        raise ValueError(f"target_trough_ng_mL must be > 0, got {target_trough_ng_mL}")
    drug_key = drug_name.lower()
    if drug_key not in _VALID_DRUGS:
        raise ValueError(f"drug_name must be one of {sorted(_VALID_DRUGS)}, got {drug_name!r}")

    new_dose = current_dose_mg * (target_trough_ng_mL / current_trough_ng_mL)
    return float(new_dose)
