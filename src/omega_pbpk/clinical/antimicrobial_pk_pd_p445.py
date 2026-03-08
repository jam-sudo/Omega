"""
Phase 445 — Antimicrobial PK/PD Target Attainment

MIC-based target attainment for antibiotics (bactericidal vs bacteriostatic).
Supports beta-lactams (%T>MIC), aminoglycosides (Cmax/MIC), fluoroquinolones
and vancomycin (AUC/MIC) using 1-compartment PK models.
"""

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AntimicrobialPKPDResult:
    drug_name: str
    pkpd_index: str  # "%T_MIC", "AUC_MIC", "Cmax_MIC"
    dose_mg: float
    dosing_interval_h: float
    mic_mg_L: float
    t_greater_mic_pct: float  # % of dosing interval above MIC
    auc_over_mic: float  # AUC24h / MIC
    cmax_over_mic: float  # Cmax / MIC
    pta_target_met: bool
    pharmacodynamic_target: float
    bactericidal_activity: str  # "achieved", "partial", "not_achieved"
    dose_recommendation: str  # "adequate", "increase_dose", "increase_frequency", "combine"
    t_half_h: float
    notes: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_CLASSES = {"beta_lactam", "aminoglycoside", "fluoroquinolone", "vancomycin"}
_VALID_ROUTES = {"iv", "oral"}

_PKPD_INDEX = {
    "beta_lactam": "%T_MIC",
    "aminoglycoside": "Cmax_MIC",
    "fluoroquinolone": "AUC_MIC",
    "vancomycin": "AUC_MIC",
}

_THRESHOLDS = {
    "beta_lactam": 40.0,  # % T>MIC
    "aminoglycoside": 10.0,  # Cmax/MIC
    "fluoroquinolone": 125.0,  # AUC/MIC
    "vancomycin": 400.0,  # AUC/MIC
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _compute_pk_iv(
    dose_mg: float, cl_L_per_h: float, vd_L: float, dosing_interval_h: float, mic_mg_L: float
):
    """Compute 1-cpt IV bolus PK metrics."""
    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)
    t_half = math.log(2) / ke
    c0 = dose_mg / vd_L  # peak (Cmax) in mg/L
    cmax = c0

    # AUC per dose (trapezoidal over dosing interval)
    n_steps = 1000
    dt = dosing_interval_h / n_steps
    times = [i * dt for i in range(n_steps + 1)]
    concs = [c0 * math.exp(-ke * t) for t in times]
    auc_dose = sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )
    auc_24h = auc_dose * (24.0 / dosing_interval_h)

    # T>MIC: time in dosing interval where C(t) > MIC
    if cmax <= mic_mg_L:
        t_mic = 0.0
    else:
        # C(t) = c0 * exp(-ke*t) = mic → t = ln(c0/mic)/ke
        t_mic = math.log(c0 / mic_mg_L) / ke
        t_mic = min(t_mic, dosing_interval_h)

    t_greater_mic_pct = (t_mic / dosing_interval_h) * 100.0 if t_mic > 0 else 0.0
    t_greater_mic_pct = max(0.0, min(100.0, t_greater_mic_pct))

    return t_half, cmax, auc_24h, t_greater_mic_pct


def _compute_pk_oral(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float,
    dosing_interval_h: float,
    mic_mg_L: float,
):
    """Compute 1-cpt oral PK metrics using numerical integration."""
    ke = cl_L_per_h / vd_L
    t_half = math.log(2) / ke

    # Handle ka == ke edge case
    if abs(ka_per_h - ke) < 1e-9:
        ka_per_h = ka_per_h * 1.0001

    coeff = (f_oral * dose_mg * ka_per_h) / (vd_L * (ka_per_h - ke))

    def c_oral(t: float) -> float:
        return coeff * (math.exp(-ke * t) - math.exp(-ka_per_h * t))

    # Numerical integration
    n_steps = 2000
    dt = dosing_interval_h / n_steps
    times = [i * dt for i in range(n_steps + 1)]
    concs = [max(0.0, c_oral(t)) for t in times]

    cmax = max(concs)
    auc_dose = sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )
    auc_24h = auc_dose * (24.0 / dosing_interval_h)

    # T>MIC by counting steps
    count_above = sum(1 for c in concs if c > mic_mg_L)
    t_greater_mic_pct = (count_above / (n_steps + 1)) * 100.0
    t_greater_mic_pct = max(0.0, min(100.0, t_greater_mic_pct))

    return t_half, cmax, auc_24h, t_greater_mic_pct


def _bactericidal_activity(pkpd_index: str, index_value: float, threshold: float) -> str:
    ratio = index_value / threshold if threshold > 0 else 0.0
    if ratio >= 1.0:
        return "achieved"
    elif ratio >= 0.5:
        return "partial"
    else:
        return "not_achieved"


def _dose_recommendation(
    pkpd_index: str, index_value: float, threshold: float, antibiotic_class: str
) -> str:
    ratio = index_value / threshold if threshold > 0 else 0.0
    if ratio >= 1.0:
        return "adequate"
    elif ratio >= 0.7:
        # Close to target — increase frequency for time-dependent, dose for conc-dependent
        if antibiotic_class == "beta_lactam":
            return "increase_frequency"
        else:
            return "increase_dose"
    elif ratio >= 0.4:
        if antibiotic_class == "beta_lactam":
            return "increase_frequency"
        else:
            return "increase_dose"
    else:
        return "combine"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_antimicrobial_pkpd(
    drug_name: str,
    antibiotic_class: str,
    dose_mg: float,
    dosing_interval_h: float,
    mic_mg_L: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 5.0,
    f_oral: float = 1.0,
    route: str = "iv",
) -> AntimicrobialPKPDResult:
    """Assess antimicrobial PK/PD target attainment.

    Parameters
    ----------
    drug_name : str
    antibiotic_class : str
        One of: "beta_lactam", "aminoglycoside", "fluoroquinolone", "vancomycin"
    dose_mg : float
        Dose in mg
    dosing_interval_h : float
        Dosing interval in hours (e.g. 8 for q8h)
    mic_mg_L : float
        Minimum inhibitory concentration in mg/L
    cl_L_per_h : float
        Clearance in L/h
    vd_L : float
        Volume of distribution in L
    ka_per_h : float
        Absorption rate constant for oral route (default 5.0 /h)
    f_oral : float
        Oral bioavailability fraction (default 1.0)
    route : str
        "iv" or "oral"

    Returns
    -------
    AntimicrobialPKPDResult
    """
    if antibiotic_class not in _VALID_CLASSES:
        raise ValueError(
            f"antibiotic_class must be one of {_VALID_CLASSES}, got '{antibiotic_class}'"
        )
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if mic_mg_L <= 0:
        raise ValueError(f"mic_mg_L must be > 0, got {mic_mg_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got '{route}'")
    if dosing_interval_h <= 0:
        raise ValueError(f"dosing_interval_h must be > 0, got {dosing_interval_h}")

    pkpd_index = _PKPD_INDEX[antibiotic_class]
    threshold = _THRESHOLDS[antibiotic_class]

    if route == "iv":
        t_half, cmax, auc_24h, t_greater_mic_pct = _compute_pk_iv(
            dose_mg, cl_L_per_h, vd_L, dosing_interval_h, mic_mg_L
        )
    else:
        t_half, cmax, auc_24h, t_greater_mic_pct = _compute_pk_oral(
            dose_mg, cl_L_per_h, vd_L, ka_per_h, f_oral, dosing_interval_h, mic_mg_L
        )

    auc_over_mic = auc_24h / mic_mg_L
    cmax_over_mic = cmax / mic_mg_L

    # Determine the index value for target assessment
    if pkpd_index == "%T_MIC":
        index_value = t_greater_mic_pct
    elif pkpd_index == "Cmax_MIC":
        index_value = cmax_over_mic
    else:  # AUC_MIC
        index_value = auc_over_mic

    pta_target_met = index_value >= threshold

    bact_activity = _bactericidal_activity(pkpd_index, index_value, threshold)
    dose_rec = _dose_recommendation(pkpd_index, index_value, threshold, antibiotic_class)

    status = "MET" if pta_target_met else "NOT MET"
    notes = (
        f"PK/PD index: {pkpd_index}; value={index_value:.1f}; "
        f"target={threshold}; target {status}. "
        f"t1/2={t_half:.2f}h, Cmax={cmax:.2f} mg/L, AUC24={auc_24h:.1f} mg*h/L."
    )

    return AntimicrobialPKPDResult(
        drug_name=drug_name,
        pkpd_index=pkpd_index,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        mic_mg_L=mic_mg_L,
        t_greater_mic_pct=t_greater_mic_pct,
        auc_over_mic=auc_over_mic,
        cmax_over_mic=cmax_over_mic,
        pta_target_met=pta_target_met,
        pharmacodynamic_target=threshold,
        bactericidal_activity=bact_activity,
        dose_recommendation=dose_rec,
        t_half_h=t_half,
        notes=notes,
    )


def compare_dosing_regimens(
    drug_name: str,
    antibiotic_class: str,
    mic_mg_L: float,
    cl_L_per_h: float,
    vd_L: float,
    doses_mg: list[float],
    intervals_h: list[float],
) -> list[AntimicrobialPKPDResult]:
    """Compare all combinations of doses and dosing intervals.

    Returns list sorted by the primary PK/PD index value descending.
    """
    pkpd_index = _PKPD_INDEX[antibiotic_class]
    results = []
    for dose in doses_mg:
        for interval in intervals_h:
            result = assess_antimicrobial_pkpd(
                drug_name=drug_name,
                antibiotic_class=antibiotic_class,
                dose_mg=dose,
                dosing_interval_h=interval,
                mic_mg_L=mic_mg_L,
                cl_L_per_h=cl_L_per_h,
                vd_L=vd_L,
            )
            results.append(result)

    def sort_key(r: AntimicrobialPKPDResult) -> float:
        if pkpd_index == "%T_MIC":
            return r.t_greater_mic_pct
        elif pkpd_index == "Cmax_MIC":
            return r.cmax_over_mic
        else:
            return r.auc_over_mic

    results.sort(key=sort_key, reverse=True)
    return results
