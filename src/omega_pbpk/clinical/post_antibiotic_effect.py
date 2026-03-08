"""Phase 532: Antibiotic Post-Antibiotic Effect (PAE) PK/PD Model.

Models continued bacterial suppression after antibiotic concentration falls
below MIC (post-antibiotic effect). Uses analytical 1-compartment PK (IV bolus).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Antibiotic class PAE parameters: (min_h, max_h, pk_pd_index)
_CLASS_PARAMS: dict = {
    "aminoglycoside": (2.0, 4.0, "AUC/MIC"),
    "fluoroquinolone": (2.0, 6.0, "AUC/MIC"),
    "beta_lactam": (0.0, 1.0, "T>MIC"),
    "macrolide": (2.0, 8.0, "AUC/MIC"),
    "oxazolidinone": (1.0, 3.0, "T>MIC"),
}

VALID_CLASSES = frozenset(_CLASS_PARAMS.keys())


@dataclass(frozen=True)
class PostAntibioticEffectResult:
    """Result of post-antibiotic effect simulation."""

    drug_name: str
    antibiotic_class: str
    dose_mg: float
    mic_mg_L: float
    cmax_mg_L: float
    cmax_mic_ratio: float
    auc_mg_h_per_L: float
    auc_mic_ratio: float
    t_above_mic_h: float
    pae_h: float
    effective_duration_h: float
    pk_pd_index: str
    efficacy_prediction: str
    notes: str


def _compute_pae(antibiotic_class: str, cmax_mic_ratio: float) -> float:
    """Compute PAE duration in hours.

    PAE = base_pae * (1 + 0.5 * log10(max(1, cmax_mic_ratio)))
    Clamped to [min_h, max_h] for the class.
    """
    min_h, max_h, _ = _CLASS_PARAMS[antibiotic_class]
    base_pae = (min_h + max_h) / 2.0
    pae = base_pae * (1.0 + 0.5 * math.log10(max(1.0, cmax_mic_ratio)))
    if pae < min_h:
        pae = min_h
    if pae > max_h:
        pae = max_h
    return pae


def _efficacy_prediction(
    antibiotic_class: str,
    auc_mic_ratio: float,
    t_above_mic_h: float,
    t_end_h: float,
) -> str:
    """Determine efficacy classification.

    - AUC/MIC > 125 (aminoglycoside, fluoroquinolone, macrolide): optimal
    - T>MIC > 50% of dosing interval (beta_lactam, oxazolidinone): optimal
    """
    _, _, pk_pd_index = _CLASS_PARAMS[antibiotic_class]
    if pk_pd_index == "T>MIC":
        pct_above = (t_above_mic_h / t_end_h) * 100.0 if t_end_h > 0 else 0.0
        return "optimal" if pct_above > 50.0 else "suboptimal"
    else:
        return "optimal" if auc_mic_ratio > 125.0 else "suboptimal"


def simulate_post_antibiotic_effect(
    drug_name: str,
    antibiotic_class: str,
    dose_mg: float,
    mic_mg_L: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 24.0,
) -> PostAntibioticEffectResult:
    """Simulate antibiotic PK/PD including post-antibiotic effect.

    Uses analytical 1-compartment IV bolus model.

    Parameters
    ----------
    drug_name:
        Name of the antibiotic.
    antibiotic_class:
        One of: aminoglycoside, fluoroquinolone, beta_lactam, macrolide, oxazolidinone.
    dose_mg:
        IV bolus dose in mg.
    mic_mg_L:
        Minimum inhibitory concentration (mg/L).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Dosing interval end time (h). Used for T>MIC% calculation.

    Returns
    -------
    PostAntibioticEffectResult
    """
    # Validate
    if antibiotic_class not in VALID_CLASSES:
        raise ValueError(
            f"antibiotic_class must be one of {sorted(VALID_CLASSES)}, got '{antibiotic_class}'"
        )
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mic_mg_L <= 0:
        raise ValueError("mic_mg_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    # 1-cpt IV bolus analytical PK
    cmax_mg_L = dose_mg / vd_L
    k_el = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # AUC for IV bolus = Dose / CL = Cmax / k_el
    auc_mg_h_per_L = dose_mg / cl_L_per_h

    # Time above MIC: C(t) = Cmax * exp(-k_el * t) > MIC
    # => t_mic = (1/k_el) * ln(Cmax/MIC)
    if cmax_mg_L > mic_mg_L:
        t_above_mic_h = (1.0 / k_el) * math.log(cmax_mg_L / mic_mg_L)
    else:
        t_above_mic_h = 0.0

    # Cmax/MIC ratio
    cmax_mic_ratio = cmax_mg_L / mic_mg_L

    # AUC/MIC ratio
    auc_mic_ratio = auc_mg_h_per_L / mic_mg_L

    # PAE duration
    pae_h = _compute_pae(antibiotic_class, cmax_mic_ratio)

    # Effective antibacterial duration
    effective_duration_h = t_above_mic_h + pae_h

    # PK/PD index
    _, _, pk_pd_index = _CLASS_PARAMS[antibiotic_class]
    # Aminoglycoside also uses Cmax/MIC as secondary index
    if antibiotic_class == "aminoglycoside":
        pk_pd_index = "AUC/MIC"

    # Efficacy
    efficacy = _efficacy_prediction(antibiotic_class, auc_mic_ratio, t_above_mic_h, t_end_h)

    notes = (
        f"Cmax/MIC={cmax_mic_ratio:.1f}, AUC/MIC={auc_mic_ratio:.1f}, "
        f"t>MIC={t_above_mic_h:.2f}h, PAE={pae_h:.2f}h"
    )

    return PostAntibioticEffectResult(
        drug_name=drug_name,
        antibiotic_class=antibiotic_class,
        dose_mg=dose_mg,
        mic_mg_L=mic_mg_L,
        cmax_mg_L=cmax_mg_L,
        cmax_mic_ratio=cmax_mic_ratio,
        auc_mg_h_per_L=auc_mg_h_per_L,
        auc_mic_ratio=auc_mic_ratio,
        t_above_mic_h=t_above_mic_h,
        pae_h=pae_h,
        effective_duration_h=effective_duration_h,
        pk_pd_index=pk_pd_index,
        efficacy_prediction=efficacy,
        notes=notes,
    )


def compare_mic_levels(
    drug_name: str,
    antibiotic_class: str,
    dose_mg: float,
    mic_values: list,
    cl_L_per_h: float,
    vd_L: float,
) -> list:
    """Compare PK/PD across different MIC levels.

    Parameters
    ----------
    drug_name:
        Drug name.
    antibiotic_class:
        Antibiotic class.
    dose_mg:
        IV bolus dose (mg).
    mic_values:
        List of MIC values (mg/L) to simulate.
    cl_L_per_h:
        Clearance (L/h).
    vd_L:
        Volume of distribution (L).

    Returns
    -------
    List of PostAntibioticEffectResult sorted by effective_duration_h descending.
    """
    results = [
        simulate_post_antibiotic_effect(drug_name, antibiotic_class, dose_mg, mic, cl_L_per_h, vd_L)
        for mic in mic_values
    ]
    results.sort(key=lambda r: r.effective_duration_h, reverse=True)
    return results
