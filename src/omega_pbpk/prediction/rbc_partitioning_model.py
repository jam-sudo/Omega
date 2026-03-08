"""
Phase 405 — Red Blood Cell Partitioning Model

Predicts drug partitioning into red blood cells and its effect on
blood-to-plasma ratio and free drug concentrations.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RBCPartitioningResult:
    drug_name: str
    logp: float
    pka_base: float  # pKa of basic group (nan if none)
    hematocrit: float  # fraction (typically 0.45)
    kp_rbc: float  # RBC-to-plasma partitioning coefficient
    blood_to_plasma_ratio: float  # Cb/Cp
    fu_plasma: float
    fu_blood: float  # fraction unbound in whole blood
    cl_blood_L_per_h: float  # CL based on blood concentration
    cl_plasma_L_per_h: float  # CL based on plasma concentration
    rbc_sequestration_pct: float  # % of total blood drug in RBCs
    drug_in_plasma_pct: float  # % in plasma water
    hemoglobin_binding: bool  # True if cationic/basic drug likely binds hemoglobin
    clinical_relevance: str  # "high"(Kp>2), "moderate"(1-2), "low"(<1)
    notes: str


def _compute_kp_rbc(logp: float, pka_base: float | None) -> float:
    """Compute RBC-to-plasma partitioning coefficient."""
    # Lipophilic component
    if logp > 0:
        kp_lipid = 10 ** (logp * 0.4)
    else:
        kp_lipid = 0.3

    if pka_base is not None:
        # Ion-trapping for basic drugs
        pH_plasma = 7.4
        pH_rbc = 7.2
        ion_trap = (1 + 10 ** (pka_base - pH_rbc)) / (1 + 10 ** (pka_base - pH_plasma))
        kp_rbc = kp_lipid * ion_trap
    else:
        kp_rbc = kp_lipid

    # Clamp to [0.1, 20.0]
    return max(0.1, min(20.0, kp_rbc))


def _clinical_relevance(kp_rbc: float) -> str:
    if kp_rbc > 2:
        return "high"
    elif kp_rbc > 1:
        return "moderate"
    else:
        return "low"


def _build_notes(
    kp_rbc: float, b_p: float, hemoglobin_binding: bool, pka_base: float | None
) -> str:
    parts = []
    if hemoglobin_binding:
        parts.append("Hemoglobin binding likely due to basic and lipophilic character.")
    if b_p > 1.5:
        parts.append(
            "High B/P ratio: plasma-based CL underestimates in vivo CL; "
            "blood-based measurements recommended."
        )
    elif b_p < 0.7:
        parts.append("Low B/P ratio: drug preferentially remains in plasma.")
    if kp_rbc > 2:
        parts.append("Significant RBC sequestration may prolong apparent half-life.")
    if pka_base is not None and pka_base > 8:
        parts.append("Strongly basic drug: pronounced pH-dependent ion-trapping in RBCs.")
    if not parts:
        parts.append("Minimal RBC partitioning; plasma measurements adequate.")
    return " ".join(parts)


def predict_rbc_partitioning(
    drug_name: str,
    logp: float,
    fu_plasma: float,
    hematocrit: float = 0.45,
    pka_base: float | None = None,
    cl_ref_L_per_h: float = 10.0,
) -> RBCPartitioningResult:
    """
    Predict RBC partitioning and derived PK parameters.

    Parameters
    ----------
    drug_name : str
    logp : float
    fu_plasma : float  fraction unbound in plasma, (0, 1]
    hematocrit : float  (0, 1)
    pka_base : float or None  pKa of basic group
    cl_ref_L_per_h : float  reference plasma CL (L/h)

    Returns
    -------
    RBCPartitioningResult
    """
    # Validation
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if not (0 < hematocrit < 1):
        raise ValueError(f"hematocrit must be in (0, 1), got {hematocrit}")
    if cl_ref_L_per_h <= 0:
        raise ValueError(f"cl_ref_L_per_h must be > 0, got {cl_ref_L_per_h}")

    kp_rbc = _compute_kp_rbc(logp, pka_base)

    # Blood-to-plasma ratio
    b_p = 1 - hematocrit + hematocrit * kp_rbc

    # Unbound fractions
    fu_blood = fu_plasma / b_p

    # Clearance conversion
    cl_plasma = cl_ref_L_per_h
    cl_blood = cl_plasma / b_p

    # Distribution in blood
    denominator = 1 - hematocrit + hematocrit * kp_rbc
    rbc_sequestration_pct = hematocrit * kp_rbc / denominator * 100
    drug_in_plasma_pct = (1 - hematocrit) / denominator * 100

    # Hemoglobin binding: basic drug with high RBC affinity
    hemoglobin_binding = pka_base is not None and pka_base > 6 and kp_rbc > 2

    clinical_rel = _clinical_relevance(kp_rbc)
    notes = _build_notes(kp_rbc, b_p, hemoglobin_binding, pka_base)

    pka_stored = pka_base if pka_base is not None else float("nan")

    return RBCPartitioningResult(
        drug_name=drug_name,
        logp=logp,
        pka_base=pka_stored,
        hematocrit=hematocrit,
        kp_rbc=kp_rbc,
        blood_to_plasma_ratio=b_p,
        fu_plasma=fu_plasma,
        fu_blood=fu_blood,
        cl_blood_L_per_h=cl_blood,
        cl_plasma_L_per_h=cl_plasma,
        rbc_sequestration_pct=rbc_sequestration_pct,
        drug_in_plasma_pct=drug_in_plasma_pct,
        hemoglobin_binding=hemoglobin_binding,
        clinical_relevance=clinical_rel,
        notes=notes,
    )


def screen_hematocrit_effect(
    drug_name: str,
    logp: float,
    fu_plasma: float,
    pka_base: float | None = None,
    hct_values: list | None = None,
) -> list:
    """
    Screen blood-to-plasma ratio across a range of hematocrit values.

    Returns list of RBCPartitioningResult sorted by blood_to_plasma_ratio ascending.
    """
    if hct_values is None:
        hct_values = [0.25, 0.35, 0.45, 0.55, 0.65]

    results = []
    for hct in hct_values:
        result = predict_rbc_partitioning(
            drug_name=drug_name,
            logp=logp,
            fu_plasma=fu_plasma,
            hematocrit=hct,
            pka_base=pka_base,
        )
        results.append(result)

    results.sort(key=lambda r: r.blood_to_plasma_ratio)
    return results
