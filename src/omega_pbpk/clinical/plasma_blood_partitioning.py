"""Plasma/blood partitioning: blood-to-plasma ratio (Rb) and its effect on CL/Vd."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BloodPartitioningResult:
    drug_name: str
    logP: float
    fu_plasma: float
    rb: float
    cl_plasma_L_per_h: float
    cl_blood_L_per_h: float
    vd_plasma_L: float
    vd_blood_L: float
    t_half_plasma_h: float
    t_half_blood_h: float
    notes: str


def predict_rb(logP: float, fu_plasma: float, hematocrit: float = 0.45) -> float:
    """Predict blood-to-plasma concentration ratio (Rb).

    Uses empirical model where lipophilic drugs partition into RBCs.

    Parameters
    ----------
    logP:
        Lipophilicity (octanol-water partition coefficient).
    fu_plasma:
        Fraction unbound in plasma (0, 1].
    hematocrit:
        Fraction of blood that is red blood cells (0, 1). Default 0.45.

    Returns
    -------
    float
        Rb = blood concentration / plasma concentration, clamped to [0.3, 5.0].
    """
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if not (0 < hematocrit < 1):
        raise ValueError(f"hematocrit must be in (0, 1), got {hematocrit}")

    fu_rbc = 1.0 / (1.0 + 10 ** (0.5 * logP))
    rb = (1.0 - hematocrit) + hematocrit * (fu_plasma / fu_rbc)
    # Clamp to physiologically plausible range
    return max(0.3, min(5.0, rb))


def correct_cl_for_blood(cl_plasma_L_per_h: float, rb: float) -> float:
    """Convert plasma clearance to blood clearance.

    CL_blood = CL_plasma / Rb

    More physiologically relevant for flow-limited drugs.
    """
    if rb <= 0:
        raise ValueError(f"rb must be positive, got {rb}")
    return cl_plasma_L_per_h / rb


def correct_vd_for_blood(
    vd_plasma_L: float,
    rb: float,
    fu_plasma: float,
    fu_blood: float | None = None,
) -> float:
    """Convert plasma volume of distribution to blood-based Vd.

    fu_blood = fu_plasma / Rb (if not provided).
    Vd_blood = Vd_plasma * fu_plasma / fu_blood = Vd_plasma * Rb.
    """
    if rb <= 0:
        raise ValueError(f"rb must be positive, got {rb}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if fu_blood is None:
        fu_blood = fu_plasma / rb
    if fu_blood <= 0:
        raise ValueError(f"fu_blood must be positive, got {fu_blood}")
    return vd_plasma_L * (fu_plasma / fu_blood)


def blood_partitioning_impact(
    drug_name: str,
    logP: float,
    fu_plasma: float,
    cl_plasma_L_per_h: float,
    vd_plasma_L: float,
    hematocrit: float = 0.45,
) -> BloodPartitioningResult:
    """Compute complete blood/plasma partitioning impact on PK parameters.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Lipophilicity.
    fu_plasma:
        Fraction unbound in plasma (0, 1].
    cl_plasma_L_per_h:
        Plasma clearance (L/h).
    vd_plasma_L:
        Volume of distribution based on plasma (L).
    hematocrit:
        Fractional hematocrit. Default 0.45.

    Returns
    -------
    BloodPartitioningResult
    """
    rb = predict_rb(logP, fu_plasma, hematocrit)
    cl_blood = correct_cl_for_blood(cl_plasma_L_per_h, rb)
    vd_blood = correct_vd_for_blood(vd_plasma_L, rb, fu_plasma)

    # t_half = 0.693 * Vd / CL
    t_half_plasma = (
        (0.693 * vd_plasma_L / cl_plasma_L_per_h) if cl_plasma_L_per_h > 0 else float("inf")
    )
    t_half_blood = (0.693 * vd_blood / cl_blood) if cl_blood > 0 else float("inf")

    if rb > 1.5:
        notes = "Drug significantly partitions into RBCs; plasma measurements underestimate blood exposure."
    elif rb < 0.7:
        notes = (
            "Drug preferentially stays in plasma; blood measurements underestimate plasma exposure."
        )
    else:
        notes = "Moderate blood-plasma partitioning; plasma and blood measurements are broadly comparable."

    return BloodPartitioningResult(
        drug_name=drug_name,
        logP=logP,
        fu_plasma=fu_plasma,
        rb=rb,
        cl_plasma_L_per_h=cl_plasma_L_per_h,
        cl_blood_L_per_h=cl_blood,
        vd_plasma_L=vd_plasma_L,
        vd_blood_L=vd_blood,
        t_half_plasma_h=t_half_plasma,
        t_half_blood_h=t_half_blood,
        notes=notes,
    )


def compare_oral_iv_plasma_blood(
    iv_auc_blood: float,
    oral_auc_plasma: float,
    dose_iv: float,
    dose_oral: float,
    rb: float,
) -> dict:
    """Estimate oral bioavailability correcting for plasma/blood measurement mismatch.

    IV AUC is measured in blood; oral AUC is measured in plasma.
    Converts oral plasma AUC to blood-equivalent before computing F.

    Parameters
    ----------
    iv_auc_blood:
        IV AUC measured in blood (mg*h/L).
    oral_auc_plasma:
        Oral AUC measured in plasma (mg*h/L).
    dose_iv:
        IV dose (mg).
    dose_oral:
        Oral dose (mg).
    rb:
        Blood-to-plasma ratio.

    Returns
    -------
    dict with keys: F_pct, auc_blood_oral, correction_note.
    """
    if rb <= 0:
        raise ValueError(f"rb must be positive, got {rb}")
    if dose_iv <= 0 or dose_oral <= 0:
        raise ValueError("Doses must be positive")

    # Convert oral plasma AUC to blood AUC
    # C_blood = C_plasma * Rb  → AUC_blood = AUC_plasma * Rb
    auc_blood_oral = oral_auc_plasma * rb

    F = (auc_blood_oral / dose_oral) / (iv_auc_blood / dose_iv)
    F_pct = F * 100.0

    correction_note = (
        f"Oral plasma AUC converted to blood AUC by multiplying by Rb={rb:.3f}. "
        f"Uncorrected F would be {(oral_auc_plasma / dose_oral) / (iv_auc_blood / dose_iv) * 100:.1f}%."
    )

    return {
        "F_pct": F_pct,
        "auc_blood_oral": auc_blood_oral,
        "correction_note": correction_note,
    }
