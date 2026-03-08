"""Phase 529 — Urinary pH Manipulation and Drug Excretion.

Models the effect of urinary pH manipulation on renal drug excretion using
Henderson-Hasselbalch ionization principles.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class UrinaryPHResult:
    """Result of urinary pH manipulation analysis."""

    drug_name: str
    pKa: float
    drug_type: str  # "acid", "base", or "neutral"
    urine_pH: float
    ionized_fraction: float
    f_reabs: float
    cl_renal_L_per_h: float
    fe_renal: float
    t_half_h: float
    alkalization_benefit: bool
    clinical_application: str
    notes: str


def _ionized_fraction(pKa: float, urine_pH: float, drug_type: str) -> float:
    """Compute ionized fraction via Henderson-Hasselbalch."""
    if drug_type == "neutral":
        return 0.0
    if drug_type == "acid":
        exponent = urine_pH - pKa
        # Guard against overflow
        if exponent > 700:
            return 1.0
        if exponent < -700:
            return 0.0
        val = 10.0**exponent
        return val / (1.0 + val)
    # base
    exponent = pKa - urine_pH
    if exponent > 700:
        return 1.0
    if exponent < -700:
        return 0.0
    val = 10.0**exponent
    return val / (1.0 + val)


def _clinical_application(pKa: float, drug_type: str) -> str:
    if drug_type == "acid" and 3.0 <= pKa <= 5.0:
        return "salicylate_overdose"
    if drug_type == "base" and 9.0 <= pKa <= 10.0:
        return "amphetamine_overdose"
    return "general"


def predict_urinary_excretion(
    drug_name: str,
    pKa: float,
    drug_type: str,
    urine_pH: float,
    fup: float = 0.1,
    vd_L: float = 50.0,
    cl_nonrenal_L_per_h: float = 5.0,
    cl_secretion_L_per_h: float = 0.0,
    gfr_L_per_h: float = 7.2,
) -> UrinaryPHResult:
    """Predict urinary drug excretion at a given urine pH.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    pKa:
        Acid dissociation constant.
    drug_type:
        One of "acid", "base", or "neutral".
    urine_pH:
        Urine pH (must be between 1 and 9 inclusive).
    fup:
        Fraction unbound in plasma (0 < fup <= 1).
    vd_L:
        Volume of distribution in litres (must be > 0).
    cl_nonrenal_L_per_h:
        Non-renal clearance in L/h (must be > 0).
    cl_secretion_L_per_h:
        Active tubular secretion clearance in L/h.
    gfr_L_per_h:
        Glomerular filtration rate in L/h (default 7.2).

    Returns
    -------
    UrinaryPHResult
    """
    if not (0 < fup <= 1):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cl_nonrenal_L_per_h <= 0:
        raise ValueError(f"cl_nonrenal_L_per_h must be > 0, got {cl_nonrenal_L_per_h}")
    if not (1.0 <= urine_pH <= 9.0):
        raise ValueError(f"urine_pH must be in [1, 9], got {urine_pH}")
    if drug_type not in {"acid", "base", "neutral"}:
        raise ValueError(f"drug_type must be 'acid', 'base', or 'neutral', got {drug_type!r}")

    ion_frac = _ionized_fraction(pKa, urine_pH, drug_type)

    # Reabsorption only for non-ionized fraction; max 95%
    f_reabs = 0.95 * (1.0 - ion_frac)
    f_reabs = max(0.0, min(0.95, f_reabs))

    # Effective renal clearance
    cl_renal = gfr_L_per_h * fup * (1.0 - f_reabs) + cl_secretion_L_per_h

    # Total clearance
    cl_total = cl_renal + cl_nonrenal_L_per_h

    fe = cl_renal / cl_total if cl_total > 0 else 0.0
    t_half = 0.693 * vd_L / cl_total if cl_total > 0 else float("inf")

    alkalization_benefit = drug_type == "acid"

    clinical_app = _clinical_application(pKa, drug_type)

    if drug_type == "neutral":
        notes = (
            "Neutral drug: ionization unaffected by pH; "
            "urinary pH manipulation has minimal effect on excretion."
        )
    elif drug_type == "acid":
        notes = (
            "Acid drug: alkalinizing urine (e.g., NaHCO3) increases ionization "
            "and reduces reabsorption, enhancing renal excretion."
        )
    else:
        notes = (
            "Basic drug: acidifying urine (e.g., NH4Cl) increases ionization "
            "and reduces reabsorption, enhancing renal excretion."
        )

    return UrinaryPHResult(
        drug_name=drug_name,
        pKa=pKa,
        drug_type=drug_type,
        urine_pH=urine_pH,
        ionized_fraction=ion_frac,
        f_reabs=f_reabs,
        cl_renal_L_per_h=cl_renal,
        fe_renal=fe,
        t_half_h=t_half,
        alkalization_benefit=alkalization_benefit,
        clinical_application=clinical_app,
        notes=notes,
    )


def compare_urine_ph(
    drug_name: str,
    pKa: float,
    drug_type: str,
    ph_values: list,
) -> list:
    """Compare urinary excretion across multiple urine pH values.

    Returns results sorted by cl_renal_L_per_h descending (highest excretion first).
    """
    results = [predict_urinary_excretion(drug_name, pKa, drug_type, ph) for ph in ph_values]
    results.sort(key=lambda r: r.cl_renal_L_per_h, reverse=True)
    return results
