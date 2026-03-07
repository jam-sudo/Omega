"""Phase 996 — Metabolic stability prediction from physicochemical descriptors.

Predicts intrinsic clearance, microsomal half-life, and hepatic extraction
ratio using microsomal stability rules (Egan 1999, Austin 2002 approaches).
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_IONIZATION = {"acid", "base", "neutral"}
_VALID_PATHWAYS = {"CYP3A4", "CYP2D6", "CYP2C9", "esterase", "other"}
_VALID_CLASSES = {"stable", "moderate", "unstable"}

# Hepatic blood flow: ~20.7 L/min for 70 kg human
_HBF_L_MIN = 20.7
_BODY_WEIGHT_KG = 70.0
# Equivalent hepatic flow in µL/min/mg protein (well-stirred model constant)
_Q_H_EQUIV = 1500.0  # µL/min/mg protein


@dataclass(frozen=True)
class MetabolicStabilityResult:
    """Predicted metabolic stability parameters."""

    drug_name: str
    logp: float
    mw: float
    psa: float
    clint_uL_min_per_mg_protein: float
    t_half_min: float
    extraction_ratio: float
    fu_mic: float
    predicted_cls_mL_min_per_kg: float
    stability_class: str
    major_metabolic_pathway: str
    notes: str


def predict_metabolic_stability(
    drug_name: str,
    logp: float,
    mw: float,
    psa: float = 80.0,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    has_ester: bool = False,
    has_aromatic_amine: bool = False,
) -> MetabolicStabilityResult:
    """Predict metabolic stability from physicochemical descriptors.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logp:
        Octanol-water partition coefficient (log scale).
    mw:
        Molecular weight in g/mol.
    psa:
        Polar surface area in Å².
    pka:
        Ionisation constant.
    ionization_type:
        One of "acid", "base", or "neutral".
    has_ester:
        Whether the compound contains an ester group.
    has_aromatic_amine:
        Whether the compound contains an aromatic amine.

    Returns
    -------
    MetabolicStabilityResult
    """
    # --- Validation ---
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got {ionization_type!r}"
        )

    # --- fu_mic (Austin 2002) ---
    log_fu_mic = -0.072 * logp + 0.067
    fu_mic = min(1.0, max(0.01, 10.0**log_fu_mic))

    # --- Intrinsic clearance (µL/min/mg protein) ---
    clint = 5.0  # base value

    if logp > 2:
        clint += 30.0  # lipophilic → more CYP metabolism
    if ionization_type == "base" and pka > 7:
        clint += 20.0  # basic amines: CYP2D6
    if mw < 300:
        clint += 15.0  # small molecules: faster metabolism
    if has_ester:
        clint += 25.0  # esterase hydrolysis is fast
    if psa > 120:
        clint -= 10.0  # polar → less CYP binding

    # Apply bounds
    clint = min(200.0, max(2.0, clint))

    # --- Microsomal half-life ---
    t_half_min = 450.0 / clint

    # --- Hepatic extraction ratio (well-stirred model) ---
    cl_int_times_fu = clint * fu_mic
    cl_blood_mL_min = (cl_int_times_fu * _Q_H_EQUIV) / (cl_int_times_fu + _Q_H_EQUIV)
    extraction_ratio = cl_blood_mL_min / _Q_H_EQUIV
    extraction_ratio = min(1.0, max(0.0, extraction_ratio))

    # --- Scaled in vivo CL (mL/min/kg) ---
    cls_mL_min_per_kg = extraction_ratio * _HBF_L_MIN * 1000.0 / _BODY_WEIGHT_KG
    cls_mL_min_per_kg = min(30.0, cls_mL_min_per_kg)

    # --- Stability class ---
    if t_half_min > 60:
        stability_class = "stable"
    elif t_half_min >= 30:
        stability_class = "moderate"
    else:
        stability_class = "unstable"

    # --- Major metabolic pathway (priority order) ---
    if has_ester:
        major_metabolic_pathway = "esterase"
    elif ionization_type == "base" and pka > 7:
        major_metabolic_pathway = "CYP2D6"
    elif ionization_type == "acid" and 3.0 <= pka <= 6.0:
        major_metabolic_pathway = "CYP2C9"
    elif logp > 2:
        major_metabolic_pathway = "CYP3A4"
    else:
        major_metabolic_pathway = "other"

    notes = (
        f"fu_mic={fu_mic:.3f}, CLint={clint:.1f} µL/min/mg, "
        f"E={extraction_ratio:.3f}, t½={t_half_min:.1f} min"
    )

    return MetabolicStabilityResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        psa=psa,
        clint_uL_min_per_mg_protein=clint,
        t_half_min=t_half_min,
        extraction_ratio=extraction_ratio,
        fu_mic=fu_mic,
        predicted_cls_mL_min_per_kg=cls_mL_min_per_kg,
        stability_class=stability_class,
        major_metabolic_pathway=major_metabolic_pathway,
        notes=notes,
    )


def screen_metabolic_stability(compounds: list) -> list:
    """Screen a list of compound dicts and return results sorted by t½ descending.

    Each element of ``compounds`` must be a dict with the key ``drug_name``
    and any keyword arguments accepted by :func:`predict_metabolic_stability`.

    Returns a list of :class:`MetabolicStabilityResult` objects sorted from
    most stable (highest t½) to least stable.
    """
    results: list[MetabolicStabilityResult] = []
    for compound in compounds:
        kwargs = dict(compound)
        name = kwargs.pop("drug_name")
        logp_val = kwargs.pop("logp")
        mw_val = kwargs.pop("mw")
        result = predict_metabolic_stability(name, logp_val, mw_val, **kwargs)
        results.append(result)

    results.sort(key=lambda r: r.t_half_min, reverse=True)
    return results
