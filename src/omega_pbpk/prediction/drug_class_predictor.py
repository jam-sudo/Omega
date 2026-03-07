"""
Phase 982 — Drug Class Predictor (prediction/drug_class_predictor.py)

Predicts drug pharmacological class from physicochemical descriptors using
rule-based scoring. Returns probability distribution over 9 drug classes.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_CLASSES = (
    "CNS",
    "cardiovascular",
    "antibiotic",
    "NSAID",
    "antifungal",
    "diuretic",
    "anticancer",
    "antihistamine",
    "other",
)

_VALID_IONIZATION = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class DrugClassPredictionResult:
    """Result of a drug class prediction."""

    drug_name: str
    predicted_class: str
    class_probabilities: dict
    bcs_class: str
    lipinski_pass: bool
    veber_pass: bool
    drug_likeness_score: float
    notes: str


def _compute_bcs_class(logp: float, psa: float, mw: float) -> str:
    """Determine BCS class from logP, PSA, and MW."""
    high_perm = psa < 90  # high permeability proxy
    high_sol = logp < 1 or (psa >= 90)  # high solubility proxy (rough)

    # More granular classification
    # I: high sol + high perm
    # II: low sol + high perm
    # III: high sol + low perm
    # IV: low sol + low perm
    if logp < 1 and psa < 90 and mw < 500:
        return "I"
    elif logp > 2 and psa < 90:
        return "II"
    elif logp < 0 and psa > 90:
        return "III"
    elif logp > 2 and psa > 120:
        return "IV"
    elif high_sol and high_perm:
        return "I"
    elif not high_sol and high_perm:
        return "II"
    elif high_sol and not high_perm:
        return "III"
    else:
        return "IV"


def predict_drug_class(
    drug_name: str,
    mw: float,
    logp: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    psa: float = 80.0,
    hbd: int = 2,
    hba: int = 4,
    rotbonds: int = 5,
) -> DrugClassPredictionResult:
    """Predict pharmacological drug class from physicochemical descriptors.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    mw:
        Molecular weight (Da).
    logp:
        Octanol-water partition coefficient.
    pka:
        Acid/base dissociation constant.
    ionization_type:
        Ionization type: "acid", "base", or "neutral".
    psa:
        Polar surface area (A^2).
    hbd:
        Number of hydrogen bond donors.
    hba:
        Number of hydrogen bond acceptors.
    rotbonds:
        Number of rotatable bonds.

    Returns
    -------
    DrugClassPredictionResult
    """
    # --- Validation ---
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if hbd < 0:
        raise ValueError("hbd must be >= 0")
    if hba < 0:
        raise ValueError("hba must be >= 0")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(f"ionization_type must be one of {_VALID_IONIZATION}")

    # --- Score each class (0-10) ---
    scores: dict = {}

    # CNS: logP 1-4, MW 150-400, PSA < 90, basic pKa > 7
    cns = 0.0
    if 1.0 <= logp <= 4.0:
        cns += 3
    if 150.0 <= mw <= 400.0:
        cns += 2
    if psa < 90.0:
        cns += 3
    if ionization_type == "base" and pka > 7.0:
        cns += 2
    scores["CNS"] = cns

    # Cardiovascular: logP 0-3, MW 200-500
    cv = 0.0
    if 0.0 <= logp <= 3.0:
        cv += 3
    if 200.0 <= mw <= 500.0:
        cv += 3
    if 40.0 <= psa <= 120.0:
        cv += 2
    if ionization_type in {"base", "neutral"}:
        cv += 2
    scores["cardiovascular"] = cv

    # Antibiotic: MW > 300, HBA > 4, PSA > 80, logP < 2
    ab = 0.0
    if mw > 300.0:
        ab += 3
    if hba > 4:
        ab += 3
    if psa > 80.0:
        ab += 2
    if logp < 2.0:
        ab += 2
    scores["antibiotic"] = ab

    # NSAID: acidic (pKa 3-6), logP 1-4, MW 150-350
    nsaid = 0.0
    if ionization_type == "acid" and 3.0 <= pka <= 6.0:
        nsaid += 4
    if 1.0 <= logp <= 4.0:
        nsaid += 3
    if 150.0 <= mw <= 350.0:
        nsaid += 3
    scores["NSAID"] = nsaid

    # Antifungal: logP > 3, MW 300-500, basic, HBA > 3
    af = 0.0
    if logp > 3.0:
        af += 3
    if 300.0 <= mw <= 500.0:
        af += 3
    if ionization_type == "base":
        af += 2
    if hba > 3:
        af += 2
    scores["antifungal"] = af

    # Diuretic: low logP < 1, acidic (pKa 6-10), PSA > 100
    di = 0.0
    if logp < 1.0:
        di += 4
    if ionization_type == "acid" and 6.0 <= pka <= 10.0:
        di += 3
    if psa > 100.0:
        di += 3
    scores["diuretic"] = di

    # Anticancer: MW > 400, HBA > 5, HBD > 2, rotbonds > 5
    ac = 0.0
    if mw > 400.0:
        ac += 3
    if hba > 5:
        ac += 3
    if hbd > 2:
        ac += 2
    if rotbonds > 5:
        ac += 2
    scores["anticancer"] = ac

    # Antihistamine: basic (pKa 8-10), logP 2-4, MW 200-400
    ah = 0.0
    if ionization_type == "base" and 8.0 <= pka <= 10.0:
        ah += 4
    if 2.0 <= logp <= 4.0:
        ah += 3
    if 200.0 <= mw <= 400.0:
        ah += 3
    scores["antihistamine"] = ah

    # Other: baseline
    scores["other"] = 1.0

    # --- Normalize scores to probabilities ---
    total_score = sum(scores.values())
    if total_score <= 0.0:
        # Uniform fallback
        n = len(_VALID_CLASSES)
        class_probabilities = {cls: 1.0 / n for cls in _VALID_CLASSES}
    else:
        class_probabilities = {cls: scores[cls] / total_score for cls in _VALID_CLASSES}

    # --- Predicted class (argmax) ---
    predicted_class = max(class_probabilities, key=lambda k: class_probabilities[k])

    # --- BCS class ---
    bcs_class = _compute_bcs_class(logp, psa, mw)

    # --- Lipinski rule of five ---
    lipinski_pass = bool(mw < 500 and logp < 5 and hbd < 5 and hba < 10)

    # --- Veber rules ---
    veber_pass = bool(psa < 140 and rotbonds < 10)

    # --- Drug-likeness score (0-100) ---
    logp_ok = 1 if (0.0 <= logp <= 5.0) else 0
    drug_likeness_score = 100.0 * (
        (1 if lipinski_pass else 0) * 0.4 + (1 if veber_pass else 0) * 0.3 + logp_ok * 0.3
    )

    notes = (
        f"predicted_class={predicted_class}, BCS={bcs_class}, "
        f"Lipinski={'pass' if lipinski_pass else 'fail'}, "
        f"Veber={'pass' if veber_pass else 'fail'}, "
        f"drug_likeness={drug_likeness_score:.1f}"
    )

    return DrugClassPredictionResult(
        drug_name=drug_name,
        predicted_class=predicted_class,
        class_probabilities=class_probabilities,
        bcs_class=bcs_class,
        lipinski_pass=lipinski_pass,
        veber_pass=veber_pass,
        drug_likeness_score=drug_likeness_score,
        notes=notes,
    )


def screen_drug_classes(compounds: list) -> list:
    """Screen a list of compound parameter dicts and return predictions sorted
    by drug_likeness_score descending.

    Each dict must contain at minimum ``drug_name`` and ``mw``; remaining
    keys are forwarded to ``predict_drug_class`` as keyword arguments.

    Parameters
    ----------
    compounds:
        List of dicts with drug parameters.

    Returns
    -------
    List of DrugClassPredictionResult sorted by drug_likeness_score descending.
    """
    results = []
    for compound in compounds:
        params = dict(compound)
        drug_name = params.pop("drug_name")
        mw = params.pop("mw")
        result = predict_drug_class(drug_name=drug_name, mw=mw, **params)
        results.append(result)

    results.sort(key=lambda r: r.drug_likeness_score, reverse=True)
    return results
