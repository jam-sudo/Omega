"""Scaffold hop PK impact predictor.

Predicts how switching between chemical scaffolds will alter key
pharmacokinetic properties based on physicochemical descriptor changes.

Scaffold hopping is a medicinal chemistry strategy to modify the core
molecular framework while retaining biological activity. Changes in
MW, logP, PSA, and pKa alter ADME properties in predictable ways.

References
----------
Bohm et al. 2004 (Drug Discov Today), Hajduk & Greer 2007 (Nat Rev Drug Discov).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaffoldHopResult:
    """Result of scaffold hop PK impact prediction."""

    original_name: str
    candidate_name: str
    delta_mw: float
    delta_logP: float
    delta_psa: float
    delta_pka: float
    predicted_vd_change: str
    predicted_absorption_change: str
    predicted_cl_change: str
    risk_flags: tuple[str, ...]
    overall_recommendation: str
    notes: str


def _validate_drug_dict(drug: dict, label: str) -> None:
    """Validate a drug dictionary has required fields with sensible values."""
    required_fields = {"name", "mw", "logP", "psa", "n_hbd", "pka", "molecule_type"}
    missing = required_fields - set(drug.keys())
    if missing:
        raise ValueError(f"{label} drug dict missing fields: {missing}")
    if drug["mw"] <= 0:
        raise ValueError(f"{label} drug mw must be > 0, got {drug['mw']}")
    if drug["psa"] < 0:
        raise ValueError(f"{label} drug psa must be >= 0, got {drug['psa']}")
    if drug["n_hbd"] < 0:
        raise ValueError(f"{label} drug n_hbd must be >= 0, got {drug['n_hbd']}")
    if not drug["name"] or not str(drug["name"]).strip():
        raise ValueError(f"{label} drug name must be a non-empty string")
    if not drug["molecule_type"] or not str(drug["molecule_type"]).strip():
        raise ValueError(f"{label} drug molecule_type must be a non-empty string")


def _predict_vd_change(delta_mw: float, delta_logP: float, delta_psa: float) -> str:
    """Predict direction of Vd change from property deltas."""
    vd_score = 0.0  # positive = increase, negative = decrease

    # Higher logP -> higher Vd (more tissue partitioning)
    if delta_logP > 1.0:
        vd_score += 1.5
    elif delta_logP > 0.5:
        vd_score += 0.5
    elif delta_logP < -1.0:
        vd_score -= 1.5
    elif delta_logP < -0.5:
        vd_score -= 0.5

    # Higher MW -> slightly lower Vd (less membrane permeation)
    if delta_mw > 100:
        vd_score -= 0.5
    elif delta_mw < -100:
        vd_score += 0.3

    # Higher PSA -> lower Vd (more polar, less tissue distribution)
    if delta_psa > 30:
        vd_score -= 0.8
    elif delta_psa < -30:
        vd_score += 0.5

    if vd_score > 0.6:
        return "increase"
    elif vd_score < -0.6:
        return "decrease"
    else:
        return "similar"


def _predict_absorption_change(
    delta_mw: float, delta_logP: float, delta_psa: float, delta_pka: float
) -> str:
    """Predict direction of oral absorption change from property deltas."""
    abs_score = 0.0  # positive = better absorption

    # Larger MW -> worse absorption (Lipinski rule)
    if delta_mw > 100:
        abs_score -= 1.5
    elif delta_mw > 50:
        abs_score -= 0.5
    elif delta_mw < -100:
        abs_score += 0.8

    # logP: moderate range is best; very high logP can impair dissolution
    if 0 < delta_logP <= 2:
        abs_score += 0.3  # slight improvement in membrane permeation
    elif delta_logP > 2:
        abs_score -= 0.5  # too lipophilic, poor aqueous solubility
    elif delta_logP < -1:
        abs_score -= 0.3  # too hydrophilic

    # Higher PSA -> lower intestinal permeability
    if delta_psa > 30:
        abs_score -= 1.2
    elif delta_psa > 10:
        abs_score -= 0.4
    elif delta_psa < -30:
        abs_score += 0.8

    # pKa change: ionisation at GI pH affects absorption
    if abs(delta_pka) > 2:
        abs_score -= 0.5  # significant ionisation state change

    if abs_score > 0.5:
        return "increase"
    elif abs_score < -0.5:
        return "decrease"
    else:
        return "similar"


def _predict_cl_change(delta_logP: float, delta_mw: float, candidate: dict) -> str:
    """Predict direction of clearance change from property deltas."""
    cl_score = 0.0  # positive = higher CL

    # Higher logP -> more CYP metabolism (lipophilic substrates)
    if delta_logP > 1.5:
        cl_score += 1.0
    elif delta_logP > 0.5:
        cl_score += 0.3
    elif delta_logP < -1.5:
        cl_score -= 1.0

    # Higher MW -> lower renal clearance, but potential for biliary
    if delta_mw > 150:
        cl_score += 0.3  # biliary excretion more likely
    elif delta_mw < -100:
        cl_score -= 0.2

    # Candidate-absolute values: very high logP -> high CL (rapid metabolism)
    candidate_logP = float(candidate.get("logP", 0))
    if candidate_logP > 5:
        cl_score += 0.5

    if cl_score > 0.5:
        return "increase"
    elif cl_score < -0.5:
        return "decrease"
    else:
        return "similar"


def _assess_risk_flags(
    original: dict,
    candidate: dict,
    delta_mw: float,
    delta_logP: float,
    delta_psa: float,
    delta_pka: float,
) -> tuple[str, ...]:
    """Identify risk flags from descriptor changes and absolute values."""
    flags: list[str] = []

    if delta_mw > 100:
        flags.append("large MW increase (>100 Da): absorption may decrease")

    candidate_logP = float(candidate.get("logP", 0))
    if candidate_logP > 5:
        flags.append(f"candidate logP={candidate_logP:.1f} exceeds 5: high lipophilicity risk")

    candidate_psa = float(candidate.get("psa", 0))
    if candidate_psa < 40:
        flags.append(f"candidate PSA={candidate_psa:.1f} < 40 A^2: potential CNS penetration")

    candidate_mw = float(candidate.get("mw", 0))
    if candidate_mw > 500:
        flags.append(f"candidate MW={candidate_mw:.1f} > 500 Da: violates Lipinski Ro5")

    candidate_n_hbd = int(candidate.get("n_hbd", 0))
    if candidate_n_hbd > 5:
        flags.append(f"candidate n_HBD={candidate_n_hbd} > 5: Lipinski violation")

    if abs(delta_pka) > 2:
        flags.append(f"delta_pKa={delta_pka:.1f}: significant ionisation state shift")

    if delta_psa > 60:
        flags.append(f"large PSA increase ({delta_psa:.1f} A^2): significant permeability decrease")

    orig_logP = float(original.get("logP", 0))
    if orig_logP <= 5 and candidate_logP > 5:
        flags.append("scaffold hop crosses logP=5 boundary: increased metabolic lability")

    return tuple(flags)


def _overall_recommendation(
    risk_flags: tuple[str, ...],
    vd_change: str,
    abs_change: str,
    cl_change: str,
    delta_mw: float,
    delta_logP: float,
    delta_psa: float,
    candidate: dict,
) -> str:
    """Determine overall recommendation based on predicted PK changes and risk flags."""
    n_flags = len(risk_flags)

    # Immediate disqualifiers
    candidate_logP = float(candidate.get("logP", 0))
    candidate_mw = float(candidate.get("mw", 0))
    candidate_psa = float(candidate.get("psa", 0))

    if candidate_logP > 7 or candidate_mw > 700:
        return "poor"

    if n_flags >= 4:
        return "risky"

    # Score favorable vs unfavorable changes
    score = 0
    if abs_change == "increase":
        score += 2
    elif abs_change == "similar":
        score += 1
    elif abs_change == "decrease":
        score -= 2

    if vd_change == "similar":
        score += 1
    elif vd_change == "decrease" and delta_logP > 0:
        score -= 1

    if cl_change == "similar":
        score += 1
    elif cl_change == "decrease":
        score += 1  # lower CL often favorable (longer half-life)
    elif cl_change == "increase":
        score -= 1

    # Lipinski compliance bonus
    if candidate_mw <= 500 and candidate_logP <= 5 and candidate_psa >= 40:
        score += 1

    if n_flags == 0 and score >= 4:
        return "favorable"
    elif n_flags <= 1 and score >= 2:
        return "acceptable"
    elif n_flags >= 3 or score <= -1:
        return "risky"
    else:
        return "acceptable"


def predict_scaffold_hop_impact(
    original_drug: dict,
    candidate_drug: dict,
) -> ScaffoldHopResult:
    """Predict PK property changes when switching between chemical scaffolds.

    Each drug dict must contain:
    - name (str): drug identifier
    - mw (float): molecular weight (Da)
    - logP (float): lipophilicity
    - psa (float): polar surface area (A^2)
    - n_hbd (int): number of H-bond donors
    - pka (float): ionisation constant
    - molecule_type (str): e.g. "small_molecule", "peptide", "macrocycle"

    Parameters
    ----------
    original_drug : dict
        Reference (current) drug properties.
    candidate_drug : dict
        New scaffold candidate properties.

    Returns
    -------
    ScaffoldHopResult
    """
    _validate_drug_dict(original_drug, "original")
    _validate_drug_dict(candidate_drug, "candidate")

    delta_mw = float(candidate_drug["mw"]) - float(original_drug["mw"])
    delta_logP = float(candidate_drug["logP"]) - float(original_drug["logP"])
    delta_psa = float(candidate_drug["psa"]) - float(original_drug["psa"])
    delta_pka = float(candidate_drug["pka"]) - float(original_drug["pka"])

    vd_change = _predict_vd_change(delta_mw, delta_logP, delta_psa)
    abs_change = _predict_absorption_change(delta_mw, delta_logP, delta_psa, delta_pka)
    cl_change = _predict_cl_change(delta_logP, delta_mw, candidate_drug)

    risk_flags = _assess_risk_flags(
        original_drug, candidate_drug, delta_mw, delta_logP, delta_psa, delta_pka
    )

    recommendation = _overall_recommendation(
        risk_flags, vd_change, abs_change, cl_change,
        delta_mw, delta_logP, delta_psa, candidate_drug,
    )

    # Build notes
    note_parts = [
        f"delta_MW={delta_mw:+.1f} Da",
        f"delta_logP={delta_logP:+.2f}",
        f"delta_PSA={delta_psa:+.1f} A^2",
        f"delta_pKa={delta_pka:+.2f}",
        f"Vd:{vd_change}, Absorption:{abs_change}, CL:{cl_change}",
    ]
    if risk_flags:
        note_parts.append(f"{len(risk_flags)} risk flag(s) identified")
    notes = "; ".join(note_parts)

    return ScaffoldHopResult(
        original_name=str(original_drug["name"]),
        candidate_name=str(candidate_drug["name"]),
        delta_mw=delta_mw,
        delta_logP=delta_logP,
        delta_psa=delta_psa,
        delta_pka=delta_pka,
        predicted_vd_change=vd_change,
        predicted_absorption_change=abs_change,
        predicted_cl_change=cl_change,
        risk_flags=risk_flags,
        overall_recommendation=recommendation,
        notes=notes,
    )


__all__ = [
    "ScaffoldHopResult",
    "predict_scaffold_hop_impact",
]
