"""Drug-Excipient Compatibility Predictor — Phase 1069.

Predicts physical/chemical compatibility between API and excipient using
rule-based mechanisms:

- Acid-base reaction: |pKa_drug - pKa_excipient| < 2  → compatibility issue
- Maillard reaction: amine drug + reducing sugar excipient → risk
- Oxidation: oxidation-sensitive drug + high-peroxide excipient → risk
- Hygroscopicity: hygroscopic drug + hygroscopic excipient → moisture-induced
  degradation risk

Compatibility score: 0-100 (100 = fully compatible)
risk_category: "compatible", "minor_concern", "moderate_concern", "incompatible"
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DrugExcipientCompatibilityResult",
    "predict_compatibility",
    "screen_excipients",
]

# ---------------------------------------------------------------------------
# Penalty weights
# ---------------------------------------------------------------------------

_PENALTY_ACID_BASE = 30.0  # |pKa_drug - pKa_excipient| < 2
_PENALTY_MAILLARD = 35.0  # amine + reducing sugar
_PENALTY_OXIDATION_MEDIUM = 20.0  # oxidation-sensitive + medium peroxide excipient
_PENALTY_OXIDATION_HIGH = 40.0  # oxidation-sensitive + high peroxide excipient
_PENALTY_HYGROSCOPIC = 20.0  # both drug and excipient hygroscopic

_ACID_BASE_PKA_THRESHOLD = 2.0  # delta pKa below which acid-base concern triggers


def _risk_category(score: float) -> str:
    """Map compatibility score to risk category."""
    if score >= 85.0:
        return "compatible"
    if score >= 65.0:
        return "minor_concern"
    if score >= 40.0:
        return "moderate_concern"
    return "incompatible"


def _recommended_action(
    acid_base_risk: bool,
    maillard_risk: bool,
    oxidation_risk: bool,
    hygroscopic_risk: bool,
    score: float,
) -> str:
    """Generate a recommended action based on risk flags and score."""
    actions: list[str] = []
    if maillard_risk:
        actions.append("avoid reducing-sugar excipient or protect with antioxidant packaging")
    if oxidation_risk:
        actions.append("use low-peroxide excipient grade and nitrogen purge")
    if acid_base_risk:
        actions.append("buffer formulation to mitigate acid-base interaction")
    if hygroscopic_risk:
        actions.append("use desiccant and moisture-barrier packaging")
    if not actions:
        return "No action required; standard manufacturing conditions acceptable."
    return "; ".join(a.capitalize() for a in actions) + "."


@dataclass(frozen=True)
class DrugExcipientCompatibilityResult:
    """Result of drug-excipient compatibility prediction (Phase 1069).

    Attributes
    ----------
    drug_name:
        Name or identifier of the API.
    excipient_name:
        Name or identifier of the excipient.
    compatibility_score:
        Numeric compatibility score in [0, 100] where 100 = fully compatible.
    risk_category:
        Qualitative risk level: "compatible", "minor_concern",
        "moderate_concern", or "incompatible".
    maillard_risk:
        True if Maillard reaction risk is predicted (amine + reducing sugar).
    oxidation_risk:
        True if oxidation risk is predicted (oxidation-sensitive drug +
        medium/high-peroxide excipient).
    acid_base_risk:
        True if acid-base interaction is predicted (|pKa_drug - pKa_excipient| < 2).
    hygroscopic_risk:
        True if moisture-induced degradation risk is predicted (both components
        are hygroscopic).
    recommended_action:
        Plain-text recommendation based on identified risks.
    notes:
        Additional notes summarising the assessment.
    """

    drug_name: str
    excipient_name: str
    compatibility_score: float
    risk_category: str
    maillard_risk: bool
    oxidation_risk: bool
    acid_base_risk: bool
    hygroscopic_risk: bool
    recommended_action: str
    notes: str


def predict_compatibility(
    drug_name: str,
    excipient_name: str,
    drug_pka: float = 7.0,
    excipient_pka: float | None = None,
    drug_has_amine: bool = False,
    excipient_is_reducing_sugar: bool = False,
    drug_oxidation_sensitive: bool = False,
    excipient_peroxide_level: str = "low",
    drug_hygroscopic: bool = False,
    excipient_hygroscopic: bool = False,
) -> DrugExcipientCompatibilityResult:
    """Predict drug-excipient compatibility from physicochemical properties.

    Parameters
    ----------
    drug_name:
        Identifier for the API.
    excipient_name:
        Identifier for the excipient.
    drug_pka:
        Most relevant pKa of the drug. Default 7.0.
    excipient_pka:
        Most relevant pKa of the excipient. ``None`` means no acid-base concern.
        Default ``None``.
    drug_has_amine:
        True if the drug contains a primary or secondary amine group.
    excipient_is_reducing_sugar:
        True if the excipient is a reducing sugar (e.g. lactose, glucose).
    drug_oxidation_sensitive:
        True if the drug is susceptible to oxidative degradation.
    excipient_peroxide_level:
        Residual peroxide content of excipient: ``"low"``, ``"medium"``, or
        ``"high"``. Default ``"low"``.
    drug_hygroscopic:
        True if the drug is hygroscopic.
    excipient_hygroscopic:
        True if the excipient is hygroscopic.

    Returns
    -------
    DrugExcipientCompatibilityResult

    Raises
    ------
    ValueError
        If *drug_name* or *excipient_name* is empty.
    """
    if not drug_name:
        raise ValueError("drug_name must not be empty.")
    if not excipient_name:
        raise ValueError("excipient_name must not be empty.")

    peroxide_level = excipient_peroxide_level.lower()

    # ------------------------------------------------------------------
    # Evaluate individual risks
    # ------------------------------------------------------------------

    # 1. Acid-base risk
    acid_base_risk = False
    ab_penalty = 0.0
    if excipient_pka is not None:
        delta_pka = abs(drug_pka - excipient_pka)
        if delta_pka < _ACID_BASE_PKA_THRESHOLD:
            acid_base_risk = True
            ab_penalty = _PENALTY_ACID_BASE

    # 2. Maillard risk
    maillard_risk = drug_has_amine and excipient_is_reducing_sugar
    maillard_penalty = _PENALTY_MAILLARD if maillard_risk else 0.0

    # 3. Oxidation risk
    oxidation_risk = False
    ox_penalty = 0.0
    if drug_oxidation_sensitive:
        if peroxide_level == "high":
            oxidation_risk = True
            ox_penalty = _PENALTY_OXIDATION_HIGH
        elif peroxide_level == "medium":
            oxidation_risk = True
            ox_penalty = _PENALTY_OXIDATION_MEDIUM

    # 4. Hygroscopic risk
    hygroscopic_risk = drug_hygroscopic and excipient_hygroscopic
    hygro_penalty = _PENALTY_HYGROSCOPIC if hygroscopic_risk else 0.0

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------
    total_penalty = ab_penalty + maillard_penalty + ox_penalty + hygro_penalty
    score = max(0.0, min(100.0, 100.0 - total_penalty))

    # ------------------------------------------------------------------
    # Qualitative outputs
    # ------------------------------------------------------------------
    risk_cat = _risk_category(score)
    action = _recommended_action(
        acid_base_risk, maillard_risk, oxidation_risk, hygroscopic_risk, score
    )

    note_parts: list[str] = [
        f"drug_pka={drug_pka:.2f}",
        f"excipient_pka={'None' if excipient_pka is None else f'{excipient_pka:.2f}'}",
        f"peroxide_level={peroxide_level}",
    ]
    if maillard_risk:
        note_parts.append("Maillard reaction risk identified")
    if oxidation_risk:
        note_parts.append(f"Oxidation risk ({peroxide_level} peroxide excipient)")
    if acid_base_risk:
        note_parts.append(
            f"Acid-base proximity (delta_pKa={abs(drug_pka - (excipient_pka or 0)):.2f})"
        )
    if hygroscopic_risk:
        note_parts.append("Hygroscopic pair — moisture control required")
    notes = "; ".join(note_parts)

    return DrugExcipientCompatibilityResult(
        drug_name=drug_name,
        excipient_name=excipient_name,
        compatibility_score=score,
        risk_category=risk_cat,
        maillard_risk=maillard_risk,
        oxidation_risk=oxidation_risk,
        acid_base_risk=acid_base_risk,
        hygroscopic_risk=hygroscopic_risk,
        recommended_action=action,
        notes=notes,
    )


def screen_excipients(
    drug_name: str,
    excipients: list[dict],
    **drug_kwargs,
) -> list[DrugExcipientCompatibilityResult]:
    """Screen a list of excipients for compatibility with a given drug.

    Parameters
    ----------
    drug_name:
        Identifier for the API.
    excipients:
        List of excipient descriptor dicts. Each dict must contain ``"name"``
        (str) and may optionally contain any keyword argument accepted by
        :func:`predict_compatibility` prefixed without ``excipient_`` or
        ``drug_`` — specifically:

        - ``"pka"`` → *excipient_pka*
        - ``"is_reducing_sugar"`` → *excipient_is_reducing_sugar*
        - ``"peroxide_level"`` → *excipient_peroxide_level*
        - ``"hygroscopic"`` → *excipient_hygroscopic*

    **drug_kwargs:
        Keyword arguments forwarded to :func:`predict_compatibility` for the
        drug (e.g. ``drug_pka``, ``drug_has_amine``, etc.).

    Returns
    -------
    list[DrugExcipientCompatibilityResult]
        Results sorted by *compatibility_score* descending (best first).

    Raises
    ------
    ValueError
        If any excipient dict is missing the ``"name"`` key.
    """
    results: list[DrugExcipientCompatibilityResult] = []
    for exc in excipients:
        if "name" not in exc:
            raise ValueError("Each excipient dict must contain a 'name' key.")
        exc_name = exc["name"]
        exc_kw: dict = {}
        if "pka" in exc:
            exc_kw["excipient_pka"] = exc["pka"]
        if "is_reducing_sugar" in exc:
            exc_kw["excipient_is_reducing_sugar"] = exc["is_reducing_sugar"]
        if "peroxide_level" in exc:
            exc_kw["excipient_peroxide_level"] = exc["peroxide_level"]
        if "hygroscopic" in exc:
            exc_kw["excipient_hygroscopic"] = exc["hygroscopic"]
        result = predict_compatibility(drug_name, exc_name, **drug_kwargs, **exc_kw)
        results.append(result)

    results.sort(key=lambda r: r.compatibility_score, reverse=True)
    return results
