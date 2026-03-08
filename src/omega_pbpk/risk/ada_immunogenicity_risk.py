"""Phase 337 — Biologic Immunogenicity Risk Scorer (ADA Predictor v2).

Scores anti-drug antibody (ADA) immunogenicity risk for biologics based on
molecular and clinical factors.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ADAImmunogenicityResult:
    """Result of ADA immunogenicity risk scoring.

    Uses plain @dataclass (not frozen) because mitigation_strategies and
    high_risk_factors are list fields.
    """

    drug_name: str
    humanization_pct: float
    route: str
    ada_risk_score: float
    ada_risk_class: str
    score_breakdown: dict
    mitigation_strategies: list
    high_risk_factors: list
    notes: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_ROUTES = {"iv", "sc", "im"}
_ROUTE_SCORES = {"iv": 5, "sc": 15, "im": 12}
_MAX_RAW = 128.0  # theoretical maximum raw score
# Breakdown of max: humanization=20 + aggregation=20 + route_sc=15
# + repeat=10 + immunosuppressed_off=8 + t_cell=20 + mhc=20 + pre_existing=15
_ADA_RISK_CLASSES = {"low", "moderate", "high"}

# Threshold above which a component is flagged as high-risk factor
_COMPONENT_HIGH_THRESHOLD = 12.0


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------


def score_ada_immunogenicity(
    drug_name: str,
    humanization_pct: float,
    aggregation_propensity_score: float,
    route: str,
    repeated_dosing: bool,
    immunosuppressed_patient: bool,
    t_cell_epitope_count: int,
    mhc_binding_score: float,
    pre_existing_antibodies: bool,
) -> ADAImmunogenicityResult:
    """Score ADA immunogenicity risk for a biologic drug.

    Parameters
    ----------
    drug_name:
        Name of the biologic drug.
    humanization_pct:
        Percentage of human sequence (0–100). Higher = lower risk.
    aggregation_propensity_score:
        Aggregation propensity on 0–10 scale. Higher = higher risk.
    route:
        Administration route: "iv", "sc", or "im".
    repeated_dosing:
        True if the drug is given repeatedly.
    immunosuppressed_patient:
        True if the patient is immunosuppressed (reduces ADA risk).
    t_cell_epitope_count:
        Number of predicted T-cell epitopes (0–20).
    mhc_binding_score:
        MHC II binding score (0–1).
    pre_existing_antibodies:
        True if patient has pre-existing antibodies to the drug.

    Returns
    -------
    ADAImmunogenicityResult
    """
    # --- Input validation ---
    if not (0.0 <= humanization_pct <= 100.0):
        raise ValueError(f"humanization_pct must be in [0, 100], got {humanization_pct}")
    if not (0.0 <= aggregation_propensity_score <= 10.0):
        raise ValueError(
            f"aggregation_propensity_score must be in [0, 10], got {aggregation_propensity_score}"
        )
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got '{route}'")

    # --- Score components (each 0–20) ---
    s_humanization = 20.0 * (1.0 - humanization_pct / 100.0)
    s_aggregation = min(20.0, aggregation_propensity_score * 2.0)
    s_route = float(_ROUTE_SCORES[route])
    s_repeat = 10.0 if repeated_dosing else 0.0
    s_immunosuppressed = 0.0 if immunosuppressed_patient else 8.0
    s_t_cell = min(20.0, t_cell_epitope_count * 1.5)
    s_mhc = min(20.0, mhc_binding_score * 20.0)
    s_pre_existing = 15.0 if pre_existing_antibodies else 0.0

    breakdown = {
        "humanization": s_humanization,
        "aggregation": s_aggregation,
        "route": s_route,
        "repeated_dosing": s_repeat,
        "immunosuppressed": s_immunosuppressed,
        "t_cell_epitopes": s_t_cell,
        "mhc_binding": s_mhc,
        "pre_existing_antibodies": s_pre_existing,
    }

    raw_total = sum(breakdown.values())
    normalized_score = raw_total / _MAX_RAW * 100.0

    # --- Risk classification ---
    if normalized_score < 25.0:
        risk_class = "low"
    elif normalized_score <= 60.0:
        risk_class = "moderate"
    else:
        risk_class = "high"

    # --- High-risk factors ---
    high_risk_factors: list[str] = [
        name for name, score in breakdown.items() if score > _COMPONENT_HIGH_THRESHOLD
    ]

    # --- Mitigation strategies ---
    mitigation_strategies: list[str] = _build_mitigations(
        humanization_pct=humanization_pct,
        aggregation_propensity_score=aggregation_propensity_score,
        route=route,
        repeated_dosing=repeated_dosing,
        immunosuppressed_patient=immunosuppressed_patient,
        t_cell_epitope_count=t_cell_epitope_count,
        mhc_binding_score=mhc_binding_score,
        pre_existing_antibodies=pre_existing_antibodies,
        risk_class=risk_class,
    )

    # --- Notes ---
    notes = (
        f"ADA risk score {normalized_score:.1f}/100 (raw {raw_total:.1f}/"
        f"{int(_MAX_RAW)}). Risk class: {risk_class}."
    )

    return ADAImmunogenicityResult(
        drug_name=drug_name,
        humanization_pct=humanization_pct,
        route=route,
        ada_risk_score=round(normalized_score, 2),
        ada_risk_class=risk_class,
        score_breakdown=breakdown,
        mitigation_strategies=mitigation_strategies,
        high_risk_factors=high_risk_factors,
        notes=notes,
    )


def _build_mitigations(
    *,
    humanization_pct: float,
    aggregation_propensity_score: float,
    route: str,
    repeated_dosing: bool,
    immunosuppressed_patient: bool,
    t_cell_epitope_count: int,
    mhc_binding_score: float,
    pre_existing_antibodies: bool,
    risk_class: str,
) -> list[str]:
    """Generate mitigation strategy recommendations."""
    strats: list[str] = []

    if humanization_pct < 90.0:
        strats.append("Increase humanization to >=90% to reduce sequence-based immunogenicity.")
    if aggregation_propensity_score > 5.0:
        strats.append("Optimize formulation to reduce aggregation (e.g., excipients, pH).")
    if route == "sc":
        strats.append(
            "Consider IV administration if clinically feasible to reduce subcutaneous"
            " immunogenicity."
        )
    elif route == "im":
        strats.append(
            "Consider IV administration to minimize intramuscular depot-related immune activation."
        )
    if repeated_dosing:
        strats.append(
            "Monitor ADA titers at each treatment cycle; consider drug holidays if titers increase."
        )
    if not immunosuppressed_patient:
        strats.append(
            "Consider co-administration of immunomodulatory agents (e.g.,"
            " methotrexate) for high-risk patients."
        )
    if t_cell_epitope_count > 5:
        strats.append("Use in-silico deimmunization to reduce T-cell epitope count in sequence.")
    if mhc_binding_score > 0.5:
        strats.append(
            "Engineer away high-affinity MHC II binding peptides identified in epitope mapping."
        )
    if pre_existing_antibodies:
        strats.append("Screen patients for pre-existing antibodies before treatment initiation.")
    if risk_class == "high" and not strats:
        strats.append(
            "Conduct comprehensive immunogenicity risk mitigation program including"
            " patient stratification and immune tolerance protocols."
        )
    if not strats:
        strats.append("Maintain standard immunogenicity monitoring per product labeling.")

    return strats


# ---------------------------------------------------------------------------
# compare_routes helper
# ---------------------------------------------------------------------------


def compare_routes(
    drug_name: str,
    humanization_pct: float,
    aggregation_propensity_score: float,
    repeated_dosing: bool,
    immunosuppressed_patient: bool,
    t_cell_epitope_count: int,
    mhc_binding_score: float,
    pre_existing_antibodies: bool,
) -> list[ADAImmunogenicityResult]:
    """Score ADA risk for all three routes and return sorted ascending by score.

    Parameters
    ----------
    drug_name:
        Name of the biologic drug.
    humanization_pct:
        Percentage of human sequence (0–100).
    aggregation_propensity_score:
        Aggregation propensity (0–10).
    repeated_dosing:
        Whether drug is given repeatedly.
    immunosuppressed_patient:
        Whether patient is immunosuppressed.
    t_cell_epitope_count:
        Number of T-cell epitopes.
    mhc_binding_score:
        MHC II binding score (0–1).
    pre_existing_antibodies:
        Whether pre-existing antibodies are present.

    Returns
    -------
    list[ADAImmunogenicityResult]
        Three results (iv, sc, im) sorted by ada_risk_score ascending.
    """
    results = [
        score_ada_immunogenicity(
            drug_name=drug_name,
            humanization_pct=humanization_pct,
            aggregation_propensity_score=aggregation_propensity_score,
            route=r,
            repeated_dosing=repeated_dosing,
            immunosuppressed_patient=immunosuppressed_patient,
            t_cell_epitope_count=t_cell_epitope_count,
            mhc_binding_score=mhc_binding_score,
            pre_existing_antibodies=pre_existing_antibodies,
        )
        for r in ("iv", "sc", "im")
    ]
    return sorted(results, key=lambda x: x.ada_risk_score)
