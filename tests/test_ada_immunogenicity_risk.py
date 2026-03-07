"""Tests for Phase 337 — ADA Immunogenicity Risk Scorer."""

import pytest

from omega_pbpk.risk.ada_immunogenicity_risk import (
    ADAImmunogenicityResult,
    compare_routes,
    score_ada_immunogenicity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_kwargs(**overrides):
    base = dict(
        drug_name="TestMab",
        humanization_pct=95.0,
        aggregation_propensity_score=2.0,
        route="iv",
        repeated_dosing=False,
        immunosuppressed_patient=False,
        t_cell_epitope_count=2,
        mhc_binding_score=0.2,
        pre_existing_antibodies=False,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = score_ada_immunogenicity(**_default_kwargs())
    assert isinstance(result, ADAImmunogenicityResult)


# ---------------------------------------------------------------------------
# 2. ada_risk_score in [0, 100]
# ---------------------------------------------------------------------------


def test_score_in_range_low_risk():
    result = score_ada_immunogenicity(**_default_kwargs())
    assert 0.0 <= result.ada_risk_score <= 100.0


def test_score_in_range_high_risk():
    result = score_ada_immunogenicity(
        **_default_kwargs(
            humanization_pct=0.0,
            aggregation_propensity_score=10.0,
            route="sc",
            repeated_dosing=True,
            immunosuppressed_patient=False,
            t_cell_epitope_count=20,
            mhc_binding_score=1.0,
            pre_existing_antibodies=True,
        )
    )
    assert 0.0 <= result.ada_risk_score <= 100.0


# ---------------------------------------------------------------------------
# 3. ada_risk_class in valid set
# ---------------------------------------------------------------------------


def test_risk_class_valid():
    result = score_ada_immunogenicity(**_default_kwargs())
    assert result.ada_risk_class in {"low", "moderate", "high"}


def test_risk_class_high_when_max():
    result = score_ada_immunogenicity(
        **_default_kwargs(
            humanization_pct=0.0,
            aggregation_propensity_score=10.0,
            route="sc",
            repeated_dosing=True,
            immunosuppressed_patient=False,
            t_cell_epitope_count=20,
            mhc_binding_score=1.0,
            pre_existing_antibodies=True,
        )
    )
    assert result.ada_risk_class == "high"


def test_risk_class_low_when_min():
    result = score_ada_immunogenicity(
        **_default_kwargs(
            humanization_pct=100.0,
            aggregation_propensity_score=0.0,
            route="iv",
            repeated_dosing=False,
            immunosuppressed_patient=True,
            t_cell_epitope_count=0,
            mhc_binding_score=0.0,
            pre_existing_antibodies=False,
        )
    )
    assert result.ada_risk_class == "low"


# ---------------------------------------------------------------------------
# 4. Fully human drug → lower score than chimeric
# ---------------------------------------------------------------------------


def test_humanization_fully_human_lower_than_chimeric():
    fully_human = score_ada_immunogenicity(**_default_kwargs(humanization_pct=100.0))
    chimeric = score_ada_immunogenicity(**_default_kwargs(humanization_pct=60.0))
    assert fully_human.ada_risk_score < chimeric.ada_risk_score


# ---------------------------------------------------------------------------
# 5. IV route → lower than SC
# ---------------------------------------------------------------------------


def test_iv_lower_than_sc():
    iv = score_ada_immunogenicity(**_default_kwargs(route="iv"))
    sc = score_ada_immunogenicity(**_default_kwargs(route="sc"))
    assert iv.ada_risk_score < sc.ada_risk_score


# ---------------------------------------------------------------------------
# 6. immunosuppressed=True → lower score than False
# ---------------------------------------------------------------------------


def test_immunosuppressed_reduces_score():
    normal = score_ada_immunogenicity(**_default_kwargs(immunosuppressed_patient=False))
    suppressed = score_ada_immunogenicity(**_default_kwargs(immunosuppressed_patient=True))
    assert suppressed.ada_risk_score < normal.ada_risk_score


# ---------------------------------------------------------------------------
# 7. pre_existing_antibodies=True → higher score
# ---------------------------------------------------------------------------


def test_pre_existing_antibodies_increases_score():
    no_ab = score_ada_immunogenicity(**_default_kwargs(pre_existing_antibodies=False))
    with_ab = score_ada_immunogenicity(**_default_kwargs(pre_existing_antibodies=True))
    assert with_ab.ada_risk_score > no_ab.ada_risk_score


# ---------------------------------------------------------------------------
# 8. high_risk_factors nonempty when risk_class == "high"
# ---------------------------------------------------------------------------


def test_high_risk_factors_nonempty_for_high_class():
    result = score_ada_immunogenicity(
        **_default_kwargs(
            humanization_pct=0.0,
            aggregation_propensity_score=10.0,
            route="sc",
            repeated_dosing=True,
            immunosuppressed_patient=False,
            t_cell_epitope_count=20,
            mhc_binding_score=1.0,
            pre_existing_antibodies=True,
        )
    )
    assert result.ada_risk_class == "high"
    assert len(result.high_risk_factors) > 0


# ---------------------------------------------------------------------------
# 9. mitigation_strategies nonempty
# ---------------------------------------------------------------------------


def test_mitigation_strategies_nonempty():
    result = score_ada_immunogenicity(**_default_kwargs())
    assert isinstance(result.mitigation_strategies, list)
    assert len(result.mitigation_strategies) > 0


# ---------------------------------------------------------------------------
# 10. compare_routes returns 3 results sorted ascending
# ---------------------------------------------------------------------------


def test_compare_routes_returns_three():
    results = compare_routes(
        drug_name="TestMab",
        humanization_pct=90.0,
        aggregation_propensity_score=3.0,
        repeated_dosing=False,
        immunosuppressed_patient=False,
        t_cell_epitope_count=3,
        mhc_binding_score=0.3,
        pre_existing_antibodies=False,
    )
    assert len(results) == 3


def test_compare_routes_sorted_ascending():
    results = compare_routes(
        drug_name="TestMab",
        humanization_pct=90.0,
        aggregation_propensity_score=3.0,
        repeated_dosing=False,
        immunosuppressed_patient=False,
        t_cell_epitope_count=3,
        mhc_binding_score=0.3,
        pre_existing_antibodies=False,
    )
    scores = [r.ada_risk_score for r in results]
    assert scores == sorted(scores)


def test_compare_routes_iv_first():
    """IV should have the lowest score and appear first."""
    results = compare_routes(
        drug_name="TestMab",
        humanization_pct=90.0,
        aggregation_propensity_score=3.0,
        repeated_dosing=False,
        immunosuppressed_patient=False,
        t_cell_epitope_count=3,
        mhc_binding_score=0.3,
        pre_existing_antibodies=False,
    )
    assert results[0].route == "iv"


def test_compare_routes_all_route_types():
    results = compare_routes(
        drug_name="TestMab",
        humanization_pct=90.0,
        aggregation_propensity_score=3.0,
        repeated_dosing=False,
        immunosuppressed_patient=False,
        t_cell_epitope_count=3,
        mhc_binding_score=0.3,
        pre_existing_antibodies=False,
    )
    routes = {r.route for r in results}
    assert routes == {"iv", "sc", "im"}


# ---------------------------------------------------------------------------
# 11. Validation errors
# ---------------------------------------------------------------------------


def test_validation_humanization_negative():
    with pytest.raises(ValueError, match="humanization_pct"):
        score_ada_immunogenicity(**_default_kwargs(humanization_pct=-1.0))


def test_validation_humanization_over_100():
    with pytest.raises(ValueError, match="humanization_pct"):
        score_ada_immunogenicity(**_default_kwargs(humanization_pct=101.0))


def test_validation_aggregation_negative():
    with pytest.raises(ValueError, match="aggregation_propensity_score"):
        score_ada_immunogenicity(**_default_kwargs(aggregation_propensity_score=-0.1))


def test_validation_aggregation_over_10():
    with pytest.raises(ValueError, match="aggregation_propensity_score"):
        score_ada_immunogenicity(**_default_kwargs(aggregation_propensity_score=10.1))


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        score_ada_immunogenicity(**_default_kwargs(route="oral"))


# ---------------------------------------------------------------------------
# 12. score_breakdown keys
# ---------------------------------------------------------------------------


def test_score_breakdown_has_all_keys():
    result = score_ada_immunogenicity(**_default_kwargs())
    expected_keys = {
        "humanization",
        "aggregation",
        "route",
        "repeated_dosing",
        "immunosuppressed",
        "t_cell_epitopes",
        "mhc_binding",
        "pre_existing_antibodies",
    }
    assert set(result.score_breakdown.keys()) == expected_keys


def test_score_breakdown_values_nonnegative():
    result = score_ada_immunogenicity(**_default_kwargs())
    for val in result.score_breakdown.values():
        assert val >= 0.0
