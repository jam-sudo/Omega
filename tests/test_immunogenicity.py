"""Tests for biologic immunogenicity prediction (Phase 157)."""

import pytest

from omega_pbpk.risk.immunogenicity import (
    ImmunogenicityResult,
    assess_immunogenicity,
    compare_routes_immunogenicity,
)


def test_negative_mw_raises():
    with pytest.raises(ValueError, match="Molecular weight"):
        assess_immunogenicity("TestAb", mw_kDa=-1.0)


def test_zero_mw_raises():
    with pytest.raises(ValueError, match="Molecular weight"):
        assess_immunogenicity("TestAb", mw_kDa=0.0)


def test_humanization_over_100_raises():
    with pytest.raises(ValueError, match="[Hh]umanization"):
        assess_immunogenicity("TestAb", mw_kDa=150.0, humanization_pct=110.0)


def test_humanization_negative_raises():
    with pytest.raises(ValueError, match="[Hh]umanization"):
        assess_immunogenicity("TestAb", mw_kDa=150.0, humanization_pct=-5.0)


def test_negative_aggregation_raises():
    with pytest.raises(ValueError, match="[Aa]ggregation"):
        assess_immunogenicity("TestAb", mw_kDa=150.0, aggregation_score=-1.0)


def test_returns_result():
    r = assess_immunogenicity("TestAb", mw_kDa=150.0)
    assert isinstance(r, ImmunogenicityResult)


def test_result_fields_present():
    r = assess_immunogenicity("TestAb", mw_kDa=150.0)
    assert r.protein_name == "TestAb"
    assert r.mw_kDa == 150.0
    assert 0 < r.ada_risk_pct <= 95
    assert r.ada_risk_category in ("low", "moderate", "high", "very_high")
    assert r.clinical_impact_prediction in ("unlikely", "possible", "probable", "expected")


def test_fully_human_lower_risk_than_murine():
    human = assess_immunogenicity("HuAb", mw_kDa=150.0, humanization_pct=100.0)
    murine = assess_immunogenicity("MurAb", mw_kDa=150.0, humanization_pct=0.0)
    assert human.ada_risk_pct < murine.ada_risk_pct


def test_iv_lower_risk_than_sc():
    iv = assess_immunogenicity("TestAb", mw_kDa=150.0, route="iv")
    sc = assess_immunogenicity("TestAb", mw_kDa=150.0, route="sc")
    assert iv.ada_risk_pct < sc.ada_risk_pct


def test_iv_lower_risk_than_im():
    iv = assess_immunogenicity("TestAb", mw_kDa=150.0, route="iv")
    im = assess_immunogenicity("TestAb", mw_kDa=150.0, route="im")
    assert iv.ada_risk_pct < im.ada_risk_pct


def test_enzyme_higher_risk_than_mab():
    mab = assess_immunogenicity("TestAb", mw_kDa=150.0, protein_type="mab")
    enzyme = assess_immunogenicity("TestEnz", mw_kDa=150.0, protein_type="enzyme")
    assert enzyme.ada_risk_pct > mab.ada_risk_pct


def test_peptide_higher_risk_than_mab():
    mab = assess_immunogenicity("TestAb", mw_kDa=150.0, protein_type="mab")
    peptide = assess_immunogenicity("TestPep", mw_kDa=10.0, protein_type="peptide")
    assert peptide.ada_risk_pct > mab.ada_risk_pct


def test_aggregation_increases_risk():
    low = assess_immunogenicity("TestAb", mw_kDa=150.0, aggregation_score=0.0)
    high = assess_immunogenicity("TestAb", mw_kDa=150.0, aggregation_score=80.0)
    assert high.ada_risk_pct > low.ada_risk_pct


def test_peg_reduces_risk():
    no_peg = assess_immunogenicity("TestAb", mw_kDa=150.0, peg_conjugated=False)
    with_peg = assess_immunogenicity("TestAb", mw_kDa=150.0, peg_conjugated=True)
    assert with_peg.ada_risk_pct < no_peg.ada_risk_pct


def test_glycosylation_reduces_risk():
    no_glyc = assess_immunogenicity("TestAb", mw_kDa=150.0, glycosylation=False)
    with_glyc = assess_immunogenicity("TestAb", mw_kDa=150.0, glycosylation=True)
    assert with_glyc.ada_risk_pct < no_glyc.ada_risk_pct


def test_longer_duration_increases_risk():
    short = assess_immunogenicity("TestAb", mw_kDa=150.0, treatment_duration_weeks=4)
    long = assess_immunogenicity("TestAb", mw_kDa=150.0, treatment_duration_weeks=104)
    assert long.ada_risk_pct > short.ada_risk_pct


def test_oncology_lower_risk_than_healthy():
    onco = assess_immunogenicity("TestAb", mw_kDa=150.0, patient_condition="oncology")
    healthy = assess_immunogenicity("TestAb", mw_kDa=150.0, patient_condition="healthy")
    assert onco.ada_risk_pct < healthy.ada_risk_pct


def test_risk_clamped_max_95():
    r = assess_immunogenicity(
        "HighRisk",
        mw_kDa=10.0,
        protein_type="enzyme",
        humanization_pct=0.0,
        aggregation_score=100.0,
        peg_conjugated=False,
        glycosylation=False,
        route="im",
        treatment_duration_weeks=200,
        patient_condition="healthy",
    )
    assert r.ada_risk_pct <= 95.0


def test_risk_clamped_min_1():
    r = assess_immunogenicity(
        "LowRisk",
        mw_kDa=150.0,
        protein_type="mab",
        humanization_pct=100.0,
        aggregation_score=0.0,
        peg_conjugated=True,
        glycosylation=True,
        route="iv",
        treatment_duration_weeks=1,
        patient_condition="oncology",
    )
    assert r.ada_risk_pct >= 1.0


def test_mitigation_list_not_empty():
    r = assess_immunogenicity("TestAb", mw_kDa=150.0)
    assert len(r.mitigation_strategies) > 0


def test_compare_routes_returns_three():
    results = compare_routes_immunogenicity("TestAb", mw_kDa=150.0)
    assert len(results) == 3


def test_compare_routes_covers_iv_sc_im():
    results = compare_routes_immunogenicity("TestAb", mw_kDa=150.0)
    routes = {r.route for r in results}
    assert routes == {"iv", "sc", "im"}


def test_compare_routes_iv_lowest():
    results = compare_routes_immunogenicity("TestAb", mw_kDa=150.0)
    by_route = {r.route: r.ada_risk_pct for r in results}
    assert by_route["iv"] < by_route["sc"]
    assert by_route["iv"] < by_route["im"]
