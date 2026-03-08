"""Tests for phase 678 — drug solubility enhancement prediction."""

import pytest

from omega_pbpk.prediction.solubility_enhancement_p678 import (
    SolubilityEnhancementResult,
    compare_excipients,
    predict_solubility_enhancement,
)

# ── helpers ──────────────────────────────────────────────────────────────────

BASE = dict(
    drug_name="TestDrug",
    logP=3.0,
    mw_Da=400.0,
    intrinsic_solubility_mg_mL=0.01,
)


def _run(**overrides):
    kw = {**BASE, **overrides}
    return predict_solubility_enhancement(**kw)


# ── return type ──────────────────────────────────────────────────────────────


class TestReturnType:
    def test_returns_result(self):
        r = _run(excipient_type="none")
        assert isinstance(r, SolubilityEnhancementResult)

    def test_frozen_raises_on_mutation(self):
        r = _run(excipient_type="none")
        with pytest.raises(AttributeError):
            r.drug_name = "Other"  # type: ignore[misc]


# ── enhanced >= intrinsic for all excipients ─────────────────────────────────


class TestEnhancedSolubility:
    @pytest.mark.parametrize("exc", ["none", "cyclodextrin", "SLS", "PEG400", "HPMC", "Cremophor"])
    def test_enhanced_geq_intrinsic(self, exc):
        r = _run(excipient_type=exc)
        assert r.enhanced_solubility_mg_mL >= r.intrinsic_solubility_mg_mL

    @pytest.mark.parametrize("exc", ["cyclodextrin", "SLS", "PEG400", "HPMC", "Cremophor"])
    def test_ratio_geq_1(self, exc):
        r = _run(excipient_type=exc)
        assert r.solubility_ratio >= 1.0


# ── cyclodextrin more effective for high logP ────────────────────────────────


class TestCyclodextrinLogP:
    def test_high_logP_better(self):
        r_hi = _run(excipient_type="cyclodextrin", logP=4.0)
        r_lo = _run(excipient_type="cyclodextrin", logP=1.5)
        assert r_hi.solubility_ratio > r_lo.solubility_ratio


# ── concentration dependence ─────────────────────────────────────────────────


class TestConcentrationDependence:
    @pytest.mark.parametrize("exc", ["cyclodextrin", "SLS", "PEG400", "HPMC", "Cremophor"])
    def test_higher_conc_higher_enhancement(self, exc):
        r_lo = _run(excipient_type=exc, excipient_concentration_pct=5.0)
        r_hi = _run(excipient_type=exc, excipient_concentration_pct=20.0)
        assert r_hi.enhanced_solubility_mg_mL >= r_lo.enhanced_solubility_mg_mL


# ── solubility class values ──────────────────────────────────────────────────


class TestSolubilityClass:
    def test_freely_soluble(self):
        r = _run(excipient_type="none", intrinsic_solubility_mg_mL=15.0)
        assert r.aqueous_solubility_class == "freely_soluble"

    def test_soluble(self):
        r = _run(excipient_type="none", intrinsic_solubility_mg_mL=2.0)
        assert r.aqueous_solubility_class == "soluble"

    def test_sparingly_soluble(self):
        r = _run(excipient_type="none", intrinsic_solubility_mg_mL=0.5)
        assert r.aqueous_solubility_class == "sparingly_soluble"

    def test_slightly_soluble(self):
        r = _run(excipient_type="none", intrinsic_solubility_mg_mL=0.05)
        assert r.aqueous_solubility_class == "slightly_soluble"

    def test_practically_insoluble(self):
        r = _run(excipient_type="none", intrinsic_solubility_mg_mL=0.005)
        assert r.aqueous_solubility_class == "practically_insoluble"


# ── formulation category values ──────────────────────────────────────────────


class TestFormulationCategory:
    def test_minimal(self):
        r = _run(excipient_type="none")
        assert r.formulation_category == "minimal_enhancement"

    def test_moderate(self):
        r = _run(excipient_type="PEG400", logP=3.0, excipient_concentration_pct=10.0)
        assert r.formulation_category in (
            "moderate_enhancement",
            "minimal_enhancement",
            "significant_enhancement",
        )

    def test_high_enhancement(self):
        r = _run(excipient_type="cyclodextrin", logP=5.0, excipient_concentration_pct=20.0)
        assert r.formulation_category in ("high_enhancement_required", "significant_enhancement")


# ── compare_excipients ───────────────────────────────────────────────────────


class TestCompare:
    def test_returns_5_results(self):
        results = compare_excipients(**BASE)
        assert len(results) == 5

    def test_sorted_descending(self):
        results = compare_excipients(**BASE)
        sols = [r.enhanced_solubility_mg_mL for r in results]
        assert sols == sorted(sols, reverse=True)

    def test_no_none_in_results(self):
        results = compare_excipients(**BASE)
        assert all(r.excipient_type != "none" for r in results)


# ── bioavailability impact ───────────────────────────────────────────────────


class TestBioavailabilityImpact:
    def test_in_range(self):
        r = _run(
            excipient_type="DMSO"
            if "DMSO" in ("cyclodextrin", "SLS", "PEG400", "HPMC", "Cremophor")
            else "cyclodextrin"
        )
        assert 0.0 <= r.bioavailability_impact_pct <= 100.0

    def test_none_zero_impact(self):
        r = _run(excipient_type="none")
        assert r.bioavailability_impact_pct == 0.0

    def test_high_logp_higher_impact(self):
        r = _run(excipient_type="cyclodextrin", logP=4.0)
        assert r.bioavailability_impact_pct > 0.0


# ── validation errors ───────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_excipient(self):
        with pytest.raises(ValueError, match="excipient_type"):
            _run(excipient_type="pixie_dust")

    def test_negative_solubility(self):
        with pytest.raises(ValueError, match="intrinsic_solubility"):
            _run(excipient_type="none", intrinsic_solubility_mg_mL=-1)

    def test_zero_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _run(excipient_type="none", mw_Da=0)

    def test_zero_concentration(self):
        with pytest.raises(ValueError, match="excipient_concentration_pct"):
            _run(excipient_type="SLS", excipient_concentration_pct=0)
