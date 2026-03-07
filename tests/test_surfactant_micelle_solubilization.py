"""
Tests for Phase 265: Surfactant Micelle Solubilization
"""

import pytest

from omega_pbpk.prediction.surfactant_micelle_solubilization import (
    MicelleResult,
    predict_micellar_solubilization,
    screen_surfactants,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _result(
    drug_name="TestDrug",
    logp=3.0,
    intrinsic=0.1,
    surfactant="polysorbate_80",
    conc_mM=1.0,
):
    return predict_micellar_solubilization(drug_name, logp, intrinsic, surfactant, conc_mM)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_micelle_result(self):
        result = _result()
        assert isinstance(result, MicelleResult)

    def test_result_is_frozen(self):
        result = _result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore


# ---------------------------------------------------------------------------
# Below CMC behaviour
# ---------------------------------------------------------------------------
class TestBelowCMC:
    def test_ratio_is_one_below_cmc(self):
        # polysorbate_80 CMC = 0.012 mM; use 0.005 mM
        result = predict_micellar_solubilization("Drug", 3.0, 0.1, "polysorbate_80", 0.005)
        assert result.micellar_solubilization_ratio == 1.0

    def test_total_equals_intrinsic_below_cmc(self):
        result = predict_micellar_solubilization("Drug", 3.0, 0.1, "polysorbate_80", 0.005)
        assert abs(result.total_solubility_mg_mL - result.intrinsic_solubility_mg_mL) < 1e-12

    def test_precipitation_risk_high_below_cmc(self):
        result = predict_micellar_solubilization("Drug", 3.0, 0.1, "polysorbate_80", 0.005)
        assert result.precipitation_risk == "high"

    def test_micelle_contribution_zero_below_cmc(self):
        result = predict_micellar_solubilization("Drug", 3.0, 0.1, "polysorbate_80", 0.005)
        assert result.micelle_contribution_pct == 0.0


# ---------------------------------------------------------------------------
# Above CMC behaviour
# ---------------------------------------------------------------------------
class TestAboveCMC:
    def test_ratio_above_one_above_cmc(self):
        result = _result(conc_mM=10.0)  # well above any reasonable CMC
        assert result.micellar_solubilization_ratio > 1.0

    def test_total_solubility_gt_intrinsic(self):
        result = _result(conc_mM=10.0)
        assert result.total_solubility_mg_mL >= result.intrinsic_solubility_mg_mL

    def test_no_precipitation_risk_above_cmc(self):
        result = _result(conc_mM=10.0)
        assert result.precipitation_risk == "none"


# ---------------------------------------------------------------------------
# LogP effect
# ---------------------------------------------------------------------------
class TestLogPEffect:
    def test_higher_logp_gives_higher_ratio(self):
        r_low = predict_micellar_solubilization("Drug", 1.0, 0.1, "polysorbate_80", 5.0)
        r_high = predict_micellar_solubilization("Drug", 5.0, 0.1, "polysorbate_80", 5.0)
        assert r_high.micellar_solubilization_ratio > r_low.micellar_solubilization_ratio


# ---------------------------------------------------------------------------
# Micelle contribution percentage
# ---------------------------------------------------------------------------
class TestMicelleContribution:
    def test_contribution_in_range_0_100(self):
        for conc in [0.005, 1.0, 50.0]:
            result = predict_micellar_solubilization("Drug", 3.0, 0.1, "polysorbate_80", conc)
            assert 0.0 <= result.micelle_contribution_pct <= 100.0

    def test_contribution_increases_with_concentration(self):
        r1 = predict_micellar_solubilization("Drug", 3.0, 0.1, "polysorbate_80", 1.0)
        r2 = predict_micellar_solubilization("Drug", 3.0, 0.1, "polysorbate_80", 100.0)
        assert r2.micelle_contribution_pct > r1.micelle_contribution_pct


# ---------------------------------------------------------------------------
# Bioavailability impact
# ---------------------------------------------------------------------------
class TestBioavailabilityImpact:
    def test_enhanced_when_ratio_above_two(self):
        # polysorbate_80 Kv=1000, logP=3 -> Kv_eff=1000; at 10 mM above CMC
        # ratio = 1 + 1000 * 9.988 / 1000 ~ 10.9
        result = predict_micellar_solubilization("Drug", 3.0, 0.1, "polysorbate_80", 10.0)
        assert result.bioavailability_impact == "enhanced"

    def test_neutral_when_ratio_near_one(self):
        # Just above CMC, small ratio
        result = predict_micellar_solubilization("Drug", 0.5, 0.1, "SDS", 8.01)
        # logp=0.5>0, Kv_eff=200*(0.5/3)=33.3; effective_conc=0.01; ratio=1+33.3*0.01/1000~1.000333
        assert result.bioavailability_impact == "neutral"


# ---------------------------------------------------------------------------
# Precipitation risk values
# ---------------------------------------------------------------------------
class TestPrecipitationRisk:
    def test_valid_risk_values(self):
        valid = {"none", "high"}
        for surf in ["polysorbate_80", "polysorbate_20", "SDS", "CTAB", "poloxamer_407"]:
            r = predict_micellar_solubilization("Drug", 3.0, 0.1, surf, 1.0)
            assert r.precipitation_risk in valid


# ---------------------------------------------------------------------------
# Screen surfactants
# ---------------------------------------------------------------------------
class TestScreenSurfactants:
    def test_returns_five_results(self):
        results = screen_surfactants("Drug", 3.0, 0.1, 1.0)
        assert len(results) == 5

    def test_all_results_are_micelle_result(self):
        results = screen_surfactants("Drug", 3.0, 0.1, 1.0)
        assert all(isinstance(r, MicelleResult) for r in results)

    def test_sorted_descending(self):
        results = screen_surfactants("Drug", 3.0, 0.1, 1.0)
        ratios = [r.micellar_solubilization_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_logp_too_low(self):
        with pytest.raises(ValueError, match="logp"):
            predict_micellar_solubilization("D", -6.0, 0.1, "SDS", 10.0)

    def test_logp_too_high(self):
        with pytest.raises(ValueError, match="logp"):
            predict_micellar_solubilization("D", 11.0, 0.1, "SDS", 10.0)

    def test_negative_intrinsic_solubility(self):
        with pytest.raises(ValueError, match="intrinsic"):
            predict_micellar_solubilization("D", 3.0, -0.1, "SDS", 10.0)

    def test_zero_intrinsic_solubility(self):
        with pytest.raises(ValueError, match="intrinsic"):
            predict_micellar_solubilization("D", 3.0, 0.0, "SDS", 10.0)

    def test_negative_surfactant_conc(self):
        with pytest.raises(ValueError, match="surfactant_conc_mM"):
            predict_micellar_solubilization("D", 3.0, 0.1, "SDS", -1.0)

    def test_unknown_surfactant(self):
        with pytest.raises(ValueError, match="Unknown surfactant"):
            predict_micellar_solubilization("D", 3.0, 0.1, "unknown_surf", 10.0)
