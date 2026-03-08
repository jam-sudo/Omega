"""Tests for Phase 1084 — Drug Protein Binding Displacement (prediction module)."""

import pytest

from omega_pbpk.prediction.protein_binding_displacement import (
    ProteinBindingDisplacementResult,
    predict_binding_displacement,
    screen_displacement_risk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base(**overrides):
    defaults = dict(
        victim_drug="warfarin",
        perpetrator_drug="sulfonamide",
        albumin_site="site_1",
        fu_victim=0.02,
        ki_displacement_mg_per_L=5.0,
        perpetrator_cmax_mg_per_L=10.0,
        hepatic_extraction_ratio=0.3,
    )
    defaults.update(overrides)
    return predict_binding_displacement(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_is_dataclass_instance(self):
        result = _base()
        assert isinstance(result, ProteinBindingDisplacementResult)

    def test_victim_drug_stored(self):
        result = _base(victim_drug="warfarin")
        assert result.victim_drug == "warfarin"

    def test_perpetrator_drug_stored(self):
        result = _base(perpetrator_drug="sulfonamide")
        assert result.perpetrator_drug == "sulfonamide"

    def test_albumin_site_stored(self):
        result = _base(albumin_site="site_2")
        assert result.albumin_site == "site_2"


# ---------------------------------------------------------------------------
# fu_victim_new behaviour
# ---------------------------------------------------------------------------


class TestFUBehaviour:
    def test_fu_victim_new_ge_baseline(self):
        result = _base()
        assert result.fu_victim_new >= result.fu_victim_baseline

    def test_higher_cmax_gives_higher_fu_new(self):
        low = _base(perpetrator_cmax_mg_per_L=1.0)
        high = _base(perpetrator_cmax_mg_per_L=50.0)
        assert high.fu_victim_new > low.fu_victim_new

    def test_zero_cmax_fu_new_equals_baseline(self):
        result = _base(perpetrator_cmax_mg_per_L=0.0)
        assert abs(result.fu_victim_new - result.fu_victim_baseline) < 1e-9

    def test_f_bound_baseline_correct(self):
        fu = 0.05
        result = _base(fu_victim=fu)
        assert abs(result.f_bound_baseline - (1.0 - fu)) < 1e-9

    def test_fu_victim_new_le_one(self):
        # Even with extreme displacement, fu must not exceed 1.0
        result = _base(
            fu_victim=0.01,
            ki_displacement_mg_per_L=0.001,
            perpetrator_cmax_mg_per_L=1000.0,
        )
        assert result.fu_victim_new <= 1.0


# ---------------------------------------------------------------------------
# delta_fu and Cmax_free_ratio
# ---------------------------------------------------------------------------


class TestDeltaFU:
    def test_delta_fu_pct_ge_zero(self):
        result = _base()
        assert result.delta_fu_pct >= 0.0

    def test_delta_fu_absolute_ge_zero(self):
        result = _base()
        assert result.delta_fu_absolute >= 0.0

    def test_cmax_free_ratio_ge_one(self):
        result = _base()
        assert result.cmax_free_ratio >= 1.0

    def test_zero_cmax_cmax_free_ratio_is_one(self):
        result = _base(perpetrator_cmax_mg_per_L=0.0)
        assert abs(result.cmax_free_ratio - 1.0) < 1e-9

    def test_higher_cmax_gives_higher_cmax_free_ratio(self):
        low = _base(perpetrator_cmax_mg_per_L=1.0)
        high = _base(perpetrator_cmax_mg_per_L=100.0)
        assert high.cmax_free_ratio > low.cmax_free_ratio


# ---------------------------------------------------------------------------
# Hepatic extraction classification
# ---------------------------------------------------------------------------


class TestHepaticClass:
    def test_hepatic_class_valid_values(self):
        valid = {"restricted", "unrestricted", "intermediate"}
        for er in [0.1, 0.5, 0.9]:
            r = _base(hepatic_extraction_ratio=er)
            assert r.hepatic_extraction_class in valid

    def test_high_er_is_restricted(self):
        result = _base(hepatic_extraction_ratio=0.8)
        assert result.hepatic_extraction_class == "restricted"

    def test_low_er_is_unrestricted(self):
        result = _base(hepatic_extraction_ratio=0.1)
        assert result.hepatic_extraction_class == "unrestricted"

    def test_mid_er_is_intermediate(self):
        result = _base(hepatic_extraction_ratio=0.5)
        assert result.hepatic_extraction_class == "intermediate"


# ---------------------------------------------------------------------------
# Clinical significance
# ---------------------------------------------------------------------------


class TestClinicalSignificance:
    def test_clinical_significance_valid_values(self):
        valid = {"negligible", "minor", "moderate"}
        result = _base()
        assert result.clinical_significance in valid

    def test_negligible_when_minimal_displacement(self):
        # Very high Ki -> tiny displacement
        result = _base(ki_displacement_mg_per_L=1000.0, perpetrator_cmax_mg_per_L=0.01)
        assert result.clinical_significance == "negligible"


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_nonempty(self):
        result = _base()
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# screen_displacement_risk
# ---------------------------------------------------------------------------


class TestScreenDisplacementRisk:
    def _perpetrators(self):
        return [
            {"name": "drug_a", "ki": 1.0, "cmax": 5.0, "site": "site_1"},
            {"name": "drug_b", "ki": 5.0, "cmax": 20.0, "site": "site_2"},
            {"name": "drug_c", "ki": 10.0, "cmax": 1.0, "site": "site_1"},
        ]

    def test_returns_correct_count(self):
        results = screen_displacement_risk("warfarin", 0.02, self._perpetrators())
        assert len(results) == 3

    def test_sorted_by_cmax_free_ratio_descending(self):
        results = screen_displacement_risk("warfarin", 0.02, self._perpetrators())
        ratios = [r.cmax_free_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_single_perpetrator_screen(self):
        results = screen_displacement_risk(
            "warfarin", 0.02, [{"name": "drug_x", "ki": 2.0, "cmax": 10.0, "site": "site_1"}]
        )
        assert len(results) == 1

    def test_victim_drug_propagated(self):
        results = screen_displacement_risk("warfarin", 0.02, self._perpetrators())
        assert all(r.victim_drug == "warfarin" for r in results)

    def test_optional_hepatic_ratio_used(self):
        perps = [
            {
                "name": "drug_a",
                "ki": 1.0,
                "cmax": 5.0,
                "site": "site_1",
                "hepatic_extraction_ratio": 0.85,
            },
        ]
        results = screen_displacement_risk("warfarin", 0.02, perps)
        assert results[0].hepatic_extraction_class == "restricted"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_albumin_site_raises_value_error(self):
        with pytest.raises(ValueError, match="albumin_site"):
            _base(albumin_site="site_3")

    def test_invalid_albumin_site_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="albumin_site"):
            _base(albumin_site="")

    def test_zero_fu_victim_raises_value_error(self):
        with pytest.raises(ValueError, match="fu_victim"):
            _base(fu_victim=0.0)

    def test_fu_victim_above_one_raises_value_error(self):
        with pytest.raises(ValueError, match="fu_victim"):
            _base(fu_victim=1.1)

    def test_zero_ki_raises_value_error(self):
        with pytest.raises(ValueError, match="ki_displacement"):
            _base(ki_displacement_mg_per_L=0.0)

    def test_negative_ki_raises_value_error(self):
        with pytest.raises(ValueError, match="ki_displacement"):
            _base(ki_displacement_mg_per_L=-1.0)

    def test_negative_cmax_raises_value_error(self):
        with pytest.raises(ValueError, match="perpetrator_cmax"):
            _base(perpetrator_cmax_mg_per_L=-0.1)

    def test_hepatic_ratio_above_one_raises_value_error(self):
        with pytest.raises(ValueError, match="hepatic_extraction_ratio"):
            _base(hepatic_extraction_ratio=1.1)

    def test_hepatic_ratio_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="hepatic_extraction_ratio"):
            _base(hepatic_extraction_ratio=-0.1)

    def test_both_albumin_sites_accepted(self):
        for site in ["site_1", "site_2"]:
            result = _base(albumin_site=site)
            assert result.albumin_site == site
