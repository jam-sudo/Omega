"""Tests for drug crystal nucleation kinetics prediction (Phase 1016)."""

import pytest

from omega_pbpk.prediction.drug_crystal_nucleation import (
    CrystalNucleationResult,
    predict_crystal_nucleation,
    screen_nucleation_stability,
)


def default_result(**kwargs) -> CrystalNucleationResult:
    """Helper: run prediction with default parameters."""
    params = dict(
        drug_name="DrugA",
        supersaturation_ratio=3.0,
        logp=2.0,
        mw=300.0,
        melting_point_celsius=170.0,
    )
    params.update(kwargs)
    return predict_crystal_nucleation(**params)


class TestReturnType:
    def test_returns_crystal_nucleation_result(self):
        result = default_result()
        assert isinstance(result, CrystalNucleationResult)

    def test_drug_name_preserved(self):
        result = default_result(drug_name="Ibuprofen")
        assert result.drug_name == "Ibuprofen"

    def test_supersaturation_ratio_preserved(self):
        result = default_result(supersaturation_ratio=5.0)
        assert result.supersaturation_ratio == pytest.approx(5.0)


class TestNucleationRate:
    def test_nucleation_rate_in_valid_range(self):
        result = default_result()
        assert 0.0 <= result.nucleation_rate <= 100.0

    def test_higher_supersaturation_higher_nucleation_rate(self):
        result_low = default_result(supersaturation_ratio=1.5)
        result_high = default_result(supersaturation_ratio=10.0)
        assert result_high.nucleation_rate > result_low.nucleation_rate


class TestInductionTime:
    def test_induction_time_positive(self):
        result = default_result()
        assert result.induction_time_min > 0

    def test_higher_supersaturation_shorter_induction_time(self):
        result_low = default_result(supersaturation_ratio=1.5)
        result_high = default_result(supersaturation_ratio=10.0)
        assert result_high.induction_time_min < result_low.induction_time_min


class TestGrowthRate:
    def test_growth_rate_positive(self):
        result = default_result()
        assert result.growth_rate_relative > 0


class TestCrystallizationTendency:
    def test_tendency_valid_values(self):
        result = default_result()
        assert result.crystallization_tendency in {"low", "moderate", "high"}

    def test_high_supersaturation_high_tendency(self):
        result = default_result(supersaturation_ratio=50.0)
        assert result.crystallization_tendency == "high"

    def test_low_supersaturation_low_tendency(self):
        result = default_result(supersaturation_ratio=1.05)
        assert result.crystallization_tendency == "low"


class TestPolymorphismRisk:
    def test_polymorphism_risk_is_bool(self):
        result = default_result()
        assert isinstance(result.polymorphism_risk, bool)

    def test_high_logp_polymorphism_risk_true(self):
        result = default_result(logp=4.0)
        assert result.polymorphism_risk is True

    def test_low_logp_polymorphism_risk_false(self):
        result = default_result(logp=1.0)
        assert result.polymorphism_risk is False


class TestGlassFormingAbility:
    def test_gfa_valid_values(self):
        result = default_result()
        assert result.glass_forming_ability in {"class_I", "class_II", "class_III"}

    def test_high_mw_low_ss_class_I(self):
        # High MW, low SS → high GFA score → class_I
        result = default_result(mw=600.0, supersaturation_ratio=1.5)
        assert result.glass_forming_ability == "class_I"

    def test_low_mw_high_ss_class_III(self):
        # Low MW, high SS → low GFA score → class_III
        result = default_result(mw=100.0, supersaturation_ratio=50.0)
        assert result.glass_forming_ability == "class_III"


class TestRecommendedStabilizer:
    def test_stabilizer_valid_values(self):
        result = default_result()
        assert result.recommended_stabilizer in {"PVP", "HPMC", "HPC", "none"}

    def test_high_tendency_pvp(self):
        result = default_result(supersaturation_ratio=50.0)
        assert result.recommended_stabilizer == "PVP"

    def test_very_low_tendency_none(self):
        result = default_result(supersaturation_ratio=1.02)
        assert result.recommended_stabilizer == "none"


class TestScreenNucleationStability:
    def _make_compounds(self):
        return [
            {"drug_name": "CompA", "supersaturation_ratio": 2.0, "logp": 1.5},
            {"drug_name": "CompB", "supersaturation_ratio": 10.0, "logp": 4.0},
            {"drug_name": "CompC", "supersaturation_ratio": 5.0, "logp": 2.5},
        ]

    def test_returns_list(self):
        result = screen_nucleation_stability(self._make_compounds())
        assert isinstance(result, list)

    def test_same_length_as_input(self):
        compounds = self._make_compounds()
        result = screen_nucleation_stability(compounds)
        assert len(result) == len(compounds)

    def test_sorted_by_induction_time_descending(self):
        result = screen_nucleation_stability(self._make_compounds())
        times = [r.induction_time_min for r in result]
        assert times == sorted(times, reverse=True)

    def test_all_results_are_crystal_nucleation_result(self):
        result = screen_nucleation_stability(self._make_compounds())
        for r in result:
            assert isinstance(r, CrystalNucleationResult)


class TestValidation:
    def test_supersaturation_equal_one_raises(self):
        with pytest.raises(ValueError):
            predict_crystal_nucleation("X", supersaturation_ratio=1.0, logp=2.0)

    def test_supersaturation_less_than_one_raises(self):
        with pytest.raises(ValueError):
            predict_crystal_nucleation("X", supersaturation_ratio=0.5, logp=2.0)

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError):
            predict_crystal_nucleation("X", supersaturation_ratio=3.0, logp=2.0, mw=-100.0)

    def test_invalid_mw_zero_raises(self):
        with pytest.raises(ValueError):
            predict_crystal_nucleation("X", supersaturation_ratio=3.0, logp=2.0, mw=0.0)
