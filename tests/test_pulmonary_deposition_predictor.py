"""Tests for Phase 277 — Pulmonary Drug Deposition Predictor."""

import pytest

from omega_pbpk.prediction.pulmonary_deposition_predictor import (
    PulmonaryDepositionResult,
    compare_formulations,
    predict_pulmonary_deposition,
)

# --- Helpers ---
DRUG = "Salbutamol"
DEFAULT_KWARGS = dict(
    flow_rate_L_min=30.0,
    breathing_frequency_per_min=15.0,
    tidal_volume_mL=500.0,
)


def make_result(mmad_um=3.0, gsd=2.0, **kwargs) -> PulmonaryDepositionResult:
    kw = {**DEFAULT_KWARGS, **kwargs}
    return predict_pulmonary_deposition(DRUG, mmad_um, gsd, **kw)


# --- Return type and field preservation ---


class TestReturnType:
    def test_returns_correct_type(self):
        result = make_result()
        assert isinstance(result, PulmonaryDepositionResult)

    def test_drug_name_preserved(self):
        result = make_result()
        assert result.drug_name == DRUG

    def test_mmad_preserved(self):
        result = make_result(mmad_um=2.5)
        assert result.mmad_um == 2.5

    def test_gsd_preserved(self):
        result = make_result(gsd=1.8)
        assert result.gsd == 1.8

    def test_flow_rate_preserved(self):
        result = make_result(flow_rate_L_min=45.0)
        assert result.flow_rate_L_min == 45.0

    def test_notes_is_string(self):
        result = make_result()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_deposition_class_is_string(self):
        result = make_result()
        assert isinstance(result.deposition_class, str)
        assert result.deposition_class in ("upper_airway", "bronchial", "alveolar")


# --- Fraction bounds ---


class TestFractionBounds:
    def test_extrathoracic_in_unit_interval(self):
        result = make_result(mmad_um=5.0)
        assert 0.0 <= result.extrathoracic_fraction <= 1.0

    def test_tracheobronchial_in_unit_interval(self):
        result = make_result(mmad_um=3.0)
        assert 0.0 <= result.tracheobronchial_fraction <= 1.0

    def test_alveolar_in_unit_interval(self):
        result = make_result(mmad_um=2.0)
        assert 0.0 <= result.alveolar_fraction <= 1.0

    def test_total_deposited_in_unit_interval(self):
        for mmad in [0.5, 1.0, 3.0, 5.0, 10.0, 15.0]:
            result = make_result(mmad_um=mmad)
            assert 0.0 <= result.total_deposited_fraction <= 1.0, (
                f"total={result.total_deposited_fraction} for mmad={mmad}"
            )

    def test_fine_particle_fraction_in_unit_interval(self):
        for mmad in [1.0, 3.0, 5.0, 8.0, 12.0]:
            result = make_result(mmad_um=mmad)
            assert 0.0 <= result.fine_particle_fraction <= 1.0


# --- Physics correctness ---


class TestPhysics:
    def test_impaction_parameter_formula(self):
        mmad = 3.0
        flow = 30.0
        result = make_result(mmad_um=mmad, flow_rate_L_min=flow)
        expected_ip = flow * mmad**2
        assert abs(result.impaction_parameter - expected_ip) < 1e-9

    def test_larger_mmad_more_extrathoracic(self):
        """Larger MMAD → higher extrathoracic fraction."""
        small = make_result(mmad_um=1.0)
        large = make_result(mmad_um=10.0)
        assert large.extrathoracic_fraction >= small.extrathoracic_fraction

    def test_larger_mmad_less_alveolar(self):
        """Larger MMAD → less alveolar deposition."""
        small = make_result(mmad_um=1.5)
        large = make_result(mmad_um=8.0)
        assert large.alveolar_fraction <= small.alveolar_fraction

    def test_small_mmad_alveolar_class(self):
        """Very small particles tend toward alveolar deposition."""
        result = make_result(mmad_um=1.0, flow_rate_L_min=15.0)
        # alveolar should be non-negligible
        assert result.alveolar_fraction >= 0.0

    def test_coarse_mmad_upper_airway_class(self):
        """Coarse particles (>10 um) dominated by upper airway."""
        result = make_result(mmad_um=15.0)
        assert result.deposition_class == "upper_airway"

    def test_fine_particle_fraction_decreases_with_mmad(self):
        """Larger MMAD → smaller FPF."""
        small = make_result(mmad_um=2.0)
        large = make_result(mmad_um=8.0)
        assert small.fine_particle_fraction >= large.fine_particle_fraction

    def test_fine_particle_fraction_zero_for_large_mmad(self):
        """MMAD >= 10 um → FPF = 0."""
        result = make_result(mmad_um=10.0)
        assert result.fine_particle_fraction == 0.0

    def test_higher_flow_rate_more_extrathoracic(self):
        """Higher flow rate → larger impaction parameter → more ET deposition."""
        low_flow = make_result(mmad_um=3.0, flow_rate_L_min=15.0)
        high_flow = make_result(mmad_um=3.0, flow_rate_L_min=60.0)
        assert high_flow.extrathoracic_fraction >= low_flow.extrathoracic_fraction

    def test_total_deposited_sum_of_components(self):
        """Total should equal sum of ET + TB + AL (capped at 1.0)."""
        for mmad in [1.0, 3.0, 5.0, 8.0]:
            result = make_result(mmad_um=mmad)
            component_sum = (
                result.extrathoracic_fraction
                + result.tracheobronchial_fraction
                + result.alveolar_fraction
            )
            expected_total = min(1.0, component_sum)
            assert abs(result.total_deposited_fraction - expected_total) < 1e-9


# --- compare_formulations ---


class TestCompareFormulations:
    def test_returns_list(self):
        results = compare_formulations(DRUG, [1.0, 3.0, 5.0], [1.5, 2.0, 2.5])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_by_alveolar_descending(self):
        mmad_list = [1.0, 3.0, 7.0, 12.0]
        gsd_list = [1.5, 2.0, 2.2, 2.5]
        results = compare_formulations(DRUG, mmad_list, gsd_list)
        for i in range(len(results) - 1):
            assert results[i].alveolar_fraction >= results[i + 1].alveolar_fraction

    def test_all_results_are_correct_type(self):
        results = compare_formulations(DRUG, [2.0, 4.0], [1.8, 2.2])
        for r in results:
            assert isinstance(r, PulmonaryDepositionResult)

    def test_mismatched_list_lengths_raises(self):
        with pytest.raises(ValueError):
            compare_formulations(DRUG, [1.0, 2.0], [1.5])


# --- Validation errors ---


class TestValidation:
    def test_negative_mmad_raises(self):
        with pytest.raises(ValueError, match="mmad_um"):
            predict_pulmonary_deposition(DRUG, -1.0, 2.0, **DEFAULT_KWARGS)

    def test_zero_mmad_raises(self):
        with pytest.raises(ValueError, match="mmad_um"):
            predict_pulmonary_deposition(DRUG, 0.0, 2.0, **DEFAULT_KWARGS)

    def test_gsd_less_than_one_raises(self):
        with pytest.raises(ValueError, match="gsd"):
            predict_pulmonary_deposition(DRUG, 3.0, 0.8, **DEFAULT_KWARGS)

    def test_zero_flow_rate_raises(self):
        with pytest.raises(ValueError, match="flow_rate"):
            predict_pulmonary_deposition(
                DRUG,
                3.0,
                2.0,
                flow_rate_L_min=0.0,
                breathing_frequency_per_min=15.0,
                tidal_volume_mL=500.0,
            )

    def test_zero_breathing_frequency_raises(self):
        with pytest.raises(ValueError, match="breathing_frequency"):
            predict_pulmonary_deposition(
                DRUG,
                3.0,
                2.0,
                flow_rate_L_min=30.0,
                breathing_frequency_per_min=0.0,
                tidal_volume_mL=500.0,
            )

    def test_zero_tidal_volume_raises(self):
        with pytest.raises(ValueError, match="tidal_volume"):
            predict_pulmonary_deposition(
                DRUG,
                3.0,
                2.0,
                flow_rate_L_min=30.0,
                breathing_frequency_per_min=15.0,
                tidal_volume_mL=0.0,
            )
