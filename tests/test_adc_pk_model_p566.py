"""Tests for Phase 566 - Antibody Drug Conjugate (ADC) PK Model."""

import pytest

from omega_pbpk.core.adc_pk_model_p566 import (
    ADCPKResult,
    simulate_adc_pk,
)


def _default_result(**kwargs):
    defaults = dict(
        drug_name="T-DM1",
        dose_mg_per_kg=3.6,
        weight_kg=70.0,
        dar_initial=4.0,
        cl_adc_L_per_h=0.005,
        vd_adc_L=5.0,
        k_deconjugation_per_h=0.003,
        cl_payload_L_per_h=2.0,
        vd_payload_L=50.0,
    )
    defaults.update(kwargs)
    return simulate_adc_pk(**defaults)


class TestReturnType:
    def test_returns_adc_pk_result(self):
        r = _default_result()
        assert isinstance(r, ADCPKResult)

    def test_list_lengths_equal(self):
        r = _default_result()
        n = len(r.times_h)
        assert len(r.c_adc_mg_L) == n
        assert len(r.c_payload_mg_L) == n
        assert len(r.dar_profile) == n

    def test_drug_name_preserved(self):
        r = _default_result(drug_name="BV")
        assert r.drug_name == "BV"


class TestInitialConditions:
    def test_c_adc_starts_at_dose_over_vd(self):
        r = _default_result()
        expected = 3.6 * 70.0 / 5.0
        assert abs(r.c_adc_mg_L[0] - expected) < 1e-6

    def test_c_payload_starts_at_zero(self):
        r = _default_result()
        assert r.c_payload_mg_L[0] == 0.0

    def test_dar_starts_at_dar_initial(self):
        r = _default_result(dar_initial=4.0)
        assert abs(r.dar_profile[0] - 4.0) < 1e-6


class TestPayloadDynamics:
    def test_payload_rises_then_falls(self):
        r = _default_result()
        # Find max payload index
        max_idx = r.c_payload_mg_L.index(max(r.c_payload_mg_L))
        assert max_idx > 0  # rises from 0
        assert max_idx < len(r.c_payload_mg_L) - 1  # falls after peak

    def test_higher_deconjugation_higher_payload_cmax(self):
        r_low = _default_result(k_deconjugation_per_h=0.001)
        r_high = _default_result(k_deconjugation_per_h=0.01)
        assert r_high.cmax_payload_mg_L > r_low.cmax_payload_mg_L


class TestADCClearance:
    def test_lower_cl_adc_higher_auc(self):
        r_low = _default_result(cl_adc_L_per_h=0.003)
        r_high = _default_result(cl_adc_L_per_h=0.01)
        assert r_low.auc_adc_mg_h_per_L > r_high.auc_adc_mg_h_per_L


class TestDAR:
    def test_dar_decreases_over_time(self):
        r = _default_result()
        assert r.dar_profile[-1] < r.dar_profile[0]

    def test_dar_starts_at_initial(self):
        r = _default_result(dar_initial=3.5)
        assert abs(r.dar_profile[0] - 3.5) < 1e-6


class TestHalfLives:
    def test_t_half_adc_positive(self):
        r = _default_result()
        assert r.t_half_adc_h > 0

    def test_t_half_payload_positive(self):
        r = _default_result()
        assert r.t_half_payload_h > 0

    def test_t_half_payload_calculation(self):
        r = _default_result(cl_payload_L_per_h=2.0, vd_payload_L=50.0)
        expected = 0.693 * 50.0 / 2.0
        assert abs(r.t_half_payload_h - expected) < 1e-6


class TestDoseLinearity:
    def test_double_dose_doubles_auc_adc(self):
        r1 = _default_result(dose_mg_per_kg=3.6)
        r2 = _default_result(dose_mg_per_kg=7.2)
        ratio = r2.auc_adc_mg_h_per_L / r1.auc_adc_mg_h_per_L
        assert abs(ratio - 2.0) < 0.1

    def test_double_dose_doubles_auc_payload(self):
        r1 = _default_result(dose_mg_per_kg=3.6)
        r2 = _default_result(dose_mg_per_kg=7.2)
        ratio = r2.auc_payload_mg_h_per_L / r1.auc_payload_mg_h_per_L
        assert abs(ratio - 2.0) < 0.1


class TestTherapeuticIndex:
    def test_therapeutic_index_positive(self):
        r = _default_result()
        assert r.therapeutic_index_estimate > 0


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(dose_mg_per_kg=0)

    def test_dar_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(dar_initial=0)

    def test_cl_adc_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(cl_adc_L_per_h=0)

    def test_vd_adc_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(vd_adc_L=0)

    def test_k_deconj_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(k_deconjugation_per_h=0)

    def test_cl_payload_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(cl_payload_L_per_h=0)

    def test_vd_payload_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(vd_payload_L=0)


class TestNotes:
    def test_notes_contains_drug_name(self):
        r = _default_result(drug_name="ADC-X")
        assert "ADC-X" in r.notes
