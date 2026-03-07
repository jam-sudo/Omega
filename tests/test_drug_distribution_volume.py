"""Tests for volume of distribution prediction — Phase 1004."""

import pytest

from omega_pbpk.prediction.drug_distribution_volume import (
    VolumeDistributionResult,
    predict_volume_of_distribution,
    screen_vd,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _predict(**kwargs):
    defaults = dict(drug_name="TestDrug", logp=2.0, pka=7.0)
    defaults.update(kwargs)
    return predict_volume_of_distribution(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic fields
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_result_instance(self):
        r = _predict()
        assert isinstance(r, VolumeDistributionResult)

    def test_vd_ss_positive(self):
        r = _predict()
        assert r.vd_ss_L_per_kg > 0.0

    def test_vd_total_positive(self):
        r = _predict()
        assert r.vd_total_L > 0.0

    def test_vd_class_valid(self):
        r = _predict()
        assert r.vd_class in {"low", "moderate", "high"}

    def test_tissue_binding_factor_positive(self):
        r = _predict()
        assert r.tissue_binding_factor > 0.0

    def test_predicted_t_half_positive(self):
        r = _predict()
        assert r.predicted_t_half_h > 0.0


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_vd_total_equals_vd_ss_times_weight(self):
        r = _predict(weight_kg=70.0)
        assert r.vd_total_L == pytest.approx(r.vd_ss_L_per_kg * 70.0, rel=1e-9)

    def test_dominant_tissue_valid(self):
        r = _predict()
        assert r.distribution_dominant_tissue in {"plasma", "muscle", "fat", "liver"}

    def test_vd_class_low_or_moderate_hydrophilic(self):
        # Very hydrophilic, high fup → smallest Vd model can produce
        # Model floor ~0.66 L/kg (tissue volumes dominate), so class is moderate
        r = _predict(logp=-3.0, fup=1.0)
        assert r.vd_class in {"low", "moderate"}
        # Confirm it is lower than a lipophilic compound
        r_lipo = _predict(logp=4.0, fup=1.0)
        assert r.vd_ss_L_per_kg < r_lipo.vd_ss_L_per_kg

    def test_vd_class_high(self):
        # Lipophilic, low fup → high Vd
        r = _predict(logp=5.0, fup=0.01)
        assert r.vd_class == "high"

    def test_t_half_formula(self):
        # t½ = 0.693 * vd_total_L / 1.0 (CL=1 L/h reference)
        r = _predict(weight_kg=70.0)
        expected = 0.693 * r.vd_total_L
        assert r.predicted_t_half_h == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Pharmacological behaviour
# ---------------------------------------------------------------------------

class TestPKBehaviour:
    def test_high_logp_higher_vd_than_low_logp(self):
        r_low = _predict(logp=-1.0, fup=0.5)
        r_high = _predict(logp=4.0, fup=0.5)
        assert r_high.vd_ss_L_per_kg > r_low.vd_ss_L_per_kg

    def test_base_higher_vd_than_neutral_same_logp(self):
        r_base = _predict(logp=2.0, ionization_type="base", fup=0.5)
        r_neutral = _predict(logp=2.0, ionization_type="neutral", fup=0.5)
        assert r_base.vd_ss_L_per_kg > r_neutral.vd_ss_L_per_kg

    def test_low_fup_higher_vd_than_high_fup(self):
        r_low = _predict(logp=2.0, fup=0.05)
        r_high = _predict(logp=2.0, fup=0.9)
        assert r_low.vd_ss_L_per_kg > r_high.vd_ss_L_per_kg


# ---------------------------------------------------------------------------
# screen_vd
# ---------------------------------------------------------------------------

class TestScreenVd:
    def test_returns_list(self):
        compounds = [
            {"drug_name": "A", "logp": 1.0, "pka": 7.0},
            {"drug_name": "B", "logp": 4.0, "pka": 7.0},
        ]
        result = screen_vd(compounds)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_sorted_descending(self):
        compounds = [
            {"drug_name": "A", "logp": -1.0, "pka": 7.0, "fup": 0.9},
            {"drug_name": "B", "logp": 4.0, "pka": 7.0, "fup": 0.1},
            {"drug_name": "C", "logp": 2.0, "pka": 7.0, "fup": 0.5},
        ]
        result = screen_vd(compounds)
        vds = [r.vd_ss_L_per_kg for r in result]
        assert vds == sorted(vds, reverse=True)

    def test_empty_list(self):
        assert screen_vd([]) == []


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_fup_zero_raises(self):
        with pytest.raises(ValueError, match="fup"):
            _predict(fup=0.0)

    def test_invalid_fup_above_one_raises(self):
        with pytest.raises(ValueError, match="fup"):
            _predict(fup=1.1)

    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError, match="ionization_type"):
            _predict(ionization_type="zwitterion")

    def test_invalid_pka_below_zero_raises(self):
        with pytest.raises(ValueError, match="pka"):
            _predict(pka=-1.0)

    def test_invalid_pka_above_14_raises(self):
        with pytest.raises(ValueError, match="pka"):
            _predict(pka=15.0)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            _predict(weight_kg=0.0)
