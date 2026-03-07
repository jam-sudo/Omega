"""Tests for gut microbiome drug metabolism model — Phase 749."""

import math

import pytest

from omega_pbpk.core.gut_microbiome_pk import (
    GutMicrobiomePKResult,
    assess_microbiome_impact,
    simulate_gut_microbiome_pk,
)

# ---------------------------------------------------------------------------
# Default kwargs
# ---------------------------------------------------------------------------

DEFAULT_KW = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    f_abs=0.9,
    ka_per_h=1.0,
    k_microb_per_h=0.2,
    f_glucuronide=0.3,
    k_biliary_per_h=0.05,
    k_beta_gluc_per_h=1.0,
    ka_react_per_h=0.5,
    cl_L_per_h=5.0,
    vd_L=50.0,
    t_end_h=24.0,
    dt_h=0.05,
)


def _run(**overrides):
    kw = dict(DEFAULT_KW)
    kw.update(overrides)
    return simulate_gut_microbiome_pk(**kw)


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


class TestGutMicrobiomePKBasic:
    def test_returns_dataclass(self):
        result = _run()
        assert isinstance(result, GutMicrobiomePKResult)

    def test_cmax_positive(self):
        result = _run()
        assert result.cmax_mg_L > 0.0

    def test_auc_positive(self):
        result = _run()
        assert result.auc_mg_h_per_L > 0.0

    def test_times_start_at_zero(self):
        result = _run()
        assert result.times_h[0] == 0.0

    def test_times_end_approx_t_end(self):
        result = _run(t_end_h=12.0)
        assert abs(result.times_h[-1] - 12.0) < 0.1

    def test_list_lengths_match(self):
        result = _run()
        n = len(result.times_h)
        assert len(result.c_plasma_mg_L) == n
        assert len(result.a_glucuronide_mg) == n

    def test_drug_name_stored(self):
        result = _run(drug_name="Aspirin")
        assert result.drug_name == "Aspirin"

    def test_dose_mg_stored(self):
        result = _run(dose_mg=200.0)
        assert result.dose_mg == 200.0

    def test_f_glucuronide_stored(self):
        result = _run(f_glucuronide=0.5)
        assert result.f_glucuronide_used == 0.5

    def test_notes_nonempty(self):
        result = _run()
        assert isinstance(result.notes, str) and len(result.notes) > 0

    def test_secondary_peak_detected_is_bool(self):
        result = _run()
        assert isinstance(result.secondary_peak_detected, bool)


# ---------------------------------------------------------------------------
# Pharmacokinetic behaviour tests
# ---------------------------------------------------------------------------


class TestGutMicrobiomePKBehaviour:
    def test_high_k_beta_gluc_more_reactivation(self):
        r_low = _run(k_beta_gluc_per_h=0.1)
        r_high = _run(k_beta_gluc_per_h=5.0)
        assert r_high.reactivation_auc_contribution > r_low.reactivation_auc_contribution

    def test_zero_k_beta_gluc_no_reactivation(self):
        result = _run(k_beta_gluc_per_h=0.0)
        assert result.reactivation_auc_contribution == pytest.approx(0.0, abs=1e-6)

    def test_zero_k_microb_higher_auc(self):
        r_no_microb = _run(k_microb_per_h=0.0)
        r_high_microb = _run(k_microb_per_h=0.5)
        assert r_no_microb.auc_mg_h_per_L > r_high_microb.auc_mg_h_per_L

    def test_microbial_loss_fraction_formula(self):
        k_microb = 0.2
        avg_transit = 3.0
        expected = 1.0 - math.exp(-k_microb * avg_transit)
        result = _run(k_microb_per_h=k_microb)
        assert abs(result.microbial_loss_fraction - expected) < 1e-9

    def test_higher_dose_proportional_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
        assert abs(ratio - 2.0) < 0.05

    def test_higher_cl_lower_auc(self):
        r_low_cl = _run(cl_L_per_h=2.0)
        r_high_cl = _run(cl_L_per_h=20.0)
        assert r_low_cl.auc_mg_h_per_L > r_high_cl.auc_mg_h_per_L


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestGutMicrobiomePKValidation:
    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0.0)

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-50.0)

    def test_invalid_f_abs_zero(self):
        with pytest.raises(ValueError):
            _run(f_abs=0.0)

    def test_invalid_f_abs_above_one(self):
        with pytest.raises(ValueError):
            _run(f_abs=1.5)

    def test_invalid_cl_zero(self):
        with pytest.raises(ValueError):
            _run(cl_L_per_h=0.0)

    def test_invalid_vd_zero(self):
        with pytest.raises(ValueError):
            _run(vd_L=0.0)


# ---------------------------------------------------------------------------
# assess_microbiome_impact tests
# ---------------------------------------------------------------------------


class TestAssessMicrobiomeImpact:
    def test_returns_dict_with_impact_level(self):
        result = _run()
        assessment = assess_microbiome_impact(result)
        assert "impact_level" in assessment

    def test_impact_level_valid_values(self):
        result = _run()
        assessment = assess_microbiome_impact(result)
        assert assessment["impact_level"] in ("low", "moderate", "high")

    def test_returns_reactivation_pct(self):
        result = _run()
        assessment = assess_microbiome_impact(result)
        assert "reactivation_pct" in assessment

    def test_high_reactivation_gives_high_impact(self):
        # Very high k_beta_gluc and k_biliary → high reactivation AUC
        result = _run(k_beta_gluc_per_h=10.0, k_biliary_per_h=0.5, f_glucuronide=0.9)
        assessment = assess_microbiome_impact(result)
        # Reactivation AUC should be significant → moderate or high
        assert assessment["impact_level"] in ("moderate", "high")

    def test_zero_reactivation_gives_low_impact(self):
        result = _run(k_beta_gluc_per_h=0.0)
        assessment = assess_microbiome_impact(result)
        assert assessment["impact_level"] == "low"
        assert assessment["reactivation_pct"] == pytest.approx(0.0, abs=1e-6)
