"""Tests for fat tissue PK model (Phase 418)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.fat_tissue_pk import (
    FatTissuePKResult,
    obesity_effect,
    simulate_fat_tissue_pk,
)


def _basic_iv(**kwargs):
    """Return a simple IV simulation result with overridable defaults."""
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=3.0,
        cl_L_per_h=5.0,
        vd_total_L=50.0,
        fat_fraction=0.20,
        fat_perfusion_rate_L_per_h=3.0,
        t_end_h=24.0,
        dt_h=0.1,
        route="iv",
    )
    defaults.update(kwargs)
    return simulate_fat_tissue_pk(**defaults)


class TestSimulateFatTissuePK:
    def test_returns_result_type(self):
        r = _basic_iv()
        assert isinstance(r, FatTissuePKResult)

    def test_times_length_matches_conc(self):
        r = _basic_iv()
        assert len(r.times_h) == len(r.c_central_mg_L)
        assert len(r.times_h) == len(r.c_fat_mg_L)

    def test_time_starts_at_zero(self):
        r = _basic_iv()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_time_ends_near_t_end(self):
        r = _basic_iv(t_end_h=24.0)
        assert r.times_h[-1] == pytest.approx(24.0, rel=1e-3)

    def test_iv_initial_conc_nonzero(self):
        r = _basic_iv()
        assert r.c_central_mg_L[0] > 0.0

    def test_oral_initial_conc_zero(self):
        r = _basic_iv(route="oral", ka_per_h=1.0)
        assert r.c_central_mg_L[0] == pytest.approx(0.0)

    def test_oral_has_absorption_peak(self):
        r = _basic_iv(route="oral", ka_per_h=1.0)
        # Concentration should rise then fall
        assert r.cmax_central > r.c_central_mg_L[0]

    def test_concentration_eventually_decreases(self):
        r = _basic_iv()
        # After t_end_h=24h, last conc should be less than initial
        assert r.c_central_mg_L[-1] < r.c_central_mg_L[0]

    def test_fat_conc_rises_for_lipophilic(self):
        r = _basic_iv(logP=4.0)
        # Fat compartment should accumulate drug
        assert max(r.c_fat_mg_L) > 0.0

    def test_cmax_central_positive(self):
        r = _basic_iv()
        assert r.cmax_central > 0.0

    def test_auc_central_positive(self):
        r = _basic_iv()
        assert r.auc_central > 0.0

    def test_t_half_positive(self):
        r = _basic_iv()
        assert r.t_half_h > 0.0

    def test_kp_fat_at_least_one(self):
        r = _basic_iv(logP=3.0)
        assert r.kp_fat >= 1.0

    def test_kp_fat_equals_logp_when_logp_gt_1(self):
        r = _basic_iv(logP=4.0)
        assert r.kp_fat == pytest.approx(4.0)

    def test_kp_fat_min_one_for_low_logp(self):
        r = _basic_iv(logP=0.5)
        assert r.kp_fat == pytest.approx(1.0)

    def test_higher_logp_more_fat_accumulation(self):
        r_low = _basic_iv(logP=1.0)
        r_high = _basic_iv(logP=5.0)
        assert max(r_high.c_fat_mg_L) > max(r_low.c_fat_mg_L)

    def test_higher_fat_fraction_affects_vd(self):
        r_low = _basic_iv(fat_fraction=0.10)
        r_high = _basic_iv(fat_fraction=0.35)
        # Different fat fractions should give different Cmax
        assert r_low.cmax_central != r_high.cmax_central

    def test_drug_name_stored(self):
        r = _basic_iv(drug_name="Amiodarone")
        assert r.drug_name == "Amiodarone"

    def test_dose_stored(self):
        r = _basic_iv(dose_mg=200.0)
        assert r.dose_mg == pytest.approx(200.0)

    def test_logp_stored(self):
        r = _basic_iv(logP=3.5)
        assert r.logP == pytest.approx(3.5)

    def test_fat_fraction_stored(self):
        r = _basic_iv(fat_fraction=0.25)
        assert r.fat_fraction == pytest.approx(0.25)

    def test_notes_non_empty(self):
        r = _basic_iv()
        assert len(r.notes) > 0

    def test_all_concentrations_non_negative(self):
        r = _basic_iv()
        assert all(c >= 0.0 for c in r.c_central_mg_L)
        assert all(c >= 0.0 for c in r.c_fat_mg_L)

    def test_invalid_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name must be a non-empty string"):
            simulate_fat_tissue_pk(
                drug_name="",
                dose_mg=100.0,
                logP=3.0,
                cl_L_per_h=5.0,
                vd_total_L=50.0,
                fat_fraction=0.20,
                fat_perfusion_rate_L_per_h=3.0,
                t_end_h=24.0,
            )

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg must be positive"):
            _basic_iv(dose_mg=-10.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_total_L must be positive"):
            _basic_iv(vd_total_L=0.0)

    def test_invalid_fat_fraction_raises(self):
        with pytest.raises(ValueError, match="fat_fraction must be in"):
            _basic_iv(fat_fraction=1.5)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route must be"):
            _basic_iv(route="subcutaneous")

    def test_invalid_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h must be positive"):
            _basic_iv(t_end_h=0.0)

    def test_invalid_dt_raises(self):
        with pytest.raises(ValueError, match="dt_h must be positive"):
            _basic_iv(dt_h=-0.1)


class TestObesityEffect:
    def test_returns_dict(self):
        result = obesity_effect(
            drug_name="LipophilicDrug",
            dose_mg=100.0,
            logP=4.0,
            cl_L_per_h=5.0,
            normal_bmi_vd=50.0,
            obese_bmi_vd=100.0,
        )
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = obesity_effect(
            drug_name="Drug",
            dose_mg=100.0,
            logP=3.0,
            cl_L_per_h=5.0,
            normal_bmi_vd=50.0,
            obese_bmi_vd=90.0,
        )
        assert "normal_bmi" in result
        assert "obese_bmi" in result
        assert "ratios_obese_vs_normal" in result

    def test_normal_fat_fraction_020(self):
        result = obesity_effect(
            drug_name="Drug",
            dose_mg=100.0,
            logP=3.0,
            cl_L_per_h=5.0,
            normal_bmi_vd=50.0,
            obese_bmi_vd=90.0,
        )
        assert result["normal_bmi"]["fat_fraction"] == pytest.approx(0.20)

    def test_obese_fat_fraction_040(self):
        result = obesity_effect(
            drug_name="Drug",
            dose_mg=100.0,
            logP=3.0,
            cl_L_per_h=5.0,
            normal_bmi_vd=50.0,
            obese_bmi_vd=90.0,
        )
        assert result["obese_bmi"]["fat_fraction"] == pytest.approx(0.40)

    def test_obese_has_higher_vd(self):
        result = obesity_effect(
            drug_name="Drug",
            dose_mg=100.0,
            logP=4.0,
            cl_L_per_h=5.0,
            normal_bmi_vd=50.0,
            obese_bmi_vd=100.0,
        )
        assert result["obese_bmi"]["vd_L"] > result["normal_bmi"]["vd_L"]

    def test_ratios_keys_present(self):
        result = obesity_effect(
            drug_name="Drug",
            dose_mg=100.0,
            logP=3.0,
            cl_L_per_h=5.0,
            normal_bmi_vd=50.0,
            obese_bmi_vd=90.0,
        )
        ratios = result["ratios_obese_vs_normal"]
        assert "cmax_ratio" in ratios
        assert "auc_ratio" in ratios
        assert "t_half_ratio" in ratios

    def test_drug_name_in_result(self):
        result = obesity_effect(
            drug_name="Amiodarone",
            dose_mg=200.0,
            logP=7.0,
            cl_L_per_h=2.0,
            normal_bmi_vd=60.0,
            obese_bmi_vd=120.0,
        )
        assert result["drug_name"] == "Amiodarone"

    def test_clinical_note_present(self):
        result = obesity_effect(
            drug_name="Drug",
            dose_mg=100.0,
            logP=3.0,
            cl_L_per_h=5.0,
            normal_bmi_vd=50.0,
            obese_bmi_vd=90.0,
        )
        assert "clinical_note" in result
        assert len(result["clinical_note"]) > 0

    def test_invalid_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name must be a non-empty string"):
            obesity_effect(
                drug_name="",
                dose_mg=100.0,
                logP=3.0,
                cl_L_per_h=5.0,
                normal_bmi_vd=50.0,
                obese_bmi_vd=90.0,
            )

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg must be positive"):
            obesity_effect(
                drug_name="Drug",
                dose_mg=0.0,
                logP=3.0,
                cl_L_per_h=5.0,
                normal_bmi_vd=50.0,
                obese_bmi_vd=90.0,
            )

    def test_invalid_normal_vd_raises(self):
        with pytest.raises(ValueError, match="normal_bmi_vd must be positive"):
            obesity_effect(
                drug_name="Drug",
                dose_mg=100.0,
                logP=3.0,
                cl_L_per_h=5.0,
                normal_bmi_vd=-10.0,
                obese_bmi_vd=90.0,
            )

    def test_invalid_obese_vd_raises(self):
        with pytest.raises(ValueError, match="obese_bmi_vd must be positive"):
            obesity_effect(
                drug_name="Drug",
                dose_mg=100.0,
                logP=3.0,
                cl_L_per_h=5.0,
                normal_bmi_vd=50.0,
                obese_bmi_vd=0.0,
            )

    def test_pk_metrics_positive(self):
        result = obesity_effect(
            drug_name="Drug",
            dose_mg=100.0,
            logP=3.0,
            cl_L_per_h=5.0,
            normal_bmi_vd=50.0,
            obese_bmi_vd=90.0,
        )
        assert result["normal_bmi"]["cmax_mg_L"] > 0.0
        assert result["obese_bmi"]["cmax_mg_L"] > 0.0
        assert result["normal_bmi"]["auc_mg_h_L"] > 0.0
        assert result["obese_bmi"]["auc_mg_h_L"] > 0.0
