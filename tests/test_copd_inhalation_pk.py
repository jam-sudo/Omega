"""Tests for COPD inhalation PK simulation (Phase 377)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.copd_inhalation_pk import (
    COPDInhalationResult,
    compare_copd_severity,
    simulate_copd_inhalation_pk,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _default(**kw):
    defaults = dict(
        drug_name="Salbutamol",
        dose_mg=2.5,
        device_type="MDI",
        cl_L_per_h=10.0,
        vd_L=50.0,
        fpf_normal=0.30,
        t_end_h=12.0,
        dt_h=0.1,
    )
    defaults.update(kw)
    return simulate_copd_inhalation_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_copd_inhalation_result(self):
        assert isinstance(_default(), COPDInhalationResult)

    def test_drug_name_preserved(self):
        r = _default(drug_name="Budesonide")
        assert r.drug_name == "Budesonide"

    def test_device_type_preserved(self):
        r = _default(device_type="DPI")
        assert r.device_type == "DPI"

    def test_copd_factor_preserved(self):
        r = _default(copd_factor=0.55)
        assert abs(r.copd_factor - 0.55) < 1e-9


# ---------------------------------------------------------------------------
# Concentrations are physically valid
# ---------------------------------------------------------------------------

class TestConcentrations:
    def test_c_plasma_normal_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.c_plasma_normal)

    def test_c_plasma_copd_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.c_plasma_copd)

    def test_cmax_normal_positive(self):
        r = _default()
        assert r.cmax_normal > 0.0

    def test_cmax_copd_positive(self):
        r = _default()
        assert r.cmax_copd > 0.0

    def test_auc_normal_positive(self):
        r = _default()
        assert r.auc_normal > 0.0

    def test_auc_copd_positive(self):
        r = _default()
        assert r.auc_copd > 0.0

    def test_times_and_profiles_same_length(self):
        r = _default()
        assert len(r.times_h) == len(r.c_plasma_normal) == len(r.c_plasma_copd)


# ---------------------------------------------------------------------------
# COPD reduces exposure vs normal
# ---------------------------------------------------------------------------

class TestCOPDvsNormal:
    def test_auc_copd_less_than_normal(self):
        r = _default()
        assert r.auc_copd < r.auc_normal

    def test_cmax_copd_less_than_normal(self):
        r = _default()
        assert r.cmax_copd < r.cmax_normal

    def test_auc_ratio_between_0_and_1(self):
        r = _default()
        assert 0.0 < r.auc_ratio_copd_normal < 1.0

    def test_lung_dose_copd_less_than_normal(self):
        r = _default()
        assert r.lung_dose_copd_mg < r.lung_dose_normal_mg

    def test_lung_dose_normal_equals_dose_times_fpf(self):
        r = _default(dose_mg=5.0, fpf_normal=0.25)
        assert abs(r.lung_dose_normal_mg - 5.0 * 0.25) < 1e-9

    def test_lung_dose_copd_equals_normal_times_factor(self):
        r = _default(dose_mg=5.0, fpf_normal=0.25, copd_factor=0.55)
        expected = 5.0 * 0.25 * 0.55
        assert abs(r.lung_dose_copd_mg - expected) < 1e-9

    def test_severe_copd_lower_auc_than_mild(self):
        mild = _default(copd_factor=0.85)
        severe = _default(copd_factor=0.40)
        assert severe.auc_copd < mild.auc_copd


# ---------------------------------------------------------------------------
# Device type effects
# ---------------------------------------------------------------------------

class TestDeviceTypes:
    def test_mdi_simulates(self):
        r = _default(device_type="MDI")
        assert r.auc_copd > 0

    def test_dpi_simulates(self):
        r = _default(device_type="DPI")
        assert r.auc_copd > 0

    def test_nebulizer_simulates(self):
        r = _default(device_type="nebulizer")
        assert r.auc_copd > 0

    def test_mdi_higher_ka_than_nebulizer_leads_to_higher_cmax(self):
        # MDI k_abs=3.0 > nebulizer k_abs=1.5, same lung dose → higher Cmax for MDI
        mdi = _default(device_type="MDI")
        neb = _default(device_type="nebulizer")
        assert mdi.cmax_normal > neb.cmax_normal


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            _default(drug_name="")

    def test_whitespace_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            _default(drug_name="   ")

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=-1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=0.0)

    def test_invalid_device_type_raises(self):
        with pytest.raises(ValueError, match="device_type"):
            _default(device_type="turbo")

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _default(cl_L_per_h=-5.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _default(vd_L=-10.0)

    def test_fpf_zero_raises(self):
        with pytest.raises(ValueError, match="fpf_normal"):
            _default(fpf_normal=0.0)

    def test_fpf_above_1_raises(self):
        with pytest.raises(ValueError, match="fpf_normal"):
            _default(fpf_normal=1.5)

    def test_copd_factor_zero_raises(self):
        with pytest.raises(ValueError, match="copd_factor"):
            _default(copd_factor=0.0)

    def test_copd_factor_above_1_raises(self):
        with pytest.raises(ValueError, match="copd_factor"):
            _default(copd_factor=1.5)

    def test_negative_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            _default(t_end_h=-1.0)

    def test_dt_too_large_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            _default(t_end_h=1.0, dt_h=2.0)


# ---------------------------------------------------------------------------
# compare_copd_severity
# ---------------------------------------------------------------------------

class TestCompareCOPDSeverity:
    def _run(self, **kw):
        defaults = dict(
            drug_name="Salbutamol",
            dose_mg=2.5,
            device_type="MDI",
            cl_L_per_h=10.0,
            vd_L=50.0,
            fpf_normal=0.30,
        )
        defaults.update(kw)
        return compare_copd_severity(**defaults)

    def test_returns_dict(self):
        assert isinstance(self._run(), dict)

    def test_has_four_stages(self):
        result = self._run()
        assert result["stages"] == ["I", "II", "III", "IV"]

    def test_four_results(self):
        result = self._run()
        assert len(result["results"]) == 4

    def test_gold_iv_lower_auc_than_gold_i(self):
        result = self._run()
        assert result["auc_by_stage"]["IV"] < result["auc_by_stage"]["I"]

    def test_gold_iv_lower_auc_than_gold_ii(self):
        result = self._run()
        assert result["auc_by_stage"]["IV"] < result["auc_by_stage"]["II"]

    def test_monotonic_decrease_across_stages(self):
        result = self._run()
        aucs = [result["auc_by_stage"][s] for s in ["I", "II", "III", "IV"]]
        for i in range(len(aucs) - 1):
            assert aucs[i] > aucs[i + 1]

    def test_all_auc_ratios_between_0_and_1(self):
        result = self._run()
        for stage, ratio in result["auc_ratios"].items():
            assert 0.0 < ratio < 1.0, f"Stage {stage}: ratio={ratio}"

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="device_type"):
            self._run(device_type="spacer")
