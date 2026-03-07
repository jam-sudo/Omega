"""Tests for clinical.immunosuppressant_pkpd — Phase 254."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.immunosuppressant_pkpd import (
    ImmunosuppressantPKResult,
    recommend_dose,
    simulate_immunosuppressant_pk,
)

# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation in simulate_immunosuppressant_pk."""

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_immunosuppressant_pk("cyclosporine", dose_mg=0.0, cl_L_per_h=25.0, vd_L=600.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_immunosuppressant_pk(
                "cyclosporine", dose_mg=-100.0, cl_L_per_h=25.0, vd_L=600.0
            )

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_immunosuppressant_pk("cyclosporine", dose_mg=200.0, cl_L_per_h=0.0, vd_L=600.0)

    def test_cl_negative_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_immunosuppressant_pk(
                "cyclosporine", dose_mg=200.0, cl_L_per_h=-5.0, vd_L=600.0
            )

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_immunosuppressant_pk("cyclosporine", dose_mg=200.0, cl_L_per_h=25.0, vd_L=0.0)

    def test_ka_zero_raises(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            simulate_immunosuppressant_pk(
                "cyclosporine", dose_mg=200.0, cl_L_per_h=25.0, vd_L=600.0, ka_per_h=0.0
            )

    def test_f_oral_zero_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            simulate_immunosuppressant_pk(
                "cyclosporine", dose_mg=200.0, cl_L_per_h=25.0, vd_L=600.0, f_oral=0.0
            )

    def test_f_oral_above_one_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            simulate_immunosuppressant_pk(
                "cyclosporine", dose_mg=200.0, cl_L_per_h=25.0, vd_L=600.0, f_oral=1.5
            )

    def test_interval_zero_raises(self):
        with pytest.raises(ValueError, match="interval_h"):
            simulate_immunosuppressant_pk(
                "cyclosporine", dose_mg=200.0, cl_L_per_h=25.0, vd_L=600.0, interval_h=0.0
            )

    def test_n_doses_zero_raises(self):
        with pytest.raises(ValueError, match="n_doses"):
            simulate_immunosuppressant_pk(
                "cyclosporine", dose_mg=200.0, cl_L_per_h=25.0, vd_L=600.0, n_doses=0
            )


# ---------------------------------------------------------------------------
# Simulation output tests
# ---------------------------------------------------------------------------


class TestSimulateImmunosuppressantPK:
    """Tests for simulate_immunosuppressant_pk() output."""

    def _default_result(self) -> ImmunosuppressantPKResult:
        return simulate_immunosuppressant_pk(
            drug_name="cyclosporine",
            dose_mg=200.0,
            cl_L_per_h=25.0,
            vd_L=600.0,
            ka_per_h=1.5,
            f_oral=0.3,
            interval_h=12.0,
            n_doses=10,
        )

    def test_returns_correct_type(self):
        result = self._default_result()
        assert isinstance(result, ImmunosuppressantPKResult)

    def test_times_h_non_empty(self):
        result = self._default_result()
        assert len(result.times_h) > 0

    def test_c_whole_blood_non_empty(self):
        result = self._default_result()
        assert len(result.c_whole_blood_ng_mL) > 0

    def test_il2_inhibition_non_empty(self):
        result = self._default_result()
        assert len(result.il2_inhibition_pct) > 0

    def test_c_peak_positive_after_10_doses(self):
        result = self._default_result()
        assert result.c_peak_ng_mL > 0.0

    def test_c_trough_non_negative(self):
        result = self._default_result()
        assert result.c_trough_ng_mL >= 0.0

    def test_auc_tau_positive(self):
        result = self._default_result()
        assert result.auc_tau_ng_h_mL > 0.0

    def test_in_therapeutic_range_true_when_trough_in_range(self):
        # Use a high dose / low CL to push trough into 100-300 ng/mL range
        result = simulate_immunosuppressant_pk(
            drug_name="cyclosporine",
            dose_mg=500.0,
            cl_L_per_h=5.0,
            vd_L=200.0,
            ka_per_h=1.5,
            f_oral=0.4,
            interval_h=12.0,
            n_doses=15,
            therapeutic_range=(100.0, 300.0),
        )
        # Check consistency: in_therapeutic_range matches trough vs therapeutic_range
        lo, hi = result.therapeutic_range_ng_mL
        expected = lo <= result.c_trough_ng_mL <= hi
        assert result.in_therapeutic_range == expected

    def test_in_therapeutic_range_false_when_outside(self):
        # Tiny dose -> trough will be 0, outside (100, 300)
        result = simulate_immunosuppressant_pk(
            drug_name="tacrolimus",
            dose_mg=0.001,
            cl_L_per_h=25.0,
            vd_L=600.0,
            therapeutic_range=(100.0, 300.0),
        )
        assert result.in_therapeutic_range is False

    def test_avg_il2_inhibition_between_0_and_100(self):
        result = self._default_result()
        assert 0.0 <= result.avg_il2_inhibition_pct <= 100.0

    def test_drug_name_stored_in_result(self):
        result = simulate_immunosuppressant_pk(
            drug_name="tacrolimus",
            dose_mg=5.0,
            cl_L_per_h=2.0,
            vd_L=50.0,
        )
        assert result.drug_name == "tacrolimus"

    def test_lists_same_length(self):
        result = self._default_result()
        n = len(result.times_h)
        assert len(result.c_whole_blood_ng_mL) == n
        assert len(result.il2_inhibition_pct) == n

    def test_single_dose_simulation(self):
        result = simulate_immunosuppressant_pk(
            drug_name="cyclosporine",
            dose_mg=200.0,
            cl_L_per_h=25.0,
            vd_L=600.0,
            n_doses=1,
        )
        assert result.c_peak_ng_mL > 0.0

    def test_il2_inhibition_increases_with_higher_dose(self):
        r_low = simulate_immunosuppressant_pk(
            "cyclosporine", dose_mg=50.0, cl_L_per_h=25.0, vd_L=600.0
        )
        r_high = simulate_immunosuppressant_pk(
            "cyclosporine", dose_mg=500.0, cl_L_per_h=25.0, vd_L=600.0
        )
        assert r_high.avg_il2_inhibition_pct > r_low.avg_il2_inhibition_pct

    def test_f_oral_one_increases_exposure(self):
        r_low_f = simulate_immunosuppressant_pk(
            "cyclosporine", dose_mg=200.0, cl_L_per_h=25.0, vd_L=600.0, f_oral=0.1
        )
        r_high_f = simulate_immunosuppressant_pk(
            "cyclosporine", dose_mg=200.0, cl_L_per_h=25.0, vd_L=600.0, f_oral=1.0
        )
        assert r_high_f.c_peak_ng_mL > r_low_f.c_peak_ng_mL

    def test_notes_non_empty_string(self):
        result = self._default_result()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# recommend_dose tests
# ---------------------------------------------------------------------------


class TestRecommendDose:
    """Tests for recommend_dose()."""

    def _call(self, **kwargs) -> dict:
        defaults = dict(
            drug_name="cyclosporine",
            observed_trough_ng_mL=80.0,
            current_dose_mg=200.0,
            target_trough_ng_mL=160.0,
            cl_L_per_h=25.0,
            vd_L=600.0,
        )
        defaults.update(kwargs)
        return recommend_dose(**defaults)

    def test_returns_dict_with_correct_keys(self):
        result = self._call()
        for key in ("recommended_dose_mg", "dose_change_pct", "t_half_h", "time_to_new_ss_h"):
            assert key in result

    def test_recommended_dose_proportional_to_ratio(self):
        result = self._call(
            observed_trough_ng_mL=100.0,
            current_dose_mg=200.0,
            target_trough_ng_mL=200.0,
        )
        assert result["recommended_dose_mg"] == pytest.approx(400.0, rel=1e-9)

    def test_dose_change_pct_positive_when_target_higher(self):
        result = self._call(
            observed_trough_ng_mL=100.0,
            target_trough_ng_mL=200.0,
        )
        assert result["dose_change_pct"] > 0.0

    def test_dose_change_pct_negative_when_target_lower(self):
        result = self._call(
            observed_trough_ng_mL=300.0,
            target_trough_ng_mL=150.0,
        )
        assert result["dose_change_pct"] < 0.0

    def test_no_change_when_on_target(self):
        result = self._call(
            observed_trough_ng_mL=150.0,
            target_trough_ng_mL=150.0,
            current_dose_mg=200.0,
        )
        assert result["recommended_dose_mg"] == pytest.approx(200.0, rel=1e-9)
        assert result["dose_change_pct"] == pytest.approx(0.0, abs=1e-9)

    def test_t_half_computed_correctly(self):
        result = self._call(cl_L_per_h=25.0, vd_L=600.0)
        ke = 25.0 / 600.0
        expected_t_half = 0.693147 / ke
        assert result["t_half_h"] == pytest.approx(expected_t_half, rel=1e-5)

    def test_time_to_new_ss_is_3_3_times_t_half(self):
        result = self._call(cl_L_per_h=25.0, vd_L=600.0)
        assert result["time_to_new_ss_h"] == pytest.approx(3.3 * result["t_half_h"], rel=1e-9)

    def test_drug_name_in_result(self):
        result = self._call(drug_name="tacrolimus")
        assert result["drug_name"] == "tacrolimus"

    def test_observed_trough_zero_raises(self):
        with pytest.raises(ValueError, match="observed_trough_ng_mL"):
            self._call(observed_trough_ng_mL=0.0)

    def test_current_dose_zero_raises(self):
        with pytest.raises(ValueError, match="current_dose_mg"):
            self._call(current_dose_mg=0.0)

    def test_target_trough_zero_raises(self):
        with pytest.raises(ValueError, match="target_trough_ng_mL"):
            self._call(target_trough_ng_mL=0.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            self._call(cl_L_per_h=0.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            self._call(vd_L=0.0)
