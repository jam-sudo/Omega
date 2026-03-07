"""Tests for Phase 582 — immunosuppressant_pk.py"""

import pytest

from omega_pbpk.clinical.immunosuppressant_pk import (
    cyclosporine_dose_adjustment,
    simulate_nti_pk,
    tacrolimus_trough_target,
    tdm_dose_recommendation,
)

# ---------------------------------------------------------------------------
# tacrolimus_trough_target
# ---------------------------------------------------------------------------


class TestTacrolimusTroughTarget:
    def test_kidney_early_phase_target(self):
        r = tacrolimus_trough_target("kidney", 1)
        assert r["target_low_ng_mL"] == 10.0
        assert r["target_high_ng_mL"] == 15.0

    def test_kidney_middle_phase_target(self):
        r = tacrolimus_trough_target("kidney", 6)
        assert r["target_low_ng_mL"] == 8.0
        assert r["target_high_ng_mL"] == 12.0

    def test_kidney_late_phase_target(self):
        r = tacrolimus_trough_target("kidney", 18)
        assert r["target_low_ng_mL"] == 5.0
        assert r["target_high_ng_mL"] == 10.0

    def test_liver_early_target(self):
        r = tacrolimus_trough_target("liver", 2)
        assert r["target_low_ng_mL"] == 8.0
        assert r["target_high_ng_mL"] == 12.0

    def test_heart_early_target(self):
        r = tacrolimus_trough_target("heart", 1)
        assert r["target_low_ng_mL"] == 12.0
        assert r["target_high_ng_mL"] == 18.0

    def test_heart_late_target(self):
        r = tacrolimus_trough_target("heart", 24)
        assert r["target_low_ng_mL"] == 8.0
        assert r["target_high_ng_mL"] == 12.0

    def test_invalid_indication_raises(self):
        with pytest.raises(ValueError):
            tacrolimus_trough_target("lung", 3)

    def test_negative_months_raises(self):
        with pytest.raises(ValueError):
            tacrolimus_trough_target("kidney", -1)

    def test_notes_returned(self):
        r = tacrolimus_trough_target("kidney", 3)
        assert isinstance(r["notes"], str)
        assert len(r["notes"]) > 0


# ---------------------------------------------------------------------------
# cyclosporine_dose_adjustment
# ---------------------------------------------------------------------------


class TestCyclosporineDoseAdjustment:
    def test_proportional_increase(self):
        # current=100 ng/mL, target=200 ng/mL → 100% increase, capped at +50%
        r = cyclosporine_dose_adjustment(100.0, 200.0, 100.0)
        # Should be capped at +50%
        assert r["new_dose_mg"] == pytest.approx(150.0, rel=1e-3)
        assert r["warning"] is not None

    def test_simple_proportional_increase_within_50_pct(self):
        r = cyclosporine_dose_adjustment(150.0, 200.0, 100.0)
        expected = 100.0 * (200.0 / 150.0)
        assert r["new_dose_mg"] == pytest.approx(expected, rel=1e-3)
        assert r["warning"] is None

    def test_capped_at_plus_50_pct(self):
        r = cyclosporine_dose_adjustment(50.0, 200.0, 100.0)
        assert r["new_dose_mg"] == pytest.approx(150.0, rel=1e-3)
        assert r["warning"] is not None

    def test_capped_at_minus_50_pct(self):
        r = cyclosporine_dose_adjustment(400.0, 100.0, 200.0)
        assert r["new_dose_mg"] == pytest.approx(100.0, rel=1e-3)
        assert r["warning"] is not None

    def test_no_adjustment_same_trough(self):
        r = cyclosporine_dose_adjustment(150.0, 150.0, 100.0)
        assert r["new_dose_mg"] == pytest.approx(100.0, rel=1e-3)
        assert r["adjustment_pct"] == pytest.approx(0.0, abs=1e-6)
        assert r["warning"] is None

    def test_negative_current_trough_raises(self):
        with pytest.raises(ValueError):
            cyclosporine_dose_adjustment(-10.0, 150.0, 100.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            cyclosporine_dose_adjustment(100.0, 150.0, -50.0)


# ---------------------------------------------------------------------------
# simulate_nti_pk
# ---------------------------------------------------------------------------


class TestSimulateNtiPk:
    def _default_sim(self, **kwargs):
        params = dict(
            drug_name="tacrolimus",
            dose_mg=5.0,
            cl_L_per_h=2.0,
            vd_L=50.0,
            f_oral=0.25,
            therapeutic_low=0.005,
            therapeutic_high=0.020,
            t_end_h=24.0,
            dt_h=0.5,
        )
        params.update(kwargs)
        return simulate_nti_pk(**params)

    def test_cmax_positive(self):
        r = self._default_sim()
        assert r["cmax"] > 0

    def test_trough_positive(self):
        r = self._default_sim()
        assert r["trough"] >= 0

    def test_fraction_in_window_between_0_and_1(self):
        r = self._default_sim()
        assert 0.0 <= r["fraction_in_window"] <= 1.0

    def test_times_and_conc_same_length(self):
        r = self._default_sim()
        assert len(r["times_h"]) == len(r["c_plasma_mg_L"])

    def test_cmax_greater_than_trough(self):
        r = self._default_sim()
        assert r["cmax"] >= r["trough"]

    def test_supratherapeutic_flag_when_high_dose(self):
        # Use a tiny window so high dose goes supratherapeutic
        r = self._default_sim(dose_mg=50.0, therapeutic_low=0.0001, therapeutic_high=0.001)
        assert r["is_supratherapeutic"] is True

    def test_subtherapeutic_flag_when_low_dose(self):
        # Use a very high therapeutic window so tiny dose is always sub-therapeutic
        r = self._default_sim(dose_mg=0.001, therapeutic_low=10.0, therapeutic_high=20.0)
        assert r["is_subtherapeutic"] is True

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            self._default_sim(dose_mg=0.0)

    def test_invalid_f_oral_raises(self):
        with pytest.raises(ValueError):
            self._default_sim(f_oral=0.0)

    def test_therapeutic_window_reversed_raises(self):
        with pytest.raises(ValueError):
            self._default_sim(therapeutic_low=5.0, therapeutic_high=1.0)


# ---------------------------------------------------------------------------
# tdm_dose_recommendation
# ---------------------------------------------------------------------------


class TestTdmDoseRecommendation:
    def test_below_target_gives_increase(self):
        r = tdm_dose_recommendation(5.0, (8.0, 12.0), 2.0)
        assert r["recommendation"] == "increase"
        assert r["new_dose_mg"] > 2.0

    def test_above_target_gives_decrease(self):
        r = tdm_dose_recommendation(20.0, (8.0, 12.0), 4.0)
        assert r["recommendation"] == "decrease"
        assert r["new_dose_mg"] < 4.0

    def test_within_target_gives_maintain(self):
        r = tdm_dose_recommendation(10.0, (8.0, 12.0), 3.0)
        assert r["recommendation"] == "maintain"

    def test_new_dose_rounded_to_half_mg(self):
        r = tdm_dose_recommendation(5.0, (8.0, 12.0), 2.0)
        # Should be rounded to nearest 0.5
        assert (r["new_dose_mg"] * 2) == round(r["new_dose_mg"] * 2)

    def test_rationale_returned(self):
        r = tdm_dose_recommendation(10.0, (8.0, 12.0), 3.0)
        assert isinstance(r["rationale"], str)

    def test_invalid_trough_raises(self):
        with pytest.raises(ValueError):
            tdm_dose_recommendation(0.0, (8.0, 12.0), 3.0)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            tdm_dose_recommendation(10.0, (8.0, 12.0), 0.0)

    def test_invalid_target_range_raises(self):
        with pytest.raises(ValueError):
            tdm_dose_recommendation(10.0, (12.0, 8.0), 3.0)
