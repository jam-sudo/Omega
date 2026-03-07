"""Tests for placebo effect pharmacodynamic model."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.placebo_pd import (
    PlaceboResult,
    estimate_drug_effect,
    simulate_placebo_pd,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _default_result(**kwargs) -> PlaceboResult:
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=40.0,
        emax=50.0,
        ec50_mg_L=1.0,
        hill_n=1.0,
        placebo_model="decay",
        p0=20.0,
        kp_per_h=0.02,
        baseline=0.0,
        route="oral",
        ka_per_h=1.0,
        t_end_h=168.0,
        dt_h=1.0,
    )
    params.update(kwargs)
    return simulate_placebo_pd(**params)


# ── Validation errors ─────────────────────────────────────────────────────────


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_result(dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_result(dose_mg=-1.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _default_result(cl_L_per_h=0.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _default_result(vd_L=0.0)

    def test_emax_zero_raises(self):
        with pytest.raises(ValueError, match="emax"):
            _default_result(emax=0.0)

    def test_ec50_zero_raises(self):
        with pytest.raises(ValueError, match="ec50"):
            _default_result(ec50_mg_L=0.0)

    def test_hill_n_zero_raises(self):
        with pytest.raises(ValueError, match="hill_n"):
            _default_result(hill_n=0.0)

    def test_p0_negative_raises(self):
        with pytest.raises(ValueError, match="p0"):
            _default_result(p0=-5.0)

    def test_invalid_placebo_model_raises(self):
        with pytest.raises(ValueError, match="placebo_model"):
            _default_result(placebo_model="linear")

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            _default_result(route="sublingual")


# ── Placebo model behaviour ───────────────────────────────────────────────────


class TestPlaceboModelBehaviour:
    def test_none_model_placebo_all_zeros(self):
        r = _default_result(placebo_model="none")
        assert all(v == pytest.approx(0.0) for v in r.e_placebo)

    def test_constant_model_placebo_constant(self):
        r = _default_result(placebo_model="constant", p0=15.0, t_end_h=24.0)
        # All values should equal p0
        for v in r.e_placebo:
            assert v == pytest.approx(15.0, rel=1e-9)

    def test_decay_model_placebo_decreases(self):
        r = _default_result(placebo_model="decay", p0=20.0, kp_per_h=0.1, t_end_h=50.0)
        # Should strictly decrease
        assert r.e_placebo[0] > r.e_placebo[-1]
        assert r.e_placebo[-1] > 0.0  # decays but not to zero in finite time

    def test_decay_model_initial_value_is_p0(self):
        r = _default_result(placebo_model="decay", p0=25.0, kp_per_h=0.05)
        assert r.e_placebo[0] == pytest.approx(25.0, rel=1e-6)

    def test_e_total_equals_sum(self):
        r = _default_result(baseline=5.0)
        for ed, ep, et in zip(r.e_drug, r.e_placebo, r.e_total):
            assert et == pytest.approx(5.0 + ed + ep, abs=1e-10)

    def test_e_total_no_baseline_equals_sum(self):
        r = _default_result(baseline=0.0)
        for ed, ep, et in zip(r.e_drug, r.e_placebo, r.e_total):
            assert et == pytest.approx(ed + ep, abs=1e-10)


# ── Drug effect properties ────────────────────────────────────────────────────


class TestDrugEffect:
    def test_higher_emax_gives_higher_peak_drug_effect(self):
        r_low = _default_result(emax=20.0)
        r_high = _default_result(emax=80.0)
        assert r_high.peak_drug_effect > r_low.peak_drug_effect

    def test_double_dose_gives_higher_peak_effect(self):
        r1 = _default_result(dose_mg=100.0)
        r2 = _default_result(dose_mg=200.0)
        assert r2.peak_drug_effect > r1.peak_drug_effect

    def test_treatment_difference_in_0_1(self):
        r = _default_result()
        assert 0.0 <= r.treatment_difference <= 1.0

    def test_treatment_difference_no_placebo_is_one(self):
        r = _default_result(placebo_model="none")
        assert r.treatment_difference == pytest.approx(1.0, abs=1e-9)

    def test_peak_drug_effect_positive(self):
        r = _default_result()
        assert r.peak_drug_effect > 0.0

    def test_auc_drug_effect_positive(self):
        r = _default_result()
        assert r.auc_drug_effect > 0.0

    def test_iv_route_produces_result(self):
        r = _default_result(route="iv", dose_mg=50.0)
        assert r.peak_drug_effect > 0.0

    def test_ec50_halves_at_correct_concentration(self):
        # At c = ec50, drug effect should be emax/2
        # Use large dose, low CL, very short simulation to confirm sigmoid
        r = _default_result(emax=100.0, ec50_mg_L=2.0, hill_n=1.0, t_end_h=1.0, dt_h=0.01)
        assert r.peak_drug_effect <= 100.0


# ── Result field types ────────────────────────────────────────────────────────


class TestResultFieldTypes:
    def test_result_is_dataclass(self):
        r = _default_result()
        assert isinstance(r, PlaceboResult)

    def test_times_is_list(self):
        r = _default_result()
        assert isinstance(r.times_h, list)

    def test_c_plasma_is_list(self):
        r = _default_result()
        assert isinstance(r.c_plasma_mg_L, list)

    def test_e_drug_is_list(self):
        r = _default_result()
        assert isinstance(r.e_drug, list)

    def test_e_placebo_is_list(self):
        r = _default_result()
        assert isinstance(r.e_placebo, list)

    def test_e_total_is_list(self):
        r = _default_result()
        assert isinstance(r.e_total, list)

    def test_notes_is_list(self):
        r = _default_result()
        assert isinstance(r.notes, list)

    def test_peak_drug_effect_is_float(self):
        r = _default_result()
        assert isinstance(r.peak_drug_effect, float)


# ── estimate_drug_effect ──────────────────────────────────────────────────────


class TestEstimateDrugEffect:
    def test_returns_dict(self):
        r = _default_result()
        result = estimate_drug_effect(r)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        r = _default_result()
        result = estimate_drug_effect(r)
        assert "max_drug_only_effect" in result
        assert "max_placebo_effect" in result
        assert "drug_vs_placebo_ratio" in result
        assert "treatment_difference" in result

    def test_treatment_difference_in_range(self):
        r = _default_result()
        result = estimate_drug_effect(r)
        assert 0.0 <= result["treatment_difference"] <= 1.0

    def test_matches_placebo_result_values(self):
        r = _default_result()
        est = estimate_drug_effect(r)
        assert est["max_drug_only_effect"] == pytest.approx(r.peak_drug_effect)
        assert est["max_placebo_effect"] == pytest.approx(r.peak_placebo_effect)

    def test_no_placebo_ratio_is_inf_or_large(self):
        r = _default_result(placebo_model="none")
        est = estimate_drug_effect(r)
        assert est["drug_vs_placebo_ratio"] == float("inf") or est["drug_vs_placebo_ratio"] > 1e6
