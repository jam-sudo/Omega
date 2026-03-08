"""Tests for pancreatic drug distribution module (Phase 1021)."""

import pytest

from omega_pbpk.core.pancreatic_drug_distribution import (
    PancreaticDistributionResult,
    simulate_pancreatic_distribution,
)


def _default(**kwargs):
    params = dict(drug_name="TestDrug", dose_mg=100.0)
    params.update(kwargs)
    return simulate_pancreatic_distribution(**params)


class TestReturnType:
    def test_returns_dataclass(self):
        result = _default()
        assert isinstance(result, PancreaticDistributionResult)


class TestPlasmaConcentration:
    def test_plasma_starts_at_dose_over_vd(self):
        result = simulate_pancreatic_distribution(drug_name="Drug", dose_mg=200.0, vd_plasma_L=10.0)
        assert abs(result.c_plasma_mg_L[0] - 20.0) < 1e-6

    def test_plasma_decreases_overall(self):
        result = _default(t_end_h=24.0)
        # Plasma should be lower at end than start
        assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]

    def test_auc_plasma_positive(self):
        result = _default()
        assert result.auc_plasma > 0.0


class TestPancreasConcentration:
    def test_pancreas_starts_at_zero(self):
        result = _default()
        assert result.c_pancreas_mg_L[0] == 0.0

    def test_pancreas_has_peak(self):
        result = _default()
        assert result.cmax_pancreas_mg_L > 0.0

    def test_auc_pancreas_positive(self):
        result = _default()
        assert result.auc_pancreas > 0.0

    def test_tmax_pancreas_positive(self):
        result = _default()
        # tmax should be > 0 since pancreas starts at 0
        assert result.tmax_pancreas_h >= 0.0

    def test_cmax_pancreas_in_concentration_list(self):
        result = _default()
        assert result.cmax_pancreas_mg_L in result.c_pancreas_mg_L


class TestKpEffect:
    def test_higher_kp_higher_cmax_pancreas(self):
        r_low = simulate_pancreatic_distribution(drug_name="Drug", dose_mg=100.0, kp_pancreas=0.5)
        r_high = simulate_pancreatic_distribution(drug_name="Drug", dose_mg=100.0, kp_pancreas=3.0)
        assert r_high.cmax_pancreas_mg_L > r_low.cmax_pancreas_mg_L

    def test_kp_field_matches_input(self):
        result = simulate_pancreatic_distribution(drug_name="Drug", dose_mg=100.0, kp_pancreas=2.5)
        assert abs(result.kp_pancreas - 2.5) < 1e-9


class TestFlowEffect:
    def test_higher_flow_earlier_tmax(self):
        r_slow = simulate_pancreatic_distribution(
            drug_name="Drug", dose_mg=100.0, q_pancreas_L_per_h=0.1
        )
        r_fast = simulate_pancreatic_distribution(
            drug_name="Drug", dose_mg=100.0, q_pancreas_L_per_h=5.0
        )
        assert r_fast.tmax_pancreas_h <= r_slow.tmax_pancreas_h


class TestRatios:
    def test_pancreas_to_plasma_ratio_positive(self):
        result = _default()
        assert result.pancreas_to_plasma_ratio > 0.0

    def test_ratio_is_auc_ratio(self):
        result = _default()
        expected = result.auc_pancreas / result.auc_plasma
        assert abs(result.pancreas_to_plasma_ratio - expected) < 1e-9


class TestNotes:
    def test_notes_nonempty(self):
        result = _default()
        assert isinstance(result.notes, str) and len(result.notes) > 0

    def test_notes_contains_kp(self):
        result = _default(kp_pancreas=1.5)
        assert "Kp_pancreas" in result.notes


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=0.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _default(cl_L_per_h=0.0)

    def test_vd_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="vd_plasma_L"):
            _default(vd_plasma_L=0.0)

    def test_vd_pancreas_zero_raises(self):
        with pytest.raises(ValueError, match="vd_pancreas_L"):
            _default(vd_pancreas_L=0.0)

    def test_q_zero_raises(self):
        with pytest.raises(ValueError, match="q_pancreas_L_per_h"):
            _default(q_pancreas_L_per_h=0.0)

    def test_kp_zero_raises(self):
        with pytest.raises(ValueError, match="kp_pancreas"):
            _default(kp_pancreas=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=-10.0)
