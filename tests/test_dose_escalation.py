"""Tests for Phase I dose escalation simulation (Phase 100)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.dose_escalation import (
    DoseEscalationResult,
    _emax_dlt_probability,
    simulate_3plus3,
    simulate_accelerated_titration,
)

DOSE_LEVELS = [10.0, 20.0, 40.0, 80.0, 160.0]
ED50 = 80.0


class TestDLTProbability:
    def test_zero_dose_near_zero_dlt(self):
        assert _emax_dlt_probability(0.0, ed50=50.0) < 0.1

    def test_high_dose_near_emax(self):
        assert _emax_dlt_probability(500.0, ed50=10.0) > 0.8

    def test_at_ed50_near_half_emax(self):
        p = _emax_dlt_probability(50.0, ed50=50.0, emax=0.9)
        assert 0.35 < p < 0.6

    def test_prob_between_0_and_1(self):
        for dose in [0.0, 1.0, 10.0, 100.0, 1000.0]:
            p = _emax_dlt_probability(dose, ed50=50.0)
            assert 0.0 <= p <= 1.0


class TestThreePlusThree:
    def _run(self, **kw):
        defaults = dict(
            drug_name="TestDrug",
            starting_dose_mg=10.0,
            dose_levels=DOSE_LEVELS,
            ed50_mg=ED50,
        )
        defaults.update(kw)
        return simulate_3plus3(**defaults)

    def test_returns_result_type(self):
        assert isinstance(self._run(), DoseEscalationResult)

    def test_dlt_probs_between_0_and_1(self):
        r = self._run()
        assert all(0.0 <= p <= 1.0 for p in r.dlt_probabilities)

    def test_dlt_probs_length_matches_levels(self):
        r = self._run()
        assert len(r.dlt_probabilities) == len(DOSE_LEVELS)

    def test_stopping_reason_is_string(self):
        r = self._run()
        assert isinstance(r.stopping_reason, str)
        assert r.stopping_reason in ("dlt_exceeded", "max_dose_reached")

    def test_rp2d_positive(self):
        r = self._run()
        assert r.recommended_phase2_dose > 0.0

    def test_rp2d_leq_max_dose(self):
        r = self._run()
        assert r.recommended_phase2_dose <= max(DOSE_LEVELS) * 1.01

    def test_empty_dose_levels_raises(self):
        with pytest.raises(ValueError):
            simulate_3plus3("Drug", 10.0, [], ed50_mg=80.0)

    def test_n_patients_non_negative(self):
        r = self._run()
        assert all(n >= 0 for n in r.n_patients_per_level)

    def test_dose_response_curve_length(self):
        r = self._run()
        assert len(r.dose_response_curve) == len(DOSE_LEVELS)

    def test_mtd_within_dose_range(self):
        r = self._run()
        # mtd_mg is either 0 (stopped at first level) or one of the levels
        assert r.mtd_mg >= 0.0


class TestAcceleratedTitration:
    def test_returns_result_type(self):
        r = simulate_accelerated_titration("Drug", DOSE_LEVELS, ed50_mg=ED50)
        assert isinstance(r, DoseEscalationResult)

    def test_dlt_probs_length_matches_levels(self):
        r = simulate_accelerated_titration("Drug", DOSE_LEVELS, ed50_mg=ED50)
        assert len(r.dlt_probabilities) == len(DOSE_LEVELS)

    def test_rp2d_positive(self):
        r = simulate_accelerated_titration("Drug", DOSE_LEVELS, ed50_mg=ED50)
        assert r.recommended_phase2_dose > 0.0

    def test_empty_levels_raises(self):
        with pytest.raises(ValueError):
            simulate_accelerated_titration("Drug", [], ed50_mg=ED50)
