"""Tests for GI absorption window modeling (Phase 65)."""
from __future__ import annotations

import pytest

from omega_pbpk.core.absorption_window import (
    AbsorptionWindowResult,
    GI_SEGMENTS,
    simulate_absorption_window,
    compare_fed_fasted_absorption,
)


class TestGISegments:
    def test_has_5_segments(self):
        assert len(GI_SEGMENTS) == 5

    def test_segment_keys(self):
        for seg in GI_SEGMENTS:
            for key in ("name", "pH", "transit_h", "length_cm", "radius_cm"):
                assert key in seg


class TestSimulateAbsorptionWindow:
    def _run(self, **kw):
        defaults = dict(drug_name="TestDrug", dose_mg=100.0, logP=2.0,
                        sol_ph7_mg_mL=5.0, dose_volume_mL=240.0)
        defaults.update(kw)
        return simulate_absorption_window(**defaults)

    def test_returns_result_type(self):
        assert isinstance(self._run(), AbsorptionWindowResult)

    def test_fa_bounded_0_to_1(self):
        assert 0.0 <= self._run().fa <= 1.0

    def test_total_absorbed_leq_dose(self):
        r = self._run(dose_mg=100.0)
        assert r.total_absorbed_mg <= 100.0 + 1e-6

    def test_high_logP_high_absorption(self):
        r_high = self._run(logP=3.0, sol_ph7_mg_mL=1.0)
        r_low = self._run(logP=-2.0, sol_ph7_mg_mL=1.0)
        assert r_high.fa >= r_low.fa

    def test_segment_absorption_is_dict(self):
        r = self._run()
        assert isinstance(r.segment_absorption, dict)

    def test_segment_keys_present(self):
        r = self._run()
        # At least one GI segment key should be present
        segment_names = {seg["name"] for seg in GI_SEGMENTS}
        assert any(k in segment_names for k in r.segment_absorption.keys())

    def test_optimal_window_nonneg(self):
        r = self._run()
        assert r.optimal_window_h >= 0.0

    def test_dose_scaling(self):
        r100 = self._run(dose_mg=100.0, logP=2.0, sol_ph7_mg_mL=10.0)
        r200 = self._run(dose_mg=200.0, logP=2.0, sol_ph7_mg_mL=10.0)
        # Both doses should absorb similarly as fraction (not solubility limited)
        assert abs(r100.fa - r200.fa) < 0.3

    def test_drug_name_preserved(self):
        r = self._run(drug_name="Metformin")
        assert r.drug_name == "Metformin"

    def test_high_solubility_increases_absorption(self):
        r_high = self._run(sol_ph7_mg_mL=10.0)
        r_low = self._run(sol_ph7_mg_mL=0.001)
        assert r_high.total_absorbed_mg >= r_low.total_absorbed_mg

    def test_acid_drug(self):
        r = self._run(drug_type="acid", pka=4.5)
        assert 0.0 <= r.fa <= 1.0

    def test_base_drug(self):
        r = self._run(drug_type="base", pka=8.5)
        assert 0.0 <= r.fa <= 1.0

    def test_neutral_drug(self):
        r = self._run(drug_type="neutral")
        assert 0.0 <= r.fa <= 1.0

    def test_warnings_is_list(self):
        r = self._run()
        assert isinstance(r.warnings, list)


class TestFedFasted:
    def test_returns_dict_with_keys(self):
        result = compare_fed_fasted_absorption("Drug", 100.0, 2.0)
        assert "fasted" in result and "fed" in result

    def test_both_results_valid(self):
        result = compare_fed_fasted_absorption("Drug", 100.0, 2.0)
        assert isinstance(result["fasted"], AbsorptionWindowResult)
        assert isinstance(result["fed"], AbsorptionWindowResult)

    def test_both_have_positive_absorbed(self):
        result = compare_fed_fasted_absorption("Drug", 100.0, 2.0)
        assert result["fasted"].total_absorbed_mg > 0
        assert result["fed"].total_absorbed_mg > 0

    def test_fa_values_reasonable(self):
        result = compare_fed_fasted_absorption("Drug", 100.0, 2.0)
        assert 0 <= result["fasted"].fa <= 1
        assert 0 <= result["fed"].fa <= 1
