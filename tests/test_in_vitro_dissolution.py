"""Tests for USP II in vitro dissolution (Phase 114)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.in_vitro_dissolution import (
    InVitroDissolutionResult,
    compare_rpm,
    simulate_usp2_dissolution,
)


def _default(**kw):
    defaults = dict(
        drug_name="Drug",
        dose_mg=100.0,
        solubility_mg_mL=1.0,
        particle_radius_um=20.0,
        rpm=50,
        t_end_min=60.0,
        dt_min=1.0,
    )
    defaults.update(kw)
    return simulate_usp2_dissolution(**defaults)


class TestUSP2Dissolution:
    def test_returns_result_type(self):
        assert isinstance(_default(), InVitroDissolutionResult)

    def test_dissolved_pct_between_0_and_100(self):
        r = _default()
        assert all(0.0 <= p <= 100.0 for p in r.dissolved_pct)

    def test_dissolved_pct_non_decreasing(self):
        r = _default()
        pcts = r.dissolved_pct
        for i in range(1, len(pcts)):
            assert pcts[i] >= pcts[i - 1] - 1e-9

    def test_times_and_pct_same_length(self):
        r = _default()
        assert len(r.times_min) == len(r.dissolved_pct)

    def test_t50_within_simulation(self):
        r = _default()
        assert 0.0 <= r.t50_min <= 60.0

    def test_de_pct_positive(self):
        r = _default()
        assert r.de_pct > 0.0

    def test_classification_valid(self):
        r = _default()
        assert r.classification in ("rapid", "moderate", "slow")

    def test_apparatus_is_usp2(self):
        r = _default()
        assert r.apparatus == "USP II"

    def test_smaller_particle_dissolves_faster(self):
        r_small = _default(particle_radius_um=5.0)
        r_large = _default(particle_radius_um=100.0)
        assert r_small.dissolved_pct[-1] >= r_large.dissolved_pct[-1] - 1e-6

    def test_higher_solubility_faster(self):
        r_high = _default(solubility_mg_mL=5.0)
        r_low = _default(solubility_mg_mL=0.1)
        assert r_high.dissolved_pct[-1] >= r_low.dissolved_pct[-1] - 1e-6

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0.0)

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError):
            _default(solubility_mg_mL=0.0)

    def test_invalid_volume_raises(self):
        with pytest.raises(ValueError):
            _default(volume_mL=0.0)


class TestCompareRPM:
    def test_returns_list(self):
        results = compare_rpm("Drug", 100.0, 1.0, [25, 50, 100])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_result_types(self):
        results = compare_rpm("Drug", 100.0, 1.0)
        assert all(isinstance(r, InVitroDissolutionResult) for r in results)

    def test_higher_rpm_dissolves_faster(self):
        results = compare_rpm("Drug", 100.0, 0.5, [25, 100])
        r_low = results[0]
        r_high = results[1]
        assert r_high.dissolved_pct[-1] >= r_low.dissolved_pct[-1] - 1e-6
