"""Tests for core/hydrogel_drug_release.py — Phase 1023."""

from __future__ import annotations

import pytest

from omega_pbpk.core.hydrogel_drug_release import (
    HydrogelReleaseResult,
    simulate_hydrogel_release,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> HydrogelReleaseResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        hydrogel_type="PEG",
        drug_diffusivity_cm2_h=1e-4,
        hydrogel_thickness_cm=0.2,
        burst_fraction=0.05,
        cl_L_per_h=5.0,
        vd_L=30.0,
        t_end_h=72.0,
        dt_h=0.5,
    )
    defaults.update(kwargs)
    return simulate_hydrogel_release(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        r = _default()
        assert isinstance(r, HydrogelReleaseResult)

    def test_drug_name_preserved(self):
        r = _default(drug_name="Hydrogel Drug")
        assert r.drug_name == "Hydrogel Drug"

    def test_hydrogel_type_preserved(self):
        r = _default(hydrogel_type="alginate")
        assert r.hydrogel_type == "alginate"

    def test_notes_nonempty(self):
        r = _default()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Fraction released
# ---------------------------------------------------------------------------


class TestFractionReleased:
    def test_starts_near_burst_fraction(self):
        burst = 0.05
        r = _default(burst_fraction=burst)
        assert r.fraction_released[0] == pytest.approx(burst, abs=1e-9)

    def test_increases_over_time(self):
        r = _default()
        assert r.fraction_released[-1] > r.fraction_released[0]

    def test_monotonically_nondecreasing(self):
        r = _default()
        for i in range(1, len(r.fraction_released)):
            assert r.fraction_released[i] >= r.fraction_released[i - 1] - 1e-12

    def test_never_exceeds_one(self):
        r = _default()
        for fr in r.fraction_released:
            assert fr <= 1.0 + 1e-9

    def test_zero_burst_starts_near_zero(self):
        r = _default(burst_fraction=0.0)
        assert r.fraction_released[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Plasma concentration
# ---------------------------------------------------------------------------


class TestPlasmaConc:
    def test_burst_nonzero_gives_initial_plasma_conc(self):
        r = _default(burst_fraction=0.1, dose_mg=100.0, vd_L=30.0)
        # Initial C = burst * dose / Vd = 0.1 * 100 / 30
        assert r.c_plasma_mg_L[0] == pytest.approx(0.1 * 100.0 / 30.0, rel=1e-6)

    def test_auc_plasma_positive(self):
        r = _default()
        assert r.auc_plasma > 0.0

    def test_cmax_plasma_positive(self):
        r = _default()
        assert r.cmax_plasma > 0.0

    def test_cmax_equals_max_plasma(self):
        r = _default()
        assert r.cmax_plasma == pytest.approx(max(r.c_plasma_mg_L), rel=1e-9)


# ---------------------------------------------------------------------------
# Release rate
# ---------------------------------------------------------------------------


class TestReleaseRate:
    def test_release_rate_nonnegative(self):
        r = _default()
        for rate in r.release_rate_mg_h:
            assert rate >= -1e-12

    def test_release_rate_decreases_over_time(self):
        # Rate should generally decrease as unreleased pool depletes
        r = _default()
        # Compare first half vs second half average
        mid = len(r.release_rate_mg_h) // 2
        avg_early = sum(r.release_rate_mg_h[:mid]) / mid
        avg_late = sum(r.release_rate_mg_h[mid:]) / (len(r.release_rate_mg_h) - mid)
        assert avg_late < avg_early


# ---------------------------------------------------------------------------
# t50 and t90 ordering
# ---------------------------------------------------------------------------


class TestReleaseTimings:
    def test_t50_lte_t90(self):
        r = _default()
        assert r.t50_release_h <= r.t90_release_h

    def test_t50_positive(self):
        r = _default()
        assert r.t50_release_h > 0.0


# ---------------------------------------------------------------------------
# Hydrogel type effect
# ---------------------------------------------------------------------------


class TestHydrogelTypeEffect:
    def test_plga_slower_than_peg(self):
        # Use a higher diffusivity so both reach 50% release, but PLGA takes longer
        r_peg = _default(hydrogel_type="PEG", drug_diffusivity_cm2_h=1e-2, t_end_h=200.0)
        r_plga = _default(hydrogel_type="PLGA", drug_diffusivity_cm2_h=1e-2, t_end_h=200.0)
        assert r_plga.t50_release_h > r_peg.t50_release_h

    def test_all_valid_types_run(self):
        for htype in ("PEG", "alginate", "chitosan", "PLGA", "collagen"):
            r = _default(hydrogel_type=htype)
            assert isinstance(r, HydrogelReleaseResult)


# ---------------------------------------------------------------------------
# Diffusivity and thickness effects
# ---------------------------------------------------------------------------


class TestParameterEffects:
    def test_higher_diffusivity_faster_release(self):
        r_slow = _default(drug_diffusivity_cm2_h=1e-5)
        r_fast = _default(drug_diffusivity_cm2_h=1e-3)
        assert r_fast.t50_release_h < r_slow.t50_release_h

    def test_thicker_hydrogel_slower_release(self):
        r_thin = _default(hydrogel_thickness_cm=0.1)
        r_thick = _default(hydrogel_thickness_cm=0.5)
        assert r_thick.t50_release_h > r_thin.t50_release_h


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=-10.0)

    def test_invalid_hydrogel_type_raises(self):
        with pytest.raises(ValueError):
            _default(hydrogel_type="unknown_polymer")

    def test_burst_fraction_one_raises(self):
        with pytest.raises(ValueError):
            _default(burst_fraction=1.0)

    def test_burst_fraction_above_one_raises(self):
        with pytest.raises(ValueError):
            _default(burst_fraction=1.5)

    def test_zero_diffusivity_raises(self):
        with pytest.raises(ValueError):
            _default(drug_diffusivity_cm2_h=0.0)

    def test_negative_diffusivity_raises(self):
        with pytest.raises(ValueError):
            _default(drug_diffusivity_cm2_h=-1e-4)

    def test_zero_thickness_raises(self):
        with pytest.raises(ValueError):
            _default(hydrogel_thickness_cm=0.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError):
            _default(cl_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            _default(vd_L=0.0)
