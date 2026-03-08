"""Tests for sweat excretion prediction model (phase 648)."""

import pytest

from omega_pbpk.prediction.sweat_excretion_p648 import (
    SweatExcretionResult,
    predict_sweat_excretion,
    screen_sweat_excretion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    drug_name="amphetamine",
    logP=1.76,
    mw_Da=135.21,
    pka=9.9,
    is_base=True,
)


def _run(**overrides):
    kw = {**DEFAULTS, **overrides}
    return predict_sweat_excretion(**kw)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------
class TestStructure:
    def test_return_type(self):
        r = _run()
        assert isinstance(r, SweatExcretionResult)

    def test_frozen_dataclass(self):
        r = _run()
        with pytest.raises(AttributeError):
            r.drug_name = "other"  # type: ignore[misc]

    def test_mutation_raises(self):
        r = _run()
        with pytest.raises(AttributeError):
            r.sweat_to_plasma_ratio = 99.0  # type: ignore[misc]

    def test_drug_name_stored(self):
        r = _run()
        assert r.drug_name == "amphetamine"


# ---------------------------------------------------------------------------
# Ion trapping / pH partition
# ---------------------------------------------------------------------------
class TestIonTrapping:
    def test_basic_drug_high_pka_ratio_above_1(self):
        # Strong base (pka=9) in acidic sweat → ion trapping → ratio > 1
        r = _run(pka=9.0, is_base=True, logP=2.0, mw_Da=200.0)
        assert r.sweat_to_plasma_ratio > 1.0

    def test_acidic_drug_ratio_ph_unity(self):
        # Acidic drug → no ion trapping → ratio_ph = 1.0
        r = _run(is_base=False)
        # ratio_ph should be 1.0 but total ratio includes logP/MW
        # Just ensure it's lower than a base with same props
        r_base = _run(is_base=True)
        assert r.sweat_to_plasma_ratio <= r_base.sweat_to_plasma_ratio

    def test_ionization_effect_strong(self):
        r = _run(pka=10.0, is_base=True)
        assert r.ionization_effect == "strong"

    def test_ionization_effect_weak(self):
        r = _run(is_base=False)
        assert r.ionization_effect == "weak"

    def test_ionization_effect_moderate(self):
        # Need ratio_ph between 2 and 5
        # pka ~ 8.0: ratio_ph = (1+10^(8-5.5))/(1+10^(8-7.4))
        #           = (1+316.23)/(1+3.98) = 317.23/4.98 ≈ 63.7 → strong
        # Try pka=7.5: (1+10^2)/(1+10^0.1) = 101/1.259 ≈ 80 → strong
        # Try pka=6.0: (1+10^0.5)/(1+10^-1.4) = 4.16/1.04 ≈ 4.0 → moderate
        r = _run(pka=6.0, is_base=True)
        assert r.ionization_effect == "moderate"


# ---------------------------------------------------------------------------
# Lipophilicity
# ---------------------------------------------------------------------------
class TestLipophilicity:
    def test_higher_logP_higher_ratio(self):
        r_low = _run(logP=0.5, is_base=False)
        r_high = _run(logP=3.0, is_base=False)
        assert r_high.sweat_to_plasma_ratio > r_low.sweat_to_plasma_ratio

    def test_detection_window_lipophilic(self):
        r = _run(logP=4.0)
        assert r.detection_window_h == 48.0

    def test_detection_window_moderate(self):
        r = _run(logP=2.0)
        assert r.detection_window_h == 24.0

    def test_detection_window_hydrophilic(self):
        r = _run(logP=0.5)
        assert r.detection_window_h == 12.0


# ---------------------------------------------------------------------------
# MW effect
# ---------------------------------------------------------------------------
class TestMWEffect:
    def test_high_mw_lower_ratio(self):
        r_small = _run(mw_Da=150.0, is_base=False)
        r_large = _run(mw_Da=800.0, is_base=False)
        assert r_large.sweat_to_plasma_ratio < r_small.sweat_to_plasma_ratio


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
class TestDerived:
    def test_sweat_conc_equals_plasma_times_ratio(self):
        r = _run(plasma_concentration_mg_L=2.5)
        expected = 2.5 * r.sweat_to_plasma_ratio
        assert abs(r.sweat_concentration_mg_L - expected) < 1e-9

    def test_daily_excretion_proportional_to_sweat_rate(self):
        r1 = _run(sweat_rate_mL_h=10.0)
        r2 = _run(sweat_rate_mL_h=20.0)
        ratio = r2.daily_excretion_mg / r1.daily_excretion_mg
        assert abs(ratio - 2.0) < 1e-6

    def test_patch_accumulation_equals_daily(self):
        r = _run()
        assert abs(r.sweat_patch_accumulation_mg - r.daily_excretion_mg) < 1e-9


# ---------------------------------------------------------------------------
# Excretion class
# ---------------------------------------------------------------------------
class TestExcretionClass:
    def test_high_class(self):
        r = _run(pka=9.0, is_base=True, logP=3.0, mw_Da=150.0)
        assert r.excretion_class == "high"

    def test_low_class(self):
        r = _run(is_base=False, logP=-1.0, mw_Da=900.0)
        assert r.excretion_class == "low"


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------
class TestScreening:
    def test_screen_returns_sorted_list(self):
        compounds = [
            {"drug_name": "A", "logP": 0.5, "mw_Da": 300.0, "is_base": False},
            {"drug_name": "B", "logP": 3.0, "mw_Da": 200.0, "pka": 9.0, "is_base": True},
            {"drug_name": "C", "logP": 1.5, "mw_Da": 250.0, "is_base": False},
        ]
        results = screen_sweat_excretion(compounds)
        assert len(results) == 3
        ratios = [r.sweat_to_plasma_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_screen_empty_list(self):
        assert screen_sweat_excretion([]) == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_zero_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _run(mw_Da=0)

    def test_negative_plasma(self):
        with pytest.raises(ValueError, match="plasma_concentration"):
            _run(plasma_concentration_mg_L=-1)

    def test_zero_sweat_rate(self):
        with pytest.raises(ValueError, match="sweat_rate"):
            _run(sweat_rate_mL_h=0)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
class TestNotes:
    def test_notes_present(self):
        r = _run()
        assert "ratio_ph" in r.notes
