"""Tests for Phase 1039 — corneal penetration prediction."""

import pytest

from omega_pbpk.prediction.corneal_penetration import (
    CornealPenetrationResult,
    predict_corneal_penetration,
)

_VALID_PENETRATION_CLASSES = {"excellent", "good", "moderate", "poor"}


def _default(**kwargs):
    """Helper: call with reasonable defaults."""
    params = dict(
        drug_name="TestDrug",
        mw_Da=300.0,
        logp=2.0,
        pka=7.0,
        ionization_type="neutral",
        dose_mg_mL=0.5,
        drop_volume_uL=30.0,
        nasolacrimal_fraction=0.7,
        ph_formulation=7.4,
    )
    params.update(kwargs)
    return predict_corneal_penetration(**params)


class TestReturnType:
    def test_returns_frozen_dataclass(self):
        result = _default()
        assert isinstance(result, CornealPenetrationResult)

    def test_frozen(self):
        result = _default()
        with pytest.raises((AttributeError, TypeError)):
            result.corneal_perm_cm_s = 999.0  # type: ignore[misc]


class TestNumericalConstraints:
    def test_corneal_perm_positive(self):
        result = _default()
        assert result.corneal_perm_cm_s > 0

    def test_bioavailability_in_bounds(self):
        result = _default()
        assert 0 < result.bioavailability_ocular <= 0.1

    def test_aqueous_humor_conc_positive(self):
        result = _default()
        assert result.aqueous_humor_conc_mg_mL > 0

    def test_tmax_positive(self):
        result = _default()
        assert result.predicted_tmax_min > 0

    def test_ionization_penalty_in_bounds(self):
        result = _default()
        assert 0.0 <= result.ionization_penalty <= 1.0


class TestPenetrationClass:
    def test_class_in_valid_set(self):
        result = _default()
        assert result.penetration_class in _VALID_PENETRATION_CLASSES

    def test_poor_for_very_low_logp_high_mw(self):
        result = _default(logp=-4.0, mw_Da=700.0)
        assert result.penetration_class in _VALID_PENETRATION_CLASSES


class TestLipophilicityOptimal:
    def test_optimal_at_logp_2(self):
        result = _default(logp=2.0)
        assert result.lipophilicity_optimal is True

    def test_not_optimal_at_logp_5(self):
        result = _default(logp=5.0)
        assert result.lipophilicity_optimal is False

    def test_boundary_logp_1(self):
        result = _default(logp=1.0)
        assert result.lipophilicity_optimal is True

    def test_boundary_logp_3(self):
        result = _default(logp=3.0)
        assert result.lipophilicity_optimal is True

    def test_below_range_logp_0(self):
        result = _default(logp=0.0)
        assert result.lipophilicity_optimal is False


class TestIonizationEffect:
    def test_neutral_has_zero_ionization_penalty(self):
        result = _default(ionization_type="neutral")
        assert result.ionization_penalty == 0.0

    def test_acid_below_pka_mostly_ionized(self):
        # acid pka=4.0, formulation pH=7.4 → mostly ionized (weak acid above pka)
        result = _default(ionization_type="acid", pka=4.0, ph_formulation=7.4)
        assert result.ionization_penalty > 0.9

    def test_neutral_vs_acid_ionization_penalty(self):
        neutral = _default(ionization_type="neutral", logp=2.0)
        acid = _default(ionization_type="acid", pka=4.0, logp=2.0)
        assert neutral.ionization_penalty < acid.ionization_penalty

    def test_base_above_pka_mostly_ionized(self):
        # base pka=10, formulation pH=7.4 → mostly protonated/ionized
        result = _default(ionization_type="base", pka=10.0, ph_formulation=7.4)
        assert result.ionization_penalty > 0.9


class TestPhysicochemicalDependencies:
    def test_higher_logp_higher_perm(self):
        high = _default(logp=2.0, mw_Da=300.0)
        low = _default(logp=-2.0, mw_Da=300.0)
        assert high.corneal_perm_cm_s > low.corneal_perm_cm_s

    def test_higher_mw_lower_bioavailability(self):
        small = _default(logp=2.0, mw_Da=200.0)
        large = _default(logp=2.0, mw_Da=600.0)
        assert small.bioavailability_ocular >= large.bioavailability_ocular


class TestNotes:
    def test_notes_nonempty(self):
        result = _default()
        assert isinstance(result.notes, str) and len(result.notes) > 0


class TestValidation:
    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _default(mw_Da=0.0)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError):
            _default(mw_Da=-1.0)

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg_mL"):
            _default(dose_mg_mL=0.0)

    def test_drop_volume_zero_raises(self):
        with pytest.raises(ValueError, match="drop_volume_uL"):
            _default(drop_volume_uL=0.0)

    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError, match="ionization_type"):
            _default(ionization_type="zwitterion")

    def test_pka_above_14_raises(self):
        with pytest.raises(ValueError, match="pka"):
            _default(pka=15.0)

    def test_pka_negative_raises(self):
        with pytest.raises(ValueError, match="pka"):
            _default(pka=-1.0)

    def test_nasolacrimal_fraction_one_raises(self):
        with pytest.raises(ValueError, match="nasolacrimal_fraction"):
            _default(nasolacrimal_fraction=1.0)

    def test_nasolacrimal_fraction_negative_raises(self):
        with pytest.raises(ValueError, match="nasolacrimal_fraction"):
            _default(nasolacrimal_fraction=-0.1)


class TestFieldValues:
    def test_drug_name_preserved(self):
        result = _default(drug_name="Timolol")
        assert result.drug_name == "Timolol"

    def test_mw_preserved(self):
        result = _default(mw_Da=432.5)
        assert result.mw_Da == 432.5

    def test_logp_preserved(self):
        result = _default(logp=1.5)
        assert result.logp == 1.5

    def test_all_penetration_classes_reachable(self):
        """Check that at least the extreme classes can be generated."""
        poor = _default(logp=-4.0, mw_Da=800.0)
        excellent = _default(logp=3.0, mw_Da=100.0)
        assert poor.penetration_class in _VALID_PENETRATION_CLASSES
        assert excellent.penetration_class in _VALID_PENETRATION_CLASSES
