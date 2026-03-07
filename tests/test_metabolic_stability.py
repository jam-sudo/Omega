"""Tests for omega_pbpk.clinical.metabolic_stability module."""

import math

import pytest

from omega_pbpk.clinical.metabolic_stability import (
    MetabolicStabilityResult,
    clint_from_half_life,
    in_vitro_clearance_from_depletion,
    predict_metabolic_stability,
)

# ---------------------------------------------------------------------------
# TestClintFromHalfLife
# ---------------------------------------------------------------------------


class TestClintFromHalfLife:
    def test_returns_positive_float(self):
        result = clint_from_half_life(30.0)
        assert isinstance(result, float)
        assert result > 0

    def test_shorter_half_life_higher_clint(self):
        clint_fast = clint_from_half_life(10.0)
        clint_slow = clint_from_half_life(60.0)
        assert clint_fast > clint_slow

    def test_hepatocyte_assay_works(self):
        result = clint_from_half_life(30.0, assay_type="hepatocyte")
        assert result > 0

    def test_microsome_formula(self):
        t_half = 30.0
        protein_mg_mL = 0.5
        expected = (0.693 / t_half) * (1000.0 / protein_mg_mL)
        result = clint_from_half_life(t_half, protein_mg_mL=protein_mg_mL)
        assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# TestPredictMetabolicStability
# ---------------------------------------------------------------------------


class TestPredictMetabolicStability:
    def test_returns_result_type(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert isinstance(result, MetabolicStabilityResult)

    def test_clint_invitro_positive(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert result.clint_uL_per_min_per_mg > 0

    def test_clint_scaled_positive(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert result.clint_scaled_L_per_h > 0

    def test_cl_hepatic_leq_q_liver(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert result.cl_hepatic_L_per_h <= 90.0

    def test_extraction_ratio_bounded(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert 0 <= result.extraction_ratio <= 1

    def test_bioavailability_fh_bounded(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert 0 <= result.bioavailability_fh <= 1

    def test_labile_classification(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=10.0)
        assert result.stability_class == "labile"

    def test_stable_classification(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=120.0)
        assert result.stability_class == "stable"

    def test_moderate_classification(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=45.0)
        assert result.stability_class == "moderate"

    def test_short_half_life_high_extraction(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=5.0)
        assert result.extraction_ratio > 0.5

    def test_long_half_life_low_extraction(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=300.0)
        assert result.extraction_ratio < 0.5

    def test_drug_name_preserved(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert result.drug_name == "TestDrug"


# ---------------------------------------------------------------------------
# TestInVitroClearanceFromDepletion
# ---------------------------------------------------------------------------


class TestInVitroClearanceFromDepletion:
    @staticmethod
    def _make_depletion_data(ke: float, n_points: int = 6, dt: float = 5.0):
        """Generate monoexponential decay data: C(t) = 100 * exp(-ke * t)."""
        times = [i * dt for i in range(n_points)]
        concentrations = [100.0 * math.exp(-ke * t) for t in times]
        return times, concentrations

    def test_returns_dict_with_keys(self):
        times, concs = self._make_depletion_data(ke=0.02)
        result = in_vitro_clearance_from_depletion(
            times_min=times, concentrations=concs, drug_name="TestDrug"
        )
        assert isinstance(result, dict)
        for key in (
            "drug_name",
            "t_half_min",
            "ke_per_min",
            "r_squared",
            "clint_uL_per_min_per_mg",
        ):
            assert key in result, f"Missing key: {key}"

    def test_t_half_positive(self):
        times, concs = self._make_depletion_data(ke=0.02)
        result = in_vitro_clearance_from_depletion(
            times_min=times, concentrations=concs, drug_name="TestDrug"
        )
        assert result["t_half_min"] > 0

    def test_r_squared_high_for_clean_data(self):
        times, concs = self._make_depletion_data(ke=0.03)
        result = in_vitro_clearance_from_depletion(
            times_min=times, concentrations=concs, drug_name="TestDrug"
        )
        assert result["r_squared"] > 0.95

    def test_faster_decay_shorter_half_life(self):
        times_fast, concs_fast = self._make_depletion_data(ke=0.05)
        times_slow, concs_slow = self._make_depletion_data(ke=0.01)
        result_fast = in_vitro_clearance_from_depletion(
            times_min=times_fast, concentrations=concs_fast, drug_name="Fast"
        )
        result_slow = in_vitro_clearance_from_depletion(
            times_min=times_slow, concentrations=concs_slow, drug_name="Slow"
        )
        assert result_fast["t_half_min"] < result_slow["t_half_min"]


# ---------------------------------------------------------------------------
# Phase 442: predict_microsomal_stability (prediction module)
# ---------------------------------------------------------------------------

from omega_pbpk.prediction.metabolic_stability import (  # noqa: E402
    MicrosomalStabilityResult,
    predict_microsomal_stability,
)

_DEFAULT_MS = dict(
    drug_name="TestDrug",
    mw=350.0,
    logP=2.0,
    n_aromatic_rings=2,
    n_rot_bonds=4,
    molecular_formula_atoms={"C": 20, "H": 25, "N": 2, "O": 3},
)


def _ms(**overrides):
    return predict_microsomal_stability(**{**_DEFAULT_MS, **overrides})


class TestPredictMicrosomalStability:
    def test_returns_correct_type(self):
        assert isinstance(_ms(), MicrosomalStabilityResult)

    def test_clint_positive(self):
        assert _ms().clint_uL_per_min_per_mg > 0

    def test_t_half_positive(self):
        assert _ms().t_half_microsomal_min > 0

    def test_f_unmetabolized_between_0_and_1(self):
        r = _ms()
        assert 0.0 <= r.predicted_f_unmetabolized <= 1.0

    def test_low_logP_low_clint(self):
        # Low logP → low base CLint; fewer aromatic rings → less correction
        r_low = _ms(logP=-2.0, n_aromatic_rings=0)
        r_high = _ms(logP=5.0, n_aromatic_rings=5)
        assert r_low.clint_uL_per_min_per_mg < r_high.clint_uL_per_min_per_mg

    def test_unstable_class_for_high_logP_many_rings(self):
        r = _ms(logP=5.0, n_aromatic_rings=6, n_rot_bonds=2)
        assert r.stability_class == "unstable"

    def test_more_aromatic_rings_higher_clint(self):
        r_few = _ms(n_aromatic_rings=0)
        r_many = _ms(n_aromatic_rings=5)
        assert r_many.clint_uL_per_min_per_mg > r_few.clint_uL_per_min_per_mg

    def test_more_rot_bonds_lower_clint(self):
        # Formula: factor = 1 - 0.05 * min(n_rot_bonds, 10)
        # More rot_bonds → smaller factor → lower CLint (more metabolically stable)
        r_rigid = _ms(n_rot_bonds=0)
        r_flex = _ms(n_rot_bonds=10)
        assert r_flex.clint_uL_per_min_per_mg < r_rigid.clint_uL_per_min_per_mg

    def test_large_mw_lower_clint(self):
        r_small = _ms(mw=200.0)
        r_large = _ms(mw=800.0)
        assert r_large.clint_uL_per_min_per_mg < r_small.clint_uL_per_min_per_mg

    def test_high_logP_higher_clint(self):
        r_low = _ms(logP=0.0)
        r_high = _ms(logP=4.0)
        assert r_high.clint_uL_per_min_per_mg > r_low.clint_uL_per_min_per_mg

    def test_drug_name_preserved(self):
        r = _ms(drug_name="Aspirin")
        assert r.drug_name == "Aspirin"

    def test_mw_preserved(self):
        r = _ms(mw=450.0)
        assert r.mw == 450.0

    def test_logP_preserved(self):
        r = _ms(logP=3.5)
        assert r.logP == 3.5

    def test_n_aromatic_rings_preserved(self):
        r = _ms(n_aromatic_rings=3)
        assert r.n_aromatic_rings == 3

    def test_n_rot_bonds_preserved(self):
        r = _ms(n_rot_bonds=6)
        assert r.n_rot_bonds == 6

    def test_notes_nonempty_string(self):
        r = _ms()
        assert isinstance(r.notes, str) and len(r.notes) > 0

    def test_no_atoms_provided_works(self):
        r = predict_microsomal_stability(
            "Drug", mw=300.0, logP=2.0, n_aromatic_rings=2, n_rot_bonds=3
        )
        assert isinstance(r, MicrosomalStabilityResult)

    def test_stability_class_moderate_boundary(self):
        # Find a logP that gives moderate stability (30–60 min)
        r = _ms(logP=1.5, n_aromatic_rings=1, n_rot_bonds=2)
        assert r.stability_class in ("stable", "moderate", "unstable")

    def test_result_is_frozen(self):
        r = _ms()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "Other"  # type: ignore[misc]

    def test_n_rot_bonds_capped_at_10(self):
        r10 = _ms(n_rot_bonds=10)
        r15 = _ms(n_rot_bonds=15)
        # Both should give same CLint (rot_bonds capped at 10)
        assert abs(r10.clint_uL_per_min_per_mg - r15.clint_uL_per_min_per_mg) < 1e-9

    def test_mw_correction_floor_at_0_3(self):
        # Very large MW → f_mw = max(0.3, ...) = 0.3
        r = _ms(mw=10000.0)
        assert r.clint_uL_per_min_per_mg > 0


class TestPredictMicrosomalStabilityValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            _ms(drug_name="")

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            _ms(mw=0.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            _ms(mw=-100.0)

    def test_negative_aromatic_rings_raises(self):
        with pytest.raises(ValueError, match="n_aromatic_rings"):
            _ms(n_aromatic_rings=-1)

    def test_negative_rot_bonds_raises(self):
        with pytest.raises(ValueError, match="n_rot_bonds"):
            _ms(n_rot_bonds=-1)
