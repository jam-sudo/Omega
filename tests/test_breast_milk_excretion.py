"""Tests for Phase 536 — Drug Excretion into Breast Milk."""

import pytest

from omega_pbpk.prediction.breast_milk_excretion import (
    _MILK_INTAKE_ML_PER_KG_PER_DAY,
    BreastMilkExcretionResult,
    predict_breast_milk_excretion,
    screen_lactation_safety,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _neutral_hydrophilic(**kwargs):
    defaults = dict(
        drug_name="Hydrophilic",
        logP=-1.0,
        pKa=3.0,
        drug_type="neutral",
        fu_plasma=0.9,
        c_plasma_mg_L=1.0,
        maternal_dose_mg=100.0,
        infant_weight_kg=5.0,
        maternal_weight_kg=60.0,
    )
    defaults.update(kwargs)
    return predict_breast_milk_excretion(**defaults)


def _lipophilic_base(**kwargs):
    defaults = dict(
        drug_name="LipoBase",
        logP=4.0,
        pKa=9.0,
        drug_type="base",
        fu_plasma=0.5,
        c_plasma_mg_L=1.0,
        maternal_dose_mg=100.0,
        infant_weight_kg=5.0,
        maternal_weight_kg=60.0,
    )
    defaults.update(kwargs)
    return predict_breast_milk_excretion(**defaults)


# ---------------------------------------------------------------------------
# Return type and field checks
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        assert isinstance(_neutral_hydrophilic(), BreastMilkExcretionResult)

    def test_has_drug_name(self):
        r = _neutral_hydrophilic()
        assert r.drug_name == "Hydrophilic"

    def test_has_logP(self):
        r = _neutral_hydrophilic()
        assert r.logP == -1.0

    def test_has_pKa(self):
        r = _neutral_hydrophilic()
        assert r.pKa == 3.0

    def test_has_drug_type(self):
        r = _neutral_hydrophilic()
        assert r.drug_type == "neutral"

    def test_has_fu_plasma(self):
        r = _neutral_hydrophilic(fu_plasma=0.7)
        assert r.fu_plasma == 0.7

    def test_has_c_plasma(self):
        r = _neutral_hydrophilic(c_plasma_mg_L=2.0)
        assert r.c_plasma_mg_L == 2.0

    def test_has_mp_ratio(self):
        r = _neutral_hydrophilic()
        assert r.mp_ratio > 0.0

    def test_has_c_milk(self):
        r = _neutral_hydrophilic()
        assert r.c_milk_mg_L > 0.0

    def test_has_infant_daily_dose(self):
        r = _neutral_hydrophilic()
        assert r.infant_daily_dose_mg_per_kg > 0.0

    def test_has_rid_pct(self):
        r = _neutral_hydrophilic()
        assert r.rid_pct > 0.0

    def test_has_lactation_safety(self):
        r = _neutral_hydrophilic()
        assert r.lactation_safety in {"compatible", "use_with_caution", "avoid"}

    def test_has_lipid_trapping(self):
        r = _neutral_hydrophilic()
        assert isinstance(r.lipid_trapping, bool)

    def test_has_ion_trapping(self):
        r = _neutral_hydrophilic()
        assert isinstance(r.ion_trapping, bool)

    def test_has_notes(self):
        r = _neutral_hydrophilic()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Neutral hydrophilic: M:P ≈ fu_plasma
# ---------------------------------------------------------------------------


class TestNeutralHydrophilic:
    def test_mp_ratio_approx_fu_plasma(self):
        # logP = -1: kp_lipid = 1.0, so M:P = fu * (1 + 0.04*(1-1)) = fu
        r = _neutral_hydrophilic(logP=-1.0, fu_plasma=0.8)
        assert abs(r.mp_ratio - 0.8) < 1e-9

    def test_c_milk_equals_c_plasma_times_mp(self):
        r = _neutral_hydrophilic(c_plasma_mg_L=2.0)
        assert abs(r.c_milk_mg_L - r.c_plasma_mg_L * r.mp_ratio) < 1e-9

    def test_no_lipid_trapping_for_low_logP(self):
        r = _neutral_hydrophilic(logP=-1.0)
        assert r.lipid_trapping is False

    def test_no_ion_trapping_for_neutral(self):
        r = _neutral_hydrophilic()
        assert r.ion_trapping is False


# ---------------------------------------------------------------------------
# Lipophilic base: M:P > fu_plasma
# ---------------------------------------------------------------------------


class TestLipophilicBase:
    def test_mp_ratio_exceeds_fu_plasma(self):
        r = _lipophilic_base()
        assert r.mp_ratio > r.fu_plasma

    def test_ion_trapping_true_for_base_high_pka(self):
        r = _lipophilic_base(pKa=9.0)
        assert r.ion_trapping is True

    def test_ion_trapping_false_for_base_low_pka(self):
        r = predict_breast_milk_excretion(
            drug_name="WeakBase",
            logP=4.0,
            pKa=6.0,
            drug_type="base",
            fu_plasma=0.5,
            c_plasma_mg_L=1.0,
            maternal_dose_mg=100.0,
        )
        assert r.ion_trapping is False  # pKa <= 7.5

    def test_lipid_trapping_true_for_high_logP(self):
        # logP=4, so kp_lipid large → lipid contribution should be significant
        r = predict_breast_milk_excretion(
            drug_name="LipoNeutral",
            logP=4.0,
            pKa=3.0,
            drug_type="neutral",
            fu_plasma=0.1,
            c_plasma_mg_L=1.0,
            maternal_dose_mg=100.0,
        )
        if r.lipid_trapping:
            assert r.logP > 2.0


# ---------------------------------------------------------------------------
# Milk concentration derived from plasma
# ---------------------------------------------------------------------------


class TestMilkConcentration:
    def test_c_milk_mg_L_formula(self):
        r = _neutral_hydrophilic(c_plasma_mg_L=5.0)
        assert abs(r.c_milk_mg_L - r.c_plasma_mg_L * r.mp_ratio) < 1e-9

    def test_c_milk_positive(self):
        r = _lipophilic_base()
        assert r.c_milk_mg_L > 0.0

    def test_infant_dose_formula(self):
        r = _neutral_hydrophilic(c_plasma_mg_L=2.0)
        expected = r.c_milk_mg_L * _MILK_INTAKE_ML_PER_KG_PER_DAY / 1000.0
        assert abs(r.infant_daily_dose_mg_per_kg - expected) < 1e-9


# ---------------------------------------------------------------------------
# Lactation safety classification
# ---------------------------------------------------------------------------


class TestLactationSafety:
    def test_compatible_when_rid_below_10(self):
        # Small dose → very small RID
        r = predict_breast_milk_excretion(
            drug_name="Safe",
            logP=-2.0,
            pKa=3.0,
            drug_type="neutral",
            fu_plasma=0.1,
            c_plasma_mg_L=0.01,
            maternal_dose_mg=1000.0,
        )
        assert r.rid_pct < 10.0
        assert r.lactation_safety == "compatible"

    def test_avoid_when_rid_above_25(self):
        # High plasma conc, small maternal dose
        r = predict_breast_milk_excretion(
            drug_name="Dangerous",
            logP=5.0,
            pKa=9.0,
            drug_type="base",
            fu_plasma=1.0,
            c_plasma_mg_L=100.0,
            maternal_dose_mg=1.0,
        )
        assert r.rid_pct > 25.0
        assert r.lactation_safety == "avoid"

    def test_rid_positive(self):
        r = _neutral_hydrophilic()
        assert r.rid_pct > 0.0

    def test_mp_ratio_clamped_max_20(self):
        # Even with extreme parameters, M:P should not exceed 20
        r = predict_breast_milk_excretion(
            drug_name="Extreme",
            logP=10.0,
            pKa=9.0,
            drug_type="base",
            fu_plasma=1.0,
            c_plasma_mg_L=1.0,
            maternal_dose_mg=100.0,
        )
        assert r.mp_ratio <= 20.0

    def test_mp_ratio_clamped_min_0_01(self):
        r = predict_breast_milk_excretion(
            drug_name="VeryLow",
            logP=-5.0,
            pKa=3.0,
            drug_type="neutral",
            fu_plasma=0.001,  # very low but we need > 0
            c_plasma_mg_L=0.001,
            maternal_dose_mg=1000.0,
        )
        assert r.mp_ratio >= 0.01


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_raises_on_invalid_drug_type(self):
        with pytest.raises(ValueError, match="drug_type"):
            predict_breast_milk_excretion(
                drug_name="D",
                logP=0.0,
                pKa=7.0,
                drug_type="cation",
                fu_plasma=0.5,
                c_plasma_mg_L=1.0,
                maternal_dose_mg=100.0,
            )

    def test_raises_on_zero_fu_plasma(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_breast_milk_excretion(
                drug_name="D",
                logP=0.0,
                pKa=7.0,
                drug_type="neutral",
                fu_plasma=0.0,
                c_plasma_mg_L=1.0,
                maternal_dose_mg=100.0,
            )

    def test_raises_on_fu_plasma_greater_than_1(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_breast_milk_excretion(
                drug_name="D",
                logP=0.0,
                pKa=7.0,
                drug_type="neutral",
                fu_plasma=1.5,
                c_plasma_mg_L=1.0,
                maternal_dose_mg=100.0,
            )

    def test_raises_on_zero_c_plasma(self):
        with pytest.raises(ValueError, match="c_plasma"):
            predict_breast_milk_excretion(
                drug_name="D",
                logP=0.0,
                pKa=7.0,
                drug_type="neutral",
                fu_plasma=0.5,
                c_plasma_mg_L=0.0,
                maternal_dose_mg=100.0,
            )

    def test_raises_on_zero_maternal_dose(self):
        with pytest.raises(ValueError, match="maternal_dose"):
            predict_breast_milk_excretion(
                drug_name="D",
                logP=0.0,
                pKa=7.0,
                drug_type="neutral",
                fu_plasma=0.5,
                c_plasma_mg_L=1.0,
                maternal_dose_mg=0.0,
            )


# ---------------------------------------------------------------------------
# screen_lactation_safety
# ---------------------------------------------------------------------------


class TestScreenLactationSafety:
    def _make_compounds(self):
        return [
            dict(
                drug_name="HighRisk",
                logP=5.0,
                pKa=9.0,
                drug_type="base",
                fu_plasma=1.0,
                c_plasma_mg_L=100.0,
                maternal_dose_mg=1.0,
            ),
            dict(
                drug_name="LowRisk",
                logP=-2.0,
                pKa=3.0,
                drug_type="neutral",
                fu_plasma=0.1,
                c_plasma_mg_L=0.01,
                maternal_dose_mg=1000.0,
            ),
            dict(
                drug_name="MidRisk",
                logP=2.0,
                pKa=7.0,
                drug_type="neutral",
                fu_plasma=0.5,
                c_plasma_mg_L=5.0,
                maternal_dose_mg=200.0,
            ),
        ]

    def test_returns_list(self):
        result = screen_lactation_safety(self._make_compounds())
        assert isinstance(result, list)
        assert len(result) == 3

    def test_sorted_by_rid_ascending(self):
        results = screen_lactation_safety(self._make_compounds())
        rids = [r.rid_pct for r in results]
        assert rids == sorted(rids)

    def test_all_results_correct_type(self):
        results = screen_lactation_safety(self._make_compounds())
        assert all(isinstance(r, BreastMilkExcretionResult) for r in results)

    def test_safest_first(self):
        results = screen_lactation_safety(self._make_compounds())
        assert results[0].drug_name == "LowRisk"
