"""Tests for Phase 200 — pka_predictor_p200.py."""

import pytest

from omega_pbpk.prediction.pka_predictor_p200 import (
    PKaResult,
    classify_ionization_at_ph,
    predict_pka,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _acid_result(**kwargs):
    defaults = dict(
        drug_name="Ibuprofen",
        smiles_fragments=[],
        mw_Da=206.3,
        logP=3.5,
        has_carboxyl=True,
    )
    defaults.update(kwargs)
    return predict_pka(**defaults)


def _amine_result(**kwargs):
    defaults = dict(
        drug_name="Amphetamine",
        smiles_fragments=[],
        mw_Da=135.2,
        logP=1.8,
        has_amine=True,
        aromatic_amine=False,
    )
    defaults.update(kwargs)
    return predict_pka(**defaults)


def _phenol_result(**kwargs):
    defaults = dict(
        drug_name="Paracetamol",
        smiles_fragments=[],
        mw_Da=151.2,
        logP=0.5,
        has_phenol=True,
    )
    defaults.update(kwargs)
    return predict_pka(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_pka_result(self):
        result = _acid_result()
        assert isinstance(result, PKaResult)

    def test_drug_name_preserved(self):
        result = predict_pka("AspirineTest", [], 180.0, 1.2, has_carboxyl=True)
        assert result.drug_name == "AspirineTest"

    def test_mw_preserved(self):
        result = _acid_result(mw_Da=300.0)
        assert result.mw_Da == pytest.approx(300.0)

    def test_logP_preserved(self):
        result = _acid_result(logP=2.5)
        assert result.logP == pytest.approx(2.5)

    def test_pka_values_is_tuple(self):
        result = _acid_result()
        assert isinstance(result.pka_values, tuple)

    def test_ionization_types_is_tuple(self):
        result = _acid_result()
        assert isinstance(result.ionization_types, tuple)

    def test_dominant_ionization_is_str(self):
        result = _acid_result()
        assert isinstance(result.dominant_ionization, str)

    def test_notes_is_str(self):
        result = _acid_result()
        assert isinstance(result.notes, str)

    def test_frozen_dataclass(self):
        result = _acid_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Carboxylic acid pKa
# ---------------------------------------------------------------------------


class TestCarboxylicAcid:
    def test_carboxyl_pka_in_range(self):
        result = _acid_result()
        assert len(result.pka_values) >= 1
        pka = result.pka_values[0]
        assert 3.0 <= pka <= 6.0

    def test_carboxyl_ionization_type_is_acid(self):
        result = _acid_result()
        assert "acid" in result.ionization_types

    def test_ewg_lowers_carboxyl_pka(self):
        no_ewg = _acid_result(smiles_fragments=[])
        ewg = _acid_result(smiles_fragments=["F", "Cl", "NO2"])
        assert ewg.pka_values[0] < no_ewg.pka_values[0]

    def test_high_logP_increases_carboxyl_pka(self):
        low = _acid_result(logP=0.5)
        high = _acid_result(logP=3.0)
        assert high.pka_values[0] >= low.pka_values[0]


# ---------------------------------------------------------------------------
# Aliphatic amine pKa
# ---------------------------------------------------------------------------


class TestAliphaticAmine:
    def test_aliphatic_amine_pka_in_range(self):
        result = _amine_result()
        assert len(result.pka_values) >= 1
        pka = result.pka_values[0]
        assert 8.0 <= pka <= 11.0

    def test_aliphatic_amine_ionization_type_is_base(self):
        result = _amine_result()
        assert "base" in result.ionization_types


# ---------------------------------------------------------------------------
# Aromatic amine pKa
# ---------------------------------------------------------------------------


class TestAromaticAmine:
    def test_aromatic_amine_pka_lower_than_aliphatic(self):
        aliphatic = _amine_result(aromatic_amine=False)
        aromatic = _amine_result(aromatic_amine=True)
        assert aromatic.pka_values[0] < aliphatic.pka_values[0]


# ---------------------------------------------------------------------------
# Phenol pKa
# ---------------------------------------------------------------------------


class TestPhenol:
    def test_phenol_pka_in_range(self):
        result = _phenol_result()
        assert len(result.pka_values) >= 1
        pka = result.pka_values[0]
        assert 8.5 <= pka <= 11.0

    def test_phenol_ionization_type_is_acid(self):
        result = _phenol_result()
        assert "acid" in result.ionization_types


# ---------------------------------------------------------------------------
# No ionizable groups
# ---------------------------------------------------------------------------


class TestNoIonizableGroups:
    def test_empty_pka_values_when_no_groups(self):
        result = predict_pka(
            "Caffeine",
            [],
            194.2,
            0.0,
            has_carboxyl=False,
            has_amine=False,
            has_phenol=False,
            has_imidazole=False,
        )
        assert result.pka_values == ()

    def test_empty_ionization_types_when_no_groups(self):
        result = predict_pka(
            "Caffeine",
            [],
            194.2,
            0.0,
        )
        assert result.ionization_types == ()

    def test_dominant_ionization_non_ionizable(self):
        result = predict_pka("Neutral", [], 200.0, 1.0)
        assert result.dominant_ionization == "non-ionizable"


# ---------------------------------------------------------------------------
# classify_ionization_at_ph
# ---------------------------------------------------------------------------


class TestClassifyIonizationAtPh:
    def test_returns_dict(self):
        result = classify_ionization_at_ph("Drug", (4.5,), ("acid",), 7.4)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = classify_ionization_at_ph("Drug", (4.5,), ("acid",), 7.4)
        assert "ionized_fraction" in result
        assert "charge_state" in result
        assert "predominant_form" in result

    def test_ionized_fraction_in_range(self):
        result = classify_ionization_at_ph("Drug", (4.5,), ("acid",), 7.4)
        assert 0.0 <= result["ionized_fraction"] <= 1.0

    def test_acid_at_high_ph_mostly_ionized(self):
        result = classify_ionization_at_ph("Acid", (4.0,), ("acid",), 10.0)
        assert result["ionized_fraction"] > 0.9

    def test_base_at_low_ph_mostly_ionized(self):
        result = classify_ionization_at_ph("Base", (10.0,), ("base",), 4.0)
        assert result["ionized_fraction"] > 0.9

    def test_acid_predominant_form_anion(self):
        # pKa=4.5 acid at pH 10 → mostly deprotonated → anion
        result = classify_ionization_at_ph("Acid", (4.5,), ("acid",), 10.0)
        assert result["predominant_form"] == "anion"

    def test_base_predominant_form_cation(self):
        # pKa=10 base at pH 4 → mostly protonated → cation
        result = classify_ionization_at_ph("Base", (10.0,), ("base",), 4.0)
        assert result["predominant_form"] == "cation"

    def test_empty_pka_values_returns_neutral(self):
        result = classify_ionization_at_ph("Neutral", (), (), 7.4)
        assert result["ionized_fraction"] == pytest.approx(0.0)
        assert result["predominant_form"] == "neutral"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_pka("X", [], 0.0, 1.0, has_carboxyl=True)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_pka("X", [], -100.0, 1.0, has_carboxyl=True)

    def test_ph_above_14_raises(self):
        with pytest.raises(ValueError, match="ph"):
            classify_ionization_at_ph("X", (4.5,), ("acid",), 15.0)

    def test_ph_below_0_raises(self):
        with pytest.raises(ValueError, match="ph"):
            classify_ionization_at_ph("X", (4.5,), ("acid",), -1.0)
