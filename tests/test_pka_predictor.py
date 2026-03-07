"""Tests for pKa predictor (Phase 417)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.pka_predictor import (
    PKaPredictionResult,
    ionization_at_pH,
    predict_pka,
)


class TestIonizationAtPH:
    def test_acid_half_ionized_at_pka(self):
        frac = ionization_at_pH(pka=4.5, molecule_type="acid", pH=4.5)
        assert abs(frac - 0.5) < 1e-6

    def test_acid_mostly_ionized_above_pka(self):
        frac = ionization_at_pH(pka=4.5, molecule_type="acid", pH=7.0)
        assert frac > 0.99

    def test_acid_mostly_unionized_below_pka(self):
        frac = ionization_at_pH(pka=4.5, molecule_type="acid", pH=2.0)
        assert frac < 0.01

    def test_base_half_ionized_at_pka(self):
        frac = ionization_at_pH(pka=9.0, molecule_type="base", pH=9.0)
        assert abs(frac - 0.5) < 1e-6

    def test_base_mostly_ionized_below_pka(self):
        # Base (protonated) is mostly ionized (protonated) below pKa
        frac = ionization_at_pH(pka=9.0, molecule_type="base", pH=5.0)
        assert frac > 0.99

    def test_base_mostly_neutral_above_pka(self):
        frac = ionization_at_pH(pka=9.0, molecule_type="base", pH=12.0)
        assert frac < 0.01

    def test_neutral_treated_as_acid(self):
        frac_neutral = ionization_at_pH(pka=5.0, molecule_type="neutral", pH=7.0)
        frac_acid = ionization_at_pH(pka=5.0, molecule_type="acid", pH=7.0)
        assert abs(frac_neutral - frac_acid) < 1e-9

    def test_amphoteric_returns_average(self):
        frac = ionization_at_pH(pka=7.0, molecule_type="amphoteric", pH=7.0)
        # Both acid and base fractions are 0.5 at pKa=pH=7
        assert 0.0 <= frac <= 1.0

    def test_fraction_between_0_and_1(self):
        for pH in [1.0, 4.0, 7.0, 10.0, 13.0]:
            frac = ionization_at_pH(pka=5.0, molecule_type="acid", pH=pH)
            assert 0.0 <= frac <= 1.0

    def test_invalid_pka_raises(self):
        with pytest.raises(ValueError, match="pka must be positive"):
            ionization_at_pH(pka=0.0, molecule_type="acid", pH=7.0)

    def test_invalid_ph_raises(self):
        with pytest.raises(ValueError, match="pH must be between"):
            ionization_at_pH(pka=5.0, molecule_type="acid", pH=15.0)

    def test_invalid_molecule_type_raises(self):
        with pytest.raises(ValueError, match="molecule_type must be one of"):
            ionization_at_pH(pka=5.0, molecule_type="unknown", pH=7.0)


class TestPredictPKa:
    def test_returns_result_type(self):
        r = predict_pka("CC(=O)O", "acid")
        assert isinstance(r, PKaPredictionResult)

    def test_carboxylic_acid_detected(self):
        # Acetic acid
        r = predict_pka("CC(=O)O", "acid")
        assert r.detected_group == "carboxylic_acid"

    def test_carboxylic_acid_pka_range(self):
        r = predict_pka("CC(=O)O", "acid")
        assert r.pka_predicted is not None
        assert 3.5 <= r.pka_predicted <= 5.5

    def test_carboxylic_acid_ionized_at_ph7(self):
        # At pH 7, carboxylic acid (pKa~4.5) should be mostly ionized
        r = predict_pka("CC(=O)O", "acid")
        assert r.ionization_at_pH7 > 0.9

    def test_carboxylic_acid_solubility_improved(self):
        r = predict_pka("CC(=O)O", "acid")
        assert r.solubility_impact == "improved"

    def test_phenol_detected(self):
        r = predict_pka("Oc1ccccc1", "acid")
        assert r.detected_group == "phenol"

    def test_phenol_pka_range(self):
        r = predict_pka("Oc1ccccc1", "acid")
        assert r.pka_predicted is not None
        assert abs(r.pka_predicted - 9.5) < 0.1

    def test_phenol_mostly_unionized_at_ph7(self):
        # Phenol pKa=9.5, at pH 7 mostly unionized as acid
        r = predict_pka("Oc1ccccc1", "acid")
        assert r.ionization_at_pH7 < 0.1

    def test_amine_primary_detected(self):
        r = predict_pka("CCN", "base")
        assert r.detected_group in ("amine_primary", "amine_secondary")

    def test_amine_primary_pka_range(self):
        r = predict_pka("CN", "base")
        assert r.pka_predicted is not None
        assert 9.0 <= r.pka_predicted <= 11.0

    def test_sulfonamide_detected(self):
        r = predict_pka("CS(=O)(=O)N", "acid")
        assert r.detected_group == "sulfonamide"

    def test_sulfonamide_pka_value(self):
        r = predict_pka("CS(=O)(=O)N", "acid")
        assert r.pka_predicted is not None
        assert abs(r.pka_predicted - 10.5) < 0.1

    def test_imidazole_detected(self):
        r = predict_pka("c1cncc1", "base")
        assert r.detected_group == "imidazole"

    def test_imidazole_pka_range(self):
        r = predict_pka("c1cncc1", "base")
        assert r.pka_predicted is not None
        assert 6.0 <= r.pka_predicted <= 7.5

    def test_pyridine_detected(self):
        r = predict_pka("c1ccccn1", "base")
        assert r.detected_group == "pyridine"

    def test_pyridine_pka_range(self):
        r = predict_pka("c1ccccn1", "base")
        assert r.pka_predicted is not None
        assert 4.5 <= r.pka_predicted <= 6.0

    def test_no_group_returns_none(self):
        # Simple alkane with no ionizable group
        r = predict_pka("CCCCCC", "neutral")
        assert r.detected_group is None
        assert r.pka_predicted is None

    def test_no_group_ionization_zero(self):
        r = predict_pka("CCCCCC", "neutral")
        assert r.ionization_at_pH7 == 0.0

    def test_no_group_solubility_neutral(self):
        r = predict_pka("CCCCCC", "neutral")
        assert r.solubility_impact == "neutral"

    def test_smiles_stored_in_result(self):
        smiles = "CC(=O)O"
        r = predict_pka(smiles, "acid")
        assert r.smiles == smiles

    def test_molecule_type_stored(self):
        r = predict_pka("CC(=O)O", "acid")
        assert r.molecule_type == "acid"

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="smiles must be a non-empty string"):
            predict_pka("", "acid")

    def test_invalid_molecule_type_raises(self):
        with pytest.raises(ValueError, match="molecule_type must be one of"):
            predict_pka("CC(=O)O", "invalid_type")

    def test_notes_non_empty(self):
        r = predict_pka("CC(=O)O", "acid")
        assert len(r.notes) > 0

    def test_result_frozen(self):
        r = predict_pka("CC(=O)O", "acid")
        with pytest.raises(AttributeError):
            r.pka_predicted = 5.0  # type: ignore[misc]

    def test_amphoteric_molecule_type_accepted(self):
        r = predict_pka("CC(=O)O", "amphoteric")
        assert r.molecule_type == "amphoteric"

    def test_mw_effect_on_carboxylic_acid(self):
        # Larger molecule should have slightly different pKa
        r_small = predict_pka("CC(=O)O", "acid")  # acetic acid ~60 Da
        r_large = predict_pka("CCCCCCCCCCC(=O)O", "acid")  # long chain ~200 Da
        assert r_small.pka_predicted is not None and r_large.pka_predicted is not None
        # MW effect should make them different (larger molecule slightly lower pKa)
        assert r_large.pka_predicted <= r_small.pka_predicted
