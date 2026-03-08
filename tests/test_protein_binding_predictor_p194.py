"""Tests for Phase 194 — Protein Binding Predictor."""

import pytest

from omega_pbpk.prediction.protein_binding_predictor_p194 import (
    ProteinBindingResult,
    predict_fu_plasma,
    screen_protein_binding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        mw_Da=350.0,
        logP=1.5,
        psa_A2=60.0,
        pka=7.0,
        ionization_type="neutral",
    )
    defaults.update(kwargs)
    return predict_fu_plasma(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_protein_binding_result(self):
        assert isinstance(_default(), ProteinBindingResult)

    def test_drug_name_preserved(self):
        result = _default(drug_name="Warfarin")
        assert result.drug_name == "Warfarin"

    def test_mw_preserved(self):
        result = _default(mw_Da=308.5)
        assert result.mw_Da == pytest.approx(308.5)

    def test_logp_preserved(self):
        result = _default(logP=2.3)
        assert result.logP == pytest.approx(2.3)

    def test_psa_preserved(self):
        result = _default(psa_A2=75.0)
        assert result.psa_A2 == pytest.approx(75.0)

    def test_pka_preserved(self):
        result = _default(pka=4.2)
        assert result.pka == pytest.approx(4.2)

    def test_ionization_type_stored_lowercase(self):
        result = _default(ionization_type="Acid")
        assert result.ionization_type == "acid"


# ---------------------------------------------------------------------------
# fu bounds and PPB relationship
# ---------------------------------------------------------------------------
class TestFuAndPPB:
    def test_fu_in_0_1(self):
        result = _default()
        assert 0.0 < result.fu_plasma <= 1.0

    def test_fu_positive(self):
        result = _default()
        assert result.fu_plasma > 0.0

    def test_ppb_equals_1_minus_fu_times_100(self):
        result = _default()
        expected_ppb = (1.0 - result.fu_plasma) * 100.0
        assert result.ppb_pct == pytest.approx(expected_ppb, rel=1e-5)

    def test_ppb_non_negative(self):
        result = _default()
        assert result.ppb_pct >= 0.0

    def test_ppb_at_most_100(self):
        result = _default()
        assert result.ppb_pct <= 100.0


# ---------------------------------------------------------------------------
# logP effect
# ---------------------------------------------------------------------------
class TestLogPEffect:
    def test_high_logp_gives_low_fu(self):
        high = _default(logP=5.0)
        low = _default(logP=0.0)
        assert high.fu_plasma < low.fu_plasma

    def test_high_logp_highly_bound(self):
        # logP=6.0 → fu_base=0.063 < 0.1 → highly_bound
        result = _default(logP=6.0, psa_A2=20.0)
        assert result.binding_class == "highly_bound"

    def test_low_logp_higher_fu(self):
        result = _default(logP=-2.0)
        assert result.fu_plasma > 0.5


# ---------------------------------------------------------------------------
# Binding class
# ---------------------------------------------------------------------------
class TestBindingClass:
    def test_highly_bound_class(self):
        # logP=6.0 → fu_base=0.063 < 0.1 → highly_bound
        result = _default(logP=6.0, psa_A2=20.0)
        assert result.binding_class == "highly_bound"

    def test_poorly_bound_class(self):
        result = _default(logP=-3.0, psa_A2=150.0)
        assert result.binding_class == "poorly_bound"

    def test_binding_class_valid_values(self):
        valid = {"highly_bound", "moderately_bound", "poorly_bound"}
        for logp in (-2.0, 1.0, 5.0):
            result = _default(logP=logp)
            assert result.binding_class in valid


# ---------------------------------------------------------------------------
# Primary binding protein
# ---------------------------------------------------------------------------
class TestBindingProtein:
    def test_acid_low_pka_binds_albumin(self):
        result = _default(ionization_type="acid", pka=3.5)
        assert result.primary_binding_protein == "albumin"

    def test_base_high_pka_binds_aag(self):
        result = _default(ionization_type="base", pka=9.0)
        assert result.primary_binding_protein == "AAG"

    def test_neutral_non_specific(self):
        result = _default(ionization_type="neutral")
        assert result.primary_binding_protein == "non-specific"


# ---------------------------------------------------------------------------
# Acid pKa modifier
# ---------------------------------------------------------------------------
class TestAcidModifier:
    def test_acid_low_pka_reduces_fu(self):
        acid_low = _default(ionization_type="acid", pka=3.0, logP=1.5)
        neutral = _default(ionization_type="neutral", pka=3.0, logP=1.5)
        assert acid_low.fu_plasma < neutral.fu_plasma

    def test_base_high_pka_reduces_fu(self):
        base_high = _default(ionization_type="base", pka=9.5, logP=1.5)
        neutral = _default(ionization_type="neutral", pka=9.5, logP=1.5)
        assert base_high.fu_plasma < neutral.fu_plasma


# ---------------------------------------------------------------------------
# screen_protein_binding
# ---------------------------------------------------------------------------
class TestScreenProteinBinding:
    def _make_drugs(self):
        return [
            {
                "drug_name": "LowBinder",
                "mw_Da": 300.0,
                "logP": -1.0,
                "psa_A2": 80.0,
                "pka": 7.0,
                "ionization_type": "neutral",
            },
            {
                "drug_name": "HighBinder",
                "mw_Da": 650.0,
                "logP": 6.0,
                "psa_A2": 20.0,
                "pka": 3.5,
                "ionization_type": "acid",
            },
            {
                "drug_name": "MidBinder",
                "mw_Da": 400.0,
                "logP": 2.0,
                "psa_A2": 55.0,
                "pka": 7.0,
                "ionization_type": "neutral",
            },
        ]

    def test_returns_list(self):
        results = screen_protein_binding(self._make_drugs())
        assert isinstance(results, list)

    def test_returns_correct_count(self):
        results = screen_protein_binding(self._make_drugs())
        assert len(results) == 3

    def test_sorted_by_fu_ascending(self):
        results = screen_protein_binding(self._make_drugs())
        fus = [r.fu_plasma for r in results]
        assert fus == sorted(fus)

    def test_all_protein_binding_result(self):
        results = screen_protein_binding(self._make_drugs())
        assert all(isinstance(r, ProteinBindingResult) for r in results)

    def test_most_bound_first(self):
        results = screen_protein_binding(self._make_drugs())
        # HighBinder should have lowest fu (most bound)
        assert results[0].drug_name == "HighBinder"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_raises_on_zero_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _default(mw_Da=0.0)

    def test_raises_on_negative_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _default(mw_Da=-100.0)

    def test_raises_on_invalid_ionization_type(self):
        with pytest.raises(ValueError, match="ionization_type"):
            _default(ionization_type="zwitter")

    def test_raises_on_invalid_ionization_type_message(self):
        with pytest.raises(ValueError):
            _default(ionization_type="amphoteric")

    def test_valid_zwitterion(self):
        result = _default(ionization_type="zwitterion")
        assert result.ionization_type == "zwitterion"

    def test_negative_psa_raises(self):
        with pytest.raises(ValueError, match="psa_A2"):
            _default(psa_A2=-10.0)
