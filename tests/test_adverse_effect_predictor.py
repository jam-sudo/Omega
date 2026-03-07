"""Tests for adverse effect predictor (Phase 427)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.adverse_effect_predictor import (
    AdverseEffectResult,
    predict_adverse_effects,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kw):
    """Return a predict_adverse_effects call with sensible defaults."""
    defaults = dict(
        drug_name="TestDrug",
        drug_class="nsaid",
        cmax_mg_L=10.0,
        auc_24h_mg_h_L=80.0,
        trough_mg_L=1.0,
        therapeutic_cmax=10.0,  # ratio = 1.0
        therapeutic_auc=80.0,  # ratio = 1.0
    )
    defaults.update(kw)
    return predict_adverse_effects(**defaults)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        r = _default()
        assert isinstance(r, AdverseEffectResult)

    def test_has_predicted_effects_list(self):
        r = _default()
        assert isinstance(r.predicted_effects, list)
        assert len(r.predicted_effects) > 0

    def test_each_effect_has_required_keys(self):
        r = _default()
        for eff in r.predicted_effects:
            assert "effect" in eff
            assert "likelihood" in eff
            assert "rationale" in eff

    def test_likelihood_values_valid(self):
        valid = {"low", "moderate", "high"}
        for drug_class in (
            "nsaid",
            "opioid",
            "antibiotic",
            "antihypertensive",
            "anticoagulant",
            "immunosuppressant",
        ):
            r = _default(drug_class=drug_class)
            for eff in r.predicted_effects:
                assert eff["likelihood"] in valid, (
                    f"{drug_class} effect {eff['effect']} likelihood invalid"
                )

    def test_overall_safety_concern_valid(self):
        valid = {"acceptable", "monitor", "high_risk"}
        for drug_class in (
            "nsaid",
            "opioid",
            "antibiotic",
            "antihypertensive",
            "anticoagulant",
            "immunosuppressant",
        ):
            r = _default(drug_class=drug_class)
            assert r.overall_safety_concern in valid

    def test_cmax_ratio_computed_correctly(self):
        r = _default(cmax_mg_L=20.0, therapeutic_cmax=10.0)
        assert abs(r.cmax_ratio - 2.0) < 1e-9

    def test_auc_ratio_computed_correctly(self):
        r = _default(auc_24h_mg_h_L=160.0, therapeutic_auc=80.0)
        assert abs(r.auc_ratio - 2.0) < 1e-9

    def test_drug_name_stored(self):
        r = _default(drug_name="Ibuprofen")
        assert r.drug_name == "Ibuprofen"

    def test_drug_class_stored(self):
        r = _default(drug_class="opioid")
        assert r.drug_class == "opioid"


# ---------------------------------------------------------------------------
# NSAID class
# ---------------------------------------------------------------------------


class TestNSAID:
    def test_gi_irritation_low_at_therapeutic(self):
        r = _default(drug_class="nsaid", cmax_mg_L=10.0, therapeutic_cmax=10.0)
        gi = next(e for e in r.predicted_effects if e["effect"] == "GI_irritation")
        assert gi["likelihood"] == "low"

    def test_gi_irritation_high_above_2x(self):
        r = _default(drug_class="nsaid", cmax_mg_L=21.0, therapeutic_cmax=10.0)
        gi = next(e for e in r.predicted_effects if e["effect"] == "GI_irritation")
        assert gi["likelihood"] == "high"

    def test_renal_toxicity_present(self):
        r = _default(drug_class="nsaid")
        effects = [e["effect"] for e in r.predicted_effects]
        assert "renal_toxicity" in effects

    def test_nsaid_has_3_effects(self):
        r = _default(drug_class="nsaid")
        assert len(r.predicted_effects) == 3


# ---------------------------------------------------------------------------
# Opioid class
# ---------------------------------------------------------------------------


class TestOpioid:
    def test_respiratory_depression_high_at_3x_cmax(self):
        r = _default(drug_class="opioid", cmax_mg_L=30.0, therapeutic_cmax=10.0)
        resp = next(e for e in r.predicted_effects if e["effect"] == "respiratory_depression")
        assert resp["likelihood"] == "high"

    def test_respiratory_depression_low_at_therapeutic(self):
        r = _default(drug_class="opioid")
        resp = next(e for e in r.predicted_effects if e["effect"] == "respiratory_depression")
        assert resp["likelihood"] == "low"

    def test_sedation_present(self):
        r = _default(drug_class="opioid")
        effects = [e["effect"] for e in r.predicted_effects]
        assert "sedation" in effects

    def test_opioid_has_3_effects(self):
        r = _default(drug_class="opioid")
        assert len(r.predicted_effects) == 3


# ---------------------------------------------------------------------------
# Antibiotic class
# ---------------------------------------------------------------------------


class TestAntibiotic:
    def test_nephrotoxicity_present(self):
        r = _default(drug_class="antibiotic")
        effects = [e["effect"] for e in r.predicted_effects]
        assert "nephrotoxicity" in effects

    def test_hepatotoxicity_present(self):
        r = _default(drug_class="antibiotic")
        effects = [e["effect"] for e in r.predicted_effects]
        assert "hepatotoxicity" in effects

    def test_aminoglycoside_nephro_low_at_therapeutic(self):
        r = predict_adverse_effects(
            drug_name="gentamicin",
            drug_class="antibiotic",
            cmax_mg_L=10.0,
            auc_24h_mg_h_L=80.0,
            trough_mg_L=1.0,
            therapeutic_cmax=10.0,
            therapeutic_auc=80.0,
        )
        nephro = next(e for e in r.predicted_effects if e["effect"] == "nephrotoxicity")
        assert nephro["likelihood"] == "low"

    def test_aminoglycoside_nephro_high_at_2x_cmax(self):
        r = predict_adverse_effects(
            drug_name="gentamicin",
            drug_class="antibiotic",
            cmax_mg_L=21.0,
            auc_24h_mg_h_L=80.0,
            trough_mg_L=1.0,
            therapeutic_cmax=10.0,
            therapeutic_auc=80.0,
        )
        nephro = next(e for e in r.predicted_effects if e["effect"] == "nephrotoxicity")
        assert nephro["likelihood"] == "high"


# ---------------------------------------------------------------------------
# Antihypertensive class
# ---------------------------------------------------------------------------


class TestAntihypertensive:
    def test_hypotension_present(self):
        r = _default(drug_class="antihypertensive")
        effects = [e["effect"] for e in r.predicted_effects]
        assert "hypotension" in effects

    def test_hypotension_high_at_2_5x_auc(self):
        r = _default(
            drug_class="antihypertensive",
            auc_24h_mg_h_L=200.0,
            therapeutic_auc=80.0,
        )
        hypo = next(e for e in r.predicted_effects if e["effect"] == "hypotension")
        assert hypo["likelihood"] == "high"

    def test_antihypertensive_has_3_effects(self):
        r = _default(drug_class="antihypertensive")
        assert len(r.predicted_effects) == 3


# ---------------------------------------------------------------------------
# Anticoagulant class
# ---------------------------------------------------------------------------


class TestAnticoagulant:
    def test_bleeding_risk_present(self):
        r = _default(drug_class="anticoagulant")
        effects = [e["effect"] for e in r.predicted_effects]
        assert "bleeding_risk" in effects

    def test_bleeding_high_above_1_5x_auc(self):
        r = _default(
            drug_class="anticoagulant",
            auc_24h_mg_h_L=121.0,
            therapeutic_auc=80.0,
        )
        bleed = next(e for e in r.predicted_effects if e["effect"] == "bleeding_risk")
        assert bleed["likelihood"] == "high"

    def test_bleeding_low_at_therapeutic(self):
        r = _default(drug_class="anticoagulant")
        bleed = next(e for e in r.predicted_effects if e["effect"] == "bleeding_risk")
        assert bleed["likelihood"] == "low"

    def test_anticoagulant_has_3_effects(self):
        r = _default(drug_class="anticoagulant")
        assert len(r.predicted_effects) == 3


# ---------------------------------------------------------------------------
# Immunosuppressant class
# ---------------------------------------------------------------------------


class TestImmunosuppressant:
    def test_infection_risk_present(self):
        r = _default(drug_class="immunosuppressant")
        effects = [e["effect"] for e in r.predicted_effects]
        assert "infection_risk" in effects

    def test_nephrotoxicity_present(self):
        r = _default(drug_class="immunosuppressant")
        effects = [e["effect"] for e in r.predicted_effects]
        assert "nephrotoxicity" in effects

    def test_infection_high_at_2x_auc(self):
        r = _default(
            drug_class="immunosuppressant",
            auc_24h_mg_h_L=161.0,
            therapeutic_auc=80.0,
        )
        inf = next(e for e in r.predicted_effects if e["effect"] == "infection_risk")
        assert inf["likelihood"] == "high"

    def test_immunosuppressant_has_3_effects(self):
        r = _default(drug_class="immunosuppressant")
        assert len(r.predicted_effects) == 3


# ---------------------------------------------------------------------------
# Overall safety classification
# ---------------------------------------------------------------------------


class TestOverallSafety:
    def test_acceptable_at_therapeutic(self):
        r = _default(drug_class="nsaid")
        assert r.overall_safety_concern == "acceptable"

    def test_high_risk_at_3x_exposure_nsaid(self):
        r = _default(
            drug_class="nsaid",
            cmax_mg_L=30.0,
            auc_24h_mg_h_L=240.0,
        )
        assert r.overall_safety_concern == "high_risk"

    def test_high_risk_at_3x_exposure_anticoagulant(self):
        r = _default(
            drug_class="anticoagulant",
            auc_24h_mg_h_L=240.0,  # 3x
        )
        assert r.overall_safety_concern == "high_risk"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_drug_class_raises(self):
        with pytest.raises(ValueError, match="drug_class"):
            _default(drug_class="chemotherapy")

    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            _default(drug_name="")

    def test_negative_cmax_raises(self):
        with pytest.raises(ValueError, match="cmax_mg_L"):
            _default(cmax_mg_L=-1.0)

    def test_negative_auc_raises(self):
        with pytest.raises(ValueError, match="auc_24h_mg_h_L"):
            _default(auc_24h_mg_h_L=-1.0)

    def test_zero_therapeutic_cmax_raises(self):
        with pytest.raises(ValueError, match="therapeutic_cmax"):
            _default(therapeutic_cmax=0.0)

    def test_zero_therapeutic_auc_raises(self):
        with pytest.raises(ValueError, match="therapeutic_auc"):
            _default(therapeutic_auc=0.0)
