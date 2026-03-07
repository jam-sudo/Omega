"""Tests for Phase 559 — logd_predictor.py (~25 tests)."""

import math

import pytest

from omega_pbpk.prediction.logd_predictor import (
    ionized_fraction,
    logd_at_multiple_ph,
    logd_from_logp_pka,
    logd_protein_binding_correction,
    predict_membrane_permeability,
)

# ---------------------------------------------------------------------------
# logd_from_logp_pka
# ---------------------------------------------------------------------------


class TestLogdFromLogpPka:
    def test_neutral_returns_logp(self):
        assert logd_from_logp_pka(2.0, 4.0, 7.4, "neutral") == 2.0

    def test_neutral_any_ph(self):
        for ph in [1.0, 4.5, 7.4, 10.0]:
            assert logd_from_logp_pka(1.5, 5.0, ph, "neutral") == 1.5

    def test_acid_high_ph_logd_less_than_logp(self):
        # At pH >> pKa, acid is mostly ionized -> logD < logP
        logd = logd_from_logp_pka(2.0, 4.0, 9.0, "acid")
        assert logd < 2.0

    def test_acid_low_ph_logd_approx_logp(self):
        # At pH << pKa, acid is mostly unionized -> logD ≈ logP
        logd = logd_from_logp_pka(2.0, 9.0, 1.0, "acid")
        assert abs(logd - 2.0) < 0.01

    def test_acid_at_pka_correction(self):
        # At pH == pKa: logD = logP - log10(2)
        logP = 2.0
        pka = 5.0
        logd = logd_from_logp_pka(logP, pka, pka, "acid")
        expected = logP - math.log10(2.0)
        assert abs(logd - expected) < 1e-9

    def test_base_low_ph_logd_less_than_logp(self):
        # At pH << pKa, base is mostly ionized -> logD < logP
        logd = logd_from_logp_pka(2.0, 9.0, 1.0, "base")
        assert logd < 2.0

    def test_base_high_ph_logd_approx_logp(self):
        # At pH >> pKa, base is mostly unionized -> logD ≈ logP
        logd = logd_from_logp_pka(2.0, 4.0, 12.0, "base")
        assert abs(logd - 2.0) < 0.01

    def test_zwitterion_between_acid_and_base(self):
        logP, pka, ph = 2.0, 7.0, 7.4
        logd_acid = logd_from_logp_pka(logP, pka, ph, "acid")
        logd_base = logd_from_logp_pka(logP, pka, ph, "base")
        logd_zwit = logd_from_logp_pka(logP, pka, ph, "zwitterion")
        assert abs(logd_zwit - (logd_acid + logd_base) / 2) < 1e-9

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="Invalid drug_type"):
            logd_from_logp_pka(2.0, 4.0, 7.4, "ampholyte")


# ---------------------------------------------------------------------------
# ionized_fraction
# ---------------------------------------------------------------------------


class TestIonizedFraction:
    def test_neutral_is_zero(self):
        assert ionized_fraction(4.0, 7.4, "neutral") == 0.0

    def test_acid_at_pka_is_half(self):
        fi = ionized_fraction(5.0, 5.0, "acid")
        assert abs(fi - 0.5) < 1e-9

    def test_acid_above_pka_mostly_ionized(self):
        # pH 10 >> pKa 5 → fi ≈ 1
        fi = ionized_fraction(5.0, 10.0, "acid")
        assert fi > 0.99

    def test_acid_below_pka_mostly_unionized(self):
        # pH 1 << pKa 7 → fi ≈ 0
        fi = ionized_fraction(7.0, 1.0, "acid")
        assert fi < 0.01

    def test_base_at_pka_is_half(self):
        fi = ionized_fraction(8.0, 8.0, "base")
        assert abs(fi - 0.5) < 1e-9

    def test_base_below_pka_mostly_ionized(self):
        # pH 1 << pKa 9 → fi ≈ 1
        fi = ionized_fraction(9.0, 1.0, "base")
        assert fi > 0.99

    def test_result_in_0_1(self):
        for drug_type in ["acid", "base", "neutral", "zwitterion"]:
            fi = ionized_fraction(7.4, 7.4, drug_type)
            assert 0.0 <= fi <= 1.0

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError):
            ionized_fraction(5.0, 7.4, "lipid")


# ---------------------------------------------------------------------------
# logd_at_multiple_ph
# ---------------------------------------------------------------------------


class TestLogdAtMultiplePh:
    def test_default_returns_4_entries(self):
        result = logd_at_multiple_ph(2.0, 4.0, "acid")
        assert len(result) == 4

    def test_default_ph_values(self):
        result = logd_at_multiple_ph(2.0, 4.0, "acid")
        phs = [r["pH"] for r in result]
        assert phs == [1.2, 4.5, 6.8, 7.4]

    def test_custom_ph_values(self):
        result = logd_at_multiple_ph(1.0, 5.0, "base", ph_values=[3.0, 7.4])
        assert len(result) == 2
        assert result[0]["pH"] == 3.0
        assert result[1]["pH"] == 7.4

    def test_dict_keys_present(self):
        result = logd_at_multiple_ph(1.5, 6.0, "acid")
        for entry in result:
            assert "pH" in entry
            assert "logD" in entry
            assert "ionized_fraction" in entry

    def test_neutral_logd_equals_logp_at_all_ph(self):
        result = logd_at_multiple_ph(3.0, 5.0, "neutral")
        for entry in result:
            assert abs(entry["logD"] - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# predict_membrane_permeability
# ---------------------------------------------------------------------------


class TestPredictMembranePermeability:
    def test_high_logd_high_permeability(self):
        result = predict_membrane_permeability(3.0)
        assert result["permeability_class"] == "high"
        assert result["papp_cm_s"] > 1e-5

    def test_low_logd_low_permeability(self):
        result = predict_membrane_permeability(-3.0)
        assert result["permeability_class"] == "low"
        assert result["papp_cm_s"] < 1e-6

    def test_moderate_logd_moderate_permeability(self):
        # logD such that papp is between 1e-6 and 1e-5
        # 10^(0.8*ld - 5.7) = 5e-6 → 0.8*ld - 5.7 = log10(5e-6) ≈ -5.3
        # ld ≈ 0.5
        result = predict_membrane_permeability(0.5)
        assert result["permeability_class"] == "moderate"

    def test_result_keys(self):
        result = predict_membrane_permeability(1.0)
        assert "logD_74" in result
        assert "papp_cm_s" in result
        assert "permeability_class" in result


# ---------------------------------------------------------------------------
# logd_protein_binding_correction
# ---------------------------------------------------------------------------


class TestLogdProteinBindingCorrection:
    def test_fu_in_0_1(self):
        result = logd_protein_binding_correction(2.0)
        assert 0 < result["fu_predicted"] <= 1.0

    def test_high_logd_lower_fu(self):
        fu_low = logd_protein_binding_correction(-1.0)["fu_predicted"]
        fu_high = logd_protein_binding_correction(4.0)["fu_predicted"]
        assert fu_high < fu_low

    def test_default_protein_conc(self):
        result = logd_protein_binding_correction(1.0)
        assert result["fu_predicted"] > 0

    def test_custom_protein_conc(self):
        result_low = logd_protein_binding_correction(2.0, protein_conc_g_L=10.0)
        result_high = logd_protein_binding_correction(2.0, protein_conc_g_L=70.0)
        # higher protein → lower fu
        assert result_high["fu_predicted"] < result_low["fu_predicted"]

    def test_result_keys(self):
        result = logd_protein_binding_correction(1.5)
        assert "logD_74" in result
        assert "fu_predicted" in result
        assert "pKd_ppb" in result
        assert "notes" in result
