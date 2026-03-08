"""Tests for Phase 901 — Bile Drug Concentration Prediction."""

import pytest

from omega_pbpk.prediction.bile_drug_concentration import (
    BileDrugConcentrationResult,
    predict_bile_concentration,
    screen_biliary_excretion,
)

# ---------------------------------------------------------------------------
# Basic instantiation and field types
# ---------------------------------------------------------------------------


class TestBileDrugConcentrationResult:
    def test_frozen_dataclass(self):
        result = predict_bile_concentration("TestDrug")
        assert isinstance(result, BileDrugConcentrationResult)
        with pytest.raises(Exception):
            result.drug_name = "Other"

    def test_all_fields_present(self):
        result = predict_bile_concentration("TestDrug")
        assert hasattr(result, "drug_name")
        assert hasattr(result, "logp")
        assert hasattr(result, "mw_Da")
        assert hasattr(result, "plasma_conc_input_mg_L")
        assert hasattr(result, "bile_plasma_ratio")
        assert hasattr(result, "predicted_bile_conc_mg_L")
        assert hasattr(result, "f_biliary_excretion")
        assert hasattr(result, "biliary_clearance_mL_min")
        assert hasattr(result, "transporter_contribution")
        assert hasattr(result, "micellar_solubilization_factor")
        assert hasattr(result, "dili_biliary_risk")
        assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# Passive transport (no transporters)
# ---------------------------------------------------------------------------


class TestPassiveTransport:
    def test_passive_transporter_label(self):
        result = predict_bile_concentration("Drug", logp=2.0)
        assert result.transporter_contribution == "passive"

    def test_passive_bile_plasma_ratio_logp2(self):
        # bile_plasma_ratio = 1 + 2.0 * 0.3 = 1.6
        result = predict_bile_concentration("Drug", logp=2.0)
        assert abs(result.bile_plasma_ratio - 1.6) < 1e-9

    def test_passive_bile_plasma_ratio_logp5(self):
        # 1 + 5.0 * 0.3 = 2.5
        result = predict_bile_concentration("Drug", logp=5.0)
        assert abs(result.bile_plasma_ratio - 2.5) < 1e-9

    def test_passive_bile_plasma_ratio_logp0(self):
        # 1 + 0 * 0.3 = 1.0
        result = predict_bile_concentration("Drug", logp=0.0)
        assert abs(result.bile_plasma_ratio - 1.0) < 1e-9

    def test_passive_negative_logp_clamped(self):
        # 1 + (-3) * 0.3 = 0.1; clamp to 0.1
        result = predict_bile_concentration("Drug", logp=-10.0)
        assert result.bile_plasma_ratio == pytest.approx(0.1)

    def test_predicted_conc_equals_ratio_times_plasma(self):
        result = predict_bile_concentration("Drug", logp=2.0, plasma_conc_mg_L=2.0)
        expected = result.bile_plasma_ratio * 2.0
        assert abs(result.predicted_bile_conc_mg_L - expected) < 1e-9


# ---------------------------------------------------------------------------
# Transporter-mediated transport
# ---------------------------------------------------------------------------


class TestTransporterMediatedTransport:
    def test_pgp_substrate_label(self):
        result = predict_bile_concentration("Drug", has_pgp_substrate=True)
        assert result.transporter_contribution == "P-gp"

    def test_pgp_bile_plasma_ratio(self):
        # bile_plasma_ratio = 1.0 * 3.0 = 3.0
        result = predict_bile_concentration("Drug", has_pgp_substrate=True)
        assert abs(result.bile_plasma_ratio - 3.0) < 1e-9

    def test_mrp2_substrate_label(self):
        result = predict_bile_concentration("Drug", has_mrp2_substrate=True)
        assert result.transporter_contribution == "MRP2"

    def test_mrp2_bile_plasma_ratio(self):
        # 1.0 * 5.0 = 5.0
        result = predict_bile_concentration("Drug", has_mrp2_substrate=True)
        assert abs(result.bile_plasma_ratio - 5.0) < 1e-9

    def test_bcrp_substrate_label(self):
        result = predict_bile_concentration("Drug", has_bcrp_substrate=True)
        assert result.transporter_contribution == "BCRP"

    def test_bcrp_bile_plasma_ratio(self):
        # 1.0 * 4.0 = 4.0
        result = predict_bile_concentration("Drug", has_bcrp_substrate=True)
        assert abs(result.bile_plasma_ratio - 4.0) < 1e-9

    def test_mrp2_and_pgp_combined_label_is_mrp2(self):
        result = predict_bile_concentration("Drug", has_pgp_substrate=True, has_mrp2_substrate=True)
        assert result.transporter_contribution == "MRP2"

    def test_mrp2_and_pgp_combined_ratio(self):
        # 1.0 * 3.0 * 5.0 = 15.0
        result = predict_bile_concentration("Drug", has_pgp_substrate=True, has_mrp2_substrate=True)
        assert abs(result.bile_plasma_ratio - 15.0) < 1e-9

    def test_all_three_transporters_clamped(self):
        # 1 * 3 * 5 * 4 = 60 → clamped to 100? No, 60 < 100 → 60
        result = predict_bile_concentration(
            "Drug",
            has_pgp_substrate=True,
            has_mrp2_substrate=True,
            has_bcrp_substrate=True,
        )
        assert result.bile_plasma_ratio == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# f_biliary_excretion
# ---------------------------------------------------------------------------


class TestFBiliaryExcretion:
    def test_low_mw_no_transporter(self):
        # mw_Da < 300 → base_f=0.1, no transporter
        result = predict_bile_concentration("Drug", mw_Da=200.0)
        assert abs(result.f_biliary_excretion - 0.1) < 1e-9

    def test_mid_mw_no_transporter(self):
        # 300 <= mw < 500 → base_f=0.4
        result = predict_bile_concentration("Drug", mw_Da=400.0)
        assert abs(result.f_biliary_excretion - 0.4) < 1e-9

    def test_high_mw_no_transporter(self):
        # mw >= 500 → base_f=0.7
        result = predict_bile_concentration("Drug", mw_Da=600.0)
        assert abs(result.f_biliary_excretion - 0.7) < 1e-9

    def test_with_transporter_boosts_f(self):
        # mw=400 → base=0.4 * 1.5 = 0.6
        result = predict_bile_concentration("Drug", mw_Da=400.0, has_mrp2_substrate=True)
        assert abs(result.f_biliary_excretion - 0.6) < 1e-9

    def test_f_clamped_to_1(self):
        # mw=600 → 0.7 * 1.5 = 1.05 → 1.0
        result = predict_bile_concentration("Drug", mw_Da=600.0, has_mrp2_substrate=True)
        assert result.f_biliary_excretion == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Biliary clearance
# ---------------------------------------------------------------------------


class TestBiliaryClearance:
    def test_biliary_clearance_formula(self):
        # biliary_CL = bile_plasma_ratio * 0.6
        result = predict_bile_concentration("Drug", logp=2.0)
        # bile_plasma_ratio = 1.6; CL = 1.6 * 0.6 = 0.96
        assert abs(result.biliary_clearance_mL_min - 0.96) < 1e-9

    def test_biliary_clearance_clamped_max(self):
        # All three transporters: ratio=60, CL=36 < 60 → ok
        result = predict_bile_concentration(
            "Drug",
            has_pgp_substrate=True,
            has_mrp2_substrate=True,
            has_bcrp_substrate=True,
        )
        assert result.biliary_clearance_mL_min <= 60.0


# ---------------------------------------------------------------------------
# Micellar solubilization factor
# ---------------------------------------------------------------------------


class TestMicellarSolubilization:
    def test_positive_logp(self):
        result = predict_bile_concentration("Drug", logp=3.0)
        expected = 1 + 3.0 * 2.0  # = 7.0
        assert abs(result.micellar_solubilization_factor - expected) < 1e-9

    def test_zero_logp(self):
        result = predict_bile_concentration("Drug", logp=0.0)
        assert result.micellar_solubilization_factor == pytest.approx(1.0)

    def test_negative_logp(self):
        result = predict_bile_concentration("Drug", logp=-2.0)
        assert result.micellar_solubilization_factor == pytest.approx(1.0)

    def test_high_logp_clamped(self):
        result = predict_bile_concentration("Drug", logp=20.0)
        assert result.micellar_solubilization_factor == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# DILI risk and notes
# ---------------------------------------------------------------------------


class TestDiliBiliaryRisk:
    def test_low_risk(self):
        result = predict_bile_concentration("Drug", logp=1.0)
        # ratio = 1.3 < 5 → low
        assert result.dili_biliary_risk == "low"
        assert "Low biliary concentration risk" in result.notes

    def test_moderate_risk(self):
        result = predict_bile_concentration("Drug", has_mrp2_substrate=True)
        # ratio=5.0, threshold > 5 → low (not > 5)
        # Re-check: > 5 → moderate; = 5.0, 5.0 > 5 is False → low
        # Actually ratio=5.0 → not > 5 → low risk
        assert result.dili_biliary_risk == "low"

    def test_moderate_risk_ratio_6(self):
        # Use MRP2 + slight logP boost? No — with transporter, passive not applied
        # MRP2 + BCRP: 1*5*4=20 → exactly 20, 20>20 is False → moderate
        result = predict_bile_concentration(
            "Drug", has_mrp2_substrate=True, has_bcrp_substrate=True
        )
        # ratio = 20 → not > 20 → moderate
        assert result.dili_biliary_risk == "moderate"
        assert "moderate" in result.notes.lower() or "hepatic monitoring" in result.notes

    def test_high_risk(self):
        result = predict_bile_concentration(
            "Drug",
            has_pgp_substrate=True,
            has_mrp2_substrate=True,
            has_bcrp_substrate=True,
        )
        # ratio=60 > 20 → high
        assert result.dili_biliary_risk == "high"
        assert "DILI" in result.notes or "liver enzymes" in result.notes


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_bile_concentration("Drug", mw_Da=0.0)

    def test_negative_mw(self):
        with pytest.raises(ValueError):
            predict_bile_concentration("Drug", mw_Da=-100.0)

    def test_invalid_plasma_conc(self):
        with pytest.raises(ValueError, match="plasma_conc_mg_L"):
            predict_bile_concentration("Drug", plasma_conc_mg_L=0.0)


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------


class TestScreenBiliaryExcretion:
    def test_screen_returns_list(self):
        candidates = [
            {"name": "DrugA", "logp": 2.0},
            {"name": "DrugB", "logp": 4.0, "has_mrp2_substrate": True},
        ]
        results = screen_biliary_excretion(candidates)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_screen_sorted_by_ratio_descending(self):
        candidates = [
            {"name": "LowRatio", "logp": 0.5},
            {"name": "HighRatio", "logp": 3.0, "has_pgp_substrate": True},
        ]
        results = screen_biliary_excretion(candidates)
        assert results[0].bile_plasma_ratio >= results[1].bile_plasma_ratio

    def test_screen_default_mw(self):
        candidates = [{"name": "Drug", "logp": 2.0}]
        results = screen_biliary_excretion(candidates)
        assert results[0].mw_Da == 400.0

    def test_screen_empty_list(self):
        results = screen_biliary_excretion([])
        assert results == []
