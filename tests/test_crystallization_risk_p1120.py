"""Tests for Phase 1120 — crystallization_risk_p1120."""

import pytest

from omega_pbpk.prediction.crystallization_risk_p1120 import (
    CrystallizationRiskResult,
    assess_crystallization_risk,
    screen_formulations,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make(
    drug_name="DrugX",
    logP=2.0,
    pka=7.0,
    ionization_type="neutral",
    glass_transition_temp_C=100.0,
    drug_loading_pct=30.0,
    storage_rh_pct=40.0,
    storage_temp_C=25.0,
):
    return assess_crystallization_risk(
        drug_name=drug_name,
        logP=logP,
        pka=pka,
        ionization_type=ionization_type,
        glass_transition_temp_C=glass_transition_temp_C,
        drug_loading_pct=drug_loading_pct,
        storage_rh_pct=storage_rh_pct,
        storage_temp_C=storage_temp_C,
    )


# ---------------------------------------------------------------------------
# Tg ratio / mobility score
# ---------------------------------------------------------------------------

class TestMobilityScore:
    def test_low_ratio_low_mobility(self):
        # T_storage ~ 298 K, Tg ~ 453 K → ratio ≈ 0.658 < 0.7 → score = 5
        res = _make(glass_transition_temp_C=180.0, storage_temp_C=25.0)
        assert res.mobility_score == pytest.approx(5.0)

    def test_ratio_0_7_to_0_8(self):
        # Need ratio in [0.7, 0.8) → Tg = 298/0.75 - 273 ≈ 124 °C; storage=25°C
        # 298/(124+273)=298/397=0.750 → score=15
        res = _make(glass_transition_temp_C=124.0, storage_temp_C=25.0)
        assert res.mobility_score == pytest.approx(15.0)
        assert 0.7 <= res.tg_ratio < 0.8

    def test_ratio_0_8_to_0_9(self):
        # 298/(75+273)=298/348=0.857 → score=25
        res = _make(glass_transition_temp_C=75.0, storage_temp_C=25.0)
        assert res.mobility_score == pytest.approx(25.0)
        assert 0.8 <= res.tg_ratio < 0.9

    def test_ratio_ge_0_9(self):
        # Storage=60°C=333K, Tg=40°C=313K → 333/313=1.06 → score=40
        res = _make(glass_transition_temp_C=40.0, storage_temp_C=60.0)
        assert res.mobility_score == pytest.approx(40.0)
        assert res.tg_ratio >= 0.9

    def test_high_storage_temp_higher_score_than_low(self):
        res_low = _make(glass_transition_temp_C=100.0, storage_temp_C=5.0)
        res_high = _make(glass_transition_temp_C=100.0, storage_temp_C=60.0)
        assert res_high.mobility_score >= res_low.mobility_score


# ---------------------------------------------------------------------------
# Hygroscopicity score
# ---------------------------------------------------------------------------

class TestHygroscopicityScore:
    def test_low_rh_base_5(self):
        res = _make(storage_rh_pct=20.0, drug_loading_pct=30.0)
        assert res.hygroscopicity_score == pytest.approx(5.0)

    def test_medium_rh_base_15(self):
        res = _make(storage_rh_pct=50.0, drug_loading_pct=30.0)
        assert res.hygroscopicity_score == pytest.approx(15.0)

    def test_high_rh_base_25(self):
        res = _make(storage_rh_pct=75.0, drug_loading_pct=30.0)
        assert res.hygroscopicity_score == pytest.approx(25.0)

    def test_high_drug_loading_adds_5(self):
        res_low_load = _make(storage_rh_pct=50.0, drug_loading_pct=30.0)
        res_high_load = _make(storage_rh_pct=50.0, drug_loading_pct=50.0)
        assert res_high_load.hygroscopicity_score == pytest.approx(res_low_load.hygroscopicity_score + 5.0)

    def test_drug_loading_40pct_boundary_no_extra(self):
        # Exactly 40% → no addition
        res = _make(storage_rh_pct=40.0, drug_loading_pct=40.0)
        assert res.hygroscopicity_score == pytest.approx(15.0)  # base 15 only


# ---------------------------------------------------------------------------
# Chemical score
# ---------------------------------------------------------------------------

class TestChemicalScore:
    def test_neutral_lipophilic_logP_above_3(self):
        res = _make(logP=4.0, ionization_type="neutral")
        assert res.chemical_score == pytest.approx(20.0)

    def test_acid_ionizable(self):
        res = _make(logP=4.0, ionization_type="acid")
        assert res.chemical_score == pytest.approx(5.0)

    def test_base_ionizable(self):
        res = _make(logP=2.0, ionization_type="base")
        assert res.chemical_score == pytest.approx(5.0)

    def test_hydrophilic_logP_below_0(self):
        res = _make(logP=-1.0, ionization_type="neutral")
        assert res.chemical_score == pytest.approx(15.0)

    def test_neutral_mid_logP(self):
        res = _make(logP=1.5, ionization_type="neutral")
        assert res.chemical_score == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Total risk score and classification
# ---------------------------------------------------------------------------

class TestRiskClassification:
    def test_total_is_sum_of_components(self):
        res = _make()
        expected = res.mobility_score + res.hygroscopicity_score + res.chemical_score
        assert res.crystallization_risk_score == pytest.approx(expected)

    def test_low_risk_class(self):
        # Glassy (Tg high), low RH, ionizable → low scores
        res = _make(
            logP=2.0,
            ionization_type="acid",
            glass_transition_temp_C=180.0,
            drug_loading_pct=20.0,
            storage_rh_pct=20.0,
            storage_temp_C=25.0,
        )
        assert res.crystallization_risk_score < 30.0
        assert res.risk_class == "low"

    def test_high_risk_class(self):
        # Near Tg, high RH, neutral lipophilic → high scores
        res = _make(
            logP=4.0,
            ionization_type="neutral",
            glass_transition_temp_C=40.0,
            drug_loading_pct=50.0,
            storage_rh_pct=80.0,
            storage_temp_C=60.0,
        )
        assert res.crystallization_risk_score >= 60.0
        assert res.risk_class == "high"

    def test_moderate_risk_class(self):
        res = _make(
            logP=2.0,
            ionization_type="neutral",
            glass_transition_temp_C=75.0,
            drug_loading_pct=30.0,
            storage_rh_pct=40.0,
            storage_temp_C=25.0,
        )
        # mobility=25, hygro=15, chemical=10 → 50 → moderate
        assert res.risk_class == "moderate"

    def test_result_type(self):
        res = _make()
        assert isinstance(res, CrystallizationRiskResult)

    def test_drug_name_preserved(self):
        res = _make(drug_name="Ibuprofen")
        assert res.drug_name == "Ibuprofen"

    def test_recommended_action_not_empty(self):
        res = _make()
        assert len(res.recommended_action) > 0


# ---------------------------------------------------------------------------
# screen_formulations
# ---------------------------------------------------------------------------

class TestScreenFormulations:
    def _make_dict(self, **kwargs):
        base = dict(
            drug_name="DrugX",
            logP=2.0,
            ionization_type="neutral",
            glass_transition_temp_C=100.0,
            drug_loading_pct=30.0,
            storage_rh_pct=40.0,
            storage_temp_C=25.0,
        )
        base.update(kwargs)
        return base

    def test_sorted_ascending(self):
        forms = [
            self._make_dict(drug_name="High", logP=4.0, glass_transition_temp_C=40.0, storage_temp_C=60.0, storage_rh_pct=75.0, drug_loading_pct=50.0),
            self._make_dict(drug_name="Low", logP=2.0, ionization_type="acid", glass_transition_temp_C=180.0, storage_rh_pct=20.0),
            self._make_dict(drug_name="Mid", logP=1.5, glass_transition_temp_C=75.0, storage_rh_pct=40.0),
        ]
        results = screen_formulations(forms)
        scores = [r.crystallization_risk_score for r in results]
        assert scores == sorted(scores)

    def test_lowest_risk_first(self):
        forms = [
            self._make_dict(drug_name="Risky", logP=4.0, ionization_type="neutral", glass_transition_temp_C=40.0, storage_temp_C=60.0),
            self._make_dict(drug_name="Safe", logP=2.0, ionization_type="acid", glass_transition_temp_C=180.0, storage_rh_pct=10.0),
        ]
        results = screen_formulations(forms)
        assert results[0].drug_name == "Safe"

    def test_single_formulation(self):
        results = screen_formulations([self._make_dict()])
        assert len(results) == 1

    def test_returns_list_of_results(self):
        forms = [self._make_dict(drug_name=f"Drug{i}") for i in range(3)]
        results = screen_formulations(forms)
        assert all(isinstance(r, CrystallizationRiskResult) for r in results)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_logP_too_low(self):
        with pytest.raises(ValueError):
            _make(logP=-6.0)

    def test_logP_too_high(self):
        with pytest.raises(ValueError):
            _make(logP=11.0)

    def test_invalid_ionization_type(self):
        with pytest.raises(ValueError):
            _make(ionization_type="zwitterion")

    def test_glass_transition_too_low(self):
        with pytest.raises(ValueError):
            _make(glass_transition_temp_C=10.0)

    def test_glass_transition_too_high(self):
        with pytest.raises(ValueError):
            _make(glass_transition_temp_C=350.0)

    def test_drug_loading_zero(self):
        with pytest.raises(ValueError):
            _make(drug_loading_pct=0.0)

    def test_drug_loading_over_100(self):
        with pytest.raises(ValueError):
            _make(drug_loading_pct=101.0)

    def test_storage_rh_negative(self):
        with pytest.raises(ValueError):
            _make(storage_rh_pct=-1.0)

    def test_storage_rh_over_100(self):
        with pytest.raises(ValueError):
            _make(storage_rh_pct=101.0)

    def test_storage_temp_too_low(self):
        with pytest.raises(ValueError):
            _make(storage_temp_C=-25.0)

    def test_storage_temp_too_high(self):
        with pytest.raises(ValueError):
            _make(storage_temp_C=85.0)
