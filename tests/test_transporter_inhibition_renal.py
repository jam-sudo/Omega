"""Tests for Phase 1083 — Drug-Induced Transporter Inhibition (OAT/OCT)."""

import pytest

from omega_pbpk.prediction.transporter_inhibition_renal import (
    RenalTransporterInhibitionResult,
    predict_renal_transporter_inhibition,
    screen_transporters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base(**overrides):
    defaults = dict(
        victim_drug="metformin",
        perpetrator_drug="cimetidine",
        transporter="OCT2",
        ki_mg_per_L=1.0,
        perpetrator_cmax_mg_per_L=5.0,
    )
    defaults.update(overrides)
    return predict_renal_transporter_inhibition(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_is_dataclass_instance(self):
        result = _base()
        assert isinstance(result, RenalTransporterInhibitionResult)

    def test_victim_drug_stored(self):
        result = _base(victim_drug="metformin")
        assert result.victim_drug == "metformin"

    def test_perpetrator_drug_stored(self):
        result = _base(perpetrator_drug="cimetidine")
        assert result.perpetrator_drug == "cimetidine"

    def test_transporter_stored(self):
        result = _base(transporter="OAT1")
        assert result.transporter == "OAT1"


# ---------------------------------------------------------------------------
# Core AUCR behaviour
# ---------------------------------------------------------------------------

class TestAUCRBehaviour:
    def test_zero_cmax_gives_aucr_one(self):
        result = _base(perpetrator_cmax_mg_per_L=0.0)
        assert abs(result.aucr - 1.0) < 1e-6

    def test_higher_cmax_gives_higher_aucr(self):
        low = _base(perpetrator_cmax_mg_per_L=1.0)
        high = _base(perpetrator_cmax_mg_per_L=10.0)
        assert high.aucr > low.aucr

    def test_lower_ki_gives_higher_aucr(self):
        weak = _base(ki_mg_per_L=10.0, perpetrator_cmax_mg_per_L=5.0)
        strong = _base(ki_mg_per_L=0.5, perpetrator_cmax_mg_per_L=5.0)
        assert strong.aucr > weak.aucr

    def test_aucr_always_ge_1(self):
        for cmax in [0.0, 0.1, 1.0, 10.0, 100.0]:
            result = _base(perpetrator_cmax_mg_per_L=cmax)
            assert result.aucr >= 1.0

    def test_cl_sec_inhibited_le_baseline(self):
        result = _base(perpetrator_cmax_mg_per_L=5.0)
        assert result.cl_sec_inhibited_L_per_h <= result.cl_sec_baseline_L_per_h

    def test_cl_sec_inhibited_approaches_zero_at_very_high_cmax(self):
        result = _base(perpetrator_cmax_mg_per_L=1e6)
        assert result.cl_sec_inhibited_L_per_h < 0.01

    def test_zero_cmax_cl_sec_inhibited_equals_baseline(self):
        result = _base(perpetrator_cmax_mg_per_L=0.0)
        assert abs(result.cl_sec_inhibited_L_per_h - result.cl_sec_baseline_L_per_h) < 1e-9


# ---------------------------------------------------------------------------
# Filtration clearance
# ---------------------------------------------------------------------------

class TestFiltrationCL:
    def test_cl_fil_uses_gfr_and_fup(self):
        result = predict_renal_transporter_inhibition(
            victim_drug="drug_a",
            perpetrator_drug="drug_b",
            transporter="OAT1",
            ki_mg_per_L=2.0,
            perpetrator_cmax_mg_per_L=4.0,
            gfr_L_per_h=6.0,
            fu_plasma=0.4,
        )
        assert abs(result.cl_fil_L_per_h - 6.0 * 0.4) < 1e-9


# ---------------------------------------------------------------------------
# DDI risk classification
# ---------------------------------------------------------------------------

class TestDDIRisk:
    def test_ddi_risk_values_valid(self):
        valid = {"none", "mild", "moderate", "severe"}
        for cmax in [0.0, 0.5, 5.0, 50.0]:
            r = _base(perpetrator_cmax_mg_per_L=cmax)
            assert r.ddi_risk in valid

    def test_none_when_aucr_lt_1_25(self):
        # very high Ki relative to Cmax → minimal inhibition → AUCR close to 1
        r = _base(ki_mg_per_L=1000.0, perpetrator_cmax_mg_per_L=0.001)
        assert r.ddi_risk == "none"

    def test_severe_when_aucr_gt_5(self):
        # Very low Ki and high Cmax on OCT2 (CL_sec=3.0 L/h)
        r = predict_renal_transporter_inhibition(
            victim_drug="metformin",
            perpetrator_drug="perpetrator",
            transporter="OCT2",
            ki_mg_per_L=0.01,
            perpetrator_cmax_mg_per_L=100.0,
            gfr_L_per_h=7.2,
            fu_plasma=0.1,
        )
        assert r.ddi_risk == "severe"


# ---------------------------------------------------------------------------
# Regulatory concern
# ---------------------------------------------------------------------------

class TestRegulatoryConcern:
    def test_is_bool(self):
        result = _base()
        assert isinstance(result.regulatory_concern, bool)

    def test_high_aucr_triggers_concern(self):
        r = _base(ki_mg_per_L=0.01, perpetrator_cmax_mg_per_L=100.0)
        assert r.regulatory_concern is True

    def test_low_aucr_no_concern(self):
        r = _base(ki_mg_per_L=1000.0, perpetrator_cmax_mg_per_L=0.001)
        assert r.regulatory_concern is False


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_recommendation_nonempty(self):
        result = _base()
        assert len(result.recommendation) > 0

    def test_notes_nonempty(self):
        result = _base()
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# screen_transporters
# ---------------------------------------------------------------------------

class TestScreenTransporters:
    def _ki_dict(self):
        return {
            "OAT1": 1.0,
            "OAT3": 2.0,
            "OCT2": 0.5,
        }

    def test_returns_correct_count(self):
        results = screen_transporters(
            "metformin", "cimetidine", self._ki_dict(), 5.0
        )
        assert len(results) == 3

    def test_sorted_by_aucr_descending(self):
        results = screen_transporters(
            "metformin", "cimetidine", self._ki_dict(), 5.0
        )
        aucrs = [r.aucr for r in results]
        assert aucrs == sorted(aucrs, reverse=True)

    def test_single_transporter_screen(self):
        results = screen_transporters(
            "tenofovir", "probenecid", {"OAT1": 0.3}, 2.0
        )
        assert len(results) == 1

    def test_five_transporter_screen(self):
        ki_dict = {t: 1.0 for t in ["OAT1", "OAT3", "OCT2", "MATE1", "P_glycoprotein_renal"]}
        results = screen_transporters("drug_v", "drug_p", ki_dict, 5.0)
        assert len(results) == 5

    def test_kwargs_forwarded(self):
        results = screen_transporters(
            "drug_v", "drug_p", {"OCT2": 1.0}, 5.0,
            gfr_L_per_h=5.0, fu_plasma=0.3
        )
        assert abs(results[0].cl_fil_L_per_h - 5.0 * 0.3) < 1e-9


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_transporter_raises_value_error(self):
        with pytest.raises(ValueError, match="transporter"):
            predict_renal_transporter_inhibition(
                "drug_a", "drug_b", "INVALID_TRANSPORTER",
                ki_mg_per_L=1.0, perpetrator_cmax_mg_per_L=5.0,
            )

    def test_zero_ki_raises_value_error(self):
        with pytest.raises(ValueError, match="ki_mg_per_L"):
            _base(ki_mg_per_L=0.0)

    def test_negative_ki_raises_value_error(self):
        with pytest.raises(ValueError, match="ki_mg_per_L"):
            _base(ki_mg_per_L=-1.0)

    def test_negative_cmax_raises_value_error(self):
        with pytest.raises(ValueError):
            _base(perpetrator_cmax_mg_per_L=-0.1)

    def test_zero_gfr_raises_value_error(self):
        with pytest.raises(ValueError, match="gfr"):
            predict_renal_transporter_inhibition(
                "a", "b", "OAT1", 1.0, 5.0, gfr_L_per_h=0.0
            )

    def test_fu_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_renal_transporter_inhibition(
                "a", "b", "OAT1", 1.0, 5.0, fu_plasma=0.0
            )

    def test_fu_above_one_raises_value_error(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_renal_transporter_inhibition(
                "a", "b", "OAT1", 1.0, 5.0, fu_plasma=1.1
            )

    def test_all_transporters_accepted(self):
        for t in ["OAT1", "OAT3", "OCT2", "MATE1", "P_glycoprotein_renal"]:
            result = predict_renal_transporter_inhibition(
                "drug_v", "drug_p", t,
                ki_mg_per_L=1.0, perpetrator_cmax_mg_per_L=2.0,
            )
            assert result.transporter == t
