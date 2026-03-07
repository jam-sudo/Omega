"""Tests for Phase 578: pk_pk_interaction."""

import pytest

from omega_pbpk.prediction.pk_pk_interaction import (
    ddi_risk_matrix,
    predict_cl_inhibition,
    predict_induction_effect,
    protein_binding_ddi,
    transporter_mediated_ddi,
)

# ---------------------------------------------------------------------------
# predict_cl_inhibition
# ---------------------------------------------------------------------------


class TestPredictClInhibition:
    def test_fm1_aucr2_victim_auc_doubles(self):
        result = predict_cl_inhibition(cl_baseline_L_per_h=10.0, fm_cyp=1.0, aucr_inhibitor=2.0)
        # CL_inhibited = 10 * (0 + 1/2) = 5; AUC ratio = 10/5 = 2
        assert result["cl_inhibited_L_per_h"] == pytest.approx(5.0, abs=1e-9)
        assert result["auc_ratio_victim"] == pytest.approx(2.0, abs=1e-9)

    def test_fm0_no_effect(self):
        result = predict_cl_inhibition(cl_baseline_L_per_h=10.0, fm_cyp=0.0, aucr_inhibitor=10.0)
        assert result["cl_inhibited_L_per_h"] == pytest.approx(10.0, abs=1e-9)
        assert result["auc_ratio_victim"] == pytest.approx(1.0, abs=1e-9)

    def test_partial_fm(self):
        # fm=0.5, aucr=2 → CL = 10*(0.5 + 0.5/2) = 10*0.75 = 7.5
        result = predict_cl_inhibition(cl_baseline_L_per_h=10.0, fm_cyp=0.5, aucr_inhibitor=2.0)
        assert result["cl_inhibited_L_per_h"] == pytest.approx(7.5, abs=1e-9)
        assert result["auc_ratio_victim"] == pytest.approx(10.0 / 7.5, abs=1e-6)

    def test_keys_present(self):
        result = predict_cl_inhibition(10.0, 0.5, 2.0)
        for key in ["cl_inhibited_L_per_h", "auc_ratio_victim", "fm_cyp", "notes"]:
            assert key in result

    def test_invalid_aucr(self):
        with pytest.raises(ValueError):
            predict_cl_inhibition(10.0, 0.5, 0.0)

    def test_invalid_fm_cyp_negative(self):
        with pytest.raises(ValueError):
            predict_cl_inhibition(10.0, -0.1, 2.0)

    def test_invalid_fm_cyp_above_one(self):
        with pytest.raises(ValueError):
            predict_cl_inhibition(10.0, 1.1, 2.0)


# ---------------------------------------------------------------------------
# predict_induction_effect
# ---------------------------------------------------------------------------


class TestPredictInductionEffect:
    def test_fold_induction_increases_cl(self):
        result = predict_induction_effect(cl_baseline_L_per_h=10.0, fm_cyp=1.0, fold_induction=3.0)
        assert result["cl_induced_L_per_h"] == pytest.approx(30.0, abs=1e-9)

    def test_auc_ratio_less_than_one(self):
        # AUC should decrease when CL increases
        result = predict_induction_effect(10.0, 1.0, 3.0)
        assert result["auc_ratio_victim"] < 1.0
        assert result["auc_ratio_victim"] == pytest.approx(1.0 / 3.0, abs=1e-9)

    def test_fm0_no_effect(self):
        result = predict_induction_effect(10.0, 0.0, 5.0)
        assert result["cl_induced_L_per_h"] == pytest.approx(10.0, abs=1e-9)
        assert result["auc_ratio_victim"] == pytest.approx(1.0, abs=1e-9)

    def test_partial_fm_induction(self):
        # fm=0.5, fold=2 → CL=10*(0.5 + 0.5*2)=15
        result = predict_induction_effect(10.0, 0.5, 2.0)
        assert result["cl_induced_L_per_h"] == pytest.approx(15.0, abs=1e-9)

    def test_keys_present(self):
        result = predict_induction_effect(10.0, 0.5, 2.0)
        for key in ["cl_induced_L_per_h", "auc_ratio_victim", "notes"]:
            assert key in result

    def test_invalid_fm_cyp(self):
        with pytest.raises(ValueError):
            predict_induction_effect(10.0, 1.5, 2.0)


# ---------------------------------------------------------------------------
# ddi_risk_matrix
# ---------------------------------------------------------------------------


class TestDdiRiskMatrix:
    def test_returns_n_perp_times_n_victim_rows(self):
        perps = [{"name": "P1", "aucr": 2.0}, {"name": "P2", "aucr": 5.0}]
        victims = [
            {"name": "V1", "fm_cyp": 1.0, "cl_baseline_L_per_h": 10.0},
            {"name": "V2", "fm_cyp": 0.5, "cl_baseline_L_per_h": 5.0},
            {"name": "V3", "fm_cyp": 0.0, "cl_baseline_L_per_h": 8.0},
        ]
        matrix = ddi_risk_matrix(perps, victims)
        assert len(matrix) == 6

    def test_high_risk_flagged(self):
        perps = [{"name": "strong_inh", "aucr": 10.0}]
        victims = [{"name": "V", "fm_cyp": 1.0, "cl_baseline_L_per_h": 10.0}]
        matrix = ddi_risk_matrix(perps, victims)
        assert matrix[0]["risk_level"] == "high"

    def test_negligible_risk_when_fm0(self):
        perps = [{"name": "P", "aucr": 10.0}]
        victims = [{"name": "V", "fm_cyp": 0.0, "cl_baseline_L_per_h": 10.0}]
        matrix = ddi_risk_matrix(perps, victims)
        assert matrix[0]["risk_level"] == "negligible"

    def test_keys_present(self):
        perps = [{"name": "P", "aucr": 2.0}]
        victims = [{"name": "V", "fm_cyp": 0.5, "cl_baseline_L_per_h": 10.0}]
        matrix = ddi_risk_matrix(perps, victims)
        for key in ["perpetrator", "victim", "predicted_auc_ratio", "risk_level"]:
            assert key in matrix[0]

    def test_empty_inputs(self):
        assert ddi_risk_matrix([], []) == []


# ---------------------------------------------------------------------------
# transporter_mediated_ddi
# ---------------------------------------------------------------------------


class TestTransporterMediatedDdi:
    def test_full_inhibition_only_passive_remains(self):
        result = transporter_mediated_ddi(
            cl_transport_L_per_h=8.0,
            cl_passive_L_per_h=2.0,
            transporter_inhibition_fraction=1.0,
        )
        assert result["cl_remaining_L_per_h"] == pytest.approx(2.0, abs=1e-9)
        assert result["auc_ratio"] == pytest.approx(5.0, abs=1e-9)

    def test_no_inhibition_no_change(self):
        result = transporter_mediated_ddi(8.0, 2.0, 0.0)
        assert result["cl_remaining_L_per_h"] == pytest.approx(10.0, abs=1e-9)
        assert result["auc_ratio"] == pytest.approx(1.0, abs=1e-9)

    def test_partial_inhibition(self):
        # cl_transport=6, cl_passive=4, inhibition=0.5
        # cl_remaining = 4 + 6*0.5 = 7; AUC ratio = 10/7
        result = transporter_mediated_ddi(6.0, 4.0, 0.5)
        assert result["cl_remaining_L_per_h"] == pytest.approx(7.0, abs=1e-9)
        assert result["auc_ratio"] == pytest.approx(10.0 / 7.0, abs=1e-6)

    def test_clinical_significance_high(self):
        result = transporter_mediated_ddi(9.0, 0.1, 1.0)
        assert result["clinical_significance"] == "high"

    def test_keys_present(self):
        result = transporter_mediated_ddi(5.0, 5.0, 0.5)
        for key in ["cl_remaining_L_per_h", "auc_ratio", "clinical_significance"]:
            assert key in result

    def test_invalid_inhibition_fraction(self):
        with pytest.raises(ValueError):
            transporter_mediated_ddi(5.0, 5.0, 1.5)


# ---------------------------------------------------------------------------
# protein_binding_ddi
# ---------------------------------------------------------------------------


class TestProteinBindingDdi:
    def test_fu_doubles_cl_doubles_auc_halves(self):
        result = protein_binding_ddi(
            fu_baseline=0.1,
            cl_baseline_L_per_h=10.0,
            fu_new=0.2,
        )
        assert result["cl_new"] == pytest.approx(20.0, abs=1e-9)
        # auc_ratio = fu_baseline / fu_new = 0.1 / 0.2 = 0.5
        assert result["auc_ratio"] == pytest.approx(0.5, abs=1e-9)

    def test_equal_fu_no_change(self):
        result = protein_binding_ddi(0.2, 10.0, 0.2)
        assert result["cl_new"] == pytest.approx(10.0, abs=1e-9)
        assert result["auc_ratio"] == pytest.approx(1.0, abs=1e-9)

    def test_fu_decreases_cl_decreases(self):
        result = protein_binding_ddi(0.2, 10.0, 0.1)
        assert result["cl_new"] == pytest.approx(5.0, abs=1e-9)
        assert result["auc_ratio"] == pytest.approx(2.0, abs=1e-9)

    def test_keys_present(self):
        result = protein_binding_ddi(0.1, 10.0, 0.2)
        for key in ["cl_new", "auc_ratio", "notes"]:
            assert key in result

    def test_invalid_fu_zero(self):
        with pytest.raises(ValueError):
            protein_binding_ddi(0.0, 10.0, 0.2)

    def test_invalid_fu_above_one(self):
        with pytest.raises(ValueError):
            protein_binding_ddi(0.5, 10.0, 1.5)
