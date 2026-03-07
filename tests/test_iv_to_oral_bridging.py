"""Tests for iv_to_oral_bridging module."""

import math

import pytest

from omega_pbpk.clinical.iv_to_oral_bridging import (
    IVOralBridgeResult,
    bridge_iv_to_oral,
    predict_oral_bioavailability,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def make_result(
    drug_name="TestDrug",
    iv_dose_mg=100.0,
    iv_auc_mg_L_h=10.0,
    cl_iv_L_per_h=10.0,
    vd_iv_L=70.0,
    oral_dose_mg=100.0,
    bcs_class="I",
    psa=60.0,
    logP=2.0,
    oral_auc_observed=None,
):
    return bridge_iv_to_oral(
        drug_name=drug_name,
        iv_dose_mg=iv_dose_mg,
        iv_auc_mg_L_h=iv_auc_mg_L_h,
        cl_iv_L_per_h=cl_iv_L_per_h,
        vd_iv_L=vd_iv_L,
        oral_dose_mg=oral_dose_mg,
        bcs_class=bcs_class,
        psa=psa,
        logP=logP,
        oral_auc_observed=oral_auc_observed,
    )


# ── basic construction ────────────────────────────────────────────────────────


def test_returns_ivoral_bridge_result():
    result = make_result()
    assert isinstance(result, IVOralBridgeResult)


def test_drug_name_preserved():
    result = make_result(drug_name="Midazolam")
    assert result.drug_name == "Midazolam"


def test_iv_pk_fields_preserved():
    result = make_result(cl_iv_L_per_h=5.0, vd_iv_L=50.0, iv_auc_mg_L_h=8.0)
    assert result.cl_iv_L_per_h == pytest.approx(5.0)
    assert result.vd_iv_L == pytest.approx(50.0)
    assert result.auc_iv_mg_L_h == pytest.approx(8.0)


def test_t_half_iv_computed():
    # t½ = 0.693 * Vd / CL
    result = make_result(cl_iv_L_per_h=10.0, vd_iv_L=70.0)
    expected = 0.693 * 70.0 / 10.0
    assert result.t_half_iv_h == pytest.approx(expected, rel=1e-4)


# ── hepatic extraction ────────────────────────────────────────────────────────


def test_high_cl_drug_low_fh():
    # CL = 75 L/h, Qh = 80 L/h → EH ≈ 0.9375, FH ≈ 0.0625
    result = make_result(cl_iv_L_per_h=75.0)
    assert result.fh_predicted < 0.1


def test_low_cl_drug_high_fh():
    # CL = 2 L/h, Qh = 80 → EH = 0.025, FH = 0.975
    result = make_result(cl_iv_L_per_h=2.0)
    assert result.fh_predicted > 0.9


# ── BCS class effects ─────────────────────────────────────────────────────────


def test_bcs_I_higher_fa_than_bcs_IV():
    r1 = make_result(bcs_class="I")
    r4 = make_result(bcs_class="IV")
    assert r1.fa_predicted > r4.fa_predicted


def test_bcs_I_fa_is_095():
    result = make_result(bcs_class="I")
    assert result.fa_predicted == pytest.approx(0.95)


def test_bcs_III_fa_is_085():
    result = make_result(bcs_class="III")
    assert result.fa_predicted == pytest.approx(0.85)


def test_bcs_II_high_psa_reduces_fa():
    r_low = make_result(bcs_class="II", psa=40.0)
    r_high = make_result(bcs_class="II", psa=100.0)
    assert r_low.fa_predicted > r_high.fa_predicted


def test_bcs_II_fa_floored_at_030():
    # Very high PSA should floor at 0.3
    result = make_result(bcs_class="II", psa=1000.0)
    assert result.fa_predicted == pytest.approx(0.3)


def test_bcs_IV_fa_floored_at_010():
    # Very high logP → floor at 0.1
    result = make_result(bcs_class="IV", logP=100.0)
    assert result.fa_predicted == pytest.approx(0.1)


# ── f_oral = fa * fg * fh ─────────────────────────────────────────────────────


def test_f_oral_equals_fa_fg_fh():
    result = make_result()
    expected = result.fa_predicted * result.fg_predicted * result.fh_predicted
    assert result.f_oral_predicted == pytest.approx(expected, rel=1e-6)


def test_high_cl_gives_low_f_oral():
    # High extraction → low bioavailability
    result = make_result(cl_iv_L_per_h=75.0, bcs_class="I")
    assert result.f_oral_predicted < 0.15


# ── tmax > 0 ──────────────────────────────────────────────────────────────────


def test_tmax_positive():
    result = make_result()
    assert result.tmax_predicted_h > 0.0


def test_cmax_positive():
    result = make_result()
    assert result.cmax_predicted_mg_L > 0.0


# ── AUC oral predicted ────────────────────────────────────────────────────────


def test_auc_oral_predicted_consistent():
    result = make_result(oral_dose_mg=100.0, cl_iv_L_per_h=10.0)
    expected = 100.0 * result.f_oral_predicted / 10.0
    assert result.auc_oral_predicted_mg_L_h == pytest.approx(expected, rel=1e-5)


# ── validation with observed AUC ──────────────────────────────────────────────


def test_prediction_error_pct_with_observed():
    # If observed AUC implies exactly the predicted F, error should be ~0
    result = make_result(
        cl_iv_L_per_h=10.0,
        oral_dose_mg=100.0,
        oral_auc_observed=100.0 * 0.5 / 10.0,  # F=0.5
    )
    # f_oral_predicted may differ from 0.5 but error should be a finite number
    assert math.isfinite(result.prediction_error_pct)
    assert not math.isnan(result.f_observed)


def test_prediction_error_pct_nan_without_observed():
    result = make_result()
    assert math.isnan(result.prediction_error_pct)
    assert math.isnan(result.f_observed)


def test_prediction_error_reasonable():
    # Use a BCS I, low-CL drug and set observed AUC equal to predicted
    cl = 5.0
    oral_dose = 100.0
    bcs = "I"
    result = make_result(
        cl_iv_L_per_h=cl,
        oral_dose_mg=oral_dose,
        bcs_class=bcs,
        logP=2.0,
        psa=60.0,
        oral_auc_observed=None,
    )
    # Now pass AUC that matches predicted exactly
    pred_auc = result.auc_oral_predicted_mg_L_h
    result2 = make_result(
        cl_iv_L_per_h=cl,
        oral_dose_mg=oral_dose,
        bcs_class=bcs,
        logP=2.0,
        psa=60.0,
        oral_auc_observed=pred_auc,
    )
    assert result2.prediction_error_pct == pytest.approx(0.0, abs=1e-4)


# ── predict_oral_bioavailability ──────────────────────────────────────────────


def test_predict_oral_bioavailability_returns_dict():
    result = predict_oral_bioavailability(
        drug_name="DrugX",
        cl_iv_L_per_h=10.0,
        vd_iv_L=70.0,
    )
    assert isinstance(result, dict)
    for key in ("fa", "fg", "fh", "f_oral"):
        assert key in result


def test_predict_oral_bioavailability_values():
    result = predict_oral_bioavailability(
        drug_name="DrugX",
        cl_iv_L_per_h=10.0,
        vd_iv_L=70.0,
        bcs_class="I",
        logP=2.0,
        psa=60.0,
    )
    assert 0 < result["fa"] <= 1.0
    assert 0 < result["fg"] <= 1.0
    assert 0 < result["fh"] <= 1.0
    assert result["f_oral"] == pytest.approx(result["fa"] * result["fg"] * result["fh"], rel=1e-6)


# ── invalid inputs ────────────────────────────────────────────────────────────


def test_invalid_bcs_class_raises():
    with pytest.raises(ValueError, match="BCS class"):
        make_result(bcs_class="V")


def test_invalid_bcs_class_in_predict_raises():
    with pytest.raises(ValueError, match="BCS class"):
        predict_oral_bioavailability("X", 10.0, 70.0, bcs_class="Z")


def test_zero_iv_dose_raises():
    with pytest.raises(ValueError):
        bridge_iv_to_oral("X", 0.0, 10.0, 10.0, 70.0, 100.0)


def test_zero_cl_raises():
    with pytest.raises(ValueError):
        bridge_iv_to_oral("X", 100.0, 10.0, 0.0, 70.0, 100.0)


def test_zero_oral_dose_raises():
    with pytest.raises(ValueError):
        bridge_iv_to_oral("X", 100.0, 10.0, 10.0, 70.0, 0.0)
