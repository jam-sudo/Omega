"""Tests for clinical/obesity_pk.py — Phase 625 (adjust_for_obesity, compute_body_metrics)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.obesity_pk import (
    ObesityPKResult,
    adjust_for_obesity,
    compute_body_metrics,
    obesity_class_from_bmi,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _result(
    drug_type: str = "hydrophilic",
    weight: float = 100.0,
    height: float = 170.0,
    sex: str = "male",
) -> ObesityPKResult:
    return adjust_for_obesity(
        drug_name="TestDrug",
        total_body_weight_kg=weight,
        height_cm=height,
        sex=sex,
        baseline_cl_L_per_h=5.0,
        baseline_vd_L=50.0,
        baseline_dose_mg=100.0,
        drug_type=drug_type,
    )


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------


def test_returns_result():
    r = _result()
    assert isinstance(r, ObesityPKResult)


def test_bmi_correct():
    # weight=90, height=170 → bmi = 90 / (1.70)^2 = 90/2.89 ≈ 31.1
    r = adjust_for_obesity(
        drug_name="Drug",
        total_body_weight_kg=90.0,
        height_cm=170.0,
        sex="male",
        baseline_cl_L_per_h=5.0,
        baseline_vd_L=50.0,
        baseline_dose_mg=100.0,
    )
    assert r.bmi == pytest.approx(90.0 / (1.70**2), rel=1e-3)


def test_obesity_class_class1():
    r = obesity_class_from_bmi(32.0)
    assert r == "class1"


def test_obesity_class_class2():
    r = obesity_class_from_bmi(37.0)
    assert r == "class2"


def test_obesity_class_class3():
    r = obesity_class_from_bmi(42.0)
    assert r == "class3"


def test_obesity_class_normal():
    r = obesity_class_from_bmi(22.0)
    assert r == "normal"


def test_ibw_male_reasonable():
    # 70 kg male 175 cm → IBW ≈ 70 kg
    metrics = compute_body_metrics(70.0, 175.0, "male")
    assert 60.0 < metrics["ibw"] < 85.0


def test_ibw_female_lower():
    # same height female → IBW < male
    m = compute_body_metrics(70.0, 175.0, "male")
    f = compute_body_metrics(70.0, 175.0, "female")
    assert f["ibw"] < m["ibw"]


def test_lbm_less_than_tbw():
    metrics = compute_body_metrics(100.0, 170.0, "male")
    assert metrics["lbm"] < 100.0


def test_abw_between_ibw_and_tbw():
    # For obese patient, IBW <= ABW <= TBW
    metrics = compute_body_metrics(120.0, 170.0, "male")
    assert metrics["ibw"] <= metrics["abw"] <= 120.0


def test_lipophilic_higher_vd():
    r_lip = _result(drug_type="lipophilic", weight=120.0)
    r_int = _result(drug_type="intermediate", weight=120.0)
    assert r_lip.vd_scaling_factor >= r_int.vd_scaling_factor


def test_hydrophilic_lower_vd():
    # Hydrophilic Vd scales with LBM/70; for obese patient LBM << TBW
    r_hyd = _result(drug_type="hydrophilic", weight=120.0)
    r_lip = _result(drug_type="lipophilic", weight=120.0)
    assert r_hyd.vd_scaling_factor < r_lip.vd_scaling_factor


def test_adjusted_vd_positive():
    r = _result()
    assert r.adjusted_vd_L > 0.0


def test_adjusted_cl_positive():
    r = _result()
    assert r.adjusted_cl_L_per_h > 0.0


def test_adjusted_t_half_positive():
    r = _result()
    assert r.adjusted_t_half_h > 0.0


def test_dose_basis_hydrophilic_ibw():
    r = _result(drug_type="hydrophilic")
    assert r.recommended_dose_basis == "ibw"


def test_dose_basis_lipophilic_tbw():
    r = _result(drug_type="lipophilic")
    assert r.recommended_dose_basis == "tbw"


def test_recommended_dose_positive():
    r = _result()
    assert r.recommended_dose_mg > 0.0


def test_compute_body_metrics_returns_dict():
    result = compute_body_metrics(80.0, 175.0, "male")
    assert isinstance(result, dict)


def test_compute_body_metrics_keys():
    result = compute_body_metrics(80.0, 175.0, "female")
    for key in ("ibw", "lbm", "abw", "bmi"):
        assert key in result


def test_invalid_sex_raises():
    with pytest.raises(ValueError):
        adjust_for_obesity(
            drug_name="Drug",
            total_body_weight_kg=100.0,
            height_cm=170.0,
            sex="other",
            baseline_cl_L_per_h=5.0,
            baseline_vd_L=50.0,
            baseline_dose_mg=100.0,
        )


def test_invalid_drug_type_raises():
    with pytest.raises(ValueError):
        adjust_for_obesity(
            drug_name="Drug",
            total_body_weight_kg=100.0,
            height_cm=170.0,
            sex="male",
            baseline_cl_L_per_h=5.0,
            baseline_vd_L=50.0,
            baseline_dose_mg=100.0,
            drug_type="amphiphilic",
        )


def test_notes_nonempty():
    r = _result()
    assert len(r.notes) > 0
