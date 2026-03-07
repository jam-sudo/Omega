"""Tests for clinical/transporter_ivive.py — Phase 348."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.transporter_ivive import (
    TransporterIVIVEResult,
    predict_transporter_ddi,
    screen_transporter_inhibitors,
    summarize_transporter_risk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = dict(
    perpetrator_name="Perpetrator",
    victim_drug="Victim",
    transporter="P-gp",
    ic50_vitro_uM=10.0,
    dose_mg_perpetrator=100.0,
    mw_perpetrator=350.0,
)


def _run(**overrides) -> TransporterIVIVEResult:
    params = {**_BASE, **overrides}
    return predict_transporter_ddi(**params)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    result = _run()
    assert isinstance(result, TransporterIVIVEResult)


def test_frozen_dataclass():
    result = _run()
    with pytest.raises((AttributeError, TypeError)):
        result.perpetrator_name = "other"  # type: ignore[misc]


def test_fields_accessible():
    result = _run()
    _ = result.r_value
    _ = result.r2_value
    _ = result.predicted_auc_ratio
    _ = result.ddi_category
    _ = result.clinical_significance
    _ = result.requires_clinical_study


# ---------------------------------------------------------------------------
# Weak inhibitor (high IC50)
# ---------------------------------------------------------------------------


def test_high_ic50_no_ddi():
    result = _run(ic50_vitro_uM=1e6, dose_mg_perpetrator=100.0)
    assert result.ddi_category == "no_ddi"


def test_high_ic50_r_value_near_one():
    result = _run(ic50_vitro_uM=1e6, dose_mg_perpetrator=100.0)
    assert result.r_value == pytest.approx(1.0, abs=0.01)


def test_high_ic50_no_clinical_study():
    result = _run(ic50_vitro_uM=1e6, dose_mg_perpetrator=100.0)
    assert not result.requires_clinical_study


# ---------------------------------------------------------------------------
# Strong inhibitor (low IC50)
# ---------------------------------------------------------------------------


def test_low_ic50_strong_ddi():
    result = _run(ic50_vitro_uM=0.001, dose_mg_perpetrator=1000.0)
    assert result.ddi_category in ("strong", "moderate")


def test_low_ic50_high_r_value():
    result = _run(ic50_vitro_uM=0.001, dose_mg_perpetrator=1000.0)
    assert result.r_value > 2.0


def test_low_ic50_requires_clinical_study():
    result = _run(ic50_vitro_uM=0.001, dose_mg_perpetrator=1000.0)
    assert result.requires_clinical_study


# ---------------------------------------------------------------------------
# Transporter-specific calculations
# ---------------------------------------------------------------------------


def test_pgp_uses_gut_concentration():
    """P-gp: c_inlet should be large (gut lumen concentration)."""
    result = _run(transporter="P-gp", dose_mg_perpetrator=500.0, mw_perpetrator=350.0,
                  ic50_vitro_uM=10.0)
    # C_gut_uM = (500/0.25 L) * 1000 / 350 >> systemic
    expected_c_gut = (500.0 / 0.25) * 1000.0 / 350.0
    assert result.c_inlet_uM == pytest.approx(expected_c_gut, rel=1e-3)


def test_bcrp_uses_gut_concentration():
    """BCRP also intestinal: same gut-lumen formula."""
    pgp = _run(transporter="P-gp")
    bcrp = _run(transporter="BCRP")
    assert bcrp.c_inlet_uM == pytest.approx(pgp.c_inlet_uM, rel=1e-3)


def test_oatp1b1_uses_portal_concentration():
    """OATP1B1: hepatic portal; C_inlet < gut C."""
    gut = _run(transporter="P-gp", dose_mg_perpetrator=100.0)
    hepatic = _run(transporter="OATP1B1", dose_mg_perpetrator=100.0)
    assert hepatic.c_inlet_uM < gut.c_inlet_uM


def test_oatp1b3_uses_portal_concentration():
    result = _run(transporter="OATP1B3", dose_mg_perpetrator=100.0,
                  mw_perpetrator=350.0, fu_plasma=0.5, vd_L=50.0)
    # C_portal = (dose/vd * fu * 1.5) * 1000 / mw
    expected = (100.0 / 50.0) * 0.5 * 1.5 * 1000.0 / 350.0
    assert result.c_inlet_uM == pytest.approx(expected, rel=1e-3)


def test_oat1_uses_systemic_concentration():
    result = _run(transporter="OAT1", dose_mg_perpetrator=100.0,
                  mw_perpetrator=350.0, fu_plasma=0.5, vd_L=50.0)
    # C_systemic = dose/vd * fu * 1000/mw
    expected = (100.0 / 50.0) * 0.5 * 1000.0 / 350.0
    assert result.c_inlet_uM == pytest.approx(expected, rel=1e-3)


def test_oct2_uses_systemic_concentration():
    """OCT2 renal: same systemic formula as OAT."""
    oat = _run(transporter="OAT3")
    oct2 = _run(transporter="OCT2")
    assert oct2.c_inlet_uM == pytest.approx(oat.c_inlet_uM, rel=1e-3)


# ---------------------------------------------------------------------------
# DDI category boundaries
# ---------------------------------------------------------------------------


def test_no_ddi_category_boundary():
    # R just below 1.1
    result = _run(transporter="OAT1", ic50_vitro_uM=1e5, dose_mg_perpetrator=1.0,
                  mw_perpetrator=350.0, fu_plasma=0.5, vd_L=50.0)
    assert result.ddi_category == "no_ddi"


def test_predicted_auc_ratio_capped_at_five():
    result = _run(ic50_vitro_uM=0.00001, dose_mg_perpetrator=5000.0)
    assert result.predicted_auc_ratio <= 5.0


def test_requires_clinical_study_threshold():
    """When AUC ratio > 1.25, clinical study is needed."""
    # Force AUC ratio slightly above 1.25 via R > 1.25
    result = _run(transporter="OAT1", ic50_vitro_uM=0.01, dose_mg_perpetrator=5000.0,
                  mw_perpetrator=350.0, fu_plasma=1.0, vd_L=10.0)
    assert result.requires_clinical_study


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_ic50_raises():
    with pytest.raises(ValueError, match="ic50"):
        _run(ic50_vitro_uM=0.0)


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose"):
        _run(dose_mg_perpetrator=0.0)


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        _run(mw_perpetrator=0.0)


def test_invalid_fu_plasma_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        _run(fu_plasma=0.0)


def test_invalid_transporter_raises():
    with pytest.raises(ValueError, match="transporter"):
        _run(transporter="INVALID")


# ---------------------------------------------------------------------------
# screen_transporter_inhibitors
# ---------------------------------------------------------------------------


def test_screen_returns_correct_count():
    compounds = [
        {"name": "A", "dose_mg": 100.0, "mw": 350.0, "ic50_uM": 1.0},
        {"name": "B", "dose_mg": 200.0, "mw": 400.0, "ic50_uM": 50.0},
        {"name": "C", "dose_mg": 50.0, "mw": 250.0, "ic50_uM": 0.5},
    ]
    results = screen_transporter_inhibitors(compounds, "Digoxin", "P-gp")
    assert len(results) == 3


def test_screen_sorted_by_r_value_descending():
    compounds = [
        {"name": "weak", "dose_mg": 10.0, "mw": 300.0, "ic50_uM": 1000.0},
        {"name": "strong", "dose_mg": 1000.0, "mw": 300.0, "ic50_uM": 0.1},
    ]
    results = screen_transporter_inhibitors(compounds, "Digoxin", "P-gp")
    assert results[0].r_value >= results[1].r_value


def test_screen_returns_ivive_result_type():
    compounds = [{"name": "X", "dose_mg": 100.0, "mw": 350.0, "ic50_uM": 5.0}]
    results = screen_transporter_inhibitors(compounds, "Digoxin", "OATP1B1")
    assert all(isinstance(r, TransporterIVIVEResult) for r in results)


# ---------------------------------------------------------------------------
# summarize_transporter_risk
# ---------------------------------------------------------------------------


def test_summarize_returns_correct_keys():
    results = [_run(transporter="P-gp"), _run(transporter="OATP1B1")]
    summary = summarize_transporter_risk(results)
    assert "high_risk_count" in summary
    assert "transporter_summary" in summary


def test_summarize_high_risk_count():
    strong = _run(ic50_vitro_uM=0.001, dose_mg_perpetrator=1000.0)
    weak = _run(ic50_vitro_uM=1e6, dose_mg_perpetrator=10.0)
    summary = summarize_transporter_risk([strong, weak])
    # strong should be moderate/strong; weak should be no_ddi
    assert summary["high_risk_count"] >= 1


def test_summarize_transporter_max_r():
    r1 = _run(transporter="P-gp", ic50_vitro_uM=0.01, dose_mg_perpetrator=500.0)
    r2 = _run(transporter="P-gp", ic50_vitro_uM=100.0, dose_mg_perpetrator=100.0)
    summary = summarize_transporter_risk([r1, r2])
    expected_max = max(r1.r_value, r2.r_value)
    assert summary["transporter_summary"]["P-gp"] == pytest.approx(expected_max, rel=1e-6)


def test_summarize_empty_list():
    summary = summarize_transporter_risk([])
    assert summary["high_risk_count"] == 0
    assert summary["transporter_summary"] == {}
