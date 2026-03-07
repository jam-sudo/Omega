"""Tests for Phase 779 — pk_summary_reporter.py."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.pk_summary_reporter import (
    PKSummaryReport,
    batch_generate_pk_reports,
    generate_pk_summary,
)

# ---------------------------------------------------------------------------
# Helper baseline parameters
# ---------------------------------------------------------------------------

BASELINE = dict(
    drug_name="DrugA",
    route="oral",
    dose_mg=100.0,
    cmax_mg_L=2.0,
    tmax_h=1.5,
    auc_mg_h_per_L=20.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    f_oral=0.8,
    tau_h=12.0,
    mec_mg_L=0.5,
    mtc_mg_L=5.0,
    time_in_therapeutic_pct=75.0,
)


# ---------------------------------------------------------------------------
# 1. Return type is PKSummaryReport
# ---------------------------------------------------------------------------
def test_return_type():
    report = generate_pk_summary(**BASELINE)
    assert isinstance(report, PKSummaryReport)


# ---------------------------------------------------------------------------
# 2. PKSummaryReport is frozen (immutable)
# ---------------------------------------------------------------------------
def test_frozen_dataclass():
    report = generate_pk_summary(**BASELINE)
    with pytest.raises((AttributeError, TypeError)):
        report.drug_name = "X"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. t_half_h = 0.693 * vd_L / cl_L_per_h
# ---------------------------------------------------------------------------
def test_t_half_computation():
    report = generate_pk_summary(**BASELINE)
    expected = 0.693 * 50.0 / 5.0
    assert abs(report.t_half_h - expected) < 1e-6


# ---------------------------------------------------------------------------
# 4. accumulation_ratio >= 1.0 always
# ---------------------------------------------------------------------------
def test_accumulation_ratio_lower_bound():
    report = generate_pk_summary(**BASELINE)
    assert report.accumulation_ratio >= 1.0


# ---------------------------------------------------------------------------
# 5. accumulation_ratio <= 10.0 always
# ---------------------------------------------------------------------------
def test_accumulation_ratio_upper_bound():
    # Very short tau relative to t_half => ratio would exceed 10 without clamping
    params = dict(BASELINE)
    params["tau_h"] = 0.1  # extremely short interval
    report = generate_pk_summary(**params)
    assert report.accumulation_ratio <= 10.0


# ---------------------------------------------------------------------------
# 6. steady_state_flag True when tau < t_half
# ---------------------------------------------------------------------------
def test_steady_state_flag_true():
    # tau=12h, t_half = 0.693*50/5 = 6.93h  -> tau > t_half => False
    # Override: make tau < t_half
    params = dict(BASELINE)
    params["tau_h"] = 5.0  # < t_half ~6.93
    report = generate_pk_summary(**params)
    assert report.steady_state_flag is True


# ---------------------------------------------------------------------------
# 7. steady_state_flag False when tau >= t_half
# ---------------------------------------------------------------------------
def test_steady_state_flag_false():
    # Default: tau=12 > t_half~6.93
    report = generate_pk_summary(**BASELINE)
    assert report.steady_state_flag is False


# ---------------------------------------------------------------------------
# 8. subtherapeutic when cmax < mec
# ---------------------------------------------------------------------------
def test_subtherapeutic_flag():
    params = dict(BASELINE)
    params["cmax_mg_L"] = 0.2  # below mec=0.5
    report = generate_pk_summary(**params)
    assert report.subtherapeutic is True
    assert report.supratherapeutic is False


# ---------------------------------------------------------------------------
# 9. supratherapeutic when cmax > mtc
# ---------------------------------------------------------------------------
def test_supratherapeutic_flag():
    params = dict(BASELINE)
    params["cmax_mg_L"] = 6.0  # above mtc=5.0
    report = generate_pk_summary(**params)
    assert report.supratherapeutic is True
    assert report.subtherapeutic is False


# ---------------------------------------------------------------------------
# 10. Both flags False when within window
# ---------------------------------------------------------------------------
def test_within_window_flags():
    report = generate_pk_summary(**BASELINE)  # cmax=2.0, window 0.5-5.0
    assert report.subtherapeutic is False
    assert report.supratherapeutic is False


# ---------------------------------------------------------------------------
# 11. Scalar field types
# ---------------------------------------------------------------------------
def test_field_types():
    report = generate_pk_summary(**BASELINE)
    assert isinstance(report.drug_name, str)
    assert isinstance(report.route, str)
    assert isinstance(report.dose_mg, float)
    assert isinstance(report.cmax_mg_L, float)
    assert isinstance(report.tmax_h, float)
    assert isinstance(report.auc_mg_h_per_L, float)
    assert isinstance(report.t_half_h, float)
    assert isinstance(report.vd_L, float)
    assert isinstance(report.cl_L_per_h, float)
    assert isinstance(report.f_oral, float)
    assert isinstance(report.accumulation_ratio, float)
    assert isinstance(report.steady_state_flag, bool)
    assert isinstance(report.subtherapeutic, bool)
    assert isinstance(report.supratherapeutic, bool)
    assert isinstance(report.mec_mg_L, float)
    assert isinstance(report.mtc_mg_L, float)
    assert isinstance(report.time_in_therapeutic_pct, float)
    assert isinstance(report.summary_text, str)
    assert isinstance(report.recommendations, str)


# ---------------------------------------------------------------------------
# 12. summary_text is non-empty string
# ---------------------------------------------------------------------------
def test_summary_text_non_empty():
    report = generate_pk_summary(**BASELINE)
    assert len(report.summary_text) > 0


# ---------------------------------------------------------------------------
# 13. recommendations is non-empty string
# ---------------------------------------------------------------------------
def test_recommendations_non_empty():
    report = generate_pk_summary(**BASELINE)
    assert len(report.recommendations) > 0


# ---------------------------------------------------------------------------
# 14. ValueError for non-positive dose_mg
# ---------------------------------------------------------------------------
def test_invalid_dose_mg():
    params = dict(BASELINE)
    params["dose_mg"] = 0.0
    with pytest.raises(ValueError):
        generate_pk_summary(**params)


# ---------------------------------------------------------------------------
# 15. ValueError for non-positive cl_L_per_h
# ---------------------------------------------------------------------------
def test_invalid_cl():
    params = dict(BASELINE)
    params["cl_L_per_h"] = -1.0
    with pytest.raises(ValueError):
        generate_pk_summary(**params)


# ---------------------------------------------------------------------------
# 16. ValueError for non-positive vd_L
# ---------------------------------------------------------------------------
def test_invalid_vd():
    params = dict(BASELINE)
    params["vd_L"] = 0.0
    with pytest.raises(ValueError):
        generate_pk_summary(**params)


# ---------------------------------------------------------------------------
# 17. ValueError for f_oral out of [0, 1]
# ---------------------------------------------------------------------------
def test_invalid_f_oral():
    params = dict(BASELINE)
    params["f_oral"] = 1.5
    with pytest.raises(ValueError):
        generate_pk_summary(**params)


# ---------------------------------------------------------------------------
# 18. ValueError when mtc <= mec
# ---------------------------------------------------------------------------
def test_invalid_therapeutic_window():
    params = dict(BASELINE)
    params["mec_mg_L"] = 5.0
    params["mtc_mg_L"] = 1.0
    with pytest.raises(ValueError):
        generate_pk_summary(**params)


# ---------------------------------------------------------------------------
# 19. batch_generate_pk_reports returns list of PKSummaryReport
# ---------------------------------------------------------------------------
def test_batch_generate_returns_list():
    params_list = [BASELINE, BASELINE]
    results = batch_generate_pk_reports(params_list)
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, PKSummaryReport)


# ---------------------------------------------------------------------------
# 20. Accumulated params passthrough correctly
# ---------------------------------------------------------------------------
def test_field_passthrough():
    report = generate_pk_summary(**BASELINE)
    assert report.drug_name == "DrugA"
    assert report.route == "oral"
    assert abs(report.dose_mg - 100.0) < 1e-9
    assert abs(report.cmax_mg_L - 2.0) < 1e-9
    assert abs(report.f_oral - 0.8) < 1e-9
    assert abs(report.mec_mg_L - 0.5) < 1e-9
    assert abs(report.mtc_mg_L - 5.0) < 1e-9
    assert abs(report.time_in_therapeutic_pct - 75.0) < 1e-9
