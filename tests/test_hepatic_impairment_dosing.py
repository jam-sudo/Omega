"""Tests for Phase 207: hepatic_impairment_dosing module."""

import pytest

from omega_pbpk.clinical.hepatic_impairment_dosing import (
    ChildPughScore,
    HepaticImpairmentResult,
    adjust_dose_hepatic,
    child_pugh_score,
)


# ---------------------------------------------------------------------------
# child_pugh_score tests
# ---------------------------------------------------------------------------


def test_normal_labs_grade_a_score_5():
    """Normal labs → Grade A, minimum score 5."""
    cp = child_pugh_score(
        bilirubin_mg_dL=1.0,
        albumin_g_dL=4.0,
        pt_seconds_prolonged=2.0,
        ascites="none",
        encephalopathy="none",
    )
    assert cp.grade == "A"
    assert cp.score == 5


def test_mild_impairment_grade_a():
    """Mild lab abnormalities → Grade A."""
    cp = child_pugh_score(
        bilirubin_mg_dL=1.8,
        albumin_g_dL=3.6,
        pt_seconds_prolonged=3.0,
        ascites="none",
        encephalopathy="none",
    )
    assert cp.grade == "A"
    assert cp.score == 5


def test_moderate_impairment_grade_b():
    """Moderate impairment → Grade B."""
    cp = child_pugh_score(
        bilirubin_mg_dL=2.5,
        albumin_g_dL=3.0,
        pt_seconds_prolonged=5.0,
        ascites="mild",
        encephalopathy="none",
    )
    assert cp.grade == "B"
    assert 7 <= cp.score <= 9


def test_severe_labs_grade_c_score_15():
    """Severe labs → Grade C, score 15."""
    cp = child_pugh_score(
        bilirubin_mg_dL=4.0,
        albumin_g_dL=2.5,
        pt_seconds_prolonged=7.0,
        ascites="moderate_severe",
        encephalopathy="grade_3_4",
    )
    assert cp.grade == "C"
    assert cp.score == 15


def test_score_boundary_grade_a_6():
    """Score 6 → Grade A."""
    cp = child_pugh_score(
        bilirubin_mg_dL=1.0,
        albumin_g_dL=3.6,
        pt_seconds_prolonged=2.0,
        ascites="mild",
        encephalopathy="none",
    )
    assert cp.grade == "A"
    assert cp.score == 6


def test_score_boundary_grade_b_7():
    """Score 7 → Grade B."""
    cp = child_pugh_score(
        bilirubin_mg_dL=2.5,
        albumin_g_dL=3.6,
        pt_seconds_prolonged=2.0,
        ascites="mild",
        encephalopathy="none",
    )
    assert cp.grade == "B"
    assert cp.score == 7


def test_score_boundary_grade_c_10():
    """Score 10 → Grade C."""
    cp = child_pugh_score(
        bilirubin_mg_dL=4.0,
        albumin_g_dL=3.6,
        pt_seconds_prolonged=2.0,
        ascites="moderate_severe",
        encephalopathy="grade_1_2",
    )
    assert cp.grade == "C"
    assert cp.score == 10


def test_invalid_ascites_raises():
    """Invalid ascites value raises ValueError."""
    with pytest.raises(ValueError, match="ascites"):
        child_pugh_score(
            bilirubin_mg_dL=1.0,
            albumin_g_dL=4.0,
            pt_seconds_prolonged=2.0,
            ascites="severe",
            encephalopathy="none",
        )


def test_invalid_encephalopathy_raises():
    """Invalid encephalopathy value raises ValueError."""
    with pytest.raises(ValueError, match="encephalopathy"):
        child_pugh_score(
            bilirubin_mg_dL=1.0,
            albumin_g_dL=4.0,
            pt_seconds_prolonged=2.0,
            ascites="none",
            encephalopathy="grade_2",
        )


def test_negative_bilirubin_raises():
    """Bilirubin <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="bilirubin"):
        child_pugh_score(
            bilirubin_mg_dL=-1.0,
            albumin_g_dL=4.0,
            pt_seconds_prolonged=2.0,
        )


# ---------------------------------------------------------------------------
# adjust_dose_hepatic tests
# ---------------------------------------------------------------------------


def _make_grade_a_cp() -> ChildPughScore:
    return child_pugh_score(1.0, 4.0, 2.0, "none", "none")


def _make_grade_b_cp() -> ChildPughScore:
    return child_pugh_score(2.5, 3.0, 5.0, "mild", "none")


def _make_grade_c_cp() -> ChildPughScore:
    return child_pugh_score(4.0, 2.5, 7.0, "moderate_severe", "grade_3_4")


def test_fe_hepatic_zero_cl_unchanged():
    """fe_hepatic=0: CL unchanged regardless of grade."""
    for cp in [_make_grade_a_cp(), _make_grade_b_cp(), _make_grade_c_cp()]:
        result = adjust_dose_hepatic(
            "TestDrug",
            cp,
            fe_hepatic=0.0,
            cl_normal_L_per_h=10.0,
            vd_L=50.0,
            dose_normal_mg=100.0,
        )
        assert abs(result.cl_adjusted_L_per_h - 10.0) < 1e-9


def test_grade_c_high_fe_hepatic_contraindicated():
    """Grade C + fe_hepatic=0.8: contraindicated=True."""
    cp = _make_grade_c_cp()
    result = adjust_dose_hepatic(
        "TestDrug",
        cp,
        fe_hepatic=0.8,
        cl_normal_L_per_h=10.0,
        vd_L=50.0,
        dose_normal_mg=100.0,
    )
    assert result.contraindicated is True


def test_grade_a_high_fe_hepatic_not_contraindicated():
    """Grade A + fe_hepatic=0.8: contraindicated=False."""
    cp = _make_grade_a_cp()
    result = adjust_dose_hepatic(
        "TestDrug",
        cp,
        fe_hepatic=0.8,
        cl_normal_L_per_h=10.0,
        vd_L=50.0,
        dose_normal_mg=100.0,
    )
    assert result.contraindicated is False


def test_grade_c_low_fe_hepatic_not_contraindicated():
    """Grade C + fe_hepatic=0.4: contraindicated=False."""
    cp = _make_grade_c_cp()
    result = adjust_dose_hepatic(
        "TestDrug",
        cp,
        fe_hepatic=0.4,
        cl_normal_L_per_h=10.0,
        vd_L=50.0,
        dose_normal_mg=100.0,
    )
    assert result.contraindicated is False


def test_dose_decreases_with_worsening_grade():
    """Dose decreases monotonically from Grade A → B → C for hepatic drug."""
    cp_a = _make_grade_a_cp()
    cp_b = _make_grade_b_cp()
    cp_c = _make_grade_c_cp()
    kwargs = dict(fe_hepatic=0.9, cl_normal_L_per_h=10.0, vd_L=50.0, dose_normal_mg=100.0)
    res_a = adjust_dose_hepatic("Drug", cp_a, **kwargs)
    res_b = adjust_dose_hepatic("Drug", cp_b, **kwargs)
    res_c = adjust_dose_hepatic("Drug", cp_c, **kwargs)
    assert res_a.dose_adjusted_mg > res_b.dose_adjusted_mg > res_c.dose_adjusted_mg


def test_half_life_increases_with_worsening_grade():
    """Adjusted half-life increases from Grade A → B → C."""
    cp_a = _make_grade_a_cp()
    cp_b = _make_grade_b_cp()
    cp_c = _make_grade_c_cp()
    kwargs = dict(fe_hepatic=0.9, cl_normal_L_per_h=10.0, vd_L=50.0, dose_normal_mg=100.0)
    res_a = adjust_dose_hepatic("Drug", cp_a, **kwargs)
    res_b = adjust_dose_hepatic("Drug", cp_b, **kwargs)
    res_c = adjust_dose_hepatic("Drug", cp_c, **kwargs)
    assert res_a.t_half_adjusted_h < res_b.t_half_adjusted_h < res_c.t_half_adjusted_h


def test_dose_reduction_pct_grade_b_approximately_50():
    """Grade B with fe_hepatic=1.0 → ~50% dose reduction."""
    cp = _make_grade_b_cp()
    result = adjust_dose_hepatic(
        "Drug",
        cp,
        fe_hepatic=1.0,
        cl_normal_L_per_h=10.0,
        vd_L=50.0,
        dose_normal_mg=100.0,
    )
    # CL factor = 1 - 1.0*(1 - 0.5) = 0.5 → 50% reduction
    assert abs(result.dose_reduction_pct - 50.0) < 1e-6


def test_fup_adjustment_high_protein_binding():
    """Highly bound drug: fup_adjustment_factor > 1 with low albumin."""
    cp = _make_grade_c_cp()  # albumin = 2.5 g/dL
    result = adjust_dose_hepatic(
        "Drug",
        cp,
        fe_hepatic=0.5,
        cl_normal_L_per_h=10.0,
        vd_L=50.0,
        dose_normal_mg=100.0,
        protein_binding=0.95,
    )
    assert result.fup_adjustment_factor > 1.0


def test_fup_adjustment_low_protein_binding():
    """Poorly bound drug: fup_adjustment_factor = 1.0."""
    cp = _make_grade_c_cp()
    result = adjust_dose_hepatic(
        "Drug",
        cp,
        fe_hepatic=0.5,
        cl_normal_L_per_h=10.0,
        vd_L=50.0,
        dose_normal_mg=100.0,
        protein_binding=0.5,
    )
    assert result.fup_adjustment_factor == 1.0


def test_returns_hepatic_impairment_result_type():
    """Return type is HepaticImpairmentResult."""
    cp = _make_grade_a_cp()
    result = adjust_dose_hepatic(
        "Warfarin",
        cp,
        fe_hepatic=0.9,
        cl_normal_L_per_h=3.0,
        vd_L=10.0,
        dose_normal_mg=5.0,
    )
    assert isinstance(result, HepaticImpairmentResult)
    assert result.drug_name == "Warfarin"
    assert result.child_pugh_grade == "A"
