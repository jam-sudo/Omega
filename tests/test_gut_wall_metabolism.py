"""Tests for core/gut_wall_metabolism.py (Phase 180)."""

import pytest

from omega_pbpk.core.gut_wall_metabolism import (
    bioavailability_with_gut_wall,
    calculate_gut_wall_extraction,
)

# ---------------------------------------------------------------------------
# calculate_gut_wall_extraction
# ---------------------------------------------------------------------------


def test_zero_clg_int_fg_one():
    """Zero intrinsic clearance → no gut extraction → Fg = 1."""
    eg, fg = calculate_gut_wall_extraction("Drug", clg_int_mL_per_min=0.0)
    assert eg == pytest.approx(0.0)
    assert fg == pytest.approx(1.0)


def test_high_clg_int_low_fg():
    """Very high CLg_int → Fg should be low (< 0.5)."""
    eg, fg = calculate_gut_wall_extraction(
        "HighExtract", clg_int_mL_per_min=1000.0, fu_gut=1.0, qg_mL_per_min=250.0
    )
    assert fg < 0.5
    assert eg + fg == pytest.approx(1.0)


def test_eg_fg_sum_to_one():
    eg, fg = calculate_gut_wall_extraction(
        "Test", clg_int_mL_per_min=100.0, fu_gut=0.3, qg_mL_per_min=250.0
    )
    assert eg + fg == pytest.approx(1.0)


def test_negative_clg_int_raises():
    with pytest.raises(ValueError, match="clg_int"):
        calculate_gut_wall_extraction("Drug", clg_int_mL_per_min=-10.0)


def test_invalid_fu_gut_raises():
    with pytest.raises(ValueError, match="fu_gut"):
        calculate_gut_wall_extraction("Drug", clg_int_mL_per_min=50.0, fu_gut=1.5)


def test_zero_qg_raises():
    with pytest.raises(ValueError, match="qg_mL_per_min"):
        calculate_gut_wall_extraction("Drug", clg_int_mL_per_min=50.0, qg_mL_per_min=0.0)


def test_fu_gut_zero_gives_fg_one():
    """No unbound drug in gut → effective CL = 0 → Fg = 1."""
    eg, fg = calculate_gut_wall_extraction("Drug", clg_int_mL_per_min=500.0, fu_gut=0.0)
    assert fg == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# bioavailability_with_gut_wall
# ---------------------------------------------------------------------------


def test_f_oral_equals_fa_fg_fh():
    result = bioavailability_with_gut_wall(
        drug_name="TestDrug",
        fa=0.8,
        cl_hepatic_L_per_h=30.0,
        vd_L=50.0,
        clg_int_mL_per_min=100.0,
        fu_gut=0.5,
        qg_mL_per_min=250.0,
    )
    expected = result.fa * result.fg * result.fh
    assert result.f_oral == pytest.approx(expected, rel=1e-6)


def test_midazolam_like_params():
    """fa=0.9, CLg_int=300 mL/min → Fg should be roughly 0.37–0.60."""
    eg, fg = calculate_gut_wall_extraction(
        "Midazolam",
        clg_int_mL_per_min=300.0,
        fu_gut=0.5,
        qg_mL_per_min=250.0,
    )
    assert 0.3 <= fg <= 0.65, f"Unexpected Fg={fg:.3f} for midazolam-like params"


def test_zero_gut_cl_full_gut_availability():
    result = bioavailability_with_gut_wall(
        drug_name="NoGutMet",
        fa=1.0,
        cl_hepatic_L_per_h=10.0,
        vd_L=20.0,
        clg_int_mL_per_min=0.0,
    )
    assert result.fg == pytest.approx(1.0)
    assert result.eg == pytest.approx(0.0)


def test_invalid_fa_negative_raises():
    with pytest.raises(ValueError, match="fa"):
        bioavailability_with_gut_wall(
            drug_name="Bad", fa=-0.1, cl_hepatic_L_per_h=10.0, vd_L=20.0, clg_int_mL_per_min=50.0
        )


def test_invalid_fa_greater_than_one_raises():
    with pytest.raises(ValueError, match="fa"):
        bioavailability_with_gut_wall(
            drug_name="Bad", fa=1.1, cl_hepatic_L_per_h=10.0, vd_L=20.0, clg_int_mL_per_min=50.0
        )


def test_invalid_clg_int_raises():
    with pytest.raises(ValueError, match="clg_int"):
        bioavailability_with_gut_wall(
            drug_name="Bad", fa=0.8, cl_hepatic_L_per_h=10.0, vd_L=20.0, clg_int_mL_per_min=-5.0
        )


def test_invalid_cl_hepatic_raises():
    with pytest.raises(ValueError, match="cl_hepatic"):
        bioavailability_with_gut_wall(
            drug_name="Bad", fa=0.8, cl_hepatic_L_per_h=-1.0, vd_L=20.0, clg_int_mL_per_min=50.0
        )


def test_result_frozen():
    result = bioavailability_with_gut_wall(
        drug_name="Test", fa=0.8, cl_hepatic_L_per_h=20.0, vd_L=30.0, clg_int_mL_per_min=100.0
    )
    with pytest.raises((AttributeError, TypeError)):
        result.fg = 0.99  # type: ignore[misc]  # frozen dataclass


def test_hepatic_extraction_calculation():
    """Validate Eh = CLh_mL / (Qh_mL + CLh_mL)."""
    # CL = 60 L/h = 1000 mL/min; Qh = 80 L/h = 1333.3 mL/min
    result = bioavailability_with_gut_wall(
        drug_name="HighHepatic",
        fa=1.0,
        cl_hepatic_L_per_h=60.0,
        vd_L=100.0,
        clg_int_mL_per_min=0.0,
        qh_L_per_h=80.0,
    )
    qh_mL = 80.0 * 1000.0 / 60.0
    clh_mL = 60.0 * 1000.0 / 60.0
    expected_eh = clh_mL / (qh_mL + clh_mL)
    assert result.eh == pytest.approx(expected_eh, rel=1e-6)
    assert result.fh == pytest.approx(1.0 - expected_eh, rel=1e-6)


def test_f_oral_between_zero_and_one():
    result = bioavailability_with_gut_wall(
        drug_name="Normal",
        fa=0.85,
        cl_hepatic_L_per_h=25.0,
        vd_L=40.0,
        clg_int_mL_per_min=150.0,
    )
    assert 0.0 <= result.f_oral <= 1.0


def test_notes_populated():
    result = bioavailability_with_gut_wall(
        drug_name="NoteTest", fa=0.7, cl_hepatic_L_per_h=10.0, vd_L=20.0, clg_int_mL_per_min=50.0
    )
    assert len(result.notes) > 0
    assert "Fg" in result.notes or "fg" in result.notes.lower() or "F_oral" in result.notes
