"""Tests for hepatic_dosing module (Phase 411)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.hepatic_dosing import (
    adjust_dose_hepatic,
    child_pugh_score,
)

# ---------------------------------------------------------------------------
# child_pugh_score — happy-path tests
# ---------------------------------------------------------------------------


def test_child_pugh_class_a_minimum():
    """Score of 5 → Class A."""
    result = child_pugh_score(
        bilirubin_mg_dL=1.5,
        albumin_g_dL=4.0,
        pt_seconds_prolonged=2.0,
        ascites="none",
        encephalopathy="none",
    )
    assert result.total_score == 5
    assert result.child_pugh_class == "A"


def test_child_pugh_class_a_maximum():
    """Score of 6 → Class A.

    bilirubin 1.5 (<2 → 1), albumin 3.6 (>3.5 → 1), PT 2s (<4 → 1),
    ascites 'mild' (2), encephalopathy 'none' (1) → total 6
    """
    result = child_pugh_score(
        bilirubin_mg_dL=1.5,
        albumin_g_dL=3.6,
        pt_seconds_prolonged=2.0,
        ascites="mild",
        encephalopathy="none",
    )
    assert result.total_score == 6
    assert result.child_pugh_class == "A"


def test_child_pugh_class_b_minimum():
    """Score of 7 → Class B.

    bilirubin 1.5 (1), albumin 3.6 (1), PT 2s (1),
    ascites 'severe' (3), encephalopathy 'none' (1) → total 7
    """
    result = child_pugh_score(
        bilirubin_mg_dL=1.5,
        albumin_g_dL=3.6,
        pt_seconds_prolonged=2.0,
        ascites="severe",
        encephalopathy="none",
    )
    assert result.total_score == 7
    assert result.child_pugh_class == "B"


def test_child_pugh_class_b_maximum():
    """Score of 9 → Class B.

    bilirubin 1.5 (1), albumin 3.6 (1), PT 2s (1),
    ascites 'severe' (3), encephalopathy 'grade_1_2' (2) → total 8.
    To reach 9: bilirubin 2.5 (2), albumin 3.6 (1), PT 2s (1),
    ascites 'severe' (3), encephalopathy 'none' (1) → 8.
    Use: bilirubin 2.5 (2), albumin 3.6 (1), PT 5s (2),
    ascites 'mild' (2), encephalopathy 'none' (1) → 8.
    Correct: bilirubin 2.5 (2), albumin 3.6 (1), PT 2s (1),
    ascites 'mild' (2), encephalopathy 'grade_1_2' (2) → 8.
    Actual 9: bilirubin 2.5 (2), albumin 3.6 (1), PT 5s (2),
    ascites 'mild' (2), encephalopathy 'grade_1_2' (2) → 9.
    """
    result = child_pugh_score(
        bilirubin_mg_dL=2.5,
        albumin_g_dL=3.6,
        pt_seconds_prolonged=5.0,
        ascites="mild",
        encephalopathy="grade_1_2",
    )
    assert result.total_score == 9
    assert result.child_pugh_class == "B"


def test_child_pugh_class_c():
    """Score of 15 → Class C."""
    result = child_pugh_score(
        bilirubin_mg_dL=4.0,
        albumin_g_dL=2.5,
        pt_seconds_prolonged=7.0,
        ascites="severe",
        encephalopathy="grade_3_4",
    )
    assert result.total_score == 15
    assert result.child_pugh_class == "C"


def test_child_pugh_bilirubin_boundary_low():
    """Bilirubin exactly 2.0 → pts 2 (2-3 bracket)."""
    result = child_pugh_score(2.0, 4.0, 2.0, "none", "none")
    assert result.bilirubin_pts == 2


def test_child_pugh_bilirubin_boundary_high():
    """Bilirubin exactly 3.0 → pts 2 (2-3 bracket inclusive)."""
    result = child_pugh_score(3.0, 4.0, 2.0, "none", "none")
    assert result.bilirubin_pts == 2


def test_child_pugh_bilirubin_above_3():
    """Bilirubin 3.1 → pts 3."""
    result = child_pugh_score(3.1, 4.0, 2.0, "none", "none")
    assert result.bilirubin_pts == 3


def test_child_pugh_albumin_boundary():
    """Albumin exactly 3.5 → pts 1 (>3.5)."""
    result = child_pugh_score(1.5, 3.5, 2.0, "none", "none")
    # 3.5 is NOT > 3.5, falls into 2.8-3.5 → pts 2
    assert result.albumin_pts == 2


def test_child_pugh_albumin_above_35():
    """Albumin 3.6 → pts 1."""
    result = child_pugh_score(1.5, 3.6, 2.0, "none", "none")
    assert result.albumin_pts == 1


def test_child_pugh_albumin_below_28():
    """Albumin 2.7 → pts 3."""
    result = child_pugh_score(1.5, 2.7, 2.0, "none", "none")
    assert result.albumin_pts == 3


def test_child_pugh_pt_boundary_4s():
    """PT 4 seconds → pts 2 (4-6 bracket)."""
    result = child_pugh_score(1.5, 4.0, 4.0, "none", "none")
    assert result.pt_pts == 2


def test_child_pugh_pt_above_6s():
    """PT > 6 seconds → pts 3."""
    result = child_pugh_score(1.5, 4.0, 6.5, "none", "none")
    assert result.pt_pts == 3


def test_child_pugh_ascites_values():
    """All ascites values map to correct points."""
    for ascites, expected in (("none", 1), ("mild", 2), ("severe", 3)):
        result = child_pugh_score(1.5, 4.0, 2.0, ascites, "none")
        assert result.ascites_pts == expected, f"ascites={ascites}"


def test_child_pugh_encephalopathy_values():
    """All encephalopathy values map to correct points."""
    for enc, expected in (("none", 1), ("grade_1_2", 2), ("grade_3_4", 3)):
        result = child_pugh_score(1.5, 4.0, 2.0, "none", enc)
        assert result.encephalopathy_pts == expected, f"enc={enc}"


def test_child_pugh_result_is_frozen():
    """ChildPughResult should be immutable."""
    result = child_pugh_score(1.5, 4.0, 2.0, "none", "none")
    with pytest.raises(AttributeError):
        result.total_score = 99  # type: ignore[misc]


def test_child_pugh_notes_contain_class():
    """Notes string contains the class label."""
    result = child_pugh_score(4.0, 2.5, 7.0, "severe", "grade_3_4")
    assert "C" in result.notes or "severe" in result.notes.lower()


# ---------------------------------------------------------------------------
# child_pugh_score — validation errors
# ---------------------------------------------------------------------------


def test_child_pugh_invalid_bilirubin():
    with pytest.raises(ValueError, match="bilirubin"):
        child_pugh_score(-1.0, 4.0, 2.0, "none", "none")


def test_child_pugh_invalid_albumin():
    with pytest.raises(ValueError, match="albumin"):
        child_pugh_score(1.5, 0.0, 2.0, "none", "none")


def test_child_pugh_invalid_pt():
    with pytest.raises(ValueError, match="pt_seconds"):
        child_pugh_score(1.5, 4.0, -1.0, "none", "none")


def test_child_pugh_invalid_ascites():
    with pytest.raises(ValueError, match="ascites"):
        child_pugh_score(1.5, 4.0, 2.0, "moderate", "none")


def test_child_pugh_invalid_encephalopathy():
    with pytest.raises(ValueError, match="encephalopathy"):
        child_pugh_score(1.5, 4.0, 2.0, "none", "grade_2")


# ---------------------------------------------------------------------------
# adjust_dose_hepatic — happy-path tests
# ---------------------------------------------------------------------------


def test_adjust_dose_class_a_no_reduction():
    """Class A: dose factor = 1.0, no dose change."""
    result = adjust_dose_hepatic("Drug_A", 100.0, "A", 0.8, 10.0, 50.0)
    assert result.dose_factor == 1.0
    assert math.isclose(result.adjusted_dose_mg, 100.0)


def test_adjust_dose_class_b_25pct_reduction():
    """Class B: dose factor = 0.75."""
    result = adjust_dose_hepatic("Drug_B", 200.0, "B", 0.8, 10.0, 50.0)
    assert result.dose_factor == 0.75
    assert math.isclose(result.adjusted_dose_mg, 150.0)


def test_adjust_dose_class_c_50pct_reduction():
    """Class C: dose factor = 0.50."""
    result = adjust_dose_hepatic("Drug_C", 200.0, "C", 0.8, 10.0, 50.0)
    assert result.dose_factor == 0.50
    assert math.isclose(result.adjusted_dose_mg, 100.0)


def test_adjust_dose_cl_adjusted():
    """CL_adjusted = CL_normal * dose_factor."""
    result = adjust_dose_hepatic("Drug_D", 100.0, "B", 0.8, 10.0, 50.0)
    assert math.isclose(result.cl_adjusted_L_per_h, 7.5)


def test_adjust_dose_t_half():
    """t_half = 0.693 * Vd / CL_adjusted."""
    result = adjust_dose_hepatic("Drug_E", 100.0, "A", 0.5, 10.0, 70.0)
    expected = 0.693 * 70.0 / 10.0
    assert math.isclose(result.t_half_adjusted_h, expected, rel_tol=1e-6)


def test_adjust_dose_t_half_class_c():
    """t_half longer in severe impairment (smaller CL_adjusted)."""
    result_a = adjust_dose_hepatic("X", 100.0, "A", 0.8, 10.0, 50.0)
    result_c = adjust_dose_hepatic("X", 100.0, "C", 0.8, 10.0, 50.0)
    assert result_c.t_half_adjusted_h > result_a.t_half_adjusted_h


def test_adjust_dose_result_is_dataclass():
    """HepaticDosingResult should be a mutable dataclass."""
    result = adjust_dose_hepatic("Drug_F", 100.0, "A", 0.8, 10.0, 50.0)
    result.notes.append("extra note")
    assert "extra note" in result.notes


def test_adjust_dose_high_hepatic_fraction_note():
    """High f_hepatic_metabolism triggers a note."""
    result = adjust_dose_hepatic("Drug_G", 100.0, "B", 0.9, 10.0, 50.0)
    combined = " ".join(result.notes).lower()
    assert "hepatic" in combined or "high" in combined or "significant" in combined


def test_adjust_dose_low_hepatic_fraction_note():
    """Low f_hepatic_metabolism triggers a different note."""
    result = adjust_dose_hepatic("Drug_H", 100.0, "C", 0.1, 10.0, 50.0)
    combined = " ".join(result.notes).lower()
    assert "low" in combined or "limited" in combined


# ---------------------------------------------------------------------------
# adjust_dose_hepatic — validation errors
# ---------------------------------------------------------------------------


def test_adjust_dose_invalid_drug_name():
    with pytest.raises(ValueError, match="drug_name"):
        adjust_dose_hepatic("", 100.0, "A", 0.8, 10.0, 50.0)


def test_adjust_dose_invalid_dose():
    with pytest.raises(ValueError, match="normal_dose"):
        adjust_dose_hepatic("Drug_I", 0.0, "A", 0.8, 10.0, 50.0)


def test_adjust_dose_invalid_class():
    with pytest.raises(ValueError, match="child_pugh_class"):
        adjust_dose_hepatic("Drug_J", 100.0, "D", 0.8, 10.0, 50.0)


def test_adjust_dose_invalid_fm():
    with pytest.raises(ValueError, match="f_hepatic_metabolism"):
        adjust_dose_hepatic("Drug_K", 100.0, "A", 1.5, 10.0, 50.0)


def test_adjust_dose_invalid_cl():
    with pytest.raises(ValueError, match="cl_normal"):
        adjust_dose_hepatic("Drug_L", 100.0, "A", 0.8, 0.0, 50.0)


def test_adjust_dose_invalid_vd():
    with pytest.raises(ValueError, match="vd_L"):
        adjust_dose_hepatic("Drug_M", 100.0, "A", 0.8, 10.0, 0.0)
