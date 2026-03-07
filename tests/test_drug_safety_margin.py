"""Tests for omega_pbpk.risk.drug_safety_margin module."""

import pytest

from omega_pbpk.risk.drug_safety_margin import (
    SafetyMarginResult,
    classify_therapeutic_window,
    compute_safety_margin,
    screen_safety_margins,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_RISK_CATEGORIES = {"very_high", "high", "moderate", "low"}
VALID_SENSITIVITIES = {"very_sensitive", "sensitive", "moderate", "robust"}


def _make_basic(**kwargs):
    defaults = dict(
        drug_name="DrugX",
        therapeutic_cmax_mg_L=1.0,
        toxic_cmax_mg_L=5.0,
        mec_mg_L=0.5,
        mtc_mg_L=4.0,
    )
    defaults.update(kwargs)
    return compute_safety_margin(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_safety_margin_result():
    result = _make_basic()
    assert isinstance(result, SafetyMarginResult)


# ---------------------------------------------------------------------------
# Therapeutic index
# ---------------------------------------------------------------------------


def test_therapeutic_index_is_ratio():
    result = compute_safety_margin(
        "DrugA",
        therapeutic_cmax_mg_L=2.0,
        toxic_cmax_mg_L=10.0,
        mec_mg_L=1.0,
        mtc_mg_L=8.0,
    )
    assert result.therapeutic_index == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Safety margin fold
# ---------------------------------------------------------------------------


def test_safety_margin_fold_gt_1():
    result = _make_basic()
    assert result.safety_margin_fold > 1.0


# ---------------------------------------------------------------------------
# Therapeutic window width
# ---------------------------------------------------------------------------


def test_therapeutic_window_width():
    # (4.0 - 0.5) / 0.5 * 100 = 700%
    result = _make_basic(mec_mg_L=0.5, mtc_mg_L=4.0)
    assert result.therapeutic_window_width == pytest.approx(700.0)


# ---------------------------------------------------------------------------
# narrow_therapeutic_window
# ---------------------------------------------------------------------------


def test_narrow_when_ti_lt_2():
    result = compute_safety_margin(
        "NarrowDrug",
        therapeutic_cmax_mg_L=2.0,
        toxic_cmax_mg_L=3.0,  # TI = 1.5 < 2
        mec_mg_L=1.0,
        mtc_mg_L=5.0,  # window = 400%
    )
    assert result.narrow_therapeutic_window is True


def test_not_narrow_when_ti_ge_2_and_window_wide():
    result = compute_safety_margin(
        "WideDrug",
        therapeutic_cmax_mg_L=1.0,
        toxic_cmax_mg_L=5.0,  # TI = 5
        mec_mg_L=0.5,
        mtc_mg_L=5.0,  # window = 900%
    )
    assert result.narrow_therapeutic_window is False


def test_narrow_when_window_lt_100_even_if_ti_ge_2():
    # TI = 3, but window = (1.5 - 1.0) / 1.0 * 100 = 50% < 100
    result = compute_safety_margin(
        "NarrowWindow",
        therapeutic_cmax_mg_L=1.0,
        toxic_cmax_mg_L=3.0,  # TI = 3
        mec_mg_L=1.0,
        mtc_mg_L=1.5,  # window = 50%
    )
    assert result.narrow_therapeutic_window is True


# ---------------------------------------------------------------------------
# risk_category
# ---------------------------------------------------------------------------


def test_risk_category_very_high_for_ti_1_5():
    result = compute_safety_margin(
        "HighRisk",
        therapeutic_cmax_mg_L=2.0,
        toxic_cmax_mg_L=3.0,  # TI = 1.5
        mec_mg_L=1.0,
        mtc_mg_L=5.0,
    )
    assert result.risk_category == "very_high"


def test_risk_category_low_for_ti_20():
    result = compute_safety_margin(
        "LowRisk",
        therapeutic_cmax_mg_L=1.0,
        toxic_cmax_mg_L=20.0,  # TI = 20
        mec_mg_L=0.5,
        mtc_mg_L=25.0,
    )
    assert result.risk_category == "low"


# ---------------------------------------------------------------------------
# dose_adjustment_sensitivity
# ---------------------------------------------------------------------------


def test_sensitivity_very_sensitive_for_ti_1_5():
    result = compute_safety_margin(
        "Sensitive",
        therapeutic_cmax_mg_L=2.0,
        toxic_cmax_mg_L=3.0,  # TI = 1.5
        mec_mg_L=1.0,
        mtc_mg_L=5.0,
    )
    assert result.dose_adjustment_sensitivity == "very_sensitive"


def test_sensitivity_robust_for_ti_15():
    result = compute_safety_margin(
        "Robust",
        therapeutic_cmax_mg_L=1.0,
        toxic_cmax_mg_L=15.0,  # TI = 15
        mec_mg_L=0.5,
        mtc_mg_L=20.0,
    )
    assert result.dose_adjustment_sensitivity == "robust"


# ---------------------------------------------------------------------------
# auc_ratio
# ---------------------------------------------------------------------------


def test_auc_ratio_when_both_provided():
    result = compute_safety_margin(
        "AUCDrug",
        therapeutic_cmax_mg_L=1.0,
        toxic_cmax_mg_L=5.0,
        mec_mg_L=0.5,
        mtc_mg_L=8.0,
        therapeutic_auc=100.0,
        toxic_auc=300.0,
    )
    assert result.auc_ratio == pytest.approx(3.0)


def test_auc_ratio_default_when_not_provided():
    result = _make_basic()
    assert result.auc_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = _make_basic()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# classify_therapeutic_window
# ---------------------------------------------------------------------------


def test_classify_returns_dict_with_expected_keys():
    result = classify_therapeutic_window(mec_mg_L=1.0, mtc_mg_L=5.0)
    assert "window_width_pct" in result
    assert "classification" in result
    assert "notes" in result


def test_classify_narrow_when_window_lt_100():
    # window = (1.4 - 1.0) / 1.0 * 100 = 40%
    result = classify_therapeutic_window(mec_mg_L=1.0, mtc_mg_L=1.4)
    assert result["classification"] == "narrow"


def test_classify_wide_when_window_gt_400():
    # window = (6.0 - 1.0) / 1.0 * 100 = 500%
    result = classify_therapeutic_window(mec_mg_L=1.0, mtc_mg_L=6.0)
    assert result["classification"] == "wide"


# ---------------------------------------------------------------------------
# screen_safety_margins
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        dict(
            drug_name="A",
            therapeutic_cmax_mg_L=1.0,
            toxic_cmax_mg_L=5.0,
            mec_mg_L=0.5,
            mtc_mg_L=8.0,
        ),
        dict(
            drug_name="B",
            therapeutic_cmax_mg_L=2.0,
            toxic_cmax_mg_L=3.0,
            mec_mg_L=1.0,
            mtc_mg_L=5.0,
        ),
    ]
    results = screen_safety_margins(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_ti_ascending():
    compounds = [
        dict(
            drug_name="Safe",
            therapeutic_cmax_mg_L=1.0,
            toxic_cmax_mg_L=20.0,
            mec_mg_L=0.5,
            mtc_mg_L=25.0,
        ),
        dict(
            drug_name="Risky",
            therapeutic_cmax_mg_L=2.0,
            toxic_cmax_mg_L=3.0,
            mec_mg_L=1.0,
            mtc_mg_L=5.0,
        ),
        dict(
            drug_name="Mid",
            therapeutic_cmax_mg_L=1.0,
            toxic_cmax_mg_L=5.0,
            mec_mg_L=0.5,
            mtc_mg_L=8.0,
        ),
    ]
    results = screen_safety_margins(compounds)
    tis = [r.therapeutic_index for r in results]
    assert tis == sorted(tis)


def test_screen_empty_returns_empty():
    assert screen_safety_margins([]) == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_when_therapeutic_cmax_zero():
    with pytest.raises(ValueError):
        compute_safety_margin(
            "X", therapeutic_cmax_mg_L=0.0, toxic_cmax_mg_L=5.0, mec_mg_L=0.5, mtc_mg_L=4.0
        )


def test_raises_when_toxic_cmax_lt_therapeutic():
    with pytest.raises(ValueError):
        compute_safety_margin(
            "X", therapeutic_cmax_mg_L=5.0, toxic_cmax_mg_L=3.0, mec_mg_L=0.5, mtc_mg_L=4.0
        )


def test_raises_when_mec_zero():
    with pytest.raises(ValueError):
        compute_safety_margin(
            "X", therapeutic_cmax_mg_L=1.0, toxic_cmax_mg_L=5.0, mec_mg_L=0.0, mtc_mg_L=4.0
        )


def test_raises_when_mtc_lt_mec():
    with pytest.raises(ValueError):
        compute_safety_margin(
            "X", therapeutic_cmax_mg_L=1.0, toxic_cmax_mg_L=5.0, mec_mg_L=3.0, mtc_mg_L=2.0
        )
