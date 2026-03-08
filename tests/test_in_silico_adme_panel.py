"""Tests for Phase 454 — in_silico_adme_panel.py"""

import pytest

from omega_pbpk.prediction.in_silico_adme_panel import (
    InSilicoADMEResult,
    predict_adme_panel,
    screen_adme_panel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(**kwargs):
    defaults = dict(
        drug_name="TestCompound",
        mw_Da=350.0,
        logp=2.0,
        pka=7.4,
        ionization_type="neutral",
    )
    defaults.update(kwargs)
    return predict_adme_panel(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic fields
# ---------------------------------------------------------------------------


def test_return_type():
    result = make_result()
    assert isinstance(result, InSilicoADMEResult)


def test_drug_name_preserved():
    result = make_result(drug_name="Aspirin")
    assert result.drug_name == "Aspirin"


# ---------------------------------------------------------------------------
# Predicted value ranges
# ---------------------------------------------------------------------------


def test_hia_between_0_and_1():
    result = make_result()
    assert 0.0 <= result.predicted_hia <= 1.0


def test_f_oral_between_0_and_1():
    result = make_result()
    assert 0.0 <= result.predicted_f_oral <= 1.0


def test_fu_plasma_between_0_and_1():
    result = make_result()
    assert 0.0 <= result.predicted_fu_plasma <= 1.0


def test_vd_positive():
    result = make_result()
    assert result.predicted_vd_L_per_kg > 0.0


def test_cl_positive():
    result = make_result()
    assert result.predicted_cl_mL_per_min_per_kg > 0.0


def test_t_half_positive():
    result = make_result()
    assert result.predicted_t_half_h > 0.0


def test_bbb_between_0_and_1():
    result = make_result()
    assert 0.0 <= result.predicted_bbb_penetration <= 1.0


def test_lipinski_score_between_0_and_5():
    result = make_result()
    assert 0 <= result.lipinski_score <= 5


def test_lead_likeness_between_0_and_100():
    result = make_result()
    assert 0.0 <= result.lead_likeness_score <= 100.0


# ---------------------------------------------------------------------------
# Trend tests
# ---------------------------------------------------------------------------


def test_higher_logp_lower_fu_plasma():
    low = make_result(logp=1.0)
    high = make_result(logp=4.0)
    assert high.predicted_fu_plasma < low.predicted_fu_plasma


def test_higher_mw_slightly_lower_hia():
    low = make_result(mw_Da=200.0)
    high = make_result(mw_Da=450.0)
    assert high.predicted_hia <= low.predicted_hia


# ---------------------------------------------------------------------------
# Substrate predictions
# ---------------------------------------------------------------------------


def test_p_gp_substrate_high_mw_high_logp_base():
    """MW > 400, logP > 2, base → P-gp substrate."""
    result = make_result(mw_Da=450.0, logp=3.0, ionization_type="base")
    assert result.predicted_p_gp_substrate is True


def test_p_gp_not_substrate_acid():
    """Acid drug should not be P-gp substrate even if MW/logP high."""
    result = make_result(mw_Da=450.0, logp=3.0, ionization_type="acid")
    assert result.predicted_p_gp_substrate is False


def test_cyp2d6_substrate_base_pka_9():
    """Base with pKa=9 (in range 7-10) → CYP2D6 substrate."""
    result = make_result(ionization_type="base", pka=9.0)
    assert result.predicted_cyp2d6_substrate is True


def test_cyp2d6_not_substrate_neutral():
    result = make_result(ionization_type="neutral", pka=9.0)
    assert result.predicted_cyp2d6_substrate is False


def test_cyp3a4_substrate_large_lipophilic():
    result = make_result(mw_Da=400.0, logp=2.0)
    assert result.predicted_cyp3a4_substrate is True


def test_cyp3a4_not_substrate_small_polar():
    result = make_result(mw_Da=200.0, logp=0.5)
    assert result.predicted_cyp3a4_substrate is False


# ---------------------------------------------------------------------------
# screen_adme_panel
# ---------------------------------------------------------------------------

COMPOUNDS = [
    {"drug_name": "CompA", "mw_Da": 300.0, "logp": 1.5, "pka": 7.0, "ionization_type": "neutral"},
    {"drug_name": "CompB", "mw_Da": 600.0, "logp": 6.0, "pka": 9.0, "ionization_type": "base"},
    {"drug_name": "CompC", "mw_Da": 250.0, "logp": 2.0, "pka": 5.0, "ionization_type": "acid"},
]


def test_screen_returns_correct_length():
    results = screen_adme_panel(COMPOUNDS)
    assert len(results) == 3


def test_screen_sorted_descending_by_lead_likeness():
    results = screen_adme_panel(COMPOUNDS)
    scores = [r.lead_likeness_score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_adme_panel("X", mw_Da=0.0, logp=2.0, pka=7.0, ionization_type="neutral")


def test_mw_negative_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_adme_panel("X", mw_Da=-100.0, logp=2.0, pka=7.0, ionization_type="neutral")


def test_bad_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_adme_panel("X", mw_Da=300.0, logp=2.0, pka=7.0, ionization_type="zwitterion")


def test_logp_too_high_raises():
    with pytest.raises(ValueError, match="logp"):
        predict_adme_panel("X", mw_Da=300.0, logp=11.0, pka=7.0, ionization_type="neutral")


def test_logp_too_low_raises():
    with pytest.raises(ValueError, match="logp"):
        predict_adme_panel("X", mw_Da=300.0, logp=-6.0, pka=7.0, ionization_type="neutral")
