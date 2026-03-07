"""Tests for pKa predictor Phase 646 additions: predict_pka_v2, classify_ionization, screen_pka."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.pka_predictor import (
    PKaPrediction,
    classify_ionization,
    predict_pka_v2,
    screen_pka,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prediction(**kwargs):
    defaults = dict(compound_name="TestCompound", logP=1.0, mw=300.0)
    defaults.update(kwargs)
    return predict_pka_v2(**defaults)


# ---------------------------------------------------------------------------
# Return type and field counts
# ---------------------------------------------------------------------------


def test_returns_pka_prediction():
    res = _make_prediction(n_cooh_groups=1)
    assert isinstance(res, PKaPrediction)


def test_n_acidic_groups_count():
    res = _make_prediction(n_cooh_groups=1, n_phenol_groups=2)
    assert res.n_acidic_groups == 3


def test_n_basic_groups_count():
    res = _make_prediction(n_primary_amine=1, n_secondary_amine=1, n_pyridine_n=1)
    assert res.n_basic_groups == 3


# ---------------------------------------------------------------------------
# pKa acid/base list ordering
# ---------------------------------------------------------------------------


def test_predicted_pka_acids_sorted_descending():
    # phenol pKa > COOH pKa → phenol should be first (highest pKa = descending)
    res = _make_prediction(n_cooh_groups=1, n_phenol_groups=1, logP=1.0)
    assert res.predicted_pka_acids[0] >= res.predicted_pka_acids[1]


def test_predicted_pka_bases_sorted_descending():
    # primary amine pKa > pyridine pKa
    res = _make_prediction(n_primary_amine=1, n_pyridine_n=1, logP=1.0)
    assert res.predicted_pka_bases[0] >= res.predicted_pka_bases[1]


# ---------------------------------------------------------------------------
# pKa range checks
# ---------------------------------------------------------------------------


def test_cooh_pka_in_range():
    res = _make_prediction(n_cooh_groups=1, logP=1.0)
    pka = res.predicted_pka_acids[0]
    assert 3.5 <= pka <= 5.0


def test_phenol_pka_in_range():
    res = _make_prediction(n_phenol_groups=1, logP=1.0)
    pka = res.predicted_pka_acids[0]
    assert 8.0 <= pka <= 11.0


def test_primary_amine_pka_in_range():
    res = _make_prediction(n_primary_amine=1, logP=1.0)
    pka = res.predicted_pka_bases[0]
    assert 8.0 <= pka <= 12.0


def test_pyridine_pka_in_range():
    res = _make_prediction(n_pyridine_n=1, logP=1.0)
    pka = res.predicted_pka_bases[0]
    assert 2.0 <= pka <= 7.0


# ---------------------------------------------------------------------------
# Physiological charge and dominant form
# ---------------------------------------------------------------------------


def test_physiological_charge_negative_for_cooh():
    res = _make_prediction(n_cooh_groups=1, logP=1.0)
    # COOH pKa ~4.15 < 7.4 → anion → negative charge
    assert res.physiological_charge < 0.0


def test_physiological_charge_positive_for_primary_amine():
    res = _make_prediction(n_primary_amine=1, logP=1.0)
    # Primary amine pKa ~10.3 > 7.4 → cation → positive charge
    assert res.physiological_charge > 0.0


def test_dominant_form_neutral_for_no_groups():
    res = _make_prediction()
    assert res.dominant_ionization_form == "neutral"


def test_dominant_form_anion_for_cooh_only():
    res = _make_prediction(n_cooh_groups=1, logP=1.0)
    assert res.dominant_ionization_form == "anion"


def test_dominant_form_cation_for_primary_amine():
    res = _make_prediction(n_primary_amine=1, logP=1.0)
    assert res.dominant_ionization_form == "cation"


# ---------------------------------------------------------------------------
# Permeability and renal impact
# ---------------------------------------------------------------------------


def test_permeability_impact_high_for_neutral():
    res = _make_prediction()
    assert res.permeability_impact == "high"


def test_renal_clearance_impact_ph_dependent_for_pyridine():
    # Use logP=5.0: pKa = 4.5 + 5*0.1 = 5.0 → in range [5,8]
    res2 = predict_pka_v2(compound_name="PyridineTest", n_pyridine_n=1, logP=5.0)
    assert res2.renal_clearance_impact == "pH-dependent"


# ---------------------------------------------------------------------------
# screen_pka
# ---------------------------------------------------------------------------


def test_screen_pka_sorted_ascending_by_charge():
    compounds = [
        {"compound_name": "Acid", "n_cooh_groups": 1, "logP": 1.0},
        {"compound_name": "Amine", "n_primary_amine": 1, "logP": 1.0},
        {"compound_name": "Neutral"},
    ]
    results = screen_pka(compounds)
    charges = [r.physiological_charge for r in results]
    assert charges == sorted(charges)


def test_screen_pka_returns_list():
    compounds = [{"compound_name": "A"}, {"compound_name": "B"}]
    results = screen_pka(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# classify_ionization
# ---------------------------------------------------------------------------


def test_classify_ionization_returns_dict():
    result = classify_ionization([{"pka": 4.2, "type": "acid"}])
    assert isinstance(result, dict)


def test_classify_ionization_has_expected_keys():
    result = classify_ionization([{"pka": 4.2, "type": "acid"}])
    assert "fraction_ionized" in result
    assert "dominant_form" in result
    assert "charge" in result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_negative_group_count_raises():
    with pytest.raises(ValueError):
        predict_pka_v2(compound_name="X", n_cooh_groups=-1)


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_pka_v2(compound_name="X", mw=0.0)


def test_validation_temperature_too_low_raises():
    with pytest.raises(ValueError, match="temperature_c"):
        predict_pka_v2(compound_name="X", temperature_c=10.0)


def test_notes_nonempty():
    res = _make_prediction(n_cooh_groups=1)
    assert len(res.notes) > 0
