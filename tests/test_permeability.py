"""Tests for omega_pbpk.prediction.permeability module."""

import math
from dataclasses import fields

import pytest

from omega_pbpk.prediction.permeability import (
    PermeabilityResult,
    _hbd_count_estimate,
    _papp_from_descriptors,
    predict_permeability,
    screen_permeability,
)

# ---------------------------------------------------------------------------
# PermeabilityResult dataclass
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "drug_name",
    "logP",
    "MW",
    "PSA",
    "HBD",
    "Papp_AB_cm_s",
    "Peff_cm_s",
    "log_Papp",
    "permeability_class",
    "fa_predicted",
    "efflux_ratio_estimate",
    "transport_mechanism",
}


def test_permeability_result_is_dataclass():
    field_names = {f.name for f in fields(PermeabilityResult)}
    assert REQUIRED_FIELDS <= field_names


# ---------------------------------------------------------------------------
# predict_permeability — basic return type and fields
# ---------------------------------------------------------------------------


@pytest.fixture
def typical_result():
    """Lisinopril-like: moderate logP, typical MW/PSA."""
    return predict_permeability("TestDrug", logP=1.5, MW=300.0, PSA=80.0)


def test_predict_returns_permeability_result(typical_result):
    assert isinstance(typical_result, PermeabilityResult)


def test_drug_name_stored(typical_result):
    assert typical_result.drug_name == "TestDrug"


def test_logp_stored(typical_result):
    assert typical_result.logP == 1.5


def test_mw_stored(typical_result):
    assert typical_result.MW == 300.0


def test_psa_stored(typical_result):
    assert typical_result.PSA == 80.0


# ---------------------------------------------------------------------------
# Papp range
# ---------------------------------------------------------------------------


def test_papp_in_valid_range(typical_result):
    assert 1e-8 <= typical_result.Papp_AB_cm_s <= 1e-4


def test_papp_range_high_logp():
    r = predict_permeability("HighLogP", logP=4.0, MW=250.0, PSA=30.0)
    assert 1e-8 <= r.Papp_AB_cm_s <= 1e-4


def test_papp_range_low_logp():
    r = predict_permeability("LowLogP", logP=-1.0, MW=400.0, PSA=150.0)
    assert 1e-8 <= r.Papp_AB_cm_s <= 1e-4


# ---------------------------------------------------------------------------
# log_Papp consistency
# ---------------------------------------------------------------------------


def test_log_papp_equals_log10_papp(typical_result):
    assert pytest.approx(typical_result.log_Papp, rel=1e-6) == math.log10(
        typical_result.Papp_AB_cm_s
    )


# ---------------------------------------------------------------------------
# Permeability class
# ---------------------------------------------------------------------------


def test_permeability_class_valid_values(typical_result):
    assert typical_result.permeability_class in {"high", "low", "intermediate"}


def test_high_papp_class_is_high():
    # Very lipophilic, small, low PSA -> high Papp
    r = predict_permeability("HighPerm", logP=5.0, MW=150.0, PSA=10.0, HBD=0)
    assert r.permeability_class == "high"
    assert r.Papp_AB_cm_s >= 1e-6


def test_low_papp_class_is_low():
    # Polar, large, high PSA -> low Papp
    r = predict_permeability("LowPerm", logP=-2.0, MW=600.0, PSA=200.0, HBD=8)
    assert r.permeability_class == "low"
    assert r.Papp_AB_cm_s < 1e-7


def test_intermediate_class_boundary():
    # Find a drug with intermediate permeability and verify class
    r = predict_permeability("MidPerm", logP=1.0, MW=350.0, PSA=100.0)
    if 1e-7 <= r.Papp_AB_cm_s < 1e-6:
        assert r.permeability_class == "intermediate"
    # If not intermediate, class must still be valid
    assert r.permeability_class in {"high", "low", "intermediate"}


# ---------------------------------------------------------------------------
# fa_predicted
# ---------------------------------------------------------------------------


def test_fa_in_unit_interval(typical_result):
    assert 0.0 <= typical_result.fa_predicted <= 1.0


def test_fa_high_for_high_permeability():
    r = predict_permeability("HighPerm", logP=5.0, MW=150.0, PSA=10.0, HBD=0)
    assert r.fa_predicted > 0.5


def test_fa_low_for_low_permeability():
    r = predict_permeability("LowPerm", logP=-2.0, MW=600.0, PSA=200.0, HBD=8)
    assert r.fa_predicted < 0.5


# ---------------------------------------------------------------------------
# Efflux ratio
# ---------------------------------------------------------------------------


def test_efflux_ratio_not_efflux_substrate():
    r = predict_permeability("Drug", logP=2.0, MW=300.0, PSA=60.0, efflux_substrate=False)
    assert r.efflux_ratio_estimate == pytest.approx(1.0)


def test_efflux_ratio_efflux_substrate_greater_than_one():
    r = predict_permeability("Drug", logP=2.0, MW=300.0, PSA=60.0, efflux_substrate=True)
    assert r.efflux_ratio_estimate > 1.0


def test_efflux_ratio_always_nonnegative(typical_result):
    assert typical_result.efflux_ratio_estimate >= 1.0


# ---------------------------------------------------------------------------
# Effect of descriptors on Papp
# ---------------------------------------------------------------------------


def test_higher_logp_gives_higher_papp():
    r_high = predict_permeability("HighLP", logP=5.0, MW=200.0, PSA=30.0, HBD=1)
    r_low = predict_permeability("LowLP", logP=1.0, MW=200.0, PSA=30.0, HBD=1)
    assert r_high.Papp_AB_cm_s > r_low.Papp_AB_cm_s


def test_higher_psa_gives_lower_papp():
    r_low_psa = predict_permeability("LowPSA", logP=3.0, MW=200.0, PSA=20.0, HBD=1)
    r_high_psa = predict_permeability("HighPSA", logP=3.0, MW=200.0, PSA=100.0, HBD=1)
    assert r_low_psa.Papp_AB_cm_s > r_high_psa.Papp_AB_cm_s


# ---------------------------------------------------------------------------
# HBD estimation when None
# ---------------------------------------------------------------------------


def test_hbd_none_uses_estimate():
    r = predict_permeability("Drug", logP=2.0, MW=300.0, PSA=60.0, HBD=None)
    # HBD field should be filled in (non-negative int)
    assert r.HBD is not None
    assert r.HBD >= 0


def test_hbd_none_result_is_valid_permeability_result():
    r = predict_permeability("Drug", logP=2.0, MW=300.0, PSA=60.0)
    assert isinstance(r, PermeabilityResult)
    assert 1e-8 <= r.Papp_AB_cm_s <= 1e-4


# ---------------------------------------------------------------------------
# transport_mechanism
# ---------------------------------------------------------------------------


def test_transport_mechanism_is_string(typical_result):
    assert isinstance(typical_result.transport_mechanism, str)


def test_transport_mechanism_not_empty(typical_result):
    assert len(typical_result.transport_mechanism) > 0


# ---------------------------------------------------------------------------
# Peff
# ---------------------------------------------------------------------------


def test_peff_positive(typical_result):
    assert typical_result.Peff_cm_s > 0.0


def test_peff_is_float(typical_result):
    assert isinstance(typical_result.Peff_cm_s, float)


# ---------------------------------------------------------------------------
# screen_permeability
# ---------------------------------------------------------------------------

DRUG_LIST = [
    {"drug_name": "DrugA", "logP": 4.0, "MW": 250.0, "PSA": 30.0},
    {"drug_name": "DrugB", "logP": 0.5, "MW": 400.0, "PSA": 120.0},
    {"drug_name": "DrugC", "logP": 2.0, "MW": 320.0, "PSA": 70.0},
    {"drug_name": "DrugD", "logP": -1.0, "MW": 500.0, "PSA": 180.0},
]


def test_screen_returns_list():
    result = screen_permeability(DRUG_LIST)
    assert isinstance(result, list)


def test_screen_returns_correct_length():
    result = screen_permeability(DRUG_LIST)
    assert len(result) == len(DRUG_LIST)


def test_screen_returns_permeability_results():
    result = screen_permeability(DRUG_LIST)
    for r in result:
        assert isinstance(r, PermeabilityResult)


def test_screen_sorted_by_papp_descending():
    result = screen_permeability(DRUG_LIST)
    papp_values = [r.Papp_AB_cm_s for r in result]
    assert papp_values == sorted(papp_values, reverse=True)


def test_screen_empty_list():
    assert screen_permeability([]) == []


def test_screen_single_drug():
    result = screen_permeability([{"drug_name": "OnlyDrug", "logP": 2.0, "MW": 300.0, "PSA": 60.0}])
    assert len(result) == 1
    assert result[0].drug_name == "OnlyDrug"


def test_screen_accepts_optional_hbd_and_efflux():
    drugs = [
        {
            "drug_name": "X",
            "logP": 2.0,
            "MW": 300.0,
            "PSA": 60.0,
            "HBD": 2,
            "efflux_substrate": True,
        },
        {"drug_name": "Y", "logP": 3.0, "MW": 250.0, "PSA": 40.0},
    ]
    result = screen_permeability(drugs)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# ValueError conditions
# ---------------------------------------------------------------------------


def test_mw_zero_raises_value_error():
    with pytest.raises(ValueError):
        predict_permeability("Drug", logP=2.0, MW=0.0, PSA=60.0)


def test_mw_negative_raises_value_error():
    with pytest.raises(ValueError):
        predict_permeability("Drug", logP=2.0, MW=-100.0, PSA=60.0)


def test_psa_negative_raises_value_error():
    with pytest.raises(ValueError):
        predict_permeability("Drug", logP=2.0, MW=300.0, PSA=-10.0)


# ---------------------------------------------------------------------------
# Private helpers: _hbd_count_estimate
# ---------------------------------------------------------------------------


def test_hbd_estimate_returns_int():
    result = _hbd_count_estimate(MW=300.0, logP=2.0, PSA=60.0)
    assert isinstance(result, int)


def test_hbd_estimate_nonnegative():
    for mw, lp, psa in [(150.0, 1.0, 20.0), (500.0, 3.0, 100.0), (800.0, -1.0, 200.0)]:
        assert _hbd_count_estimate(mw, lp, psa) >= 0


def test_hbd_estimate_higher_psa_more_hbd():
    low = _hbd_count_estimate(MW=300.0, logP=2.0, PSA=30.0)
    high = _hbd_count_estimate(MW=300.0, logP=2.0, PSA=150.0)
    assert high >= low


# ---------------------------------------------------------------------------
# Private helpers: _papp_from_descriptors
# ---------------------------------------------------------------------------


def test_papp_from_descriptors_returns_float():
    result = _papp_from_descriptors(logP=2.0, MW=300.0, PSA=60.0, HBD=2)
    assert isinstance(result, float)


def test_papp_from_descriptors_in_range():
    result = _papp_from_descriptors(logP=2.0, MW=300.0, PSA=60.0, HBD=2)
    assert 1e-8 <= result <= 1e-4


def test_papp_from_descriptors_monotone_logp():
    low = _papp_from_descriptors(logP=1.0, MW=200.0, PSA=30.0, HBD=1)
    high = _papp_from_descriptors(logP=5.0, MW=200.0, PSA=30.0, HBD=1)
    assert high > low


def test_papp_from_descriptors_monotone_psa():
    low_psa = _papp_from_descriptors(logP=3.0, MW=200.0, PSA=20.0, HBD=1)
    high_psa = _papp_from_descriptors(logP=3.0, MW=200.0, PSA=100.0, HBD=1)
    assert low_psa > high_psa
