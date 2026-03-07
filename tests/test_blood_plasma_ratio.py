"""Tests for Phase 738 — Blood-to-Plasma Ratio Predictor."""

from omega_pbpk.prediction.blood_plasma_ratio import (
    BloodPlasmaRatioResult,
    predict_blood_plasma_ratio,
    screen_blood_plasma_ratio,
)


def test_returns_blood_plasma_ratio_result():
    result = predict_blood_plasma_ratio("TestDrug")
    assert isinstance(result, BloodPlasmaRatioResult)


def test_compound_name_stored():
    result = predict_blood_plasma_ratio("Warfarin")
    assert result.compound_name == "Warfarin"


def test_fields_echoed_back():
    result = predict_blood_plasma_ratio("DrugA", logp=2.5, charge_at_ph74=1.0, fu_plasma=0.3)
    assert result.logp == 2.5
    assert result.charge_at_ph74 == 1.0
    assert result.fu_plasma == 0.3


def test_rb_between_0_4_and_3_0():
    for logp in [-5.0, 0.0, 5.0, 20.0]:
        result = predict_blood_plasma_ratio("X", logp=logp)
        assert 0.4 <= result.rb <= 3.0


def test_rb_clamped_low():
    result = predict_blood_plasma_ratio("Low", logp=-10.0, charge_at_ph74=-2.0, fu_plasma=0.01)
    assert result.rb >= 0.4


def test_rb_clamped_high():
    result = predict_blood_plasma_ratio("High", logp=50.0, charge_at_ph74=2.0, fu_plasma=0.9)
    assert result.rb <= 3.0


def test_high_logp_gives_higher_rb_than_low_logp():
    low = predict_blood_plasma_ratio("Low", logp=0.0, charge_at_ph74=0.0, fu_plasma=0.5)
    high = predict_blood_plasma_ratio("High", logp=5.0, charge_at_ph74=0.0, fu_plasma=0.5)
    assert high.rb > low.rb


def test_basic_drug_higher_rb_than_neutral():
    neutral = predict_blood_plasma_ratio("Neutral", logp=2.0, charge_at_ph74=0.0, fu_plasma=0.5)
    basic = predict_blood_plasma_ratio("Basic", logp=2.0, charge_at_ph74=1.0, fu_plasma=0.5)
    assert basic.rb > neutral.rb


def test_acidic_drug_lower_rb_than_neutral():
    neutral = predict_blood_plasma_ratio("Neutral", logp=2.0, charge_at_ph74=0.0, fu_plasma=0.5)
    acidic = predict_blood_plasma_ratio("Acidic", logp=2.0, charge_at_ph74=-1.0, fu_plasma=0.5)
    assert acidic.rb < neutral.rb


def test_high_ppb_lowers_rb():
    unbound = predict_blood_plasma_ratio("Free", logp=2.0, charge_at_ph74=0.0, fu_plasma=0.8)
    bound = predict_blood_plasma_ratio("Bound", logp=2.0, charge_at_ph74=0.0, fu_plasma=0.05)
    assert bound.rb < unbound.rb


def test_cl_blood_computed_when_cl_plasma_provided():
    result = predict_blood_plasma_ratio("DrugCL", cl_plasma_L_per_h=10.0)
    assert result.cl_blood_L_per_h is not None


def test_cl_blood_none_when_cl_plasma_not_provided():
    result = predict_blood_plasma_ratio("DrugNoCL")
    assert result.cl_blood_L_per_h is None


def test_cl_blood_equals_cl_plasma_divided_by_rb():
    cl_plasma = 8.0
    result = predict_blood_plasma_ratio(
        "DrugX", logp=1.5, fu_plasma=0.4, cl_plasma_L_per_h=cl_plasma
    )
    assert result.cl_blood_L_per_h is not None
    assert abs(result.cl_blood_L_per_h - cl_plasma / result.rb) < 1e-9


def test_rbc_affinity_valid_values():
    valid = {"low", "neutral", "high"}
    for logp in [-2.0, 2.0, 10.0]:
        result = predict_blood_plasma_ratio("X", logp=logp)
        assert result.rbc_affinity in valid


def test_rbc_affinity_low_when_rb_below_0_8():
    result = predict_blood_plasma_ratio("Low", logp=-3.0, charge_at_ph74=-1.0, fu_plasma=0.05)
    assert result.rb < 0.8
    assert result.rbc_affinity == "low"


def test_rbc_affinity_high_when_rb_above_1_2():
    # logp=10, charge=+1 → rb_base=(0.55+0.5)*1.2=1.26 → rb=1.26 > 1.2
    result = predict_blood_plasma_ratio("High", logp=10.0, charge_at_ph74=1.0, fu_plasma=0.9)
    assert result.rb > 1.2
    assert result.rbc_affinity == "high"


def test_impact_on_vd_underestimate_when_rb_below_1():
    result = predict_blood_plasma_ratio("Low", logp=-3.0, charge_at_ph74=-1.0, fu_plasma=0.05)
    assert result.rb < 1.0
    assert result.impact_on_vd == "may underestimate Vd"


def test_impact_on_vd_accurate_when_rb_ge_1():
    # logp=10, charge=+1 → rb=1.26 >= 1.0
    result = predict_blood_plasma_ratio("High", logp=10.0, charge_at_ph74=1.0, fu_plasma=0.9)
    assert result.rb >= 1.0
    assert result.impact_on_vd == "accurate"


def test_notes_nonempty():
    result = predict_blood_plasma_ratio("AnyDrug")
    assert len(result.notes) > 0


def test_screen_returns_list():
    compounds = [
        {"compound_name": "A", "logp": 1.0},
        {"compound_name": "B", "logp": 5.0, "charge_at_ph74": 1.0},
    ]
    results = screen_blood_plasma_ratio(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_rb_descending():
    compounds = [
        {"compound_name": "Low", "logp": -2.0, "charge_at_ph74": -1.0, "fu_plasma": 0.05},
        {"compound_name": "High", "logp": 8.0, "charge_at_ph74": 1.0, "fu_plasma": 0.9},
        {"compound_name": "Mid", "logp": 2.0},
    ]
    results = screen_blood_plasma_ratio(compounds)
    rbs = [r.rb for r in results]
    assert rbs == sorted(rbs, reverse=True)
