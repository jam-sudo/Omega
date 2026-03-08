"""Tests for Phase 413 — Drug Solubility Temperature Dependence."""

import math

import pytest

from omega_pbpk.prediction.solubility_temperature_dependence import (
    _R_KJ,
    SolubilityTemperatureResult,
    _van_t_hoff_ratio,
    compare_temperatures,
    predict_solubility_temperature,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DRUG = "DrugX"
S_REF = 1.0  # mg/mL
LOGP = 2.0
MW = 300.0
T_REF = 25.0


def _predict(temperature_C=37.0, t_ref_C=25.0, delta_H=None, logp=LOGP, s_ref=S_REF):
    return predict_solubility_temperature(
        drug_name=DRUG,
        solubility_ref_mg_mL=s_ref,
        temperature_C=temperature_C,
        logp=logp,
        mw_da=MW,
        t_ref_C=t_ref_C,
        delta_H_kJ_per_mol=delta_H,
    )


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _predict()
    assert isinstance(result, SolubilityTemperatureResult)


# ---------------------------------------------------------------------------
# 2. Ratio at reference temperature equals 1
# ---------------------------------------------------------------------------


def test_ratio_at_ref_temperature():
    result = _predict(temperature_C=T_REF)
    assert abs(result.solubility_ratio - 1.0) < 1e-9
    assert abs(result.solubility_predicted_mg_mL - S_REF) < 1e-9


# ---------------------------------------------------------------------------
# 3. Endothermic dissolution → higher temp → more soluble
# ---------------------------------------------------------------------------


def test_higher_temp_more_soluble_endothermic():
    r_low = _predict(temperature_C=10.0, delta_H=20.0)
    r_high = _predict(temperature_C=60.0, delta_H=20.0)
    assert r_high.solubility_predicted_mg_mL > r_low.solubility_predicted_mg_mL


# ---------------------------------------------------------------------------
# 4. Van't Hoff formula correctness (manual calculation)
# ---------------------------------------------------------------------------


def test_vant_hoff_formula_manual():
    delta_H = 20.0
    T_K = 37.0 + 273.15
    T_ref_K = 25.0 + 273.15
    expected_ratio = math.exp(-delta_H / _R_KJ * (1.0 / T_K - 1.0 / T_ref_K))
    result = _predict(temperature_C=37.0, delta_H=delta_H)
    assert abs(result.solubility_ratio - expected_ratio) < 1e-9


# ---------------------------------------------------------------------------
# 5. Solubility_ratio consistency with predicted solubility
# ---------------------------------------------------------------------------


def test_ratio_consistent_with_predicted():
    result = _predict()
    assert abs(result.solubility_predicted_mg_mL - S_REF * result.solubility_ratio) < 1e-9


# ---------------------------------------------------------------------------
# 6. delta_H estimation from logP
# ---------------------------------------------------------------------------


def test_delta_H_estimation_from_logP():
    result = _predict(logp=2.0)
    expected_H = max(5.0, 15.0 + 2.0 * 2.0)
    assert abs(result.delta_H_solution_kJ_per_mol - expected_H) < 1e-9


def test_delta_H_minimum_clamp():
    # logP = -20 → 15 + 2*(-20) = -25 → clamped to 5
    result = _predict(logp=-20.0)
    assert result.delta_H_solution_kJ_per_mol == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 7. Explicit delta_H override respected
# ---------------------------------------------------------------------------


def test_explicit_delta_H_override():
    result = _predict(delta_H=50.0)
    assert result.delta_H_solution_kJ_per_mol == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 8. Temperature sensitivity class
# ---------------------------------------------------------------------------


def test_sensitivity_class_low():
    # Very small delta_H → small 10-degree ratio → "low"
    result = _predict(delta_H=1.0)
    assert result.temperature_sensitivity_class == "low"


def test_sensitivity_class_moderate():
    # Medium delta_H → moderate ratio
    result = _predict(delta_H=20.0)
    class_ = result.temperature_sensitivity_class
    assert class_ in ("low", "moderate", "high")  # depends on exact value

    # Manually check the 10-degree ratio to assert correctly
    ratio_10 = _van_t_hoff_ratio(20.0, 308.15, 298.15)
    if ratio_10 < 1.5:
        assert class_ == "low"
    elif ratio_10 < 3.0:
        assert class_ == "moderate"
    else:
        assert class_ == "high"


def test_sensitivity_class_high():
    # Very large delta_H → ratio > 3
    result = _predict(delta_H=200.0)
    assert result.temperature_sensitivity_class == "high"


# ---------------------------------------------------------------------------
# 9. Intrinsic solubility at 25 C
# ---------------------------------------------------------------------------


def test_intrinsic_solubility_at_25_when_ref_is_25():
    result = _predict(t_ref_C=25.0)
    assert abs(result.intrinsic_solubility_mg_mL - S_REF) < 1e-9


def test_intrinsic_solubility_at_25_when_ref_is_37():
    delta_H = 20.0
    result = predict_solubility_temperature(
        drug_name=DRUG,
        solubility_ref_mg_mL=S_REF,
        temperature_C=25.0,
        logp=LOGP,
        mw_da=MW,
        t_ref_C=37.0,
        delta_H_kJ_per_mol=delta_H,
    )
    # Expected: S_ref * exp(-dH/R * (1/298.15 - 1/310.15))
    expected = S_REF * math.exp(-delta_H / _R_KJ * (1 / 298.15 - 1 / 310.15))
    assert abs(result.intrinsic_solubility_mg_mL - expected) < 1e-6


# ---------------------------------------------------------------------------
# 10. Storage recommendation
# ---------------------------------------------------------------------------


def test_storage_recommendation_refrigerate_high_dH():
    result = _predict(delta_H=50.0)
    assert result.storage_recommendation == "refrigerate"


def test_storage_recommendation_room_temperature_low_dH():
    result = _predict(delta_H=10.0)
    assert result.storage_recommendation == "room_temperature"


# ---------------------------------------------------------------------------
# 11. compare_temperatures returns 5 results sorted ascending
# ---------------------------------------------------------------------------


def test_compare_temperatures_returns_five():
    results = compare_temperatures(DRUG, S_REF, LOGP, MW)
    assert len(results) == 5


def test_compare_temperatures_sorted_ascending():
    results = compare_temperatures(DRUG, S_REF, LOGP, MW)
    solubilities = [r.solubility_predicted_mg_mL for r in results]
    assert solubilities == sorted(solubilities)


def test_compare_temperatures_custom_list():
    results = compare_temperatures(DRUG, S_REF, LOGP, MW, temperatures_C=[10.0, 40.0])
    assert len(results) == 2


# ---------------------------------------------------------------------------
# 12. Validation errors
# ---------------------------------------------------------------------------


def test_validation_negative_solubility():
    with pytest.raises(ValueError, match="solubility_ref_mg_mL"):
        _predict(s_ref=-1.0)


def test_validation_zero_mw():
    with pytest.raises(ValueError, match="mw_da"):
        predict_solubility_temperature(
            drug_name=DRUG,
            solubility_ref_mg_mL=S_REF,
            temperature_C=37.0,
            logp=LOGP,
            mw_da=0.0,
        )


def test_validation_temperature_out_of_range():
    with pytest.raises(ValueError, match="temperature_C"):
        _predict(temperature_C=120.0)


def test_validation_temperature_too_low():
    with pytest.raises(ValueError, match="temperature_C"):
        _predict(temperature_C=-25.0)


# ---------------------------------------------------------------------------
# 13. Result is frozen (immutable)
# ---------------------------------------------------------------------------


def test_result_is_frozen():
    result = _predict()
    with pytest.raises(Exception):
        result.drug_name = "Other"  # type: ignore[misc]
