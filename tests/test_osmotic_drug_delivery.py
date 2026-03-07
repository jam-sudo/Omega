"""
Tests for Phase 875: Osmotic Drug Delivery Model
"""

import pytest

from omega_pbpk.core.osmotic_drug_delivery import (
    OsmoticDrugDeliveryResult,
    compare_osmotic_designs,
    simulate_osmotic_delivery,
)

# --- Return type and basic field tests ---


def test_returns_correct_type():
    result = simulate_osmotic_delivery("DrugA", 100.0)
    assert isinstance(result, OsmoticDrugDeliveryResult)


def test_fields_preserved():
    result = simulate_osmotic_delivery(
        "TestDrug", 200.0, t_release_h=6.0, lag_h=0.5, osmotic_type="single_layer"
    )
    assert result.drug_name == "TestDrug"
    assert result.total_dose_mg == 200.0
    assert result.t_release_h == 6.0
    assert result.lag_h == 0.5
    assert result.osmotic_type == "single_layer"


def test_times_list_length():
    result = simulate_osmotic_delivery("DrugA", 100.0, t_end_h=10.0, dt_h=0.5)
    assert len(result.times_h) > 0
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.a_core_mg)


def test_cmax_positive():
    result = simulate_osmotic_delivery("DrugA", 100.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_osmotic_delivery("DrugA", 100.0)
    assert result.auc_mg_h_per_L > 0.0


def test_fraction_released_le_one():
    result = simulate_osmotic_delivery("DrugA", 100.0, t_release_h=8.0, t_end_h=24.0)
    assert result.fraction_released <= 1.0


def test_fraction_released_positive():
    result = simulate_osmotic_delivery("DrugA", 100.0)
    assert result.fraction_released > 0.0


# --- Lag time behavior ---


def test_longer_lag_later_tmax():
    short_lag = simulate_osmotic_delivery("DrugA", 100.0, lag_h=0.5)
    long_lag = simulate_osmotic_delivery("DrugA", 100.0, lag_h=3.0)
    assert long_lag.tmax_h >= short_lag.tmax_h


def test_zero_lag_releases_earlier():
    no_lag = simulate_osmotic_delivery("DrugA", 100.0, lag_h=0.0)
    with_lag = simulate_osmotic_delivery("DrugA", 100.0, lag_h=2.0)
    assert no_lag.tmax_h <= with_lag.tmax_h


# --- Zero-order duration ---


def test_zero_order_duration_approx_t_release():
    # For single_layer, the zero-order release duration should be close to t_release_h
    result = simulate_osmotic_delivery(
        "DrugA", 100.0, t_release_h=8.0, lag_h=1.0, t_end_h=24.0, dt_h=0.05
    )
    # Allow 10% tolerance
    assert abs(result.zero_order_duration_h - 8.0) < 1.0


def test_zero_order_duration_positive():
    result = simulate_osmotic_delivery("DrugA", 100.0)
    assert result.zero_order_duration_h > 0.0


# --- Osmotic type effects ---


def test_push_pull_later_peak_than_single_layer():
    single = simulate_osmotic_delivery("DrugA", 100.0, lag_h=1.0, osmotic_type="single_layer")
    push_pull = simulate_osmotic_delivery("DrugA", 100.0, lag_h=1.0, osmotic_type="push_pull")
    # Push-pull has lag += 1.0 so peak should be later
    assert push_pull.tmax_h >= single.tmax_h


def test_dual_layer_returns_result():
    result = simulate_osmotic_delivery("DrugA", 100.0, osmotic_type="dual_layer")
    assert isinstance(result, OsmoticDrugDeliveryResult)
    assert result.osmotic_type == "dual_layer"


def test_push_pull_osmotic_type():
    result = simulate_osmotic_delivery("DrugA", 100.0, osmotic_type="push_pull")
    assert result.osmotic_type == "push_pull"
    # Push-pull adds lag
    assert result.lag_h == 1.0  # stored original lag
    assert result.cmax_mg_L > 0.0


# --- t_half_effective ---


def test_t_half_effective_before_end():
    result = simulate_osmotic_delivery("DrugA", 100.0, t_end_h=24.0)
    # 50% of dose should be released well before 24h
    assert result.t_half_effective_h < 24.0


def test_t_half_effective_after_lag():
    result = simulate_osmotic_delivery("DrugA", 100.0, lag_h=2.0)
    # t_half must be after lag since no release before lag
    assert result.t_half_effective_h >= 2.0


# --- Notes ---


def test_notes_is_string():
    result = simulate_osmotic_delivery("DrugA", 100.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# --- compare_osmotic_designs ---


def test_compare_osmotic_designs_returns_list():
    designs = [
        {"osmotic_type": "single_layer", "t_release_h": 8.0, "lag_h": 1.0},
        {"osmotic_type": "dual_layer", "t_release_h": 12.0, "lag_h": 1.0},
        {"osmotic_type": "push_pull", "t_release_h": 8.0, "lag_h": 1.0},
    ]
    results = compare_osmotic_designs("DrugA", 100.0, designs)
    assert len(results) == 3


def test_compare_osmotic_designs_sorted_by_auc_descending():
    designs = [
        {"osmotic_type": "single_layer", "t_release_h": 4.0, "lag_h": 0.5},
        {"osmotic_type": "single_layer", "t_release_h": 12.0, "lag_h": 2.0},
        {"osmotic_type": "push_pull", "t_release_h": 8.0, "lag_h": 1.0},
    ]
    results = compare_osmotic_designs("DrugA", 100.0, designs)
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_osmotic_designs_single_design():
    designs = [{"osmotic_type": "single_layer", "t_release_h": 8.0, "lag_h": 1.0}]
    results = compare_osmotic_designs("DrugA", 100.0, designs)
    assert len(results) == 1


# --- Validation ---


def test_validation_negative_dose():
    with pytest.raises(ValueError, match="total_dose_mg must be > 0"):
        simulate_osmotic_delivery("DrugA", -10.0)


def test_validation_zero_dose():
    with pytest.raises(ValueError, match="total_dose_mg must be > 0"):
        simulate_osmotic_delivery("DrugA", 0.0)


def test_validation_negative_t_release():
    with pytest.raises(ValueError, match="t_release_h must be > 0"):
        simulate_osmotic_delivery("DrugA", 100.0, t_release_h=-1.0)


def test_validation_negative_lag():
    with pytest.raises(ValueError, match="lag_h must be >= 0"):
        simulate_osmotic_delivery("DrugA", 100.0, lag_h=-0.5)


def test_validation_invalid_osmotic_type():
    with pytest.raises(ValueError, match="osmotic_type"):
        simulate_osmotic_delivery("DrugA", 100.0, osmotic_type="unknown_type")


def test_validation_zero_vd():
    with pytest.raises(ValueError, match="vd_L must be > 0"):
        simulate_osmotic_delivery("DrugA", 100.0, vd_L=0.0)
