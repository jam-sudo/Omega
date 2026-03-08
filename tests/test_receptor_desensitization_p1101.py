"""Tests for Phase 1101 — Receptor Desensitization Kinetics."""

import pytest

from omega_pbpk.core.receptor_desensitization_p1101 import (
    ReceptorDesensitizationResult,
    compare_dosing_intervals,
    simulate_receptor_desensitization,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_result():
    return simulate_receptor_desensitization(
        drug_name="DrugA",
        dose_mg=10.0,
        tau_h=8.0,
        n_doses=5,
    )


@pytest.fixture
def short_interval_result():
    return simulate_receptor_desensitization(
        drug_name="DrugA",
        dose_mg=10.0,
        tau_h=2.0,
        n_doses=5,
    )


@pytest.fixture
def long_interval_result():
    return simulate_receptor_desensitization(
        drug_name="DrugA",
        dose_mg=10.0,
        tau_h=24.0,
        n_doses=5,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type(base_result):
    assert isinstance(base_result, ReceptorDesensitizationResult)


def test_return_type_fields(base_result):
    assert hasattr(base_result, "drug_name")
    assert hasattr(base_result, "tachyphylaxis_ratio")
    assert hasattr(base_result, "recovery_time_h")


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_ra_fraction_starts_at_one(base_result):
    assert abs(base_result.ra_fraction[0] - 1.0) < 1e-9


def test_rd_fraction_starts_at_zero(base_result):
    assert abs(base_result.rd_fraction[0] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Mass conservation: Ra + Rd <= 1
# ---------------------------------------------------------------------------


def test_receptor_fractions_sum_le_one(base_result):
    for i, (ra, rd) in enumerate(zip(base_result.ra_fraction, base_result.rd_fraction)):
        total = ra + rd
        assert total <= 1.0 + 1e-9, f"Ra+Rd={total:.6f} > 1.0 at index {i}"


# ---------------------------------------------------------------------------
# Tachyphylaxis
# ---------------------------------------------------------------------------


def test_tachyphylaxis_ratio_le_one(base_result):
    assert base_result.tachyphylaxis_ratio <= 1.0 + 1e-6


def test_peak_effect_doseN_le_dose1(base_result):
    assert base_result.peak_effect_doseN <= base_result.peak_effect_dose1 + 1e-6


def test_tachyphylaxis_ratio_positive(base_result):
    assert base_result.tachyphylaxis_ratio >= 0.0


# ---------------------------------------------------------------------------
# Effect bounds
# ---------------------------------------------------------------------------


def test_effect_pct_in_valid_range(base_result):
    for eff in base_result.effect_pct:
        assert 0.0 <= eff <= 100.0, f"effect_pct={eff} out of [0,100]"


def test_ra_fraction_in_valid_range(base_result):
    for ra in base_result.ra_fraction:
        assert 0.0 <= ra <= 1.0 + 1e-9, f"ra_fraction={ra} out of [0,1]"


def test_rd_fraction_in_valid_range(base_result):
    for rd in base_result.rd_fraction:
        assert 0.0 <= rd <= 1.0 + 1e-9, f"rd_fraction={rd} out of [0,1]"


# ---------------------------------------------------------------------------
# Recovery time
# ---------------------------------------------------------------------------


def test_recovery_time_positive(base_result):
    assert base_result.recovery_time_h >= 0.0


def test_recovery_time_is_float(base_result):
    assert isinstance(base_result.recovery_time_h, float)


# ---------------------------------------------------------------------------
# Dosing interval effect on tachyphylaxis
# ---------------------------------------------------------------------------


def test_longer_tau_less_tachyphylaxis(short_interval_result, long_interval_result):
    assert (
        long_interval_result.tachyphylaxis_ratio >= short_interval_result.tachyphylaxis_ratio - 1e-6
    )


# ---------------------------------------------------------------------------
# compare_dosing_intervals
# ---------------------------------------------------------------------------


def test_compare_dosing_intervals_count():
    tau_list = [4.0, 8.0, 12.0, 24.0]
    results = compare_dosing_intervals("DrugA", 10.0, tau_list)
    assert len(results) == len(tau_list)


def test_compare_dosing_intervals_sorted_descending():
    tau_list = [2.0, 8.0, 24.0]
    results = compare_dosing_intervals("DrugA", 10.0, tau_list)
    ratios = [r.tachyphylaxis_ratio for r in results]
    for i in range(len(ratios) - 1):
        assert ratios[i] >= ratios[i + 1] - 1e-9, f"Not sorted descending: {ratios}"


def test_compare_dosing_intervals_returns_list():
    results = compare_dosing_intervals("DrugA", 10.0, [4.0, 8.0])
    assert isinstance(results, list)


def test_compare_dosing_intervals_correct_type():
    results = compare_dosing_intervals("DrugA", 10.0, [4.0, 8.0])
    assert all(isinstance(r, ReceptorDesensitizationResult) for r in results)


# ---------------------------------------------------------------------------
# Array lengths
# ---------------------------------------------------------------------------


def test_array_lengths_consistent(base_result):
    n = len(base_result.times_h)
    assert len(base_result.c_drug_mg_L) == n
    assert len(base_result.ra_fraction) == n
    assert len(base_result.rd_fraction) == n
    assert len(base_result.effect_pct) == n


# ---------------------------------------------------------------------------
# Parameter storage
# ---------------------------------------------------------------------------


def test_result_stores_drug_name(base_result):
    assert base_result.drug_name == "DrugA"


def test_result_stores_dose(base_result):
    assert abs(base_result.dose_mg - 10.0) < 1e-9


def test_result_stores_tau_h(base_result):
    assert abs(base_result.tau_h - 8.0) < 1e-9


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_negative_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_receptor_desensitization("DrugA", dose_mg=-1.0)


def test_validation_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_receptor_desensitization("DrugA", dose_mg=0.0)


def test_validation_negative_tau():
    with pytest.raises(ValueError, match="tau_h"):
        simulate_receptor_desensitization("DrugA", dose_mg=10.0, tau_h=-1.0)


def test_validation_zero_n_doses():
    with pytest.raises(ValueError, match="n_doses"):
        simulate_receptor_desensitization("DrugA", dose_mg=10.0, n_doses=0)


def test_validation_negative_ke():
    with pytest.raises(ValueError, match="ke_drug"):
        simulate_receptor_desensitization("DrugA", dose_mg=10.0, ke_drug=-0.1)


def test_validation_negative_vd():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_receptor_desensitization("DrugA", dose_mg=10.0, vd_L=-50.0)


def test_validation_negative_ec50():
    with pytest.raises(ValueError, match="ec50"):
        simulate_receptor_desensitization("DrugA", dose_mg=10.0, ec50=-0.1)


def test_validation_negative_emax():
    with pytest.raises(ValueError, match="emax"):
        simulate_receptor_desensitization("DrugA", dose_mg=10.0, emax=-100.0)
