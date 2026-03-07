"""Tests for Phase 366: Particle Size Distribution Effect on Dissolution."""

import pytest

from omega_pbpk.prediction.particle_size_distribution_dissolution import (
    PolydisperseDissoResult,
    simulate_polydisperse_dissolution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        d50_um=50.0,
        span=1.5,
        intrinsic_solubility_mg_mL=5.0,
        dose_mg=100.0,
        drug_density_g_cm3=1.2,
        t_end_min=60.0,
        dt_min=0.5,
    )
    params.update(kwargs)
    return simulate_polydisperse_dissolution(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = default_result()
    assert isinstance(result, PolydisperseDissoResult)


# ---------------------------------------------------------------------------
# Array consistency
# ---------------------------------------------------------------------------


def test_times_and_dissolved_pct_same_length():
    result = default_result()
    assert len(result.times_min) == len(result.dissolved_pct)


def test_times_min_starts_at_zero():
    result = default_result()
    assert result.times_min[0] == pytest.approx(0.0)


def test_array_nonempty():
    result = default_result()
    assert len(result.times_min) > 1


# ---------------------------------------------------------------------------
# dissolved_pct properties
# ---------------------------------------------------------------------------


def test_dissolved_pct_non_decreasing():
    result = default_result()
    for i in range(1, len(result.dissolved_pct)):
        assert result.dissolved_pct[i] >= result.dissolved_pct[i - 1] - 1e-9


def test_dissolved_pct_in_range():
    result = default_result()
    for p in result.dissolved_pct:
        assert 0.0 <= p <= 100.0 + 1e-9


def test_dissolved_pct_starts_at_zero():
    result = default_result()
    assert result.dissolved_pct[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# t50, t90 properties
# ---------------------------------------------------------------------------


def test_t50_positive_or_inf():
    result = default_result()
    if result.t50_min != float("inf"):
        assert result.t50_min > 0.0


def test_t90_positive_or_inf():
    result = default_result()
    if result.t90_min != float("inf"):
        assert result.t90_min > 0.0


def test_t50_leq_t90():
    result = default_result()
    assert result.t50_min <= result.t90_min


def test_t50_reached_high_solubility():
    result = simulate_polydisperse_dissolution(
        drug_name="Soluble",
        d50_um=10.0,
        span=1.0,
        intrinsic_solubility_mg_mL=50.0,
        dose_mg=50.0,
        t_end_min=60.0,
        dt_min=0.25,
    )
    assert result.t50_min < float("inf")


# ---------------------------------------------------------------------------
# Polydisperse vs monodisperse
# ---------------------------------------------------------------------------


def test_polydisperse_faster_attribute_is_bool():
    result = default_result()
    assert isinstance(result.polydisperse_faster, bool)


def test_polydisperse_faster_with_high_span():
    """High span introduces fine particles → faster than monodisperse."""
    result = simulate_polydisperse_dissolution(
        drug_name="Drug",
        d50_um=100.0,
        span=3.0,
        intrinsic_solubility_mg_mL=10.0,
        dose_mg=200.0,
        t_end_min=120.0,
        dt_min=0.5,
    )
    # With span=3 the fines are very small; polydisperse is often faster
    # We just check the flag is consistent with the t90 comparison
    if result.t90_min < result.monodisperse_t90_min:
        assert result.polydisperse_faster is True
    else:
        assert result.polydisperse_faster is False


# ---------------------------------------------------------------------------
# Smaller d50 → faster dissolution
# ---------------------------------------------------------------------------


def test_smaller_d50_faster_dissolution():
    r_small = default_result(d50_um=10.0, t_end_min=30.0)
    r_large = default_result(d50_um=100.0, t_end_min=30.0)
    # Smaller d50 should dissolve more within the same time window
    assert r_small.max_dissolved_pct >= r_large.max_dissolved_pct


def test_smaller_d50_earlier_t50():
    r_small = simulate_polydisperse_dissolution(
        drug_name="Drug",
        d50_um=5.0,
        span=1.0,
        intrinsic_solubility_mg_mL=10.0,
        dose_mg=50.0,
        t_end_min=60.0,
        dt_min=0.25,
    )
    r_large = simulate_polydisperse_dissolution(
        drug_name="Drug",
        d50_um=200.0,
        span=1.0,
        intrinsic_solubility_mg_mL=10.0,
        dose_mg=50.0,
        t_end_min=60.0,
        dt_min=0.25,
    )
    assert r_small.t50_min <= r_large.t50_min


# ---------------------------------------------------------------------------
# Higher span → more fines → faster dissolution
# ---------------------------------------------------------------------------


def test_higher_span_more_dissolved():
    r_narrow = simulate_polydisperse_dissolution(
        drug_name="Drug",
        d50_um=50.0,
        span=0.5,
        intrinsic_solubility_mg_mL=5.0,
        dose_mg=100.0,
        t_end_min=30.0,
        dt_min=0.5,
    )
    r_wide = simulate_polydisperse_dissolution(
        drug_name="Drug",
        d50_um=50.0,
        span=3.0,
        intrinsic_solubility_mg_mL=5.0,
        dose_mg=100.0,
        t_end_min=30.0,
        dt_min=0.5,
    )
    # Wider span has more fines so should dissolve >= narrow span
    assert r_wide.max_dissolved_pct >= r_narrow.max_dissolved_pct


# ---------------------------------------------------------------------------
# Dissolution efficiency
# ---------------------------------------------------------------------------


def test_dissolution_efficiency_in_range():
    result = default_result()
    assert 0.0 <= result.dissolution_efficiency_pct <= 100.0


def test_dissolution_efficiency_positive():
    result = simulate_polydisperse_dissolution(
        drug_name="Drug",
        d50_um=5.0,
        span=1.0,
        intrinsic_solubility_mg_mL=50.0,
        dose_mg=50.0,
        t_end_min=30.0,
        dt_min=0.5,
    )
    assert result.dissolution_efficiency_pct > 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_error_d50_zero():
    with pytest.raises(ValueError, match="d50_um"):
        simulate_polydisperse_dissolution(
            drug_name="X", d50_um=0.0, intrinsic_solubility_mg_mL=1.0, dose_mg=100.0
        )


def test_error_d50_negative():
    with pytest.raises(ValueError, match="d50_um"):
        simulate_polydisperse_dissolution(
            drug_name="X", d50_um=-5.0, intrinsic_solubility_mg_mL=1.0, dose_mg=100.0
        )


def test_error_span_zero():
    with pytest.raises(ValueError, match="span"):
        simulate_polydisperse_dissolution(
            drug_name="X", d50_um=50.0, span=0.0, intrinsic_solubility_mg_mL=1.0, dose_mg=100.0
        )


def test_error_solubility_zero():
    with pytest.raises(ValueError, match="intrinsic_solubility_mg_mL"):
        simulate_polydisperse_dissolution(
            drug_name="X", d50_um=50.0, intrinsic_solubility_mg_mL=0.0, dose_mg=100.0
        )


def test_error_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_polydisperse_dissolution(
            drug_name="X", d50_um=50.0, intrinsic_solubility_mg_mL=1.0, dose_mg=0.0
        )


def test_error_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_polydisperse_dissolution(
            drug_name="X", d50_um=50.0, intrinsic_solubility_mg_mL=1.0, dose_mg=-100.0
        )
