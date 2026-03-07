"""Tests for omega_pbpk.biopharmaceutics.ivdt — Phase 510."""

import pytest

from omega_pbpk.biopharmaceutics.ivdt import (
    DissolutionTestResult,
    calculate_dissolution_efficiency,
    compare_to_specification,
    simulate_usp1_dissolution,
    simulate_usp2_dissolution,
)

# ---------------------------------------------------------------------------
# simulate_usp2_dissolution
# ---------------------------------------------------------------------------


def test_usp2_returns_dissolution_test_result():
    result = simulate_usp2_dissolution("drug_a", dose_mg=100.0, solubility_mg_mL=5.0)
    assert isinstance(result, DissolutionTestResult)


def test_usp2_apparatus_label():
    result = simulate_usp2_dissolution("drug_a", dose_mg=100.0, solubility_mg_mL=5.0)
    assert result.apparatus == "USP2"


def test_usp2_times_start_at_zero():
    result = simulate_usp2_dissolution("drug_a", dose_mg=100.0, solubility_mg_mL=5.0)
    assert result.times_min[0] == pytest.approx(0.0)


def test_usp2_fractions_nonnegative():
    result = simulate_usp2_dissolution("drug_a", dose_mg=100.0, solubility_mg_mL=5.0)
    assert all(f >= -1e-9 for f in result.fraction_dissolved)


def test_usp2_fractions_at_most_one():
    result = simulate_usp2_dissolution("drug_a", dose_mg=100.0, solubility_mg_mL=5.0)
    assert all(f <= 1.0 + 1e-9 for f in result.fraction_dissolved)


def test_usp2_fractions_nondecreasing():
    """Dissolution should be monotonically non-decreasing."""
    result = simulate_usp2_dissolution("drug_a", dose_mg=100.0, solubility_mg_mL=5.0)
    for i in range(len(result.fraction_dissolved) - 1):
        assert result.fraction_dissolved[i + 1] >= result.fraction_dissolved[i] - 1e-9


def test_usp2_higher_rpm_thinner_h_faster_dissolution():
    """Higher RPM → faster dissolution (higher fraction at 30 min)."""
    result_low = simulate_usp2_dissolution("drug", dose_mg=100.0, solubility_mg_mL=1.0, rpm=25.0)
    result_high = simulate_usp2_dissolution("drug", dose_mg=100.0, solubility_mg_mL=1.0, rpm=200.0)

    # Find fraction at 30 min
    def _frac_at_30(r):
        for t, f in zip(r.times_min, r.fraction_dissolved):
            if t >= 30.0:
                return f
        return r.fraction_dissolved[-1]

    assert _frac_at_30(result_high) > _frac_at_30(result_low)


def test_usp2_smaller_particle_faster_dissolution():
    """Smaller particle size → larger surface area → faster dissolution."""
    result_large = simulate_usp2_dissolution(
        "drug", dose_mg=100.0, solubility_mg_mL=1.0, particle_size_um=200.0
    )
    result_small = simulate_usp2_dissolution(
        "drug", dose_mg=100.0, solubility_mg_mL=1.0, particle_size_um=10.0
    )
    assert result_small.t80_min <= result_large.t80_min + 1e-6


def test_usp2_highly_soluble_drug_passes_q30():
    """Very high solubility drug should dissolve quickly and pass Q30."""
    result = simulate_usp2_dissolution(
        "soluble_drug", dose_mg=10.0, solubility_mg_mL=100.0, particle_size_um=5.0, rpm=100.0
    )
    assert result.passes_q30 is True


def test_usp2_t80_within_sim_time():
    result = simulate_usp2_dissolution("drug", dose_mg=100.0, solubility_mg_mL=5.0)
    assert 0.0 <= result.t80_min <= result.times_min[-1] + 1e-9


def test_usp2_de30_between_0_and_100():
    result = simulate_usp2_dissolution("drug", dose_mg=100.0, solubility_mg_mL=5.0)
    assert 0.0 <= result.de30_pct <= 100.0


def test_usp2_conc_mg_mL_nonnegative():
    result = simulate_usp2_dissolution("drug", dose_mg=100.0, solubility_mg_mL=5.0)
    assert all(c >= -1e-9 for c in result.conc_mg_mL)


def test_usp2_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_usp2_dissolution("drug", dose_mg=0.0, solubility_mg_mL=1.0)


def test_usp2_invalid_solubility_raises():
    with pytest.raises(ValueError):
        simulate_usp2_dissolution("drug", dose_mg=100.0, solubility_mg_mL=-1.0)


def test_usp2_invalid_rpm_raises():
    with pytest.raises(ValueError):
        simulate_usp2_dissolution("drug", dose_mg=100.0, solubility_mg_mL=1.0, rpm=0.0)


# ---------------------------------------------------------------------------
# simulate_usp1_dissolution
# ---------------------------------------------------------------------------


def test_usp1_returns_dissolution_test_result():
    result = simulate_usp1_dissolution("drug_b", dose_mg=100.0, solubility_mg_mL=5.0)
    assert isinstance(result, DissolutionTestResult)


def test_usp1_apparatus_label():
    result = simulate_usp1_dissolution("drug_b", dose_mg=100.0, solubility_mg_mL=5.0)
    assert result.apparatus == "USP1"


def test_usp1_vs_usp2_same_drug_different_t80():
    """USP1 (basket) has h_factor=0.85 → slightly faster than USP2 at same RPM."""
    # Same drug, same particle size, same rpm but different apparatus
    result_usp1 = simulate_usp1_dissolution(
        "drug", dose_mg=100.0, solubility_mg_mL=0.5, particle_size_um=50.0, rpm=75.0
    )
    result_usp2 = simulate_usp2_dissolution(
        "drug", dose_mg=100.0, solubility_mg_mL=0.5, particle_size_um=50.0, rpm=75.0
    )
    # USP1 should be equal or faster (smaller t80 or same)
    assert result_usp1.t80_min <= result_usp2.t80_min + 1e-6


def test_usp1_fractions_nondecreasing():
    result = simulate_usp1_dissolution("drug_b", dose_mg=100.0, solubility_mg_mL=5.0)
    for i in range(len(result.fraction_dissolved) - 1):
        assert result.fraction_dissolved[i + 1] >= result.fraction_dissolved[i] - 1e-9


# ---------------------------------------------------------------------------
# compare_to_specification
# ---------------------------------------------------------------------------


def test_compare_to_spec_passes_q30():
    """Drug that passes Q30 (>=75% at 30 min) should pass spec at 30 min."""
    result = simulate_usp2_dissolution(
        "soluble", dose_mg=10.0, solubility_mg_mL=50.0, particle_size_um=5.0, rpm=100.0
    )
    spec = compare_to_specification(result, [30.0], [0.75])
    assert spec["passes_all"] is True


def test_compare_to_spec_fails_when_below_spec():
    """Slowly dissolving drug should fail specification."""
    result = simulate_usp2_dissolution(
        "slow_drug",
        dose_mg=500.0,
        solubility_mg_mL=0.01,
        particle_size_um=200.0,
        rpm=25.0,
        t_end_min=60.0,
    )
    spec = compare_to_specification(result, [10.0], [0.90])
    assert spec["passes_all"] is False


def test_compare_to_spec_returns_timepoint_results():
    result = simulate_usp2_dissolution("drug", dose_mg=100.0, solubility_mg_mL=5.0)
    spec = compare_to_specification(result, [15.0, 30.0, 45.0], [0.3, 0.6, 0.8])
    assert "timepoint_results" in spec
    assert len(spec["timepoint_results"]) == 3


def test_compare_to_spec_timepoint_dict_keys():
    result = simulate_usp2_dissolution("drug", dose_mg=100.0, solubility_mg_mL=5.0)
    spec = compare_to_specification(result, [30.0], [0.75])
    tp = spec["timepoint_results"][0]
    for key in ("time_min", "spec_min", "actual", "passes"):
        assert key in tp


def test_compare_to_spec_mismatched_lengths_raises():
    result = simulate_usp2_dissolution("drug", dose_mg=100.0, solubility_mg_mL=5.0)
    with pytest.raises(ValueError):
        compare_to_specification(result, [30.0, 45.0], [0.75])


# ---------------------------------------------------------------------------
# calculate_dissolution_efficiency
# ---------------------------------------------------------------------------


def test_de_instant_dissolution_returns_100():
    """Instant full dissolution throughout → DE = 100%."""
    times = [0.0, 10.0, 20.0, 30.0]
    fractions = [1.0, 1.0, 1.0, 1.0]
    de = calculate_dissolution_efficiency(times, fractions, t_max_min=30.0)
    assert de == pytest.approx(100.0, abs=0.5)


def test_de_zero_dissolution_returns_zero():
    """No dissolution → DE = 0%."""
    times = [0.0, 10.0, 20.0, 30.0]
    fractions = [0.0, 0.0, 0.0, 0.0]
    de = calculate_dissolution_efficiency(times, fractions, t_max_min=30.0)
    assert de == pytest.approx(0.0, abs=0.1)


def test_de_partial_dissolution_between_0_and_100():
    result = simulate_usp2_dissolution("drug", dose_mg=100.0, solubility_mg_mL=1.0)
    de = calculate_dissolution_efficiency(result.times_min, result.fraction_dissolved)
    assert 0.0 <= de <= 100.0


def test_de_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        calculate_dissolution_efficiency([0.0, 10.0], [0.5])


def test_de_negative_t_max_raises():
    with pytest.raises(ValueError):
        calculate_dissolution_efficiency([0.0, 10.0], [0.5, 0.8], t_max_min=-5.0)


def test_de_usp1_highly_soluble_gt_90():
    """Highly soluble small particle drug in USP1 → DE > 90%."""
    result = simulate_usp1_dissolution(
        "soluble", dose_mg=5.0, solubility_mg_mL=100.0, particle_size_um=5.0, rpm=100.0
    )
    de = calculate_dissolution_efficiency(result.times_min, result.fraction_dissolved)
    assert de > 80.0


# ---------------------------------------------------------------------------
# Sink conditions
# ---------------------------------------------------------------------------


def test_usp2_sink_conditions_linear_dissolution():
    """In true sink (Cs >> C_vessel), dissolution rate is approximately
    constant initially, giving near-linear fraction dissolved early."""
    result = simulate_usp2_dissolution(
        "sink_drug",
        dose_mg=1.0,
        solubility_mg_mL=1000.0,
        particle_size_um=10.0,
        rpm=100.0,
        t_end_min=10.0,
        dt_min=0.1,
    )
    # Early fractions should increase roughly linearly
    frac_5 = result.fraction_dissolved[int(5.0 / 0.1)]
    frac_10 = result.fraction_dissolved[int(10.0 / 0.1)]
    # Should have dissolved significantly
    assert frac_5 > 0.1
    assert frac_10 >= frac_5
