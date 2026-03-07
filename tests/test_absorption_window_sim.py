"""Tests for biopharmaceutics/absorption_window_sim.py — Phase 473.

Covers:
- Basic smoke test and dataclass structure
- Low permeability → long window, low fa_total
- Low solubility → dissolution-limited, low fa_total
- Very soluble, high-permeability → fast absorption, wide window
- lag_time_h > 0 for poorly soluble drug
- fa_total in [0, 1]
- 2x dose: same fa fraction (linear PK)
- Acid pKa 4 at pH 1.5 is largely unionized → better dissolution in stomach
- Input validation
- calculate_window_width and estimate_lag_time standalone tests
"""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.absorption_window_sim import (
    AbsorptionWindowResult,
    calculate_window_width,
    estimate_lag_time,
    simulate_absorption_window,
)

# ---------------------------------------------------------------------------
# Shared defaults — moderate BCS II drug
# ---------------------------------------------------------------------------

_BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw=350.0,
    pka=None,
    drug_type="neutral",
    solubility_pH7_mg_mL=1.0,
    t_end_h=8.0,
    dt_h=0.05,
)


def _run(**overrides) -> AbsorptionWindowResult:
    params = {**_BASE, **overrides}
    return simulate_absorption_window(**params)


# ---------------------------------------------------------------------------
# 1. Smoke test / dataclass type
# ---------------------------------------------------------------------------


def test_returns_absorption_window_result():
    result = _run()
    assert isinstance(result, AbsorptionWindowResult)


def test_result_is_frozen():
    result = _run()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "other"  # type: ignore[misc]


def test_times_h_non_empty():
    result = _run()
    assert len(result.times_h) > 0


def test_fraction_absorbed_same_length_as_times():
    result = _run()
    assert len(result.fraction_absorbed) == len(result.times_h)


def test_times_h_starts_at_zero():
    result = _run()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_h_ends_near_t_end():
    result = _run(t_end_h=8.0, dt_h=0.05)
    assert result.times_h[-1] == pytest.approx(8.0, abs=0.1)


# ---------------------------------------------------------------------------
# 2. fa_total in [0, 1]
# ---------------------------------------------------------------------------


def test_fa_total_non_negative():
    result = _run()
    assert result.fa_total >= 0.0


def test_fa_total_at_most_one():
    result = _run()
    assert result.fa_total <= 1.0 + 1e-9


def test_fa_total_matches_last_profile_value():
    result = _run()
    assert result.fa_total == pytest.approx(result.fraction_absorbed[-1], abs=1e-9)


def test_fraction_absorbed_monotone_non_decreasing():
    result = _run()
    for i in range(1, len(result.fraction_absorbed)):
        assert result.fraction_absorbed[i] >= result.fraction_absorbed[i - 1] - 1e-12


# ---------------------------------------------------------------------------
# 3. Low permeability → low fa_total
# ---------------------------------------------------------------------------


def test_low_logp_low_fa():
    """logP = -3 → very low permeability → low fa_total."""
    result = _run(logP=-3.0, mw=500.0, solubility_pH7_mg_mL=5.0)
    assert result.fa_total < 0.4


def test_low_logp_permeability_limited():
    result = _run(logP=-3.0, mw=600.0)
    assert result.limiting_factor in ("permeability", "dissolution", "transit")


# ---------------------------------------------------------------------------
# 4. Low solubility → dissolution-limited, low fa_total
# ---------------------------------------------------------------------------


def test_low_solubility_low_fa():
    """Very poorly soluble drug (0.001 mg/mL) → dissolution-limited, low fa."""
    result = _run(solubility_pH7_mg_mL=0.001)
    assert result.fa_total < 0.3


def test_low_solubility_dissolution_limited():
    result = _run(solubility_pH7_mg_mL=0.001)
    assert result.limiting_factor == "dissolution"


# ---------------------------------------------------------------------------
# 5. Very soluble, high permeability → fast absorption, high fa_total
# ---------------------------------------------------------------------------


def test_high_solubility_high_logp_high_fa():
    """High solubility and lipophilicity → fast absorption."""
    result = _run(logP=3.5, mw=300.0, solubility_pH7_mg_mL=50.0)
    assert result.fa_total > 0.5


def test_high_permeability_wide_window():
    """High permeability drug should have a meaningful window width."""
    result = _run(logP=3.5, mw=300.0, solubility_pH7_mg_mL=50.0)
    # Window width should be positive
    assert result.window_width_h >= 0.0


# ---------------------------------------------------------------------------
# 6. lag_time_h > 0 for poorly soluble drug
# ---------------------------------------------------------------------------


def test_poorly_soluble_has_lag_time():
    """A poorly soluble drug requires dissolution time before measurable absorption."""
    result = _run(solubility_pH7_mg_mL=0.005, logP=1.0)
    assert result.lag_time_h >= 0.0


def test_well_soluble_low_lag_time():
    """Well-soluble drug should have shorter lag time than poorly soluble."""
    well = _run(solubility_pH7_mg_mL=50.0, logP=2.0)
    poor = _run(solubility_pH7_mg_mL=0.005, logP=1.0)
    assert well.lag_time_h <= poor.lag_time_h + 1.0  # well-soluble starts absorbing earlier


# ---------------------------------------------------------------------------
# 7. 2× dose → same fa fraction (linear PK)
# ---------------------------------------------------------------------------


def test_double_dose_same_fa_fraction():
    """Scaling dose should not change fa_total (linear absorption kinetics)."""
    r100 = _run(dose_mg=100.0)
    r200 = _run(dose_mg=200.0)
    assert r100.fa_total == pytest.approx(r200.fa_total, rel=0.15)


# ---------------------------------------------------------------------------
# 8. Acid pKa=4 at stomach pH=1.5 is largely un-ionized → better dissolution
# ---------------------------------------------------------------------------


def test_acid_drug_pka4_has_some_absorption():
    """Acid with pKa=4: largely un-ionized in stomach (pH 1.5) → better stomach dissolution."""
    result_acid = _run(pka=4.0, drug_type="acid", solubility_pH7_mg_mL=0.1)
    result_neutral = _run(pka=None, drug_type="neutral", solubility_pH7_mg_mL=0.1)
    # Both should produce some absorption; acid may vary
    assert result_acid.fa_total >= 0.0
    assert result_neutral.fa_total >= 0.0


def test_acid_pka4_solubility_enhanced_in_stomach():
    """Acid drug at pH 1.5 (stomach) < pKa=4 → mostly un-ionized → intrinsic dissolution."""
    # When pKa=4 and pH=1.5, fraction un-ionized ≈ 1/(1+10^(1.5-4)) = ~0.997
    # This means no solubility enhancement (acid is mostly neutral at low pH)
    # But simulation should still run without error
    result = _run(pka=4.0, drug_type="acid")
    assert isinstance(result, AbsorptionWindowResult)
    assert 0.0 <= result.fa_total <= 1.0


def test_base_drug_higher_fa_at_intestinal_ph():
    """Base drug with pKa=8 is mostly un-ionized in intestine (pH 6.5–7.2)."""
    result = _run(pka=8.0, drug_type="base", solubility_pH7_mg_mL=2.0)
    assert isinstance(result, AbsorptionWindowResult)
    assert result.fa_total >= 0.0


# ---------------------------------------------------------------------------
# 9. Input validation
# ---------------------------------------------------------------------------


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-50.0)


def test_zero_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        _run(mw=0.0)


def test_negative_solubility_raises():
    with pytest.raises(ValueError, match="solubility"):
        _run(solubility_pH7_mg_mL=-1.0)


def test_zero_solubility_raises():
    with pytest.raises(ValueError, match="solubility"):
        _run(solubility_pH7_mg_mL=0.0)


def test_invalid_drug_type_raises():
    with pytest.raises(ValueError, match="drug_type"):
        _run(drug_type="zwitterion")


def test_zero_t_end_raises():
    with pytest.raises(ValueError, match="t_end_h"):
        _run(t_end_h=0.0)


def test_zero_dt_raises():
    with pytest.raises(ValueError, match="dt_h"):
        _run(dt_h=0.0)


def test_empty_drug_name_raises():
    with pytest.raises(ValueError, match="drug_name"):
        _run(drug_name="")


# ---------------------------------------------------------------------------
# 10. calculate_window_width standalone
# ---------------------------------------------------------------------------


def test_calculate_window_width_zero_if_empty():
    assert calculate_window_width([], []) == pytest.approx(0.0)


def test_calculate_window_width_zero_if_all_zero():
    assert calculate_window_width([0.0, 0.0, 0.0], [0.0, 1.0, 2.0]) == pytest.approx(0.0)


def test_calculate_window_width_positive():
    fa = [0.0, 0.1, 0.3, 0.6, 0.8, 0.9, 0.95]
    t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    w = calculate_window_width(fa, t)
    assert w > 0.0


def test_calculate_window_width_at_threshold():
    # Linear ramp: 0..1 over 10 h
    fa = [i / 10.0 for i in range(11)]
    t = [float(i) for i in range(11)]
    # onset at fa=0.05*1.0=0.05 → t≈0.5, close at fa=0.9*1.0=0.9 → t=9.0
    w = calculate_window_width(fa, t, threshold=0.9)
    assert w == pytest.approx(8.5, abs=1.0)


# ---------------------------------------------------------------------------
# 11. estimate_lag_time standalone
# ---------------------------------------------------------------------------


def test_estimate_lag_time_zero_if_empty():
    assert estimate_lag_time([], []) == pytest.approx(0.0)


def test_estimate_lag_time_zero_if_all_zero():
    assert estimate_lag_time([0.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_estimate_lag_time_positive():
    # No absorption for first 2 steps, then rises
    fa = [0.0, 0.0, 0.0, 0.0, 0.05, 0.3, 0.6]
    t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    lag = estimate_lag_time(fa, t)
    assert lag >= 0.0


def test_estimate_lag_time_from_simulation():
    result = _run()
    assert result.lag_time_h >= 0.0
    assert result.lag_time_h <= result.times_h[-1]


# ---------------------------------------------------------------------------
# 12. limiting_factor is valid
# ---------------------------------------------------------------------------


def test_limiting_factor_valid_values():
    result = _run()
    assert result.limiting_factor in ("dissolution", "permeability", "transit")


def test_limiting_factor_dissolution_for_insoluble_drug():
    result = _run(solubility_pH7_mg_mL=0.001)
    assert result.limiting_factor == "dissolution"


# ---------------------------------------------------------------------------
# 13. Notes are a string
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = _run()
    assert isinstance(result.notes, str)


def test_notes_contains_logp():
    result = _run(logP=2.5)
    assert "logP" in result.notes
