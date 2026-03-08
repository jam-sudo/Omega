"""Tests for pancreatic drug distribution PK model (Phase 496)."""

import pytest

from omega_pbpk.core.pancreatic_drug_distribution_p496 import (
    PancreaticDistributionResult,
    _classify_therapeutic_adequacy,
    _compute_kp_pancreas,
    screen_antibiotics,
    simulate_pancreatic_distribution,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_result():
    return simulate_pancreatic_distribution(
        drug_name="TestAntibiotic",
        dose_mg=500.0,
        logP=1.5,
        cl_sys_L_per_h=8.0,
        vd_plasma_L=10.0,
        vd_panc_L=1.0,
        Q_panc_L_per_h=1.2,
        t_end_h=24.0,
        dt_h=0.1,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type(base_result):
    assert isinstance(base_result, PancreaticDistributionResult)


# ---------------------------------------------------------------------------
# Kp calculation
# ---------------------------------------------------------------------------


def test_kp_positive_logp():
    kp = _compute_kp_pancreas(2.0)
    assert abs(kp - (0.5 + 0.2 * 2.0)) < 1e-9


def test_kp_zero_logp():
    kp = _compute_kp_pancreas(0.0)
    assert abs(kp - 0.5) < 1e-9


def test_kp_negative_logp_clamps_to_0_5():
    kp = _compute_kp_pancreas(-3.0)
    assert abs(kp - 0.5) < 1e-9  # max(0, logP)=0, so kp=0.5


def test_kp_clamped_max():
    kp = _compute_kp_pancreas(100.0)
    assert kp == 8.0


def test_kp_clamped_min():
    # Would need logP very negative, but since max(0, logP)=0 gives 0.5 > 0.1
    kp = _compute_kp_pancreas(-999.0)
    assert kp == 0.5  # 0.5 + 0 = 0.5, not below 0.1


def test_higher_logp_higher_kp():
    kp1 = _compute_kp_pancreas(1.0)
    kp2 = _compute_kp_pancreas(3.0)
    assert kp2 > kp1


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_c_plasma_starts_at_dose_over_vd():
    result = simulate_pancreatic_distribution(
        drug_name="A",
        dose_mg=200.0,
        logP=1.0,
        cl_sys_L_per_h=5.0,
        vd_plasma_L=8.0,
    )
    expected = 200.0 / 8.0
    assert abs(result.c_plasma_mg_L[0] - expected) < 1e-9


def test_c_pancreas_starts_at_zero(base_result):
    assert base_result.c_pancreas_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Profile metrics
# ---------------------------------------------------------------------------


def test_cmax_pancreas_positive(base_result):
    assert base_result.cmax_pancreas_mg_L > 0.0


def test_auc_pancreas_positive(base_result):
    assert base_result.auc_pancreas_mg_h_per_L > 0.0


def test_auc_plasma_positive(base_result):
    assert base_result.auc_plasma_mg_h_per_L > 0.0


def test_tmax_pancreas_within_range(base_result):
    assert 0.0 <= base_result.tmax_pancreas_h <= 24.0


def test_kp_pancreas_stored(base_result):
    expected = _compute_kp_pancreas(1.5)
    assert abs(base_result.kp_pancreas - expected) < 1e-9


def test_list_lengths_consistent(base_result):
    n = len(base_result.times_h)
    assert len(base_result.c_plasma_mg_L) == n
    assert len(base_result.c_pancreas_mg_L) == n


# ---------------------------------------------------------------------------
# Higher logP -> higher Kp
# ---------------------------------------------------------------------------


def test_higher_logp_higher_kp_in_result():
    r_low = simulate_pancreatic_distribution("L", 100.0, logP=0.0, cl_sys_L_per_h=5.0)
    r_high = simulate_pancreatic_distribution("H", 100.0, logP=4.0, cl_sys_L_per_h=5.0)
    assert r_high.kp_pancreas > r_low.kp_pancreas


# ---------------------------------------------------------------------------
# Therapeutic adequacy classification
# ---------------------------------------------------------------------------


def test_adequacy_insufficient():
    assert _classify_therapeutic_adequacy(0.3) == "insufficient"


def test_adequacy_adequate_lower_bound():
    assert _classify_therapeutic_adequacy(0.5) == "adequate"


def test_adequacy_adequate_upper_bound():
    assert _classify_therapeutic_adequacy(1.5) == "adequate"


def test_adequacy_high():
    assert _classify_therapeutic_adequacy(2.0) == "high"


def test_adequacy_field_is_string(base_result):
    assert base_result.therapeutic_adequacy in {"insufficient", "adequate", "high"}


# ---------------------------------------------------------------------------
# screen_antibiotics
# ---------------------------------------------------------------------------


def test_screen_antibiotics_returns_list():
    results = screen_antibiotics(100.0, [1.0, 2.0, 3.0], 5.0)
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_antibiotics_sorted_by_kp_descending():
    results = screen_antibiotics(100.0, [0.5, 2.0, 4.0], 5.0)
    kps = [r.kp_pancreas for r in results]
    assert kps == sorted(kps, reverse=True)


def test_screen_antibiotics_drug_names():
    results = screen_antibiotics(100.0, [1.0, 2.0], 5.0)
    drug_names = {r.drug_name for r in results}
    assert "Drug_logP1.0" in drug_names or "Drug_logP2.0" in drug_names


def test_screen_antibiotics_result_types():
    results = screen_antibiotics(50.0, [1.0, 3.0], 3.0)
    for r in results:
        assert isinstance(r, PancreaticDistributionResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pancreatic_distribution("X", 0.0, 1.0, 5.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pancreatic_distribution("X", -10.0, 1.0, 5.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_pancreatic_distribution("X", 100.0, 1.0, 0.0)


def test_validation_vd_plasma_zero():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_pancreatic_distribution("X", 100.0, 1.0, 5.0, vd_plasma_L=0.0)


def test_validation_vd_panc_zero():
    with pytest.raises(ValueError, match="vd_panc_L"):
        simulate_pancreatic_distribution("X", 100.0, 1.0, 5.0, vd_panc_L=0.0)


def test_validation_q_panc_zero():
    with pytest.raises(ValueError, match="Q_panc_L_per_h"):
        simulate_pancreatic_distribution("X", 100.0, 1.0, 5.0, Q_panc_L_per_h=0.0)


# ---------------------------------------------------------------------------
# notes and other fields
# ---------------------------------------------------------------------------


def test_notes_is_string(base_result):
    assert isinstance(base_result.notes, str)


def test_notes_contains_kp(base_result):
    assert "Kp_pancreas" in base_result.notes


def test_panc_to_plasma_ratio_non_negative(base_result):
    assert base_result.panc_to_plasma_ratio >= 0.0
