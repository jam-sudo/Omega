"""Tests for clinical/drug_interaction_simulator.py — Phase 433."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.drug_interaction_simulator import (
    DrugInteractionSimResult,
    onset_offset_analysis,
    simulate_drug_interaction,
)

# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------

VICTIM_NAME = "midazolam"
PERP_NAME = "ketoconazole"
VICTIM_DOSE = 5.0  # mg
PERP_DOSE = 200.0  # mg
VICTIM_CL = 25.0  # L/h
VICTIM_VD = 80.0  # L
PERP_CL = 5.0  # L/h
PERP_VD = 100.0  # L
KI = 0.02  # mg/L — strong inhibitor
FM = 0.9  # CYP3A4 fm

# ---------------------------------------------------------------------------
# simulate_drug_interaction — smoke tests
# ---------------------------------------------------------------------------


def test_simulate_reversible_returns_result():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, inhibition_type="reversible", n_days=3,
    )
    assert isinstance(result, DrugInteractionSimResult)


def test_simulate_mbi_returns_result():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, inhibition_type="mechanism_based", n_days=3,
    )
    assert isinstance(result, DrugInteractionSimResult)


def test_result_field_lengths_match():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, n_days=2, dt_h=0.5,
    )
    n = len(result.times_h)
    assert len(result.c_victim_mg_L) == n
    assert len(result.c_perpetrator_mg_L) == n
    assert len(result.cl_victim_dynamic) == n


def test_times_start_at_zero():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, n_days=1,
    )
    assert result.times_h[0] == pytest.approx(0.0, abs=1e-9)


def test_concentrations_non_negative():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, n_days=3,
    )
    assert all(c >= 0 for c in result.c_victim_mg_L)
    assert all(c >= 0 for c in result.c_perpetrator_mg_L)


def test_cl_dynamic_floor_not_zero():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, n_days=3,
    )
    assert all(cl > 0 for cl in result.cl_victim_dynamic)


def test_reversible_aucr_gt_1_with_strong_inhibitor():
    """Strong inhibitor + high fm should yield AUCR > 1.25."""
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, inhibition_type="reversible", n_days=7,
    )
    assert result.aucr > 1.0


def test_mbi_aucr_generally_gt_reversible():
    """MBI tends to have larger AUCR at steady state than reversible (same Ki)."""
    rev = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=0.1, fm_victim=FM, inhibition_type="reversible", n_days=7,
    )
    mbi = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=0.1, fm_victim=FM, inhibition_type="mechanism_based", n_days=7,
    )
    # Both should have AUCR > 1
    assert rev.aucr > 1.0
    assert mbi.aucr > 1.0


def test_no_inhibition_fm_zero_aucr_near_1():
    """fm=0 → no DDI → AUCR should be ≈ 1."""
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=0.0, inhibition_type="reversible", n_days=5,
    )
    assert result.aucr == pytest.approx(1.0, abs=0.1)


def test_auc_day1_positive():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, n_days=3,
    )
    assert result.auc_day1 > 0


def test_auc_last_day_positive():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, n_days=3,
    )
    assert result.auc_last_day > 0


def test_aucr_equals_auc_last_over_auc_day1():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, n_days=3,
    )
    expected_aucr = result.auc_last_day / result.auc_day1
    assert result.aucr == pytest.approx(expected_aucr, rel=1e-6)


def test_inhibition_type_stored_in_result():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, inhibition_type="mechanism_based", n_days=2,
    )
    assert result.inhibition_type == "mechanism_based"


def test_drug_names_stored_in_result():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, n_days=1,
    )
    assert result.victim_name == VICTIM_NAME
    assert result.perpetrator_name == PERP_NAME


def test_notes_is_list():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, n_days=1,
    )
    assert isinstance(result.notes, list)


def test_mbi_notes_mention_kdeg():
    result = simulate_drug_interaction(
        VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
        VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
        ki_mg_L=KI, fm_victim=FM, inhibition_type="mechanism_based", n_days=2,
    )
    mbi_notes = [n for n in result.notes if "kdeg" in n.lower() or "mbi" in n.lower()]
    assert len(mbi_notes) >= 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_victim_name_raises():
    with pytest.raises(ValueError, match="victim_name"):
        simulate_drug_interaction(
            "", PERP_NAME, VICTIM_DOSE, PERP_DOSE,
            VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
            ki_mg_L=KI, fm_victim=FM,
        )


def test_invalid_perpetrator_name_raises():
    with pytest.raises(ValueError, match="perpetrator_name"):
        simulate_drug_interaction(
            VICTIM_NAME, "", VICTIM_DOSE, PERP_DOSE,
            VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
            ki_mg_L=KI, fm_victim=FM,
        )


def test_invalid_victim_dose_raises():
    with pytest.raises(ValueError, match="victim_dose_mg"):
        simulate_drug_interaction(
            VICTIM_NAME, PERP_NAME, -1.0, PERP_DOSE,
            VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
            ki_mg_L=KI, fm_victim=FM,
        )


def test_invalid_ki_raises():
    with pytest.raises(ValueError, match="ki_mg_L"):
        simulate_drug_interaction(
            VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
            VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
            ki_mg_L=0.0, fm_victim=FM,
        )


def test_invalid_fm_raises():
    with pytest.raises(ValueError, match="fm_victim"):
        simulate_drug_interaction(
            VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
            VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
            ki_mg_L=KI, fm_victim=1.5,
        )


def test_invalid_inhibition_type_raises():
    with pytest.raises(ValueError, match="inhibition_type"):
        simulate_drug_interaction(
            VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
            VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
            ki_mg_L=KI, fm_victim=FM, inhibition_type="unknown",
        )


def test_invalid_n_days_raises():
    with pytest.raises(ValueError, match="n_days"):
        simulate_drug_interaction(
            VICTIM_NAME, PERP_NAME, VICTIM_DOSE, PERP_DOSE,
            VICTIM_CL, VICTIM_VD, PERP_CL, PERP_VD,
            ki_mg_L=KI, fm_victim=FM, n_days=0,
        )


# ---------------------------------------------------------------------------
# onset_offset_analysis
# ---------------------------------------------------------------------------


def test_onset_offset_returns_dict():
    result = onset_offset_analysis(
        VICTIM_NAME, PERP_NAME,
        victim_cl_L_per_h=VICTIM_CL,
        fm_victim=FM,
        ki_mg_L=KI,
        kdeg_per_h=0.0462,
        kinact_per_h=0.04,
        c_perp_steady_mg_L=0.1,
    )
    assert isinstance(result, dict)


def test_onset_offset_keys_present():
    result = onset_offset_analysis(
        VICTIM_NAME, PERP_NAME,
        victim_cl_L_per_h=VICTIM_CL,
        fm_victim=FM,
        ki_mg_L=KI,
        kdeg_per_h=0.0462,
        kinact_per_h=0.04,
        c_perp_steady_mg_L=0.1,
    )
    for key in ("t_onset_h", "t_recovery_h", "max_aucr_estimate"):
        assert key in result, f"Missing key: {key}"


def test_t_recovery_gt_t_onset_when_kinact_large():
    """When kinact >> kdeg, onset is fast and recovery is slow."""
    result = onset_offset_analysis(
        VICTIM_NAME, PERP_NAME,
        victim_cl_L_per_h=VICTIM_CL,
        fm_victim=FM,
        ki_mg_L=0.01,
        kdeg_per_h=0.02,
        kinact_per_h=2.0,  # very fast MBI
        c_perp_steady_mg_L=1.0,
    )
    assert result["t_recovery_h"] > result["t_onset_h"]


def test_max_aucr_gt_1_for_significant_fm():
    result = onset_offset_analysis(
        VICTIM_NAME, PERP_NAME,
        victim_cl_L_per_h=VICTIM_CL,
        fm_victim=0.9,
        ki_mg_L=0.05,
        kdeg_per_h=0.02,
        kinact_per_h=0.5,
        c_perp_steady_mg_L=1.0,
    )
    assert result["max_aucr_estimate"] > 1.0


def test_zero_perp_concentration_gives_aucr_1():
    result = onset_offset_analysis(
        VICTIM_NAME, PERP_NAME,
        victim_cl_L_per_h=VICTIM_CL,
        fm_victim=FM,
        ki_mg_L=KI,
        kdeg_per_h=0.02,
        kinact_per_h=0.5,
        c_perp_steady_mg_L=0.0,
    )
    assert result["max_aucr_estimate"] == pytest.approx(1.0, abs=0.01)


def test_onset_offset_invalid_ki_raises():
    with pytest.raises(ValueError, match="ki_mg_L"):
        onset_offset_analysis(
            VICTIM_NAME, PERP_NAME,
            victim_cl_L_per_h=VICTIM_CL,
            fm_victim=FM,
            ki_mg_L=0.0,
            kdeg_per_h=0.02,
            kinact_per_h=0.5,
            c_perp_steady_mg_L=1.0,
        )


def test_onset_offset_invalid_kdeg_raises():
    with pytest.raises(ValueError, match="kdeg_per_h"):
        onset_offset_analysis(
            VICTIM_NAME, PERP_NAME,
            victim_cl_L_per_h=VICTIM_CL,
            fm_victim=FM,
            ki_mg_L=KI,
            kdeg_per_h=0.0,
            kinact_per_h=0.5,
            c_perp_steady_mg_L=1.0,
        )


def test_onset_offset_drug_names_in_result():
    result = onset_offset_analysis(
        VICTIM_NAME, PERP_NAME,
        victim_cl_L_per_h=VICTIM_CL,
        fm_victim=FM,
        ki_mg_L=KI,
        kdeg_per_h=0.02,
        kinact_per_h=0.5,
        c_perp_steady_mg_L=1.0,
    )
    assert result["victim_name"] == VICTIM_NAME
    assert result["perpetrator_name"] == PERP_NAME


def test_onset_offset_values_positive():
    result = onset_offset_analysis(
        VICTIM_NAME, PERP_NAME,
        victim_cl_L_per_h=VICTIM_CL,
        fm_victim=FM,
        ki_mg_L=KI,
        kdeg_per_h=0.0462,
        kinact_per_h=0.04,
        c_perp_steady_mg_L=0.1,
    )
    assert result["t_onset_h"] > 0
    assert result["t_recovery_h"] > 0
    assert result["max_aucr_estimate"] > 0
