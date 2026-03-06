"""Tests for enzyme turnover and mechanism-based inactivation model (Phase 216)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.enzyme_turnover import (
    EnzymeTurnoverResult,
    mbi_half_life,
    recovery_time,
    simulate_enzyme_turnover,
)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
BASE = dict(
    enzyme_name="CYP3A4",
    drug_name="Clarithromycin",
    inhibitor_conc_mg_L=1.0,
    kinact_per_h=0.5,
    ki_mg_L=0.1,
    ksyn_nmol_per_h=10.0,
    kdeg_per_h=0.02,
    e0_nmol=500.0,
    t_end_h=168.0,
    dt_h=0.5,
)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_negative_inhibitor_conc_raises(self):
        with pytest.raises(ValueError, match="inhibitor_conc_mg_L"):
            simulate_enzyme_turnover(**{**BASE, "inhibitor_conc_mg_L": -1.0})

    def test_negative_kinact_raises(self):
        with pytest.raises(ValueError, match="kinact_per_h"):
            simulate_enzyme_turnover(**{**BASE, "kinact_per_h": -0.1})

    def test_zero_ki_raises(self):
        with pytest.raises(ValueError, match="ki_mg_L"):
            simulate_enzyme_turnover(**{**BASE, "ki_mg_L": 0.0})

    def test_negative_ki_raises(self):
        with pytest.raises(ValueError, match="ki_mg_L"):
            simulate_enzyme_turnover(**{**BASE, "ki_mg_L": -1.0})

    def test_zero_ksyn_raises(self):
        with pytest.raises(ValueError, match="ksyn_nmol_per_h"):
            simulate_enzyme_turnover(**{**BASE, "ksyn_nmol_per_h": 0.0})

    def test_zero_kdeg_raises(self):
        with pytest.raises(ValueError, match="kdeg_per_h"):
            simulate_enzyme_turnover(**{**BASE, "kdeg_per_h": 0.0})

    def test_zero_e0_raises(self):
        with pytest.raises(ValueError, match="e0_nmol"):
            simulate_enzyme_turnover(**{**BASE, "e0_nmol": 0.0})


# ---------------------------------------------------------------------------
# Result type and fields
# ---------------------------------------------------------------------------
class TestResultFields:
    def test_returns_correct_type(self):
        result = simulate_enzyme_turnover(**BASE)
        assert isinstance(result, EnzymeTurnoverResult)

    def test_required_fields_present(self):
        result = simulate_enzyme_turnover(**BASE)
        for field in (
            "enzyme_name",
            "drug_name",
            "inhibitor_conc_mg_L",
            "kinact_per_h",
            "ki_mg_L",
            "times_h",
            "enzyme_amount_nmol",
            "e_baseline_nmol",
            "e_final_nmol",
            "pct_remaining_at_end",
            "mbi_half_life_h",
            "strong_inhibitor",
            "notes",
        ):
            assert hasattr(result, field), f"Missing field: {field}"

    def test_times_and_amounts_same_length(self):
        result = simulate_enzyme_turnover(**BASE)
        assert len(result.times_h) == len(result.enzyme_amount_nmol)

    def test_enzyme_name_preserved(self):
        result = simulate_enzyme_turnover(**BASE)
        assert result.enzyme_name == "CYP3A4"

    def test_drug_name_preserved(self):
        result = simulate_enzyme_turnover(**BASE)
        assert result.drug_name == "Clarithromycin"


# ---------------------------------------------------------------------------
# No-inhibitor behavior
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestNoInhibitor:
    def test_kinact_zero_enzyme_stays_at_baseline(self):
        """With kinact=0, enzyme should remain at baseline (ksyn/kdeg)."""
        result = simulate_enzyme_turnover(
            **{
                **BASE,
                "kinact_per_h": 0.0,
                "e0_nmol": 500.0,
                "ksyn_nmol_per_h": 10.0,
                "kdeg_per_h": 0.02,
            }
        )
        e_baseline = 10.0 / 0.02  # 500 nmol
        assert result.e_final_nmol == pytest.approx(e_baseline, rel=0.05)

    def test_zero_inhibitor_conc_enzyme_near_baseline(self):
        """With inhibitor_conc=0, no MBI occurs."""
        result = simulate_enzyme_turnover(**{**BASE, "inhibitor_conc_mg_L": 0.0})
        e_baseline = result.e_baseline_nmol
        assert result.e_final_nmol == pytest.approx(e_baseline, rel=0.05)

    def test_zero_inhibitor_notes_indicate_no_mbi(self):
        result = simulate_enzyme_turnover(**{**BASE, "inhibitor_conc_mg_L": 0.0})
        assert "No MBI" in result.notes or "baseline" in result.notes.lower()


# ---------------------------------------------------------------------------
# Inhibition behavior
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestInhibition:
    def test_high_inhibitor_depletes_enzyme(self):
        """High inhibitor concentration should significantly deplete enzyme."""
        result = simulate_enzyme_turnover(
            **{**BASE, "inhibitor_conc_mg_L": 100.0, "kinact_per_h": 1.0, "ki_mg_L": 0.1}
        )
        assert result.pct_remaining_at_end < 50.0

    def test_pct_remaining_in_valid_range(self):
        result = simulate_enzyme_turnover(**BASE)
        assert 0.0 <= result.pct_remaining_at_end <= 100.0

    def test_strong_inhibitor_flag_when_low_remaining(self):
        """strong_inhibitor=True when pct_remaining < 20."""
        result = simulate_enzyme_turnover(
            **{**BASE, "inhibitor_conc_mg_L": 100.0, "kinact_per_h": 2.0, "ki_mg_L": 0.01}
        )
        if result.pct_remaining_at_end < 20.0:
            assert result.strong_inhibitor is True

    def test_strong_inhibitor_false_when_high_remaining(self):
        """strong_inhibitor=False when enzyme is mostly intact."""
        result = simulate_enzyme_turnover(
            **{**BASE, "inhibitor_conc_mg_L": 0.001, "kinact_per_h": 0.01, "ki_mg_L": 100.0}
        )
        if result.pct_remaining_at_end >= 20.0:
            assert result.strong_inhibitor is False

    def test_enzyme_amounts_nonneg(self):
        result = simulate_enzyme_turnover(**BASE)
        assert all(e >= 0 for e in result.enzyme_amount_nmol)

    def test_e_baseline_equals_ksyn_over_kdeg(self):
        result = simulate_enzyme_turnover(**BASE)
        expected_baseline = BASE["ksyn_nmol_per_h"] / BASE["kdeg_per_h"]
        assert result.e_baseline_nmol == pytest.approx(expected_baseline)

    def test_more_inhibitor_lower_pct_remaining(self):
        """Higher inhibitor concentration leads to lower enzyme remaining."""
        low = simulate_enzyme_turnover(**{**BASE, "inhibitor_conc_mg_L": 0.1})
        high = simulate_enzyme_turnover(**{**BASE, "inhibitor_conc_mg_L": 10.0})
        assert high.pct_remaining_at_end < low.pct_remaining_at_end


# ---------------------------------------------------------------------------
# mbi_half_life function
# ---------------------------------------------------------------------------
class TestMbiHalfLife:
    def test_returns_positive(self):
        t_half = mbi_half_life(0.5, 0.1, 1.0, kdeg=0.02)
        assert t_half > 0

    def test_no_inhibitor_returns_natural_turnover_halflife(self):
        """With no inhibitor, t_half = ln(2) / kdeg."""
        kdeg = 0.02
        t_half = mbi_half_life(kinact=0.5, ki=0.1, inhibitor_conc=0.0, kdeg=kdeg)
        expected = math.log(2) / kdeg
        assert t_half == pytest.approx(expected, rel=1e-6)

    def test_higher_conc_shorter_halflife(self):
        """More inhibitor → shorter apparent inactivation half-life."""
        low = mbi_half_life(0.5, 0.1, 0.1, kdeg=0.02)
        high = mbi_half_life(0.5, 0.1, 10.0, kdeg=0.02)
        assert high < low

    def test_mbi_halflife_stored_in_result(self):
        result = simulate_enzyme_turnover(**BASE)
        assert result.mbi_half_life_h > 0

    def test_invalid_ki_raises(self):
        with pytest.raises(ValueError, match="ki"):
            mbi_half_life(0.5, 0.0, 1.0)

    def test_invalid_inhibitor_conc_raises(self):
        with pytest.raises(ValueError, match="inhibitor_conc"):
            mbi_half_life(0.5, 0.1, -1.0)


# ---------------------------------------------------------------------------
# recovery_time function
# ---------------------------------------------------------------------------
class TestRecoveryTime:
    def test_returns_positive(self):
        t = recovery_time(0.02)
        assert t > 0

    def test_slower_kdeg_longer_recovery(self):
        """Smaller kdeg → slower enzyme resynthesis → longer recovery."""
        fast = recovery_time(0.1)
        slow = recovery_time(0.01)
        assert slow > fast

    def test_90pct_recovery_default(self):
        kdeg = 0.02
        t = recovery_time(kdeg, target_recovery_pct=0.9)
        expected = -math.log(1.0 - 0.9) / kdeg
        assert t == pytest.approx(expected, rel=1e-6)

    def test_higher_target_longer_recovery(self):
        t90 = recovery_time(0.02, 0.9)
        t99 = recovery_time(0.02, 0.99)
        assert t99 > t90

    def test_invalid_kdeg_raises(self):
        with pytest.raises(ValueError, match="kdeg_per_h"):
            recovery_time(0.0)

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match="target_recovery_pct"):
            recovery_time(0.02, target_recovery_pct=1.5)
