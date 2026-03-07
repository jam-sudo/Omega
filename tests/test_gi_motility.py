"""Tests for core/gi_motility.py — Phase 492."""

from __future__ import annotations

import pytest

from omega_pbpk.core.gi_motility import (
    GIMotilityResult,
    calculate_transit_time,
    colon_transit_time,
    gastric_emptying_half_time,
    simulate_gi_transit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(
    drug_name="TestDrug",
    dose_mg=100.0,
    state="fasted",
    formulation="immediate_release",
    t_end=8.0,
    dt=0.05,
):
    return simulate_gi_transit(
        drug_name=drug_name,
        dose_mg=dose_mg,
        state=state,
        formulation=formulation,
        t_end_h=t_end,
        dt_h=dt,
    )


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------

class TestGIMotilityResultStructure:
    def test_returns_dataclass(self):
        r = _run()
        assert isinstance(r, GIMotilityResult)

    def test_drug_name_stored(self):
        r = _run(drug_name="Ibuprofen")
        assert r.drug_name == "Ibuprofen"

    def test_state_stored(self):
        r = _run(state="fed")
        assert r.state == "fed"

    def test_formulation_stored(self):
        r = _run(formulation="modified_release")
        assert r.formulation == "modified_release"

    def test_list_lengths_consistent(self):
        r = _run(t_end=4.0, dt=0.1)
        n = len(r.times_h)
        assert len(r.mass_stomach_mg) == n
        assert len(r.mass_si_mg) == n
        assert len(r.mass_colon_mg) == n
        assert len(r.mass_absorbed_mg) == n
        assert len(r.mass_excreted_mg) == n

    def test_times_start_at_zero(self):
        r = _run()
        assert r.times_h[0] == pytest.approx(0.0, abs=1e-9)

    def test_times_end_at_t_end(self):
        r = _run(t_end=6.0)
        assert r.times_h[-1] == pytest.approx(6.0, abs=0.1)


# ---------------------------------------------------------------------------
# Mass conservation
# ---------------------------------------------------------------------------

class TestMassConservation:
    def test_initial_mass_equals_dose(self):
        dose = 200.0
        r = _run(dose_mg=dose)
        total_initial = (
            r.mass_stomach_mg[0]
            + r.mass_si_mg[0]
            + r.mass_colon_mg[0]
            + r.mass_absorbed_mg[0]
            + r.mass_excreted_mg[0]
        )
        assert total_initial == pytest.approx(dose, rel=1e-6)

    def test_mass_conservation_at_all_times(self):
        dose = 150.0
        r = _run(dose_mg=dose, t_end=8.0, dt=0.05)
        for i in range(len(r.times_h)):
            total = (
                r.mass_stomach_mg[i]
                + r.mass_si_mg[i]
                + r.mass_colon_mg[i]
                + r.mass_absorbed_mg[i]
                + r.mass_excreted_mg[i]
            )
            assert total == pytest.approx(dose, rel=0.02), (
                f"Mass conservation failed at t={r.times_h[i]:.2f}h: total={total:.4f}"
            )

    def test_all_masses_nonneg(self):
        r = _run(dose_mg=100.0, t_end=8.0)
        for i in range(len(r.times_h)):
            assert r.mass_stomach_mg[i] >= -1e-9
            assert r.mass_si_mg[i] >= -1e-9
            assert r.mass_colon_mg[i] >= -1e-9
            assert r.mass_absorbed_mg[i] >= -1e-9
            assert r.mass_excreted_mg[i] >= -1e-9


# ---------------------------------------------------------------------------
# Fed vs fasted
# ---------------------------------------------------------------------------

class TestFedVsFasted:
    def test_fed_slower_gastric_emptying(self):
        r_fasted = _run(state="fasted")
        r_fed = _run(state="fed")
        # Fed state → stomach empties slower → higher stomach mass at early time
        # Compare stomach at t~1h (index corresponding to 1h)
        idx = min(20, len(r_fasted.times_h) - 1)  # 20*0.05=1h
        assert r_fed.mass_stomach_mg[idx] > r_fasted.mass_stomach_mg[idx]

    def test_gastric_emptying_t50_fed_greater_than_fasted(self):
        r_fasted = _run(state="fasted")
        r_fed = _run(state="fed")
        assert r_fed.gastric_emptying_t50_h > r_fasted.gastric_emptying_t50_h

    def test_absorbed_mass_is_dose_dependent(self):
        r = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        # Absorbed mass should scale roughly with dose
        assert r2.mass_absorbed_mg[-1] > r.mass_absorbed_mg[-1]


# ---------------------------------------------------------------------------
# Formulation effects
# ---------------------------------------------------------------------------

class TestFormulationEffects:
    def test_modified_release_slower_absorption(self):
        r_ir = _run(formulation="immediate_release", t_end=8.0)
        r_mr = _run(formulation="modified_release", t_end=8.0)
        # Immediate release absorbs more by t=8h
        assert r_ir.mass_absorbed_mg[-1] > r_mr.mass_absorbed_mg[-1]

    def test_extended_release_slower_than_immediate(self):
        r_ir = _run(formulation="immediate_release", t_end=8.0)
        r_er = _run(formulation="extended_release", t_end=8.0)
        assert r_ir.mass_absorbed_mg[-1] > r_er.mass_absorbed_mg[-1]


# ---------------------------------------------------------------------------
# gastric_emptying_half_time
# ---------------------------------------------------------------------------

class TestGastricEmptyingHalfTime:
    def test_fed_greater_than_fasted(self):
        t_fasted = gastric_emptying_half_time(state="fasted")
        t_fed = gastric_emptying_half_time(state="fed")
        assert t_fed > t_fasted

    def test_fasted_default_value(self):
        t = gastric_emptying_half_time(state="fasted")
        assert t == pytest.approx(0.5, rel=0.2)

    def test_fed_default_value(self):
        t = gastric_emptying_half_time(state="fed")
        assert t == pytest.approx(2.0, rel=0.3)

    def test_high_calorie_meal_slows_emptying(self):
        t_low = gastric_emptying_half_time(state="fed", meal_calorie_kcal=300.0)
        t_high = gastric_emptying_half_time(state="fed", meal_calorie_kcal=800.0)
        assert t_high > t_low

    def test_modified_release_slower_emptying(self):
        t_ir = gastric_emptying_half_time(state="fasted", formulation="immediate_release")
        t_mr = gastric_emptying_half_time(state="fasted", formulation="modified_release")
        assert t_mr > t_ir

    def test_invalid_state_raises(self):
        with pytest.raises(ValueError):
            gastric_emptying_half_time(state="unknown")


# ---------------------------------------------------------------------------
# colon_transit_time
# ---------------------------------------------------------------------------

class TestColonTransitTime:
    def test_returns_positive(self):
        t = colon_transit_time()
        assert t > 0

    def test_fed_longer_than_fasted(self):
        t_fasted = colon_transit_time(state="fasted")
        t_fed = colon_transit_time(state="fed")
        assert t_fed >= t_fasted

    def test_fasted_in_reasonable_range(self):
        t = colon_transit_time(state="fasted")
        assert 12.0 <= t <= 48.0

    def test_invalid_state_raises(self):
        with pytest.raises(ValueError):
            colon_transit_time(state="invalid")


# ---------------------------------------------------------------------------
# calculate_transit_time
# ---------------------------------------------------------------------------

class TestCalculateTransitTime:
    def test_stomach_fasted(self):
        t = calculate_transit_time("fasted", "stomach")
        assert t > 0

    def test_colon_segment(self):
        t = calculate_transit_time("fasted", "colon")
        assert t == pytest.approx(colon_transit_time("fasted"), rel=1e-6)

    def test_fed_stomach_longer_than_fasted(self):
        t_f = calculate_transit_time("fasted", "stomach")
        t_fed = calculate_transit_time("fed", "stomach")
        assert t_fed > t_f

    def test_small_intestine_alias(self):
        t1 = calculate_transit_time("fasted", "small_intestine")
        t2 = calculate_transit_time("fasted", "si")
        assert t1 == pytest.approx(t2, rel=1e-6)

    def test_segment_values_positive(self):
        for seg in ("stomach", "duodenum", "jejunum", "ileum", "colon"):
            assert calculate_transit_time("fasted", seg) > 0

    def test_invalid_segment_raises(self):
        with pytest.raises(ValueError):
            calculate_transit_time("fasted", "unknown_segment")

    def test_invalid_state_raises(self):
        with pytest.raises(ValueError):
            calculate_transit_time("unknown", "stomach")


# ---------------------------------------------------------------------------
# fa_estimated
# ---------------------------------------------------------------------------

class TestFaEstimated:
    def test_fa_in_0_1(self):
        r = _run(t_end=24.0)
        assert 0.0 <= r.fa_estimated <= 1.0

    def test_immediate_release_higher_fa(self):
        r_ir = _run(formulation="immediate_release", t_end=24.0)
        r_mr = _run(formulation="modified_release", t_end=24.0)
        assert r_ir.fa_estimated >= r_mr.fa_estimated


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_invalid_state_raises(self):
        with pytest.raises(ValueError):
            simulate_gi_transit("X", 100.0, state="unknown")

    def test_invalid_formulation_raises(self):
        with pytest.raises(ValueError):
            simulate_gi_transit("X", 100.0, formulation="suppository")

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_gi_transit("X", 0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_gi_transit("X", -50.0)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError):
            simulate_gi_transit("X", 100.0, t_end_h=0.0)

    def test_negative_dt_raises(self):
        with pytest.raises(ValueError):
            simulate_gi_transit("X", 100.0, dt_h=-0.01)
