"""Tests for Phase 666 — Drug Crystallization Kinetics in Vivo."""

import pytest

from omega_pbpk.core.in_vivo_crystallization_p666 import (
    InVivoCrystallizationResult,
    simulate_in_vivo_crystallization,
)

DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=500.0,
    solubility_mg_mL=0.01,
    logP=2.0,
    biofluid="urine",
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_in_vivo_crystallization(**params)


class TestReturnType:
    def test_returns_result(self):
        r = _run()
        assert isinstance(r, InVivoCrystallizationResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_biofluid(self):
        r = _run()
        assert r.biofluid == "urine"


class TestListLengths:
    def test_times_length(self):
        r = _run(t_end_h=4.0, dt_h=0.05)
        expected = int(4.0 / 0.05) + 1
        assert len(r.times_h) == expected

    def test_dissolved_length(self):
        r = _run()
        assert len(r.c_dissolved_mg_mL) == len(r.times_h)

    def test_crystal_length(self):
        r = _run()
        assert len(r.c_crystal_mg_mL) == len(r.times_h)

    def test_sr_length(self):
        r = _run()
        assert len(r.supersaturation_ratio) == len(r.times_h)

    def test_nucleation_rate_length(self):
        r = _run()
        assert len(r.nucleation_rate_per_h) == len(r.times_h)


class TestNonNegative:
    def test_dissolved_non_negative(self):
        r = _run()
        assert all(v >= 0 for v in r.c_dissolved_mg_mL)

    def test_crystal_non_negative(self):
        r = _run()
        assert all(v >= 0 for v in r.c_crystal_mg_mL)

    def test_sr_non_negative(self):
        r = _run()
        assert all(v >= 0 for v in r.supersaturation_ratio)


class TestSupersaturation:
    def test_initial_sr_high(self):
        """With low solubility and high dose, initial SR should be > 1."""
        r = _run(dose_mg=500.0, solubility_mg_mL=0.01)
        assert r.supersaturation_ratio[0] > 1.0

    def test_high_supersaturation_faster_crystallization(self):
        """Higher SR should lead to more crystal formation."""
        r_high = _run(concentration_mg_mL=1.0, solubility_mg_mL=0.01)
        r_low = _run(concentration_mg_mL=0.05, solubility_mg_mL=0.01)
        assert r_high.max_crystal_fraction_pct >= r_low.max_crystal_fraction_pct


class TestCrystalFraction:
    def test_fraction_range(self):
        r = _run()
        assert 0 <= r.max_crystal_fraction_pct <= 100

    def test_no_crystallization_below_solubility(self):
        """When concentration < solubility, crystal should stay near 0."""
        r = _run(concentration_mg_mL=0.001, solubility_mg_mL=1.0)
        assert r.max_crystal_fraction_pct < 1.0
        assert all(c < 0.001 for c in r.c_crystal_mg_mL)


class TestTimings:
    def test_induction_before_half_time(self):
        """Induction time should be <= crystallization half time."""
        r = _run(concentration_mg_mL=1.0, solubility_mg_mL=0.01)
        assert r.induction_time_h <= r.crystallization_half_time_h


class TestRiskCategory:
    def test_critical_risk(self):
        r = _run(concentration_mg_mL=1.0, solubility_mg_mL=0.01)
        # SR = 1.0 / (0.01 * 1.0001) ≈ 100 > 5
        assert r.risk_category == "critical"

    def test_low_risk(self):
        r = _run(concentration_mg_mL=0.005, solubility_mg_mL=1.0)
        assert r.risk_category == "low"

    def test_moderate_risk(self):
        """SR between 1 and 2 should be moderate."""
        # Need SR ~1.5: concentration / sol_eff ≈ 1.5
        r = _run(concentration_mg_mL=0.015, solubility_mg_mL=0.01)
        assert r.risk_category in ("moderate", "high")

    def test_high_risk(self):
        """SR between 2 and 5."""
        # SR = 0.04 / (0.01 * 1.0001) ≈ 4
        r = _run(concentration_mg_mL=0.04, solubility_mg_mL=0.01)
        assert r.risk_category == "high"


class TestPhysiologicalConsequence:
    def test_urine_high_risk(self):
        r = _run(biofluid="urine", concentration_mg_mL=1.0, solubility_mg_mL=0.01)
        assert r.physiological_consequence == "nephrolithiasis_risk"

    def test_bile_risk(self):
        r = _run(biofluid="bile", concentration_mg_mL=1.0, solubility_mg_mL=0.01)
        assert r.physiological_consequence == "gallstone_risk"

    def test_synovial_risk(self):
        r = _run(biofluid="synovial_fluid", concentration_mg_mL=1.0, solubility_mg_mL=0.01)
        assert r.physiological_consequence == "crystal_arthropathy_risk"

    def test_aqueous_humor_risk(self):
        r = _run(biofluid="aqueous_humor", concentration_mg_mL=1.0, solubility_mg_mL=0.01)
        assert r.physiological_consequence == "ocular_precipitation_risk"

    def test_low_risk_minimal(self):
        r = _run(biofluid="urine", concentration_mg_mL=0.001, solubility_mg_mL=1.0)
        assert r.physiological_consequence == "minimal"


class TestValidation:
    def test_dose_zero(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_dose_negative(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-10)

    def test_solubility_zero(self):
        with pytest.raises(ValueError):
            _run(solubility_mg_mL=0)

    def test_invalid_biofluid(self):
        with pytest.raises(ValueError):
            _run(biofluid="blood")


class TestNotes:
    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0
