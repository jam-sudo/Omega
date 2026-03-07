"""Tests for Phase 826 — ligand_binding_kinetics module."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.ligand_binding_kinetics import (
    LigandBindingResult,
    screen_drug_concentrations,
    simulate_binding_kinetics,
)

# ---------------------------------------------------------------------------
# Default parameters used across tests
# ---------------------------------------------------------------------------

DEFAULT = {
    "drug_name": "DrugA",
    "target_name": "TargetX",
    "drug_conc_nM": 10.0,
    "receptor_conc_nM": 100.0,
    "kon_per_nM_per_h": 0.1,  # nM^-1 h^-1
    "koff_per_h": 1.0,  # h^-1  → KD = 10 nM
    "t_end_h": 24.0,
    "dt_h": 0.01,
}


def _run(**overrides) -> LigandBindingResult:
    params = {**DEFAULT, **overrides}
    return simulate_binding_kinetics(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_ligand_binding_result(self):
        result = _run()
        assert isinstance(result, LigandBindingResult)

    def test_result_is_frozen(self):
        result = _run()
        with pytest.raises((AttributeError, TypeError)):
            result.kd_nM = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Derived parameter correctness
# ---------------------------------------------------------------------------


class TestDerivedParameters:
    def test_kd_equals_koff_over_kon(self):
        kon = 0.05
        koff = 2.0
        result = _run(kon_per_nM_per_h=kon, koff_per_h=koff)
        expected_kd = koff / kon
        assert math.isclose(result.kd_nM, expected_kd, rel_tol=1e-9)

    def test_t_half_complex(self):
        koff = 0.5
        result = _run(koff_per_h=koff)
        expected = math.log(2.0) / koff
        assert math.isclose(result.t_half_complex_h, expected, rel_tol=1e-9)

    def test_residence_time_equals_1_over_koff(self):
        koff = 2.0
        result = _run(koff_per_h=koff)
        expected = 1.0 / koff
        assert math.isclose(result.residence_time_h, expected, rel_tol=1e-9)

    def test_stored_drug_name(self):
        result = _run(drug_name="Aspirin")
        assert result.drug_name == "Aspirin"

    def test_stored_target_name(self):
        result = _run(target_name="COX1")
        assert result.target_name == "COX1"


# ---------------------------------------------------------------------------
# Equilibrium behaviour
# ---------------------------------------------------------------------------


class TestEquilibrium:
    def test_complex_positive_for_positive_drug(self):
        result = _run()
        assert result.complex_at_equilibrium_nM > 0.0

    def test_occupancy_positive_for_positive_drug(self):
        result = _run()
        assert result.occupancy_at_equilibrium_pct > 0.0

    def test_occupancy_le_100(self):
        result = _run()
        assert result.occupancy_at_equilibrium_pct <= 100.0

    def test_high_conc_approaches_100_pct_occupancy(self):
        # Drug at 1000x KD → near-saturation
        kd = DEFAULT["koff_per_h"] / DEFAULT["kon_per_nM_per_h"]  # 10 nM
        result = _run(drug_conc_nM=kd * 1000.0, t_end_h=48.0)
        assert result.occupancy_at_equilibrium_pct > 90.0

    def test_occupancy_at_kd_approx_50_pct(self):
        # When drug_conc == KD and receptor_conc << KD (free-drug approx holds),
        # equilibrium occupancy should be ~50%.
        # KD = koff/kon = 1.0 / 0.1 = 10 nM
        # Use receptor_conc = 0.1 nM (100x below KD) so free drug ≈ drug_conc.
        kd = DEFAULT["koff_per_h"] / DEFAULT["kon_per_nM_per_h"]  # 10 nM
        result = _run(drug_conc_nM=kd, receptor_conc_nM=0.1, t_end_h=48.0)
        assert abs(result.occupancy_at_equilibrium_pct - 50.0) < 10.0

    def test_higher_conc_higher_occupancy(self):
        result_low = _run(drug_conc_nM=1.0, t_end_h=24.0)
        result_high = _run(drug_conc_nM=100.0, t_end_h=24.0)
        assert result_high.occupancy_at_equilibrium_pct > result_low.occupancy_at_equilibrium_pct

    def test_complex_zero_when_zero_drug(self):
        result = _run(drug_conc_nM=0.0)
        assert result.complex_at_equilibrium_nM == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Selectivity index
# ---------------------------------------------------------------------------


class TestSelectivityIndex:
    def test_selectivity_index_1_when_no_competitor(self):
        result = _run(competing_kd_nM=0.0)
        assert math.isclose(result.selectivity_index, 1.0, rel_tol=1e-9)

    def test_selectivity_gt_1_when_competitor_kd_higher(self):
        # competitor KD > drug KD → drug is more selective → index > 1
        kd = DEFAULT["koff_per_h"] / DEFAULT["kon_per_nM_per_h"]  # 10 nM
        result = _run(competing_kd_nM=kd * 100.0)
        assert result.selectivity_index > 1.0

    def test_selectivity_lt_1_when_competitor_kd_lower(self):
        # competitor KD < drug KD → drug is less selective
        kd = DEFAULT["koff_per_h"] / DEFAULT["kon_per_nM_per_h"]  # 10 nM
        result = _run(competing_kd_nM=kd * 0.01)
        assert result.selectivity_index < 1.0

    def test_selectivity_equals_ratio(self):
        kon = 0.1
        koff = 1.0
        kd = koff / kon  # 10 nM
        competing_kd = 500.0
        result = _run(
            kon_per_nM_per_h=kon,
            koff_per_h=koff,
            competing_kd_nM=competing_kd,
        )
        expected = competing_kd / kd
        assert math.isclose(result.selectivity_index, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# screen_drug_concentrations
# ---------------------------------------------------------------------------


class TestScreenDrugConcentrations:
    def test_returns_list(self):
        concs = [1.0, 5.0, 10.0, 50.0]
        results = screen_drug_concentrations(
            drug_name="DrugA",
            target_name="TargetX",
            concentrations_nM=concs,
            receptor_conc_nM=100.0,
            kon_per_nM_per_h=0.1,
            koff_per_h=1.0,
        )
        assert isinstance(results, list)

    def test_returns_correct_length(self):
        concs = [1.0, 5.0, 10.0, 50.0, 100.0]
        results = screen_drug_concentrations(
            drug_name="DrugA",
            target_name="TargetX",
            concentrations_nM=concs,
            receptor_conc_nM=100.0,
            kon_per_nM_per_h=0.1,
            koff_per_h=1.0,
        )
        assert len(results) == len(concs)

    def test_sorted_ascending_by_conc(self):
        concs = [50.0, 1.0, 100.0, 10.0]
        results = screen_drug_concentrations(
            drug_name="DrugA",
            target_name="TargetX",
            concentrations_nM=concs,
            receptor_conc_nM=100.0,
            kon_per_nM_per_h=0.1,
            koff_per_h=1.0,
        )
        drug_concs = [r.drug_conc_nM for r in results]
        assert drug_concs == sorted(drug_concs)

    def test_each_element_is_result_type(self):
        concs = [5.0, 20.0]
        results = screen_drug_concentrations(
            drug_name="DrugA",
            target_name="TargetX",
            concentrations_nM=concs,
            receptor_conc_nM=100.0,
            kon_per_nM_per_h=0.1,
            koff_per_h=1.0,
        )
        for r in results:
            assert isinstance(r, LigandBindingResult)

    def test_occupancy_increases_with_conc(self):
        concs = [1.0, 10.0, 100.0]
        results = screen_drug_concentrations(
            drug_name="DrugA",
            target_name="TargetX",
            concentrations_nM=concs,
            receptor_conc_nM=100.0,
            kon_per_nM_per_h=0.1,
            koff_per_h=1.0,
        )
        occupancies = [r.occupancy_at_equilibrium_pct for r in results]
        # Occupancy should be monotonically increasing
        assert occupancies[0] < occupancies[1] < occupancies[2]


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_nonempty(self):
        result = _run()
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_negative_drug_conc_raises(self):
        with pytest.raises(ValueError):
            _run(drug_conc_nM=-1.0)

    def test_negative_receptor_conc_raises(self):
        with pytest.raises(ValueError):
            _run(receptor_conc_nM=-5.0)

    def test_zero_kon_raises(self):
        with pytest.raises(ValueError):
            _run(kon_per_nM_per_h=0.0)

    def test_negative_kon_raises(self):
        with pytest.raises(ValueError):
            _run(kon_per_nM_per_h=-0.01)

    def test_zero_koff_raises(self):
        with pytest.raises(ValueError):
            _run(koff_per_h=0.0)

    def test_negative_koff_raises(self):
        with pytest.raises(ValueError):
            _run(koff_per_h=-1.0)
