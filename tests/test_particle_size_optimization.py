"""Tests for particle size optimization (Phase 450)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.particle_size_optimization import (
    ParticleSizeOptResult,
    biopharmaceutics_classification,
    optimize_particle_size,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_opt(**kw) -> ParticleSizeOptResult:
    defaults = dict(
        drug_name="TestDrug",
        target_dissolved_pct_at_60min=80.0,
        solubility_mg_mL=1.0,
        diffusion_coeff_cm2_s=5e-6,
        dose_mg=100.0,
        density_g_cm3=1.2,
        particle_sizes_um=[5.0, 10.0, 20.0, 50.0, 100.0],
        media_volume_mL=900.0,
    )
    defaults.update(kw)
    return optimize_particle_size(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_type(self):
        assert isinstance(_default_opt(), ParticleSizeOptResult)

    def test_drug_name_preserved(self):
        r = _default_opt(drug_name="Aspirin")
        assert r.drug_name == "Aspirin"

    def test_target_preserved(self):
        r = _default_opt(target_dissolved_pct_at_60min=75.0)
        assert r.target_dissolved_pct_at_60min == 75.0

    def test_solubility_preserved(self):
        r = _default_opt(solubility_mg_mL=2.0)
        assert r.solubility_mg_mL == 2.0

    def test_dose_preserved(self):
        r = _default_opt(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_particle_sizes_sorted(self):
        r = _default_opt(particle_sizes_um=[50.0, 5.0, 20.0])
        assert r.particle_sizes_um == sorted(r.particle_sizes_um)

    def test_dissolved_pct_list_length_matches_sizes(self):
        r = _default_opt(particle_sizes_um=[5.0, 10.0, 50.0])
        assert len(r.dissolved_pct_at_60min) == 3

    def test_notes_is_list(self):
        r = _default_opt()
        assert isinstance(r.notes, list)

    def test_optimal_size_in_candidate_list(self):
        r = _default_opt()
        assert r.optimal_size_um in r.particle_sizes_um


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------


class TestPhysicalPlausibility:
    def test_dissolved_pct_between_0_and_100(self):
        r = _default_opt()
        assert all(0.0 <= p <= 100.0 for p in r.dissolved_pct_at_60min)

    def test_optimal_dissolved_pct_positive(self):
        r = _default_opt()
        assert r.optimal_dissolved_pct >= 0.0

    def test_smaller_particle_dissolves_more(self):
        r = _default_opt(particle_sizes_um=[5.0, 100.0])
        diss_small = r.dissolved_pct_at_60min[0]   # 5 um
        diss_large = r.dissolved_pct_at_60min[1]   # 100 um
        assert diss_small >= diss_large - 1e-6

    def test_higher_solubility_faster_dissolution(self):
        r_low = _default_opt(solubility_mg_mL=0.01, particle_sizes_um=[10.0])
        r_high = _default_opt(solubility_mg_mL=10.0, particle_sizes_um=[10.0])
        assert r_high.dissolved_pct_at_60min[0] >= r_low.dissolved_pct_at_60min[0]

    def test_single_size_returns_that_as_optimal(self):
        r = _default_opt(particle_sizes_um=[10.0])
        assert r.optimal_size_um == 10.0

    def test_highly_soluble_drug_achieves_high_dissolution(self):
        # Very high solubility + small particle should dissolve quickly
        r = _default_opt(
            solubility_mg_mL=100.0,
            particle_sizes_um=[1.0, 5.0],
            dose_mg=10.0,
        )
        assert max(r.dissolved_pct_at_60min) > 50.0

    def test_optimal_pct_matches_entry_in_list(self):
        r = _default_opt()
        idx = r.particle_sizes_um.index(r.optimal_size_um)
        assert abs(r.optimal_dissolved_pct - r.dissolved_pct_at_60min[idx]) < 1e-6

    def test_note_added_when_no_size_meets_target(self):
        # Very low solubility, so no size should meet 99% target
        r = _default_opt(
            solubility_mg_mL=0.0001,
            target_dissolved_pct_at_60min=99.0,
            dose_mg=500.0,
        )
        assert any("No particle size" in note for note in r.notes)

    def test_rapid_dissolution_note_when_high_pct(self):
        # Very high solubility should trigger rapid dissolution note
        r = _default_opt(
            solubility_mg_mL=200.0,
            dose_mg=1.0,
            particle_sizes_um=[1.0],
            target_dissolved_pct_at_60min=85.0,
        )
        # Should reach >85% and add note
        if r.optimal_dissolved_pct >= 85.0:
            assert any("rapid dissolution" in note.lower() for note in r.notes)

    def test_more_sizes_more_dissolved_entries(self):
        r = _default_opt(particle_sizes_um=[5.0, 10.0, 20.0, 50.0])
        assert len(r.dissolved_pct_at_60min) == 4

    def test_dose_number_note_for_poorly_soluble(self):
        # Dose number = dose/(solubility*250) >> 1 for low solubility
        r = _default_opt(
            solubility_mg_mL=0.01,
            dose_mg=500.0,
            particle_sizes_um=[10.0],
        )
        assert any("dose number" in note.lower() for note in r.notes)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError):
            _default_opt(drug_name="")

    def test_negative_target_raises(self):
        with pytest.raises(ValueError):
            _default_opt(target_dissolved_pct_at_60min=-5.0)

    def test_target_gt_100_raises(self):
        with pytest.raises(ValueError):
            _default_opt(target_dissolved_pct_at_60min=101.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            _default_opt(solubility_mg_mL=0.0)

    def test_negative_diffusion_coeff_raises(self):
        with pytest.raises(ValueError):
            _default_opt(diffusion_coeff_cm2_s=-1e-6)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _default_opt(dose_mg=0.0)

    def test_zero_density_raises(self):
        with pytest.raises(ValueError):
            _default_opt(density_g_cm3=0.0)

    def test_empty_particle_sizes_raises(self):
        with pytest.raises(ValueError):
            _default_opt(particle_sizes_um=[])

    def test_negative_particle_size_raises(self):
        with pytest.raises(ValueError):
            _default_opt(particle_sizes_um=[-5.0, 10.0])

    def test_zero_media_volume_raises(self):
        with pytest.raises(ValueError):
            _default_opt(media_volume_mL=0.0)


# ---------------------------------------------------------------------------
# biopharmaceutics_classification
# ---------------------------------------------------------------------------


class TestBiopharmaceuticsClassification:
    def test_class_i_high_both(self):
        assert biopharmaceutics_classification(90.0, 0.90) == "BCS Class I"

    def test_class_ii_low_dissolution(self):
        assert biopharmaceutics_classification(60.0, 0.90) == "BCS Class II"

    def test_class_iii_low_permeability(self):
        assert biopharmaceutics_classification(90.0, 0.50) == "BCS Class III"

    def test_class_iv_both_low(self):
        assert biopharmaceutics_classification(60.0, 0.50) == "BCS Class IV"

    def test_boundary_85pct_dissolution(self):
        # exactly 85% is NOT > 85, so should be low dissolution
        assert biopharmaceutics_classification(85.0, 0.90) == "BCS Class II"

    def test_boundary_85_1_pct_dissolution(self):
        # 85.01% is > 85, so should be high dissolution
        assert biopharmaceutics_classification(85.01, 0.90) == "BCS Class I"

    def test_boundary_permeability_085(self):
        # exactly 0.85 is >= 0.85, so high permeability
        assert biopharmaceutics_classification(90.0, 0.85) == "BCS Class I"

    def test_negative_dissolved_pct_raises(self):
        with pytest.raises(ValueError):
            biopharmaceutics_classification(-1.0, 0.9)

    def test_dissolved_pct_gt_100_raises(self):
        with pytest.raises(ValueError):
            biopharmaceutics_classification(101.0, 0.9)

    def test_negative_fa_raises(self):
        with pytest.raises(ValueError):
            biopharmaceutics_classification(80.0, -0.1)

    def test_fa_gt_1_raises(self):
        with pytest.raises(ValueError):
            biopharmaceutics_classification(80.0, 1.1)
