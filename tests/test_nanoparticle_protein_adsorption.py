"""Tests for prediction/nanoparticle_protein_adsorption.py — Phase 821."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.nanoparticle_protein_adsorption import (
    NanoProteinAdsorptionResult,
    predict_adsorption,
    screen_proteins,
)

# ---------------------------------------------------------------------------
# Default parameters used across tests
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "nanoparticle_name": "PLGA-NP",
    "protein_name": "Albumin",
    "nanoparticle_diameter_nm": 150.0,
    "protein_conc_mol_per_L": 5e-5,  # 50 µM albumin
    "ka_L_per_mol": 1e6,
    "kd_mol_per_L": 1e-7,
    "protein_mw_kDa": 66.5,
    "protein_diameter_nm": 7.0,
}

PROTEIN_LIBRARY = [
    {
        "name": "Albumin",
        "conc_mol_per_L": 5e-5,
        "ka_L_per_mol": 1e6,
        "kd_mol_per_L": 1e-7,
        "mw_kDa": 66.5,
        "diameter_nm": 7.0,
    },
    {
        "name": "Fibrinogen",
        "conc_mol_per_L": 1e-6,
        "ka_L_per_mol": 2e6,
        "kd_mol_per_L": 5e-7,
        "mw_kDa": 340.0,
        "diameter_nm": 45.0,
    },
    {
        "name": "IgG",
        "conc_mol_per_L": 5e-8,
        "ka_L_per_mol": 5e5,
        "kd_mol_per_L": 1e-6,
        "mw_kDa": 150.0,
        "diameter_nm": 10.0,
    },
]


def _run(**overrides) -> NanoProteinAdsorptionResult:
    params = {**DEFAULT_PARAMS, **overrides}
    return predict_adsorption(**params)


# ---------------------------------------------------------------------------
# Return type and field sanity
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = _run()
        assert isinstance(result, NanoProteinAdsorptionResult)

    def test_nanoparticle_name_preserved(self):
        result = _run(nanoparticle_name="LNP-001")
        assert result.nanoparticle_name == "LNP-001"

    def test_protein_name_preserved(self):
        result = _run(protein_name="Fibrinogen")
        assert result.protein_name == "Fibrinogen"

    def test_surface_area_positive(self):
        result = _run()
        assert result.surface_area_nm2 > 0.0

    def test_surface_area_formula(self):
        # 4 * pi * (d/2)^2
        d = 100.0
        result = _run(nanoparticle_diameter_nm=d)
        expected = 4.0 * math.pi * (d / 2.0) ** 2
        assert abs(result.surface_area_nm2 - expected) < 1e-6

    def test_notes_is_str(self):
        result = _run()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Langmuir isotherm behaviour
# ---------------------------------------------------------------------------


class TestLangmuirIsotherm:
    def test_equilibrium_coverage_in_unit_interval(self):
        result = _run()
        assert 0.0 <= result.equilibrium_coverage <= 1.0

    def test_higher_conc_higher_coverage(self):
        low = _run(protein_conc_mol_per_L=1e-8)
        high = _run(protein_conc_mol_per_L=1e-3)
        assert high.equilibrium_coverage > low.equilibrium_coverage

    def test_zero_conc_gives_zero_coverage(self):
        result = _run(protein_conc_mol_per_L=0.0)
        assert result.equilibrium_coverage == pytest.approx(0.0)

    def test_very_high_conc_coverage_approaches_one(self):
        result = _run(protein_conc_mol_per_L=1.0)  # 1 mol/L >> Kd
        assert result.equilibrium_coverage > 0.99

    def test_coverage_at_kd_is_half(self):
        # When C = Kd, theta = 0.5
        kd = 1e-6
        result = _run(protein_conc_mol_per_L=kd, kd_mol_per_L=kd)
        assert result.equilibrium_coverage == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Adsorption quantities
# ---------------------------------------------------------------------------


class TestAdsorptionQuantities:
    def test_adsorbed_molecules_non_negative(self):
        result = _run()
        assert result.adsorbed_molecules_per_np >= 0.0

    def test_corona_thickness_non_negative(self):
        result = _run()
        assert result.corona_thickness_nm >= 0.0

    def test_zero_conc_zero_corona(self):
        result = _run(protein_conc_mol_per_L=0.0)
        assert result.corona_thickness_nm == pytest.approx(0.0)

    def test_larger_np_more_adsorbed_molecules(self):
        small = _run(nanoparticle_diameter_nm=50.0)
        large = _run(nanoparticle_diameter_nm=200.0)
        assert large.adsorbed_molecules_per_np > small.adsorbed_molecules_per_np

    def test_max_adsorption_positive(self):
        result = _run()
        assert result.max_adsorption_mol_per_nm2 > 0.0


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------


class TestStabilityClassification:
    def test_valid_classifications(self):
        valid = {"stable", "moderate", "unstable"}
        for conc in [0.0, 1e-7, 1e-3]:
            result = _run(protein_conc_mol_per_L=conc)
            assert result.stability_classification in valid

    def test_high_coverage_is_stable(self):
        # Very high concentration → theta > 0.7 → stable
        result = _run(protein_conc_mol_per_L=1.0)
        assert result.stability_classification == "stable"

    def test_zero_coverage_is_unstable(self):
        result = _run(protein_conc_mol_per_L=0.0)
        assert result.stability_classification == "unstable"


# ---------------------------------------------------------------------------
# Physical quantities
# ---------------------------------------------------------------------------


class TestPhysicalQuantities:
    def test_binding_energy_negative_for_favorable(self):
        # ka >> kd → strong binding → negative dG
        result = _run(ka_L_per_mol=1e8, kd_mol_per_L=1e-9)
        assert result.binding_energy_kJ_per_mol < 0.0

    def test_desorption_rate_positive(self):
        result = _run()
        assert result.desorption_rate_per_h > 0.0


# ---------------------------------------------------------------------------
# screen_proteins
# ---------------------------------------------------------------------------


class TestScreenProteins:
    def test_returns_list(self):
        results = screen_proteins("PLGA-NP", 150.0, PROTEIN_LIBRARY)
        assert isinstance(results, list)
        assert len(results) == len(PROTEIN_LIBRARY)

    def test_all_elements_correct_type(self):
        results = screen_proteins("PLGA-NP", 150.0, PROTEIN_LIBRARY)
        for r in results:
            assert isinstance(r, NanoProteinAdsorptionResult)

    def test_sorted_descending_by_coverage(self):
        results = screen_proteins("PLGA-NP", 150.0, PROTEIN_LIBRARY)
        coverages = [r.equilibrium_coverage for r in results]
        assert coverages == sorted(coverages, reverse=True)

    def test_empty_library_returns_empty(self):
        results = screen_proteins("NP", 100.0, [])
        assert results == []

    def test_nanoparticle_name_propagated(self):
        results = screen_proteins("Gold-NP", 100.0, PROTEIN_LIBRARY)
        for r in results:
            assert r.nanoparticle_name == "Gold-NP"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_invalid_diameter_zero(self):
        with pytest.raises(ValueError, match="nanoparticle_diameter_nm"):
            _run(nanoparticle_diameter_nm=0.0)

    def test_invalid_diameter_negative(self):
        with pytest.raises(ValueError, match="nanoparticle_diameter_nm"):
            _run(nanoparticle_diameter_nm=-10.0)

    def test_invalid_ka_zero(self):
        with pytest.raises(ValueError, match="ka_L_per_mol"):
            _run(ka_L_per_mol=0.0)

    def test_invalid_ka_negative(self):
        with pytest.raises(ValueError, match="ka_L_per_mol"):
            _run(ka_L_per_mol=-1e6)

    def test_invalid_kd_zero(self):
        with pytest.raises(ValueError, match="kd_mol_per_L"):
            _run(kd_mol_per_L=0.0)

    def test_invalid_kd_negative(self):
        with pytest.raises(ValueError, match="kd_mol_per_L"):
            _run(kd_mol_per_L=-1e-7)

    def test_invalid_protein_conc_negative(self):
        with pytest.raises(ValueError, match="protein_conc_mol_per_L"):
            _run(protein_conc_mol_per_L=-1e-5)
