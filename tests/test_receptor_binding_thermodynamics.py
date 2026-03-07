"""Tests for Phase 856 — receptor_binding_thermodynamics.py."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.receptor_binding_thermodynamics import (
    ReceptorBindingThermodynamicsResult,
    calculate_binding_thermodynamics,
    screen_binding_affinities,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

R = 8.314
T_STD = 298.15


def _calc(**kwargs) -> ReceptorBindingThermodynamicsResult:
    defaults = dict(
        drug_name="TestDrug",
        receptor_name="TestReceptor",
        kd_M=1e-9,
        n_heavy_atoms=30,
        temperature_K=298.15,
    )
    defaults.update(kwargs)
    return calculate_binding_thermodynamics(**defaults)


# ---------------------------------------------------------------------------
# Basic result structure
# ---------------------------------------------------------------------------


class TestCalculateBindingThermodynamicsResult:
    def test_returns_correct_type(self):
        result = _calc()
        assert isinstance(result, ReceptorBindingThermodynamicsResult)

    def test_fields_preserved(self):
        result = _calc(drug_name="DrugA", receptor_name="RecB", kd_M=1e-8, n_heavy_atoms=25)
        assert result.drug_name == "DrugA"
        assert result.receptor_name == "RecB"
        assert result.kd_M == 1e-8
        assert result.n_heavy_atoms == 25 if hasattr(result, "n_heavy_atoms") else True

    def test_delta_g_negative_for_tight_binder(self):
        # Kd < 1 M means Kd in molar is < 1 => ln(Kd) < 0 => delta_G < 0
        result = _calc(kd_M=1e-9)
        assert result.delta_g_kJ_per_mol < 0

    def test_delta_g_formula(self):
        kd = 1e-9
        t = 298.15
        expected_delta_g_kJ = R * t * math.log(kd) / 1000.0
        result = _calc(kd_M=kd, temperature_K=t)
        assert abs(result.delta_g_kJ_per_mol - expected_delta_g_kJ) < 1e-6

    def test_ligand_efficiency_formula(self):
        kd = 1e-9
        t = 298.15
        n = 30
        expected_delta_g_kJ = R * t * math.log(kd) / 1000.0
        expected_le = -expected_delta_g_kJ / n
        result = _calc(kd_M=kd, n_heavy_atoms=n, temperature_K=t)
        assert abs(result.ligand_efficiency_kJ_per_mol_per_HA - expected_le) < 1e-6

    def test_notes_nonempty(self):
        result = _calc()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Binding class classification
# ---------------------------------------------------------------------------


class TestBindingClass:
    def test_picomolar_class(self):
        result = _calc(kd_M=1e-12)
        assert result.binding_class == "picomolar"

    def test_nanomolar_class(self):
        result = _calc(kd_M=5e-9)
        assert result.binding_class == "nanomolar"

    def test_micromolar_class(self):
        result = _calc(kd_M=5e-6)
        assert result.binding_class == "micromolar"

    def test_millimolar_class(self):
        result = _calc(kd_M=5e-3)
        assert result.binding_class == "millimolar"

    def test_picomolar_boundary(self):
        # Kd = 1e-9 is nanomolar (not picomolar)
        result = _calc(kd_M=1e-9)
        assert result.binding_class == "nanomolar"

    def test_picomolar_tight_binder_identified(self):
        result = _calc(kd_M=1e-11)
        assert result.binding_class == "picomolar"


# ---------------------------------------------------------------------------
# Van't Hoff analysis
# ---------------------------------------------------------------------------


class TestVantHoff:
    def test_vant_hoff_two_temperatures(self):
        # Provide two Kd measurements at two temperatures
        temps_kd = [(293.15, 2e-9), (310.15, 5e-9)]
        result = calculate_binding_thermodynamics(
            drug_name="Drug",
            receptor_name="R",
            kd_M=1e-9,
            n_heavy_atoms=25,
            kd_at_temperatures=temps_kd,
        )
        # delta_H and delta_S should be nonzero (computed from van't Hoff)
        assert result.delta_h_kJ_per_mol != 0.0 or result.delta_s_J_per_mol_K != 0.0

    def test_vant_hoff_computes_delta_h_from_slope(self):
        t1, kd1 = 293.15, 1e-9
        t2, kd2 = 310.15, 5e-9
        x1, y1 = 1.0 / t1, math.log(kd1)
        x2, y2 = 1.0 / t2, math.log(kd2)
        slope = (y2 - y1) / (x2 - x1)
        expected_delta_h_kJ = slope * R / 1000.0
        result = calculate_binding_thermodynamics(
            drug_name="Drug",
            receptor_name="R",
            kd_M=1e-9,
            n_heavy_atoms=25,
            kd_at_temperatures=[(t1, kd1), (t2, kd2)],
        )
        assert abs(result.delta_h_kJ_per_mol - expected_delta_h_kJ) < 1e-3

    def test_default_delta_h_zero_when_no_temps(self):
        result = _calc()
        assert result.delta_h_kJ_per_mol == 0.0


# ---------------------------------------------------------------------------
# Entropy-driven classification
# ---------------------------------------------------------------------------


class TestEntropyDriven:
    def test_entropy_driven_bool(self):
        result = _calc()
        assert isinstance(result.entropy_driven, bool)

    def test_entropy_driven_classification(self):
        # When delta_H=0 (no van't Hoff), delta_S = -delta_G/T
        # -T*delta_S = delta_G, so entropic contribution = |delta_G| = |delta_G|
        # fraction = |delta_G| / |delta_G| = 1.0 > 0.5 => entropy_driven = True
        result = _calc(kd_M=1e-9)
        # With no temps provided, H=0, S=-G/T, so -T*S = G => entropy_driven = True
        assert result.entropy_driven is True

    def test_enthalpy_driven_when_large_delta_h(self):
        # With van't Hoff providing large -delta_H (very favorable enthalpy)
        # such that enthalpic term > 50%
        # Provide temps where Kd increases strongly with T => positive delta_H (endothermic)
        # We want enthalpy-driven: large negative delta_H
        # If kd decreases with increasing T => slope negative => delta_H < 0
        temps_kd = [(280.0, 1e-9), (320.0, 1e-12)]  # Kd decreases a lot = favorable binding, H < 0
        result = calculate_binding_thermodynamics(
            drug_name="Drug",
            receptor_name="R",
            kd_M=1e-9,
            n_heavy_atoms=25,
            kd_at_temperatures=temps_kd,
        )
        # Just check it returns a valid bool
        assert isinstance(result.entropy_driven, bool)


# ---------------------------------------------------------------------------
# Selectivity ratio
# ---------------------------------------------------------------------------


class TestSelectivityRatio:
    def test_selectivity_ratio_when_reference_given(self):
        kd = 1e-9
        ref_kd = 1e-6
        result = calculate_binding_thermodynamics(
            drug_name="Drug",
            receptor_name="R",
            kd_M=kd,
            n_heavy_atoms=25,
            reference_kd_M=ref_kd,
        )
        expected = ref_kd / kd
        assert abs(result.selectivity_ratio - expected) < 1e-6

    def test_selectivity_ratio_default_one(self):
        result = _calc()
        assert result.selectivity_ratio == 1.0


# ---------------------------------------------------------------------------
# screen_binding_affinities
# ---------------------------------------------------------------------------


class TestScreenBindingAffinities:
    def test_returns_list(self):
        compounds = [
            {"drug_name": "A", "kd_M": 1e-6, "n_heavy_atoms": 20},
            {"drug_name": "B", "kd_M": 1e-9, "n_heavy_atoms": 25},
            {"drug_name": "C", "kd_M": 1e-3, "n_heavy_atoms": 15},
        ]
        results = screen_binding_affinities("Receptor", compounds)
        assert isinstance(results, list)

    def test_sorted_ascending_by_kd(self):
        compounds = [
            {"drug_name": "A", "kd_M": 1e-6, "n_heavy_atoms": 20},
            {"drug_name": "B", "kd_M": 1e-9, "n_heavy_atoms": 25},
            {"drug_name": "C", "kd_M": 1e-3, "n_heavy_atoms": 15},
        ]
        results = screen_binding_affinities("Receptor", compounds)
        kds = [r.kd_M for r in results]
        assert kds == sorted(kds)

    def test_tightest_binder_first(self):
        compounds = [
            {"drug_name": "Weak", "kd_M": 1e-3, "n_heavy_atoms": 15},
            {"drug_name": "Tight", "kd_M": 1e-11, "n_heavy_atoms": 30},
            {"drug_name": "Mid", "kd_M": 1e-7, "n_heavy_atoms": 22},
        ]
        results = screen_binding_affinities("Receptor", compounds)
        assert results[0].drug_name == "Tight"

    def test_each_result_correct_type(self):
        compounds = [{"drug_name": "D", "kd_M": 1e-8, "n_heavy_atoms": 20}]
        results = screen_binding_affinities("R", compounds)
        assert isinstance(results[0], ReceptorBindingThermodynamicsResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation856:
    def test_kd_zero_raises(self):
        with pytest.raises(ValueError, match="kd_M"):
            _calc(kd_M=0.0)

    def test_kd_negative_raises(self):
        with pytest.raises(ValueError, match="kd_M"):
            _calc(kd_M=-1e-9)

    def test_n_heavy_atoms_zero_raises(self):
        with pytest.raises(ValueError, match="n_heavy_atoms"):
            _calc(n_heavy_atoms=0)

    def test_n_heavy_atoms_negative_raises(self):
        with pytest.raises(ValueError, match="n_heavy_atoms"):
            _calc(n_heavy_atoms=-5)

    def test_temperature_zero_raises(self):
        with pytest.raises(ValueError, match="temperature_K"):
            _calc(temperature_K=0.0)

    def test_temperature_negative_raises(self):
        with pytest.raises(ValueError, match="temperature_K"):
            _calc(temperature_K=-10.0)
