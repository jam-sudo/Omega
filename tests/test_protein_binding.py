"""Tests for saturable protein binding module (Phase 167)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.protein_binding import (
    ProteinBindingResult,
    binding_curve,
    simulate_protein_binding,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    drug_name="TestDrug",
    total_conc_mg_L=10.0,
)


class TestSimulateProteinBinding:
    """Tests for simulate_protein_binding."""

    def test_default_parameters_work(self):
        res = simulate_protein_binding(**DEFAULTS)
        assert isinstance(res, ProteinBindingResult)
        assert res.drug_name == "TestDrug"

    def test_low_concentration_high_binding(self):
        """At low concentration, most drug should be bound."""
        res = simulate_protein_binding(drug_name="DrugA", total_conc_mg_L=0.01)
        assert res.percent_bound > 50.0

    def test_high_concentration_saturation(self):
        """At very high concentration, fup should increase (binding saturates)."""
        low = simulate_protein_binding(drug_name="DrugA", total_conc_mg_L=0.1)
        high = simulate_protein_binding(drug_name="DrugA", total_conc_mg_L=1000.0)
        assert high.fup > low.fup

    def test_fup_in_valid_range(self):
        res = simulate_protein_binding(**DEFAULTS)
        assert 0.0 <= res.fup <= 1.0

    def test_percent_bound_plus_fup_equals_100(self):
        res = simulate_protein_binding(**DEFAULTS)
        assert abs(res.percent_bound + res.fup * 100.0 - 100.0) < 1e-9

    def test_free_plus_bound_equals_total(self):
        res = simulate_protein_binding(**DEFAULTS)
        assert abs(res.free_conc_mg_L + res.bound_conc_mg_L - res.total_conc_mg_L) < 1e-9

    def test_binding_ratio_r_le_n_sites(self):
        res = simulate_protein_binding(**DEFAULTS, n_sites=2)
        assert res.binding_ratio_r <= 2.0 + 1e-9

    def test_saturation_pct_range(self):
        res = simulate_protein_binding(**DEFAULTS)
        assert 0.0 <= res.saturation_pct <= 100.0 + 1e-9

    def test_notes_non_empty(self):
        res = simulate_protein_binding(**DEFAULTS)
        assert isinstance(res.notes, str)
        assert len(res.notes) > 0

    def test_multiple_n_sites_increases_capacity(self):
        """More binding sites should increase binding at same concentration."""
        r1 = simulate_protein_binding(drug_name="D", total_conc_mg_L=10.0, n_sites=1)
        r2 = simulate_protein_binding(drug_name="D", total_conc_mg_L=10.0, n_sites=4)
        assert r2.percent_bound >= r1.percent_bound

    def test_very_high_concentration_fup_approaches_one(self):
        res = simulate_protein_binding(drug_name="D", total_conc_mg_L=1e6)
        assert res.fup > 0.99

    def test_very_low_concentration_high_binding(self):
        res = simulate_protein_binding(drug_name="D", total_conc_mg_L=1e-4)
        assert res.percent_bound > 80.0

    def test_known_numerical_check(self):
        """Verify against hand-computed values for specific parameters."""
        # Parameters: total=10 mg/L, ka=0.5, protein=40 g/L, mw_drug=300, mw_protein=67000, n=1
        # ka_umol = 0.5 * 300 / 1000 = 0.15
        # P_total_umol = 40 / 67000 * 1e6 = 597.01...
        # total_umol = 10 / 300 * 1e6 = 33333.33...
        # a = 0.15, b = 1 + 1*597.01*0.15 - 33333.33*0.15 = 1 + 89.55 - 5000 = -4909.45
        # c = -33333.33
        # disc = 4909.45^2 + 4*0.15*33333.33 = 24102695 + 20000 = 24122695
        # free = (4909.45 + sqrt(24122695)) / 0.3
        # sqrt(24122695) ~ 4911.49
        # free ~ (4909.45 + 4911.49) / 0.3 ~ 32736.5
        # fup ~ 32736.5 / 33333.33 ~ 0.9821
        res = simulate_protein_binding(
            drug_name="Check",
            total_conc_mg_L=10.0,
            ka_L_per_mg=0.5,
            protein_conc_g_L=40.0,
            mw_drug=300.0,
            mw_protein=67000.0,
            n_sites=1,
        )
        assert abs(res.fup - 0.9821) < 0.01

    def test_frozen_dataclass(self):
        res = simulate_protein_binding(**DEFAULTS)
        with pytest.raises(AttributeError):
            res.fup = 0.5  # type: ignore[misc]


class TestValidation:
    """Tests for input validation."""

    def test_zero_concentration_raises(self):
        with pytest.raises(ValueError, match="total_conc_mg_L"):
            simulate_protein_binding(drug_name="D", total_conc_mg_L=0.0)

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError, match="total_conc_mg_L"):
            simulate_protein_binding(drug_name="D", total_conc_mg_L=-1.0)

    def test_zero_ka_raises(self):
        with pytest.raises(ValueError, match="ka_L_per_mg"):
            simulate_protein_binding(drug_name="D", total_conc_mg_L=1.0, ka_L_per_mg=0.0)

    def test_negative_ka_raises(self):
        with pytest.raises(ValueError, match="ka_L_per_mg"):
            simulate_protein_binding(drug_name="D", total_conc_mg_L=1.0, ka_L_per_mg=-0.1)

    def test_n_sites_zero_raises(self):
        with pytest.raises(ValueError, match="n_sites"):
            simulate_protein_binding(drug_name="D", total_conc_mg_L=1.0, n_sites=0)

    def test_protein_conc_zero_raises(self):
        with pytest.raises(ValueError, match="protein_conc_g_L"):
            simulate_protein_binding(drug_name="D", total_conc_mg_L=1.0, protein_conc_g_L=0.0)

    def test_mw_drug_zero_raises(self):
        with pytest.raises(ValueError, match="mw_drug"):
            simulate_protein_binding(drug_name="D", total_conc_mg_L=1.0, mw_drug=0.0)

    def test_mw_protein_zero_raises(self):
        with pytest.raises(ValueError, match="mw_protein"):
            simulate_protein_binding(drug_name="D", total_conc_mg_L=1.0, mw_protein=0.0)


class TestBindingCurve:
    """Tests for binding_curve."""

    def test_returns_correct_count(self):
        concs = [1.0, 10.0, 100.0]
        results = binding_curve("D", concs)
        assert len(results) == 3

    def test_each_result_has_correct_drug_name(self):
        results = binding_curve("MyDrug", [1.0, 5.0])
        for r in results:
            assert r.drug_name == "MyDrug"

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            binding_curve("D", [])

    def test_monotone_increasing_fup_at_high_concs(self):
        """As concentration increases enough, fup should increase (saturation)."""
        concs = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        results = binding_curve("D", concs)
        fups = [r.fup for r in results]
        # fup should be non-decreasing (monotone) at these concentrations
        for i in range(1, len(fups)):
            assert fups[i] >= fups[i - 1] - 1e-12

    def test_binding_curve_saturation_pct_valid(self):
        concs = [0.01, 1.0, 100.0, 10000.0]
        results = binding_curve("D", concs)
        for r in results:
            assert 0.0 <= r.saturation_pct <= 100.0 + 1e-9

    def test_binding_curve_fup_valid_range(self):
        concs = [0.01, 1.0, 100.0]
        results = binding_curve("D", concs)
        for r in results:
            assert 0.0 <= r.fup <= 1.0
