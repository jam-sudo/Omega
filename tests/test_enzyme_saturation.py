"""Tests for Michaelis-Menten enzyme kinetics and saturation."""

from __future__ import annotations

import pytest

from omega_pbpk.core.enzyme_saturation import (
    MMPKResult,
    apparent_km_with_inhibitor,
    dose_proportionality_mm,
    mm_velocity,
    saturation_fraction,
    simulate_mm_pk,
)


class TestMMVelocity:
    def test_s_equals_km_gives_half_vmax(self):
        km = 10.0
        vmax = 100.0
        v = mm_velocity(substrate_conc_uM=km, vmax_nmol_min=vmax, km_uM=km)
        assert v == pytest.approx(vmax / 2.0)

    def test_s_much_greater_than_km_approaches_vmax(self):
        km = 1.0
        vmax = 100.0
        v = mm_velocity(substrate_conc_uM=1000.0, vmax_nmol_min=vmax, km_uM=km)
        assert v > 0.99 * vmax

    def test_zero_substrate_zero_velocity(self):
        v = mm_velocity(substrate_conc_uM=0.0, vmax_nmol_min=100.0, km_uM=10.0)
        assert v == pytest.approx(0.0)

    def test_positive_values(self):
        v = mm_velocity(substrate_conc_uM=5.0, vmax_nmol_min=50.0, km_uM=10.0)
        assert v > 0.0

    def test_invalid_km_zero(self):
        with pytest.raises(ValueError):
            mm_velocity(substrate_conc_uM=5.0, vmax_nmol_min=100.0, km_uM=0.0)

    def test_invalid_vmax_zero(self):
        with pytest.raises(ValueError):
            mm_velocity(substrate_conc_uM=5.0, vmax_nmol_min=0.0, km_uM=10.0)

    def test_invalid_negative_substrate(self):
        with pytest.raises(ValueError):
            mm_velocity(substrate_conc_uM=-1.0, vmax_nmol_min=100.0, km_uM=10.0)


class TestSaturationFraction:
    def test_s_equals_km_gives_0_5(self):
        km = 10.0
        f = saturation_fraction(substrate_conc_uM=km, km_uM=km)
        assert f == pytest.approx(0.5)

    def test_zero_substrate_zero_saturation(self):
        f = saturation_fraction(substrate_conc_uM=0.0, km_uM=10.0)
        assert f == pytest.approx(0.0)

    def test_high_substrate_approaches_1(self):
        f = saturation_fraction(substrate_conc_uM=10000.0, km_uM=1.0)
        assert f > 0.999

    def test_saturation_in_0_1(self):
        for s in [0.1, 1.0, 5.0, 50.0, 500.0]:
            f = saturation_fraction(substrate_conc_uM=s, km_uM=10.0)
            assert 0.0 <= f <= 1.0

    def test_invalid_km_zero(self):
        with pytest.raises(ValueError):
            saturation_fraction(substrate_conc_uM=5.0, km_uM=0.0)


class TestApparentKm:
    def test_competitive_increases_km(self):
        km_app = apparent_km_with_inhibitor(
            km_uM=10.0, ki_uM=5.0, inhibitor_conc_uM=5.0, inhibition_type="competitive"
        )
        assert km_app > 10.0
        assert km_app == pytest.approx(20.0)

    def test_uncompetitive_decreases_km(self):
        km_app = apparent_km_with_inhibitor(
            km_uM=10.0, ki_uM=5.0, inhibitor_conc_uM=5.0, inhibition_type="uncompetitive"
        )
        assert km_app < 10.0
        assert km_app == pytest.approx(5.0)

    def test_mixed_inhibition(self):
        km_app = apparent_km_with_inhibitor(
            km_uM=10.0, ki_uM=5.0, inhibitor_conc_uM=5.0, inhibition_type="mixed"
        )
        # Km_app = 10 * 2 / 1.5 ≈ 13.33
        assert km_app == pytest.approx(10.0 * 2.0 / 1.5, rel=1e-3)

    def test_zero_inhibitor_unchanged(self):
        km_app = apparent_km_with_inhibitor(
            km_uM=10.0, ki_uM=5.0, inhibitor_conc_uM=0.0, inhibition_type="competitive"
        )
        assert km_app == pytest.approx(10.0)

    def test_invalid_inhibition_type(self):
        with pytest.raises(ValueError):
            apparent_km_with_inhibitor(
                km_uM=10.0, ki_uM=5.0, inhibitor_conc_uM=5.0, inhibition_type="noncompetitive"
            )

    def test_invalid_km_zero(self):
        with pytest.raises(ValueError):
            apparent_km_with_inhibitor(km_uM=0.0, ki_uM=5.0, inhibitor_conc_uM=5.0)

    def test_invalid_ki_zero(self):
        with pytest.raises(ValueError):
            apparent_km_with_inhibitor(km_uM=10.0, ki_uM=0.0, inhibitor_conc_uM=5.0)


class TestDoseProportionalityMM:
    def test_returns_list_of_dicts(self):
        results = dose_proportionality_mm(doses_mg=[10.0, 100.0, 1000.0])
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_correct_number_of_results(self):
        doses = [10.0, 50.0, 100.0, 500.0]
        results = dose_proportionality_mm(doses_mg=doses)
        assert len(results) == len(doses)

    def test_high_dose_non_proportional_auc(self):
        """High dose should give disproportionately higher AUC (non-linear)."""
        doses = [10.0, 1000.0]
        results = dose_proportionality_mm(doses_mg=doses, km_mg_L=5.0, vmax_mg_h=50.0)
        # Dose-normalized AUC should increase at higher doses (saturation increases AUC)
        dna_low = results[0]["dose_normalized_auc"]
        dna_high = results[1]["dose_normalized_auc"]
        assert dna_high > dna_low

    def test_low_dose_linear_regime(self):
        """Very low dose (S << Km) should give near-linear AUC."""
        doses = [0.01, 0.02]  # Very low doses
        results = dose_proportionality_mm(doses_mg=doses, km_mg_L=100.0, vmax_mg_h=100.0)
        # AUC should roughly double when dose doubles
        auc_ratio = results[1]["auc_mg_h_L"] / results[0]["auc_mg_h_L"]
        assert auc_ratio == pytest.approx(2.0, rel=0.05)

    def test_dict_keys_present(self):
        results = dose_proportionality_mm(doses_mg=[100.0])
        expected_keys = {
            "dose_mg",
            "cmax_mg_L",
            "cl_eff_L_per_h",
            "auc_mg_h_L",
            "dose_normalized_auc",
        }
        assert set(results[0].keys()) == expected_keys

    def test_invalid_km_zero(self):
        with pytest.raises(ValueError):
            dose_proportionality_mm(doses_mg=[100.0], km_mg_L=0.0)

    def test_invalid_vmax_zero(self):
        with pytest.raises(ValueError):
            dose_proportionality_mm(doses_mg=[100.0], vmax_mg_h=0.0)


class TestSimulateMMPK:
    def test_returns_mmpk_result(self):
        result = simulate_mm_pk(
            drug_name="Drug",
            dose_mg=100.0,
            vd_L=50.0,
            km_mg_L=10.0,
            vmax_mg_h=100.0,
        )
        assert isinstance(result, MMPKResult)

    def test_cmax_positive(self):
        result = simulate_mm_pk(
            drug_name="Drug",
            dose_mg=100.0,
            vd_L=50.0,
            km_mg_L=10.0,
            vmax_mg_h=100.0,
        )
        assert result.cmax > 0.0

    def test_auc_positive(self):
        result = simulate_mm_pk(
            drug_name="Drug",
            dose_mg=100.0,
            vd_L=50.0,
            km_mg_L=10.0,
            vmax_mg_h=100.0,
        )
        assert result.auc > 0.0

    def test_higher_dose_higher_saturation(self):
        """Higher dose should lead to higher saturation at Cmax."""
        r_low = simulate_mm_pk(
            drug_name="Drug",
            dose_mg=10.0,
            vd_L=50.0,
            km_mg_L=10.0,
            vmax_mg_h=100.0,
        )
        r_high = simulate_mm_pk(
            drug_name="Drug",
            dose_mg=500.0,
            vd_L=50.0,
            km_mg_L=10.0,
            vmax_mg_h=100.0,
        )
        assert r_high.saturation_at_cmax > r_low.saturation_at_cmax

    def test_higher_dose_non_linear_t_half(self):
        """Higher dose (saturating) → longer initial t_half."""
        r_low = simulate_mm_pk(
            drug_name="Drug",
            dose_mg=1.0,
            vd_L=50.0,
            km_mg_L=5.0,
            vmax_mg_h=50.0,
        )
        r_high = simulate_mm_pk(
            drug_name="Drug",
            dose_mg=500.0,
            vd_L=50.0,
            km_mg_L=5.0,
            vmax_mg_h=50.0,
        )
        assert r_high.t_half_initial_h > r_low.t_half_initial_h

    def test_iv_initial_conc_equals_dose_over_vd(self):
        dose = 200.0
        vd = 40.0
        result = simulate_mm_pk(
            drug_name="Drug",
            dose_mg=dose,
            vd_L=vd,
            km_mg_L=10.0,
            vmax_mg_h=50.0,
            route="iv",
        )
        assert result.c_plasma_mg_L[0] == pytest.approx(dose / vd)

    def test_oral_initial_conc_zero(self):
        result = simulate_mm_pk(
            drug_name="Drug",
            dose_mg=100.0,
            vd_L=50.0,
            km_mg_L=10.0,
            vmax_mg_h=100.0,
            route="oral",
        )
        assert result.c_plasma_mg_L[0] == pytest.approx(0.0)

    def test_saturation_at_cmax_in_0_1(self):
        result = simulate_mm_pk(
            drug_name="Drug",
            dose_mg=100.0,
            vd_L=50.0,
            km_mg_L=10.0,
            vmax_mg_h=100.0,
        )
        assert 0.0 <= result.saturation_at_cmax <= 1.0

    def test_invalid_km_zero(self):
        with pytest.raises(ValueError):
            simulate_mm_pk(
                drug_name="Drug",
                dose_mg=100.0,
                vd_L=50.0,
                km_mg_L=0.0,
                vmax_mg_h=100.0,
            )

    def test_invalid_vmax_zero(self):
        with pytest.raises(ValueError):
            simulate_mm_pk(
                drug_name="Drug",
                dose_mg=100.0,
                vd_L=50.0,
                km_mg_L=10.0,
                vmax_mg_h=0.0,
            )

    def test_drug_name_preserved(self):
        result = simulate_mm_pk(
            drug_name="TestMed",
            dose_mg=100.0,
            vd_L=50.0,
            km_mg_L=10.0,
            vmax_mg_h=100.0,
        )
        assert result.drug_name == "TestMed"
