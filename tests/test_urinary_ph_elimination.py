"""Tests for clinical/urinary_ph_elimination.py — Phase 523."""

import pytest

from omega_pbpk.clinical.urinary_ph_elimination import (
    UrinaryPhResult,
    compare_acidic_alkaline,
    ionized_fraction,
    reabsorption_fraction,
    renal_cl_adjusted,
    simulate_urinary_ph_effect,
)

# ---------------------------------------------------------------------------
# ionized_fraction
# ---------------------------------------------------------------------------


class TestIonizedFraction:
    def test_weak_acid_at_pka_is_half(self):
        """At pH == pKa, exactly 50% ionized."""
        fi = ionized_fraction(ph=5.0, pka=5.0, drug_type="weak_acid")
        assert abs(fi - 0.5) < 1e-9

    def test_weak_acid_high_ph_mostly_ionized(self):
        """Weak acid at pH >> pKa → nearly all ionized."""
        fi = ionized_fraction(ph=9.0, pka=4.0, drug_type="weak_acid")
        assert fi > 0.99

    def test_weak_acid_low_ph_mostly_unionized(self):
        """Weak acid at pH << pKa → mostly un-ionized."""
        fi = ionized_fraction(ph=1.0, pka=5.0, drug_type="weak_acid")
        assert fi < 0.01

    def test_weak_base_at_pka_is_half(self):
        """Weak base at pH == pKa → 50% ionized."""
        fi = ionized_fraction(ph=8.0, pka=8.0, drug_type="weak_base")
        assert abs(fi - 0.5) < 1e-9

    def test_weak_base_low_ph_mostly_ionized(self):
        """Weak base at pH << pKa → nearly all ionized."""
        fi = ionized_fraction(ph=2.0, pka=8.0, drug_type="weak_base")
        assert fi > 0.99

    def test_weak_base_high_ph_mostly_unionized(self):
        """Weak base at pH >> pKa → mostly un-ionized."""
        fi = ionized_fraction(ph=12.0, pka=7.0, drug_type="weak_base")
        assert fi < 0.01

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            ionized_fraction(ph=7.0, pka=5.0, drug_type="neutral")

    def test_return_in_range(self):
        for ph in [4.0, 5.0, 6.0, 7.0, 8.0]:
            fi = ionized_fraction(ph, pka=6.5, drug_type="weak_acid")
            assert 0.0 <= fi <= 1.0


# ---------------------------------------------------------------------------
# reabsorption_fraction
# ---------------------------------------------------------------------------


class TestReabsorptionFraction:
    def test_high_ionization_low_reabsorption(self):
        """Alkaline urine → weak acid mostly ionized → little reabsorption."""
        f = reabsorption_fraction(ph_urine=8.0, pka=4.5, drug_type="weak_acid")
        assert f < 0.05

    def test_low_ionization_high_reabsorption(self):
        """Acidic urine → weak acid mostly un-ionized → high reabsorption."""
        f = reabsorption_fraction(ph_urine=5.0, pka=9.0, drug_type="weak_acid")
        assert f > 0.90

    def test_max_reabsorption_capped(self):
        """Reabsorption cannot exceed f_reabsorption_max."""
        f = reabsorption_fraction(
            ph_urine=5.0, pka=10.0, drug_type="weak_acid", f_reabsorption_max=0.8
        )
        assert f <= 0.8

    def test_result_in_range(self):
        for ph in [5.0, 6.0, 7.0, 8.0]:
            f = reabsorption_fraction(ph, pka=6.5, drug_type="weak_base")
            assert 0.0 <= f <= 0.95


# ---------------------------------------------------------------------------
# renal_cl_adjusted
# ---------------------------------------------------------------------------


class TestRenalClAdjusted:
    def test_alkaline_urine_increases_cl_for_weak_acid(self):
        """Alkaline urine → more ionization → less reabsorption → higher CL."""
        cl_acid = renal_cl_adjusted(7.2, 0.0, ph_urine=5.0, pka=6.0, drug_type="weak_acid")
        cl_alk = renal_cl_adjusted(7.2, 0.0, ph_urine=8.0, pka=6.0, drug_type="weak_acid")
        assert cl_alk > cl_acid

    def test_acidic_urine_increases_cl_for_weak_base(self):
        """Acidic urine → more ionization of weak base → higher CL."""
        cl_acid = renal_cl_adjusted(7.2, 0.0, ph_urine=5.0, pka=8.0, drug_type="weak_base")
        cl_alk = renal_cl_adjusted(7.2, 0.0, ph_urine=8.0, pka=8.0, drug_type="weak_base")
        assert cl_acid > cl_alk

    def test_secretion_adds_to_cl(self):
        """Secretion increases effective renal CL."""
        cl_no_sec = renal_cl_adjusted(7.2, 0.0, ph_urine=7.0, pka=6.0, drug_type="weak_acid")
        cl_sec = renal_cl_adjusted(7.2, 2.0, ph_urine=7.0, pka=6.0, drug_type="weak_acid")
        assert cl_sec > cl_no_sec

    def test_zero_filtration_zero_cl_when_fully_reabsorbed(self):
        """Zero filtration + zero secretion → zero CL regardless of pH."""
        cl = renal_cl_adjusted(0.0, 0.0, ph_urine=7.0, pka=6.0, drug_type="weak_acid")
        assert cl == 0.0


# ---------------------------------------------------------------------------
# simulate_urinary_ph_effect
# ---------------------------------------------------------------------------


class TestSimulateUrinaryPhEffect:
    def test_returns_four_results_by_default(self):
        results = simulate_urinary_ph_effect(
            drug_name="TestDrug",
            dose_mg=100.0,
            pka=6.5,
            drug_type="weak_acid",
            cl_nonrenal_L_per_h=2.0,
            vd_L=50.0,
        )
        assert len(results) == 4

    def test_result_type(self):
        results = simulate_urinary_ph_effect(
            drug_name="Aspirin",
            dose_mg=325.0,
            pka=3.5,
            drug_type="weak_acid",
            cl_nonrenal_L_per_h=1.5,
            vd_L=30.0,
        )
        assert all(isinstance(r, UrinaryPhResult) for r in results)

    def test_ph_values_match(self):
        ph_vals = [5.0, 7.4]
        results = simulate_urinary_ph_effect(
            drug_name="Drug",
            dose_mg=100.0,
            pka=6.0,
            drug_type="weak_acid",
            cl_nonrenal_L_per_h=1.0,
            vd_L=40.0,
            ph_urine_values=ph_vals,
        )
        assert [r.ph_urine for r in results] == ph_vals

    def test_weak_acid_alkaline_urine_higher_cl(self):
        """Weak acid: alkaline urine → higher renal CL."""
        results = simulate_urinary_ph_effect(
            drug_name="WeakAcid",
            dose_mg=100.0,
            pka=6.0,
            drug_type="weak_acid",
            cl_nonrenal_L_per_h=1.0,
            vd_L=40.0,
            ph_urine_values=[5.0, 8.0],
        )
        assert results[1].cl_renal_L_per_h > results[0].cl_renal_L_per_h

    def test_weak_acid_alkaline_urine_lower_auc(self):
        """Weak acid: alkaline → higher CL → shorter t½ → lower AUC."""
        results = simulate_urinary_ph_effect(
            drug_name="WeakAcid",
            dose_mg=100.0,
            pka=6.0,
            drug_type="weak_acid",
            cl_nonrenal_L_per_h=1.0,
            vd_L=40.0,
            ph_urine_values=[5.0, 8.0],
        )
        assert results[1].auc < results[0].auc  # alkaline < acidic

    def test_weak_base_alkaline_urine_lower_cl(self):
        """Weak base: alkaline → less ionized → more reabsorption → lower renal CL."""
        results = simulate_urinary_ph_effect(
            drug_name="WeakBase",
            dose_mg=100.0,
            pka=8.5,
            drug_type="weak_base",
            cl_nonrenal_L_per_h=1.0,
            vd_L=40.0,
            ph_urine_values=[5.0, 8.0],
        )
        assert results[0].cl_renal_L_per_h > results[1].cl_renal_L_per_h

    def test_weak_base_alkaline_urine_higher_auc(self):
        """Weak base: alkaline → lower CL → longer t½ → higher AUC."""
        results = simulate_urinary_ph_effect(
            drug_name="WeakBase",
            dose_mg=100.0,
            pka=8.5,
            drug_type="weak_base",
            cl_nonrenal_L_per_h=1.0,
            vd_L=40.0,
            ph_urine_values=[5.0, 8.0],
        )
        assert results[1].auc > results[0].auc

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            simulate_urinary_ph_effect(
                drug_name="X",
                dose_mg=100.0,
                pka=5.0,
                drug_type="amphoteric",
                cl_nonrenal_L_per_h=1.0,
                vd_L=30.0,
            )

    def test_auc_positive(self):
        results = simulate_urinary_ph_effect(
            drug_name="Drug",
            dose_mg=200.0,
            pka=7.0,
            drug_type="weak_acid",
            cl_nonrenal_L_per_h=3.0,
            vd_L=70.0,
        )
        assert all(r.auc > 0 for r in results)

    def test_t_half_positive(self):
        results = simulate_urinary_ph_effect(
            drug_name="Drug",
            dose_mg=100.0,
            pka=5.5,
            drug_type="weak_base",
            cl_nonrenal_L_per_h=2.0,
            vd_L=60.0,
        )
        assert all(r.t_half_h > 0 for r in results)


# ---------------------------------------------------------------------------
# compare_acidic_alkaline
# ---------------------------------------------------------------------------


class TestCompareAcidicAlkaline:
    def test_returns_both_results(self):
        out = compare_acidic_alkaline(
            drug_name="Aspirin",
            dose_mg=325.0,
            pka=3.5,
            drug_type="weak_acid",
            cl_nonrenal_L_per_h=1.5,
            vd_L=30.0,
        )
        assert "acidic_result" in out
        assert "alkaline_result" in out

    def test_ph_values_correct(self):
        out = compare_acidic_alkaline(
            drug_name="Drug",
            dose_mg=100.0,
            pka=5.0,
            drug_type="weak_acid",
            cl_nonrenal_L_per_h=1.0,
            vd_L=40.0,
        )
        assert out["acidic_result"].ph_urine == 5.0
        assert out["alkaline_result"].ph_urine == 8.0

    def test_auc_ratio_present(self):
        out = compare_acidic_alkaline(
            drug_name="Drug",
            dose_mg=100.0,
            pka=7.0,
            drug_type="weak_base",
            cl_nonrenal_L_per_h=2.0,
            vd_L=50.0,
        )
        assert "auc_ratio" in out
        assert out["auc_ratio"] > 0

    def test_clinical_relevance_string(self):
        out = compare_acidic_alkaline(
            drug_name="Drug",
            dose_mg=100.0,
            pka=6.5,
            drug_type="weak_acid",
            cl_nonrenal_L_per_h=1.0,
            vd_L=40.0,
        )
        assert isinstance(out["clinical_relevance"], str)
        assert len(out["clinical_relevance"]) > 0

    def test_t_half_ratio_present(self):
        out = compare_acidic_alkaline(
            drug_name="Drug",
            dose_mg=100.0,
            pka=5.0,
            drug_type="weak_acid",
            cl_nonrenal_L_per_h=1.0,
            vd_L=40.0,
        )
        assert "t_half_ratio" in out
        assert out["t_half_ratio"] > 0
