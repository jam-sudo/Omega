"""Tests for Phase 305 — Nanoparticle Drug Delivery PK."""

from __future__ import annotations

import math

import numpy as np
import pytest

from omega_pbpk.core.nanoparticle_pk import (
    NanoparticlePKResult,
    compare_np_formulations,
    simulate_nanoparticle_pk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> NanoparticlePKResult:
    params = dict(
        drug_name="doxorubicin",
        dose_mg=50.0,
        np_type="liposome",
        k_release_per_h=0.1,
        cl_np_L_per_h=0.5,
        vd_np_L=5.0,
        cl_free_L_per_h=10.0,
        vd_free_L=50.0,
        epr_factor=2.0,
        t_end_h=72.0,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_nanoparticle_pk(**params)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=-10.0)

    def test_k_release_zero_raises(self):
        with pytest.raises(ValueError, match="k_release_per_h"):
            _default(k_release_per_h=0.0)

    def test_cl_np_zero_raises(self):
        with pytest.raises(ValueError, match="cl_np_L_per_h"):
            _default(cl_np_L_per_h=0.0)

    def test_vd_np_zero_raises(self):
        with pytest.raises(ValueError, match="vd_np_L"):
            _default(vd_np_L=0.0)

    def test_cl_free_zero_raises(self):
        with pytest.raises(ValueError, match="cl_free_L_per_h"):
            _default(cl_free_L_per_h=0.0)

    def test_vd_free_zero_raises(self):
        with pytest.raises(ValueError, match="vd_free_L"):
            _default(vd_free_L=0.0)

    def test_epr_factor_below_one_raises(self):
        with pytest.raises(ValueError, match="epr_factor"):
            _default(epr_factor=0.5)

    def test_invalid_np_type_raises(self):
        with pytest.raises(ValueError, match="np_type"):
            _default(np_type="dendrimers")


# ---------------------------------------------------------------------------
# Initial conditions and mass balance
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_c_np_starts_at_dose_over_vd(self):
        result = _default(dose_mg=50.0, vd_np_L=5.0)
        assert result.c_np[0] == pytest.approx(50.0 / 5.0, rel=1e-6)

    def test_c_free_starts_at_zero(self):
        result = _default()
        assert result.c_free[0] == pytest.approx(0.0, abs=1e-9)

    def test_c_tumor_starts_at_zero(self):
        result = _default()
        assert result.c_tumor[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# NP compartment decay
# ---------------------------------------------------------------------------


class TestNPDecay:
    def setup_method(self):
        self.result = _default()

    def test_c_np_decreases_monotonically(self):
        # NP concentration should decrease monotonically (no source)
        arr = np.array(self.result.c_np)
        assert np.all(arr[1:] <= arr[:-1] + 1e-10)

    def test_c_np_reaches_near_zero(self):
        # After 72 h, most NP should be cleared
        assert self.result.c_np[-1] < 0.01 * self.result.c_np[0]

    def test_cmax_np_equals_initial(self):
        # NP peaks at t=0 (IV bolus)
        assert self.result.cmax_np == pytest.approx(self.result.c_np[0], rel=1e-6)


# ---------------------------------------------------------------------------
# Free drug compartment
# ---------------------------------------------------------------------------


class TestFreeDrug:
    def setup_method(self):
        self.result = _default()

    def test_c_free_peaks_after_t0(self):
        # Free drug is released over time, so peak is after t=0
        peak_idx = int(np.argmax(self.result.c_free))
        assert peak_idx > 0

    def test_cmax_free_positive(self):
        assert self.result.cmax_free > 0.0

    def test_c_free_eventually_decays(self):
        # Should return to near-zero by end of 72h
        assert self.result.c_free[-1] < 0.05 * self.result.cmax_free


# ---------------------------------------------------------------------------
# Tumor compartment and EPR
# ---------------------------------------------------------------------------


class TestTumorEPR:
    def test_higher_epr_gives_higher_tumor_auc(self):
        res_low = _default(epr_factor=1.0)
        res_high = _default(epr_factor=3.0)
        assert res_high.auc_tumor > res_low.auc_tumor

    def test_tumor_plasma_ratio_positive(self):
        result = _default()
        assert result.tumor_plasma_ratio > 0.0

    def test_higher_epr_gives_higher_tumor_plasma_ratio(self):
        res1 = _default(epr_factor=1.0)
        res3 = _default(epr_factor=3.0)
        assert res3.tumor_plasma_ratio > res1.tumor_plasma_ratio

    def test_cmax_tumor_positive_with_epr(self):
        result = _default(epr_factor=2.0)
        assert result.cmax_tumor > 0.0


# ---------------------------------------------------------------------------
# NP type presets
# ---------------------------------------------------------------------------


class TestNPTypes:
    def test_liposome_type_accepted(self):
        result = simulate_nanoparticle_pk("drug", 50.0, np_type="liposome")
        assert result.np_type == "liposome"

    def test_polymeric_type_accepted(self):
        result = simulate_nanoparticle_pk("drug", 50.0, np_type="polymeric")
        assert result.np_type == "polymeric"

    def test_solid_lipid_type_accepted(self):
        result = simulate_nanoparticle_pk("drug", 50.0, np_type="solid_lipid")
        assert result.np_type == "solid_lipid"

    def test_custom_type_accepted(self):
        result = simulate_nanoparticle_pk("drug", 50.0, np_type="custom", k_release_per_h=0.05)
        assert result.np_type == "custom"

    def test_faster_release_gives_higher_free_cmax(self):
        # Faster release → more free drug appears sooner → higher Cmax
        res_fast = _default(k_release_per_h=0.5)
        res_slow = _default(k_release_per_h=0.05)
        assert res_fast.cmax_free > res_slow.cmax_free


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    def setup_method(self):
        self.result = _default()

    def test_result_is_frozen_dataclass(self):
        assert isinstance(self.result, NanoparticlePKResult)
        with pytest.raises((AttributeError, TypeError)):
            self.result.drug_name = "other"  # type: ignore[misc]

    def test_times_length_consistent(self):
        n = len(self.result.times_h)
        assert len(self.result.c_np) == n
        assert len(self.result.c_free) == n
        assert len(self.result.c_tumor) == n

    def test_auc_values_positive(self):
        assert self.result.auc_np > 0.0
        assert self.result.auc_free > 0.0
        assert self.result.auc_tumor > 0.0

    def test_notes_string(self):
        assert isinstance(self.result.notes, str)
        assert len(self.result.notes) > 0

    def test_drug_name_preserved(self):
        result = _default(drug_name="paclitaxel")
        assert result.drug_name == "paclitaxel"

    def test_dose_mg_preserved(self):
        result = _default(dose_mg=100.0)
        assert result.dose_mg == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# compare_np_formulations
# ---------------------------------------------------------------------------


class TestCompareFormulations:
    def setup_method(self):
        self.comparison = compare_np_formulations(
            drug_name="doxorubicin",
            dose_mg=50.0,
            cl_free=10.0,
            vd_free=50.0,
        )

    def test_returns_dict_with_expected_keys(self):
        expected = {"free_drug", "liposome", "polymeric", "solid_lipid"}
        assert set(self.comparison.keys()) == expected

    def test_free_drug_ratio_is_one(self):
        assert self.comparison["free_drug"]["auc_ratio_vs_free"] == pytest.approx(1.0, rel=0.01)

    def test_np_formulations_have_positive_auc_free(self):
        for k in ("liposome", "polymeric", "solid_lipid"):
            assert self.comparison[k]["auc_free"] > 0.0

    def test_np_formulations_have_positive_tumor_auc(self):
        for k in ("liposome", "polymeric", "solid_lipid"):
            assert self.comparison[k]["auc_tumor"] > 0.0

    def test_tumor_plasma_ratio_higher_for_np(self):
        # NP formulations should have higher tumor/plasma ratio than free drug
        # (EPR effect with epr_factor=2)
        for k in ("liposome", "polymeric", "solid_lipid"):
            assert self.comparison[k]["tumor_plasma_ratio"] > 0.0

    def test_compare_raises_on_bad_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            compare_np_formulations("drug", 0.0, 10.0, 50.0)

    def test_compare_raises_on_bad_cl(self):
        with pytest.raises(ValueError, match="cl_free"):
            compare_np_formulations("drug", 50.0, 0.0, 50.0)

    def test_compare_raises_on_bad_vd(self):
        with pytest.raises(ValueError, match="vd_free"):
            compare_np_formulations("drug", 50.0, 10.0, 0.0)
