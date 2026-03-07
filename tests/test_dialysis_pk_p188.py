"""Tests for Phase 188 — dialysis_pk_p188.py."""

import pytest

from omega_pbpk.core.dialysis_pk_p188 import (
    DialysisSessionResult,
    compare_dialysis_modalities,
    simulate_dialysis_session,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DEFAULT_KWARGS = dict(
    drug_name="VancomycinP188",
    dose_mg=1000.0,
    cl_renal_L_per_h=0.5,
    vd_L=50.0,
    cl_dialysis_L_per_h=15.0,
    session_duration_h=4.0,
    interdialysis_h=44.0,
    n_sessions=3,
    modality="HD",
)


def _make_result(**overrides) -> DialysisSessionResult:
    kw = {**_DEFAULT_KWARGS, **overrides}
    return simulate_dialysis_session(**kw)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dialysis_session_result(self):
        r = _make_result()
        assert isinstance(r, DialysisSessionResult)

    def test_drug_name_preserved(self):
        r = _make_result(drug_name="Gentamicin")
        assert r.drug_name == "Gentamicin"

    def test_dose_mg_preserved(self):
        r = _make_result(dose_mg=500.0)
        assert r.dose_mg == pytest.approx(500.0)

    def test_modality_preserved(self):
        r = _make_result(modality="PD")
        assert r.modality == "PD"

    def test_vd_L_preserved(self):
        r = _make_result(vd_L=40.0)
        assert r.vd_L == pytest.approx(40.0)

    def test_cl_dialysis_preserved(self):
        r = _make_result(cl_dialysis_L_per_h=12.0)
        assert r.cl_dialysis_L_per_h == pytest.approx(12.0)

    def test_notes_is_string(self):
        r = _make_result()
        assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# Concentration profile
# ---------------------------------------------------------------------------


class TestConcentrationProfile:
    def test_times_h_is_list(self):
        r = _make_result()
        assert isinstance(r.times_h, list)

    def test_c_plasma_is_list(self):
        r = _make_result()
        assert isinstance(r.c_plasma_mg_L, list)

    def test_c_plasma_starts_positive(self):
        r = _make_result()
        assert r.c_plasma_mg_L[0] > 0.0

    def test_times_start_at_zero(self):
        r = _make_result()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_same_length_times_concentrations(self):
        r = _make_result()
        assert len(r.times_h) == len(r.c_plasma_mg_L)

    def test_concentrations_non_negative(self):
        r = _make_result()
        for c in r.c_plasma_mg_L:
            assert c >= 0.0


# ---------------------------------------------------------------------------
# Cmax / Cmin relationships
# ---------------------------------------------------------------------------


class TestCmaxCmin:
    def test_cmax_positive(self):
        r = _make_result()
        assert r.cmax_mg_L > 0.0

    def test_cmax_greater_than_cmin_pre(self):
        r = _make_result()
        assert r.cmax_mg_L >= r.cmin_pre_dialysis_mg_L

    def test_cmax_greater_than_cmin_post(self):
        r = _make_result()
        assert r.cmax_mg_L >= r.cmin_post_dialysis_mg_L

    def test_cmin_pre_greater_than_cmin_post(self):
        # Dialysis removes drug, so post-session < pre-session
        r = _make_result(cl_dialysis_L_per_h=15.0)
        assert r.cmin_pre_dialysis_mg_L >= r.cmin_post_dialysis_mg_L


# ---------------------------------------------------------------------------
# Fraction removed
# ---------------------------------------------------------------------------


class TestFractionRemoved:
    def test_fraction_removed_in_range(self):
        r = _make_result()
        assert 0.0 <= r.fraction_removed_per_session <= 1.0

    def test_higher_cl_dialysis_higher_fraction_removed(self):
        r_low = _make_result(cl_dialysis_L_per_h=5.0)
        r_high = _make_result(cl_dialysis_L_per_h=25.0)
        assert r_high.fraction_removed_per_session > r_low.fraction_removed_per_session

    def test_zero_cl_dialysis_and_zero_cl_renal_zero_fraction_removed(self):
        # With no dialysis clearance AND no renal clearance, nothing is removed during session
        r = simulate_dialysis_session(
            drug_name="Test",
            dose_mg=100.0,
            cl_renal_L_per_h=0.0,
            vd_L=20.0,
            cl_dialysis_L_per_h=0.0,
            session_duration_h=4.0,
            interdialysis_h=44.0,
            n_sessions=2,
            modality="none",
        )
        assert r.fraction_removed_per_session == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Supplemental dose
# ---------------------------------------------------------------------------


class TestSupplementalDose:
    def test_supplement_zero_when_no_target(self):
        r = _make_result(target_Cmin_mg_L=0.0)
        assert r.supplement_dose_mg == pytest.approx(0.0)

    def test_supplement_positive_when_target_high(self):
        # Set target very high to ensure post-dialysis < target
        r = _make_result(target_Cmin_mg_L=1000.0)
        assert r.supplement_dose_mg > 0.0

    def test_supplement_zero_when_target_already_met(self):
        # Very low target that's always met
        r = _make_result(target_Cmin_mg_L=1e-10)
        assert r.supplement_dose_mg == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compare_dialysis_modalities
# ---------------------------------------------------------------------------


class TestCompareDialysisModalities:
    def test_returns_list(self):
        results = compare_dialysis_modalities(
            drug_name="DrugX",
            dose_mg=500.0,
            vd_L=40.0,
        )
        assert isinstance(results, list)

    def test_returns_three_results(self):
        results = compare_dialysis_modalities(
            drug_name="DrugX",
            dose_mg=500.0,
            vd_L=40.0,
        )
        assert len(results) == 3

    def test_all_dialysis_session_results(self):
        results = compare_dialysis_modalities(
            drug_name="DrugX",
            dose_mg=500.0,
            vd_L=40.0,
        )
        for r in results:
            assert isinstance(r, DialysisSessionResult)

    def test_modalities_include_hd_pd_none(self):
        results = compare_dialysis_modalities(
            drug_name="DrugX",
            dose_mg=500.0,
            vd_L=40.0,
        )
        modalities = {r.modality for r in results}
        assert modalities == {"HD", "PD", "none"}

    def test_hd_removes_more_than_pd(self):
        results = compare_dialysis_modalities(
            drug_name="DrugX",
            dose_mg=500.0,
            vd_L=40.0,
            hd_cl_L_per_h=15.0,
            pd_cl_L_per_h=3.0,
        )
        hd = next(r for r in results if r.modality == "HD")
        pd = next(r for r in results if r.modality == "PD")
        assert hd.fraction_removed_per_session > pd.fraction_removed_per_session


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _make_result(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            _make_result(dose_mg=-100.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            _make_result(vd_L=0.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError):
            _make_result(vd_L=-10.0)

    def test_negative_cl_renal_raises(self):
        with pytest.raises(ValueError):
            _make_result(cl_renal_L_per_h=-1.0)
