"""Tests for Phase 685 — Cardiac Output Effects on PK."""

import pytest

from omega_pbpk.core.cardiac_output_pk_p685 import (
    CardiacOutputPKResult,
    simulate_cardiac_output_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = dict(
    drug_name="DrugX",
    dose_mg=100.0,
    logP=2.5,
    cl_base_L_per_h=30.0,  # high extraction
    vd_base_L=50.0,
)


def _run(**overrides):
    kw = {**_BASE, **overrides}
    return simulate_cardiac_output_pk(**kw)


# ---------------------------------------------------------------------------
# Return type / structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, CardiacOutputPKResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "DrugX"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 100.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_c_plasma_is_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list)
        assert len(r.c_plasma_mg_L) == len(r.times_h)

    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# Valid conditions
# ---------------------------------------------------------------------------


class TestConditions:
    @pytest.mark.parametrize("cond", ["normal", "heart_failure", "exercise", "sepsis"])
    def test_valid_conditions_accepted(self, cond):
        r = _run(condition=cond)
        assert r.condition == cond

    def test_invalid_condition_raises(self):
        with pytest.raises(ValueError, match="condition"):
            _run(condition="unknown")


# ---------------------------------------------------------------------------
# Cardiac output effects
# ---------------------------------------------------------------------------


class TestCOEffects:
    def test_low_co_higher_auc_high_extraction(self):
        """Heart failure (low CO) → lower CL → higher AUC for high-extraction drugs."""
        r_normal = _run(cardiac_output_L_per_min=5.0)
        r_hf = _run(cardiac_output_L_per_min=3.0)
        assert r_hf.auc_mg_h_per_L > r_normal.auc_mg_h_per_L

    def test_high_co_lower_auc_high_extraction(self):
        """Exercise (high CO) → higher CL → lower AUC for high-extraction drugs."""
        r_normal = _run(cardiac_output_L_per_min=5.0)
        r_ex = _run(cardiac_output_L_per_min=15.0)
        assert r_ex.auc_mg_h_per_L < r_normal.auc_mg_h_per_L

    def test_low_extraction_cl_independent_of_co(self):
        """Low-extraction drug: CL doesn't change with CO."""
        r_norm = _run(cl_base_L_per_h=2.0, cardiac_output_L_per_min=5.0)
        r_high = _run(cl_base_L_per_h=2.0, cardiac_output_L_per_min=10.0)
        assert r_norm.cl_effective_L_per_h == r_high.cl_effective_L_per_h

    def test_normal_co_baseline(self):
        r = _run(cardiac_output_L_per_min=5.0)
        assert r.cardiac_output_L_per_min == 5.0


# ---------------------------------------------------------------------------
# PK parameters
# ---------------------------------------------------------------------------


class TestPKParameters:
    def test_cmax_positive(self):
        r = _run()
        assert r.cmax_mg_L > 0

    def test_auc_positive(self):
        r = _run()
        assert r.auc_mg_h_per_L > 0

    def test_t_half_positive(self):
        r = _run()
        assert r.t_half_h > 0

    def test_cl_effective_positive(self):
        r = _run()
        assert r.cl_effective_L_per_h > 0

    def test_vd_effective_positive(self):
        r = _run()
        assert r.vd_effective_L > 0

    def test_tmax_within_range(self):
        r = _run()
        assert 0 <= r.tmax_h <= 24.0


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_double_dose_double_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_mg_L / r1.cmax_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_double_dose_double_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
        assert abs(ratio - 2.0) < 0.1


# ---------------------------------------------------------------------------
# Plasma concentrations non-negative
# ---------------------------------------------------------------------------


class TestPlasmaConcentrations:
    def test_all_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_starts_at_zero(self):
        r = _run()
        assert r.c_plasma_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_base"):
            _run(cl_base_L_per_h=0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_base"):
            _run(vd_base_L=-5)

    def test_negative_co_raises(self):
        with pytest.raises(ValueError, match="cardiac_output"):
            _run(cardiac_output_L_per_min=-1)

    def test_zero_co_raises(self):
        with pytest.raises(ValueError, match="cardiac_output"):
            _run(cardiac_output_L_per_min=0)
