"""Unit tests for the Param range validation gateway (_param_guard).

Validates that out-of-range ML/plugin predictions are caught before being
passed to the ODE solver, preventing solver divergence or silent errors.
"""

from __future__ import annotations

import pytest

from omega_pbpk.validation._param_guard import ParamViolation, check_drug_params

# ---------------------------------------------------------------------------
# Fixtures: valid & invalid param sets
# ---------------------------------------------------------------------------

VALID_PARAMS = dict(
    clint_hepatic_L_per_h=10.0,
    clint_gut_L_per_h=0.5,
    fup=0.5,
    rbp=1.0,
    mw=300.0,
    logP=2.0,
    peff=2.0,
    ka_per_h=1.5,
    kp={"liver": 5.0, "kidney": 2.0, "muscle": 1.0},
)


# ---------------------------------------------------------------------------
# Happy-path: all valid params pass without exception
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidParams:
    def test_valid_params_no_violations(self):
        violations = check_drug_params(**VALID_PARAMS, raise_on_violation=False)
        assert violations == []

    def test_valid_params_no_raise(self):
        check_drug_params(**VALID_PARAMS, raise_on_violation=True)  # no exception

    def test_none_params_skipped(self):
        """None values for any param are silently skipped."""
        violations = check_drug_params(
            clint_hepatic_L_per_h=None,
            fup=None,
            raise_on_violation=False,
        )
        assert violations == []

    def test_no_params_no_violations(self):
        violations = check_drug_params(raise_on_violation=False)
        assert violations == []


# ---------------------------------------------------------------------------
# fup boundary cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFupBounds:
    def test_fup_zero_raises(self):
        with pytest.raises(ValueError, match="fup"):
            check_drug_params(fup=0.0)

    def test_fup_negative_raises(self):
        with pytest.raises(ValueError, match="fup"):
            check_drug_params(fup=-0.1)

    def test_fup_exactly_one_valid(self):
        check_drug_params(fup=1.0)  # no exception

    def test_fup_above_one_raises(self):
        with pytest.raises(ValueError, match="fup"):
            check_drug_params(fup=1.001)

    def test_fup_minimum_boundary_valid(self):
        check_drug_params(fup=1e-4)  # exactly at lower bound


# ---------------------------------------------------------------------------
# CLint boundary cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClintBounds:
    def test_negative_clint_hepatic_raises(self):
        with pytest.raises(ValueError, match="clint_hepatic_L_per_h"):
            check_drug_params(clint_hepatic_L_per_h=-1.0)

    def test_zero_clint_valid(self):
        check_drug_params(clint_hepatic_L_per_h=0.0)  # zero clearance is valid

    def test_clint_above_max_raises(self):
        with pytest.raises(ValueError, match="clint_hepatic_L_per_h"):
            check_drug_params(clint_hepatic_L_per_h=600.0)

    def test_clint_at_max_boundary_valid(self):
        check_drug_params(clint_hepatic_L_per_h=500.0)


# ---------------------------------------------------------------------------
# Kp boundary cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKpBounds:
    def test_negative_kp_raises(self):
        with pytest.raises(ValueError, match="kp\\[liver\\]"):
            check_drug_params(kp={"liver": -0.5})

    def test_zero_kp_raises(self):
        with pytest.raises(ValueError, match="kp\\[brain\\]"):
            check_drug_params(kp={"brain": 0.0})

    def test_very_large_kp_raises(self):
        with pytest.raises(ValueError, match="kp\\[adipose\\]"):
            check_drug_params(kp={"adipose": 10000.0})

    def test_valid_kp_dict_passes(self):
        check_drug_params(kp={"liver": 5.0, "kidney": 2.0})  # no exception

    def test_kp_at_max_boundary_valid(self):
        check_drug_params(kp={"liver": 1000.0})  # exactly at upper bound


# ---------------------------------------------------------------------------
# Multiple violations reported together
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultipleViolations:
    def test_multiple_violations_all_reported(self):
        violations = check_drug_params(
            fup=-0.1,
            clint_hepatic_L_per_h=-5.0,
            kp={"liver": -1.0},
            raise_on_violation=False,
        )
        assert len(violations) == 3

    def test_multiple_violations_raise_lists_all(self):
        with pytest.raises(ValueError) as exc_info:
            check_drug_params(fup=2.0, mw=-10.0)
        msg = str(exc_info.value)
        assert "fup" in msg
        assert "mw" in msg

    def test_violation_str_format(self):
        violations = check_drug_params(fup=5.0, raise_on_violation=False)
        assert len(violations) == 1
        v = violations[0]
        assert isinstance(v, ParamViolation)
        assert v.param == "fup"
        assert "5" in str(v)


# ---------------------------------------------------------------------------
# MW, logP, rbp bounds
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOtherBounds:
    def test_low_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            check_drug_params(mw=10.0)

    def test_very_high_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            check_drug_params(mw=5000.0)

    def test_extreme_logP_raises(self):
        with pytest.raises(ValueError, match="logP"):
            check_drug_params(logP=20.0)

    def test_valid_logP_range(self):
        check_drug_params(logP=-5.0)  # hydrophilic extreme
        check_drug_params(logP=10.0)  # lipophilic extreme

    def test_rbp_below_min_raises(self):
        with pytest.raises(ValueError, match="rbp"):
            check_drug_params(rbp=0.0)
