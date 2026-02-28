import pytest
from omega_pbpk.validation._param_guard import validate_drug_params


def test_negative_clint_raises():
    with pytest.raises(ValueError, match="clint_hepatic"):
        validate_drug_params(clint_hepatic=-1.0, clint_gut=0, clr=0, fup=0.5, rbp=1.0)


def test_negative_clr_raises():
    with pytest.raises(ValueError, match="clr"):
        validate_drug_params(clint_hepatic=0, clint_gut=0, clr=-5.0, fup=0.5, rbp=1.0)


def test_fup_zero_raises():
    with pytest.raises(ValueError, match="fup"):
        validate_drug_params(clint_hepatic=0, clint_gut=0, clr=0, fup=0.0, rbp=1.0)


def test_fup_above_one_raises():
    with pytest.raises(ValueError, match="fup"):
        validate_drug_params(clint_hepatic=0, clint_gut=0, clr=0, fup=1.5, rbp=1.0)


def test_negative_kp_raises():
    with pytest.raises(ValueError, match="kp"):
        validate_drug_params(
            clint_hepatic=0, clint_gut=0, clr=0, fup=0.5, rbp=1.0, kp={"liver": -0.1}
        )


def test_valid_params_pass():
    validate_drug_params(
        clint_hepatic=10.0,
        clint_gut=5.0,
        clr=2.0,
        fup=0.1,
        rbp=0.9,
        kp={"liver": 2.0, "muscle": 0.5},
        ka=1.5,
    )


def test_extreme_clint_raises():
    with pytest.raises(ValueError, match="clint_hepatic"):
        validate_drug_params(clint_hepatic=99999.0, clint_gut=0, clr=0, fup=0.5, rbp=1.0)
