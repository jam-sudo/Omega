"""Tests for Phase 338 — Peptide Oral Absorption Predictor."""

import pytest

from omega_pbpk.prediction.peptide_oral_absorption_p338 import (
    PeptideAbsorptionResult,
    predict_peptide_oral_absorption,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_kwargs(**overrides):
    base = dict(
        drug_name="TestPeptide",
        n_residues=5,
        mw_Da=600.0,
        logP=1.0,
        psa_angstrom2=120.0,
        n_hbd=3,
        n_hba=6,
        has_cyclic_structure=False,
        has_n_methylation=False,
        charge_at_ph7=0,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Return type (frozen dataclass)
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_peptide_oral_absorption(**_default_kwargs())
    assert isinstance(result, PeptideAbsorptionResult)


def test_frozen_dataclass():
    result = predict_peptide_oral_absorption(**_default_kwargs())
    with pytest.raises((AttributeError, TypeError)):
        result.fa = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. fa in (0, 1)
# ---------------------------------------------------------------------------


def test_fa_in_range():
    result = predict_peptide_oral_absorption(**_default_kwargs())
    assert 0.0 < result.fa < 1.0


def test_fa_in_range_extreme_values():
    result = predict_peptide_oral_absorption(
        **_default_kwargs(logP=-5.0, psa_angstrom2=300.0, n_hbd=20, n_hba=30)
    )
    assert 0.0 < result.fa < 1.0


# ---------------------------------------------------------------------------
# 3. f_oral in (0, 1)
# ---------------------------------------------------------------------------


def test_f_oral_in_range():
    result = predict_peptide_oral_absorption(**_default_kwargs())
    assert 0.0 <= result.f_oral < 1.0


# ---------------------------------------------------------------------------
# 4. proteolytic_stability in (0, 1]
# ---------------------------------------------------------------------------


def test_proteolytic_stability_in_range():
    result = predict_peptide_oral_absorption(**_default_kwargs())
    assert 0.0 < result.proteolytic_stability <= 1.0


def test_proteolytic_stability_in_range_large_peptide():
    result = predict_peptide_oral_absorption(**_default_kwargs(n_residues=50))
    assert 0.0 < result.proteolytic_stability <= 1.0


# ---------------------------------------------------------------------------
# 5. Cyclic peptide → higher F_oral than linear equivalent
# ---------------------------------------------------------------------------


def test_cyclic_higher_f_oral_than_linear():
    linear = predict_peptide_oral_absorption(
        **_default_kwargs(has_cyclic_structure=False, has_n_methylation=False)
    )
    cyclic = predict_peptide_oral_absorption(
        **_default_kwargs(has_cyclic_structure=True, has_n_methylation=False)
    )
    assert cyclic.f_oral > linear.f_oral


# ---------------------------------------------------------------------------
# 6. N-methylation improves absorption
# ---------------------------------------------------------------------------


def test_n_methylation_improves_absorption():
    without = predict_peptide_oral_absorption(**_default_kwargs(has_n_methylation=False))
    with_nm = predict_peptide_oral_absorption(**_default_kwargs(has_n_methylation=True))
    assert with_nm.f_oral > without.f_oral


# ---------------------------------------------------------------------------
# 7. Larger peptide → lower proteolytic stability
# ---------------------------------------------------------------------------


def test_larger_peptide_lower_proteolytic_stability():
    small = predict_peptide_oral_absorption(**_default_kwargs(n_residues=3))
    large = predict_peptide_oral_absorption(**_default_kwargs(n_residues=20))
    assert large.proteolytic_stability < small.proteolytic_stability


# ---------------------------------------------------------------------------
# 8. lipinski_violations in [0, 5]
# ---------------------------------------------------------------------------


def test_lipinski_violations_in_range():
    result = predict_peptide_oral_absorption(**_default_kwargs())
    assert 0 <= result.lipinski_violations <= 5


def test_lipinski_violations_max():
    result = predict_peptide_oral_absorption(
        **_default_kwargs(
            mw_Da=1500.0,
            logP=6.0,
            psa_angstrom2=200.0,
            n_hbd=10,
            n_hba=15,
        )
    )
    assert result.lipinski_violations == 5


def test_lipinski_violations_zero():
    result = predict_peptide_oral_absorption(
        **_default_kwargs(
            mw_Da=300.0,
            logP=2.0,
            psa_angstrom2=80.0,
            n_hbd=2,
            n_hba=5,
        )
    )
    assert result.lipinski_violations == 0


# ---------------------------------------------------------------------------
# 9. bcs_class in valid set
# ---------------------------------------------------------------------------


def test_bcs_class_valid():
    result = predict_peptide_oral_absorption(**_default_kwargs())
    assert result.bcs_class in {"I", "II", "III", "IV"}


def test_bcs_class_iv_for_low_fa():
    """High PSA and many HBDs → low fa → BCS IV."""
    result = predict_peptide_oral_absorption(
        **_default_kwargs(
            psa_angstrom2=300.0,
            n_hbd=15,
            logP=-3.0,
            n_hba=20,
        )
    )
    assert result.bcs_class == "IV"


def test_bcs_class_i_for_high_fa():
    """Very high logP, low PSA → high fa → BCS I."""
    result = predict_peptide_oral_absorption(
        **_default_kwargs(
            n_residues=2,
            logP=5.0,
            psa_angstrom2=10.0,
            n_hbd=0,
            n_hba=1,
        )
    )
    assert result.bcs_class == "I"


# ---------------------------------------------------------------------------
# 10. absorption_enhancer_needed = True when F_oral < 0.1
# ---------------------------------------------------------------------------


def test_absorption_enhancer_needed_for_low_f_oral():
    result = predict_peptide_oral_absorption(
        **_default_kwargs(
            n_residues=30,
            psa_angstrom2=300.0,
            n_hbd=15,
            logP=-3.0,
            n_hba=20,
        )
    )
    assert result.f_oral < 0.1
    assert result.absorption_enhancer_needed is True


def test_absorption_enhancer_not_needed_for_good_f_oral():
    result = predict_peptide_oral_absorption(
        **_default_kwargs(
            n_residues=2,
            logP=5.0,
            psa_angstrom2=10.0,
            n_hbd=0,
            n_hba=1,
        )
    )
    assert result.f_oral >= 0.1
    assert result.absorption_enhancer_needed is False


# ---------------------------------------------------------------------------
# 11. Validation errors
# ---------------------------------------------------------------------------


def test_validation_n_residues_too_small():
    with pytest.raises(ValueError, match="n_residues"):
        predict_peptide_oral_absorption(**_default_kwargs(n_residues=0))


def test_validation_n_residues_too_large():
    with pytest.raises(ValueError, match="n_residues"):
        predict_peptide_oral_absorption(**_default_kwargs(n_residues=51))


def test_validation_mw_nonpositive():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_peptide_oral_absorption(**_default_kwargs(mw_Da=0.0))


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_peptide_oral_absorption(**_default_kwargs(mw_Da=-100.0))


def test_validation_charge_must_be_int():
    with pytest.raises(ValueError, match="charge_at_ph7"):
        predict_peptide_oral_absorption(**_default_kwargs(charge_at_ph7=1.5))  # type: ignore[arg-type]


def test_validation_n_residues_must_be_int():
    with pytest.raises(ValueError, match="n_residues"):
        predict_peptide_oral_absorption(**_default_kwargs(n_residues=5.5))  # type: ignore[arg-type]
