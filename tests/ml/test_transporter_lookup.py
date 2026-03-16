"""Tests for the P-gp / transporter lookup module."""

from omega_pbpk.ml.models.adme.transporter_lookup import (
    get_transporter_flags,
    is_pgp_substrate,
)

VERAPAMIL_SMILES = "CC(C)C(CCCN(C)CCC1=CC(=C(C=C1)OC)OC)(C#N)C2=CC(=C(C=C2)OC)OC"


def test_verapamil_is_pgp_by_name():
    assert is_pgp_substrate(drug_name="verapamil") is True


def test_verapamil_is_pgp_by_smiles():
    assert is_pgp_substrate(smiles=VERAPAMIL_SMILES) is True


def test_digoxin_is_pgp_by_name():
    assert is_pgp_substrate(drug_name="digoxin") is True


def test_caffeine_not_pgp():
    assert is_pgp_substrate(drug_name="caffeine") is False


def test_unknown_drug_returns_none():
    assert get_transporter_flags(drug_name="totally_fake_xyz") is None


def test_get_flags_returns_correct_keys():
    flags = get_transporter_flags(drug_name="verapamil")
    assert flags is not None
    assert flags["pgp_substrate"] == 1
