"""Tests for the applicability domain filter."""

from omega_pbpk.ml.applicability import ApplicabilityResult, check_applicability

CAFFEINE_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
CLOPIDOGREL_SMILES = "COC(=O)[C@@H](C1=CC=CC=C1Cl)N2CCC3=C(C2)C=CS3"
IBUPROFEN_SMILES = "CC(C)CC1=CC=C(CC(C)C(=O)O)C=C1"
VERAPAMIL_SMILES = "CC(C)C(CCCN(C)CCC1=CC(=C(C=C1)OC)OC)(C#N)C2=CC(=C(C=C2)OC)OC"


def test_caffeine_is_tractable():
    """Caffeine — small, simple xanthine — should be fully tractable."""
    result = check_applicability(CAFFEINE_SMILES)
    assert result.tractable is True
    assert len(result.flags) == 0
    assert result.confidence == "high"


def test_prodrug_ester_detected():
    """Clopidogrel has a methyl-ester prodrug motif and should be flagged."""
    result = check_applicability(CLOPIDOGREL_SMILES)
    assert result.tractable is False
    assert "prodrug_ester" in result.flags


def test_result_has_confidence():
    """Every result must carry a valid confidence level."""
    result = check_applicability(CAFFEINE_SMILES)
    assert result.confidence in ("high", "medium", "low")


def test_simple_carboxylic_acid_not_flagged():
    """Ibuprofen has -C(=O)OH (carboxylic acid), NOT an ester — should not flag."""
    result = check_applicability(IBUPROFEN_SMILES)
    assert "prodrug_ester" not in result.flags


def test_verapamil_medium_confidence():
    """Verapamil is a known P-gp substrate — should get medium confidence."""
    result = check_applicability(VERAPAMIL_SMILES, drug_name="verapamil")
    assert result.confidence == "medium"
    assert "pgp_substrate" in result.flags


def test_known_untractable_by_name():
    """Drugs in the untractable set should be flagged regardless of SMILES."""
    result = check_applicability(CAFFEINE_SMILES, drug_name="temozolomide")
    assert result.tractable is False
    assert "known_untractable" in result.flags
    assert result.confidence == "low"


def test_invalid_smiles():
    """Garbage SMILES should produce invalid_smiles flag."""
    result = check_applicability("NOT_A_SMILES!!!")
    assert result.tractable is False
    assert "invalid_smiles" in result.flags


def test_result_is_frozen():
    """ApplicabilityResult should be immutable."""
    result = check_applicability(CAFFEINE_SMILES)
    assert isinstance(result, ApplicabilityResult)
    try:
        result.tractable = False  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except AttributeError:
        pass  # expected — frozen dataclass
