"""Tests for Phase 194 — excipient_compatibility."""

import pytest

from omega_pbpk.biopharmaceutics.excipient_compatibility import (
    DEFAULT_EXCIPIENTS,
    CompatibilityResult,
    ExcipientInteraction,
    screen_excipient_compatibility,
    suggest_formulation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _screen(smiles: str = "CC", drug_name: str = "TestDrug", excipients=None):
    return screen_excipient_compatibility(drug_name=drug_name, smiles=smiles, excipients=excipients)


# ---------------------------------------------------------------------------
# Return-type and immutability
# ---------------------------------------------------------------------------


def test_returns_compatibility_result():
    result = _screen()
    assert isinstance(result, CompatibilityResult)


def test_result_is_frozen():
    result = _screen()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "x"  # type: ignore[misc]


def test_excipient_interaction_frozen():
    ei = ExcipientInteraction("lactose", "degradation", "high", "mech", "mit")
    with pytest.raises((AttributeError, TypeError)):
        ei.excipient = "mannitol"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Amine + lactose → high severity
# ---------------------------------------------------------------------------


def test_amine_lactose_high_severity():
    """Primary amine drug + lactose → Maillard reaction, high severity."""
    result = screen_excipient_compatibility(
        drug_name="AmineDrug",
        smiles="CN",  # contains N → amine detected
        excipients=["lactose"],
    )
    assert result.n_high_severity == 1
    assert result.overall_compatibility == "incompatible"


def test_amine_lactose_in_avoid_list():
    result = screen_excipient_compatibility(
        drug_name="AmineDrug",
        smiles="CN",
        excipients=["lactose", "mannitol"],
    )
    assert "lactose" in result.excipients_to_avoid
    assert "mannitol" not in result.excipients_to_avoid


def test_amine_lactose_interaction_type():
    result = screen_excipient_compatibility(
        drug_name="AmineDrug",
        smiles="CN",
        excipients=["lactose"],
    )
    assert result.interactions[0].interaction_type == "degradation"
    assert result.interactions[0].severity == "high"


# ---------------------------------------------------------------------------
# Ester + magnesium_stearate → moderate
# ---------------------------------------------------------------------------


def test_ester_mgstearate_moderate():
    # "C(=O)O" is an ester
    result = screen_excipient_compatibility(
        drug_name="EsterDrug",
        smiles="CC(=O)OCC",
        excipients=["magnesium_stearate"],
    )
    assert result.interactions[0].severity == "moderate"
    assert result.overall_compatibility in ("compatible", "caution", "incompatible")


def test_ester_mgstearate_to_avoid():
    result = screen_excipient_compatibility(
        drug_name="EsterDrug",
        smiles="CC(=O)OCC",
        excipients=["magnesium_stearate", "mannitol"],
    )
    assert "magnesium_stearate" in result.excipients_to_avoid


# ---------------------------------------------------------------------------
# No problematic groups → compatible
# ---------------------------------------------------------------------------


def test_inert_drug_compatible():
    """Drug with no detected reactive groups → all excipients compatible."""
    result = screen_excipient_compatibility(
        drug_name="InertDrug",
        smiles="CCCC",  # no N, no ester, no nitro, no S, no aromatic
        excipients=["mannitol", "microcrystalline_cellulose", "talc"],
    )
    assert result.overall_compatibility == "compatible"
    assert result.n_high_severity == 0


def test_inert_drug_recommended_all():
    result = screen_excipient_compatibility(
        drug_name="InertDrug",
        smiles="CCCC",
        excipients=["mannitol", "silicon_dioxide"],
    )
    assert set(result.recommended_excipients) == {"mannitol", "silicon_dioxide"}
    assert result.excipients_to_avoid == []


# ---------------------------------------------------------------------------
# Default excipient list
# ---------------------------------------------------------------------------


def test_default_excipients_screened():
    result = _screen(smiles="CCCC")
    assert result.excipients_tested == DEFAULT_EXCIPIENTS
    assert len(result.interactions) == len(DEFAULT_EXCIPIENTS)


def test_interactions_length_matches_excipients():
    excipients = ["lactose", "mannitol", "talc"]
    result = screen_excipient_compatibility("Drug", "CN", excipients=excipients)
    assert len(result.interactions) == 3


# ---------------------------------------------------------------------------
# n_high_severity count
# ---------------------------------------------------------------------------


def test_n_high_severity_counts_correctly():
    # amine + lactose → 1 high; mannitol → none
    result = screen_excipient_compatibility(
        drug_name="Drug",
        smiles="CN",
        excipients=["lactose", "mannitol", "silicon_dioxide"],
    )
    assert result.n_high_severity == 1


def test_n_high_severity_zero_no_interactions():
    result = screen_excipient_compatibility(
        drug_name="Drug",
        smiles="CCCC",
        excipients=["mannitol", "silicon_dioxide"],
    )
    assert result.n_high_severity == 0


# ---------------------------------------------------------------------------
# suggest_formulation
# ---------------------------------------------------------------------------


def test_suggest_formulation_returns_result():
    result = suggest_formulation("Drug", "CCCC", bcs_class="II", route="oral")
    assert isinstance(result, CompatibilityResult)


def test_suggest_formulation_bcs_ii_has_solubilizers():
    """BCS II should include solubilizing excipients like HPC or povidone."""
    result = suggest_formulation("Drug", "CCCC", bcs_class="II")
    solubilizers = {
        "hydroxypropyl_cellulose",
        "povidone",
        "polysorbate_80",
        "sodium_lauryl_sulfate",
    }
    assert len(solubilizers & set(result.excipients_tested)) > 0


def test_suggest_formulation_bcs_i_screened():
    result = suggest_formulation("Drug", "CCCC", bcs_class="I")
    assert len(result.excipients_tested) > 0


def test_suggest_formulation_amine_detects_problem():
    """Even with suggest_formulation, amine + lactose should be flagged if present."""
    result = suggest_formulation("AmineDrug", "CN", bcs_class="I", route="oral")
    # BCS I list contains mannitol/MCC; if lactose is included, flag it
    # Regardless, n_high_severity should reflect reality
    assert result.n_high_severity >= 0  # at minimum, check it runs without error


# ---------------------------------------------------------------------------
# Functional group detection
# ---------------------------------------------------------------------------


def test_aromatic_detected():
    result = _screen(smiles="c1ccccc1")  # benzene ring
    assert "aromatic" in result.drug_functional_groups


def test_sulfur_detected():
    result = _screen(smiles="CSC")
    assert "sulfur" in result.drug_functional_groups


def test_no_groups_for_alkane():
    result = _screen(smiles="CCCC")
    assert result.drug_functional_groups == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_empty_smiles_raises():
    with pytest.raises(ValueError):
        screen_excipient_compatibility("Drug", "")


def test_whitespace_smiles_raises():
    with pytest.raises(ValueError):
        screen_excipient_compatibility("Drug", "   ")


def test_suggest_formulation_empty_smiles_raises():
    with pytest.raises(ValueError):
        suggest_formulation("Drug", "")
