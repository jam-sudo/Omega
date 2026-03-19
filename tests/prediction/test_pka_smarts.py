# tests/prediction/test_pka_smarts.py
"""RDKit SMARTS-based functional group detection in pKa predictor.

Covers:
  - Known acid/base/neutral drugs from gold-24
  - Critical regression: diazepam must NOT be classified as amine
  - Caffeine (xanthine) must NOT be classified as imidazole base
"""

import pytest

from omega_pbpk.prediction.pka_predictor import _detect_functional_group

# (smiles, expected_group_name_or_None)
_CASES = [
    # === Carboxylic acids ===
    ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", "carboxylic_acid"),  # ibuprofen
    ("NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl", "carboxylic_acid"),  # furosemide
    ("OC(=O)CC1(CN)CCCCC1", "carboxylic_acid"),  # gabapentin (COOH present)
    # === Enol-lactone ===
    ("CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "enol_lactone"),  # warfarin (4-OH-coumarin)
    # === Secondary amines (bases) ===
    ("CC(C)NCC(O)COc1cccc2ccccc12", "amine_secondary"),  # propranolol
    ("CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "amine_secondary"),  # atenolol
    ("COCCc1ccc(OCC(O)CNC(C)C)cc1", "amine_secondary"),  # metoprolol
    # === Primary amines ===
    ("C[C@@H](N)Cc1ccccc1", "amine_primary"),  # d-amphetamine
    # === Critical neutrals — must return None ===
    # Diazepam: N=C (azomethine) + amide N → all N are non-basic
    ("CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", None),  # diazepam
    # Caffeine: xanthine ring, all N fully substituted (no NH)
    ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", None),  # caffeine
    # Fluconazole: triazole N are aromatic, pKa ~2 → effectively neutral
    ("OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F", None),  # fluconazole (triazoles)
]


@pytest.mark.parametrize("smiles,expected_group", _CASES)
def test_detect_functional_group(smiles, expected_group):
    group, pka = _detect_functional_group(smiles)
    assert group == expected_group, (
        f"SMILES={smiles!r}: got group={group!r}, expected={expected_group!r}"
    )
    if expected_group is not None:
        assert pka is not None, f"pKa must not be None when group={expected_group!r}"
    else:
        assert pka is None, f"pKa must be None when no group detected, got {pka}"


def test_diazepam_not_amine():
    """Critical: diazepam azomethine N=C must NOT be detected as amine."""
    group, _ = _detect_functional_group("CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21")
    assert group is None, f"Diazepam classified as {group!r} — azomethine guard failed"


def test_caffeine_not_imidazole_base():
    """Caffeine xanthine has no free NH — must not be detected as imidazole."""
    group, _ = _detect_functional_group("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
    assert group is None, f"Caffeine classified as {group!r} — all N are substituted"


def test_ibuprofen_is_carboxylic_acid():
    """Ibuprofen COOH must be detected (feeds into acid Kp D-fix)."""
    group, pka = _detect_functional_group("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
    assert group == "carboxylic_acid"
    assert 3.0 < pka < 6.0, f"Carboxylic acid pKa should be 3–6, got {pka}"


def test_warfarin_enol_lactone():
    """Warfarin 4-OH-coumarin must be detected as enol_lactone (not phenol or acid)."""
    group, pka = _detect_functional_group("CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O")
    assert group == "enol_lactone", f"Warfarin: expected enol_lactone, got {group!r}"
    assert pka is not None
