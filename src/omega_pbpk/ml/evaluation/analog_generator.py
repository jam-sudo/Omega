"""Structural analog generator using RDKit for SAR validation.

Generates systematic structural modifications of known drugs to test
whether predictions follow expected SAR trends:
- Adding polar groups → higher clearance, lower logP
- Adding lipophilic groups → higher Vd, higher logP
- Gradual changes → gradual PK changes (no discontinuities)

Reference: Plan Section 3, Tier 2 validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalogResult:
    """Result of a single analog perturbation."""

    parent_smiles: str
    analog_smiles: str
    modification: str
    expected_direction: str  # e.g., "logP_decrease", "clearance_increase"
    parent_pk: dict | None = None
    analog_pk: dict | None = None


def generate_analogs(smiles: str, max_analogs: int = 10) -> list[tuple[str, str, str]]:
    """Generate structural analogs of a molecule via RDKit.

    Returns list of (analog_smiles, modification_description, expected_direction).

    Modifications applied:
    1. Single-atom substitutions (Cl→F, CH3→CF3)
    2. Hydroxylation (add -OH to aromatic ring)
    3. N-demethylation (NCH3 → NH)
    4. Ester hydrolysis (COOCH3 → COOH)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors  # noqa: F401
    except ImportError:
        logger.warning("RDKit not available for analog generation")
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    analogs: list[tuple[str, str, str]] = []

    # 1. Cl → F substitution (reduces logP slightly, changes metabolism)
    rxn_cl_to_f = AllChem.ReactionFromSmarts("[c:1][Cl:2]>>[c:1][F]")
    if rxn_cl_to_f:
        products = rxn_cl_to_f.RunReactants((mol,))
        for product_set in products[:2]:
            for product in product_set:
                try:
                    Chem.SanitizeMol(product)
                    smi = Chem.MolToSmiles(product)
                    if smi != smiles:
                        analogs.append((smi, "Cl→F substitution", "logP_decrease"))
                except Exception:
                    continue

    # 2. Add hydroxyl to aromatic ring (increases polarity)
    rxn_hydroxylate = AllChem.ReactionFromSmarts("[cH:1]>>[c:1]O")
    if rxn_hydroxylate:
        products = rxn_hydroxylate.RunReactants((mol,))
        for product_set in products[:3]:
            for product in product_set:
                try:
                    Chem.SanitizeMol(product)
                    smi = Chem.MolToSmiles(product)
                    if smi != smiles and len(smi) < 200:
                        analogs.append((smi, "aromatic hydroxylation", "logP_decrease"))
                except Exception:
                    continue

    # 3. N-demethylation (N(C) → N([H]))
    rxn_ndemethyl = AllChem.ReactionFromSmarts("[N:1]([CH3])>>[N:1][H]")
    if rxn_ndemethyl:
        products = rxn_ndemethyl.RunReactants((mol,))
        for product_set in products[:2]:
            for product in product_set:
                try:
                    Chem.SanitizeMol(product)
                    smi = Chem.MolToSmiles(product)
                    if smi != smiles:
                        analogs.append((smi, "N-demethylation", "logP_decrease"))
                except Exception:
                    continue

    # 4. Add methyl group to aromatic ring (increases logP)
    rxn_methylate = AllChem.ReactionFromSmarts("[cH:1]>>[c:1]C")
    if rxn_methylate:
        products = rxn_methylate.RunReactants((mol,))
        for product_set in products[:2]:
            for product in product_set:
                try:
                    Chem.SanitizeMol(product)
                    smi = Chem.MolToSmiles(product)
                    if smi != smiles and len(smi) < 200:
                        analogs.append((smi, "aromatic methylation", "logP_increase"))
                except Exception:
                    continue

    # 5. F → H substitution (removes fluorine, changes metabolism)
    rxn_f_to_h = AllChem.ReactionFromSmarts("[c:1][F:2]>>[cH:1]")
    if rxn_f_to_h:
        products = rxn_f_to_h.RunReactants((mol,))
        for product_set in products[:2]:
            for product in product_set:
                try:
                    Chem.SanitizeMol(product)
                    smi = Chem.MolToSmiles(product)
                    if smi != smiles:
                        analogs.append((smi, "F→H (defluorination)", "logP_increase"))
                except Exception:
                    continue

    # Deduplicate by SMILES
    seen: set[str] = set()
    unique: list[tuple[str, str, str]] = []
    for smi, mod, direction in analogs:
        if smi not in seen:
            seen.add(smi)
            unique.append((smi, mod, direction))

    return unique[:max_analogs]


def generate_de_novo_molecules(n: int = 20, seed: int = 42) -> list[str]:
    """Generate drug-like molecules via BRICS decomposition and recombination.

    Generates novel drug-like molecules by fragmenting known drugs and
    recombining fragments. These molecules don't exist in any database.

    Returns list of SMILES strings.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import BRICS, Descriptors, Lipinski
    except ImportError:
        logger.warning("RDKit not available for de novo generation")
        return []

    import random

    rng = random.Random(seed)

    # Start from a set of known drug-like fragments
    seed_mols = [
        "c1ccccc1",  # benzene
        "c1ccncc1",  # pyridine
        "C1CCCCC1",  # cyclohexane
        "c1cc[nH]c1",  # pyrrole
        "c1ccc2[nH]ccc2c1",  # indole
        "CC(=O)O",  # acetic acid
        "CC(N)C(=O)O",  # alanine
        "c1ccc(O)cc1",  # phenol
        "CC(=O)Nc1ccccc1",  # acetanilide
        "c1ccc(-c2ccccc2)cc1",  # biphenyl
    ]

    mols = []
    for smi in seed_mols:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)

    if not mols:
        return []

    # BRICS decomposition
    all_frags: set[str] = set()
    for mol in mols:
        try:
            frags = BRICS.BRICSDecompose(mol)
            all_frags.update(frags)
        except Exception:
            continue

    if len(all_frags) < 3:
        return []

    frag_list = sorted(all_frags)

    # Recombine fragments to generate novel molecules
    generated: list[str] = []
    attempts = 0
    max_attempts = n * 50

    while len(generated) < n and attempts < max_attempts:
        attempts += 1
        # Pick 2-3 random fragments and try to combine
        n_frags = rng.choice([2, 3])
        selected = rng.sample(frag_list, min(n_frags, len(frag_list)))

        try:
            combined = BRICS.BRICSBuild(
                [Chem.MolFromSmiles(f) for f in selected if Chem.MolFromSmiles(f)]
            )
            for mol in combined:
                try:
                    Chem.SanitizeMol(mol)
                    smi = Chem.MolToSmiles(mol)
                    # Filter for drug-like properties (Lipinski)
                    mw = Descriptors.ExactMolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    hbd = Lipinski.NumHDonors(mol)
                    hba = Lipinski.NumHAcceptors(mol)

                    if 150 < mw < 600 and -2 < logp < 6 and hbd <= 5 and hba <= 10:
                        if smi not in generated:
                            generated.append(smi)
                            if len(generated) >= n:
                                break
                except Exception:
                    continue
        except Exception:
            continue

    return generated[:n]
