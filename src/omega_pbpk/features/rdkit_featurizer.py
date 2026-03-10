"""RDKit-based molecular featurizer for ADME prediction.

Extracts 15 molecular descriptors from SMILES strings.
Gracefully falls back to zero vectors when RDKit is not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, MolSurf, rdMolDescriptors

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.info("RDKit not available; RDKitFeaturizer will return zero vectors.")


@dataclass(frozen=True)
class FeatureVector:
    """Container for a computed molecular feature vector.

    Attributes:
        descriptors: 1-D float64 array of length 15 with descriptor values.
        feature_names: List of 15 descriptor name strings.
    """

    descriptors: NDArray[np.float64]
    feature_names: list[str]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape of the descriptor array."""
        return self.descriptors.shape


class RDKitFeaturizer:
    """Compute 15 molecular descriptors from a SMILES string.

    When RDKit is installed, real descriptor values are computed.
    When RDKit is not available, a zero vector is returned so that
    downstream models can still run (with degraded accuracy).

    Class attributes:
        FEATURE_NAMES: Ordered list of the 15 descriptor names.
    """

    FEATURE_NAMES: list[str] = [
        "MW",
        "LogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotBonds",
        "AromaticRings",
        "FractionCSP3",
        "MolMR",
        "HeavyAtoms",
        "RingCount",
        "SaturatedRings",
        "HeteroAtoms",
        "NumStereoCenters",
        "BertzCT_norm",
    ]

    def featurize(self, smiles: str) -> FeatureVector:
        """Compute the 15-descriptor feature vector for a single SMILES.

        Args:
            smiles: A valid SMILES string.

        Returns:
            FeatureVector with descriptors of shape (15,) and the
            canonical FEATURE_NAMES list.

        Notes:
            If RDKit is not available, or if the SMILES cannot be parsed,
            a zero vector is returned and a warning is logged.
        """
        n = len(self.FEATURE_NAMES)
        if not HAS_RDKIT:
            return FeatureVector(
                descriptors=np.zeros(n, dtype=np.float64),
                feature_names=list(self.FEATURE_NAMES),
            )

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("RDKitFeaturizer: could not parse SMILES %r; returning zeros.", smiles)
            return FeatureVector(
                descriptors=np.zeros(n, dtype=np.float64),
                feature_names=list(self.FEATURE_NAMES),
            )

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = MolSurf.TPSA(mol)
        hbd = float(Lipinski.NumHDonors(mol))
        hba = float(Lipinski.NumHAcceptors(mol))
        rot_bonds = float(Lipinski.NumRotatableBonds(mol))
        aromatic_rings = float(Descriptors.NumAromaticRings(mol))
        fraction_csp3 = float(Descriptors.FractionCSP3(mol))
        mol_mr = float(Crippen.MolMR(mol))
        heavy_atoms = float(mol.GetNumHeavyAtoms())
        ring_count = float(rdMolDescriptors.CalcNumRings(mol))
        saturated_rings = float(rdMolDescriptors.CalcNumSaturatedRings(mol))
        hetero_atoms = float(rdMolDescriptors.CalcNumHeteroatoms(mol))
        stereo_centers = float(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
        bertz_ct = float(Descriptors.BertzCT(mol))
        # Normalise BertzCT so it is order-of-magnitude comparable to other descriptors
        bertz_ct_norm = bertz_ct / 1000.0

        descriptors = np.array(
            [
                mw,
                logp,
                tpsa,
                hbd,
                hba,
                rot_bonds,
                aromatic_rings,
                fraction_csp3,
                mol_mr,
                heavy_atoms,
                ring_count,
                saturated_rings,
                hetero_atoms,
                stereo_centers,
                bertz_ct_norm,
            ],
            dtype=np.float64,
        )

        return FeatureVector(
            descriptors=descriptors,
            feature_names=list(self.FEATURE_NAMES),
        )

    def featurize_batch(self, smiles_list: list[str]) -> NDArray[np.float64]:
        """Compute descriptor matrix for a list of SMILES strings.

        Args:
            smiles_list: List of N SMILES strings.

        Returns:
            Array of shape (N, 15) where each row is the descriptor vector
            for the corresponding SMILES (zeros for unparseable entries).
        """
        n = len(smiles_list)
        n_features = len(self.FEATURE_NAMES)
        result = np.zeros((n, n_features), dtype=np.float64)
        for i, smi in enumerate(smiles_list):
            result[i] = self.featurize(smi).descriptors
        return result
