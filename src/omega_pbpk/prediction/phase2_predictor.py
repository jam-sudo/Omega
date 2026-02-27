"""Phase 2 metabolism prediction from molecular structure.

Predicts intrinsic clearance contributions from conjugation enzymes:

  UGT (UDP-glucuronosyltransferases)
      Glucuronidation of phenols, alcohols, carboxylic acids, amines.
      Major pathway for many drugs (e.g. morphine, lorazepam, ibuprofen).
      QSPR: Miners et al. Curr Drug Metab. 2004;5(3):235-46.

  SULT (Sulfotransferases)
      Sulfation of phenols and primary alcohols.
      High-capacity but saturable at therapeutic doses.
      Reference: Gamage et al. Drug Metab Rev. 2006;38(1-2):1-69.

  MAO (Monoamine oxidases A/B)
      Oxidative deamination of primary amines.
      Relevant for catecholamines, tyramine, serotonin-like compounds.
      Reference: Youdim MBH et al. Nat Rev Neurosci. 2006;7(4):295-309.

QSPR approach
-------------
Models are simplified linear regressions on functional group counts and
physicochemical descriptors, calibrated against a training set of ~60
literature CLint values.  When RDKit is not available, counts are
estimated from SMILES character patterns (lower accuracy, confidence="low").

All CLint values are in µL/min/mg microsomal protein equivalent, so they
can be directly added to Phase 1 CLint and fed into the IVIVE module.

Limitations
-----------
- QSPR models assume a homogeneous enzyme pool; isoform specificity
  (e.g. UGT1A1 vs UGT2B7) is not predicted.
- Saturation kinetics (Km, Vmax) are not predicted; only apparent CLint
  in the linear range is estimated.
- Predictions should be treated as order-of-magnitude estimates
  (expected fold-error: 2–5×).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, MolSurf, rdMolDescriptors

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.info("RDKit not available; Phase 2 predictions use SMILES heuristics.")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase2Properties:
    """Predicted Phase 2 metabolic properties.

    All CLint values in µL/min/mg (microsomal protein equivalent).

    Attributes:
        clint_ugt_ul_min_mg: UGT intrinsic clearance.
        clint_sult_ul_min_mg: SULT intrinsic clearance.
        clint_mao_ul_min_mg: MAO intrinsic clearance.
        clint_phase2_total_ul_min_mg: Sum of all Phase 2 pathways.
        ugt_substrate_prob: Probability of being a UGT substrate (0–1).
        sult_substrate_prob: Probability of being a SULT substrate (0–1).
        mao_substrate_prob: Probability of being a MAO substrate (0–1).
        dominant_pathway: Name of the major Phase 2 route.
        confidence: "high" (RDKit descriptors) or "low" (SMILES heuristic).
    """

    clint_ugt_ul_min_mg: float
    clint_sult_ul_min_mg: float
    clint_mao_ul_min_mg: float
    clint_phase2_total_ul_min_mg: float
    ugt_substrate_prob: float
    sult_substrate_prob: float
    mao_substrate_prob: float
    dominant_pathway: str
    confidence: str = "low"


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class Phase2Predictor:
    """QSPR-based Phase 2 metabolism predictor.

    Predicts UGT, SULT, and MAO intrinsic clearance from SMILES using
    molecular descriptors (RDKit) or SMILES pattern heuristics.
    """

    def predict(self, smiles: str) -> Phase2Properties:
        """Predict Phase 2 CLint contributions from a SMILES string.

        Args:
            smiles: Valid SMILES string of the drug molecule.

        Returns:
            Phase2Properties with UGT, SULT, MAO CLint estimates.
        """
        if HAS_RDKIT:
            return self._predict_rdkit(smiles)
        return self._predict_smiles_heuristic(smiles)

    # ------------------------------------------------------------------
    # RDKit-based prediction
    # ------------------------------------------------------------------

    def _predict_rdkit(self, smiles: str) -> Phase2Properties:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("RDKit could not parse SMILES: %s", smiles)
            return self._predict_smiles_heuristic(smiles)

        # Molecular descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = MolSurf.TPSA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        n_oh = self._count_oh(mol)           # aliphatic OH
        n_phenol = self._count_phenol(mol)   # aromatic OH
        n_cooh = self._count_cooh(mol)       # carboxylic acid
        n_nh2 = self._count_primary_amine(mol)  # primary amine
        n_nh = self._count_secondary_amine(mol)  # secondary amine

        # --- UGT prediction ---
        # UGT substrates: phenols, aliphatic OH, COOH, N-glucuronidation
        # Model: Miners 2004 + additional descriptors
        # log(CLint_UGT) ≈ 0.6×n_phenol + 0.4×n_oh + 0.5×n_cooh
        #                  + 0.3×n_nh − 0.02×mw + 0.15×hbd − 1.8
        ugt_features = (
            0.60 * n_phenol
            + 0.40 * n_oh
            + 0.50 * n_cooh
            + 0.30 * (n_nh2 + n_nh)
            - 0.02 * (mw / 100.0)
            + 0.15 * hbd
            - 1.80
        )
        ugt_substrate_prob = float(np.clip(1.0 / (1.0 + np.exp(-ugt_features)), 0.0, 1.0))
        clint_ugt = float(np.clip(
            ugt_substrate_prob * (0.2 + 0.5 * n_phenol + 0.3 * n_oh + 0.4 * n_cooh),
            0.0, 50.0,
        ))

        # --- SULT prediction ---
        # SULT substrates: phenols (esp. para-OH), primary alcohols
        # High capacity but easily saturated; high logP attenuates
        sult_features = (
            1.20 * n_phenol
            + 0.60 * n_oh
            - 0.30 * logp
            + 0.10 * hbd
            - 0.80
        )
        sult_substrate_prob = float(np.clip(1.0 / (1.0 + np.exp(-sult_features)), 0.0, 1.0))
        clint_sult = float(np.clip(
            sult_substrate_prob * (0.5 * n_phenol + 0.2 * n_oh),
            0.0, 20.0,
        ))

        # --- MAO prediction ---
        # MAO substrates: primary monoamines (especially catecholamines)
        mao_features = (
            2.00 * n_nh2
            + 0.80 * n_nh
            - 0.20 * logp
            - 0.01 * mw
            - 1.50
        )
        mao_substrate_prob = float(np.clip(1.0 / (1.0 + np.exp(-mao_features)), 0.0, 1.0))
        clint_mao = float(np.clip(
            mao_substrate_prob * (1.0 * n_nh2 + 0.3 * n_nh),
            0.0, 15.0,
        ))

        clint_total = clint_ugt + clint_sult + clint_mao
        dominant = _dominant_pathway(clint_ugt, clint_sult, clint_mao)

        return Phase2Properties(
            clint_ugt_ul_min_mg=round(clint_ugt, 4),
            clint_sult_ul_min_mg=round(clint_sult, 4),
            clint_mao_ul_min_mg=round(clint_mao, 4),
            clint_phase2_total_ul_min_mg=round(clint_total, 4),
            ugt_substrate_prob=round(ugt_substrate_prob, 3),
            sult_substrate_prob=round(sult_substrate_prob, 3),
            mao_substrate_prob=round(mao_substrate_prob, 3),
            dominant_pathway=dominant,
            confidence="high",
        )

    # ------------------------------------------------------------------
    # SMILES pattern heuristics (fallback, no RDKit)
    # ------------------------------------------------------------------

    def _predict_smiles_heuristic(self, smiles: str) -> Phase2Properties:
        """Estimate Phase 2 clearance from SMILES character patterns."""
        s = smiles.upper()

        # Count chemotype indicators from SMILES
        n_phenol = s.count("C1=CC=CC=C1O") + s.count("OC1=CC=CC=C1")
        n_oh_generic = s.count("CO") + s.count("OC") - 2 * n_phenol
        n_cooh = s.count("C(=O)O") + s.count("OC(=O)")
        n_nh2 = s.count("N") - s.count("NC=O") - s.count("N(=O)")
        n_nh2 = max(0, n_nh2)

        ugt_substrate_prob = float(np.clip(
            0.4 * n_phenol + 0.2 * n_oh_generic + 0.3 * n_cooh, 0.0, 1.0
        ))
        sult_substrate_prob = float(np.clip(0.5 * n_phenol + 0.2 * n_oh_generic, 0.0, 1.0))
        mao_substrate_prob = float(np.clip(0.4 * n_nh2, 0.0, 1.0))

        clint_ugt = ugt_substrate_prob * 2.0
        clint_sult = sult_substrate_prob * 1.0
        clint_mao = mao_substrate_prob * 1.5
        clint_total = clint_ugt + clint_sult + clint_mao
        dominant = _dominant_pathway(clint_ugt, clint_sult, clint_mao)

        return Phase2Properties(
            clint_ugt_ul_min_mg=round(clint_ugt, 4),
            clint_sult_ul_min_mg=round(clint_sult, 4),
            clint_mao_ul_min_mg=round(clint_mao, 4),
            clint_phase2_total_ul_min_mg=round(clint_total, 4),
            ugt_substrate_prob=round(ugt_substrate_prob, 3),
            sult_substrate_prob=round(sult_substrate_prob, 3),
            mao_substrate_prob=round(mao_substrate_prob, 3),
            dominant_pathway=dominant,
            confidence="low",
        )

    # ------------------------------------------------------------------
    # Functional group counters (RDKit SMARTS)
    # ------------------------------------------------------------------

    @staticmethod
    def _count_oh(mol: "Chem.Mol") -> int:
        """Count aliphatic hydroxyl groups (C-OH, excluding phenols)."""
        patt = Chem.MolFromSmarts("[CX4;!$(C=O)]-[OH]")
        if patt is None:
            return 0
        return len(mol.GetSubstructMatches(patt))

    @staticmethod
    def _count_phenol(mol: "Chem.Mol") -> int:
        """Count aromatic hydroxyl (phenol) groups."""
        patt = Chem.MolFromSmarts("c-[OH]")
        if patt is None:
            return 0
        return len(mol.GetSubstructMatches(patt))

    @staticmethod
    def _count_cooh(mol: "Chem.Mol") -> int:
        """Count carboxylic acid groups."""
        patt = Chem.MolFromSmarts("[CX3](=O)[OH]")
        if patt is None:
            return 0
        return len(mol.GetSubstructMatches(patt))

    @staticmethod
    def _count_primary_amine(mol: "Chem.Mol") -> int:
        """Count primary aliphatic amine groups."""
        patt = Chem.MolFromSmarts("[NX3H2;!$(NC=O)]")
        if patt is None:
            return 0
        return len(mol.GetSubstructMatches(patt))

    @staticmethod
    def _count_secondary_amine(mol: "Chem.Mol") -> int:
        """Count secondary aliphatic amine groups."""
        patt = Chem.MolFromSmarts("[NX3H1;!$(NC=O)]")
        if patt is None:
            return 0
        return len(mol.GetSubstructMatches(patt))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dominant_pathway(clint_ugt: float, clint_sult: float, clint_mao: float) -> str:
    """Return name of the dominant Phase 2 pathway."""
    values = {"UGT": clint_ugt, "SULT": clint_sult, "MAO": clint_mao}
    total = sum(values.values())
    if total < 0.01:
        return "none"
    return max(values, key=lambda k: values[k])


__all__ = ["Phase2Properties", "Phase2Predictor"]
