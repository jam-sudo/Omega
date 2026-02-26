"""SMILES → ADME property prediction using QSPR models.

Uses RDKit molecular descriptors when available; falls back to
MW/logP-based estimation otherwise.

Predicts 9 ADME properties:
  1. logP (octanol-water partition coefficient)
  2. logS (aqueous solubility)
  3. Peff (intestinal permeability, ×10⁻⁴ cm/s)
  4. fup (fraction unbound in plasma)
  5. Rbp (blood-to-plasma ratio)
  6. CLint_3A4 (CYP3A4 intrinsic clearance, µL/min/pmol)
  7. CLint_2D6 (CYP2D6 intrinsic clearance, µL/min/pmol)
  8. hERG IC50 (µM) — cardiac safety
  9. MW (molecular weight, g/mol)
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, MolSurf

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.info("RDKit not available; using simplified QSPR models.")


@dataclass(frozen=True)
class ADMEProperties:
    """Predicted ADME properties from QSPR models."""

    mw: float
    logP: float
    logS: float
    peff: float  # ×10⁻⁴ cm/s
    fup: float
    rbp: float
    clint_3a4: float  # µL/min/pmol
    clint_2d6: float  # µL/min/pmol
    herg_ic50_uM: float
    confidence: str = "low"  # low, medium, high


class ADMEPredictor:
    """QSPR-based ADME property predictor.

    Uses multiple linear regression models trained on internal datasets.
    When RDKit is available, uses 2D molecular descriptors.
    Otherwise, uses simplified correlations.

    Nearest-neighbor confidence scoring is applied using reference data
    from data/adme_reference.csv when available.
    """

    _ref_data: list[dict] | None = None
    _featurizer: Any | None = None

    def _load_reference_data(self) -> None:
        """Load ADME reference data from CSV into self._ref_data.

        Searches for data/adme_reference.csv relative to this file:
        prediction/ -> omega_pbpk/ -> src/ -> Omega/ -> data/
        """
        ref_path = (
            Path(__file__).parent  # prediction/
            .parent                 # omega_pbpk/
            .parent                 # src/
            .parent                 # Omega/
            / "data"
            / "adme_reference.csv"
        )
        if not ref_path.exists():
            logger.debug("adme_reference.csv not found at %s; NN correction disabled.", ref_path)
            self._ref_data = []
            return

        rows: list[dict] = []
        with open(ref_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        self._ref_data = rows
        logger.debug("Loaded %d reference compounds from %s.", len(rows), ref_path)

    def _nearest_neighbor_distance(self, smiles: str) -> float:
        """Compute the minimum Euclidean distance from query to reference data.

        Uses the RDKitFeaturizer (MW, LogP, TPSA, ...) when available.
        Falls back to a 2-feature proxy (logP, MW) derived from the CSV rows
        when the featurizer produces zeros (no RDKit) or ref_data is absent.

        Args:
            smiles: Query SMILES string.

        Returns:
            Minimum Euclidean distance to nearest reference compound,
            or 0.0 if no reference data are available.
        """
        if not self._ref_data:
            return 0.0

        # Lazy-init featurizer
        if self._featurizer is None:
            try:
                from omega_pbpk.features.rdkit_featurizer import RDKitFeaturizer
                self._featurizer = RDKitFeaturizer()
            except Exception:
                self._featurizer = False  # mark as unavailable

        # Featurize query
        query_vec: np.ndarray | None = None
        if self._featurizer and self._featurizer is not False:
            fv = self._featurizer.featurize(smiles)
            if np.any(fv.descriptors != 0.0):
                query_vec = fv.descriptors  # shape (15,)

        if query_vec is not None:
            # Build reference matrix using same 2-feature proxy subset
            # (logP=index 1, MW=index 0) for fair comparison
            ref_vecs = []
            for row in self._ref_data:
                try:
                    ref_vecs.append([float(row["mw"]), float(row["logP"])])
                except (KeyError, ValueError):
                    continue
            if not ref_vecs:
                return 0.0
            # Use MW and logP columns from query vector
            q2 = np.array([query_vec[0], query_vec[1]])
            ref_arr = np.array(ref_vecs, dtype=np.float64)
            diffs = ref_arr - q2
            dists = np.linalg.norm(diffs, axis=1)
            return float(dists.min())

        # Fallback: use heuristic MW/logP estimates
        est_mw = self._estimate_mw(smiles)
        est_logp = self._estimate_logp(smiles)
        q2 = np.array([est_mw, est_logp])
        ref_vecs = []
        for row in self._ref_data:
            try:
                ref_vecs.append([float(row["mw"]), float(row["logP"])])
            except (KeyError, ValueError):
                continue
        if not ref_vecs:
            return 0.0
        ref_arr = np.array(ref_vecs, dtype=np.float64)
        diffs = ref_arr - q2
        dists = np.linalg.norm(diffs, axis=1)
        return float(dists.min())

    def predict(self, smiles: str) -> ADMEProperties:
        """Predict ADME properties from SMILES string.

        Loads reference data on first call to apply nearest-neighbor
        confidence scoring. Confidence is set to:
          "high"   if NN distance < 1.5
          "medium" if NN distance < 3.5
          "low"    otherwise

        Args:
            smiles: A valid SMILES string.

        Returns:
            ADMEProperties dataclass with updated confidence.
        """
        if self._ref_data is None:
            self._load_reference_data()

        if HAS_RDKIT:
            base_result = self._predict_rdkit(smiles)
        else:
            base_result = self._predict_simplified(smiles)

        distance = self._nearest_neighbor_distance(smiles)

        # Reference data absent or empty -> keep model-default confidence
        if self._ref_data:
            if distance < 1.5:
                confidence = "high"
            elif distance < 3.5:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = base_result.confidence

        # ADMEProperties is frozen; rebuild with updated confidence
        return ADMEProperties(
            mw=base_result.mw,
            logP=base_result.logP,
            logS=base_result.logS,
            peff=base_result.peff,
            fup=base_result.fup,
            rbp=base_result.rbp,
            clint_3a4=base_result.clint_3a4,
            clint_2d6=base_result.clint_2d6,
            herg_ic50_uM=base_result.herg_ic50_uM,
            confidence=confidence,
        )

    def _predict_rdkit(self, smiles: str) -> ADMEProperties:
        """Full RDKit-based QSPR prediction."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = MolSurf.TPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        Lipinski.NumRotatableBonds(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)

        # logS = -0.01 × MW - 0.92 × logP + 0.025 × TPSA - 0.19 × aromatic_rings + 1.58
        log_s = -0.01 * mw - 0.92 * logp + 0.025 * tpsa - 0.19 * aromatic_rings + 1.58

        # Peff (×10⁻⁴ cm/s) from logP and TPSA
        # Higher logP → higher Peff; higher TPSA → lower Peff
        peff = 10 ** (0.4 * logp - 0.01 * tpsa + 0.3)
        peff = np.clip(peff, 0.01, 100.0)

        # fup from logP and pKa-related descriptors
        # Higher logP → lower fup (more protein binding)
        fup = 1.0 / (1.0 + 10 ** (0.58 * logp - 1.2))
        fup = np.clip(fup, 0.001, 1.0)

        # Rbp from logP
        rbp = 0.55 + 0.35 * (1.0 / (1.0 + 10 ** (1.0 - 0.3 * logp)))
        rbp = np.clip(rbp, 0.5, 2.0)

        # CLint_3A4 from MW, logP, and flexibility
        clint_3a4 = 10 ** (0.2 * logp + 0.003 * mw - 0.05 * hbd - 1.5)
        clint_3a4 = np.clip(clint_3a4, 0.01, 100.0)

        # CLint_2D6 — lower baseline, depends on basicity
        clint_2d6 = 10 ** (0.15 * logp + 0.002 * mw + 0.1 * hba - 2.0)
        clint_2d6 = np.clip(clint_2d6, 0.01, 50.0)

        # hERG IC50 from lipophilicity and aromaticity
        herg = 10 ** (-0.4 * logp + 0.01 * tpsa - 0.15 * aromatic_rings + 2.5)
        herg = np.clip(herg, 0.01, 1000.0)

        return ADMEProperties(
            mw=round(mw, 2),
            logP=round(float(logp), 2),
            logS=round(float(log_s), 2),
            peff=round(float(peff), 3),
            fup=round(float(fup), 4),
            rbp=round(float(rbp), 3),
            clint_3a4=round(float(clint_3a4), 3),
            clint_2d6=round(float(clint_2d6), 3),
            herg_ic50_uM=round(float(herg), 2),
            confidence="medium",
        )

    def _predict_simplified(self, smiles: str) -> ADMEProperties:
        """Simplified prediction without RDKit (using SMILES heuristics)."""
        # Estimate MW from SMILES length (very rough)
        mw = self._estimate_mw(smiles)
        logp = self._estimate_logp(smiles)

        log_s = -0.01 * mw - 0.8 * logp + 1.0
        peff = 10 ** (0.3 * logp + 0.2)
        peff = np.clip(peff, 0.01, 100.0)
        fup = 1.0 / (1.0 + 10 ** (0.5 * logp - 1.0))
        fup = np.clip(fup, 0.001, 1.0)
        rbp = 0.6 + 0.2 * (logp > 2)
        clint_3a4 = 10 ** (0.15 * logp - 1.2)
        clint_2d6 = 10 ** (0.1 * logp - 1.5)
        herg = 10 ** (-0.3 * logp + 2.0)

        return ADMEProperties(
            mw=round(mw, 2),
            logP=round(float(logp), 2),
            logS=round(float(log_s), 2),
            peff=round(float(peff), 3),
            fup=round(float(fup), 4),
            rbp=round(float(rbp), 3),
            clint_3a4=round(float(clint_3a4), 3),
            clint_2d6=round(float(clint_2d6), 3),
            herg_ic50_uM=round(float(herg), 2),
            confidence="low",
        )

    @staticmethod
    def _estimate_mw(smiles: str) -> float:
        """Rough MW estimate from SMILES character composition."""
        atom_weights_upper = {
            "C": 12,
            "N": 14,
            "O": 16,
            "S": 32,
            "F": 19,
            "Cl": 35.5,
            "Br": 80,
            "I": 127,
            "P": 31,
        }
        # Aromatic atoms in SMILES: c, n, o, s (lowercase)
        atom_weights_lower = {"c": 12, "n": 14, "o": 16, "s": 32}
        mw = 0.0
        i = 0
        while i < len(smiles):
            if i + 1 < len(smiles) and smiles[i : i + 2] in ("Cl", "Br"):
                mw += atom_weights_upper[smiles[i : i + 2]]
                i += 2
            elif smiles[i] in atom_weights_upper:
                mw += atom_weights_upper[smiles[i]]
                i += 1
            elif smiles[i] in atom_weights_lower:
                mw += atom_weights_lower[smiles[i]]
                i += 1
            else:
                i += 1
        # Add hydrogens estimate (~10% of heavy atom mass)
        return max(mw * 1.1, 1.0)

    @staticmethod
    def _estimate_logp(smiles: str) -> float:
        """Rough logP estimate from SMILES."""
        n_aromatic = smiles.count("c") + smiles.count("C1") * 0.5
        n_polar = smiles.count("O") + smiles.count("N") + smiles.count("S")
        n_halogen = smiles.count("F") + smiles.count("Cl") + smiles.count("Br")
        heavy = sum(1 for c in smiles if c.isalpha() and c not in "()[]")
        return 0.1 * heavy + 0.3 * n_halogen - 0.4 * n_polar + 0.05 * n_aromatic

    def predict_from_dict(self, props: dict[str, float]) -> ADMEProperties | None:
        """Create ADMEProperties from a dictionary (e.g., from YAML)."""
        return ADMEProperties(
            mw=props.get("mw", 300.0),
            logP=props.get("logP", 2.0),
            logS=props.get("logS", -3.0),
            peff=props.get("peff", 1.0),
            fup=props.get("fup", 0.5),
            rbp=props.get("rbp", 1.0),
            clint_3a4=props.get("clint_3a4", 0.1),
            clint_2d6=props.get("clint_2d6", 0.01),
            herg_ic50_uM=props.get("herg_ic50_uM", 100.0),
            confidence="high",
        )
