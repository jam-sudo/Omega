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
    # Conformal prediction intervals (90% coverage)
    fup_lo: float = 0.0
    fup_hi: float = 1.0
    clint_3a4_lo: float = 0.0
    clint_3a4_hi: float = 0.0
    peff_lo: float = 0.0
    peff_hi: float = 0.0
    rbp_lo: float = 0.0
    rbp_hi: float = 0.0


class ADMEPredictor:
    """QSPR-based ADME property predictor.

    Uses polynomial ridge regression models trained on reference data
    (data/adme_reference.csv) when available, with split-conformal
    prediction intervals.  When RDKit is available, also uses 2D
    molecular descriptors for logS, hERG, and CLint_2D6.

    Nearest-neighbor confidence scoring uses the 6-feature polynomial
    space (normalised) for better discrimination.
    """

    _ref_data: list[dict] | None = None
    _featurizer: Any | None = None

    # Polynomial model coefficients (set by _fit_qspr_models)
    _coeff_fup: np.ndarray | None = None
    _coeff_clint: np.ndarray | None = None
    _coeff_peff: np.ndarray | None = None
    _coeff_rbp: np.ndarray | None = None
    # Conformal quantiles (90th percentile of relative residuals)
    _q90_fup: float = 0.5
    _q90_clint: float = 0.5
    _q90_peff: float = 0.5
    _q90_rbp: float = 0.5
    # Normalised reference feature matrix for NN distance
    _ref_poly_mat: np.ndarray | None = None
    _ref_feat_std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Reference data loading
    # ------------------------------------------------------------------

    def _load_reference_data(self) -> None:
        """Load ADME reference data from CSV into self._ref_data.

        Searches for data/adme_reference.csv relative to this file:
        prediction/ -> omega_pbpk/ -> src/ -> Omega/ -> data/
        """
        ref_path = (
            Path(
                __file__
            ).parent.parent.parent.parent  # prediction/  # omega_pbpk/  # src/  # Omega/
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

        self._fit_qspr_models()

    # ------------------------------------------------------------------
    # Polynomial feature construction
    # ------------------------------------------------------------------

    @staticmethod
    def _poly_features(mw: float, logp: float) -> np.ndarray:
        """Return degree-2 polynomial features [1, mw, logP, mw², logP², mw·logP]."""
        return np.array([1.0, mw, logp, mw * mw, logp * logp, mw * logp])

    # ------------------------------------------------------------------
    # Model fitting (called once after reference data is loaded)
    # ------------------------------------------------------------------

    def _fit_qspr_models(self) -> None:
        """Fit polynomial ridge regression on reference data with conformal calibration.

        Uses degree-2 features [1, mw, logP, mw², logP², mw·logP].
        80% of rows are used for training, the remaining 20% for
        split-conformal calibration (90th-percentile relative residual).
        """
        rows = self._ref_data
        if not rows or len(rows) < 5:
            return

        n = len(rows)
        try:
            mw_arr = np.array([float(r["mw"]) for r in rows])
            logp_arr = np.array([float(r["logP"]) for r in rows])
        except (KeyError, ValueError) as exc:
            logger.warning("QSPR fitting skipped: bad reference data (%s).", exc)
            return

        # Polynomial feature matrix [N x 6]
        A = np.column_stack(
            [
                np.ones(n),
                mw_arr,
                logp_arr,
                mw_arr**2,
                logp_arr**2,
                mw_arr * logp_arr,
            ]
        )

        # Targets (log-transformed where appropriate)
        try:
            fup_arr = np.clip(np.array([float(r["fup"]) for r in rows]), 1e-4, 0.9999)
            clint_arr = np.array([float(r["clint_3a4_uL_min_pmol"]) for r in rows])
            peff_arr = np.array([float(r["peff_cm_s"]) for r in rows])
            rbp_arr = np.array([float(r["rbp"]) for r in rows])
        except (KeyError, ValueError) as exc:
            logger.warning("QSPR fitting skipped: missing target column (%s).", exc)
            return

        y_fup = np.clip(np.log10(fup_arr / (1.0 - fup_arr)), -4.0, 4.0)
        y_clint = np.log10(np.clip(clint_arr, 1e-6, None) + 0.01)
        y_peff = np.log10(np.clip(peff_arr, 1e-10, None) * 1e4 + 0.001)
        y_rbp = rbp_arr

        # Train / calibration split
        n_train = max(5, int(0.8 * n))
        A_train, A_cal = A[:n_train], A[n_train:]

        def _fit_one(y_full: np.ndarray) -> tuple[np.ndarray, float]:
            y_tr, y_ca = y_full[:n_train], y_full[n_train:]
            # OLS via normal equations (well-conditioned with 6 features, 150+ rows)
            try:
                coeffs = np.linalg.lstsq(A_train, y_tr, rcond=None)[0]
            except np.linalg.LinAlgError:
                return np.zeros(6), 0.5
            if len(y_ca) >= 2:
                y_pred_ca = A_cal @ coeffs
                rel_res = np.abs(y_pred_ca - y_ca) / (np.abs(y_ca) + 1e-6)
                q90 = float(np.percentile(rel_res, 90))
            else:
                q90 = 0.5
            return coeffs, q90

        try:
            self._coeff_fup, self._q90_fup = _fit_one(y_fup)
            self._coeff_clint, self._q90_clint = _fit_one(y_clint)
            self._coeff_peff, self._q90_peff = _fit_one(y_peff)
            self._coeff_rbp, self._q90_rbp = _fit_one(y_rbp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("QSPR polynomial fitting failed: %s", exc)
            self._coeff_fup = None
            return

        # Build normalised reference matrix for NN scoring
        self._ref_poly_mat = A
        std = A.std(axis=0)
        std = np.where(std < 1e-10, 1.0, std)
        self._ref_feat_std = std

        logger.debug(
            "QSPR models fitted (n=%d train, %d cal). q90: fup=%.2f clint=%.2f peff=%.2f rbp=%.2f",
            n_train,
            n - n_train,
            self._q90_fup,
            self._q90_clint,
            self._q90_peff,
            self._q90_rbp,
        )

    # ------------------------------------------------------------------
    # Predict from fitted polynomial models
    # ------------------------------------------------------------------

    def _predict_poly(self, mw: float, logp: float) -> dict:
        """Return predictions from fitted polynomial models (dict of floats).

        Returns an empty dict if models are not fitted.
        """
        if self._coeff_fup is None:
            return {}

        poly = self._poly_features(mw, logp)

        # fup (logit space → back-transform)
        logit_fup = float(poly @ self._coeff_fup)
        fup = float(np.clip(1.0 / (1.0 + 10.0 ** (-logit_fup)), 0.001, 1.0))
        fup_lo = max(0.001, fup / (1.0 + self._q90_fup))
        fup_hi = min(1.0, fup * (1.0 + self._q90_fup))

        # clint_3a4 (log space → back-transform)
        log_clint = float(poly @ self._coeff_clint)
        clint_3a4 = float(np.clip(10.0**log_clint - 0.01, 0.01, 100.0))
        clint_lo = max(0.001, clint_3a4 / (1.0 + self._q90_clint))
        clint_hi = clint_3a4 * (1.0 + self._q90_clint)

        # peff (log scale of ×10⁻⁴ cm/s → back-transform; stored as ×10⁻⁴ cm/s)
        log_peff = float(poly @ self._coeff_peff)
        peff = float(np.clip(10.0**log_peff - 0.001, 0.01, 100.0))
        peff_lo = max(0.001, peff / (1.0 + self._q90_peff))
        peff_hi = peff * (1.0 + self._q90_peff)

        # rbp (direct)
        rbp = float(np.clip(poly @ self._coeff_rbp, 0.5, 3.0))
        rbp_lo = max(0.5, rbp / (1.0 + self._q90_rbp))
        rbp_hi = min(3.0, rbp * (1.0 + self._q90_rbp))

        return {
            "fup": fup,
            "fup_lo": fup_lo,
            "fup_hi": fup_hi,
            "clint_3a4": clint_3a4,
            "clint_3a4_lo": clint_lo,
            "clint_3a4_hi": clint_hi,
            "peff": peff,
            "peff_lo": peff_lo,
            "peff_hi": peff_hi,
            "rbp": rbp,
            "rbp_lo": rbp_lo,
            "rbp_hi": rbp_hi,
        }

    # ------------------------------------------------------------------
    # Nearest-neighbour confidence distance
    # ------------------------------------------------------------------

    def _nearest_neighbor_distance(self, smiles: str) -> float:
        """Compute the minimum Euclidean distance from query to reference data.

        Uses the normalised 6-feature polynomial space when models are
        fitted (better discrimination).  Falls back to a 2-feature
        proxy (logP, MW) otherwise.

        Args:
            smiles: Query SMILES string.

        Returns:
            Minimum Euclidean distance to nearest reference compound,
            or 0.0 if no reference data are available.
        """
        if not self._ref_data:
            return 0.0

        # Obtain MW and logP for the query molecule
        mw_q: float | None = None
        logp_q: float | None = None

        # Lazy-init featurizer
        if self._featurizer is None:
            try:
                from omega_pbpk.features.rdkit_featurizer import RDKitFeaturizer

                self._featurizer = RDKitFeaturizer()
            except Exception:
                self._featurizer = False  # mark as unavailable

        if self._featurizer and self._featurizer is not False:
            fv = self._featurizer.featurize(smiles)
            if np.any(fv.descriptors != 0.0):
                mw_q = float(fv.descriptors[0])
                logp_q = float(fv.descriptors[1])

        if mw_q is None:
            mw_q = self._estimate_mw(smiles)
            logp_q = self._estimate_logp(smiles)

        # Use normalised 6-feature poly space when available
        if self._ref_poly_mat is not None and self._ref_feat_std is not None:
            q_poly = self._poly_features(mw_q, logp_q)
            q_norm = q_poly / self._ref_feat_std
            ref_norm = self._ref_poly_mat / self._ref_feat_std
            dists = np.linalg.norm(ref_norm - q_norm, axis=1)
            return float(dists.min())

        # Fallback: 2-feature [mw, logP]
        q2 = np.array([mw_q, logp_q])
        ref_vecs = []
        for row in self._ref_data:
            try:
                ref_vecs.append([float(row["mw"]), float(row["logP"])])
            except (KeyError, ValueError):
                continue
        if not ref_vecs:
            return 0.0
        ref_arr = np.array(ref_vecs, dtype=np.float64)
        dists = np.linalg.norm(ref_arr - q2, axis=1)
        return float(dists.min())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            fup_lo=base_result.fup_lo,
            fup_hi=base_result.fup_hi,
            clint_3a4_lo=base_result.clint_3a4_lo,
            clint_3a4_hi=base_result.clint_3a4_hi,
            peff_lo=base_result.peff_lo,
            peff_hi=base_result.peff_hi,
            rbp_lo=base_result.rbp_lo,
            rbp_hi=base_result.rbp_hi,
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

        # Use polynomial models when fitted; else fall back to hardcoded equations
        poly = self._predict_poly(float(mw), float(logp))

        if poly:
            fup = poly["fup"]
            fup_lo = poly["fup_lo"]
            fup_hi = poly["fup_hi"]
            clint_3a4 = poly["clint_3a4"]
            clint_3a4_lo = poly["clint_3a4_lo"]
            clint_3a4_hi = poly["clint_3a4_hi"]
            peff = poly["peff"]
            peff_lo = poly["peff_lo"]
            peff_hi = poly["peff_hi"]
            rbp = poly["rbp"]
            rbp_lo = poly["rbp_lo"]
            rbp_hi = poly["rbp_hi"]
        else:
            # Hardcoded fallback equations
            peff = float(np.clip(10 ** (0.4 * logp - 0.01 * tpsa + 0.3), 0.01, 100.0))
            fup = float(np.clip(1.0 / (1.0 + 10 ** (0.58 * logp - 1.2)), 0.001, 1.0))
            rbp = float(np.clip(0.55 + 0.35 * (1.0 / (1.0 + 10 ** (1.0 - 0.3 * logp))), 0.5, 2.0))
            clint_3a4 = float(
                np.clip(10 ** (0.2 * logp + 0.003 * mw - 0.05 * hbd - 1.5), 0.01, 100.0)
            )
            fup_lo, fup_hi = 0.0, 1.0
            clint_3a4_lo, clint_3a4_hi = 0.0, 0.0
            peff_lo, peff_hi = 0.0, 0.0
            rbp_lo, rbp_hi = 0.0, 0.0

        # CLint_2D6 — lower baseline, depends on basicity
        clint_2d6 = float(np.clip(10 ** (0.15 * logp + 0.002 * mw + 0.1 * hba - 2.0), 0.01, 50.0))

        # hERG IC50 from lipophilicity and aromaticity
        herg = float(
            np.clip(10 ** (-0.4 * logp + 0.01 * tpsa - 0.15 * aromatic_rings + 2.5), 0.01, 1000.0)
        )

        return ADMEProperties(
            mw=round(mw, 2),
            logP=round(float(logp), 2),
            logS=round(float(log_s), 2),
            peff=round(peff, 3),
            fup=round(fup, 4),
            rbp=round(rbp, 3),
            clint_3a4=round(clint_3a4, 3),
            clint_2d6=round(clint_2d6, 3),
            herg_ic50_uM=round(herg, 2),
            confidence="medium",
            fup_lo=round(fup_lo, 4),
            fup_hi=round(fup_hi, 4),
            clint_3a4_lo=round(clint_3a4_lo, 3),
            clint_3a4_hi=round(clint_3a4_hi, 3),
            peff_lo=round(peff_lo, 4),
            peff_hi=round(peff_hi, 4),
            rbp_lo=round(rbp_lo, 4),
            rbp_hi=round(rbp_hi, 4),
        )

    def _predict_simplified(self, smiles: str) -> ADMEProperties:
        """Simplified prediction without RDKit (using SMILES heuristics)."""
        mw = self._estimate_mw(smiles)
        logp = self._estimate_logp(smiles)

        log_s = -0.01 * mw - 0.8 * logp + 1.0

        # Use polynomial models when fitted; else fall back to hardcoded equations
        poly = self._predict_poly(mw, logp)

        if poly:
            fup = poly["fup"]
            fup_lo = poly["fup_lo"]
            fup_hi = poly["fup_hi"]
            clint_3a4 = poly["clint_3a4"]
            clint_3a4_lo = poly["clint_3a4_lo"]
            clint_3a4_hi = poly["clint_3a4_hi"]
            peff = poly["peff"]
            peff_lo = poly["peff_lo"]
            peff_hi = poly["peff_hi"]
            rbp = poly["rbp"]
            rbp_lo = poly["rbp_lo"]
            rbp_hi = poly["rbp_hi"]
        else:
            peff = float(np.clip(10 ** (0.3 * logp + 0.2), 0.01, 100.0))
            fup = float(np.clip(1.0 / (1.0 + 10 ** (0.5 * logp - 1.0)), 0.001, 1.0))
            rbp = 0.6 + 0.2 * (logp > 2)
            clint_3a4 = float(np.clip(10 ** (0.15 * logp - 1.2), 0.01, 100.0))
            fup_lo, fup_hi = 0.0, 1.0
            clint_3a4_lo, clint_3a4_hi = 0.0, 0.0
            peff_lo, peff_hi = 0.0, 0.0
            rbp_lo, rbp_hi = 0.0, 0.0

        clint_2d6 = float(np.clip(10 ** (0.1 * logp - 1.5), 0.01, 50.0))
        herg = float(np.clip(10 ** (-0.3 * logp + 2.0), 0.01, 1000.0))

        return ADMEProperties(
            mw=round(mw, 2),
            logP=round(float(logp), 2),
            logS=round(float(log_s), 2),
            peff=round(peff, 3),
            fup=round(fup, 4),
            rbp=round(rbp, 3),
            clint_3a4=round(clint_3a4, 3),
            clint_2d6=round(clint_2d6, 3),
            herg_ic50_uM=round(herg, 2),
            confidence="low",
            fup_lo=round(fup_lo, 4),
            fup_hi=round(fup_hi, 4),
            clint_3a4_lo=round(clint_3a4_lo, 3),
            clint_3a4_hi=round(clint_3a4_hi, 3),
            peff_lo=round(peff_lo, 4),
            peff_hi=round(peff_hi, 4),
            rbp_lo=round(rbp_lo, 4),
            rbp_hi=round(rbp_hi, 4),
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
