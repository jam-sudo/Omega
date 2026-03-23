"""Adaptive Conformal UQ — molecular similarity-based variable-width prediction intervals.

Drugs similar to the calibration set get tighter intervals; novel drugs get wider
intervals.  This replaces fixed-width conformal prediction with a *local* conformal
approach using k-nearest-neighbor nonconformity scores in Morgan fingerprint
Tanimoto space.

Algorithm
---------
1. **Calibration (fit)**: For each calibration drug compute
       nonconformity_score = |log10(Cmax_obs / Cmax_pred)|
   and store the Morgan fingerprint + score.

2. **Prediction (predict_interval)**: For a new molecule
   a. Compute Morgan FP and find k-nearest calibration molecules (Tanimoto).
   b. Collect the nonconformity scores of those k neighbors.
   c. Compute the (1-alpha)-quantile *q* of those scores, with the standard
      finite-sample correction: quantile rank = ceil((k+1)*(1-alpha)) / k.
   d. Return the interval [Cmax_pred * 10^(-q), Cmax_pred * 10^(+q)].

When no calibration data is available (model files missing), predict_interval()
returns None so the pipeline can gracefully fall back to fixed-width UQ.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_MODEL_DIR = _REPO_ROOT / "models" / "corrections" / "conformal"

# Defaults
_DEFAULT_K = 30  # number of nearest neighbors (recalibrated 2026-03-22)
_DEFAULT_ALPHA = 0.10  # 90 % prediction interval
_FP_RADIUS = 2
_FP_NBITS = 2048

# Fallback quantile when fewer than k neighbors (wide interval)
_FALLBACK_Q = 1.0  # 10x fold range


class AdaptiveConformal:
    """Adaptive conformal prediction intervals using molecular similarity.

    Parameters
    ----------
    model_dir : Path | None
        Directory for saving / loading calibration data.  When None, uses
        ``models/corrections/conformal/`` relative to repo root.
    k : int
        Number of nearest neighbors for local conformal quantile.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        k: int = _DEFAULT_K,
    ) -> None:
        self._model_dir = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
        self._k = k

        # Calibration state -------------------------------------------------
        self._smiles: list[str] = []
        self._fps: list[Any] = []  # RDKit ExplicitBitVect objects
        self._scores: np.ndarray = np.array([], dtype=np.float64)
        self._fitted = False

        # Attempt to load from disk
        self._try_load()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether calibration data has been loaded or computed."""
        return self._fitted and len(self._fps) > 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        smiles_list: list[str],
        cmax_obs_list: list[float],
        cmax_pred_list: list[float],
    ) -> None:
        """Calibrate from observed / predicted Cmax pairs.

        Computes nonconformity scores and Morgan fingerprints, then saves to
        ``self._model_dir``.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES for each calibration molecule.
        cmax_obs_list, cmax_pred_list : list[float]
            Observed and predicted Cmax values (same units, mg/L).
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        assert len(smiles_list) == len(cmax_obs_list) == len(cmax_pred_list), (
            "smiles_list, cmax_obs_list, cmax_pred_list must have equal length"
        )

        fps: list[Any] = []
        scores: list[float] = []
        valid_smiles: list[str] = []

        for smi, obs, pred in zip(smiles_list, cmax_obs_list, cmax_pred_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning("AdaptiveConformal.fit: invalid SMILES skipped: %s", smi)
                continue
            if obs <= 0 or pred <= 0:
                logger.warning(
                    "AdaptiveConformal.fit: non-positive Cmax skipped (obs=%.4g, pred=%.4g) for %s",
                    obs,
                    pred,
                    smi,
                )
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, _FP_RADIUS, nBits=_FP_NBITS)
            score = abs(np.log10(obs / pred))

            fps.append(fp)
            scores.append(score)
            valid_smiles.append(smi)

        if len(fps) == 0:
            logger.error("AdaptiveConformal.fit: no valid calibration molecules!")
            return

        self._fps = fps
        self._scores = np.array(scores, dtype=np.float64)
        self._smiles = valid_smiles
        self._fitted = True

        # Save to disk
        self._save()

        logger.info(
            "AdaptiveConformal: calibrated on %d molecules "
            "(mean score %.3f, median %.3f, max %.3f)",
            len(self._fps),
            float(self._scores.mean()),
            float(np.median(self._scores)),
            float(self._scores.max()),
        )

    def predict_interval(
        self,
        smiles: str,
        cmax_pred: float,
        alpha: float = _DEFAULT_ALPHA,
    ) -> dict[str, float] | None:
        """Compute adaptive conformal prediction interval for a molecule.

        Parameters
        ----------
        smiles : str
            Query molecule SMILES.
        cmax_pred : float
            Point prediction of Cmax (mg/L).
        alpha : float
            Miscoverage rate.  Default 0.10 gives a 90 % prediction interval.

        Returns
        -------
        dict | None
            Keys: lower, upper, width, similarity, q.
            Returns None if the model is not fitted or SMILES is invalid.
        """
        if not self.is_fitted:
            return None

        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug("AdaptiveConformal: invalid SMILES '%s'", smiles)
            return None

        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, _FP_RADIUS, nBits=_FP_NBITS)

        # Compute Tanimoto similarities to all calibration molecules
        sims = np.array(
            [DataStructs.TanimotoSimilarity(query_fp, fp) for fp in self._fps],
            dtype=np.float64,
        )

        # Find k-nearest neighbors
        k = min(self._k, len(self._fps))
        top_k_idx = np.argsort(-sims)[:k]  # indices of highest similarity
        neighbor_scores = self._scores[top_k_idx]
        neighbor_sims = sims[top_k_idx]

        mean_similarity = float(neighbor_sims.mean())

        # Local conformal quantile with finite-sample correction
        # Per Vovk et al., quantile rank = ceil((k+1)*(1-alpha)) / k
        quantile_rank = min(np.ceil((k + 1) * (1.0 - alpha)) / k, 1.0)
        q = float(np.quantile(neighbor_scores, quantile_rank))

        # Safety: ensure q is at least a small value (no zero-width intervals)
        q = max(q, 0.01)

        lower = cmax_pred * 10 ** (-q)
        upper = cmax_pred * 10 ** (+q)
        width = upper / lower  # fold-range

        return {
            "lower": float(lower),
            "upper": float(upper),
            "width": float(width),
            "similarity": mean_similarity,
            "q": q,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Save calibration data to model_dir."""
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # Save scores
        np.save(self._model_dir / "nonconformity_scores.npy", self._scores)

        # Save fingerprints as bit strings (portable, no pickle)
        fp_bits = [fp.ToBitString() for fp in self._fps]

        meta = {
            "n_calibration": len(self._fps),
            "fp_radius": _FP_RADIUS,
            "fp_nbits": _FP_NBITS,
            "k": self._k,
            "smiles": self._smiles,
            "fp_bits": fp_bits,
            "score_mean": float(self._scores.mean()),
            "score_median": float(np.median(self._scores)),
            "score_max": float(self._scores.max()),
        }
        with open(self._model_dir / "adaptive_conformal_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("AdaptiveConformal: saved to %s", self._model_dir)

    def _try_load(self) -> None:
        """Attempt to load calibration data from model_dir."""
        meta_path = self._model_dir / "adaptive_conformal_meta.json"
        scores_path = self._model_dir / "nonconformity_scores.npy"

        if not meta_path.exists() or not scores_path.exists():
            logger.debug(
                "AdaptiveConformal: no calibration data at %s",
                self._model_dir,
            )
            return

        try:
            from rdkit.Chem import DataStructs

            with open(meta_path) as f:
                meta = json.load(f)

            self._scores = np.load(scores_path)
            self._smiles = meta["smiles"]
            self._k = meta.get("k", _DEFAULT_K)

            # Reconstruct fingerprints from bit strings
            fp_bits = meta["fp_bits"]
            nbits = meta.get("fp_nbits", _FP_NBITS)
            self._fps = []
            for bits_str in fp_bits:
                fp = DataStructs.ExplicitBitVect(nbits)
                for i, ch in enumerate(bits_str):
                    if ch == "1":
                        fp.SetBit(i)
                self._fps.append(fp)

            assert len(self._fps) == len(self._scores), (
                f"FP count ({len(self._fps)}) != score count ({len(self._scores)})"
            )

            self._fitted = True
            logger.info(
                "AdaptiveConformal: loaded %d calibration molecules from %s",
                len(self._fps),
                self._model_dir,
            )
        except Exception as exc:
            logger.warning("AdaptiveConformal: failed to load: %s", exc)
            self._fitted = False
