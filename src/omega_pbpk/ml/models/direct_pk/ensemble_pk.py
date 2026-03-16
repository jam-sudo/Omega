"""PBPK + ML ensemble for Cmax prediction.

Blends PBPK and direct ML Cmax predictions using a confidence-weighted
geometric mean. Scales the PBPK C(t) curve to match the ensemble Cmax.

Weight schedule:
- High confidence: 60% PBPK, 40% ML (PBPK trusted)
- Medium confidence: 40% PBPK, 60% ML (ML slightly preferred)
- Low confidence: 20% PBPK, 80% ML (ML strongly preferred)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# PBPK weight by confidence level (rest goes to ML)
# Conservative: high-confidence PBPK predictions are trusted fully.
# ML only contributes when PBPK confidence is medium/low.
_PBPK_WEIGHTS = {
    "high": 1.0,  # Trust PBPK fully when confident
    "medium": 0.7,  # Light ML adjustment
    "low": 0.4,  # ML takes the lead
}


def ensemble_cmax(
    cmax_pbpk: float,
    cmax_ml: float,
    confidence: str = "medium",
) -> float:
    """Blend PBPK and ML Cmax predictions.

    Uses geometric mean with confidence-dependent weighting:
        Cmax_final = Cmax_pbpk^w * Cmax_ml^(1-w)

    Args:
        cmax_pbpk: PBPK pipeline Cmax prediction (mg/L)
        cmax_ml: Direct ML Cmax prediction (mg/L)
        confidence: PBPK confidence level ("high"/"medium"/"low")

    Returns:
        Ensemble Cmax (mg/L)
    """
    if cmax_pbpk <= 0 or cmax_ml <= 0:
        return max(cmax_pbpk, cmax_ml, 1e-12)

    w = _PBPK_WEIGHTS.get(confidence, 0.4)
    log_ensemble = w * np.log(cmax_pbpk) + (1 - w) * np.log(cmax_ml)
    return float(np.exp(log_ensemble))


def scale_ct_curve(
    time_h: np.ndarray,
    cp_mg_L: np.ndarray,
    target_cmax: float,
) -> np.ndarray:
    """Scale a C(t) curve so its Cmax matches the target.

    Preserves the curve shape (absorption/elimination profile)
    while adjusting the magnitude.
    """
    current_cmax = np.max(cp_mg_L)
    if current_cmax <= 0:
        return cp_mg_L

    scale_factor = target_cmax / current_cmax
    return cp_mg_L * scale_factor
