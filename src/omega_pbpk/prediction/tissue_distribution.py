"""
tissue_distribution.py — Tissue Distribution Predictor

Predict tissue-to-plasma concentration ratios (Kp) for 13 major organs using
logP-based QSPR equations (simplified Rodgers-Rowland-like empirical model).

Reference: Rodgers & Rowland (2006) J Pharm Sci; Poulin & Theil (2002).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "TissueKpResult",
    "predict_tissue_kp_qspr",
    "predict_vss_from_kp",
    "tissue_distribution_profile",
]

# ---------------------------------------------------------------------------
# Tissue lipid fractions (volume fraction of neutral lipids, dimensionless)
# ---------------------------------------------------------------------------

_TISSUE_LIPID: dict[str, float] = {
    "adipose": 0.79,
    "bone": 0.07,
    "brain": 0.04,
    "gut": 0.03,
    "heart": 0.01,
    "kidney": 0.01,
    "liver": 0.04,
    "lung": 0.02,
    "muscle": 0.01,
    "skin": 0.01,
    "spleen": 0.01,
    "thymus": 0.01,
    "gonads": 0.03,
}

# Physiological tissue volumes (L/kg, 70 kg human reference)
# From Rodgers & Rowland (2006) supplementary; ICRP 2002
_TISSUE_VOLUMES_L_PER_KG: dict[str, float] = {
    "adipose": 0.214,
    "bone": 0.086,
    "brain": 0.020,
    "gut": 0.017,
    "heart": 0.005,
    "kidney": 0.004,
    "liver": 0.026,
    "lung": 0.010,
    "muscle": 0.400,
    "skin": 0.037,
    "spleen": 0.003,
    "thymus": 0.003,
    "gonads": 0.002,
}

_VP_L_PER_KG = 0.04  # plasma volume


# ---------------------------------------------------------------------------
# Result dataclass (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TissueKpResult:
    """Tissue-to-plasma Kp predictions for a given drug physicochemistry."""

    logP: float
    fu_plasma: float
    bp_ratio: float
    kp_values: dict[str, float]  # tissue -> Kp (dimensionless)
    vss_L_per_kg: float
    highest_kp_tissue: str
    lowest_kp_tissue: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(fu_plasma: float, bp_ratio: float) -> None:
    """Raise ValueError for out-of-range physicochemical inputs."""
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}.")
    if bp_ratio <= 0.0:
        raise ValueError(f"bp_ratio must be > 0, got {bp_ratio}.")


def _compute_kp(logP: float, fu_plasma: float, lip: float) -> float:
    """
    Simplified empirical Kp equation.

    Kp = (0.5 + logP * lip) / (fu_plasma + 0.01)

    This captures:
    - Baseline partitioning offset (0.5)
    - logP-driven lipophilic sequestration (logP * lip)
    - fu_plasma dependency (higher unbound fraction -> lower tissue retention)
    - Small offset (0.01) prevents division by zero
    """
    numerator = 0.5 + logP * lip
    denominator = fu_plasma + 0.01
    kp = numerator / denominator
    # Kp must be positive; if logP is very negative, clamp to a minimum
    return max(kp, 0.001)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_tissue_kp_qspr(
    logP: float,
    pka_acid: float | None,
    pka_base: float | None,
    fu_plasma: float,
    bp_ratio: float = 1.0,
) -> TissueKpResult:
    """
    Predict tissue-to-plasma Kp for 13 major tissues using logP-based QSPR.

    Uses a simplified Rodgers-Rowland-like empirical model:
        Kp = (0.5 + logP * lip[tissue]) / (fu_plasma + 0.01)

    pka_acid and pka_base are accepted for future extensibility (ionisation
    corrections) but are not used in the current simplified model.

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient (any float).
    pka_acid:
        Acidic pKa (or None if not applicable).
    pka_base:
        Basic pKa (or None if not applicable).
    fu_plasma:
        Fraction unbound in plasma; must be in (0, 1].
    bp_ratio:
        Blood-to-plasma concentration ratio (>0). Used in Vss calculation.

    Returns
    -------
    TissueKpResult
    """
    _validate_inputs(fu_plasma, bp_ratio)

    kp_values: dict[str, float] = {}
    for tissue, lip in _TISSUE_LIPID.items():
        kp_values[tissue] = round(_compute_kp(logP, fu_plasma, lip), 4)

    vss = predict_vss_from_kp(kp_values, fu_plasma, bp_ratio)

    highest_kp_tissue = max(kp_values, key=lambda t: kp_values[t])
    lowest_kp_tissue = min(kp_values, key=lambda t: kp_values[t])

    pka_note = ""
    if pka_acid is not None:
        pka_note += f" pKa_acid={pka_acid}."
    if pka_base is not None:
        pka_note += f" pKa_base={pka_base}."

    notes = (
        f"QSPR Kp for logP={logP:.2f}, fu_plasma={fu_plasma:.3f}, "
        f"bp_ratio={bp_ratio:.2f}.{pka_note} "
        f"Highest Kp: {highest_kp_tissue} ({kp_values[highest_kp_tissue]:.2f}). "
        f"Vss = {vss:.3f} L/kg."
    )

    return TissueKpResult(
        logP=logP,
        fu_plasma=fu_plasma,
        bp_ratio=bp_ratio,
        kp_values=kp_values,
        vss_L_per_kg=round(vss, 4),
        highest_kp_tissue=highest_kp_tissue,
        lowest_kp_tissue=lowest_kp_tissue,
        notes=notes,
    )


def predict_vss_from_kp(
    kp_values: dict[str, float],
    fu_plasma: float,
    bp_ratio: float = 1.0,
) -> float:
    """
    Estimate volume of distribution at steady state (Vss, L/kg).

    Vss = sum_t(Vt * Kp_t) + Vp

    where Vt are physiological tissue volumes (L/kg) and Vp = 0.04 L/kg.

    bp_ratio is stored for completeness; the Vss formula here follows the
    Rodgers-Rowland tissue-summation approach which already embeds the
    blood partitioning via fu_plasma in the Kp values.

    Parameters
    ----------
    kp_values:
        Dict mapping tissue name to Kp (from predict_tissue_kp_qspr).
    fu_plasma:
        Fraction unbound in plasma (>0).
    bp_ratio:
        Blood-to-plasma ratio (>0, stored but not applied here as Kp
        already incorporates plasma binding).

    Returns
    -------
    float: Vss in L/kg
    """
    if fu_plasma <= 0:
        raise ValueError(f"fu_plasma must be > 0, got {fu_plasma}.")
    if bp_ratio <= 0:
        raise ValueError(f"bp_ratio must be > 0, got {bp_ratio}.")

    vss = _VP_L_PER_KG
    for tissue, vol in _TISSUE_VOLUMES_L_PER_KG.items():
        kp = kp_values.get(tissue, 1.0)
        vss += vol * kp

    return vss


def tissue_distribution_profile(
    logP_range: list[float],
    fu_plasma: float,
) -> list[TissueKpResult]:
    """
    Compute tissue Kp predictions across a range of logP values.

    Useful for understanding how lipophilicity drives tissue distribution.

    Parameters
    ----------
    logP_range:
        List of logP values to evaluate.
    fu_plasma:
        Fraction unbound in plasma (held constant across the range).

    Returns
    -------
    list[TissueKpResult], one entry per logP value.
    """
    if not logP_range:
        raise ValueError("logP_range must contain at least one value.")
    _validate_inputs(fu_plasma, 1.0)  # bp_ratio=1.0 is always valid

    return [
        predict_tissue_kp_qspr(
            logP=lp,
            pka_acid=None,
            pka_base=None,
            fu_plasma=fu_plasma,
        )
        for lp in logP_range
    ]
