"""Predict drug concentration in target organs from plasma PK.

Uses instantaneous equilibrium (Kp-based) to compute tissue concentrations
and AUC/Cmax metrics for a panel of organs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz

# Recognised organ names
_VALID_ORGANS = frozenset(
    {"liver", "kidney", "brain", "lung", "heart", "muscle", "fat"}
)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrganConcentrationResult:
    """Result of organ concentration prediction.

    Uses a regular @dataclass (not frozen) because list fields are present.
    """

    organ_name: str
    kp_organ: float
    fu_plasma: float
    fu_tissue: float
    times_h: list[float] = field(default_factory=list)
    c_tissue_mg_L: list[float] = field(default_factory=list)
    auc_tissue: float = 0.0
    cmax_tissue: float = 0.0
    tissue_plasma_auc_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Single-organ prediction
# ---------------------------------------------------------------------------

def predict_organ_concentration(
    plasma_concs: list[float],
    times_h: list[float],
    organ_name: str,
    kp_organ: float,
    fu_plasma: float,
    fu_tissue: float,
) -> OrganConcentrationResult:
    """Predict drug concentration in a target organ from plasma PK.

    Uses the instantaneous Kp-based equilibrium assumption:
        C_tissue(t) = Kp_organ * C_plasma(t)

    Unbound concentrations:
        Cu_plasma(t) = fu_plasma * C_plasma(t)
        Cu_tissue(t) = fu_tissue * C_tissue(t)

    At true steady-state: Cu_plasma = Cu_tissue (validates Kp consistency).

    Parameters
    ----------
    plasma_concs : list[float]
        Plasma concentration time course (mg/L). Must be non-empty.
    times_h : list[float]
        Corresponding time points (h). Same length as plasma_concs.
    organ_name : str
        Target organ name. One of: liver, kidney, brain, lung, heart, muscle, fat.
    kp_organ : float
        Organ-to-plasma partition coefficient (total drug). Must be > 0.
    fu_plasma : float
        Fraction unbound in plasma (0, 1]. Must be in (0, 1].
    fu_tissue : float
        Fraction unbound in tissue (0, 1]. Must be in (0, 1].

    Returns
    -------
    OrganConcentrationResult
    """
    # --- Input validation ---
    if not plasma_concs:
        raise ValueError("plasma_concs must be non-empty")
    if not times_h:
        raise ValueError("times_h must be non-empty")
    if len(plasma_concs) != len(times_h):
        raise ValueError("plasma_concs and times_h must have the same length")
    if organ_name not in _VALID_ORGANS:
        raise ValueError(
            f"organ_name must be one of {sorted(_VALID_ORGANS)}, got '{organ_name}'"
        )
    if kp_organ <= 0:
        raise ValueError("kp_organ must be > 0")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if not (0 < fu_tissue <= 1.0):
        raise ValueError("fu_tissue must be in (0, 1]")

    cp = np.maximum(np.asarray(plasma_concs, dtype=np.float64), 0.0)
    t = np.asarray(times_h, dtype=np.float64)

    # --- Tissue concentration (instantaneous equilibrium) ---
    c_tissue = kp_organ * cp  # total tissue
    # Unbound concentrations (for informational note)
    cu_plasma_ss = fu_plasma * float(cp[-1]) if len(cp) > 0 else 0.0
    cu_tissue_ss = fu_tissue * float(c_tissue[-1]) if len(c_tissue) > 0 else 0.0

    # --- AUC via trapezoidal rule ---
    auc_plasma = float(np_trapz(cp, t))
    auc_tissue = float(np_trapz(c_tissue, t))

    cmax_tissue = float(np.max(c_tissue))
    tissue_plasma_ratio = auc_tissue / auc_plasma if auc_plasma > 0 else 0.0

    # --- Notes ---
    notes: list[str] = []
    # Check Kp consistency: at steady state Cu_plasma ≈ Cu_tissue
    # implies Kp_organ = fu_plasma / fu_tissue
    kp_expected_ss = fu_plasma / fu_tissue
    kp_ratio = kp_organ / kp_expected_ss if kp_expected_ss > 0 else float("inf")
    if abs(kp_ratio - 1.0) < 0.2:
        notes.append(
            "Kp consistent with unbound steady-state equilibrium "
            f"(expected Kp_ss={kp_expected_ss:.2f})"
        )
    else:
        notes.append(
            f"Kp_organ ({kp_organ:.2f}) deviates from unbound SS prediction "
            f"({kp_expected_ss:.2f}): active transport or non-specific binding likely"
        )
    if organ_name == "brain" and kp_organ < 0.1:
        notes.append("Very low brain penetration (Kp_brain < 0.1)")
    if cu_plasma_ss > 0 and abs(cu_plasma_ss - cu_tissue_ss) / cu_plasma_ss > 0.5:
        notes.append(
            f"Unbound concentrations differ at t_end: "
            f"Cu_plasma={cu_plasma_ss:.3f}, Cu_tissue={cu_tissue_ss:.3f} mg/L"
        )

    return OrganConcentrationResult(
        organ_name=organ_name,
        kp_organ=kp_organ,
        fu_plasma=fu_plasma,
        fu_tissue=fu_tissue,
        times_h=t.tolist(),
        c_tissue_mg_L=c_tissue.tolist(),
        auc_tissue=auc_tissue,
        cmax_tissue=cmax_tissue,
        tissue_plasma_auc_ratio=tissue_plasma_ratio,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Multi-organ profile
# ---------------------------------------------------------------------------

def organ_exposure_profile(
    plasma_concs: list[float],
    times_h: list[float],
    organ_kp_dict: dict,
    fu_plasma: float,
    fu_tissue: float = 0.1,
) -> dict:
    """Compute organ exposure profile for multiple organs simultaneously.

    Parameters
    ----------
    plasma_concs : list[float]
        Plasma concentration time course (mg/L). Must be non-empty.
    times_h : list[float]
        Corresponding time points (h). Same length as plasma_concs.
    organ_kp_dict : dict
        Mapping of organ_name -> Kp_organ (e.g. {"liver": 3.5, "brain": 0.1}).
        All organ names must be valid; all Kp values must be > 0.
    fu_plasma : float
        Fraction unbound in plasma (0, 1].
    fu_tissue : float
        Fraction unbound in tissue (0, 1]. Applied uniformly to all organs.

    Returns
    -------
    dict
        Keys are organ names; each value is a dict with:
        - 'c_tissue_mg_L': list[float]
        - 'auc_tissue': float
        - 'cmax_tissue': float
        - 'tissue_plasma_auc_ratio': float
        - 'result': OrganConcentrationResult
    """
    if not plasma_concs:
        raise ValueError("plasma_concs must be non-empty")
    if not times_h:
        raise ValueError("times_h must be non-empty")
    if len(plasma_concs) != len(times_h):
        raise ValueError("plasma_concs and times_h must have the same length")
    if not organ_kp_dict:
        raise ValueError("organ_kp_dict must be non-empty")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if not (0 < fu_tissue <= 1.0):
        raise ValueError("fu_tissue must be in (0, 1]")

    profile: dict = {}
    for organ_name, kp in organ_kp_dict.items():
        result = predict_organ_concentration(
            plasma_concs=plasma_concs,
            times_h=times_h,
            organ_name=organ_name,
            kp_organ=kp,
            fu_plasma=fu_plasma,
            fu_tissue=fu_tissue,
        )
        profile[organ_name] = {
            "c_tissue_mg_L": result.c_tissue_mg_L,
            "auc_tissue": result.auc_tissue,
            "cmax_tissue": result.cmax_tissue,
            "tissue_plasma_auc_ratio": result.tissue_plasma_auc_ratio,
            "result": result,
        }

    return profile


__all__ = [
    "OrganConcentrationResult",
    "predict_organ_concentration",
    "organ_exposure_profile",
]
