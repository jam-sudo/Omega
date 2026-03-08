"""Phase 474: Drug Stability in Biological Matrices.

Models first-order drug degradation in various biological matrices including
plasma, blood, liver microsomes, intestinal microsomes, brain homogenate, and urine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

VALID_MATRICES = {
    "plasma",
    "blood",
    "liver_microsomes",
    "intestinal_microsomes",
    "brain_homogenate",
    "urine",
}

# Baseline k_deg (1/h) for a typical drug (logP=2, no ester, no reactive groups)
_K_BASE: dict[str, float] = {
    "plasma": 0.02,
    "blood": 0.05,
    "liver_microsomes": 0.5,
    "intestinal_microsomes": 0.3,
    "brain_homogenate": 0.1,
    "urine": 0.01,
}

# Matrices subject to esterase lability
_ESTERASE_MATRICES = {"plasma", "blood"}

# Matrices subject to CYP-based logP and functional group adjustments
_CYP_MATRICES = {"liver_microsomes", "intestinal_microsomes"}

# Reactive functional groups that increase CYP reactivity
_REACTIVE_GROUPS = {"aldehyde", "nitro", "Michael_acceptor"}


@dataclass(frozen=True)
class BiomatrixStabilityResult:
    """Result of drug stability prediction in a biological matrix."""

    drug_name: str
    matrix: str
    logP: float
    ester_group: bool
    functional_groups: tuple
    k_deg_per_h: float
    t_half_h: float
    pct_remaining_30min: float
    pct_remaining_60min: float
    pct_remaining_2h: float
    stability_class: str
    notes: str


def _classify_stability(t_half_h: float) -> str:
    """Classify stability based on half-life."""
    if t_half_h < 0.5:
        return "labile"
    elif t_half_h < 2.0:
        return "unstable"
    elif t_half_h < 6.0:
        return "moderate"
    else:
        return "stable"


def _compute_k_deg(
    matrix: str,
    logP: float,
    ester_group: bool,
    functional_groups: tuple,
) -> float:
    """Compute the degradation rate constant for the given matrix."""
    k_deg = _K_BASE[matrix]

    # Esterase lability: multiply by 5 for plasma/blood
    if ester_group and matrix in _ESTERASE_MATRICES:
        k_deg *= 5.0

    # logP effect for CYP-rich matrices
    if matrix in _CYP_MATRICES:
        k_deg *= 1.0 + 0.1 * max(0.0, logP)

    # Reactive functional groups in CYP matrices
    if matrix in _CYP_MATRICES and functional_groups:
        for group in functional_groups:
            if group in _REACTIVE_GROUPS:
                k_deg *= 2.0

    return k_deg


def predict_biomatrix_stability(
    drug_name: str,
    matrix: str,
    logP: float = 2.0,
    ester_group: bool = False,
    functional_groups: tuple | None = None,
) -> BiomatrixStabilityResult:
    """Predict drug stability in a biological matrix.

    Uses first-order degradation: C(t) = C0 * exp(-k_deg * t).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    matrix:
        Biological matrix. One of: plasma, blood, liver_microsomes,
        intestinal_microsomes, brain_homogenate, urine.
    logP:
        Lipophilicity.
    ester_group:
        Whether the drug contains an ester group (subject to esterase hydrolysis).
    functional_groups:
        Tuple of reactive functional group names. Recognised: aldehyde, nitro,
        Michael_acceptor.

    Returns
    -------
    BiomatrixStabilityResult
    """
    if matrix not in VALID_MATRICES:
        raise ValueError(f"matrix must be one of {sorted(VALID_MATRICES)}, got '{matrix}'")

    if functional_groups is None:
        functional_groups = ()

    k_deg = _compute_k_deg(matrix, logP, ester_group, functional_groups)

    if k_deg <= 0:
        raise ValueError("k_deg must be > 0")

    t_half_h = math.log(2) / k_deg

    pct_remaining_30min = 100.0 * math.exp(-k_deg * 0.5)
    pct_remaining_60min = 100.0 * math.exp(-k_deg * 1.0)
    pct_remaining_2h = 100.0 * math.exp(-k_deg * 2.0)

    stability_class = _classify_stability(t_half_h)

    modifiers = []
    if ester_group and matrix in _ESTERASE_MATRICES:
        modifiers.append("esterase lability (5x)")
    if matrix in _CYP_MATRICES:
        modifiers.append(f"logP CYP effect (+{0.1 * max(0.0, logP):.2f}x)")
    reactive_found = [g for g in functional_groups if g in _REACTIVE_GROUPS]
    if reactive_found:
        modifiers.append(f"reactive groups {reactive_found} (2x each)")

    notes_str = f"k_deg={k_deg:.4f}/h; t1/2={t_half_h:.2f}h; class={stability_class}."
    if modifiers:
        notes_str += " Modifiers: " + ", ".join(modifiers) + "."

    return BiomatrixStabilityResult(
        drug_name=drug_name,
        matrix=matrix,
        logP=logP,
        ester_group=ester_group,
        functional_groups=functional_groups,
        k_deg_per_h=k_deg,
        t_half_h=t_half_h,
        pct_remaining_30min=pct_remaining_30min,
        pct_remaining_60min=pct_remaining_60min,
        pct_remaining_2h=pct_remaining_2h,
        stability_class=stability_class,
        notes=notes_str,
    )


def screen_matrices(
    drug_name: str,
    logP: float = 2.0,
    ester_group: bool = False,
    functional_groups: tuple | None = None,
) -> list:
    """Screen drug stability across all 6 biological matrices.

    Returns a list of BiomatrixStabilityResult objects sorted by t_half_h
    ascending (most labile first).
    """
    results = [
        predict_biomatrix_stability(
            drug_name=drug_name,
            matrix=matrix,
            logP=logP,
            ester_group=ester_group,
            functional_groups=functional_groups,
        )
        for matrix in sorted(VALID_MATRICES)
    ]
    return sorted(results, key=lambda r: r.t_half_h)
