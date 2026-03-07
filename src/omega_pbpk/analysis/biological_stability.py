"""Drug stability in biological matrices.

Predicts drug degradation in plasma, whole blood, liver microsomes,
and GI fluids using empirical and mechanistic models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PlasmaStabilityResult",
    "MicrosomalStabilityResult",
    "GIStabilityResult",
    "BloodStabilityResult",
    "plasma_stability",
    "microsomal_stability",
    "gi_fluid_stability",
    "blood_stability",
]

# Hepatic blood flow (L/h) — standard value
_QH_L_PER_H = 90.0

# Default GI pH values: stomach, duodenum, jejunum, ileum
_DEFAULT_GI_PH = [1.5, 6.0, 6.5, 7.0]
_GI_REGION_NAMES = ["stomach", "duodenum", "jejunum", "ileum"]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlasmaStabilityResult:
    """Result of plasma stability prediction."""

    compound_name: str
    t_half_plasma_min: float
    k_deg_per_min: float
    pct_remaining_30min: float
    pct_remaining_60min: float
    pct_remaining_120min: float
    stability_class: str  # very_stable / stable / moderate / unstable
    notes: str


@dataclass(frozen=True)
class MicrosomalStabilityResult:
    """Result of microsomal stability prediction."""

    compound_name: str
    clint_uL_per_min_per_mg: float
    t_half_microsomes_min: float
    cl_hepatic_L_per_h: float
    extraction_ratio: float
    stability_class: str  # very_stable / stable / moderate / unstable
    notes: str


@dataclass(frozen=True)
class GIStabilityResult:
    """Result of GI fluid stability prediction."""

    compound_name: str
    ph_values: list[float]
    k_deg_per_h: list[float]
    pct_remaining_2h: list[float]
    most_labile_region: str
    notes: str


@dataclass(frozen=True)
class BloodStabilityResult:
    """Result of whole blood stability prediction."""

    compound_name: str
    t_half_blood_min: float
    k_deg_per_min: float
    pct_remaining_30min: float
    is_ester: bool
    notes: str


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _classify_plasma_stability(t_half_min: float) -> str:
    """Classify stability based on plasma half-life."""
    if t_half_min > 120:
        return "very_stable"
    elif t_half_min >= 60:
        return "stable"
    elif t_half_min >= 30:
        return "moderate"
    else:
        return "unstable"


def _pct_remaining(k_deg_per_min: float, t_min: float) -> float:
    """Percent drug remaining after time t_min with first-order degradation."""
    return 100.0 * math.exp(-k_deg_per_min * t_min)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plasma_stability(
    compound_name: str,
    t_half_plasma_min: float | None = None,
    logP: float = 2.0,
    pka: float = 7.0,
    fu_plasma: float = 0.1,
) -> PlasmaStabilityResult:
    """Predict drug stability in plasma.

    Parameters
    ----------
    compound_name:
        Identifier for the compound.
    t_half_plasma_min:
        Observed plasma half-life (min). If provided, used directly.
        Otherwise estimated from physicochemical properties.
    logP:
        Octanol-water partition coefficient.
    pka:
        pKa of ionisable group (informational; not used in current estimate).
    fu_plasma:
        Fraction unbound in plasma (0, 1].

    Returns
    -------
    PlasmaStabilityResult
    """
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")

    if t_half_plasma_min is not None:
        if t_half_plasma_min <= 0:
            raise ValueError(f"t_half_plasma_min must be > 0, got {t_half_plasma_min}")
        t_half = t_half_plasma_min
        notes = "t_half provided directly"
    else:
        # Empirical estimate: t_half = 30 * exp(0.2 * logP) * fu_plasma^0.3
        t_half = 30.0 * math.exp(0.2 * logP) * (fu_plasma**0.3)
        notes = f"t_half estimated from logP={logP:.2f}, fu_plasma={fu_plasma:.3f}"

    k_deg = math.log(2) / t_half
    pct_30 = _pct_remaining(k_deg, 30.0)
    pct_60 = _pct_remaining(k_deg, 60.0)
    pct_120 = _pct_remaining(k_deg, 120.0)
    stability_class = _classify_plasma_stability(t_half)

    return PlasmaStabilityResult(
        compound_name=compound_name,
        t_half_plasma_min=t_half,
        k_deg_per_min=k_deg,
        pct_remaining_30min=pct_30,
        pct_remaining_60min=pct_60,
        pct_remaining_120min=pct_120,
        stability_class=stability_class,
        notes=notes,
    )


def microsomal_stability(
    compound_name: str,
    clint_uL_per_min_per_mg: float = 5.0,
    protein_mg_per_mL: float = 0.5,
    fu_mic: float = 1.0,
    liver_weight_g: float = 1500.0,
    hepatocyte_density: float = 1e8,
) -> MicrosomalStabilityResult:
    """Predict hepatic clearance from microsomal intrinsic clearance.

    Scales CLint to in-vivo hepatic clearance using the well-stirred model.

    Parameters
    ----------
    compound_name:
        Identifier for the compound.
    clint_uL_per_min_per_mg:
        In-vitro intrinsic clearance (uL/min/mg protein). Must be > 0.
    protein_mg_per_mL:
        Microsomal protein concentration (mg/mL).
    fu_mic:
        Fraction unbound in microsomes (0, 1].
    liver_weight_g:
        Liver weight (g), default 1500 g for a 70-kg human.
    hepatocyte_density:
        Hepatocyte density (cells/g liver), informational.

    Returns
    -------
    MicrosomalStabilityResult
    """
    if clint_uL_per_min_per_mg <= 0:
        raise ValueError(f"clint_uL_per_min_per_mg must be > 0, got {clint_uL_per_min_per_mg}")
    if fu_mic <= 0 or fu_mic > 1:
        raise ValueError(f"fu_mic must be in (0, 1], got {fu_mic}")
    if liver_weight_g <= 0:
        raise ValueError(f"liver_weight_g must be > 0, got {liver_weight_g}")

    # Scale CLint to whole-liver in vivo (L/h)
    # CLint_hep (uL/min) = CLint * protein_mg_per_mL * volume_per_g * liver_weight
    # Using 1 mL/g microsomal suspension volume scaling; convert uL->L and min->h
    clint_hep_L_per_h = (
        clint_uL_per_min_per_mg * protein_mg_per_mL * liver_weight_g * 60.0 / 1e6 * fu_mic
    )

    # Well-stirred model: CL_H = QH * fu * CLint / (QH + fu * CLint)
    # Here fu_mic already incorporated into CLint_hep; use fu_mic=1 in formula
    cl_h = _QH_L_PER_H * clint_hep_L_per_h / (_QH_L_PER_H + clint_hep_L_per_h)
    extraction_ratio = cl_h / _QH_L_PER_H

    # Approximate microsomal t_half from incubation clearance
    # V_inc_mL = 1 mL (standard assay volume)
    v_inc_mL = 1.0
    protein_total_mg = protein_mg_per_mL * v_inc_mL
    # CLint in uL/min for the incubation
    clint_inc_uL_per_min = clint_uL_per_min_per_mg * protein_total_mg
    # t_half in incubation = ln(2) * V_inc / CLint  (V in uL)
    v_inc_uL = v_inc_mL * 1000.0
    t_half_mic = math.log(2) * v_inc_uL / max(clint_inc_uL_per_min, 1e-9)

    # Classify based on CLint
    if clint_uL_per_min_per_mg < 10:
        stability_class = "stable"
    elif clint_uL_per_min_per_mg < 30:
        stability_class = "moderate"
    elif clint_uL_per_min_per_mg < 100:
        stability_class = "unstable"
    else:
        stability_class = "very_unstable"

    notes = (
        f"CLint_hep={clint_hep_L_per_h:.3f} L/h; "
        f"ER={extraction_ratio:.3f}; "
        f"well-stirred model, QH={_QH_L_PER_H} L/h"
    )

    return MicrosomalStabilityResult(
        compound_name=compound_name,
        clint_uL_per_min_per_mg=clint_uL_per_min_per_mg,
        t_half_microsomes_min=t_half_mic,
        cl_hepatic_L_per_h=cl_h,
        extraction_ratio=extraction_ratio,
        stability_class=stability_class,
        notes=notes,
    )


def gi_fluid_stability(
    compound_name: str,
    ph_values: list[float] | None = None,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    logP: float = 2.0,
) -> GIStabilityResult:
    """Predict drug stability across GI fluid pH values.

    Degradation rates depend on acid/base lability and logP.

    Parameters
    ----------
    compound_name:
        Identifier for the compound.
    ph_values:
        List of pH values to evaluate. Defaults to [1.5, 6.0, 6.5, 7.0].
    pka_acid:
        pKa of acidic group. Compounds with pka_acid < 4 are acid-labile.
    pka_base:
        pKa of basic group. Triggers base-labile degradation pathway.
    logP:
        Octanol-water partition coefficient.

    Returns
    -------
    GIStabilityResult
    """
    if ph_values is None:
        ph_values = list(_DEFAULT_GI_PH)

    if not ph_values:
        raise ValueError("ph_values must not be empty")

    k_deg_list: list[float] = []
    pct_remaining_list: list[float] = []

    for ph in ph_values:
        k_acid = 0.0
        k_base = 0.0
        k_neutral = 0.001 * logP if logP > 0 else 1e-6

        if pka_acid is not None and pka_acid < 4:
            k_acid = 0.1 * math.exp(-0.5 * ph)

        if pka_base is not None:
            k_base = 0.01 * math.exp(0.3 * (ph - 7.0))

        k_total = max(k_acid + k_base + k_neutral, 0.0)
        k_deg_list.append(k_total)
        # Percent remaining after 2 h (k in h^-1)
        pct = 100.0 * math.exp(-k_total * 2.0)
        pct_remaining_list.append(pct)

    # Identify most labile region (highest k_deg)
    max_idx = k_deg_list.index(max(k_deg_list))
    # Map index to region name or pH label
    if ph_values == list(_DEFAULT_GI_PH):
        most_labile = _GI_REGION_NAMES[max_idx]
    else:
        most_labile = f"pH={ph_values[max_idx]:.1f}"

    notes = (
        f"pka_acid={pka_acid}, pka_base={pka_base}, logP={logP:.2f}; "
        "k_acid + k_base + k_neutral model"
    )

    return GIStabilityResult(
        compound_name=compound_name,
        ph_values=list(ph_values),
        k_deg_per_h=k_deg_list,
        pct_remaining_2h=pct_remaining_list,
        most_labile_region=most_labile,
        notes=notes,
    )


def blood_stability(
    compound_name: str,
    t_half_blood_min: float | None = None,
    is_ester: bool = False,
    logP: float = 2.0,
) -> BloodStabilityResult:
    """Predict drug stability in whole blood.

    Esters are rapidly hydrolyzed by blood esterases. Non-esters are more
    stable and primarily degraded by oxidative or non-enzymatic pathways.

    Parameters
    ----------
    compound_name:
        Identifier for the compound.
    t_half_blood_min:
        Observed blood half-life (min). If provided, used directly.
    is_ester:
        True if the compound contains an ester functional group.
    logP:
        Octanol-water partition coefficient (influences non-ester stability).

    Returns
    -------
    BloodStabilityResult
    """
    if t_half_blood_min is not None:
        if t_half_blood_min <= 0:
            raise ValueError(f"t_half_blood_min must be > 0, got {t_half_blood_min}")
        t_half = t_half_blood_min
        notes = "t_half provided directly"
    else:
        if is_ester:
            # Esters: t_half 5-60 min, shorter for higher logP (more enzyme access)
            # logP range roughly -2 to 6 → t_half 5–60 min
            t_half = max(5.0, 60.0 - 9.0 * logP)
            t_half = min(t_half, 60.0)
            notes = f"ester hydrolysis by blood esterases; logP={logP:.2f}"
        else:
            # Non-esters: t_half = 120 + 30 * logP
            t_half = 120.0 + 30.0 * logP
            notes = f"non-ester; t_half estimated from logP={logP:.2f}"

    k_deg = math.log(2) / t_half
    pct_30 = _pct_remaining(k_deg, 30.0)

    return BloodStabilityResult(
        compound_name=compound_name,
        t_half_blood_min=t_half,
        k_deg_per_min=k_deg,
        pct_remaining_30min=pct_30,
        is_ester=is_ester,
        notes=notes,
    )
