"""Phase 19: Extended IVIVE engine — Caco-2 permeability scaling and PPB correction.

References:
  Sun D et al. Curr Top Med Chem 2004 (Papp→Peff regression)
  Yang J et al. Curr Drug Metab 2007 (gut availability)
  Houston JB, Carlile DJ. Drug Metab Rev 1997 (hepatic CLint scaling, in ivive.py)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omega_pbpk.clinical.ivive import (
    HUMAN_LIVER_WEIGHT_G,
    MPPGL,
    IVIVEResult,
    scale_hepatocyte_clint,
    scale_microsomal_clint,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAPP_PEFF_SLOPE = 0.6532  # Sun D et al., Curr Top Med Chem 2004
PAPP_PEFF_INTERCEPT = -0.3036
Q_GUT_L_PER_H = 18.0  # Yang et al. 2007
GUT_WEIGHT_G = 18.0  # empirical, ~1 % of liver CYP3A4

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Caco2IVIVEResult:
    """Result of Caco-2 permeability scaling to intestinal absorption."""

    papp_ab_cm_s: float
    papp_ba_cm_s: float | None
    peff_10_4_cm_s: float
    efflux_ratio: float
    fa_predicted: float
    fg_predicted: float
    absorption_classification: str  # "high", "moderate", "low"
    pgp_flag: str  # "passive", "efflux_substrate", "strong_efflux_substrate"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PPBIVIVEResult:
    """Result of plasma-protein binding and microsomal binding correction."""

    fup_measured: float
    fup_corrected: float
    fu_mic: float
    fu_mic_source: str  # "measured", "estimated_logP", "default"
    rbp: float
    rbp_source: str  # "measured", "estimated"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class IVIVEBundleResult:
    """Combined IVIVE result bundling clearance, permeability, and binding."""

    clearance: IVIVEResult
    permeability: Caco2IVIVEResult | None
    binding: PPBIVIVEResult
    drug_params: dict[str, float | str]
    confidence: str  # "high", "medium", "low"
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# scale_caco2_papp
# ---------------------------------------------------------------------------


def scale_caco2_papp(
    papp_ab_cm_s: float,
    papp_ba_cm_s: float | None = None,
    clint_gut_uL_min_mg: float = 0.0,
    q_gut_L_per_h: float = Q_GUT_L_PER_H,
) -> Caco2IVIVEResult:
    """Scale Caco-2 Papp to intestinal Peff and predict oral absorption fraction.

    Args:
        papp_ab_cm_s: Apical→basolateral Papp (cm/s).
        papp_ba_cm_s: Basolateral→apical Papp (cm/s); None if not measured.
        clint_gut_uL_min_mg: Intestinal microsomal CLint (µL/min/mg protein).
        q_gut_L_per_h: Gut blood flow (L/h); default 18 L/h (Yang 2007).

    Returns:
        Caco2IVIVEResult with absorption predictions.
    """
    warnings: list[str] = []

    # 1. Papp → Peff (Sun 2004 regression)
    papp_1e6 = papp_ab_cm_s * 1e6  # convert to ×10⁻⁶ cm/s
    if papp_1e6 <= 0:
        papp_1e6 = 1e-6
        warnings.append("papp_ab_cm_s was non-positive; clamped to 1e-12 cm/s equivalent")

    log_peff = PAPP_PEFF_SLOPE * math.log10(papp_1e6) + PAPP_PEFF_INTERCEPT
    peff_10_4_cm_s = 10.0**log_peff
    peff_10_4_cm_s = max(0.01, min(peff_10_4_cm_s, 100.0))

    # 2. Fraction absorbed (absorption number approximation)
    an = peff_10_4_cm_s * 1.7
    fa = 1.0 - math.exp(-2.0 * an)
    fa = max(0.0, min(fa, 1.0))

    # 3. Efflux ratio and P-gp flag
    if papp_ba_cm_s is not None:
        efflux_ratio = papp_ba_cm_s / max(papp_ab_cm_s, 1e-15)
    else:
        efflux_ratio = 1.0

    if efflux_ratio < 2.0:
        pgp_flag = "passive"
    elif efflux_ratio < 10.0:
        pgp_flag = "efflux_substrate"
    else:
        pgp_flag = "strong_efflux_substrate"

    # 4. Absorption classification (based on Papp ×10⁻⁶ cm/s)
    if papp_1e6 > 10.0:
        absorption_classification = "high"
    elif papp_1e6 >= 1.0:
        absorption_classification = "moderate"
    else:
        absorption_classification = "low"

    # 5. Gut availability Fg (Yang 2007)
    if clint_gut_uL_min_mg > 0.0:
        clint_gut_L_per_h = clint_gut_uL_min_mg * MPPGL * GUT_WEIGHT_G / 1e6 * 60
        fg = q_gut_L_per_h / (q_gut_L_per_h + clint_gut_L_per_h)
        fg = max(0.0, min(fg, 1.0))
    else:
        fg = 1.0

    return Caco2IVIVEResult(
        papp_ab_cm_s=papp_ab_cm_s,
        papp_ba_cm_s=papp_ba_cm_s,
        peff_10_4_cm_s=peff_10_4_cm_s,
        efflux_ratio=efflux_ratio,
        fa_predicted=fa,
        fg_predicted=fg,
        absorption_classification=absorption_classification,
        pgp_flag=pgp_flag,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# scale_ppb
# ---------------------------------------------------------------------------


def scale_ppb(
    fup: float,
    fu_mic: float | None = None,
    logP: float | None = None,
    rbp: float | None = None,
    hematocrit: float = 0.45,
) -> PPBIVIVEResult:
    """Correct for plasma-protein binding and microsomal binding.

    Args:
        fup: Fraction unbound in plasma (0–1).
        fu_mic: Fraction unbound in microsomes (measured, 0–1); None to estimate.
        logP: Octanol–water partition coefficient; used to estimate fu_mic if needed.
        rbp: Blood-to-plasma ratio (measured); None to estimate.
        hematocrit: Blood hematocrit fraction; default 0.45.

    Returns:
        PPBIVIVEResult with corrected binding parameters.
    """
    warnings: list[str] = []

    # fu_mic determination
    if fu_mic is not None:
        fu_mic_val = fu_mic
        fu_mic_source = "measured"
    elif logP is not None:
        fu_mic_val = 1.0 / (1.0 + 0.0072 * 10.0**logP)
        fu_mic_source = "estimated_logP"
    else:
        fu_mic_val = 1.0
        fu_mic_source = "default"
        warnings.append("fu_mic not provided and logP missing; defaulting fu_mic=1.0")

    # rbp determination
    if rbp is not None:
        rbp_val = rbp
        rbp_source = "measured"
    else:
        rbp_val = 1.0 + hematocrit * (1.0 / max(fup, 1e-6) - 1.0) * 0.1
        rbp_val = max(0.55, min(rbp_val, 1.5))
        rbp_source = "estimated"

    return PPBIVIVEResult(
        fup_measured=fup,
        fup_corrected=fup,
        fu_mic=fu_mic_val,
        fu_mic_source=fu_mic_source,
        rbp=rbp_val,
        rbp_source=rbp_source,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# run_ivive_bundle
# ---------------------------------------------------------------------------


def run_ivive_bundle(
    clint_uL_min: float,
    clint_system: str = "microsomes",
    fup: float = 0.1,
    fu_mic: float | None = None,
    logP: float | None = None,
    papp_ab_cm_s: float | None = None,
    papp_ba_cm_s: float | None = None,
    rbp: float | None = None,
    mw: float = 300.0,
    drug_name: str = "ivive_compound",
    clint_gut_uL_min_mg: float = 0.0,
    liver_weight_g: float = HUMAN_LIVER_WEIGHT_G,
    q_liver_L_per_h: float = 96.6,
) -> IVIVEBundleResult:
    """Run full IVIVE bundle: clearance + permeability + protein binding.

    Args:
        clint_uL_min: In vitro CLint in µL/min/mg (microsomes) or µL/min/million cells
            (hepatocytes).
        clint_system: "microsomes" or "hepatocytes".
        fup: Fraction unbound in plasma.
        fu_mic: Microsomal unbound fraction (measured); None to estimate.
        logP: logP for estimating fu_mic if not measured.
        papp_ab_cm_s: Caco-2 A→B Papp (cm/s); None skips permeability scaling.
        papp_ba_cm_s: Caco-2 B→A Papp (cm/s); None if not available.
        rbp: Blood-to-plasma ratio (measured); None to estimate.
        mw: Molecular weight (Da).
        drug_name: Identifier for the compound.
        clint_gut_uL_min_mg: Intestinal microsomal CLint (µL/min/mg) for Fg calculation.
        liver_weight_g: Liver weight (g); default 1500 g (70 kg adult).
        q_liver_L_per_h: Hepatic blood flow (L/h); default 96.6 L/h.

    Returns:
        IVIVEBundleResult aggregating all sub-results.
    """
    # 1. Binding correction
    binding = scale_ppb(fup, fu_mic, logP, rbp)
    fu_mic_effective = binding.fu_mic

    # 2. Hepatic clearance scaling
    if clint_system == "microsomes":
        cl = scale_microsomal_clint(
            clint_uL_min_mg=clint_uL_min,
            fup=fup,
            fu_mic=fu_mic_effective,
            liver_weight_g=liver_weight_g,
            q_liver_L_per_h=q_liver_L_per_h,
        )
    elif clint_system == "hepatocytes":
        cl = scale_hepatocyte_clint(
            clint_uL_min_million_cells=clint_uL_min,
            fup=fup,
            liver_weight_g=liver_weight_g,
            q_liver_L_per_h=q_liver_L_per_h,
        )
    else:
        raise ValueError(
            f"clint_system must be 'microsomes' or 'hepatocytes', got {clint_system!r}"
        )

    # 3. Permeability scaling (optional)
    if papp_ab_cm_s is not None:
        permeability = scale_caco2_papp(papp_ab_cm_s, papp_ba_cm_s, clint_gut_uL_min_mg)
    else:
        permeability = None

    # 4. Drug parameter summary
    drug_params: dict[str, float | str] = {
        "name": drug_name,
        "mw": mw,
        "fup": fup,
        "rbp": binding.rbp,
        "clint_hepatic_L_per_h": cl.clint_liver_L_per_h,
        "peff": permeability.peff_10_4_cm_s if permeability is not None else 1.0,
    }

    # 5. Confidence scoring: count measured inputs
    measured_count = 0
    if fu_mic is not None:
        measured_count += 1
    if papp_ab_cm_s is not None:
        measured_count += 1
    if rbp is not None:
        measured_count += 1

    if measured_count >= 3:
        confidence = "high"
    elif measured_count >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    # 6. Aggregate warnings
    all_warnings: list[str] = []
    all_warnings.extend(binding.warnings)
    if permeability is not None:
        all_warnings.extend(permeability.warnings)

    return IVIVEBundleResult(
        clearance=cl,
        permeability=permeability,
        binding=binding,
        drug_params=drug_params,
        confidence=confidence,
        warnings=tuple(all_warnings),
    )
