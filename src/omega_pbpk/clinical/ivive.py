"""In Vitro to In Vivo Extrapolation (IVIVE) for hepatic CLint.

Scales microsomal/hepatocyte intrinsic clearance to in vivo hepatic CL.
Reference: Houston JB, Carlile DJ. Drug Metab Rev 1997;29:891-922.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Human liver scaling factors (per 70 kg adult)
HUMAN_LIVER_WEIGHT_G = 1500.0       # g
MPPGL = 40.0                         # mg microsomal protein per g liver
HPGL = 120e6                         # hepatocytes per g liver (120 million/g)


@dataclass(frozen=True)
class IVIVEResult:
    """Result of IVIVE calculation."""
    clint_in_vitro: float             # µL/min/mg protein (or /million cells)
    system: str                       # "microsomes" or "hepatocytes"
    clint_liver_L_per_h: float        # Scaled to whole-liver CLint (L/h)
    clh_well_stirred_L_per_h: float   # Hepatic CL via well-stirred model
    fu_mic: float                     # Microsomal binding correction
    fup: float                        # Fraction unbound in plasma


def scale_microsomal_clint(
    clint_uL_min_mg: float,
    fup: float = 0.1,
    fu_mic: float = 1.0,
    liver_weight_g: float = HUMAN_LIVER_WEIGHT_G,
    mppgl: float = MPPGL,
    q_liver_L_per_h: float = 96.6,
) -> IVIVEResult:
    """Scale microsomal CLint to in vivo hepatic clearance.

    Steps:
    1. Correct for microsomal binding: CLint_corrected = CLint / fu_mic
    2. Scale to whole liver: CLint_liver = CLint_corrected × MPPGL × liver_weight
    3. Convert units: µL/min → mL/min → L/h
    4. Apply well-stirred model: CLh = Q × fup × CLint / (Q + fup × CLint)

    Args:
        clint_uL_min_mg: In vitro CLint (µL/min/mg microsomal protein)
        fup: Fraction unbound in plasma (0–1)
        fu_mic: Fraction unbound in microsomes (1.0 if not measured)
        liver_weight_g: Liver weight (g), default 1500g
        mppgl: mg microsomal protein per g liver
        q_liver_L_per_h: Hepatic blood flow (L/h)
    """
    # Step 1: Binding correction
    clint_corrected = clint_uL_min_mg / max(fu_mic, 0.001)

    # Step 2: Scale to whole liver (µL/min/mg × mg/g × g liver)
    clint_liver_uL_per_min = clint_corrected * mppgl * liver_weight_g

    # Step 3: Unit conversion: µL/min → L/h (/1000 /1000 × 60)
    clint_liver_L_per_h = clint_liver_uL_per_min * 60 / 1e6

    # Step 4: Well-stirred model
    clh = (q_liver_L_per_h * fup * clint_liver_L_per_h) / (
        q_liver_L_per_h + fup * clint_liver_L_per_h
    )

    return IVIVEResult(
        clint_in_vitro=clint_uL_min_mg,
        system="microsomes",
        clint_liver_L_per_h=clint_liver_L_per_h,
        clh_well_stirred_L_per_h=float(clh),
        fu_mic=fu_mic,
        fup=fup,
    )


def scale_hepatocyte_clint(
    clint_uL_min_million_cells: float,
    fup: float = 0.1,
    liver_weight_g: float = HUMAN_LIVER_WEIGHT_G,
    hpgl: float = HPGL,
    q_liver_L_per_h: float = 96.6,
) -> IVIVEResult:
    """Scale hepatocyte CLint to in vivo hepatic clearance.

    Args:
        clint_uL_min_million_cells: CLint (µL/min/million hepatocytes)
        fup: Fraction unbound in plasma
        liver_weight_g: Liver weight (g)
        hpgl: Hepatocytes per gram liver
        q_liver_L_per_h: Hepatic blood flow (L/h)
    """
    # Scale to whole liver (µL/min/10^6 cells × 10^6 cells/g × g)
    clint_liver_uL_per_min = clint_uL_min_million_cells * (hpgl / 1e6) * liver_weight_g

    # Unit conversion
    clint_liver_L_per_h = clint_liver_uL_per_min * 60 / 1e6

    # Well-stirred model
    clh = (q_liver_L_per_h * fup * clint_liver_L_per_h) / (
        q_liver_L_per_h + fup * clint_liver_L_per_h
    )

    return IVIVEResult(
        clint_in_vitro=clint_uL_min_million_cells,
        system="hepatocytes",
        clint_liver_L_per_h=clint_liver_L_per_h,
        clh_well_stirred_L_per_h=float(clh),
        fu_mic=1.0,
        fup=fup,
    )


def estimate_clint_for_target_clh(
    target_clh_L_per_h: float,
    fup: float = 0.1,
    system: str = "microsomes",
    q_liver_L_per_h: float = 96.6,
) -> float:
    """Back-calculate the in vitro CLint needed to achieve a target CLh.

    Useful for calibrating drug parameters.
    Returns CLint in µL/min/mg (microsomes) or µL/min/million cells (hepatocytes).
    """
    # Invert well-stirred: CLint = Q × CLh / (fup × (Q - CLh))
    denom = fup * max(q_liver_L_per_h - target_clh_L_per_h, 0.001)
    clint_liver_L_per_h = q_liver_L_per_h * target_clh_L_per_h / denom

    # Convert L/h → µL/min
    clint_liver_uL_per_min = clint_liver_L_per_h * 1e6 / 60

    if system == "microsomes":
        return clint_liver_uL_per_min / (MPPGL * HUMAN_LIVER_WEIGHT_G)
    else:
        return clint_liver_uL_per_min / ((HPGL / 1e6) * HUMAN_LIVER_WEIGHT_G)
