"""Drug metabolism rate prediction from structural features.

Predicts in vitro metabolic clearance rates using empirical physicochemical
rules and scales to in vivo hepatic clearance using the well-stirred model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "MetabolismRateResult",
    "predict_metabolism_rate",
    "compare_structural_modifications",
]

_VALID_STABILITY = ("high", "moderate", "low")


@dataclass(frozen=True)
class MetabolismRateResult:
    """Result of in vitro / in vivo metabolism rate prediction."""

    compound_name: str

    # Predicted in vitro rates
    clint_hep_uL_per_min_per_mg: float  # Human liver microsome CLint
    clint_s9_uL_per_min_per_mg: float  # S9 fraction CLint (lower than HLM)
    clint_hepatocyte_uL_per_min_per_cell: float  # Hepatocyte CLint

    # CYP attribution
    predicted_fm_cyp3a4: float  # Fraction metabolized by CYP3A4
    predicted_fm_cyp2d6: float
    predicted_fm_cyp2c9: float
    predicted_fm_cyp2c19: float
    predicted_fm_cyp1a2: float
    fm_total: float  # Should sum to <=1.0 (less if non-CYP metabolism)

    # Scaled in vivo
    cl_hep_predicted_L_per_h: float  # In vivo hepatic CL (well-stirred)
    cl_sys_predicted_L_per_h: float  # Systemic CL estimate
    t_half_predicted_h: float  # Predicted t1/2

    # High-risk metabolism flags
    cyp3a4_dominant: bool  # CYP3A4 > 70% -> DDI vulnerable
    cyp2d6_substrate: bool  # CYP2D6 > 30% -> PGx-sensitive
    non_cyp_metabolism: bool  # fm_total < 0.7 -> non-CYP pathways likely

    # Stability prediction
    microsomal_stability: str  # "high" / "moderate" / "low"

    notes: str


def predict_metabolism_rate(
    compound_name: str,
    logP: float,
    mw: float,
    n_hbd: int,
    n_hba: int,
    pka_base: float | None = None,
    pka_acid: float | None = None,
    fu_plasma: float = 0.5,
    vd_L: float = 70.0,
    hepatic_blood_flow_L_per_h: float = 90.0,
) -> MetabolismRateResult:
    """Predict metabolic clearance rates from physicochemical properties.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol). Must be > 0.
    n_hbd:
        Number of hydrogen bond donors.
    n_hba:
        Number of hydrogen bond acceptors.
    pka_base:
        Basic pKa (e.g. amine). If drug is basic (pKa > 7), CYP2D6/3A4 likely.
    pka_acid:
        Acidic pKa (e.g. carboxylic acid). If pKa < 5, CYP2C9 likely.
    fu_plasma:
        Unbound fraction in plasma (0, 1].
    vd_L:
        Volume of distribution (L). Used for t1/2 calculation.
    hepatic_blood_flow_L_per_h:
        Hepatic blood flow (L/h). Default 90 L/h (adult human).

    Returns
    -------
    MetabolismRateResult
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if hepatic_blood_flow_L_per_h <= 0:
        raise ValueError("hepatic_blood_flow_L_per_h must be > 0")

    # ------------------------------------------------------------------ #
    # 1. Base CLint estimate from logP, MW, H-bond count                  #
    # ------------------------------------------------------------------ #
    exponent = 0.5 * logP - 0.005 * mw + 0.3 * (n_hbd + n_hba) * 0.1
    clint_base = 10.0**exponent

    # Clamp to physiological range
    clint_base = max(0.1, min(500.0, clint_base))

    # Structural adjustments
    adj = 1.0
    notes_parts: list[str] = []

    if mw > 500:
        adj *= 0.5
        notes_parts.append("high MW (>500): reduced CLint")

    if pka_base is not None and pka_base > 7:
        adj *= 1.3
        notes_parts.append("basic drug (pKa_base>7): CYP2D6/3A4 preference")

    if pka_acid is not None and pka_acid < 5:
        adj *= 0.8
        notes_parts.append("acidic drug (pKa_acid<5): slower metabolism")

    if logP > 4:
        adj *= 1.5
        notes_parts.append("high logP (>4): faster oxidation")

    clint_hep = max(0.1, min(500.0, clint_base * adj))

    # S9 and hepatocyte CLint
    clint_s9 = clint_hep * 0.3
    clint_hepatocyte = clint_hep / 45.0 * 1e6

    # ------------------------------------------------------------------ #
    # 2. CYP attribution                                                  #
    # ------------------------------------------------------------------ #
    fm_3a4 = 0.6
    fm_2d6 = 0.1
    fm_2c9 = 0.1
    fm_2c19 = 0.1
    fm_1a2 = 0.1

    # Adjustments based on structural indicators
    if pka_base is not None and pka_base > 7:
        fm_2d6 += 0.3
        fm_3a4 -= 0.2

    if pka_acid is not None and pka_acid < 5:
        fm_2c9 += 0.3
        fm_3a4 -= 0.2

    if logP < 1:
        fm_1a2 += 0.2
        fm_3a4 -= 0.1

    # Floor at 0
    fm_3a4 = max(0.0, fm_3a4)
    fm_2d6 = max(0.0, fm_2d6)
    fm_2c9 = max(0.0, fm_2c9)
    fm_2c19 = max(0.0, fm_2c19)
    fm_1a2 = max(0.0, fm_1a2)

    # Normalize so total does not exceed 1.0
    total = fm_3a4 + fm_2d6 + fm_2c9 + fm_2c19 + fm_1a2
    if total > 1.0:
        scale = 1.0 / total
        fm_3a4 *= scale
        fm_2d6 *= scale
        fm_2c9 *= scale
        fm_2c19 *= scale
        fm_1a2 *= scale
        total = 1.0

    fm_total = total

    # ------------------------------------------------------------------ #
    # 3. In vivo scaling (well-stirred model)                             #
    # ------------------------------------------------------------------ #
    # CLint in vitro (uL/min/mg) -> in vivo (L/h)
    # 1500 g liver; 45 mg microsomal protein / g liver; 60 min/h; /1e6 mL->L
    clint_invivo_L_per_h = clint_hep * (60.0 / 1000.0) * 45.0 * 1500.0 / 1000.0

    # Well-stirred hepatic CL
    q = hepatic_blood_flow_L_per_h
    fu_times_clint = fu_plasma * clint_invivo_L_per_h
    cl_hep = q * fu_times_clint / (q + fu_times_clint)

    cl_sys = cl_hep  # hepatic assumed primary

    if cl_sys > 0:
        t_half = 0.693 * vd_L / cl_sys
    else:
        t_half = float("inf")

    # ------------------------------------------------------------------ #
    # 4. Flags and stability classification                               #
    # ------------------------------------------------------------------ #
    cyp3a4_dominant = fm_3a4 > 0.7
    cyp2d6_substrate = fm_2d6 > 0.3
    non_cyp_metabolism = fm_total < 0.7

    if clint_hep < 10:
        microsomal_stability = "high"
    elif clint_hep <= 100:
        microsomal_stability = "moderate"
    else:
        microsomal_stability = "low"

    notes = "; ".join(notes_parts) if notes_parts else "No notable adjustments"

    return MetabolismRateResult(
        compound_name=compound_name,
        clint_hep_uL_per_min_per_mg=clint_hep,
        clint_s9_uL_per_min_per_mg=clint_s9,
        clint_hepatocyte_uL_per_min_per_cell=clint_hepatocyte,
        predicted_fm_cyp3a4=fm_3a4,
        predicted_fm_cyp2d6=fm_2d6,
        predicted_fm_cyp2c9=fm_2c9,
        predicted_fm_cyp2c19=fm_2c19,
        predicted_fm_cyp1a2=fm_1a2,
        fm_total=fm_total,
        cl_hep_predicted_L_per_h=cl_hep,
        cl_sys_predicted_L_per_h=cl_sys,
        t_half_predicted_h=t_half,
        cyp3a4_dominant=cyp3a4_dominant,
        cyp2d6_substrate=cyp2d6_substrate,
        non_cyp_metabolism=non_cyp_metabolism,
        microsomal_stability=microsomal_stability,
        notes=notes,
    )


def compare_structural_modifications(
    base_compound: dict,
    modifications: list[dict],
) -> list[MetabolismRateResult]:
    """Compare metabolic properties of structural modifications vs a base compound.

    Parameters
    ----------
    base_compound:
        Dict with keys: compound_name, logP, mw, n_hbd, n_hba, and optional
        pka_base, pka_acid, fu_plasma, vd_L, hepatic_blood_flow_L_per_h.
    modifications:
        List of dicts with the same keys. Each dict overrides fields from
        base_compound.

    Returns
    -------
    List of MetabolismRateResult sorted by cl_sys_predicted_L_per_h ascending
    (most metabolically stable compound first).
    """
    results: list[MetabolismRateResult] = []

    for mod in modifications:
        params = {**base_compound, **mod}
        result = predict_metabolism_rate(
            compound_name=params.get("compound_name", "unknown"),
            logP=params["logP"],
            mw=params["mw"],
            n_hbd=params["n_hbd"],
            n_hba=params["n_hba"],
            pka_base=params.get("pka_base"),
            pka_acid=params.get("pka_acid"),
            fu_plasma=params.get("fu_plasma", 0.5),
            vd_L=params.get("vd_L", 70.0),
            hepatic_blood_flow_L_per_h=params.get("hepatic_blood_flow_L_per_h", 90.0),
        )
        results.append(result)

    return sorted(results, key=lambda r: r.cl_sys_predicted_L_per_h)
