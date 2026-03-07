"""Clearance scaling from in vitro to in vivo (IVIVE) and across species.

Provides microsomal CLint scaling to hepatic clearance (well-stirred model),
simple allometric CL scaling, and multi-species log-linear regression for
human CL prediction.

References
----------
- Houston JB & Carlile DJ, Drug Metab Rev. 1997 (hepatic CLint scaling)
- Hallifax D & Houston JB, Drug Metab Dispos. 2006 (fu_mic correction)
- Mahmood I & Balian JD, Clin Pharmacokinet. 1996 (allometric scaling)
- Boxenbaum H, J Pharmacokinet Biopharm. 1982 (interspecies allometry)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CLScalingResult",
    "scale_clint_to_clh",
    "allometric_cl_scaling",
    "multi_species_cl_scaling",
]


@dataclass(frozen=True)
class CLScalingResult:
    """Summary of clearance scaling prediction.

    Attributes
    ----------
    compound_name : Name of the compound.
    clint_in_vitro : In vitro CLint (uL/min/mg microsomal protein).
    clh_predicted : Predicted hepatic clearance (L/h).
    er : Hepatic extraction ratio (0-1).
    method : Scaling method used.
    notes : Summary text.
    """

    compound_name: str
    clint_in_vitro: float
    clh_predicted: float
    er: float
    method: str
    notes: str


def scale_clint_to_clh(
    clint_uL_per_min_per_mg: float,
    liver_weight_g: float = 1500.0,
    protein_conc_mg_per_mL: float = 1.0,
    fup: float = 1.0,
    rbp: float = 1.0,
    qh_L_per_h: float = 90.0,
) -> dict:
    """Scale microsomal CLint to hepatic clearance (well-stirred model).

    Scaling steps:
    1. CLint_scaled (L/h) = clint * protein_conc * liver_weight * 60 / 1e6
    2. fu_mic correction (Hallifax & Houston 2006):
       fu_mic = 1 / (1 + 0.072 * protein_conc_mg_per_mL)
    3. CLint_u = CLint_scaled / fu_mic
    4. CLh = Qh * (fup * CLint_u / rbp) / (Qh + fup * CLint_u / rbp)
       [well-stirred hepatic model]

    Parameters
    ----------
    clint_uL_per_min_per_mg : Intrinsic clearance from microsomal assay (uL/min/mg protein).
    liver_weight_g : Human liver weight (g), default 1500 g.
    protein_conc_mg_per_mL : Microsomal protein concentration in assay (mg/mL), default 1.0.
    fup : Fraction unbound in plasma (0-1).
    rbp : Blood-to-plasma concentration ratio.
    qh_L_per_h : Hepatic blood flow (L/h), default 90 L/h.

    Returns
    -------
    dict with keys: clint_scaled_L_per_h, fu_mic, clint_u_L_per_h,
                    clh_L_per_h, er, notes

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if clint_uL_per_min_per_mg <= 0:
        raise ValueError("clint_uL_per_min_per_mg must be > 0")
    if liver_weight_g <= 0:
        raise ValueError("liver_weight_g must be > 0")
    if protein_conc_mg_per_mL <= 0:
        raise ValueError("protein_conc_mg_per_mL must be > 0")
    if not (0 < fup <= 1):
        raise ValueError("fup must be in (0, 1]")
    if rbp <= 0:
        raise ValueError("rbp must be > 0")
    if qh_L_per_h <= 0:
        raise ValueError("qh_L_per_h must be > 0")

    # Scale to in vivo CLint (L/h)
    # uL/min/mg * mg_protein/mL * mL/g_liver * g_liver * 1000 mL/L * 60 min/h / 1e6 uL/mL
    # Simplified: clint * protein_conc * liver_weight * 60 / 1e6
    clint_scaled_L_per_h = (
        clint_uL_per_min_per_mg * protein_conc_mg_per_mL * liver_weight_g * 60.0 / 1e6
    )

    # Microsomal protein binding correction (Hallifax & Houston 2006)
    fu_mic = 1.0 / (1.0 + 0.072 * protein_conc_mg_per_mL)

    # Unbound CLint
    clint_u_L_per_h = clint_scaled_L_per_h / fu_mic

    # Well-stirred hepatic model
    fu_clint_u = fup * clint_u_L_per_h / rbp
    clh_L_per_h = qh_L_per_h * fu_clint_u / (qh_L_per_h + fu_clint_u)

    er = clh_L_per_h / qh_L_per_h

    notes = (
        f"Microsomal IVIVE: CLint={clint_uL_per_min_per_mg:.2f} uL/min/mg -> "
        f"CLint_scaled={clint_scaled_L_per_h:.3f} L/h, fu_mic={fu_mic:.3f}, "
        f"CLint_u={clint_u_L_per_h:.3f} L/h, CLh={clh_L_per_h:.3f} L/h, ER={er:.3f}. "
        f"fup={fup}, rbp={rbp}, Qh={qh_L_per_h} L/h."
    )

    return {
        "clint_scaled_L_per_h": clint_scaled_L_per_h,
        "fu_mic": fu_mic,
        "clint_u_L_per_h": clint_u_L_per_h,
        "clh_L_per_h": clh_L_per_h,
        "er": er,
        "notes": notes,
    }


def allometric_cl_scaling(
    cl_animal_L_per_h: float,
    bw_animal_kg: float,
    bw_human_kg: float = 70.0,
    exponent: float = 0.75,
) -> dict:
    """Scale clearance from animal to human using simple allometry.

    CL_human = CL_animal * (BW_human / BW_animal) ^ exponent

    Parameters
    ----------
    cl_animal_L_per_h : Animal clearance (L/h).
    bw_animal_kg : Animal body weight (kg).
    bw_human_kg : Human body weight (kg), default 70 kg.
    exponent : Allometric exponent, default 0.75 (standard for CL).

    Returns
    -------
    dict with keys: cl_human_L_per_h, scaling_factor, exponent, notes

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if cl_animal_L_per_h <= 0:
        raise ValueError("cl_animal_L_per_h must be > 0")
    if bw_animal_kg <= 0:
        raise ValueError("bw_animal_kg must be > 0")
    if bw_human_kg <= 0:
        raise ValueError("bw_human_kg must be > 0")

    scaling_factor = (bw_human_kg / bw_animal_kg) ** exponent
    cl_human_L_per_h = cl_animal_L_per_h * scaling_factor

    notes = (
        f"Allometric scaling: CL_animal={cl_animal_L_per_h:.3f} L/h "
        f"(BW={bw_animal_kg} kg) -> CL_human={cl_human_L_per_h:.3f} L/h "
        f"(BW={bw_human_kg} kg), exponent={exponent}, factor={scaling_factor:.3f}."
    )

    return {
        "cl_human_L_per_h": cl_human_L_per_h,
        "scaling_factor": scaling_factor,
        "exponent": exponent,
        "notes": notes,
    }


def multi_species_cl_scaling(
    species_data: list,
    bw_human_kg: float = 70.0,
) -> dict:
    """Scale clearance from multiple species to human using log-linear regression.

    Fits: log(CL) = intercept + exponent * log(BW)
    Then predicts human CL at bw_human_kg.

    Parameters
    ----------
    species_data : List of dicts, each with keys:
        - species (str): species name
        - cl_L_per_h (float): clearance (L/h)
        - bw_kg (float): body weight (kg)
    bw_human_kg : Human body weight for prediction (kg), default 70.

    Returns
    -------
    dict with keys: predicted_cl_human_L_per_h, fitted_exponent, r_squared,
                    species_used, notes

    Raises
    ------
    ValueError : If fewer than 2 species provided or invalid data.
    """
    if len(species_data) < 2:
        raise ValueError("At least 2 species required for multi-species scaling")

    # Build log-log data
    log_bw = []
    log_cl = []
    species_used = []

    for entry in species_data:
        cl = entry.get("cl_L_per_h", 0.0)
        bw = entry.get("bw_kg", 0.0)
        sp = entry.get("species", "unknown")

        if cl <= 0:
            raise ValueError(f"cl_L_per_h must be > 0 for species {sp}")
        if bw <= 0:
            raise ValueError(f"bw_kg must be > 0 for species {sp}")

        log_bw.append(math.log(bw))
        log_cl.append(math.log(cl))
        species_used.append(sp)

    n = len(log_bw)

    # Linear regression: log_cl = intercept + exponent * log_bw
    sum_x = sum(log_bw)
    sum_y = sum(log_cl)
    sum_xx = sum(x * x for x in log_bw)
    sum_xy = sum(x * y for x, y in zip(log_bw, log_cl))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        raise ValueError("Collinear data — cannot fit log-linear regression")

    exponent = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - exponent * sum_x) / n

    # Predict human CL
    log_cl_human = intercept + exponent * math.log(bw_human_kg)
    cl_human_L_per_h = math.exp(log_cl_human)

    # R-squared
    mean_y = sum_y / n
    ss_tot = sum((y - mean_y) ** 2 for y in log_cl)
    ss_res = sum((y - (intercept + exponent * x)) ** 2 for x, y in zip(log_bw, log_cl))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    r_squared = max(0.0, min(1.0, r_squared))

    notes = (
        f"Multi-species allometry ({n} species: {', '.join(species_used)}). "
        f"Fitted exponent={exponent:.3f}, intercept={intercept:.3f}, R2={r_squared:.3f}. "
        f"Predicted CL_human={cl_human_L_per_h:.3f} L/h at BW={bw_human_kg} kg."
    )

    return {
        "predicted_cl_human_L_per_h": cl_human_L_per_h,
        "fitted_exponent": exponent,
        "r_squared": r_squared,
        "species_used": species_used,
        "notes": notes,
    }
