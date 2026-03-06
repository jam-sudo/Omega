"""Protein binding displacement DDI — competitive plasma protein binding."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BindingDrug:
    """Drug with plasma protein binding parameters."""

    name: str
    fup: float  # fraction unbound in plasma (0, 1]
    fu_tissue: float  # fraction unbound in tissue (0, 1]
    vd_L: float  # volume of distribution at baseline fup (L)
    cl_L_per_h: float  # total clearance at baseline fup (L/h)
    ka_albumin: float = 0.0  # association constant for albumin (L/μmol), optional
    ka_agp: float = 0.0  # association constant for α1-acid glycoprotein, optional


@dataclass(frozen=True)
class DisplacementResult:
    """Result of protein binding displacement analysis."""

    victim_name: str
    perpetrator_name: str
    fup_baseline: float
    fup_displaced: float
    fup_ratio: float  # fup_displaced / fup_baseline
    vd_displaced_L: float  # Vd after displacement
    cl_displaced_L_per_h: float  # CL after displacement (low-extraction drugs)
    auc_ratio: float  # AUC_displaced / AUC_baseline (for low-extraction)
    cmax_ratio: float  # Cmax ratio (= fup_ratio for low-extraction)
    extraction_ratio: float  # hepatic extraction ratio (0–1)
    is_high_extraction: bool  # E > 0.7
    clinical_significance: str  # "none", "minor", "moderate", "major"
    warnings: list[str] = field(default_factory=list)


def _hepatic_extraction(fup: float, cl_int_L_per_h: float, q_liver: float = 90.0) -> float:
    """Well-stirred model: E = fup * CLint / (Q + fup * CLint)."""
    numerator = fup * cl_int_L_per_h
    return numerator / (q_liver + numerator)


def _vd_after_displacement(
    vd_baseline: float,
    fup_baseline: float,
    fup_new: float,
    fu_tissue: float,
) -> float:
    """Vd after fup change: Vd_new = Vp + (fup_new/fu_tissue)*Vt.

    Uses: Vd = Vp + (fup/fut)*Vt → rearrange to get Vt, then recompute.
    Simplified: Vd_new ≈ Vd_baseline * (1 + (fup_new - fup_baseline) * Vt_apparent / Vd_baseline)
    Exact: Vd_new = Vp + (fup_new / fu_tissue) * Vt
    where Vt = (Vd_baseline - Vp) * fu_tissue / fup_baseline, Vp = 4.0 L (typical plasma volume)
    """
    vp = 4.0  # plasma volume, L
    vt = max((vd_baseline - vp) * fu_tissue / fup_baseline, 0.0)
    return vp + (fup_new / fu_tissue) * vt


def predict_displacement(
    victim: BindingDrug,
    perpetrator_fup_effect: float,
    cl_int_L_per_h: float | None = None,
    q_liver_L_per_h: float = 90.0,
    perpetrator_name: str = "perpetrator",
) -> DisplacementResult:
    """Predict PK changes due to protein binding displacement.

    Parameters
    ----------
    victim : BindingDrug
        The drug whose binding is being displaced.
    perpetrator_fup_effect : float
        New fup of victim after displacement (0, 1].
    cl_int_L_per_h : float | None
        Intrinsic hepatic clearance (L/h). If None, estimated from CL and fup.
    q_liver_L_per_h : float
        Hepatic blood flow (L/h), default 90.
    perpetrator_name : str
        Name of displacing drug.
    """
    warnings: list[str] = []

    fup_base = float(np.clip(victim.fup, 1e-6, 1.0))
    fup_new = float(np.clip(perpetrator_fup_effect, 1e-6, 1.0))

    if fup_new <= fup_base:
        warnings.append("perpetrator_fup_effect is not greater than baseline fup — no displacement")

    fup_ratio = fup_new / fup_base

    # Estimate intrinsic clearance from baseline CL and fup (well-stirred)
    if cl_int_L_per_h is None:
        # CL = E * Q → E = CL / Q; E = fup*CLint/(Q + fup*CLint) → CLint = E*Q / (fup*(1-E))
        e_base = victim.cl_L_per_h / q_liver_L_per_h
        e_base = min(e_base, 0.999)
        if e_base > 0 and (1.0 - e_base) > 0:
            cl_int = e_base * q_liver_L_per_h / (fup_base * (1.0 - e_base))
        else:
            cl_int = victim.cl_L_per_h / fup_base
    else:
        cl_int = cl_int_L_per_h

    # Extraction ratios
    e_base = _hepatic_extraction(fup_base, cl_int, q_liver_L_per_h)
    e_new = _hepatic_extraction(fup_new, cl_int, q_liver_L_per_h)
    is_high = e_base >= 0.6

    # Displaced clearance
    if is_high:
        # High-extraction: CL ≈ Q (blood flow limited) — displacement barely changes CL
        cl_displaced = q_liver_L_per_h * e_new
        warnings.append(
            f"High-extraction drug (E={e_base:.2f}): CL is blood-flow limited; "
            "displacement has minimal effect on CL but increases free Cmax"
        )
    else:
        # Low-extraction: CL ≈ fup * CLint — proportional to fup
        cl_displaced = fup_new * cl_int

    # Displaced Vd
    vd_displaced = _vd_after_displacement(victim.vd_L, fup_base, fup_new, victim.fu_tissue)

    # Unbound AUC ratio: for low-extraction, CL ∝ fup so AUC_total ∝ 1/fup,
    # but AUC_unbound = AUC_total * fup → stays constant (ratio = 1.0).
    # For high-extraction, CL is blood-flow limited; total AUC changes with fup.
    if is_high:
        auc_ratio = victim.cl_L_per_h / cl_displaced if victim.cl_L_per_h > 0 else 1.0
    else:
        auc_ratio = 1.0  # unbound AUC unchanged for low-extraction drugs

    # Cmax ratio (free drug): for low-extraction, total Cmax unchanged, free Cmax ∝ fup
    cmax_ratio = fup_ratio if not is_high else 1.0

    # Clinical significance based on fup_ratio
    if fup_ratio < 1.25:
        significance = "none"
    elif fup_ratio < 1.5:
        significance = "minor"
        warnings.append("Minor displacement: monitor drug levels")
    elif fup_ratio < 2.0:
        significance = "moderate"
        warnings.append("Moderate displacement: dose adjustment may be needed")
    else:
        significance = "major"
        warnings.append("Major displacement (>2-fold fup increase): significant clinical risk")

    return DisplacementResult(
        victim_name=victim.name,
        perpetrator_name=perpetrator_name,
        fup_baseline=fup_base,
        fup_displaced=fup_new,
        fup_ratio=fup_ratio,
        vd_displaced_L=vd_displaced,
        cl_displaced_L_per_h=cl_displaced,
        auc_ratio=auc_ratio,
        cmax_ratio=cmax_ratio,
        extraction_ratio=e_base,
        is_high_extraction=is_high,
        clinical_significance=significance,
        warnings=warnings,
    )


def multi_drug_displacement(
    victim: BindingDrug,
    perpetrators: list[tuple[str, float]],  # (name, delta_fup_additive)
    cl_int_L_per_h: float | None = None,
    q_liver_L_per_h: float = 90.0,
) -> list[DisplacementResult]:
    """Predict displacement from multiple perpetrators (additive fup increase).

    Parameters
    ----------
    perpetrators : list[tuple[str, float]]
        Each tuple is (name, additional_fup). Fup increases are additive up to max 1.0.
    """
    results = []
    cumulative_fup = victim.fup

    for name, delta_fup in perpetrators:
        new_fup = min(cumulative_fup + delta_fup, 1.0)
        result = predict_displacement(
            victim=victim,
            perpetrator_fup_effect=new_fup,
            cl_int_L_per_h=cl_int_L_per_h,
            q_liver_L_per_h=q_liver_L_per_h,
            perpetrator_name=name,
        )
        results.append(result)
        cumulative_fup = new_fup

    return results


__all__ = [
    "BindingDrug",
    "DisplacementResult",
    "predict_displacement",
    "multi_drug_displacement",
]
