"""Phase 348 — Drug Diffusion in Tumor Interstitium.

Models radial drug diffusion through tumor interstitium from blood vessel to
necrotic core using a simplified 3-zone shell model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Characteristic diffusion barrier (cm^2): ratio zone_thickness^2 / D_ref
# where D_ref ~ 1e-6 cm^2/s is a reference; barrier = k_diff * (zone_um * 1e-4)^2
# k_diff = 0.1 (dimensionless scaling of the diffusion resistance per zone)
_K_DIFF = 0.1
# Zone thickness in micrometers
_ZONE_THICKNESS_UM = 100.0
# Reference diffusivity for barrier normalization (cm^2/s)
_D_REFERENCE = 1e-6

_VALID_PENETRATION_CLASSES = ("excellent", "moderate", "poor")


@dataclass(frozen=True)
class TumorDrugDiffusionResult:
    """Result of tumor drug diffusion model."""

    drug_name: str
    plasma_conc_mg_L: float
    mw_Da: float
    logP: float
    effective_diffusivity_cm2_per_s: float
    fu_plasma: float
    kp_tumor: float
    c_zone1_mg_L: float
    c_zone2_mg_L: float
    c_zone3_mg_L: float
    penetration_ratio_z3_z1: float
    tumor_to_plasma_ratio: float
    penetration_class: str
    notes: str


def _estimate_diffusivity(mw_Da: float) -> float:
    """Estimate effective diffusivity from MW: D = 6e-7 / sqrt(MW) cm^2/s."""
    return 6e-7 / math.sqrt(mw_Da)


def _classify_penetration(penetration_ratio: float) -> str:
    if penetration_ratio > 0.5:
        return "excellent"
    if penetration_ratio >= 0.2:
        return "moderate"
    return "poor"


def _build_notes(
    mw_Da: float,
    logP: float,
    protein_binding_pct: float,
    diffusivity: float,
    penetration_ratio: float,
) -> str:
    parts: list[str] = []
    if mw_Da > 500:
        parts.append("high MW (>500 Da): reduced diffusion through interstitium")
    if logP > 3.0:
        parts.append("high logP: elevated tumor Kp due to lipophilicity")
    if logP < 0.0:
        parts.append("negative logP: hydrophilic drug, Kp_tumor near 1")
    if protein_binding_pct > 90.0:
        parts.append("high protein binding (>90%): low free drug available")
    if penetration_ratio < 0.2:
        parts.append("poor penetration: may not reach hypoxic/necrotic core")
    elif penetration_ratio > 0.5:
        parts.append("excellent penetration: drug reaches necrotic core effectively")
    parts.append(f"effective diffusivity: {diffusivity:.2e} cm^2/s")
    return "; ".join(parts)


def predict_tumor_drug_diffusion(
    drug_name: str,
    plasma_conc_mg_L: float,
    mw_Da: float,
    logP: float,
    protein_binding_pct: float,
    tumor_radius_um: float = 200.0,
    diffusivity_cm2_per_s: float | None = None,
    interstitial_pH: float = 6.8,
    charge_at_ph7: int = 0,
) -> TumorDrugDiffusionResult:
    """Predict drug diffusion through tumor interstitium (3-zone shell model).

    Zones:
        Zone 1: perivascular (0-100 um from vessel) — well perfused
        Zone 2: intermediate (100-200 um) — diffusion limited
        Zone 3: hypoxic/necrotic (>200 um) — poor penetration

    Parameters
    ----------
    drug_name : str
    plasma_conc_mg_L : float
    mw_Da : float
    logP : float
    protein_binding_pct : float  0-100
    tumor_radius_um : float  default 200 um
    diffusivity_cm2_per_s : float | None  estimated from MW if None
    interstitial_pH : float  default 6.8
    charge_at_ph7 : int  charge state at pH 7
    """
    if plasma_conc_mg_L <= 0.0:
        raise ValueError("plasma_conc_mg_L must be > 0")
    if mw_Da <= 0.0:
        raise ValueError("mw_Da must be > 0")
    if not (0.0 <= protein_binding_pct <= 100.0):
        raise ValueError("protein_binding_pct must be in [0, 100]")
    if tumor_radius_um <= 0.0:
        raise ValueError("tumor_radius_um must be > 0")

    # Unbound fraction
    fu = 1.0 - protein_binding_pct / 100.0

    # Effective diffusivity
    if diffusivity_cm2_per_s is None:
        D = _estimate_diffusivity(mw_Da)
    else:
        D = diffusivity_cm2_per_s

    # Tumor partition coefficient: higher logP → more uptake
    kp_tumor = 1.0 + 0.5 * max(0.0, logP)

    # Free plasma concentration
    c_plasma_free = plasma_conc_mg_L * fu

    # Zone 1: perivascular — direct perfusion from free plasma
    c_zone1 = c_plasma_free * kp_tumor

    # Zone 2 & 3: exponential decay across each 100 um zone.
    # Decay factor: exp(-k_diff * D_ref / D) — higher D relative to D_ref means
    # less resistance (exponent closer to 0 → decay_factor closer to 1).
    # With D_ref=1e-6 and k_diff=0.1: exponent = -0.1 * 1e-6 / D
    # For D=6e-7 (small MW): exp(-0.167) ≈ 0.846; for D=2e-8 (large MW): exp(-5) ≈ 0.007
    decay_factor = math.exp(-_K_DIFF * _D_REFERENCE / D)

    c_zone2 = c_zone1 * decay_factor
    c_zone3 = c_zone2 * decay_factor

    # Penetration efficiency: zone3 / zone1
    penetration_ratio = c_zone3 / c_zone1 if c_zone1 > 0.0 else 0.0

    # Tumor-to-plasma ratio: zone1 relative to free plasma
    tumor_to_plasma_ratio = c_zone1 / c_plasma_free if c_plasma_free > 0.0 else 0.0

    penetration_class = _classify_penetration(penetration_ratio)
    notes = _build_notes(mw_Da, logP, protein_binding_pct, D, penetration_ratio)

    return TumorDrugDiffusionResult(
        drug_name=drug_name,
        plasma_conc_mg_L=plasma_conc_mg_L,
        mw_Da=mw_Da,
        logP=logP,
        effective_diffusivity_cm2_per_s=D,
        fu_plasma=fu,
        kp_tumor=kp_tumor,
        c_zone1_mg_L=c_zone1,
        c_zone2_mg_L=c_zone2,
        c_zone3_mg_L=c_zone3,
        penetration_ratio_z3_z1=penetration_ratio,
        tumor_to_plasma_ratio=tumor_to_plasma_ratio,
        penetration_class=penetration_class,
        notes=notes,
    )


def compare_drugs(drug_list: list[dict]) -> list[TumorDrugDiffusionResult]:
    """Compare multiple drugs by tumor penetration ratio.

    Parameters
    ----------
    drug_list : list[dict]  each dict contains kwargs for predict_tumor_drug_diffusion

    Returns
    -------
    list[TumorDrugDiffusionResult]  sorted by penetration_ratio_z3_z1 descending
    """
    results = [predict_tumor_drug_diffusion(**params) for params in drug_list]
    return sorted(results, key=lambda r: r.penetration_ratio_z3_z1, reverse=True)
