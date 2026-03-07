"""Metabolic stability prediction in liver microsomes and hepatocytes (Phase 442).

Empirical physicochemical descriptor-based model for predicting CLint and t½
in human liver microsomes (HLM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (frozen — no mutable fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicrosomalStabilityResult:
    """Result of microsomal stability prediction.

    Attributes
    ----------
    drug_name : str
        Name of the drug compound.
    mw : float
        Molecular weight (g/mol).
    logP : float
        Lipophilicity.
    n_aromatic_rings : int
        Number of aromatic rings.
    n_rot_bonds : int
        Number of rotatable bonds.
    clint_uL_per_min_per_mg : float
        Intrinsic clearance (µL/min/mg microsomal protein).
    t_half_microsomal_min : float
        Microsomal half-life (minutes) at 45 mg protein/mL incubation.
    stability_class : str
        "stable" (t½ > 60 min), "moderate" (30–60 min), or "unstable" (< 30 min).
    predicted_f_unmetabolized : float
        Fraction unmetabolized at steady state (0–1).
    notes : str
        Descriptor values and correction factors used.
    """

    drug_name: str
    mw: float
    logP: float
    n_aromatic_rings: int
    n_rot_bonds: int
    clint_uL_per_min_per_mg: float
    t_half_microsomal_min: float
    stability_class: str
    predicted_f_unmetabolized: float
    notes: str


# ---------------------------------------------------------------------------
# Model implementation
# ---------------------------------------------------------------------------

_PROTEIN_CONC_MG_PER_ML = 45.0  # mg microsomal protein / mL incubation


def _base_clint(logP: float) -> float:
    """Base CLint = 15 * exp(0.3 * logP) µL/min/mg protein."""
    return 15.0 * math.exp(0.3 * logP)


def _aromatic_ring_correction(n_aromatic_rings: int) -> float:
    """More aromatic rings → more CYP oxidation sites: factor = 1 + 0.15 * n_rings."""
    return 1.0 + 0.15 * n_aromatic_rings


def _flexibility_correction(n_rot_bonds: int) -> float:
    """Rigid molecules (fewer rot bonds) are more metabolically stable.

    factor = 1 - 0.05 * min(n_rot_bonds, 10)
    """
    return 1.0 - 0.05 * min(n_rot_bonds, 10)


def _mw_correction(mw: float) -> float:
    """Large MW molecules are less efficiently metabolized.

    factor = max(0.3, 1 - (mw - 400) / 1000)
    """
    return max(0.3, 1.0 - (mw - 400.0) / 1000.0)


def _stability_class(t_half_min: float) -> str:
    """Classify microsomal stability based on half-life."""
    if t_half_min > 60.0:
        return "stable"
    if t_half_min >= 30.0:
        return "moderate"
    return "unstable"


def predict_microsomal_stability(
    drug_name: str,
    mw: float,
    logP: float,
    n_aromatic_rings: int,
    n_rot_bonds: int,
    molecular_formula_atoms: dict[str, int] | None = None,
) -> MicrosomalStabilityResult:
    """Predict microsomal stability using an empirical physicochemical model.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    mw : float
        Molecular weight (g/mol). Must be > 0.
    logP : float
        Lipophilicity (log octanol/water partition coefficient).
    n_aromatic_rings : int
        Number of aromatic rings. Must be >= 0.
    n_rot_bonds : int
        Number of rotatable bonds. Must be >= 0.
    molecular_formula_atoms : dict[str, int] or None
        Atomic composition dict e.g. {"C": 20, "H": 25, "N": 2, "O": 3}.
        Currently stored for traceability; reserved for future heteroatom corrections.

    Returns
    -------
    MicrosomalStabilityResult
        Predicted CLint, t½, stability class, and fraction unmetabolized.
    """
    # --- Input validation ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if n_aromatic_rings < 0:
        raise ValueError(f"n_aromatic_rings must be >= 0, got {n_aromatic_rings}")
    if n_rot_bonds < 0:
        raise ValueError(f"n_rot_bonds must be >= 0, got {n_rot_bonds}")

    if molecular_formula_atoms is None:
        molecular_formula_atoms = {}

    # Validate atom counts if provided
    for elem, count in molecular_formula_atoms.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError(
                f"molecular_formula_atoms['{elem}'] must be a non-negative integer, got {count}"
            )

    # --- Empirical model ---
    clint_base = _base_clint(logP)
    f_aromatic = _aromatic_ring_correction(n_aromatic_rings)
    f_flex = _flexibility_correction(n_rot_bonds)
    f_mw = _mw_correction(mw)

    clint = clint_base * f_aromatic * f_flex * f_mw
    clint = max(clint, 0.001)  # floor to avoid division issues

    # t½ in microsomes: t½ = 0.693 / (CLint * protein_conc / 1000)
    # CLint in µL/min/mg, protein_conc in mg/mL → k_deg in min⁻¹
    k_deg_per_min = clint * _PROTEIN_CONC_MG_PER_ML / 1000.0
    t_half_min = 0.693 / k_deg_per_min

    stab_class = _stability_class(t_half_min)

    # Fraction unmetabolized at steady state (approximately exp(-CL * tau / Vd))
    # Simple surrogate: f_unmet = exp(-k_deg_per_min * 60) for a 1h incubation
    f_unmet = math.exp(-k_deg_per_min * 60.0)
    f_unmet = float(max(0.0, min(1.0, f_unmet)))

    # Build notes
    atom_str = (
        ", ".join(f"{k}:{v}" for k, v in sorted(molecular_formula_atoms.items()))
        if molecular_formula_atoms
        else "not provided"
    )
    notes = (
        f"CLint_base={clint_base:.2f}; f_aromatic={f_aromatic:.3f}; "
        f"f_flex={f_flex:.3f}; f_mw={f_mw:.3f}; atoms=[{atom_str}]"
    )

    return MicrosomalStabilityResult(
        drug_name=drug_name,
        mw=mw,
        logP=logP,
        n_aromatic_rings=n_aromatic_rings,
        n_rot_bonds=n_rot_bonds,
        clint_uL_per_min_per_mg=float(clint),
        t_half_microsomal_min=float(t_half_min),
        stability_class=stab_class,
        predicted_f_unmetabolized=f_unmet,
        notes=notes,
    )


__all__ = [
    "MicrosomalStabilityResult",
    "predict_microsomal_stability",
    "MetabolicStabilityResult",
    "classify_stability",
    "predict_primary_pathway",
    "predict_metabolic_stability",
]


# ---------------------------------------------------------------------------
# Phase 618 — predict_metabolic_stability (microsomal data → in vivo CL)
# ---------------------------------------------------------------------------

_QH_mL_per_min = 1500.0  # hepatic blood flow


@dataclass(frozen=True)
class MetabolicStabilityResult:
    """Result of Phase 618 metabolic stability prediction from measured microsomal t½."""

    drug_name: str
    t_half_microsomal_min: float  # in vitro t½ (minutes)
    clint_mL_per_min_per_mg: float  # intrinsic CL
    clint_scaled_mL_per_min: float  # in vivo scaled CLint
    hepatic_cl_L_per_h: float  # predicted in vivo hepatic CL
    extraction_ratio: float  # 0-1
    stability_class: str  # "unstable","moderate","stable","very_stable"
    primary_metabolic_pathway: str  # "CYP3A4","CYP2D6","CYP2C9","other","none"
    metabolite_toxicity_risk: bool  # reactive metabolite risk
    plasma_t_half_estimate_h: float  # estimated in vivo t½ (assuming hepatic only)
    in_vitro_to_in_vivo_factor: float  # MPPGL × liver_weight scaling
    notes: str


def classify_stability(clint_mL_per_min_per_mg: float) -> str:
    """Classify stability: unstable>100, moderate>30, stable>10, very_stable≤10 (mL/min/mg)."""
    if clint_mL_per_min_per_mg > 100:
        return "unstable"
    elif clint_mL_per_min_per_mg > 30:
        return "moderate"
    elif clint_mL_per_min_per_mg > 10:
        return "stable"
    else:
        return "very_stable"


def predict_primary_pathway(logP: float, pka_base: float | None, mw: float) -> str:
    """CYP3A4: logP>2, mw>300; CYP2D6: pka_base>7; CYP2C9: logP<2, mw<400; else other."""
    # CYP2D6 takes priority over CYP3A4
    if pka_base is not None and pka_base > 7:
        return "CYP2D6"
    if logP > 2 and mw > 300:
        return "CYP3A4"
    if logP < 2 and mw < 400:
        return "CYP2C9"
    return "other"


def predict_metabolic_stability(
    drug_name: str,
    t_half_microsomal_min: float,  # measured microsomal t½
    protein_mg_per_mL: float = 0.5,  # microsomal protein concentration
    fu_mic: float = 1.0,  # fraction unbound in microsomes
    logP: float = 2.0,
    mw: float = 400.0,
    pka_base: float | None = None,
    n_aromatic_rings: int = 0,
    vd_L: float = 50.0,
    fu_plasma: float = 0.5,
    mppgl: float = 45.0,
    liver_weight_g: float = 1500.0,
) -> MetabolicStabilityResult:
    """Predict metabolic stability and in vivo hepatic CL from microsomal data."""
    # Validation
    if t_half_microsomal_min <= 0:
        raise ValueError(f"t_half_microsomal_min must be > 0, got {t_half_microsomal_min}")
    if protein_mg_per_mL <= 0:
        raise ValueError(f"protein_mg_per_mL must be > 0, got {protein_mg_per_mL}")
    if not (0 < fu_mic <= 1.0):
        raise ValueError(f"fu_mic must be in (0, 1], got {fu_mic}")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if mppgl <= 0:
        raise ValueError(f"mppgl must be > 0, got {mppgl}")
    if liver_weight_g <= 0:
        raise ValueError(f"liver_weight_g must be > 0, got {liver_weight_g}")

    # CLint calculation (µL/min/mg protein, converted to mL/min/mg by ×1000 / 1000 = ×1)
    # Standard formula: CLint [µL/min/mg] = ln(2) / t_half [min] / protein [mg/mL] * 1000
    # Store as µL/min/mg (matching threshold scale: unstable>100, moderate>30, stable>10)
    clint_raw = math.log(2) / t_half_microsomal_min / protein_mg_per_mL * 1000.0
    # Adjust for fu_mic: only unbound drug metabolized
    clint_corrected = clint_raw / fu_mic
    # Clamp
    clint_corrected = max(0.001, min(1000.0, clint_corrected))

    # In vitro to in vivo scaling factor
    # CLint [µL/min/mg] → in vivo CLint [mL/min]:
    # CLint_scaled [mL/min] = CLint [µL/min/mg] / 1000 * mppgl [mg/g liver] * liver_weight [g]
    iviv_factor = mppgl * liver_weight_g / 1000.0  # converts µL/min/mg to mL/min per g liver

    # Scale to in vivo CLint (mL/min)
    clint_scaled = clint_corrected / 1000.0 * mppgl * liver_weight_g

    # Hepatic CL via well-stirred model (mL/min)
    qh = _QH_mL_per_min
    cl_h_mL_per_min = qh * fu_plasma * clint_scaled / (qh + fu_plasma * clint_scaled)

    # Extraction ratio
    er = cl_h_mL_per_min / qh
    er = max(0.0, min(1.0, er))

    # Convert hepatic CL to L/h
    cl_h_L_per_h = cl_h_mL_per_min * 60.0 / 1000.0

    # Stability class based on corrected CLint
    stability_class = classify_stability(clint_corrected)

    # Primary metabolic pathway
    if stability_class == "very_stable":
        primary_pathway = "none"
    else:
        primary_pathway = predict_primary_pathway(logP, pka_base, mw)

    # Metabolite toxicity risk
    metabolite_toxicity_risk = n_aromatic_rings >= 2 or logP > 4

    # Plasma t½ estimate (hours) — using hepatic CL only
    plasma_t_half_h = math.log(2) * vd_L / cl_h_L_per_h

    # Notes
    notes_parts = [
        f"CLint={clint_corrected:.2f} mL/min/mg ({stability_class}); ER={er:.2f}; pathway={primary_pathway}"
    ]
    if metabolite_toxicity_risk:
        notes_parts.append("Reactive metabolite risk: high logP or multiple aromatic rings")
    if fu_mic < 0.5:
        notes_parts.append(f"Low fu_mic ({fu_mic:.2f}): CLint corrected upward")
    notes = ". ".join(notes_parts)

    return MetabolicStabilityResult(
        drug_name=drug_name,
        t_half_microsomal_min=t_half_microsomal_min,
        clint_mL_per_min_per_mg=clint_corrected,
        clint_scaled_mL_per_min=clint_scaled,
        hepatic_cl_L_per_h=cl_h_L_per_h,
        extraction_ratio=er,
        stability_class=stability_class,
        primary_metabolic_pathway=primary_pathway,
        metabolite_toxicity_risk=metabolite_toxicity_risk,
        plasma_t_half_estimate_h=plasma_t_half_h,
        in_vitro_to_in_vivo_factor=iviv_factor,
        notes=notes,
    )
