"""Plasma protein binding prediction from physicochemical properties.

Phase 718 adds: ProteinBindingResult, predict_albumin_binding, predict_agp_binding,
predict_total_fup, screen_protein_binding — albumin/AGP binding with mechanistic Kd.


Plasma proteins that bind drugs:
- Albumin: primary binder for acidic and neutral drugs (Kd 10⁻⁵–10⁻⁷ M).
- Alpha-1-acid glycoprotein (AGP): primary binder for basic drugs.

The empirical model estimates the free (unbound) fraction fu_plasma from:
    logP (lipophilicity surrogate), PSA (polarity), logD at pH 7.4 (ionisation).

The model also predicts the effect of disease states (hypoalbuminemia, renal
failure, hepatic failure) on protein binding.

References:
    Trainor (2007) Expert Opin Drug Discov; Bohnert & Gan (2013) J Med Chem;
    Rowland & Tozer Clinical Pharmacokinetics (2011).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_MOLECULE_TYPES = {"acid", "base", "neutral", "zwitterion"}
_VALID_DISEASES = {"hypoalbuminemia", "renal_failure", "hepatic_failure"}

# Disease effect multipliers on fu
_DISEASE_FU_MULTIPLIER: dict[str, float] = {
    "hypoalbuminemia": 2.0,  # reduced albumin → less binding → higher fu
    "renal_failure": 1.5,  # uremic compounds displace drugs
    "hepatic_failure": 1.8,  # reduced albumin synthesis
}


# ---------------------------------------------------------------------------
# Result dataclass (frozen — no mutable fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProteinBindingPredResult:
    """Predicted plasma protein binding parameters."""

    mw: float
    logP: float
    psa: float
    pka: float
    molecule_type: str
    logD_pH74: float
    fu_plasma: float
    primary_protein: str
    protein_binding_pct: float
    binding_class: str  # "low", "moderate", "high"
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _binding_class(fu: float) -> str:
    """Classify binding level from fu_plasma."""
    pct = (1.0 - fu) * 100.0
    if pct < 50.0:
        return "low"
    elif pct <= 90.0:
        return "moderate"
    else:
        return "high"


def _compute_fu(logP: float, psa: float, logD_pH74: float) -> float:
    """Compute combined fu estimate from three empirical sub-models."""
    # Base fu from logP: higher lipophilicity → more binding → lower fu
    fu_base = 1.0 / (1.0 + 10 ** (logP * 0.5))

    # PSA correction: polar surface area reduces hydrophobic binding
    fu_psa = fu_base * (1.0 + psa / 200.0)

    # logD correction: lower logD at pH 7.4 means more ionised → less binding
    fu_logD = fu_base * (1.0 + max(0.0, 2.0 - logD_pH74) / 5.0)

    # Combined: arithmetic mean of three estimates, clamped to [0.001, 1.0]
    fu_combined = (fu_base + fu_psa + fu_logD) / 3.0
    return max(0.001, min(1.0, fu_combined))


def _validate_inputs(
    mw: float,
    logP: float,
    psa: float,
    pka: float,
    molecule_type: str,
    logD_pH74: float,
) -> None:
    """Validate physicochemical inputs."""
    if mw <= 0:
        raise ValueError(f"mw must be positive; got {mw}.")
    if psa < 0:
        raise ValueError(f"psa must be non-negative; got {psa}.")
    mol_type = molecule_type.lower()
    if mol_type not in _VALID_MOLECULE_TYPES:
        raise ValueError(
            f"molecule_type must be one of {_VALID_MOLECULE_TYPES}; got '{molecule_type}'."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_protein_binding(
    mw: float,
    logP: float,
    psa: float,
    pka: float,
    molecule_type: str,
    logD_pH74: float,
) -> ProteinBindingPredResult:
    """Predict plasma protein binding from physicochemical descriptors.

    Args:
        mw: Molecular weight (Da).
        logP: Octanol-water partition coefficient (log units).
        psa: Polar surface area (Å²).
        pka: Most acidic or basic pKa (relevant ionisation constant).
        molecule_type: One of "acid", "base", "neutral", "zwitterion".
        logD_pH74: Distribution coefficient at physiological pH 7.4.

    Returns:
        ProteinBindingPredResult with fu_plasma, primary protein, binding class.

    Raises:
        ValueError: On invalid inputs.
    """
    _validate_inputs(mw, logP, psa, pka, molecule_type, logD_pH74)

    mol_type = molecule_type.lower()
    fu_base = _compute_fu(logP, psa, logD_pH74)

    # Bases bind preferentially to AGP → apply additional factor
    if mol_type == "base":
        fu_plasma = max(0.001, min(1.0, fu_base * 0.8))
        primary_protein = "AGP"
    else:
        fu_plasma = fu_base
        primary_protein = "Albumin"

    protein_binding_pct = (1.0 - fu_plasma) * 100.0
    b_class = _binding_class(fu_plasma)

    notes = (
        f"Empirical model: fu_base={fu_base:.3f} from logP={logP:.2f}. "
        f"Primary protein: {primary_protein}. "
        f"Binding class: {b_class} ({protein_binding_pct:.1f}% bound)."
    )

    return ProteinBindingPredResult(
        mw=mw,
        logP=logP,
        psa=psa,
        pka=pka,
        molecule_type=mol_type,
        logD_pH74=logD_pH74,
        fu_plasma=fu_plasma,
        primary_protein=primary_protein,
        protein_binding_pct=protein_binding_pct,
        binding_class=b_class,
        notes=notes,
    )


def disease_effect_on_binding(
    mw: float,
    logP: float,
    psa: float,
    pka: float,
    molecule_type: str,
    logD_pH74: float,
    disease: str,
) -> dict[str, float]:
    """Estimate how a disease state alters plasma protein binding.

    Args:
        mw: Molecular weight (Da).
        logP: Octanol-water partition coefficient.
        psa: Polar surface area (Å²).
        pka: Ionisation constant.
        molecule_type: One of "acid", "base", "neutral", "zwitterion".
        logD_pH74: Distribution coefficient at pH 7.4.
        disease: One of "hypoalbuminemia", "renal_failure", "hepatic_failure".

    Returns:
        dict with keys: normal_fu, disease_fu, fu_ratio.

    Raises:
        ValueError: On invalid inputs or unrecognised disease.
    """
    _validate_inputs(mw, logP, psa, pka, molecule_type, logD_pH74)
    disease_key = disease.lower()
    if disease_key not in _VALID_DISEASES:
        raise ValueError(f"disease must be one of {_VALID_DISEASES}; got '{disease}'.")

    normal_result = predict_protein_binding(mw, logP, psa, pka, molecule_type, logD_pH74)
    normal_fu = normal_result.fu_plasma

    multiplier = _DISEASE_FU_MULTIPLIER[disease_key]
    disease_fu = max(0.001, min(1.0, normal_fu * multiplier))
    fu_ratio = disease_fu / normal_fu if normal_fu > 0 else float("nan")

    return {
        "normal_fu": normal_fu,
        "disease_fu": disease_fu,
        "fu_ratio": fu_ratio,
    }


# ---------------------------------------------------------------------------
# Phase 718 — Albumin/AGP mechanistic binding prediction
# ---------------------------------------------------------------------------

__all__ = [
    "ProteinBindingResult",
    "predict_albumin_binding",
    "predict_agp_binding",
    "predict_total_fup",
    "screen_protein_binding",
]

_ALBUMIN_BMAX_UM = 600.0  # typical plasma albumin ~42 g/L = ~630 µM
_AGP_CONC_UM = 25.0  # ~1 g/L AGP at MW 41 kDa ≈ 24 µM


@dataclass(frozen=True)
class ProteinBindingResult:
    """Comprehensive plasma protein binding prediction (Phase 718).

    Attributes
    ----------
    compound_name : Compound identifier.
    logP : Lipophilicity.
    mw : Molecular weight (Da).
    psa : Polar surface area (Å²).
    fub_albumin : Fraction unbound to albumin (0–1).
    fub_agp : Fraction unbound to AGP (0–1).
    fup_total : Combined fraction unbound in plasma (0–1).
    protein_binding_pct : Percent protein bound ((1-fup_total)*100).
    binding_class : 'low', 'moderate', 'high', or 'very_high'.
    drug_drug_interaction_risk : 'low', 'moderate', or 'high'.
    notes : Summary text.
    """

    compound_name: str
    logP: float
    mw: float
    psa: float
    fub_albumin: float
    fub_agp: float
    fup_total: float
    protein_binding_pct: float
    binding_class: str
    drug_drug_interaction_risk: str
    notes: str


def predict_albumin_binding(
    logP: float,
    mw: float,
    psa: float,
    n_h_donors: int = 1,
    charge_at_74: int = 0,
) -> dict:
    """Predict albumin (HSA) binding from physicochemical descriptors.

    Lipophilic, large, low-PSA drugs bind more to albumin.
    Hydrogen bond donors slightly reduce binding.
    Acidic drugs (charge_at_74=-1) have enhanced Site I/II binding.

    Parameters
    ----------
    logP : Lipophilicity.
    mw : Molecular weight (Da).
    psa : Polar surface area (Å²).
    n_h_donors : Number of H-bond donors.
    charge_at_74 : Formal charge at pH 7.4 (-1=acid, 0=neutral, +1=base).

    Returns
    -------
    dict with fub_albumin, kd_uM, bound_fraction, notes.
    """
    # logit of fraction unbound: more lipophilic / larger / less polar → more bound
    logit_fub = -(0.5 * logP + 0.01 * mw - 0.01 * psa - 0.3 * n_h_donors)

    # Acidic drugs have enhanced albumin Site I/II binding
    if charge_at_74 == -1:
        logit_fub -= 0.5

    fub = 1.0 / (1.0 + math.exp(-logit_fub))
    fub_albumin = max(0.01, min(0.99, fub))
    bound_fraction = 1.0 - fub_albumin

    # Kd: fub = 1 / (1 + Bmax/Kd) → Kd = Bmax * fub / (1 - fub)
    if bound_fraction > 0:
        kd_uM = _ALBUMIN_BMAX_UM * fub_albumin / bound_fraction
    else:
        kd_uM = 1e6

    kd_uM = min(kd_uM, 1e6)

    notes = (
        f"HSA binding: logP={logP}, MW={mw:.1f}, PSA={psa:.1f}, "
        f"H-donors={n_h_donors}, charge={charge_at_74}. "
        f"fub_albumin={fub_albumin:.3f}, Kd={kd_uM:.1f} µM."
    )

    return {
        "fub_albumin": fub_albumin,
        "kd_uM": kd_uM,
        "bound_fraction": bound_fraction,
        "notes": notes,
    }


def predict_agp_binding(
    logP: float,
    pka_base: float | None = None,
    charge_at_74: int = 0,
) -> dict:
    """Predict AGP (alpha-1-acid glycoprotein) binding.

    Basic drugs bind significantly to AGP. Lipophilicity drives
    non-specific hydrophobic interaction.

    Parameters
    ----------
    logP : Lipophilicity.
    pka_base : Basic pKa if applicable (None for neutral/acidic drugs).
    charge_at_74 : Formal charge at pH 7.4.

    Returns
    -------
    dict with fub_agp, kd_agp_uM, notes.
    """
    logp_excess = max(0.0, logP - 2.0)
    fub_agp_base = 1.0 / (1.0 + logp_excess * 0.5)

    # Basic drugs (pKa > 7) bind ~2x more to AGP
    if pka_base is not None and pka_base > 7.0:
        fub_agp = fub_agp_base * 0.5
    else:
        fub_agp = fub_agp_base

    fub_agp = max(0.01, min(0.99, fub_agp))
    bound_agp = 1.0 - fub_agp

    if bound_agp > 0:
        kd_agp_uM = _AGP_CONC_UM * fub_agp / bound_agp
    else:
        kd_agp_uM = 1e6

    kd_agp_uM = min(kd_agp_uM, 1e6)

    notes = (
        f"AGP binding: logP={logP}, pKa_base={pka_base}, "
        f"fub_agp={fub_agp:.3f}, Kd={kd_agp_uM:.2f} µM."
    )

    return {
        "fub_agp": fub_agp,
        "kd_agp_uM": kd_agp_uM,
        "notes": notes,
    }


def _p718_binding_class(fup: float) -> str:
    """Classify binding level by fup (Phase 718 version)."""
    if fup > 0.5:
        return "low"
    if fup >= 0.1:
        return "moderate"
    if fup >= 0.01:
        return "high"
    return "very_high"


def _p718_ddi_risk(pb_pct: float) -> str:
    """DDI risk from protein binding percent."""
    if pb_pct >= 95.0:
        return "high"
    if pb_pct >= 80.0:
        return "moderate"
    return "low"


def predict_total_fup(
    logP: float,
    mw: float,
    psa: float,
    pka_base: float | None = None,
    pka_acid: float | None = None,
    n_h_donors: int = 1,
    compound_name: str = "compound",
) -> ProteinBindingResult:
    """Predict combined plasma protein binding (albumin + AGP).

    Parameters
    ----------
    logP : Lipophilicity.
    mw : Molecular weight (Da). Must be > 0.
    psa : Polar surface area (Å²). Must be >= 0.
    pka_base : Basic pKa (if applicable).
    pka_acid : Acidic pKa (if applicable).
    n_h_donors : Number of H-bond donors.
    compound_name : Name/identifier for the compound.

    Returns
    -------
    ProteinBindingResult
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")

    # Determine charge character
    if pka_acid is not None and (pka_base is None or pka_acid < pka_base):
        charge_at_74 = -1
    elif pka_base is not None and pka_base > 7.0:
        charge_at_74 = 1
    else:
        charge_at_74 = 0

    alb = predict_albumin_binding(
        logP=logP,
        mw=mw,
        psa=psa,
        n_h_donors=n_h_donors,
        charge_at_74=charge_at_74,
    )
    agp = predict_agp_binding(
        logP=logP,
        pka_base=pka_base,
        charge_at_74=charge_at_74,
    )

    fub_albumin = alb["fub_albumin"]
    fub_agp = agp["fub_agp"]

    # Combined fup: independent parallel binding with offset floor
    fup_combined = max(0.01, min(1.0, fub_albumin * fub_agp + 0.05))

    pb_pct = (1.0 - fup_combined) * 100.0
    cls = _p718_binding_class(fup_combined)
    ddi = _p718_ddi_risk(pb_pct)

    notes = (
        f"{compound_name}: logP={logP}, MW={mw:.1f}, PSA={psa:.1f}. "
        f"fub_albumin={fub_albumin:.3f}, fub_agp={fub_agp:.3f}, "
        f"fup_total={fup_combined:.3f} ({pb_pct:.1f}% bound). "
        f"Binding class: {cls}. DDI risk: {ddi}."
    )

    return ProteinBindingResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        psa=psa,
        fub_albumin=fub_albumin,
        fub_agp=fub_agp,
        fup_total=fup_combined,
        protein_binding_pct=pb_pct,
        binding_class=cls,
        drug_drug_interaction_risk=ddi,
        notes=notes,
    )


def screen_protein_binding(compounds: list[dict]) -> list[ProteinBindingResult]:
    """Screen multiple compounds for protein binding.

    Each compound dict must have keys: logP, mw, psa.
    Optional keys: pka_base, pka_acid, n_h_donors, compound_name.

    Results are sorted ascending by fup_total (most bound first).

    Parameters
    ----------
    compounds : List of compound parameter dicts.

    Returns
    -------
    list[ProteinBindingResult] sorted by fup_total ascending.
    """
    results = []
    for i, comp in enumerate(compounds):
        name = comp.get("compound_name", f"compound_{i + 1}")
        result = predict_total_fup(
            logP=comp["logP"],
            mw=comp["mw"],
            psa=comp["psa"],
            pka_base=comp.get("pka_base"),
            pka_acid=comp.get("pka_acid"),
            n_h_donors=comp.get("n_h_donors", 1),
            compound_name=name,
        )
        results.append(result)

    results.sort(key=lambda r: r.fup_total)
    return results
