"""Plasma protein binding prediction from physicochemical properties.

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
