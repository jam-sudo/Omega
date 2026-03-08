"""Phase 1086 — Dissolution-Absorption Rate-Limiting Step Identifier.

Identifies whether dissolution or absorption is the rate-limiting step
in oral drug delivery. Critical for formulation strategy selection.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_R_INTESTINE_CM = 0.175  # intestinal radius (cm)
_DIFFUSIVITY_CM2_H = 2.16e-2  # aqueous diffusivity ~6e-6 cm²/s → cm²/h
_VOLUME_GI_ML = 250.0  # standard GI volume for dose number
_K_ABS_NORM_PEFF = 1e-4  # reference Peff for absorption number normalisation (cm/s)


# ---------------------------------------------------------------------------
# k_diss formula
# ---------------------------------------------------------------------------


def _calc_k_diss(
    solubility_mg_mL: float,
    drug_density_g_mL: float,
    particle_size_um: float,
) -> float:
    """Noyes-Whitney simplified k_diss in 1/h.

    k_diss = 3.6 * S / (rho * (d/10)^2)
    where S in mg/mL, rho in g/mL, d in um.

    Tuned so 10 um + 0.1 mg/mL → ~0.36/h; 100 um → ~0.0036/h.
    """
    return 3.6 * solubility_mg_mL / (drug_density_g_mL * (particle_size_um / 10.0) ** 2)


# ---------------------------------------------------------------------------
# k_abs formula
# ---------------------------------------------------------------------------


def _calc_k_abs(peff_cm_per_s: float) -> float:
    """k_abs in 1/h, normalised to reference Peff.

    k_abs_per_h = peff_cm_per_s / 1e-4  (dimensionless relative rate)
    """
    return peff_cm_per_s / _K_ABS_NORM_PEFF


# ---------------------------------------------------------------------------
# BCS dose number and absorption number
# ---------------------------------------------------------------------------


def _dose_number(dose_mg: float, solubility_mg_mL: float) -> float:
    """Do = dose / (solubility * 250 mL)."""
    return dose_mg / (solubility_mg_mL * _VOLUME_GI_ML)


def _absorption_number(peff_cm_per_s: float) -> float:
    """An = Peff / Peff_ref (normalised)."""
    return peff_cm_per_s / _K_ABS_NORM_PEFF


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _classify_rate_limiting(k_diss: float, k_abs: float, do: float, an: float) -> str:
    if do > 1.0 and an < 1.0:
        return "permeability"
    if do > 1.0 and an >= 1.0:
        return "dissolution"
    if do <= 1.0 and an < 1.0:
        return "absorption"
    # do <= 1 and an >= 1 → check k values
    if k_diss > 5.0 * k_abs:
        return "absorption"
    if k_abs > 5.0 * k_diss:
        return "dissolution"
    return "both"


def _bcs_class(do: float, an: float) -> str:
    if do <= 1.0 and an >= 1.0:
        return "I"
    if do > 1.0 and an >= 1.0:
        return "II"
    if do <= 1.0 and an < 1.0:
        return "III"
    return "IV"


def _f_absorbed_predicted(do: float, an: float) -> float:
    """Simplified absorption prediction: min(1, 1/(1+Do)) * min(1, An)."""
    f_solubility = min(1.0, 1.0 / (1.0 + do))
    f_perm = min(1.0, an)
    return f_solubility * f_perm


def _formulation_strategy(rate_limiting: str, bcs_class: str) -> str:
    strategies = {
        "dissolution": (
            "Particle size reduction (micronisation/nanosizing), amorphous solid dispersion, "
            "or solubilisation (surfactants, cyclodextrins) to improve dissolution rate."
        ),
        "absorption": (
            "Permeability enhancement (penetration enhancers, prodrug design, lipid-based "
            "formulations) to improve membrane transport."
        ),
        "permeability": (
            "Permeability enhancement is primary; solubility improvement secondary. "
            "Consider prodrug or lipid formulation strategies."
        ),
        "both": (
            "Dual strategy required: improve both solubility/dissolution rate and "
            "membrane permeability (e.g., self-emulsifying drug delivery system)."
        ),
    }
    return strategies.get(rate_limiting, "Standard oral formulation.")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimitingStepResult:
    drug_name: str
    dose_mg: float
    dose_number: float
    absorption_number: float
    k_diss_per_h: float
    k_abs_per_h: float
    rate_limiting_step: str
    bcs_class_predicted: str
    f_absorbed_predicted: float
    formulation_strategy: str
    notes: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def identify_rate_limiting_step(
    drug_name: str,
    dose_mg: float,
    particle_size_um: float = 100.0,
    solubility_mg_mL: float = 0.1,
    peff_cm_per_s: float = 1e-4,
    drug_density_g_mL: float = 1.2,
) -> RateLimitingStepResult:
    """Identify the rate-limiting step for oral absorption.

    Parameters
    ----------
    drug_name:
        Label for the compound.
    dose_mg:
        Administered dose in mg.
    particle_size_um:
        Mean particle diameter in micrometres.
    solubility_mg_mL:
        Aqueous solubility in mg/mL.
    peff_cm_per_s:
        Effective intestinal permeability in cm/s.
    drug_density_g_mL:
        Apparent drug particle density in g/mL.

    Returns
    -------
    RateLimitingStepResult
    """
    # --- Validation -------------------------------------------------------
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if particle_size_um <= 0.0:
        raise ValueError(f"particle_size_um must be > 0, got {particle_size_um}")
    if solubility_mg_mL <= 0.0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")
    if peff_cm_per_s <= 0.0:
        raise ValueError(f"peff_cm_per_s must be > 0, got {peff_cm_per_s}")

    # --- Calculations -----------------------------------------------------
    k_diss = _calc_k_diss(solubility_mg_mL, drug_density_g_mL, particle_size_um)
    k_abs = _calc_k_abs(peff_cm_per_s)
    do = _dose_number(dose_mg, solubility_mg_mL)
    an = _absorption_number(peff_cm_per_s)

    rls = _classify_rate_limiting(k_diss, k_abs, do, an)
    bcs = _bcs_class(do, an)
    f_abs = _f_absorbed_predicted(do, an)
    strategy = _formulation_strategy(rls, bcs)

    notes = (
        f"Do={do:.3f}, An={an:.3f}, k_diss={k_diss:.4f}/h, k_abs={k_abs:.4f}/h. "
        f"BCS class {bcs}; rate-limiting: {rls}."
    )

    return RateLimitingStepResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dose_number=round(do, 6),
        absorption_number=round(an, 6),
        k_diss_per_h=round(k_diss, 6),
        k_abs_per_h=round(k_abs, 6),
        rate_limiting_step=rls,
        bcs_class_predicted=bcs,
        f_absorbed_predicted=round(max(0.0, min(1.0, f_abs)), 6),
        formulation_strategy=strategy,
        notes=notes,
    )


def compare_formulations(
    drug_name: str,
    dose_mg: float,
    particle_sizes: list[float],
    **kwargs,
) -> list[RateLimitingStepResult]:
    """Compare multiple particle sizes for the same drug.

    Returns results sorted by f_absorbed_predicted descending.
    """
    results = [
        identify_rate_limiting_step(drug_name, dose_mg, particle_size_um=ps, **kwargs)
        for ps in particle_sizes
    ]
    results.sort(key=lambda r: r.f_absorbed_predicted, reverse=True)
    return results
