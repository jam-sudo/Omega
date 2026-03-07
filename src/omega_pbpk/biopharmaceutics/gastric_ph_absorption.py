"""Gastric pH Effect on Drug Absorption.

Phase 364: Model how gastric pH (fasted/fed, PPI use, achlorhydria) affects
ionization and dissolution of weak acids/bases.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_SCENARIOS = {"fasted", "fed", "ppi", "achlorhydria"}
VALID_PKA_TYPES = {"acid", "base", "neutral"}

_SCENARIO_PH: dict[str, dict[str, float]] = {
    "fasted": {"gastric_ph": 1.5, "intestinal_ph": 6.5},
    "fed": {"gastric_ph": 4.0, "intestinal_ph": 6.8},
    "ppi": {"gastric_ph": 6.5, "intestinal_ph": 6.5},
    "achlorhydria": {"gastric_ph": 7.0, "intestinal_ph": 6.5},
}


@dataclass(frozen=True)
class GastricPHResult:
    """Result of gastric pH absorption simulation."""

    drug_name: str
    pka_type: str
    pka_value: float
    gastric_ph: float
    intestinal_ph: float

    # Ionization state
    fraction_ionized_stomach: float
    fraction_unionized_stomach: float
    fraction_ionized_intestine: float
    fraction_unionized_intestine: float

    # Solubility
    intrinsic_solubility_mg_mL: float
    solubility_stomach_mg_mL: float
    solubility_intestine_mg_mL: float

    # Absorption impact
    fa_stomach: float
    fa_intestine: float
    fa_total: float

    # pH scenario
    scenario: str
    ph_effect_direction: str
    comparison_ref_fa: float
    ph_effect_magnitude_pct: float

    notes: str


def calculate_ph_dependent_solubility(
    pka_type: str,
    pka_value: float,
    intrinsic_solubility_mg_mL: float,
    ph: float,
) -> float:
    """Calculate pH-dependent solubility using Henderson-Hasselbalch.

    Parameters
    ----------
    pka_type:
        "acid", "base", or "neutral".
    pka_value:
        pKa value of the drug.
    intrinsic_solubility_mg_mL:
        Intrinsic (unionized) solubility in mg/mL.
    ph:
        pH at which to calculate solubility.

    Returns
    -------
    float
        pH-dependent solubility in mg/mL.
    """
    if pka_type not in VALID_PKA_TYPES:
        raise ValueError(f"pka_type must be one of {VALID_PKA_TYPES}")
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError("intrinsic_solubility_mg_mL must be positive")

    if pka_type == "acid":
        return intrinsic_solubility_mg_mL * (1.0 + 10.0 ** (ph - pka_value))
    if pka_type == "base":
        return intrinsic_solubility_mg_mL * (1.0 + 10.0 ** (pka_value - ph))
    # neutral
    return intrinsic_solubility_mg_mL


def calculate_fraction_ionized(
    pka_type: str,
    pka_value: float,
    ph: float,
) -> float:
    """Calculate fraction ionized at a given pH.

    Parameters
    ----------
    pka_type:
        "acid", "base", or "neutral".
    pka_value:
        pKa value.
    ph:
        pH at which to calculate ionization.

    Returns
    -------
    float
        Fraction ionized (0 to 1).
    """
    if pka_type not in VALID_PKA_TYPES:
        raise ValueError(f"pka_type must be one of {VALID_PKA_TYPES}")

    if pka_type == "acid":
        return 1.0 / (1.0 + 10.0 ** (pka_value - ph))
    if pka_type == "base":
        return 1.0 / (1.0 + 10.0 ** (ph - pka_value))
    # neutral
    return 0.0


def _compute_fa(
    pka_type: str,
    pka_value: float,
    intrinsic_solubility_mg_mL: float,
    dose_mg: float,
    peff_cm_per_s: float,
    gastric_ph: float,
    intestinal_ph: float,
    stomach_volume_mL: float,
    intestinal_volume_mL: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute absorption fractions; returns (fa_stomach, fa_intestine, fa_total,
    fi_stomach, fu_stomach, fi_intestine, fu_intestine, s_stomach, s_intestine)."""
    s_stomach = calculate_ph_dependent_solubility(
        pka_type, pka_value, intrinsic_solubility_mg_mL, gastric_ph
    )
    s_intestine = calculate_ph_dependent_solubility(
        pka_type, pka_value, intrinsic_solubility_mg_mL, intestinal_ph
    )

    fi_stomach = calculate_fraction_ionized(pka_type, pka_value, gastric_ph)
    fu_stomach = 1.0 - fi_stomach
    fi_intestine = calculate_fraction_ionized(pka_type, pka_value, intestinal_ph)
    fu_intestine = 1.0 - fi_intestine

    # Stomach dissolution
    dissolved_stomach = min(dose_mg, s_stomach * stomach_volume_mL)
    # Stomach absorption: ~10% contribution, capped at 30%
    fa_stomach = min(0.3, (dissolved_stomach / dose_mg) * 0.1)

    # Intestine: remaining dose
    remaining = dose_mg * (1.0 - fa_stomach)
    dissolved_intestine = min(remaining, s_intestine * intestinal_volume_mL)

    # Absorption efficiency: Peff scaled (Peff / 1e-4 as reference)
    peff_factor = min(1.0, peff_cm_per_s / 1e-4)
    fa_intestine = dissolved_intestine / dose_mg * peff_factor
    fa_total = min(1.0, fa_stomach + fa_intestine)

    return (
        fa_stomach,
        fa_intestine,
        fa_total,
        fi_stomach,
        fu_stomach,
        fi_intestine,
        fu_intestine,
        s_stomach,
        s_intestine,
    )


def simulate_gastric_ph_absorption(
    drug_name: str,
    pka_type: str,
    pka_value: float,
    intrinsic_solubility_mg_mL: float,
    dose_mg: float,
    peff_cm_per_s: float,
    scenario: str = "fasted",
    stomach_volume_mL: float = 250.0,
    dose_volume_mL: float = 250.0,
    intestinal_volume_mL: float = 500.0,
) -> GastricPHResult:
    """Simulate gastric pH effect on drug absorption.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    pka_type:
        "acid", "base", or "neutral".
    pka_value:
        pKa value.
    intrinsic_solubility_mg_mL:
        Intrinsic (unionized) solubility (mg/mL).
    dose_mg:
        Administered dose (mg).
    peff_cm_per_s:
        Effective permeability (cm/s).
    scenario:
        pH scenario: "fasted", "fed", "ppi", or "achlorhydria".
    stomach_volume_mL:
        Gastric fluid volume (mL).
    dose_volume_mL:
        Volume of water taken with dose (mL); included in stomach volume.
    intestinal_volume_mL:
        Intestinal fluid volume (mL).

    Returns
    -------
    GastricPHResult
    """
    # Input validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if peff_cm_per_s <= 0:
        raise ValueError("peff_cm_per_s must be positive")
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError("intrinsic_solubility_mg_mL must be positive")
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"scenario must be one of {VALID_SCENARIOS}")
    if pka_type not in VALID_PKA_TYPES:
        raise ValueError(f"pka_type must be one of {VALID_PKA_TYPES}")

    ph_params = _SCENARIO_PH[scenario]
    gastric_ph = ph_params["gastric_ph"]
    intestinal_ph = ph_params["intestinal_ph"]

    # Effective stomach volume (includes dose volume)
    effective_stomach_vol = stomach_volume_mL + dose_volume_mL

    (
        fa_stomach,
        fa_intestine,
        fa_total,
        fi_stomach,
        fu_stomach,
        fi_intestine,
        fu_intestine,
        s_stomach,
        s_intestine,
    ) = _compute_fa(
        pka_type,
        pka_value,
        intrinsic_solubility_mg_mL,
        dose_mg,
        peff_cm_per_s,
        gastric_ph,
        intestinal_ph,
        effective_stomach_vol,
        intestinal_volume_mL,
    )

    # Reference: fasted scenario
    ref_ph = _SCENARIO_PH["fasted"]
    (
        _,
        _,
        fa_ref,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _compute_fa(
        pka_type,
        pka_value,
        intrinsic_solubility_mg_mL,
        dose_mg,
        peff_cm_per_s,
        ref_ph["gastric_ph"],
        ref_ph["intestinal_ph"],
        effective_stomach_vol,
        intestinal_volume_mL,
    )

    ph_effect_magnitude_pct = (fa_total - fa_ref) / max(0.01, fa_ref) * 100.0
    if ph_effect_magnitude_pct > 10.0:
        ph_effect_direction = "increases_absorption"
    elif ph_effect_magnitude_pct < -10.0:
        ph_effect_direction = "decreases_absorption"
    else:
        ph_effect_direction = "minimal"

    notes_parts = [f"Scenario: {scenario} (gastric pH={gastric_ph}, intestinal pH={intestinal_ph})"]
    if pka_type == "neutral":
        notes_parts.append("Neutral drug: pH has no ionization effect")
    elif pka_type == "acid" and scenario in ("ppi", "achlorhydria"):
        notes_parts.append("High gastric pH reduces acid dissolution/absorption")
    elif pka_type == "base" and scenario in ("ppi", "achlorhydria"):
        notes_parts.append("High gastric pH improves base dissolution")
    notes = "; ".join(notes_parts)

    return GastricPHResult(
        drug_name=drug_name,
        pka_type=pka_type,
        pka_value=pka_value,
        gastric_ph=gastric_ph,
        intestinal_ph=intestinal_ph,
        fraction_ionized_stomach=fi_stomach,
        fraction_unionized_stomach=fu_stomach,
        fraction_ionized_intestine=fi_intestine,
        fraction_unionized_intestine=fu_intestine,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        solubility_stomach_mg_mL=s_stomach,
        solubility_intestine_mg_mL=s_intestine,
        fa_stomach=fa_stomach,
        fa_intestine=fa_intestine,
        fa_total=fa_total,
        scenario=scenario,
        ph_effect_direction=ph_effect_direction,
        comparison_ref_fa=fa_ref,
        ph_effect_magnitude_pct=ph_effect_magnitude_pct,
        notes=notes,
    )


def compare_ph_scenarios(
    drug_name: str,
    pka_type: str,
    pka_value: float,
    intrinsic_solubility_mg_mL: float,
    dose_mg: float,
    peff_cm_per_s: float,
) -> dict:
    """Compare absorption across all four pH scenarios.

    Returns
    -------
    dict
        Keys are scenario names ("fasted", "fed", "ppi", "achlorhydria").
        Values are GastricPHResult objects. Also includes "highest_fa_scenario"
        and "lowest_fa_scenario" keys.
    """
    results = {}
    for scenario in VALID_SCENARIOS:
        results[scenario] = simulate_gastric_ph_absorption(
            drug_name=drug_name,
            pka_type=pka_type,
            pka_value=pka_value,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
            dose_mg=dose_mg,
            peff_cm_per_s=peff_cm_per_s,
            scenario=scenario,
        )

    fa_values = {s: results[s].fa_total for s in results}
    results["highest_fa_scenario"] = max(fa_values, key=lambda s: fa_values[s])
    results["lowest_fa_scenario"] = min(fa_values, key=lambda s: fa_values[s])
    return results
