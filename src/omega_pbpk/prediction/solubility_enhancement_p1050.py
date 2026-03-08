"""Phase 1050 — Drug Solubility Enhancement Techniques prediction."""

from dataclasses import dataclass

_VALID_TECHNIQUES = {
    "amorphous_dispersion",
    "cyclodextrin_complexation",
    "particle_size_reduction",
    "salt_form",
    "cosolvency",
}


@dataclass(frozen=True)
class SolubilityEnhancementResult:
    drug_name: str
    technique: str
    baseline_solubility_mg_mL: float
    enhanced_solubility_mg_mL: float
    enhancement_fold: float
    mechanism: str
    predicted_bioavailability_gain: float
    stability_risk: str
    feasibility_score: float
    recommended: bool
    notes: str


def _baseline_solubility(logp: float, mw_Da: float) -> float:
    """Compute rough baseline solubility in mg/mL."""
    s0 = (10 ** (-0.5 * logp)) * 100.0 / mw_Da
    return max(0.001, min(100.0, s0))


def _amorphous_dispersion(
    melting_point_C: float,
    polymer_type: str,
) -> tuple:
    """Return (enhancement, mechanism, stability_risk, feasibility_score)."""
    enhancement = max(2.0, min(100.0, (melting_point_C - 100.0) / 50.0 + 2.0))

    if melting_point_C > 200.0:
        stability_risk = "low"
    elif melting_point_C > 150.0:
        stability_risk = "moderate"
    else:
        stability_risk = "high"

    polymer_params = {
        "HPMC": (75.0, 1.0),
        "PVP": (70.0, 1.1),
        "PVP-VA": (80.0, 1.05),
    }
    feasibility, polymer_mult = polymer_params.get(polymer_type, (75.0, 1.0))
    enhancement = max(2.0, min(100.0, enhancement * polymer_mult))

    return enhancement, "amorphous", stability_risk, feasibility


def _cyclodextrin_complexation(
    logp: float,
    mw_Da: float,
    cd_type: str,
) -> tuple:
    """Return (enhancement, mechanism, stability_risk, feasibility_score)."""
    cd_params = {
        "HP-beta-CD": (1.2, 0.01, 85.0),
        "beta-CD": (1.0, 0.005, 65.0),
        "SBE-CD": (1.4, 0.01, 90.0),
    }
    k_mult, cd_conc, feasibility = cd_params.get(cd_type, (1.2, 0.01, 85.0))

    k_base = 1000.0 * (10 ** (0.2 * logp)) / max(100.0, mw_Da)
    k_stability = max(50.0, min(50000.0, k_base))

    enhancement = 1.0 + k_stability * k_mult * cd_conc
    enhancement = max(1.0, min(200.0, enhancement))

    return enhancement, "complexation", "low", feasibility


def _particle_size_reduction(logp: float) -> tuple:
    """Return (enhancement, mechanism, stability_risk, feasibility_score)."""
    if logp > 1.0:
        enhancement = max(2.0, min(10.0, 100.0 / logp))
    else:
        enhancement = 5.0
    return enhancement, "particle_size", "moderate", 70.0


def _salt_form_calc(
    pka: float,
    ionization_type: str,
    salt_form: str,
) -> tuple:
    """Return (enhancement, mechanism, stability_risk, feasibility_score)."""
    salt_mult_map = {
        "HCl": 1.2,
        "Na": 1.0,
        "Ca": 0.9,
        "Mg": 0.8,
        "phosphate": 1.1,
    }
    salt_mult = salt_mult_map.get(salt_form, 1.0)

    if ionization_type == "neutral":
        enhancement = 1.5
    else:
        base_enhancement = 10 ** (abs(pka - 7.4) * 0.5)
        enhancement = base_enhancement

    enhancement *= salt_mult
    enhancement = max(1.0, enhancement)

    return enhancement, "salt_form", "low", 90.0


def _cosolvency(logp: float, cosolvent_pct: float) -> tuple:
    """Return (enhancement, mechanism, stability_risk, feasibility_score)."""
    sigma = 0.5 * logp
    enhancement = 10 ** (sigma * cosolvent_pct / 100.0)
    enhancement = max(1.0, min(100.0, enhancement))
    return enhancement, "cosolvency", "low", 80.0


def predict_solubility_enhancement(
    drug_name: str,
    mw_Da: float,
    logp: float,
    melting_point_C: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    technique: str = "amorphous_dispersion",
    polymer_type: str = "HPMC",
    cd_type: str = "HP-beta-CD",
    salt_form: str = "HCl",
    cosolvent_pct: float = 10.0,
) -> SolubilityEnhancementResult:
    """Predict solubility improvement from a pharmaceutical enhancement technique."""
    # Validation
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if technique not in _VALID_TECHNIQUES:
        raise ValueError(f"technique must be one of {_VALID_TECHNIQUES}")
    if not (0.0 <= cosolvent_pct <= 100.0):
        raise ValueError("cosolvent_pct must be in [0, 100]")
    if technique == "amorphous_dispersion" and melting_point_C <= 0:
        raise ValueError("melting_point_C must be > 0 for amorphous_dispersion")

    s0 = _baseline_solubility(logp, mw_Da)

    if technique == "amorphous_dispersion":
        enhancement, mechanism, stability_risk, feasibility = _amorphous_dispersion(
            melting_point_C, polymer_type
        )
    elif technique == "cyclodextrin_complexation":
        enhancement, mechanism, stability_risk, feasibility = _cyclodextrin_complexation(
            logp, mw_Da, cd_type
        )
    elif technique == "particle_size_reduction":
        enhancement, mechanism, stability_risk, feasibility = _particle_size_reduction(logp)
    elif technique == "salt_form":
        enhancement, mechanism, stability_risk, feasibility = _salt_form_calc(
            pka, ionization_type, salt_form
        )
    else:  # cosolvency
        enhancement, mechanism, stability_risk, feasibility = _cosolvency(logp, cosolvent_pct)

    enhanced_solubility = s0 * enhancement
    predicted_bioavailability_gain = min(0.5, (enhancement - 1.0) / (enhancement - 1.0 + 100.0))
    recommended = enhancement >= 10.0 and feasibility >= 50.0

    notes = (
        f"{drug_name}: {technique} -> {enhancement:.1f}x fold enhancement; "
        f"baseline={s0:.4f} mg/mL, enhanced={enhanced_solubility:.4f} mg/mL; "
        f"stability={stability_risk}, feasibility={feasibility:.0f}"
    )

    return SolubilityEnhancementResult(
        drug_name=drug_name,
        technique=technique,
        baseline_solubility_mg_mL=s0,
        enhanced_solubility_mg_mL=enhanced_solubility,
        enhancement_fold=enhancement,
        mechanism=mechanism,
        predicted_bioavailability_gain=predicted_bioavailability_gain,
        stability_risk=stability_risk,
        feasibility_score=feasibility,
        recommended=recommended,
        notes=notes,
    )


def compare_enhancement_techniques(
    drug_name: str,
    mw_Da: float,
    logp: float,
    melting_point_C: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
) -> list:
    """Compare all 5 enhancement techniques, sorted by enhancement_fold descending."""
    techniques = list(_VALID_TECHNIQUES)
    results = []
    for tech in techniques:
        result = predict_solubility_enhancement(
            drug_name=drug_name,
            mw_Da=mw_Da,
            logp=logp,
            melting_point_C=melting_point_C,
            pka=pka,
            ionization_type=ionization_type,
            technique=tech,
        )
        results.append(result)
    results.sort(key=lambda r: r.enhancement_fold, reverse=True)
    return results
