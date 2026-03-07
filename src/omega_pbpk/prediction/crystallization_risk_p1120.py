"""Phase 1120 — Crystallization risk assessment for supersaturated drug solutions (ASD/SDD)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrystallizationRiskResult:
    drug_name: str
    logP: float
    ionization_type: str
    glass_transition_temp_C: float
    drug_loading_pct: float
    storage_rh_pct: float
    storage_temp_C: float
    tg_ratio: float  # T_storage_K / T_glass_K
    mobility_score: float
    hygroscopicity_score: float
    chemical_score: float
    crystallization_risk_score: float
    risk_class: str  # "low", "moderate", "high"
    recommended_action: str
    notes: str


_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


def _mobility_score(tg_ratio: float) -> float:
    """0–40 based on T_storage / T_glass (Kelvin ratio)."""
    if tg_ratio < 0.7:
        return 5.0
    if tg_ratio < 0.8:
        return 15.0
    if tg_ratio < 0.9:
        return 25.0
    return 40.0


def _hygroscopicity_score(storage_rh_pct: float, drug_loading_pct: float) -> float:
    """0–30 based on RH and drug loading."""
    if storage_rh_pct < 30.0:
        base = 5.0
    elif storage_rh_pct <= 60.0:
        base = 15.0
    else:
        base = 25.0

    if drug_loading_pct > 40.0:
        base += 5.0

    return base


def _chemical_score(logP: float, ionization_type: str) -> float:
    """0–30 based on molecular properties."""
    if logP > 3.0 and ionization_type == "neutral":
        return 20.0
    if ionization_type in {"acid", "base"}:
        return 5.0
    if logP < 0.0:
        return 15.0
    return 10.0


def _risk_class(score: float) -> str:
    if score < 30.0:
        return "low"
    if score < 60.0:
        return "moderate"
    return "high"


def _recommended_action(risk_class: str) -> str:
    actions = {
        "low": "Standard packaging; monitor at 25°C/60% RH",
        "moderate": "Use desiccant; consider refrigerated storage; periodic stability testing",
        "high": "Reduce drug loading or storage RH; reformulate with stabilizing polymer; store at 2-8°C",
    }
    return actions[risk_class]


def assess_crystallization_risk(
    drug_name: str,
    logP: float,
    pka: float,
    ionization_type: str,
    glass_transition_temp_C: float,
    drug_loading_pct: float,
    storage_rh_pct: float,
    storage_temp_C: float,
) -> CrystallizationRiskResult:
    """Assess crystallization risk for an amorphous solid dispersion."""
    # Validation
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(f"ionization_type must be one of {_VALID_IONIZATION_TYPES}")
    if not (20.0 <= glass_transition_temp_C <= 300.0):
        raise ValueError(
            f"glass_transition_temp_C must be in [20, 300], got {glass_transition_temp_C}"
        )
    if not (0.0 < drug_loading_pct <= 100.0):
        raise ValueError(f"drug_loading_pct must be in (0, 100], got {drug_loading_pct}")
    if not (0.0 <= storage_rh_pct <= 100.0):
        raise ValueError(f"storage_rh_pct must be in [0, 100], got {storage_rh_pct}")
    if not (-20.0 <= storage_temp_C <= 80.0):
        raise ValueError(f"storage_temp_C must be in [-20, 80], got {storage_temp_C}")

    # Convert to Kelvin
    t_storage_k = storage_temp_C + 273.15
    t_glass_k = glass_transition_temp_C + 273.15
    tg_ratio = t_storage_k / t_glass_k

    mob = _mobility_score(tg_ratio)
    hygro = _hygroscopicity_score(storage_rh_pct, drug_loading_pct)
    chem = _chemical_score(logP, ionization_type)

    total = mob + hygro + chem
    rc = _risk_class(total)
    action = _recommended_action(rc)

    notes = (
        f"Tg ratio={tg_ratio:.3f}; mobility={mob:.0f}, hygro={hygro:.0f}, "
        f"chemical={chem:.0f}; total={total:.0f} → {rc}"
    )

    return CrystallizationRiskResult(
        drug_name=drug_name,
        logP=logP,
        ionization_type=ionization_type,
        glass_transition_temp_C=glass_transition_temp_C,
        drug_loading_pct=drug_loading_pct,
        storage_rh_pct=storage_rh_pct,
        storage_temp_C=storage_temp_C,
        tg_ratio=tg_ratio,
        mobility_score=mob,
        hygroscopicity_score=hygro,
        chemical_score=chem,
        crystallization_risk_score=total,
        risk_class=rc,
        recommended_action=action,
        notes=notes,
    )


def screen_formulations(formulations: list[dict]) -> list[CrystallizationRiskResult]:
    """Screen a list of formulation dicts; returns results sorted by risk score ascending."""
    results = []
    for f in formulations:
        result = assess_crystallization_risk(
            drug_name=f["drug_name"],
            logP=f["logP"],
            pka=f.get("pka", 7.0),
            ionization_type=f["ionization_type"],
            glass_transition_temp_C=f["glass_transition_temp_C"],
            drug_loading_pct=f["drug_loading_pct"],
            storage_rh_pct=f["storage_rh_pct"],
            storage_temp_C=f["storage_temp_C"],
        )
        results.append(result)
    results.sort(key=lambda r: r.crystallization_risk_score)
    return results
