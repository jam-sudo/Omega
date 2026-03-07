"""
Phase 962 — Comprehensive lipophilicity profiling.

logD at multiple pH values, membrane partitioning, permeability estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10

__all__ = ["LipophilicityProfileResult", "profile_lipophilicity", "screen_lipophilicity"]


@dataclass(frozen=True)
class LipophilicityProfileResult:
    drug_name: str
    logp: float
    pka: float
    ionization_type: str
    logd_at_gastric_ph: float  # pH 1.2
    logd_at_intestinal_ph: float  # pH 6.8
    logd_at_colonic_ph: float  # pH 7.4
    logd_plasma_ph: float  # pH 7.4 (same as colonic)
    membrane_partition_coeff: float
    skin_permeability_cm_h: float
    fu_predicted: float
    optimal_for_oral: bool
    optimal_for_cns: bool
    lipophilicity_class: str
    efflux_risk: bool
    notes: str


def _logd(logp: float, pka: float, ionization_type: str, ph: float) -> float:
    """Compute logD at a given pH for acid, base, or neutral."""
    if ionization_type == "neutral":
        return logp
    elif ionization_type == "acid":
        # logD = logP - log10(1 + 10^(pH - pKa))
        exponent = ph - pka
        # clamp to avoid overflow
        exponent = max(-50.0, min(50.0, exponent))
        correction = log10(1.0 + 10.0**exponent)
        return logp - correction
    else:  # base
        # logD = logP - log10(1 + 10^(pKa - pH))
        exponent = pka - ph
        exponent = max(-50.0, min(50.0, exponent))
        correction = log10(1.0 + 10.0**exponent)
        return logp - correction


def profile_lipophilicity(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
) -> LipophilicityProfileResult:
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got '{ionization_type}'"
        )

    logd_gastric = _logd(logp, pka, ionization_type, 1.2)
    logd_intestinal = _logd(logp, pka, ionization_type, 6.8)
    logd_colonic = _logd(logp, pka, ionization_type, 7.4)
    logd_plasma = logd_colonic  # pH 7.4

    # Membrane partitioning: logKp_membrane = 0.65 * logP - 0.5
    logkp_membrane = 0.65 * logp - 0.5
    membrane_partition_coeff = 10.0**logkp_membrane

    # Skin permeability: logKm_skin = 0.71 * logP - 6.3  (units: cm/h)
    logkm_skin = 0.71 * logp - 6.3
    skin_permeability_cm_h = 10.0**logkm_skin

    # fu predicted: logfu ≈ -0.5 * logP, clamped to [0, 1]
    logfu = -0.5 * logp
    fu_raw = 10.0**logfu
    fu_predicted = max(0.0, min(1.0, fu_raw))

    optimal_for_oral = -1.0 <= logp <= 5.0
    optimal_for_cns = 1.0 <= logp <= 3.0

    if logp < 0:
        lipophilicity_class = "hydrophilic"
    elif logp > 3:
        lipophilicity_class = "lipophilic"
    else:
        lipophilicity_class = "moderate"

    # P-gp efflux risk: logP > 4 AND pKa > 7 (basic lipophilic)
    efflux_risk = (logp > 4) and (pka > 7)

    notes = (
        f"logP={logp:.2f}; pKa={pka:.2f}; type={ionization_type}; "
        f"logD(pH1.2)={logd_gastric:.2f}; logD(pH6.8)={logd_intestinal:.2f}; "
        f"logD(pH7.4)={logd_colonic:.2f}; class={lipophilicity_class}; "
        f"efflux_risk={efflux_risk}"
    )

    return LipophilicityProfileResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        logd_at_gastric_ph=logd_gastric,
        logd_at_intestinal_ph=logd_intestinal,
        logd_at_colonic_ph=logd_colonic,
        logd_plasma_ph=logd_plasma,
        membrane_partition_coeff=membrane_partition_coeff,
        skin_permeability_cm_h=skin_permeability_cm_h,
        fu_predicted=fu_predicted,
        optimal_for_oral=optimal_for_oral,
        optimal_for_cns=optimal_for_cns,
        lipophilicity_class=lipophilicity_class,
        efflux_risk=efflux_risk,
        notes=notes,
    )


def screen_lipophilicity(compounds: list) -> list:
    """Screen a list of compound dicts and return results sorted by logp ascending."""
    results = []
    for comp in compounds:
        r = profile_lipophilicity(
            drug_name=comp["drug_name"],
            logp=comp["logp"],
            pka=comp["pka"],
            ionization_type=comp["ionization_type"],
        )
        results.append(r)
    results.sort(key=lambda x: x.logp)
    return results
