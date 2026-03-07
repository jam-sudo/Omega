"""Drug efflux ratio prediction for P-gp/BCRP substrates — Phase 511."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EffluxImpactResult:
    """Result from assess_efflux_impact."""

    efflux_ratio: float
    kp_uu_effective: float
    pgp_concern: str
    gut_bioavailability_impact: float
    cns_penetration_impact: float
    recommended_action: str
    notes: str


# ---------------------------------------------------------------------------
# QSAR for efflux ratio
# ---------------------------------------------------------------------------

_ER_MIN = 1.0
_ER_MAX = 20.0


def predict_efflux_ratio(
    logP: float,
    mw: float,
    psa_A2: float,
    n_hbd: int,
    has_amine: bool = False,
) -> float:
    """Predict efflux ratio (ER) from physicochemical descriptors.

    QSAR rules:
    - Base ER = 1.0
    - +2.0 per 100 Da above MW 400
    - +1.5 if PSA > 60 Å² and logP < 2 (hydrophilic, polar)
    - +1.0 if has basic amine (heuristic: n_hbd >= 2 and PSA > 40)
    - -0.5 if logP > 4 (hydrophobic → less P-gp)
    - Clamped to [1.0, 20.0]
    """
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")

    er = 1.0

    # MW contribution: +2.0 per 100 Da above 400
    if mw > 400.0:
        er += 2.0 * ((mw - 400.0) / 100.0)

    # Hydrophilic polar contribution
    if psa_A2 > 60.0 and logP < 2.0:
        er += 1.5

    # Basic amine heuristic
    if has_amine or (n_hbd >= 2 and psa_A2 > 40.0):
        er += 1.0

    # High logP reduces P-gp efflux
    if logP > 4.0:
        er -= 0.5

    return float(max(_ER_MIN, min(_ER_MAX, er)))


def predict_pgp_substrate(
    logP: float,
    mw: float,
    psa_A2: float,
    n_hbd: int,
) -> bool:
    """Predict P-glycoprotein substrate status.

    Rule: MW > 400 AND PSA > 60 → True.
    """
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")
    return mw > 400.0 and psa_A2 > 60.0


def predict_bcrp_substrate(
    logP: float,
    mw: float,
    psa_A2: float,
) -> bool:
    """Predict BCRP substrate status.

    Rule: MW > 350 AND (PSA < 40 OR logP > 3) → True.
    """
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")
    return mw > 350.0 and (psa_A2 < 40.0 or logP > 3.0)


def assess_efflux_impact(
    efflux_ratio: float,
    fu_plasma: float = 0.1,
    kp_uu_baseline: float = 1.0,
) -> EffluxImpactResult:
    """Assess the pharmacokinetic impact of transporter-mediated efflux.

    Parameters
    ----------
    efflux_ratio:
        Predicted or measured efflux ratio (ER). Must be >= 1.
    fu_plasma:
        Unbound fraction in plasma. Must be in (0, 1].
    kp_uu_baseline:
        Unbound brain-to-plasma concentration ratio at baseline.

    Returns
    -------
    EffluxImpactResult with CNS and gut impact assessments.
    """
    if efflux_ratio < 1.0:
        raise ValueError(f"efflux_ratio must be >= 1.0, got {efflux_ratio}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if kp_uu_baseline <= 0.0:
        raise ValueError(f"kp_uu_baseline must be positive, got {kp_uu_baseline}")

    # CNS: kp_uu reduced by ER
    kp_uu_effective = kp_uu_baseline / efflux_ratio

    # Gut bioavailability: fraction remaining = 1 / ER (efflux reduces absorption)
    gut_bioavailability_impact = 1.0 / efflux_ratio

    # CNS penetration as fraction of baseline
    cns_penetration_impact = kp_uu_effective / kp_uu_baseline

    # Concern level
    if efflux_ratio < 2.0:
        pgp_concern = "none"
        recommended_action = "No efflux-related dose adjustment needed."
        notes = "Low efflux ratio; transporter effect negligible."
    elif efflux_ratio < 5.0:
        pgp_concern = "mild"
        recommended_action = "Monitor for reduced bioavailability; consider dose adjustment."
        notes = "Mild P-gp/BCRP efflux; may reduce oral bioavailability and CNS penetration."
    elif efflux_ratio < 10.0:
        pgp_concern = "moderate"
        recommended_action = (
            "Consider efflux inhibitor co-administration or formulation strategies."
        )
        notes = "Moderate efflux; likely to reduce oral FA and limit CNS exposure significantly."
    else:
        pgp_concern = "severe"
        recommended_action = (
            "High efflux burden; reassess therapeutic target or use parenteral route."
        )
        notes = "Severe efflux ratio (>10); CNS and oral bioavailability will be markedly reduced."

    return EffluxImpactResult(
        efflux_ratio=efflux_ratio,
        kp_uu_effective=kp_uu_effective,
        pgp_concern=pgp_concern,
        gut_bioavailability_impact=gut_bioavailability_impact,
        cns_penetration_impact=cns_penetration_impact,
        recommended_action=recommended_action,
        notes=notes,
    )


def rank_efflux_concern(drugs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank a list of drugs by predicted efflux ratio (descending).

    Each drug dict must contain keys: 'name', 'logP', 'mw', 'psa_A2', 'n_hbd'.
    Optional key: 'has_amine' (bool, default False).

    Returns a list of dicts with original fields plus 'efflux_ratio', 'pgp_substrate',
    'bcrp_substrate', sorted by efflux_ratio descending.
    """
    if not isinstance(drugs, list):
        raise TypeError("drugs must be a list of dicts")

    results = []
    for drug in drugs:
        if not isinstance(drug, dict):
            raise TypeError("Each entry in drugs must be a dict")

        name = drug.get("name", "unknown")
        logP = float(drug["logP"])
        mw = float(drug["mw"])
        psa_A2 = float(drug["psa_A2"])
        n_hbd = int(drug["n_hbd"])
        has_amine = bool(drug.get("has_amine", False))

        er = predict_efflux_ratio(logP, mw, psa_A2, n_hbd, has_amine)
        pgp = predict_pgp_substrate(logP, mw, psa_A2, n_hbd)
        bcrp = predict_bcrp_substrate(logP, mw, psa_A2)

        entry = dict(drug)
        entry.update(
            {
                "name": name,
                "efflux_ratio": er,
                "pgp_substrate": pgp,
                "bcrp_substrate": bcrp,
            }
        )
        results.append(entry)

    results.sort(key=lambda d: d["efflux_ratio"], reverse=True)
    return results
