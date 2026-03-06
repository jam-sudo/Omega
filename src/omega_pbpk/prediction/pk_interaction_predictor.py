"""PK-PK interaction prediction — CYP inhibition and induction AUCR."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PKInteractionResult:
    perpetrator: str
    victim: str
    mechanism: str
    aucr_predicted: float
    cmax_ratio: float
    risk_level: str  # none / mild / moderate / severe
    fda_recommendation: str
    confidence: float


def _aucr_cyp_inhibition(
    ki_uM: float,
    inhibitor_conc_uM: float,
    fm: float = 0.8,
) -> float:
    """AUCR from competitive CYP inhibition.

    R1 = 1 + I/Ki; AUCR = 1 / (1 - fm + fm/R1)
    """
    r1 = 1.0 + inhibitor_conc_uM / max(ki_uM, 1e-9)
    aucr = 1.0 / (1.0 - fm + fm / r1)
    return float(min(max(aucr, 1.0), 20.0))


def _aucr_induction(
    emax: float,
    ec50_uM: float,
    inducer_conc_uM: float,
    fm: float = 0.8,
) -> float:
    """AUCR from enzyme induction (AUC decreases, AUCR < 1).

    fold = 1 + emax*C/(ec50+C); AUCR = 1/(1 - fm + fm/fold)
    """
    if inducer_conc_uM <= 0:
        return 1.0
    fold = 1.0 + emax * inducer_conc_uM / (max(ec50_uM, 1e-9) + inducer_conc_uM)
    # Induction increases CLint by fold → AUC decreases → AUCR = 1-fm+fm/fold < 1
    aucr = 1.0 - fm + fm / fold
    return float(aucr)


def _risk_level_inhibition(aucr: float) -> str:
    if aucr < 1.25:
        return "none"
    elif aucr < 2.0:
        return "mild"
    elif aucr < 5.0:
        return "moderate"
    else:
        return "severe"


def _risk_level_induction(aucr: float) -> str:
    if aucr >= 0.8:
        return "none"
    elif aucr >= 0.5:
        return "moderate"
    else:
        return "severe"


def _fda_recommendation(mechanism: str, risk: str) -> str:
    if risk == "none":
        return "No dose adjustment needed"
    elif mechanism == "inhibition":
        mapping = {
            "mild": "Monitor; dose adjustment may not be needed",
            "moderate": "Reduce victim drug dose; monitor closely",
            "severe": "Contraindicated or strong dose reduction required",
        }
        return mapping.get(risk, "Consult label")
    else:
        mapping = {
            "moderate": "Consider dose increase; monitor efficacy",
            "severe": "Avoid combination or significantly increase victim dose",
        }
        return mapping.get(risk, "Consult label")


def predict_pk_interaction(
    perpetrator: str,
    victim: str,
    ki_uM: float | None = None,
    emax: float | None = None,
    ec50_uM: float | None = None,
    inhibitor_conc_uM: float = 1.0,
    inducer_conc_uM: float = 1.0,
    fm: float = 0.8,
) -> PKInteractionResult:
    """Predict PK interaction between perpetrator and victim."""
    aucr = 1.0
    mechanisms = []

    if ki_uM is not None:
        aucr_inh = _aucr_cyp_inhibition(ki_uM, inhibitor_conc_uM, fm)
        aucr *= aucr_inh
        mechanisms.append("inhibition")

    if emax is not None and ec50_uM is not None:
        aucr_ind = _aucr_induction(emax, ec50_uM, inducer_conc_uM, fm)
        aucr *= aucr_ind
        mechanisms.append("induction")

    if not mechanisms:
        mechanisms = ["unknown"]

    mechanism = "+".join(mechanisms)

    if "inhibition" in mechanism:
        risk = _risk_level_inhibition(aucr)
    else:
        risk = _risk_level_induction(aucr)

    cmax_ratio = float(aucr**0.5)
    confidence = 0.9 if fm > 0.5 else 0.7
    recommendation = _fda_recommendation(mechanism, risk)

    return PKInteractionResult(
        perpetrator=perpetrator,
        victim=victim,
        mechanism=mechanism,
        aucr_predicted=aucr,
        cmax_ratio=cmax_ratio,
        risk_level=risk,
        fda_recommendation=recommendation,
        confidence=confidence,
    )


def screen_interaction_pairs(pairs: list[dict]) -> list[PKInteractionResult]:
    """Screen a list of perpetrator-victim pairs for PK interactions."""
    results = []
    for p in pairs:
        r = predict_pk_interaction(
            perpetrator=p.get("perpetrator", "unknown"),
            victim=p.get("victim", "unknown"),
            ki_uM=p.get("ki_uM"),
            emax=p.get("emax"),
            ec50_uM=p.get("ec50_uM"),
            inhibitor_conc_uM=p.get("inhibitor_conc_uM", 1.0),
            inducer_conc_uM=p.get("inducer_conc_uM", 1.0),
            fm=p.get("fm", 0.8),
        )
        results.append(r)
    return results


__all__ = [
    "PKInteractionResult",
    "predict_pk_interaction",
    "screen_interaction_pairs",
    "_aucr_cyp_inhibition",
    "_aucr_induction",
]
