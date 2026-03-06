"""Transporter-mediated drug-drug interaction (DDI) assessment."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "TransporterDDIResult",
    "assess_transporter_ddi",
    "batch_transporter_screen",
]

_VALID_TRANSPORTERS = {"OATP1B1", "OATP1B3", "P-gp", "BCRP", "OCT2", "MATE1", "MATE2K"}
_VALID_INHIBITION_TYPES = {"competitive", "noncompetitive"}


@dataclass
class TransporterDDIResult:
    victim_drug: str
    perpetrator_drug: str
    transporter: str
    inhibition_type: str
    ki_uM: float
    victim_conc_uM: float | None
    inhibitor_conc_uM: float
    r_value: float
    aucr_predicted: float
    clinical_relevance: str
    fda_recommendation: str


def _r_value(
    inhibitor_conc_uM: float,
    ki_uM: float,
    inhibition_type: str = "competitive",
) -> float:
    """Compute R value (1 + I/Ki), clamped to >= 1.0."""
    r = 1.0 + inhibitor_conc_uM / ki_uM
    return max(1.0, r)


def _clinical_relevance(aucr: float) -> str:
    if aucr >= 5.0:
        return "contraindicated"
    if aucr >= 2.0:
        return "strong"
    if aucr >= 1.25:
        return "moderate"
    return "weak"


def _fda_recommendation(relevance: str) -> str:
    return {
        "contraindicated": "Avoid combination",
        "strong": "Use with caution, consider dose reduction",
        "moderate": "Monitor for toxicity",
        "weak": "No specific action required",
    }[relevance]


def assess_transporter_ddi(
    victim_drug: str,
    perpetrator_drug: str,
    transporter: str,
    ki_uM: float,
    victim_conc_uM: float | None = None,
    inhibitor_conc_uM: float = 0.1,
    inhibition_type: str = "competitive",
) -> TransporterDDIResult:
    """Assess transporter-mediated DDI for a single perpetrator/victim pair.

    Parameters
    ----------
    victim_drug:
        Name of the victim (substrate) drug.
    perpetrator_drug:
        Name of the perpetrator (inhibitor) drug.
    transporter:
        Transporter name (OATP1B1, OATP1B3, P-gp, BCRP, OCT2, MATE1, MATE2K).
    ki_uM:
        Inhibition constant in µM. Must be > 0.
    victim_conc_uM:
        Victim drug concentration in µM (informational only).
    inhibitor_conc_uM:
        Inhibitor (perpetrator) concentration in µM. Must be >= 0.
    inhibition_type:
        'competitive' or 'noncompetitive'.

    Returns
    -------
    TransporterDDIResult

    Raises
    ------
    ValueError
        If ki_uM <= 0 or inhibitor_conc_uM < 0.
    """
    if ki_uM <= 0:
        raise ValueError(f"ki_uM must be > 0, got {ki_uM}")
    if inhibitor_conc_uM < 0:
        raise ValueError(f"inhibitor_conc_uM must be >= 0, got {inhibitor_conc_uM}")

    r = _r_value(inhibitor_conc_uM, ki_uM, inhibition_type)
    aucr = r
    relevance = _clinical_relevance(aucr)
    recommendation = _fda_recommendation(relevance)

    return TransporterDDIResult(
        victim_drug=victim_drug,
        perpetrator_drug=perpetrator_drug,
        transporter=transporter,
        inhibition_type=inhibition_type,
        ki_uM=ki_uM,
        victim_conc_uM=victim_conc_uM,
        inhibitor_conc_uM=inhibitor_conc_uM,
        r_value=r,
        aucr_predicted=aucr,
        clinical_relevance=relevance,
        fda_recommendation=recommendation,
    )


def batch_transporter_screen(
    perpetrator_drug: str,
    inhibitor_conc_uM: float,
    transporter_ki_dict: dict[str, float],
) -> list[TransporterDDIResult]:
    """Screen a perpetrator drug across multiple transporters.

    Parameters
    ----------
    perpetrator_drug:
        Name of the perpetrator (inhibitor) drug.
    inhibitor_conc_uM:
        Inhibitor concentration in µM applied to all transporters.
    transporter_ki_dict:
        Mapping of transporter name -> Ki in µM.

    Returns
    -------
    List of TransporterDDIResult sorted by aucr_predicted descending.
    """
    results: list[TransporterDDIResult] = []
    for transporter, ki in transporter_ki_dict.items():
        result = assess_transporter_ddi(
            victim_drug="substrate",
            perpetrator_drug=perpetrator_drug,
            transporter=transporter,
            ki_uM=ki,
            inhibitor_conc_uM=inhibitor_conc_uM,
        )
        results.append(result)
    results.sort(key=lambda r: r.aucr_predicted, reverse=True)
    return results
