"""Pericardial fluid drug distribution prediction.

Predicts drug concentration in pericardial fluid, relevant for
intrapericardial drug delivery (antiarrhythmics, anti-inflammatory agents).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PericardialDistributionResult",
    "predict_pericardial_distribution",
    "screen_intrapericardial_drugs",
]


@dataclass(frozen=True)
class PericardialDistributionResult:
    drug_name: str
    logp: float
    mw_Da: float
    route: str
    dose_mg: float
    pericardial_plasma_ratio: float
    predicted_pericardial_conc_mg_L: float
    plasma_conc_input_mg_L: float
    pericardial_residence_h: float
    epicardial_penetration_fraction: float
    local_cardiac_advantage: float
    notes: str


def predict_pericardial_distribution(
    drug_name: str,
    dose_mg: float = 1.0,
    logp: float = 2.0,
    mw_Da: float = 300.0,
    route: str = "systemic",
    plasma_conc_mg_L: float = 1.0,
) -> PericardialDistributionResult:
    """Predict drug distribution in pericardial fluid.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose administered in mg. Must be > 0.
    logp:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons. Must be > 0.
    route:
        Administration route: "intrapericardial" or "systemic".
    plasma_conc_mg_L:
        Plasma drug concentration in mg/L (input for systemic route).

    Returns
    -------
    PericardialDistributionResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if route not in {"intrapericardial", "systemic"}:
        raise ValueError(f"route must be 'intrapericardial' or 'systemic', got {route!r}")

    safe_plasma = max(plasma_conc_mg_L, 0.001)

    if route == "systemic":
        ratio = 0.05 + logp * 0.02
        ratio = max(0.01, min(ratio, 0.5))
        pericardial_conc = ratio * plasma_conc_mg_L
        pericardial_plasma_ratio = ratio
    else:
        # intrapericardial: direct injection into ~30 mL volume
        v_pericardial_L = 0.030
        pericardial_conc = dose_mg / v_pericardial_L
        pericardial_plasma_ratio = pericardial_conc / safe_plasma
        pericardial_plasma_ratio = max(1.0, min(pericardial_plasma_ratio, 1000.0))

    # Residence time in pericardial space
    if route == "systemic":
        residence = 0.5 + logp * 0.2
        residence = max(0.1, min(residence, 5.0))
    else:
        residence = 2.0 + logp * 1.0
        residence = max(1.0, min(residence, 48.0))

    # Epicardial penetration fraction
    if logp > 2:
        epicardial_penetration = 0.3
    elif logp > 0:
        epicardial_penetration = 0.15
    else:
        epicardial_penetration = 0.05

    # Local cardiac advantage
    local_advantage = pericardial_conc / safe_plasma
    local_advantage = max(0.1, min(local_advantage, 1000.0))

    # Notes
    if route == "intrapericardial" and local_advantage > 10:
        notes = (
            "High local cardiac advantage — intrapericardial delivery "
            "achieves targeted cardiac exposure"
        )
    elif epicardial_penetration > 0.2:
        notes = "Good epicardial penetration for cardiac PD effects"
    else:
        notes = "Limited pericardial distribution via systemic route"

    return PericardialDistributionResult(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        route=route,
        dose_mg=dose_mg,
        pericardial_plasma_ratio=pericardial_plasma_ratio,
        predicted_pericardial_conc_mg_L=pericardial_conc,
        plasma_conc_input_mg_L=plasma_conc_mg_L,
        pericardial_residence_h=residence,
        epicardial_penetration_fraction=epicardial_penetration,
        local_cardiac_advantage=local_advantage,
        notes=notes,
    )


def screen_intrapericardial_drugs(
    candidates: list[dict],
) -> list[PericardialDistributionResult]:
    """Screen candidate drugs for intrapericardial delivery.

    Parameters
    ----------
    candidates:
        List of dicts with keys: "name" (str), "logp" (float),
        optional "mw_Da" (float, default 300).

    Returns
    -------
    List of PericardialDistributionResult sorted by epicardial_penetration_fraction
    descending.
    """
    results = []
    for cand in candidates:
        name = cand["name"]
        logp = float(cand["logp"])
        mw_Da = float(cand.get("mw_Da", 300.0))
        result = predict_pericardial_distribution(
            drug_name=name,
            logp=logp,
            mw_Da=mw_Da,
            route="intrapericardial",
        )
        results.append(result)
    results.sort(key=lambda r: r.epicardial_penetration_fraction, reverse=True)
    return results
