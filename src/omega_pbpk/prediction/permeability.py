"""Caco-2/PAMPA permeability prediction from physicochemical descriptors."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PermeabilityResult",
    "predict_permeability",
    "screen_permeability",
]


@dataclass
class PermeabilityResult:
    drug_name: str
    logP: float
    MW: float
    PSA: float
    HBD: int
    Papp_AB_cm_s: float
    Peff_cm_s: float
    log_Papp: float
    permeability_class: str
    fa_predicted: float
    efflux_ratio_estimate: float
    transport_mechanism: str


def _hbd_count_estimate(MW: float, logP: float, PSA: float) -> int:
    return round(max(0.0, PSA / 13.0 - logP * 0.2))


def _papp_from_descriptors(logP: float, MW: float, PSA: float, HBD: int) -> float:
    exponent = 0.4 * logP - 0.01 * MW - 0.02 * PSA - 0.3 * HBD - 5.0
    base = 10.0**exponent
    return max(1e-8, min(1e-4, base))


def predict_permeability(
    drug_name: str,
    logP: float,
    MW: float,
    PSA: float,
    HBD: int | None = None,
    efflux_substrate: bool = False,
) -> PermeabilityResult:
    if MW <= 0:
        raise ValueError("MW must be positive")
    if PSA < 0:
        raise ValueError("PSA must be non-negative")

    if HBD is None:
        HBD = _hbd_count_estimate(MW, logP, PSA)

    Papp = _papp_from_descriptors(logP, MW, PSA, HBD)
    Peff = Papp * 0.4
    log_Papp = math.log10(Papp)

    if Papp >= 1e-6:
        permeability_class = "high"
    elif Papp < 1e-7:
        permeability_class = "low"
    else:
        permeability_class = "intermediate"

    fa = 1.0 / (1.0 + math.exp(-(log_Papp + 5.5) * 2.0))

    if efflux_substrate:
        efflux_ratio = 2.0 + max(logP - 2.0, 0.0) * 0.5
    else:
        efflux_ratio = 1.0

    if logP > 1:
        transport_mechanism = "passive_transcellular"
    elif PSA > 140:
        transport_mechanism = "passive_paracellular"
    else:
        transport_mechanism = "mixed"

    return PermeabilityResult(
        drug_name=drug_name,
        logP=logP,
        MW=MW,
        PSA=PSA,
        HBD=HBD,
        Papp_AB_cm_s=Papp,
        Peff_cm_s=Peff,
        log_Papp=log_Papp,
        permeability_class=permeability_class,
        fa_predicted=fa,
        efflux_ratio_estimate=efflux_ratio,
        transport_mechanism=transport_mechanism,
    )


def screen_permeability(compounds: list[dict]) -> list[PermeabilityResult]:
    results = []
    for compound in compounds:
        result = predict_permeability(
            drug_name=compound["drug_name"],
            logP=compound["logP"],
            MW=compound["MW"],
            PSA=compound["PSA"],
            HBD=compound.get("HBD"),
            efflux_substrate=compound.get("efflux_substrate", False),
        )
        results.append(result)
    return sorted(results, key=lambda r: r.Papp_AB_cm_s, reverse=True)
