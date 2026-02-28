from __future__ import annotations

from typing import Literal

from omega_pbpk.plugins.base import PluginBase
from omega_pbpk.contracts import DrugSpec


class HeuristicKpPlugin(PluginBase):
    """Poulin & Theil 또는 Rodgers & Rowland Kp 추정을 PluginBase로 래핑."""

    name = "heuristic_kp"
    provides = frozenset({"kp"})

    def __init__(self, method: Literal["poulin_theil", "rodgers_rowland"] = "poulin_theil") -> None:
        self.method = method

    def predict(self, spec: DrugSpec) -> dict[str, float | dict]:
        pka = spec.pka[0] if spec.pka else None

        if self.method == "rodgers_rowland":
            from omega_pbpk.core.heuristics import rodgers_rowland_kp, _TISSUE_FACTORS

            kp_dict: dict[str, float] = {
                tissue: rodgers_rowland_kp(
                    logP=spec.logP,
                    pka=pka,
                    compound_type=spec.compound_type,
                    tissue_name=tissue,
                    fup=spec.fup,
                )
                for tissue in _TISSUE_FACTORS
            }
        else:
            # poulin_theil (default)
            from omega_pbpk.core.heuristics import estimate_all_kp

            kp_dict = estimate_all_kp(
                logP=spec.logP,
                pka=pka,
                drug_type=spec.compound_type,
                fup=spec.fup,
            )

        return {"kp": kp_dict}

    def confidence(self, spec: DrugSpec) -> float:
        return 0.6
