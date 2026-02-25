from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from physio_sim.qsp.base import BaseQSPModel
from physio_sim.qsp.registry import register_qsp_model


@register_qsp_model("turnover")
class TurnoverBiomarkerModel(BaseQSPModel):
    def validate_params(self, config: Mapping[str, float]) -> None:
        required = ("kin", "kout", "emax", "ec50_mg_per_L")
        for key in required:
            if key not in config:
                msg = f"Missing required parameter for turnover model: {key}"
                raise ValueError(msg)
        values = {key: float(config[key]) for key in required}
        if values["kin"] < 0:
            raise ValueError("kin must be non-negative")
        if values["kout"] <= 0:
            raise ValueError("kout must be positive")
        if values["emax"] < 0:
            raise ValueError("emax must be non-negative")
        if values["ec50_mg_per_L"] <= 0:
            raise ValueError("ec50_mg_per_L must be positive")
        hill = float(config.get("hill", 1.0))
        if hill <= 0:
            raise ValueError("hill must be positive")

    def state_names(self) -> tuple[str, ...]:
        return ("B_biomarker",)

    def initial_state(self, config: Mapping[str, float]) -> NDArray[np.float64]:
        kin = float(config["kin"])
        kout = float(config["kout"])
        b0 = float(config.get("B0", kin / max(kout, 1e-12)))
        return np.array([max(0.0, b0)], dtype=float)

    def rhs(
        self,
        t: float,
        state: NDArray[np.float64],
        signals: Mapping[str, float],
        config: Mapping[str, float],
    ) -> NDArray[np.float64]:
        _ = t
        b = max(0.0, float(state[0]))
        kin = float(config["kin"])
        kout = float(config["kout"])
        emax = float(config["emax"])
        ec50 = float(config["ec50_mg_per_L"])
        hill = float(config.get("hill", 1.0))
        c_signal = max(0.0, float(signals["C_signal"]))

        c_hill = c_signal**hill
        effect = (emax * c_hill) / (ec50**hill + c_hill + 1e-12)
        db = kin - kout * b - effect * b
        if b <= 0.0 and db < 0.0:
            db = 0.0
        return np.array([db], dtype=float)
