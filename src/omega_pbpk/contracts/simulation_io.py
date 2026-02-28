from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Regimen:
    dose_mg: float
    route: Literal["oral", "iv", "sc"] = "oral"
    interval_h: float = 24.0
    n_doses: int = 1

    def validate(self) -> None:
        if self.dose_mg <= 0:
            raise ValueError(f"dose_mg must be > 0, got {self.dose_mg}")
        if self.interval_h <= 0:
            raise ValueError(f"interval_h must be > 0, got {self.interval_h}")
        if self.n_doses < 1:
            raise ValueError(f"n_doses must be >= 1, got {self.n_doses}")


@dataclass
class PKCurve:
    t: NDArray[np.float64]
    C_plasma: NDArray[np.float64]
    C_tissue_dict: dict[str, NDArray[np.float64]] = field(default_factory=dict)


@dataclass(frozen=True)
class PKMetrics:
    Cmax: float
    Tmax: float
    AUC: float
    t_half: float


@dataclass(frozen=True)
class ADMEOutput:
    Fa: float = 1.0  # fraction absorbed
    Fg: float = 1.0  # fraction surviving gut
    Fh: float = 1.0  # fraction surviving liver
    CLint: float = 0.0  # intrinsic clearance L/h
    fu: float = 0.5  # fraction unbound
    Vd: float = 30.0  # volume of distribution L
    confidence: Literal["low", "medium", "high"] = "low"


@dataclass(frozen=True)
class Uncertainty:
    ci_lower: tuple[float, ...]
    ci_upper: tuple[float, ...]
    method: Literal["mc", "ensemble", "conformal", "quantile"] = "mc"
    coverage: float = 0.90
