from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omega_pbpk.contracts import DrugSpec, PatientSpec, Regimen


@dataclass
class SimulationResult:
    """ODE 시뮬레이션 결과."""

    t: NDArray[np.float64]
    amounts: NDArray[np.float64]  # shape (n_states, n_timepoints)
    plasma_concentration: NDArray[np.float64]  # shape (n_timepoints,)
    drug_name: str = ""
    route: str = "oral"
    dose_mg: float = 0.0
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def Cmax(self) -> float:
        return float(np.max(self.plasma_concentration))

    @property
    def Tmax(self) -> float:
        idx = int(np.argmax(self.plasma_concentration))
        return float(self.t[idx])


class SimulationEngine(ABC):
    """ODE 시뮬레이션 엔진 추상 인터페이스."""

    @abstractmethod
    def run(
        self,
        drug: DrugSpec,
        patient: PatientSpec,
        regimen: Regimen,
        t_end_h: float = 24.0,
        n_points: int = 241,
        solver_config: dict | None = None,
    ) -> SimulationResult:
        """시뮬레이션 실행 후 SimulationResult 반환."""
        ...

    @abstractmethod
    def pk_summary(self, result: SimulationResult) -> dict[str, float]:
        """Cmax, Tmax, AUC, t_half 등 PK 지표 딕셔너리 반환."""
        ...
