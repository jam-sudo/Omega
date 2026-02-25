from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray


class BaseQSPModel(ABC):
    @abstractmethod
    def state_names(self) -> tuple[str, ...]:
        """Names of model states in output order."""

    @abstractmethod
    def initial_state(self, config: Mapping[str, float]) -> NDArray[np.float64]:
        """Initial QSP state vector."""


    def validate_params(self, params: Mapping[str, float]) -> None:
        """Validate model parameter values before runtime execution."""
        return None

    @abstractmethod
    def rhs(
        self,
        t: float,
        state: NDArray[np.float64],
        signals: Mapping[str, float],
        config: Mapping[str, float],
    ) -> NDArray[np.float64]:
        """QSP ODE right-hand-side."""
