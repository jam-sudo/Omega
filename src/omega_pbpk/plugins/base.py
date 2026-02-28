from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from omega_pbpk.contracts import DrugSpec, PatientSpec, Regimen


class PluginBase(ABC):
    """모든 ML 플러그인의 기반 추상 클래스."""

    @property
    @abstractmethod
    def name(self) -> str:
        """플러그인 이름."""
        ...

    @property
    @abstractmethod
    def provides(self) -> frozenset[str]:
        """이 플러그인이 DrugSpec에 채워주는 필드명 집합."""
        ...

    @abstractmethod
    def predict(self, spec: DrugSpec) -> dict[str, float | dict]:
        """DrugSpec을 받아 provides 집합에 해당하는 필드값 반환."""
        ...

    def confidence(self, spec: DrugSpec) -> float:
        """예측 신뢰도 [0, 1]. 서브클래스에서 오버라이드 가능."""
        return 0.5

    def apply(self, spec: DrugSpec) -> DrugSpec:
        """predict() 결과를 DrugSpec에 병합한 새 DrugSpec 반환."""
        updates = self.predict(spec)
        missing = self.provides - set(updates.keys())
        if missing:
            raise TypeError(
                f"Plugin '{self.name}' declared provides={self.provides!r} "
                f"but predict() did not return keys: {missing!r}"
            )
        return DrugSpec(**{**spec.__dict__, **updates})  # type: ignore[arg-type]


@runtime_checkable
class SurrogateModelPlugin(Protocol):
    """서러게이트 모델 플러그인 인터페이스 (Protocol)."""

    def predict_pk(
        self,
        drug: DrugSpec,
        patient: PatientSpec,
        regimen: Regimen,
    ) -> tuple[list[float], list[float], dict[str, float]]:
        """(t_list, C_plasma_list, pk_metrics_dict) 반환."""
        ...
