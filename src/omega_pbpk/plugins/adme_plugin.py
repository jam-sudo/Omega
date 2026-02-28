from __future__ import annotations

from omega_pbpk.contracts import DrugSpec
from omega_pbpk.plugins.base import PluginBase


class ADMEPredictorPlugin(PluginBase):
    """기존 ADMEPredictor를 PluginBase 인터페이스로 래핑."""

    name = "adme_predictor_qspr"
    provides = frozenset({"fup", "rbp", "peff", "clint_hepatic_L_per_h"})

    def __init__(self) -> None:
        from omega_pbpk.prediction.adme_predictor import ADMEPredictor

        self._predictor = ADMEPredictor()

    def predict(self, spec: DrugSpec) -> dict[str, float | dict]:
        """SMILES 또는 DrugSpec의 physicochemical 특성으로 ADME 예측."""
        smiles = spec.smiles or ""
        if smiles:
            props = self._predictor.predict(smiles)
        else:
            # SMILES 없을 때: DrugSpec의 physicochemical 값으로 fallback
            props = self._predictor.predict_from_dict(
                {
                    "mw": spec.mw,
                    "logP": spec.logP,
                    "fup": spec.fup,
                    "rbp": spec.rbp,
                    "peff": spec.peff,
                }
            )

        return {
            "fup": float(props.fup),
            "rbp": float(props.rbp),
            "peff": float(props.peff),
            # clint_hepatic_L_per_h: IVIVE 변환은 Drug 내부에서 처리하므로 0 반환
            "clint_hepatic_L_per_h": 0.0,
        }

    def confidence(self, spec: DrugSpec) -> float:
        return 0.4 if spec.smiles else 0.2
