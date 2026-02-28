"""ParameterNetPlugin — rule-based PBPK parameter scaling.

MVP 구현: PGx CYP scaling + 체중/연령 allometric scaling.
추후 ML NN으로 교체 가능한 PluginBase 구조.
"""

from __future__ import annotations

from omega_pbpk.contracts.drug_spec import DrugSpec
from omega_pbpk.contracts.patient_spec import PatientSpec
from omega_pbpk.plugins.base import PluginBase

_REFERENCE_BW_KG = 70.0
_REFERENCE_GFR_ML_MIN = 125.0


class ParameterNetPlugin(PluginBase):
    """Rule-based PBPK parameter personalizer.

    입력: PatientSpec (환자 공변량)
    출력: DrugSpec 업데이트 (CLint 스케일링 등)
    """

    name = "parameter_net_rule_based"
    provides = frozenset({"clint_hepatic_L_per_h", "clr_L_per_h"})

    def __init__(self, patient: PatientSpec | None = None):
        self._patient = patient or PatientSpec()

    def set_patient(self, patient: PatientSpec) -> None:
        self._patient = patient

    def predict(self, spec: DrugSpec) -> dict[str, float | dict]:
        """환자 공변량 기반 CLint, CLR 스케일링."""
        # 1. PGx scaling: CYP3A4 activity 기반
        hepatic_scale = self._patient.hepatic_cl_factor * self._patient.cyp3a4_activity
        # 2. Renal scaling: GFR 기반 (정상 GFR=125 mL/min 기준)
        renal_scale = self._patient.renal_cl_factor * (
            self._patient.gfr_mL_min / _REFERENCE_GFR_ML_MIN
        )
        # 3. 체중 allometric scaling (기준: 70 kg)
        bw_ratio = self._patient.body_weight_kg / _REFERENCE_BW_KG
        bw_scale = bw_ratio**0.75  # allometric exponent

        return {
            "clint_hepatic_L_per_h": spec.clint_hepatic_L_per_h * hepatic_scale * bw_scale,
            "clr_L_per_h": spec.clr_L_per_h * renal_scale * bw_scale,
        }

    def confidence(self, spec: DrugSpec) -> float:
        return 0.7  # rule-based > ML stub

    def apply_pgx(self, spec: DrugSpec, diplotype: str, enzyme: str = "CYP3A4") -> DrugSpec:
        """PGx diplotype 기반 CLint 스케일링.

        Args:
            spec: 입력 DrugSpec.
            diplotype: 예) "*1/*4" (CPIC 형식).
            enzyme: CYP 효소명 (기본값 "CYP3A4").

        Returns:
            CLint가 PGx 스케일링된 새 DrugSpec.
        """
        from omega_pbpk.pharmacogenomics.cyp_polymorphism import PGxAnalyzer

        analyzer = PGxAnalyzer()
        scaling = analyzer.clint_scaling(gene=enzyme, diplotype=diplotype)
        updated = dict(spec.__dict__)
        updated["clint_hepatic_L_per_h"] = spec.clint_hepatic_L_per_h * scaling
        return DrugSpec(**updated)
