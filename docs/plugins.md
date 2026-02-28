# Plugin Development Guide

Omega ML-Drug-Sim의 플러그인 시스템은 ML 모델, QSPR 방정식, 실험 측정값 등 다양한 파라미터 소스를 `DrugSpec` 및 `PatientSpec` 에 적용할 수 있는 교체 가능한 아키텍처를 제공합니다.

---

## PluginBase 구현 방법

커스텀 플러그인은 `omega_pbpk.plugins.base.PluginBase` ABC를 상속받고 세 가지 추상 멤버를 구현합니다.

```python
from omega_pbpk.plugins.base import PluginBase
from omega_pbpk.contracts import DrugSpec


class MyADMEPlugin(PluginBase):
    # 1. 플러그인 식별자 (레지스트리/로그에 사용)
    name = "my_adme_v1"

    # 2. 이 플러그인이 DrugSpec에 채워주는 필드명 집합
    provides = frozenset({"fup", "rbp"})

    def predict(self, spec: DrugSpec) -> dict[str, float | dict]:
        """DrugSpec을 입력으로 받아 `provides` 에 해당하는 필드값을 반환."""
        return {
            "fup": my_model.predict_fup(spec.smiles),
            "rbp": my_model.predict_rbp(spec.smiles),
        }

    def confidence(self, spec: DrugSpec) -> float:
        """예측 신뢰도 [0, 1]. 기본값 0.5 — 오버라이드 권장."""
        return 0.85
```

> `predict()` 반환 dict에 `provides` 에 선언된 모든 키가 없으면 `TypeError` 가 발생합니다.

### apply() 흐름

`apply()` 는 `PluginBase` 에 이미 구현되어 있습니다. 직접 오버라이드할 필요 없습니다.

```python
def apply(self, spec: DrugSpec) -> DrugSpec:
    updates = self.predict(spec)        # 서브클래스 predict() 호출
    # provides 집합 vs updates 키 검증
    return DrugSpec(**{**spec.__dict__, **updates})   # 새 불변 인스턴스 반환
```

---

## 사용 예시

```python
from omega_pbpk.adapters.yaml_loader import load_drug_spec
from omega_pbpk.engine.interface import SimulationEngine
from omega_pbpk.contracts import PatientSpec, Regimen

# 1. YAML에서 DrugSpec 로드
spec = load_drug_spec("compounds/caffeine.yaml")

# 2. 플러그인으로 파라미터 예측 및 적용
plugin = MyADMEPlugin()
updated_spec = plugin.apply(spec)   # fup, rbp 업데이트된 새 DrugSpec 반환
updated_spec.validate()             # 물리 범위 검증 (오류 없으면 통과)

# 3. PatientSpec과 Regimen 준비
patient = PatientSpec(body_weight_kg=65.0, age_years=45, sex="female")
regimen = Regimen(dose_mg=200.0, route="oral")

# 4. ODE 엔진으로 시뮬레이션
from omega_pbpk.engine.ode_engine import WholeBodyPBPKEngine

engine = WholeBodyPBPKEngine()
result = engine.run(updated_spec, patient, regimen, t_end_h=24.0)

# 5. PK 지표 추출
pk = engine.pk_summary(result)
print(f"Cmax = {pk['Cmax']:.2f} mg/L")
print(f"AUC  = {pk['AUC']:.1f} mg·h/L")
print(f"t½   = {pk['t_half']:.1f} h")
```

---

## 플러그인 유형

| 유형 | 클래스 | `provides` | 설명 |
|------|--------|-----------|------|
| ADME | `ADMEPredictorPlugin` | `fup`, `rbp`, `peff`, `clint_hepatic_L_per_h` | SMILES → ADME 속성 (QSPR 기반) |
| Kp | `HeuristicKpPlugin` | `kp` | 조직 분배계수 (Poulin-Theil 또는 Rodgers-Rowland) |
| ParameterNet | `ParameterNetPlugin` | `clint_hepatic_L_per_h`, `clr_L_per_h` | 환자 공변량 기반 청소율 스케일링 |

---

## 내장 플러그인 상세

### ADMEPredictorPlugin

SMILES 또는 물리화학적 특성에서 ADME 파라미터를 QSPR 방정식으로 예측합니다.

```python
from omega_pbpk.plugins.adme_plugin import ADMEPredictorPlugin

plugin = ADMEPredictorPlugin()
updated = plugin.apply(spec)
# updated.fup, updated.rbp, updated.peff, updated.clint_hepatic_L_per_h 업데이트됨
```

- SMILES 있을 때 신뢰도: `0.4`
- SMILES 없을 때 신뢰도: `0.2` (물리화학적 값 fallback)

### HeuristicKpPlugin

Poulin-Theil 또는 Rodgers-Rowland 방법으로 조직 분배계수 딕셔너리(`kp`)를 계산합니다.

```python
from omega_pbpk.plugins.heuristic_kp import HeuristicKpPlugin

# Poulin-Theil (기본값)
kp_plugin = HeuristicKpPlugin(method="poulin_theil")
updated = kp_plugin.apply(spec)   # spec.kp 딕셔너리가 채워진 새 DrugSpec

# Rodgers-Rowland
kp_plugin_rr = HeuristicKpPlugin(method="rodgers_rowland")
updated_rr = kp_plugin_rr.apply(spec)
```

- 신뢰도: `0.6`

### ParameterNetPlugin

환자 공변량(체중, GFR, CYP 활성도)을 기반으로 CLint와 CLR을 스케일링합니다.

```python
from omega_pbpk.plugins.parameter_net import ParameterNetPlugin
from omega_pbpk.contracts import PatientSpec

patient = PatientSpec(
    body_weight_kg=50.0,
    gfr_mL_min=60.0,       # 신장 기능 저하
    cyp3a4_activity=0.5,   # PGx: IM 표현형
)
param_plugin = ParameterNetPlugin(patient=patient)
personalized_spec = param_plugin.apply(spec)
# clint_hepatic_L_per_h, clr_L_per_h가 환자 공변량 기반으로 스케일링됨
```

PGx diplotype 기반 스케일링:

```python
pgx_spec = param_plugin.apply_pgx(spec, diplotype="*1/*4", enzyme="CYP3A4")
```

- 신뢰도: `0.7`

---

## 플러그인 파이프라인 조합

여러 플러그인을 순서대로 적용할 수 있습니다.

```python
from omega_pbpk.plugins.adme_plugin import ADMEPredictorPlugin
from omega_pbpk.plugins.heuristic_kp import HeuristicKpPlugin
from omega_pbpk.plugins.parameter_net import ParameterNetPlugin
from omega_pbpk.contracts import PatientSpec

patient = PatientSpec(body_weight_kg=80.0, gfr_mL_min=100.0)

pipeline = [
    ADMEPredictorPlugin(),              # step 1: fup, rbp, peff
    HeuristicKpPlugin("rodgers_rowland"), # step 2: kp
    ParameterNetPlugin(patient),         # step 3: 환자 개인화 CLint, CLR
]

current_spec = spec
for plugin in pipeline:
    current_spec = plugin.apply(current_spec)

current_spec.validate()
```

---

## SurrogateModelPlugin (Protocol)

ODE 대신 ML 서러게이트로 PK 곡선을 예측할 때 사용합니다. `Protocol` 기반이므로 상속 없이도 구현 가능합니다.

```python
from omega_pbpk.plugins.base import SurrogateModelPlugin
from omega_pbpk.contracts import DrugSpec, PatientSpec, Regimen


class MySurrogatePlugin:
    """SurrogateModelPlugin Protocol 구현체."""

    def predict_pk(
        self,
        drug: DrugSpec,
        patient: PatientSpec,
        regimen: Regimen,
    ) -> tuple[list[float], list[float], dict[str, float]]:
        """(시간 목록, 혈장농도 목록, PK 지표 딕셔너리) 반환."""
        t = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        c = my_nn.forward(drug, patient, regimen)
        metrics = {"Cmax": max(c), "AUC": trapezoidal(t, c)}
        return t, c, metrics


# runtime_checkable Protocol 확인
assert isinstance(MySurrogatePlugin(), SurrogateModelPlugin)  # True
```

---

## 테스트 작성 가이드

플러그인 단위 테스트 예시:

```python
import pytest
from omega_pbpk.contracts import DrugSpec
from my_package.plugins import MyADMEPlugin


@pytest.fixture
def caffeine_spec():
    return DrugSpec(
        name="Caffeine",
        smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        mw=194.19,
        fup=0.5,  # 예측 전 기본값
    )


def test_my_adme_plugin_provides_correct_fields(caffeine_spec):
    plugin = MyADMEPlugin()
    updated = plugin.apply(caffeine_spec)
    assert 0 < updated.fup <= 1.0
    assert updated.rbp > 0


def test_my_adme_plugin_confidence(caffeine_spec):
    plugin = MyADMEPlugin()
    conf = plugin.confidence(caffeine_spec)
    assert 0.0 <= conf <= 1.0


def test_updated_spec_passes_validation(caffeine_spec):
    plugin = MyADMEPlugin()
    updated = plugin.apply(caffeine_spec)
    updated.validate()  # ValueError 발생하면 안 됨
```

---

## 주의사항

1. `provides` 에 선언한 필드를 `predict()` 에서 모두 반환하지 않으면 `TypeError` 가 발생합니다.
2. `DrugSpec` 은 `frozen=True` 이므로 직접 수정 불가 — `apply()` 가 새 인스턴스를 반환합니다.
3. `kp` 필드는 `dict[str, float]` 타입입니다. `predict()` 반환 시 `{"kp": {...}}` 형태로 반환하세요.
4. `confidence()` 는 선택 사항이지만 구현을 권장합니다. 기본값은 `0.5` 입니다.
