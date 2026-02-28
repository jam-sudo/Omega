# Omega ML-Drug-Sim — Data Contracts

모든 계약 클래스는 `omega_pbpk.contracts` 에서 임포트합니다.

```python
from omega_pbpk.contracts import (
    DrugSpec, PatientSpec, Regimen,
    PKCurve, PKMetrics, ADMEOutput, Uncertainty,
)
```

모든 계약 클래스는 **`frozen=True` dataclass** 로, 생성 후 변경이 불가능합니다.

---

## DrugSpec

화합물의 물리화학적·약동학적 파라미터를 담는 불변 레코드입니다.

| 필드 | 타입 | 기본값 | 단위 | 설명 |
|------|------|--------|------|------|
| `name` | `str` | (필수) | — | 화합물명 |
| `smiles` | `str \| None` | `None` | — | SMILES 문자열 (없으면 `None`) |
| `mw` | `float` | `300.0` | g/mol | 분자량 |
| `logP` | `float` | `2.0` | 무차원 | 옥탄올-수 분배계수 (log) |
| `pka` | `list[float]` | `[7.0]` | — | pKa 값 목록 |
| `compound_type` | `"neutral" \| "acid" \| "base" \| "zwitterion"` | `"neutral"` | — | 이온화 유형 |
| `fup` | `float` | `0.5` | 무차원 | 혈장 비결합 분율 (0, 1] |
| `rbp` | `float` | `1.0` | 무차원 | 혈액:혈장 농도 비 (> 0) |
| `clint_hepatic_L_per_h` | `float` | `0.0` | L/h | 간 고유 청소율 (≥ 0) |
| `clint_gut_L_per_h` | `float` | `0.0` | L/h | 장 고유 청소율 (≥ 0) |
| `clr_L_per_h` | `float` | `0.0` | L/h | 신장 청소율 (≥ 0) |
| `peff` | `float` | `1.0` | cm/s × 10⁻⁴ | 장관 유효 투과도 (≥ 0) |
| `solubility_mg_mL` | `float` | `1.0` | mg/mL | 용해도 (> 0) |
| `kp` | `dict[str, float]` | `{}` | 무차원 | 조직별 분배계수 (각 값 ≥ 0) |
| `permeability_limited` | `dict[str, dict[str, float]]` | `{}` | — | 투과 제한 조직 파라미터 |
| `param_source` | `"yaml" \| "python" \| "ml_predicted" \| "measured"` | `"yaml"` | — | 파라미터 출처 |
| `prediction_confidence` | `str` | `"unknown"` | — | 예측 신뢰도 레이블 |

### validate() 검증 조건

| 조건 | 오류 메시지 예시 |
|------|----------------|
| `0 < fup <= 1` | `"fup must be in (0, 1], got 0.0"` |
| `rbp > 0` | `"rbp must be > 0, got -0.5"` |
| `mw > 0` | `"mw must be > 0, got 0.0"` |
| `clint_hepatic_L_per_h >= 0` | `"clint_hepatic_L_per_h must be >= 0, got -1.0"` |
| `clr_L_per_h >= 0` | `"clr_L_per_h must be >= 0"` |
| `peff >= 0` | `"peff must be >= 0"` |
| `solubility_mg_mL > 0` | `"solubility_mg_mL must be > 0"` |
| `kp[organ] >= 0` (전체 조직) | `"kp[liver] must be >= 0"` |

### JSON 예제

```json
{
  "name": "Caffeine",
  "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
  "mw": 194.19,
  "logP": -0.07,
  "pka": [0.5],
  "compound_type": "neutral",
  "fup": 0.65,
  "rbp": 1.0,
  "clint_hepatic_L_per_h": 1.8,
  "clint_gut_L_per_h": 0.0,
  "clr_L_per_h": 0.1,
  "peff": 2.5,
  "solubility_mg_mL": 21.7,
  "kp": {"liver": 1.5, "brain": 0.9, "muscle": 0.7},
  "param_source": "yaml",
  "prediction_confidence": "high"
}
```

### API 입출력 스키마 예제

**POST /simulate** 요청 body (DrugSpec 서브셋):

```json
{
  "drug": {
    "name": "Caffeine",
    "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "mw": 194.19,
    "fup": 0.65
  },
  "patient": {"body_weight_kg": 70.0},
  "regimen": {"dose_mg": 200.0, "route": "oral"}
}
```

---

## PatientSpec

환자 생리학적 공변량을 담는 불변 레코드입니다.

| 필드 | 타입 | 기본값 | 단위 | 설명 |
|------|------|--------|------|------|
| `body_weight_kg` | `float` | `70.0` | kg | 체중 (> 0) |
| `age_years` | `int` | `30` | years | 나이 (≥ 0) |
| `sex` | `"male" \| "female"` | `"male"` | — | 성별 |
| `height_cm` | `float` | `170.0` | cm | 신장 |
| `gfr_mL_min` | `float` | `125.0` | mL/min | 사구체 여과율 (≥ 0) |
| `cardiac_output_L_h` | `float` | `390.0` | L/h | 심박출량 (> 0) |
| `child_pugh` | `"normal" \| "A" \| "B" \| "C"` | `"normal"` | — | Child-Pugh 간 기능 점수 |
| `cyp3a4_activity` | `float` | `1.0` | 무차원 | CYP3A4 활성도 (EM 기준 상대값) |
| `cyp2d6_activity` | `float` | `1.0` | 무차원 | CYP2D6 활성도 |
| `cyp2c9_activity` | `float` | `1.0` | 무차원 | CYP2C9 활성도 |
| `hepatic_cl_factor` | `float` | `1.0` | 무차원 | 간 청소율 스케일링 인수 |
| `renal_cl_factor` | `float` | `1.0` | 무차원 | 신장 청소율 스케일링 인수 |

### validate() 검증 조건

| 조건 | 오류 메시지 예시 |
|------|----------------|
| `body_weight_kg > 0` | `"body_weight_kg must be > 0"` |
| `age_years >= 0` | `"age_years must be >= 0"` |
| `gfr_mL_min >= 0` | `"gfr_mL_min must be >= 0"` |
| `cardiac_output_L_h > 0` | `"cardiac_output_L_h must be > 0"` |

### JSON 예제

```json
{
  "body_weight_kg": 65.0,
  "age_years": 45,
  "sex": "female",
  "height_cm": 162.0,
  "gfr_mL_min": 110.0,
  "cardiac_output_L_h": 360.0,
  "child_pugh": "normal",
  "cyp3a4_activity": 0.5,
  "cyp2d6_activity": 0.0,
  "cyp2c9_activity": 1.0,
  "hepatic_cl_factor": 1.0,
  "renal_cl_factor": 0.88
}
```

---

## Regimen

투약 계획을 담는 불변 레코드입니다.

| 필드 | 타입 | 기본값 | 단위 | 설명 |
|------|------|--------|------|------|
| `dose_mg` | `float` | (필수) | mg | 1회 투여 용량 (> 0) |
| `route` | `"oral" \| "iv" \| "sc"` | `"oral"` | — | 투여 경로 |
| `interval_h` | `float` | `24.0` | h | 투여 간격 (> 0) |
| `n_doses` | `int` | `1` | — | 총 투여 횟수 (≥ 1) |

### validate() 검증 조건

| 조건 | 오류 메시지 예시 |
|------|----------------|
| `dose_mg > 0` | `"dose_mg must be > 0"` |
| `interval_h > 0` | `"interval_h must be > 0"` |
| `n_doses >= 1` | `"n_doses must be >= 1"` |

### JSON 예제

```json
{"dose_mg": 200.0, "route": "oral", "interval_h": 12.0, "n_doses": 7}
```

---

## PKCurve

ODE 시뮬레이션 결과 농도-시간 데이터를 담는 레코드입니다.

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `t` | `NDArray[float64]` | h | 시간 벡터 |
| `C_plasma` | `NDArray[float64]` | mg/L | 혈장 농도 시계열 |
| `C_tissue_dict` | `dict[str, NDArray[float64]]` | mg/L | 조직별 농도 시계열 |

### API 응답 예제

```json
{
  "t": [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
  "C_plasma": [0.0, 0.02, 0.31, 0.87, 1.42, 1.21, 0.63, 0.28, 0.04],
  "C_tissue_dict": {
    "liver": [0.0, 0.05, 0.72, 1.95, 3.11, 2.64, 1.37, 0.61, 0.09],
    "muscle": [0.0, 0.001, 0.04, 0.18, 0.51, 0.72, 0.58, 0.33, 0.06]
  }
}
```

---

## PKMetrics

PK 요약 지표를 담는 불변 레코드입니다.

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `Cmax` | `float` | mg/L | 최고 혈장 농도 |
| `Tmax` | `float` | h | Cmax 도달 시간 |
| `AUC` | `float` | mg·h/L | AUC₀₋ₜ (사다리꼴 적분) |
| `t_half` | `float` | h | 소실 반감기 |

### JSON 예제

```json
{"Cmax": 1.42, "Tmax": 1.5, "AUC": 12.8, "t_half": 5.1}
```

### API 응답 스키마 예제

**POST /simulate** 응답:

```json
{
  "pk_metrics": {
    "Cmax": 1.42,
    "Tmax": 1.5,
    "AUC": 12.8,
    "t_half": 5.1
  },
  "curve": {
    "t": [0.0, 0.5, 1.0, "..."],
    "C_plasma": [0.0, 0.31, 0.87, "..."]
  },
  "uncertainty": {
    "ci_lower": [0.0, 0.22, 0.67],
    "ci_upper": [0.0, 0.45, 1.12],
    "method": "conformal",
    "coverage": 0.9
  }
}
```

---

## ADMEOutput

ADME 예측 결과를 담는 불변 레코드입니다.

| 필드 | 타입 | 기본값 | 단위 | 설명 |
|------|------|--------|------|------|
| `Fa` | `float` | `1.0` | 무차원 | 흡수 분율 [0, 1] |
| `Fg` | `float` | `1.0` | 무차원 | 장관 생존 분율 (first-pass) |
| `Fh` | `float` | `1.0` | 무차원 | 간 생존 분율 (first-pass) |
| `CLint` | `float` | `0.0` | L/h | 고유 청소율 |
| `fu` | `float` | `0.5` | 무차원 | 비결합 분율 |
| `Vd` | `float` | `30.0` | L | 분포 용적 |
| `confidence` | `"low" \| "medium" \| "high"` | `"low"` | — | 예측 신뢰도 |

### JSON 예제

```json
{
  "Fa": 0.95,
  "Fg": 0.88,
  "Fh": 0.72,
  "CLint": 2.4,
  "fu": 0.65,
  "Vd": 38.0,
  "confidence": "medium"
}
```

---

## Uncertainty

불확실성 구간 정보를 담는 불변 레코드입니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `ci_lower` | `tuple[float, ...]` | (필수) | 신뢰하한 (시간 벡터와 동일 길이) |
| `ci_upper` | `tuple[float, ...]` | (필수) | 신뢰상한 |
| `method` | `"mc" \| "ensemble" \| "conformal" \| "quantile"` | `"mc"` | 불확실성 추정 방법 |
| `coverage` | `float` | `0.90` | 명목 커버리지 확률 (예: 0.90 = 90% CI) |

### JSON 예제

```json
{
  "ci_lower": [0.0, 0.18, 0.54, 1.01, 0.87, 0.39, 0.14],
  "ci_upper": [0.0, 0.47, 1.23, 1.98, 1.72, 0.94, 0.31],
  "method": "conformal",
  "coverage": 0.9
}
```

---

## 계약 간 관계

```
DrugSpec ──── (apply) ──── PluginBase.apply() ──── DrugSpec (업데이트)
    │
    │  +  PatientSpec  +  Regimen
    │
    └──── SimulationEngine.run() ──── SimulationResult
                                            │
                                      PKCurve · PKMetrics
                                      Uncertainty · ADMEOutput
```
