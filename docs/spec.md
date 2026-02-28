# Omega ML-Drug-Sim — System Specification

## Architecture Overview

```
YAML / SMILES
     │
     ▼
┌─────────────┐
│  adapters/  │   yaml_loader.py · population_adapter.py
│ (입력 변환)  │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────┐
│         contracts/               │
│  DrugSpec · PatientSpec          │
│  Regimen · PKCurve · PKMetrics   │
│  ADMEOutput · Uncertainty        │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│          plugins/                │
│  PluginBase ABC                  │
│  ┌─────────────────────────────┐ │
│  │ ADMEPredictorPlugin         │ │  fup, rbp, peff, clint
│  │ HeuristicKpPlugin           │ │  kp (Poulin-Theil / R&R)
│  │ ParameterNetPlugin          │ │  clint_hepatic, clr (환자 개인화)
│  └─────────────────────────────┘ │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│          engine/                 │
│  SimulationEngine ABC            │
│  WholeBodyPBPKEngine             │
│  (35-state ODE, LSODA solver)    │
└──────┬───────────────────────────┘
       │
       ▼
  PKResult / SimulationResult
  (Cmax, Tmax, AUC, t½, plasma & tissue curves)

                 ┌──────────────────────────┐
                 │  surrogate/              │
                 │  PKSurrogate (NumPy MLP) │
                 │  data_generator.py       │
                 └──────────────────────────┘
                 (ODE 대신 빠른 근사 예측)

                 ┌──────────────────────────┐
                 │  experimental/           │
                 │  physio_sim/ (QSP)       │
                 └──────────────────────────┘
                 (격리된 QSP 프로토타입)
```

---

## Module Boundaries

| 모듈 | 책임 | 주요 파일 |
|------|------|----------|
| `contracts/` | 불변 데이터 계약 (frozen dataclass) | `drug_spec.py`, `patient_spec.py`, `simulation_io.py` |
| `adapters/` | 입력 포맷 변환 (YAML, 가상집단 CSV) | `yaml_loader.py`, `population_adapter.py` |
| `plugins/` | ML 플러그인 인터페이스 및 구현 | `base.py`, `adme_plugin.py`, `heuristic_kp.py`, `parameter_net.py` |
| `engine/` | ODE 시뮬레이션 추상 인터페이스 및 구현 | `interface.py`, `ode_engine.py` |
| `core/` | 35-state ODE 엔진 (물리 모델) | `body.py`, `organ.py`, `heuristics.py` |
| `surrogate/` | ML 서러게이트 모델 | `__init__.py` (PKSurrogate), `data_generator.py` |
| `validation/` | 파라미터 가드 & 벤치마크 | `_param_guard.py`, `benchmarks.py` |
| `experimental/` | 격리된 QSP/physio 프로토타입 코드 | `physio_sim/` |

---

## Design Principles

### 1. 하드코딩 최소화
구조적 제약(질량 보존, 비음수 농도, 기본 PBPK 생리학)은 ODE 엔진에 유지하되, 화합물-특이적 미지 파라미터(fup, Kp, CLint 등)는 ML 플러그인으로 채웁니다.

### 2. 플러그인 교체 가능
`PluginBase` ABC를 구현한 모든 클래스가 ODE 엔진과 결합 가능합니다. QSPR 모델, GNN, 실험 측정값 등 다양한 예측 소스를 교체 없이 등록할 수 있습니다.

### 3. 데이터 계약 불변
`DrugSpec`, `PatientSpec`, `Regimen` 등은 `frozen=True` dataclass로 타입 안전성과 불변성을 보장합니다. 플러그인은 항상 새 인스턴스를 반환하며 원본을 변경하지 않습니다.

### 4. 하위 호환
기존 `Drug` / `WholeBodyPBPK` 코드는 `adapters/` 래퍼를 통해 계속 동작합니다. 신규 코드는 `contracts/` → `plugins/` → `engine/` 흐름을 사용합니다.

### 5. 격리된 실험 코드
QSP, 미완성 모델 등 실험적 코드는 `experimental/` 패키지에 격리하여 핵심 경로에 영향을 주지 않습니다.

---

## Key Interfaces

### PluginBase (plugins/base.py)

```python
class PluginBase(ABC):
    name: str                        # 플러그인 식별자
    provides: frozenset[str]         # DrugSpec 필드명 집합

    def predict(self, spec: DrugSpec) -> dict[str, float | dict]: ...
    def confidence(self, spec: DrugSpec) -> float: ...    # [0, 1]
    def apply(self, spec: DrugSpec) -> DrugSpec: ...      # predict() + merge
```

### SimulationEngine (engine/interface.py)

```python
class SimulationEngine(ABC):
    def run(
        self,
        drug: DrugSpec,
        patient: PatientSpec,
        regimen: Regimen,
        t_end_h: float = 24.0,
        n_points: int = 241,
        solver_config: dict | None = None,
    ) -> SimulationResult: ...

    def pk_summary(self, result: SimulationResult) -> dict[str, float]: ...
```

---

## Extras / Installation

| Extra | 내용 | 설치 |
|-------|------|------|
| (기본) | NumPy, SciPy, Pydantic, PyYAML, Typer, Pandas | `pip install -e .` |
| `ml` | PyTorch, RDKit (GNN 플러그인) | `pip install -e ".[ml]"` |
| `api` | FastAPI, Uvicorn, httpx | `pip install -e ".[api]"` |
| `viz` | Matplotlib | `pip install -e ".[viz]"` |
| `dev` | pytest, ruff, mypy | `pip install -e ".[dev]"` |
| `all` | 전체 | `pip install -e ".[all]"` |

---

## Version

Current: `0.9.0` (pyproject.toml)
