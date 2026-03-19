# Plan Final Synthesis: 임상 우선 + 현대화된 도구 스택

> **Created:** 2026-03-18
> **Supersedes:** plan-inno.md (v0.9 session) + plan-inno1.md (clinical first)
> **Method:** 양 플랜 비판적 종합 + 실제 설치 현황 기반 도구 매트릭스
> **Goal:** 진정한 all-clinical AAFE < 2.2 (stretch < 2.0), External AAFE < 2.5

---

## 1. 비판적 분석: 양 플랜의 결함

### plan-inno.md (v0.9) — 결함 목록

| # | 결함 | 심각도 | 근거 |
|---|------|--------|------|
| F1 | **순환 벤치마크**: AAFE 1.513은 13개 임상 + 11개 합성 혼합 측정 | **CRITICAL** | 11개 약물 = Omega 1-cpt 자체 생성 CSV |
| F2 | **잘못된 수정 순서**: CLint_gut → Fluconazole Vd → ML 재훈련으로 우선순위 설정 | HIGH | pKa 통합이 더 근본적 구조 수정 (CLAUDE.md 구조적 문제 #1) |
| F3 | **"수렴 선언" 시기상조**: 순환 목적함수에서 수렴, 실제 임상 목표 아님 | HIGH | all-clinical AAFE 미측정 |
| F4 | **CLint/VDss 약물별 앵커**: midazolam/warfarin 직접 앵커 = 하드코딩에 준함 | MEDIUM | Key Decision 5 위반 경계 |
| F5 | **Chemprop v2 미활용**: v1.6.1 설치돼 있으나 XGBoost 고수 | MEDIUM | TDC 벤치마크에서 v2가 15-30% 우세 |

### plan-inno1.md (Clinical First) — 결함 목록

| # | 결함 | 심각도 | 근거 |
|---|------|--------|------|
| F6 | **pKa 통합 누락**: Phase C 후보에도 없음 | **CRITICAL** | CLAUDE.md: "가장 큰 구조적 문제" — pka_predictor.py 존재하나 연결 안 됨 |
| F7 | **LOO CV on gold-24**: 24개 샘플 LOO는 분산 허용 불가 | HIGH | 올바른 방법: expanded set(127약물) 훈련 + gold-24 holdout |
| F8 | **구식 적용도 도메인**: Tanimoto similarity gating = 2010년대 방법 | MEDIUM | MAPIE conformal + Chemprop v2 내장 불확실성 사용 |
| F9 | **Sobol 중복**: B.2 유한차분 민감도 = 기존 Sobol 인프라와 중복 | LOW | `scripts/sensitivity_per_drug.py` 대신 기존 Sobol 활용 |
| F10 | **설치된 도구 미활용**: torchdiffeq 0.2.5, optuna 4.7.0 이미 존재 | LOW | 즉시 활용 가능 |

### 두 플랜이 모두 놓친 것

1. **pKa가 FIRST 구조 수정**: 모든 이온화 약물의 Kp에 영향 (R&R/Berezhkovskiy). propranolol(pKa=9.5), diazepam, verapamil, metoprolol, atenolol 등이 잘못된 조직 분배 계수 사용 중
2. **fup-pKa 의존성**: fup 재보정이 왜 위험한지(Key Decision 18)는 pKa가 없어 fup 예측 환경이 불안정하기 때문 → pKa 먼저
3. **Chemprop v2**: 이미 v1.6.1 설치됨 → v2(2.x)로 업그레이드만 하면 됨
4. **torchdiffeq**: 이미 설치됨 → 인구 시뮬레이션 가속화 즉시 가능

---

## 2. 현재 설치 현황 (venv 기준 — 실제 확인)

### 이미 사용 가능 (즉시 활용)

| 도구 | 버전 | 현재 활용도 | 권장 활용 |
|------|------|------------|---------|
| **chemprop** | 1.6.1 (v1) | 미사용 | v2(2.x)로 업그레이드 후 CLint/fup/VDss 교체 |
| **torchdiffeq** | 0.2.5 | 미사용 | Population sim ODE 가속화 (N=1000+) |
| **optuna** | 4.7.0 | 미사용 | ADME 모델 하이퍼파라미터 자동 최적화 |
| **torch** | 2.5.0 | 간접 사용 | Chemprop v2 백엔드 |
| **torch-geometric** | 2.7.0 | 미사용 | 필요 시 분자 GNN 기반 |
| **admet_ai** | 1.4.0 | 비활성화 | 선택적 속성만 활성화 재평가 |
| **rdkit** | 2023.9.6 | 사용 중 | 유지 |
| **xgboost** | 3.2.0 | 사용 중 | Chemprop v2와 앙상블 또는 교체 |
| **sklearn** | 1.7.2 | 사용 중 | MAPIE 기반으로 conformal 교체 |
| **PyTDC** | 1.1.15 | 훈련 데이터 | Chemprop v2 훈련 데이터 소스 |

### 신규 설치 필요 (무료, 검증된 라이브러리)

| 도구 | 라이선스 | 설치 명령 | 용도 |
|------|---------|---------|------|
| **dimorphite-dl** | Apache 2.0 | `pip install dimorphite-dl` | pKa + 이온화 상태 예측 — Phase 1 핵심 |
| **MAPIE** | BSD-3 | `pip install mapie` | 엄밀한 conformal 예측 (현재 커스텀 대체) |
| **SHAP** | MIT | `pip install shap` | ADME 모델 오류 진단 자동화 |
| **chemprop v2** | MIT | `pip install "chemprop>=2.0"` | v1 → v2 업그레이드 |
| **polars** | MIT | `pip install polars` | pandas 대체, 10-50x 빠른 벤치마크 데이터 처리 |
| **wandb** | MIT (free tier) | `pip install wandb` | 실험 추적 (MEMORY.md 수동 업데이트 보완) |
| **Europe PMC client** | Apache 2.0 | `pip install europepmc` | 임상 PK 논문 자동 검색 |

### 평가 후 선택 (현재는 보류)

| 도구 | 이유 |
|------|------|
| JAX/diffrax | torchdiffeq 이미 설치됨 — 중복 |
| pharmpy/mrgsolve | ODE 엔진 교체는 현재 범위 밖 |
| ChemBERTa-2 | Chemprop v2 CYP 분류 성능 확인 후 결정 |
| Uni-Mol | GPU 환경 필요, 현재 CPU 우선 |

---

## 3. 임상 데이터 현황 — 11개 약물 미해결

**현재**: 13/24 약물에 임상 참조값
**미해결 11개**: metoprolol, propranolol, d_amphetamine, atorvastatin, carbamazepine, diazepam, digoxin, fluoxetine, nifedipine, phenytoin, gabapentin

> **MEMORY.md 재해석**: "Remaining 9 synthetic refs confirmed OK: don't change"는
> *합성 CSV 자체는 정확하다는 뜻이 아님* — "1-cpt 모델 파라미터 내에서 자기 일관성이 있다"는 뜻.
> plan-inno1이 옳음: 이는 순환 참조 → 임상 데이터로 교체해야 진정한 벤치마크

| 약물 | 권장 임상 소스 | 복잡도 | 비고 |
|------|------------|-------|------|
| metoprolol | FDA label NDA 019962 | 낮음 | 100mg 단회 경구 |
| propranolol | FDA label NDA 016418 | 낮음 | 80mg 단회 경구 |
| d_amphetamine | FDA label ANDA 040425 | 낮음 | 10mg 단회 경구 |
| atorvastatin | FDA label NDA 020702 | 낮음 | 40mg 단회 경구 |
| carbamazepine | PK-DB or FDA label | 중간 | 자가유도 주의, 단회 투여 필요 |
| diazepam | PK-DB (Bertilsson 1973 등) | 낮음 | 10mg 경구 |
| digoxin | FDA label NDA 009460 | 낮음 | 0.5mg 경구 (서방형 아님) |
| fluoxetine | PK-DB | 높음 | 활성 대사체(norfluoxetine), 단회 단순화 |
| nifedipine | FDA label NDA 018276 | 중간 | 서방형 vs 속방형 구분 필요 |
| phenytoin | FDA label + PK-DB | 높음 | 비선형 PK, 300mg 단회 |
| gabapentin | FDA label NDA 021435 | 낮음 | 300mg 단회 경구 |

---

## 4. 올바른 수정 순서 (결정적 시퀀스)

```
Phase 0: 임상 데이터 완성 (11개 약물)
    └─ 진정한 all-clinical 기준선 측정
           │
           ▼
Phase 1: pKa 통합 (dimorphite-dl)
    └─ 이온화 의존 Kp 활성화 → 모든 이온화 약물 영향
    └─ fup-logD 상관관계 변화 → Phase 2 기반 마련
           │
           ▼
Phase 2: ADME 모델 현대화 (Chemprop v2 + Optuna)
    └─ CLint/fup/VDss: XGBoost → Chemprop v2
    └─ Optuna 자동 하이퍼파라미터 최적화
    └─ fup 재보정 안전성 재평가 (pKa 수정 후)
           │
           ▼
Phase 3: CLint_gut 아키텍처 수정 (fup 수정 후 가능)
    └─ 독립적 장벽 CLint 공식 (hepatic IVIVE에서 분리)
    └─ midazolam 3.99x → ~2x 목표
           │
           ▼
Phase 4: ML Corrections v2 (MAPIE + 임상 목표)
    └─ 임상 Cmax 대상 Pre-ODE/Post-ODE 재훈련
    └─ MAPIE conformal UQ (현재 커스텀 구현 교체)
    └─ 적용도 도메인: Chemprop v2 latent space 거리
           │
           ▼
Phase 5: 파이프라인 가속화 + 실험 추적
    └─ torchdiffeq: Population sim 가속화
    └─ wandb: 실험 이력 자동화
           │
           ▼
Phase 6: 외부 검증 + 논문 최종
    └─ External AAFE < 2.5 목표
```

**핵심 순서 원칙:**
1. pKa가 fup보다 먼저 (fup 예측은 logD에 의존 → pKa 없으면 logD 불안정)
2. fup가 CLint_gut보다 먼저 (K=1.7은 fup 환경에 맞춰 경험적 보정됨)
3. 구조적 수정이 ML 재훈련보다 먼저 (ML이 보정해야 할 잔차를 줄여야 함)

---

## Phase 0: 임상 데이터 완성

**목표**: 24/24 (최소 20/24) 약물 임상 Cmax 확보

### Task 0.1: 자동화 수집 스크립트

```python
# scripts/collect_clinical_cmax.py
# 소스 우선순위:
# 1. PK-DB REST API v2: https://pkdb.pk-db.com/api/v1/
# 2. FDA API: https://api.fda.gov/drug/label.json
# 3. DailyMed API: https://dailymed.nlm.nih.gov/dailymed/services/
# 4. Europe PMC REST API (pip install europepmc)
```

- [ ] **Step 1**: `europepmc` + `requests` 활용한 자동 PK 데이터 수집 스크립트 작성
- [ ] **Step 2**: FDA API `application_number` 기반 레이블 Cmax 추출
- [ ] **Step 3**: 자동 수집 결과 검토 + 수동 보완 (fluoxetine, phenytoin 특히 주의)
- [ ] **Step 4**: `data/clinical/gold24_reference_cmax.json` 업데이트
- [ ] **Step 5**: 각 항목에 `data_quality` (fda_label_exact/clinical_exact/clinical_dose_normalized/synthetic) 기록
- [ ] **Step 6**: 커밋

### Task 0.2: 진정한 기준선 측정

- [ ] **Step 1**: `python scripts/run_full_benchmark.py` (all-clinical 참조값)
- [ ] **Step 2**: 결과를 `results/clinical_baseline_2026-03-18.json`에 저장
- [ ] **Step 3**: MEMORY.md 업데이트 (honest all-clinical AAFE 기록)
- [ ] **Step 4**: 커밋

**수용 기준**: ≥20/24 임상 데이터. 미확보 약물은 "synthetic" 플래그 유지, 집계에서 분리.

---

## Phase 1: pKa 통합 (가장 중요한 구조적 수정)

**근거**: CLAUDE.md 최대 구조적 문제. `pka_predictor.py` 존재하나 파이프라인에 연결 안 됨.
현재: 모든 약물 중성 취급 → 이온화 의존 Kp(R&R) 비활성화.

**영향 약물 (pKa가 잘못된 Kp 유발):**
- propranolol (pKa 9.5 염기): 폐/신장 Kp 크게 오류
- metoprolol (pKa 9.7 염기): 유사 오류
- verapamil (pKa 8.9 염기): 확인 필요
- atenolol (pKa 9.6 염기): 친수성 + 염기 이중 문제
- diazepam (pKa 3.4 염기): 영향 낮음
- 산성 약물: D-fix(Phase 3a.2)로 이미 부분 수정됨

### Tool: dimorphite-dl (Apache 2.0)

```bash
pip install dimorphite-dl
```

```python
from dimorphite_dl import DimorphiteDL
dl = DimorphiteDL(min_ph=7.0, max_ph=7.4, max_variants=2)
variants = dl.protonate(smiles)  # 생리적 pH에서 주 이온화 상태 반환
```

### Task 1.1: dimorphite-dl 설치 + pka_predictor.py 연동

- [ ] **Step 1**: `pip install dimorphite-dl` (venv)
- [ ] **Step 2**: `src/omega_pbpk/ml/models/adme/pka_predictor.py` 읽기 → 현재 구현 파악
- [ ] **Step 3**: dimorphite-dl 백엔드 연결 또는 교체 (현재 RDKit 기반 pKa vs dimorphite-dl 정확도 비교)
- [ ] **Step 4**: `_build_drug()` 경로에 pKa 주입 인터페이스 설계

### Task 1.2: Berezhkovskiy/R&R Kp 이온화 수정

- [ ] **Step 1**: `src/omega_pbpk/pipeline/heuristics.py` 읽기 → `berezhkovskiy_kp()` 확인
- [ ] **Step 2**: 염기 D-fix 구현
  - 현재: acids에만 D-fix 적용 (Phase 3a.2)
  - 추가: bases에도 `logD = logP - log(1 + 10^(pKa - pH))` 적용 (단, R&R phospholipid term은 별도 — 검증 필요)
- [ ] **Step 3**: compound_type='base' 전용 테스트: propranolol Kp 오류 감소 확인
- [ ] **Step 4**: 벤치마크 + 오류 상쇄 모니터링 (`run_measured_ablation.py`)
- [ ] **Step 5**: 수용 기준 통과 시 커밋. 실패 시 acids-only D-fix 유지.

**수용 기준**: AAFE ↓ OR 변화 없음 (염기 약물 개선, 산성 약물 회귀 없음)

---

## Phase 2: ADME 모델 현대화

### Tool: Chemprop v2 (MIT) — v1.6.1에서 업그레이드

**왜 v2인가:**
- 더 나은 불확실성 추정 (evidential regression 내장)
- Multi-task 학습: CLint + fup + VDss 동시 훈련 가능
- TDC ADME 벤치마크에서 v1 대비 ~15-25% RMSE 개선
- PyTDC 1.1.15 (이미 설치됨)와 완전 호환

```bash
pip install "chemprop>=2.0"  # pyproject.toml ml-new에 이미 명시됨
```

**API 변화 (v1 → v2):**
```python
# v1 (현재)
from chemprop.models import MoleculeModel
# v2 (대상)
from chemprop.models import MPNN
from chemprop.data import MoleculeDataset, MoleculeDataLoader
from chemprop.nn import RegressionFFN, BondMessagePassing
```

### Tool: Optuna (MIT) — 이미 설치됨, 미활용

```python
# 즉시 사용 가능
import optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, pruner=optuna.pruners.MedianPruner())
```

### Task 2.1: Chemprop v2 업그레이드

- [ ] **Step 1**: `pip install "chemprop>=2.0"` (호환성 테스트)
- [ ] **Step 2**: `src/omega_pbpk/ml/models/adme/xgboost_clint.py` 읽기 → Chemprop v2 병행 모델 작성
- [ ] **Step 3**: TDC CLint 데이터 + 기존 18 앵커로 Chemprop v2 CLint 훈련
- [ ] **Step 4**: Gold-24 holdout LOOCV: Chemprop v2 vs XGBoost RMSE 비교
- [ ] **Step 5**: 더 나은 쪽 선택 OR 앙상블 (Chemprop v2 AAFE + XGBoost AAFE 비교)

### Task 2.2: fup 모델 재평가 (pKa 통합 후)

- [ ] **Step 1**: pKa 통합 후 XGBoost fup 예측값 변화 확인
- [ ] **Step 2**: Chemprop v2 fup 모델 훈련 (TDC + pKa-aware features)
- [ ] **Step 3**: isotonic regression 재평가 (Key Decision 18 조건 변화 확인)
  - pKa 수정이 fup 예측 환경 변화 → 이전에 악화 +0.088이었던 것이 개선될 수 있음
- [ ] **Step 4**: 수용 기준 통과 시 업데이트

### Task 2.3: Optuna 하이퍼파라미터 최적화

- [ ] **Step 1**: `scripts/tune_adme_models.py` 작성
  - Optuna study: XGBoost + Chemprop v2 모두 포함
  - Objective: Gold-24 holdout AAFE (LOOCV)
  - n_trials=100, TPESampler + MedianPruner
- [ ] **Step 2**: CLint / fup / VDss 각각 최적화
- [ ] **Step 3**: 최적 하이퍼파라미터로 재훈련 + 커밋

### Task 2.4: SHAP 기반 오류 진단 통합

```bash
pip install shap
```

- [ ] **Step 1**: `scripts/diagnose_drug_errors.py` 작성 (plan-inno1 B.1 대체)
  - `shap.TreeExplainer(xgb_model)` → per-drug SHAP values
  - 각 약물의 Cmax 오류 원인 자동 분류: fup / CLint / VDss / 구조적
- [ ] **Step 2**: SHAP 진단 결과로 Phase 3 fix 우선순위 결정

---

## Phase 3: CLint_gut 아키텍처 수정 (fup 수정 완료 후)

**전제 조건**: Phase 2 fup 재보정 완료. K=1.7은 Phase 2 결과에 따라 재평가.

**현재 버그 (Key Decision 17):**
```python
# 현재 — 잘못된 공식
gut_clint = clint_L_per_h * 1.7
# clint_L_per_h는 역산된 hepatic CL (22-223× CLh_target)
# midazolam: 991 L/h × 1.7 = 1681 L/h >> QH_gut = 90 L/h → Fg ≈ 0 (실측 0.44)

# 올바른 공식 (Pang & Rowland 1977)
# Fg_CL_int = clint_3a4 [µL/min/pmol] × intestinal_cyp3a4_content [pmol]
# intestinal CYP3A4: 70 pmol/mg × 18g × 40mg/g ≈ 50,400 pmol
# Fg = exp(-Fg_CL_int / Q_gut) 또는 Fg = Q_gut / (Q_gut + Fg_CL_int_scaled)
```

### Task 3.1: 독립적 장벽 CLint 공식 구현

- [ ] **Step 1**: `src/omega_pbpk/pipeline/__init__.py` — `_gut_wall_extraction()` 또는 관련 함수 위치 확인
- [ ] **Step 2**: 새 공식 구현:
  ```python
  # intestinal CYP3A4 함량 기반 독립 Fg 산출
  intestinal_cyp3a4_pmol = 50400  # 70 pmol/mg × 18g × 40mg/g
  fg_clint_uL_per_min = clint_3a4 * intestinal_cyp3a4_pmol * fup_gut
  fg_clint_L_per_h = fg_clint_uL_per_min * 60 / 1e6
  Fg = q_gut_L_per_h / (q_gut_L_per_h + fg_clint_L_per_h)
  ```
- [ ] **Step 3**: midazolam 테스트: Fg 예측값 vs 실측 0.44 비교
- [ ] **Step 4**: 전체 벤치마크 + 오류 상쇄 모니터링 (필수)
- [ ] **Step 5**: 수용 기준 통과 시 커밋 (AAFE ↓, ≤2 회귀, 새 >3x 없음)

### Task 3.2: torchdiffeq 활용 (이미 설치됨)

- [ ] **Step 1**: 현재 scipy odeint vs torchdiffeq 속도 벤치마크 (단일 약물)
- [ ] **Step 2**: Population simulation (N=1000+) 가속화:
  ```python
  from torchdiffeq import odeint
  # batched ODE: [N_subjects, N_states] → GPU 병렬
  ```
- [ ] **Step 3**: Population sim 속도 10x+ 개선 확인 시 `run_population_sim.py` 업데이트

---

## Phase 4: ML Corrections v2

**전제**: Phase 0 (임상 데이터 24/24), Phase 1-3 (구조적 수정) 완료 후.

### Tool: MAPIE (BSD-3)

```bash
pip install mapie
```

```python
from mapie.regression import MapieRegressor
from mapie.conformity_scores import AbsoluteConformityScore

mapie = MapieRegressor(
    estimator=base_regressor,
    method="plus",  # cross-conformal (더 효율적)
    cv=10,
)
mapie.fit(X_train, y_train)
y_pred, y_pi = mapie.predict(X_test, alpha=0.1)  # 90% PI
```

**현재 vs MAPIE:**
- 현재 `adaptive_conformal.py`: 커스텀 구현, 커버리지 보장 불완전
- MAPIE "plus" method: PAC-conformal 보장, sklearn 완전 호환

### Task 4.1: Pre-ODE Corrector 재훈련 (임상 목표)

- [ ] **Step 1**: 훈련 목표 재설정 (합성 CSV 제거 → 임상 Cmax 사용)
- [ ] **Step 2**: 훈련 세트: expanded 127-drug + Phase 0 신규 임상 데이터 (~140개)
- [ ] **Step 3**: **검증**: gold-24 holdout (LOO 아님 — 24개 표본 LOO는 분산 과다)
- [ ] **Step 4**: Chemprop v2 기반 Pre-ODE corrector (ADME feature 공유)
- [ ] **Step 5**: Optuna로 max_depth/n_estimators 최적화
- [ ] **Step 6**: Gold-24 holdout AAFE 기록 → 기준선 대비 개선 확인

### Task 4.2: Post-ODE Corrector 재훈련

- [ ] **Step 1**: 잔차 목표 재설정: log(clinical_cmax / ode_pred_cmax)
- [ ] **Step 2**: Ridge + Chemprop v2 스태킹 (XGBoost 대체 또는 병행)
- [ ] **Step 3**: Gold-24 holdout 검증

### Task 4.3: MAPIE conformal UQ 교체

- [ ] **Step 1**: `src/omega_pbpk/ml/corrections/adaptive_conformal.py` 읽기
- [ ] **Step 2**: MAPIE `MapieRegressor` 적용 (현재 커스텀 로직 교체)
- [ ] **Step 3**: 90% PI 실제 커버리지 측정 (목표: 88-92%)
- [ ] **Step 4**: 현재 커버리지 vs MAPIE 비교 리포트

### Task 4.4: 적용도 도메인 gating (개선)

```python
# plan-inno1 D.3 개선: Tanimoto → Chemprop v2 latent space 거리
# Chemprop v2: forward() 중간 레이어 임베딩 추출 가능
emb_train = model.get_embeddings(X_train)
emb_query = model.get_embeddings(X_query)
dist = cosine_distance(emb_query, emb_train.min(axis=0))
# dist 크면 외삽 → MAPIE conformal interval 자동으로 넓어짐 (추가 gating 불필요)
```

- [ ] **Step 1**: Chemprop v2 임베딩 기반 적용도 평가 구현
- [ ] **Step 2**: 외삽 약물(external 8개)에 대한 불확실성 자동 확대 확인

---

## Phase 5: 파이프라인 가속화 + 실험 추적

### Task 5.1: wandb 실험 추적 (무료 tier)

```bash
pip install wandb && wandb login  # free account
```

- [ ] **Step 1**: `scripts/run_full_benchmark.py`에 wandb logging 추가
  ```python
  wandb.log({"aafe": aafe, "pct_2fold": pct_2fold, **per_drug_fe})
  ```
- [ ] **Step 2**: 모든 ADME 훈련 스크립트에 wandb 추가 (run마다 자동 기록)
- [ ] **Step 3**: MEMORY.md 수동 업데이트 → wandb 대시보드로 보완

### Task 5.2: polars 도입 (대용량 데이터 처리)

```bash
pip install polars
```

- [ ] 벤치마크/데이터 처리 스크립트에서 pandas 병목 구간 polars로 교체
- [ ] 주요 효과: expanded 127-drug 데이터셋 로딩 속도 개선

---

## Phase 6: 외부 검증 + 논문 최종

**현재 외부 AAFE: 2.95 (8개 약물) → 목표 < 2.5**

### Task 6.1: 외부 검증 세트 확장

- [ ] Europe PMC API로 추가 임상 PK 데이터 수집 (목표 20개)
- [ ] 기준: SMILES 사용 가능 + 단회 경구 + 건강인 + 시뮬레이션 범위 내 약물
- [ ] External AAFE < 2.5 달성 후 논문 최종 업데이트

### Task 6.2: 논문 업데이트 (결과에 따라)

- 모든 수치에 bootstrap CI (이미 구현됨)
- 방법론 섹션: dimorphite-dl, Chemprop v2, MAPIE conformal 기술
- 외부 AAFE < 2.5 시 "outperforms Bayer" claim 유지

---

## 5. 불변 원칙 (양 플랜 통합 + 강화)

1. **임상 우선**: 합성 참조값 대상 최적화 또는 주장 금지
2. **오류 상쇄 모니터링**: 모든 ADME/구조 수정 후 `run_measured_ablation.py` 실행 필수
3. **수용 기준 (Phase C/1/2/3)**: AAFE ↓ AND ≤2개 약물 회귀 AND 새 >3x 오류 없음
4. **pKa 우선**: fup 재보정 및 CLint_gut 수정은 반드시 pKa 통합 후
5. **K=1.7 잠금**: Phase 2 fup 수정 결과 확인 전 CLint_gut K 변경 금지
6. **약물별 파이프라인 하드코딩 금지**: CLint/VDss 앵커는 ML 훈련 데이터로만
7. **임계값 동결**: `_GUT_WALL_CLint3A4_THRESHOLD=2.6`, `VDss threshold=4.5`
8. **Ridge는 dead code**: 건드리지 않음
9. **외부 검증이 진정한 성능**: in-sample AAFE는 진행 모니터링용, 성과 지표 아님
10. **한 번에 하나씩**: 각 수정 후 반드시 벤치마크 실행 (복합 수정 금지)

---

## 6. 성공 기준

| 지표 | 현재 (부분 순환) | 목표 | 스트레치 |
|------|---------------|------|---------|
| 임상 커버리지 | 13/24 (54%) | ≥20/24 (83%) | 24/24 (100%) |
| Cmax AAFE (all-clinical) | **미측정** | < 2.2 | < 2.0 |
| %2-fold (all-clinical) | **미측정** | ≥ 50% | ≥ 65% |
| External AAFE (8 drugs) | 2.95 | < 2.5 | < 2.0 |
| >5x 오류 | 미측정 | 0개 | 0개 |
| pKa 활성화 | 미적용 | 모든 약물 pKa 예측 | — |

**참조**: Simcyp/GastroPlus AAFE ~1.5-2.0은 *측정된* ADME 입력 사용.
우리는 SMILES만으로 ADME 예측 → 2.0-2.5가 현실적 목표.
External AAFE < 2.0은 업계 최고 수준.

---

## 7. 도구 설치 명령 (순서대로)

```bash
# Phase 0-1 전 (즉시)
source .venv/bin/activate
pip install dimorphite-dl europepmc shap mapie polars wandb

# Phase 2 전 (Chemprop v2 업그레이드)
pip install "chemprop>=2.0"
# 주의: v1 API와 완전히 다름 — 기존 chemprop 코드 재작성 필요
# (v1.6.1은 이미 설치됨, v2로 replace됨)

# torchdiffeq, optuna는 이미 설치됨 — 추가 설치 불필요
```

```toml
# pyproject.toml ml-new 그룹 업데이트
[project.optional-dependencies]
ml-new = [
    "chemprop>=2.0",       # v1.6.1 → v2.x
    "dimorphite-dl>=1.3",  # NEW — pKa
    "mapie>=0.8",          # NEW — conformal UQ
    "shap>=0.44",          # NEW — 오류 진단
    "polars>=0.20",        # NEW — 빠른 데이터 처리
    "wandb>=0.17",         # NEW — 실험 추적
    "europepmc>=1.1",      # NEW — 임상 문헌 API
    "torchdiffeq>=0.2",    # 이미 있음
    "optuna>=3.0",         # 이미 있음 (4.7.0)
    "torch>=2.0",          # 이미 있음 (2.5.0)
    "xgboost>=2.0",        # 이미 있음 (3.2.0)
    "PyTDC>=1.0",          # 이미 있음 (1.1.15)
]
```

---

## 8. 이전 플랜과의 관계

| 기존 계획 | 처리 |
|---------|------|
| plan-inno Phase 3b Priority 1 (CLint_gut) | **흡수** → Phase 3 (순서 조정: pKa 후) |
| plan-inno Phase 3b Priority 2 (Fluconazole Vd) | **Phase 3 후 재평가** (pKa+CLint_gut 수정 후 자연 해결 가능) |
| plan-inno Phase 3b Priority 3 (ML retrain) | **흡수** → Phase 4 (임상 목표로 개선) |
| plan-inno1 Phase A (임상 데이터) | **채택** → Phase 0 |
| plan-inno1 Phase B (진단) | **개선** → SHAP 기반 자동화 (Task 2.4) |
| plan-inno1 Phase C (구조 수정) | **흡수** → Phase 1-3 (pKa 우선 추가) |
| plan-inno1 Phase D (ML corrections) | **채택 + 개선** → Phase 4 (MAPIE + Chemprop v2) |
| ML-Hybrid Phase 3: Multi-task ADME | **Chemprop v2 multi-task으로 구현** |
| ML-Hybrid Phase 3: BCS classification | **Phase 3 후 재평가** |

---

## 자가피드백 수렴 로그

| 라운드 | 변경 | 상태 |
|-------|------|------|
| 1 | plan-inno 분석: F1-F5 결함 식별 | 결함 목록 완성 |
| 2 | plan-inno1 분석: F6-F10 결함 식별 | 결함 목록 완성 |
| 3 | 실제 설치 현황 확인 (chemprop 1.6.1, torchdiffeq, optuna 설치됨) | 도구 매트릭스 구체화 |
| 4 | 임상 데이터 현황 확인 (13/24, 11개 synthetic 특정) | Phase 0 구체화 |
| 5 | pKa가 누락된 가장 중요한 수정임 확인 → Phase 1로 격상 | 순서 확정 |
| 6 | 전체 시퀀스 검토: 종속성 오류 없음, 모든 gap 해결됨 | **CONVERGED** |
