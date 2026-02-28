# Omega ML-Drug-Sim Roadmap

---

## Milestone 0 — Repo Cleanup + Data Contracts (완료)

- [x] `physio_sim/` QSP 코드 → `experimental/` 격리
- [x] `contracts/` 데이터 계약 정의 (DrugSpec, PatientSpec, Regimen, PKCurve, PKMetrics, ADMEOutput, Uncertainty)
- [x] YAML 단일 소스 어댑터 (`adapters/yaml_loader.py`)
- [x] 패키징 정리 — extras 분리 (ml / api / viz / dev / all)
- [x] CI 통합 (coverage gate 70%, ruff, mypy)
- [x] Param 검증 게이트웨이 (`validation/_param_guard.py`)

---

## Milestone 1 — Plugin Interface + MVP Surrogate (완료)

- [x] `PluginBase` ABC + `SimulationEngine` ABC 정의 (`plugins/base.py`, `engine/interface.py`)
- [x] ADME 플러그인 어댑터 (`ADMEPredictorPlugin`)
- [x] Kp 플러그인 어댑터 (`HeuristicKpPlugin` — Poulin-Theil / Rodgers-Rowland)
- [x] Surrogate 18D 입력 확장 + Conformal Prediction UQ
- [ ] `tests/` 재편 + 통합 테스트 (`tests/integration/`) 커버리지 완성

---

## Milestone 2 — Personalization + Uncertainty (완료/일부 진행 중)

- [x] `VirtualPopulation` → `PatientSpec` (공변량 기반 환자 입력 표준화)
- [x] `ParameterNetPlugin` (rule-based: PGx CYP scaling + allometric BW scaling)
- [ ] Synthetic Dataset 2,000개 생성 + 서러게이트 재학습

---

## Milestone 3 — API / CLI / Docs (진행 중)

- [ ] CLI 4-verb 재구조화 (`simulate`, `predict`, `population`, `report`)
- [ ] FastAPI 신규 엔드포인트 (`/simulate`, `/validate`, `/uncertainty`)
- [ ] 핵심 문서 완성
  - [x] `docs/spec.md` — 아키텍처 및 모듈 경계
  - [x] `docs/data_contracts.md` — 전체 계약 필드 테이블
  - [x] `docs/plugins.md` — 커스텀 플러그인 개발 가이드
  - [x] `docs/roadmap.md` — M0~M3 반영 전면 개정

---

## Post-M3 — GNN / UDE (계획)

- GNN ADME predictor (ChEMBL/internal 학습 데이터 확보 후)
- UDE (Universal Differential Equations) ResidualCorrector
- Coverage gate 80% 달성
- Bayesian parameter estimation against clinical PK data (공식 V&V 보고서)
- Docker 컨테이너화 + PyPI 배포
