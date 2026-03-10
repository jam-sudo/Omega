# Omega PBPK — Implementation Plan v3 (Parallel Execution)

> **Quick reference:** `/CLAUDE.md` (auto-loaded every session)
> **Progress tracker:** `~/.claude/projects/-home-ubuntu-Omega/memory/plan_v3.md`
> **Date:** 2026-03-08 | **Plan version:** v3.1 (parallel branch model)

---

## 1. Vision

Omega is an **AI/ML-driven pharmacokinetic prediction platform**.
- Input: SMILES string (+ optional patient/dosing info)
- Output: Full PK profile with calibrated uncertainty
- No manual parameter entry required
- Learns from data, generalizes to unseen compounds

The ODE engine is infrastructure: training data generator, validation oracle, explainability backbone.

### Three Levels

| Level | Capability | Architecture |
|-------|-----------|-------------|
| **1** | SMILES → ML ADME → ODE → PK | ADMET-AI + ensemble → existing ODE |
| **2** | SMILES → GNN → PK params → ODE → C(t) | Hybrid neural-mechanistic (end-to-end) |
| **3** | Molecule + patient + dosing → PK | Foundation model with few-shot adaptation |

---

## 2. Parallel Branch Architecture

### Dependency Graph

```
           0A (ODE fixes)─────────────────→ A (ODE training data)──┐
           [no deps]                                                │
                                                                    ├──→ L2 (Level 2 training)──→ L3
           0B (ML infra)──→ B (Level 1 ADME)──────────────────────┤
           [no deps]    └──→ C (GNN architecture)──────────────────┘
                                                                    │
           D (Clinical data pipeline)───────────────────────────────┘
           [no deps]

           E (Phase parameter extraction)
           [no deps, independent]
```

### Branch Specifications

| Branch | Git Branch | From | Merge To | Unlocks | Duration |
|--------|-----------|------|----------|---------|----------|
| **0A** | `fix/ode-critical-bugs` | `main` | `main` | A | 1 day |
| **0B** | `feat/ml-infrastructure` | `main` | `main` | B, C | 1-2 days |
| **A** | `feat/ode-training-data` | `main` | `main` | L2 | 1-2 weeks |
| **B** | `feat/adme-ml-level1` | `main` | `main` | L2, Level 1 | 1-2 weeks |
| **C** | `feat/gnn-architecture` | `main` | `main` | L2 | 1-2 weeks |
| **D** | `feat/clinical-data-pipeline` | `main` | `main` | L3 | 2-3 weeks |
| **E** | `feat/phase-param-extraction` | `main` | `main` | — | Ongoing |
| **L2** | `feat/level2-training` | `main` | `main` | L3 | 2-3 weeks |
| **L3** | `feat/foundation-model` | `main` | `main` | — | 4-6 weeks |

### Parallel Timeline

```
Week 1:   [0A]━━━━  [0B]━━━━━━  [D starts]━━━━━━━━━━━━  [E starts]━━━━
Week 2:   [A starts]━━━━━━━━━━  [B starts]━━━━━━━━━━━━  [C starts]━━━━
Week 3:   [A]━━━━━━━━━━━━━━━━  [B]━━━━━━━━━━ Level 1!  [C]━━━━━━━━━━
Week 4:   [L2 merge + training starts]━━━━━━━━━━━━━━━━  [D]━━━━━━━━━━
Week 5:   [L2]━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Level 2!    [D]━━━━━━━━━━
Week 6-10: [L3 starts]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Level 3!
```

**Sequential timeline: ~17 weeks. Parallel timeline: ~10 weeks. Savings: ~40%.**

---

## 3. Branch Details

### Branch 0A: ODE Critical Bugs

**Purpose:** Make ODE output trustworthy for ML training data generation.

| Task | File | Action |
|------|------|--------|
| Remove negative state clipping from RHS | `src/omega_pbpk/core/body.py` | Remove `np.maximum(y, 0.0)` from ODE RHS. Add post-solve warning if any state < -1e-6. |
| Fix mass balance tolerance | `src/omega_pbpk/validation/` | Change from absolute `1e-3` to relative `dose_mg * 1e-3`. |
| Guard DDI division-by-zero | `src/omega_pbpk/core/body.py` | If `ki <= 0`, return unmodified CLint. |

**Validation:** 5-drug benchmark must pass. Mass balance ±0.5%.

---

### Branch 0B: ML Infrastructure

**Purpose:** Create clean workspace for ML development.

| Task | Action |
|------|--------|
| Create `src/omega_pbpk/ml/` | Full directory tree with placeholder files. Subdirs: data/, features/, models/adme/, models/surrogate/, models/foundation/, training/, evaluation/, registry/. |
| Add ML deps | `pyproject.toml` new extras group `ml-new`: admet-ai, chemprop, torchdiffeq, xgboost, optuna, PyTDC, torch-geometric. |
| Delete dead code | Remove `experimental/` and `docking/`. Grep for imports first. |
| Define MLADMEPredictor ABC | `src/omega_pbpk/ml/models/adme/__init__.py` — abstract base class with `predict(smiles) -> ADMEProperties`. |
| Create `tests/ml/` | Directory with conftest.py and shared fixtures. |

**Validation:** Full test suite passes (nothing broken).

---

### Branch B: Level 1 ADME ML

**Purpose:** Replace polynomial ADME predictor with SOTA ML ensemble.

**ADMET-AI property mapping (CRITICAL — get units right):**

| ADMET-AI Endpoint | Our Property | Conversion |
|-------------------|-------------|------------|
| Lipophilicity | logP | Direct |
| PPBR | fup | `fup = 1 - ppbr/100` |
| Clearance_Hepatocyte | clint_3a4 | Verify µL/min/pmol (may need IVIVE scaling) |
| Caco2_Wang | peff | Convert to ×10⁻⁴ cm/s |
| Solubility_AqSolDB | logS | Direct |
| hERG | herg_ic50_uM | Direct |
| CYP2D6_Substrate | clint_2d6 | Categorical → scaling heuristic |

**RBP (custom model):** XGBoost + Morgan FP (2048-bit) trained on `data/adme_reference.csv` (153 compounds). This is the only property not covered by ADMET-AI.

**Ensemble:** ADMET-AI (primary) + XGBoost (RBP + backup) + current polynomial (fallback).

**Exit criteria:** ADME AAFE < 3.0. End-to-end PK ≤2-fold for ≥70% of 20+ drugs.

---

### Branch A: ODE Training Data

**Purpose:** Generate large-scale training data + differentiable ODE surrogate.

| Task | Output |
|------|--------|
| Generate 50K ODE profiles | Full C(t) curves (241 timepoints) + PK metrics. HDF5/parquet. |
| Generate 100K 1-cpt profiles | Analytical PK. Microseconds each. Low-fidelity pre-training data. |
| Train differentiable surrogate | PyTorch MLP: PK params → C(t). AAFE < 1.5 vs real ODE. |

**Key design:** Differentiable surrogate enables backprop from PK loss through "ODE" to GNN parameters during Level 2 training. Real ODE used at inference.

---

### Branch C: GNN Architecture

**Purpose:** Build GNN encoder + parameter head for Level 2.

| Component | Architecture |
|-----------|-------------|
| Graph builder | SMILES → PyG Data (atom features ~30-dim, bond features ~10-dim) |
| GNN encoder | 3-layer D-MPNN, 256-dim output, attention pooling |
| Parameter head | MLP: 256 → 256 → 128 → ~20 PK params. Softplus (positive), sigmoid (bounded). |
| Physics losses | MSE + mass conservation + non-negativity + monotonic terminal + param plausibility |

**Validation:** Forward pass works. All predicted params in valid physical ranges.

---

### Branch D: Clinical Data Pipeline

**Purpose:** Collect real clinical PK data for Level 3 training. **NO DEPENDENCIES — start immediately.**

| Source | Method | Target |
|--------|--------|--------|
| PK-DB | REST API (pk-db.com/api/v1/) | ~500+ drugs, C(t) curves |
| FDA DailyMed | Download PK sections + LLM extraction | ~1000 drugs, summary PK params |
| TDC | `pip install PyTDC`, direct API | ADME benchmarks, 1K-10K per endpoint |

All data cached locally in `data/ml/` with `data/ml/README.md` documenting provenance.

---

### Branch E: Phase Parameter Extraction

**Purpose:** Extract domain knowledge from phase files into structured data. **NO DEPENDENCIES.**

Convert 549 hardcoded phase files into a `data/phases_registry.yaml` parameter table + single parameterized `TissueCompartmentSimulator`. Preserves knowledge without code duplication.

---

### Level 2 Integration (requires A + B + C merged)

Wire: `smiles_to_graph` → `MolecularEncoder` → `PKParameterHead` → `DifferentiableODESurrogate` → PK.

Training: multi-fidelity curriculum (1-cpt → 35-state → clinical).
Inference: predicted params → **real ODE** (not surrogate).

**Exit criteria:** AAFE < 2.0. Predicted params meaningful. Inference < 500ms.

---

### Level 3 Foundation (requires L2 + D merged)

Add patient encoder (64-dim) + dosing encoder (32-dim) + cross-attention fusion.
Fine-tune on real clinical data from Branch D.
Few-shot adaptation via MAML/Reptile.

**Exit criteria:** Patient covariates. Few-shot <5 obs. Generalizes to novel compounds.

---

## 4. Technology Decisions

| Decision | Choice | Why | Alternatives Rejected |
|----------|--------|-----|----------------------|
| ADME predictor | ADMET-AI (pretrained) | SOTA, #1 on TDC, pip-installable, MIT | Train from scratch (slow), XGBoost-only (less accurate) |
| Level 2 arch | Hybrid neural-mechanistic | Interpretable, data-efficient, Bayer 2024 validated | Pure Neural ODE (black-box), PINN (hard to train), Transformer+Diffusion (data-hungry) |
| Backprop through ODE | Differentiable surrogate | Practical for 35-state system | torchdiffeq adjoint (complex to wrap scipy), JAX (smaller ecosystem) |
| ML framework | PyTorch + PyG | Largest ecosystem, pretrained models | JAX (smaller), DeepChem (rigid) |
| Clinical data | PK-DB + FDA labels | Real C(t) curves, open access | ChEMBL (in vitro only), manual curation (slow) |
| Training strategy | Multi-fidelity (3-stage) | Data-efficient, each stage corrects biases | Single dataset (wastes cheap data) |
| Phase files | Don't consolidate | Doesn't block ML, high refactoring risk | Consolidate first (weeks of delay) |

---

## 5. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| ADMET-AI unit mismatch | HIGH | Assertion tests in B.1, explicit conversion layer |
| ODE bugs corrupt training data | HIGH | 0A fixes are hard prerequisite for Branch A |
| Differentiable surrogate ≠ real ODE | MEDIUM | Validate with real ODE at inference; AAFE < 1.5 |
| RBP no public model | MEDIUM | Custom XGBoost on 153+ compounds |
| PK-DB API stability | MEDIUM | Cache all data locally on first fetch |
| Branch merge conflicts | LOW | Each branch touches different files; 0A/0B merge first |

---

## 6. Cross-References

| File | Purpose | Auto-loaded? |
|------|---------|-------------|
| `/home/ubuntu/Omega/CLAUDE.md` | Project instructions, branch map, rules | Yes |
| `~/.claude/.../memory/MEMORY.md` | Session memory summary | Yes |
| `~/.claude/.../memory/plan_v3.md` | Task-level progress tracker | On demand |
| `/home/ubuntu/Omega/docs/plan.md` | This file — full reference | On demand |
| `/home/ubuntu/Omega/docs/roadmap.md` | Legacy milestones (M0-M3) | On demand |
| `/home/ubuntu/Omega/REVIEW.md` | Code review findings | On demand |
| `/home/ubuntu/Omega/AGENTS.md` | Agent team structure | On demand |

---

## 7. Published References

### Validating Our Approach
1. **Bayer (2023-2024):** GCN → 11 molecular properties → PBPK, 1.62-2.35x fold error
2. **PBPK-iPINNs (2025):** Inverse PINNs for PBPK brain models
3. **Physiologically Informed DL (Feb 2026):** Transformers + Neural ODEs, cross-species PK
4. **Hybrid Transformer-Based PBPK (2025):** Transformer + mechanistic PK

### Tools & Libraries
- ADMET-AI: https://github.com/swansonk14/admet_ai (MIT, v2.0.1)
- Chemprop: https://github.com/chemprop/chemprop (MIT, v2)
- TDC: https://tdcommons.ai/ (MIT)
- torchdiffeq: https://github.com/rtqichen/torchdiffeq (MIT)
- PK-DB: https://pk-db.com/ (open access)
- PyTorch Geometric: https://pyg.org/ (MIT)

### Data Sources
- TDC ADMET Benchmark: 22 tasks, 1K-10K compounds per endpoint
- PK-DB: ~800 clinical PK studies, open REST API
- FDA DailyMed: ~2000 drug labels with PK sections
- adme_reference.csv: 153 compounds (in-repo)
