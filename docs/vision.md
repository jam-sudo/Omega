# Omega PBPK -- Vision

## What Omega Is

Omega is an AI/ML-driven pharmacokinetic prediction platform.
SMILES string in, PK profile out -- powered by learned models, not manual parameterization.

The traditional PBPK workflow requires hours of literature search, measured ADME data,
and expert curation before a single simulation can run. Omega replaces that front-end
with machine learning: a molecular structure is all you need.

The mechanistic ODE engine still exists -- it generates training data, provides
explainability, and anchors predictions in physiology -- but it is infrastructure,
not the user-facing product.

## Key Differentiators

### 1. SMILES to PK (Zero-Touch Prediction)

No measured ADME data required. Predict PK for compounds before they are synthesized.
This enables virtual screening at scale: evaluate thousands of candidates on PK
properties before committing to synthesis and in vitro assays.

Commercial tools (Simcyp, GastroPlus, PK-Sim) require measured clearance, solubility,
permeability, and protein binding as inputs. Without those, they cannot run.

### 2. Differentiable Pipeline and Inverse Design

The GNN encoder, parameter head, and differentiable ODE surrogate form a fully
differentiable pipeline from molecular graph to PK profile. This enables:

- **Forward mode:** SMILES to predicted Cmax, AUC, half-life.
- **Inverse mode:** Given a target PK profile, backpropagate gradients into molecular
  feature space to identify structural modifications that achieve it.

No commercial PBPK tool supports gradient-based inverse design. This is a capability
that does not exist in the current landscape.

### 3. Sub-Second Speed, API-First

Target prediction latency is under 500 ms per compound. The interface is a Python API
and CLI, not a desktop GUI. This means:

- Screen 100K compounds in hours, not months.
- Integrate directly into LIMS, ELN, and automated screening platforms.
- Run as a microservice behind internal web tools.

Commercial tools are GUI-first and designed for single-compound, expert-driven workflows.

### 4. Multi-Fidelity ML Training

Omega uses a three-tier training strategy to maximize data efficiency:

| Tier | Data Source | Scale | Purpose |
|------|-----------|-------|---------|
| 1 | 1-compartment analytical PK | ~100K compounds | Broad chemical space coverage |
| 2 | 35-state PBPK ODE simulations | ~10K compounds | Physiological grounding |
| 3 | Clinical PK observations | Hundreds of compounds | Real-world calibration |

Transfer learning at each tier lets the model learn general pharmacokinetic structure
from cheap synthetic data, then fine-tune on scarce clinical observations.

### 5. Few-Shot Clinical Adaptation (Level 3)

The foundation model, once trained, can be adapted to a specific patient with fewer
than 5 observed concentrations. This targets therapeutic drug monitoring (TDM) scenarios
where rapid, individualized PK prediction has direct clinical value.

### 6. Open Source

Simcyp licenses cost $50-100K/year. GastroPlus is similarly priced. PK-Sim is open
source but lacks ML integration. Omega is open source with ML as a first-class citizen.

## Comparison with Commercial Tools

| Capability | Simcyp | GastroPlus | PK-Sim | Omega |
|-----------|--------|------------|--------|-------|
| SMILES to PK (no measured data) | No | No | No | Yes |
| Inverse design (gradient-based) | No | No | No | Yes |
| Sub-second prediction | No | No | No | Yes |
| ML-integrated pipeline | No | Partial | No | Yes |
| Open source | No | No | Yes | Yes |
| Regulatory acceptance | Yes | Yes | Yes | Not yet |
| Population PBPK | Yes | Yes | Yes | Level 3 (planned) |
| API/CLI native | No | No | Partial | Yes |

## Current Limitations

Honesty about where Omega stands today:

- **Regulatory acceptance:** None. Commercial tools have decades of regulatory track
  record. Omega would need prospective validation studies before regulatory use.
- **Level 2 (GNN encoder) is in development.** The current production path uses
  Level 1 (ADMET-AI + XGBoost ensemble) which meets AAFE < 3.0 but not the Level 2
  target of AAFE < 2.0.
- **Level 3 (few-shot adaptation) is architecture-complete but untrained.** Clinical
  data pipeline exists but training has not begun.
- **Benchmark coverage is limited.** Current validation uses ~20 drugs. Expanding to
  50+ is needed for credible claims.
- **Population variability modeling is not yet implemented.** Commercial tools handle
  virtual populations natively.

## Potential Future Capabilities

- **Active learning for drug programs:** Given a set of candidates, recommend which
  compound to measure next for maximum information gain.
- **Multi-objective optimization:** Jointly optimize Cmax, AUC, half-life, and safety
  margins (e.g., hERG) in molecular design space.
- **Virtual population PK trials:** Generate synthetic trial data for dose selection
  without patient enrollment, once population variability modeling is added.

## Exit Criteria

| Level | Criteria | Status |
|-------|---------|--------|
| 1 | SMILES to PK profile. ADME AAFE < 3.0. PK within 2-fold for 70%+ of 20+ drugs. | Partially met |
| 2 | Sub-500ms prediction. AAFE < 2.0. Physically meaningful predicted parameters. | In development |
| 3 | Patient covariates. Few-shot (< 5 obs). Generalizes to novel compounds. | Planned |
