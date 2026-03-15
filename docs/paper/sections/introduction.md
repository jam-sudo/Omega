# 1. Introduction

## 1.1 The Challenge of Pharmacokinetic Prediction

Pharmacokinetics (PK) — the study of how drugs are absorbed, distributed, metabolized, and eliminated — determines whether a promising molecule becomes a viable medicine. Poor PK properties account for approximately 40% of clinical trial failures, making early PK characterization essential for efficient drug discovery (Kola & Landis, 2004). Yet obtaining PK data traditionally requires either expensive in vivo studies or labor-intensive physiologically based pharmacokinetic (PBPK) modeling, where expert pharmacokineticists manually parameterize complex models for each compound.

Commercial PBPK platforms — Simcyp (Certara), GastroPlus (Simulation Plus), and PK-Sim (Open Systems Pharmacology) — encode decades of physiological knowledge into sophisticated mechanistic models. However, these tools require extensive manual input: physicochemical properties, in vitro clearance data, permeability measurements, and protein binding assays. Setting up a single compound typically takes hours to days, requires domain expertise, and remains inaccessible to many research groups due to cost (commercial licenses exceeding $50,000/year) or expertise requirements.

The pharmaceutical industry therefore faces a fundamental tension: the compounds most in need of PK prediction — novel drug candidates in early discovery — are precisely those for which experimental data are most scarce.

## 1.2 Machine Learning Approaches to PK Prediction

Recent advances in molecular property prediction, particularly graph neural networks (GNNs) and message-passing neural networks (MPNNs), have enabled accurate prediction of absorption, distribution, metabolism, and excretion (ADME) properties directly from molecular structure (Yang et al., 2019; Swanson et al., 2024). ADMET-AI, a Chemprop-based D-MPNN ensemble, currently ranks first on the Therapeutics Data Commons (TDC) ADMET benchmark, predicting over 40 endpoints from SMILES strings alone (Swanson et al., 2024).

However, predicting individual ADME properties is fundamentally different from predicting integrated PK behavior. A drug's concentration-time profile emerges from the nonlinear interplay of absorption kinetics, tissue distribution, hepatic metabolism, and renal elimination — dynamics that cannot be captured by simple property-to-PK mappings. Pure deep learning approaches that attempt to learn PK directly from molecular structure (e.g., DeepPK, PkSolver) achieve moderate accuracy but sacrifice the mechanistic interpretability that is essential for regulatory acceptance and clinical decision-making.

## 1.3 Hybrid Neural-Mechanistic Models

A promising middle ground is the hybrid approach: using machine learning to predict mechanistically meaningful parameters (clearance, volume of distribution, permeability), then feeding these into a physics-based PBPK model to simulate PK dynamics. This preserves mechanistic interpretability — every prediction traces through named physiological parameters — while leveraging ML's ability to generalize across chemical space.

Gruber et al. (2024) demonstrated this approach at Bayer AG, combining a graph convolutional network (GCN) with a whole-body PBPK model trained end-to-end on human PK data. Their hybrid model achieved exposure fold change errors of 1.87 (oral) and 1.86 (intravenous) in healthy subjects. However, this model relies on Bayer's proprietary in vitro assay data for pre-training and remains closed-source, limiting reproducibility and accessibility.

## 1.4 Contribution

We present Omega, an open-source hybrid neural-mechanistic platform that predicts human PK directly from SMILES strings. Our key contributions are:

1. **Comparable accuracy with public data only.** Omega achieves Cmax AAFE 1.90 on 20 benchmark drugs using exclusively public data and tools, matching the Bayer proprietary model (1.87) without access to internal assay data.

2. **Sub-100ms inference.** At 73 milliseconds per compound (warm start), Omega enables high-throughput virtual screening of compound libraries — approximately 1,000x faster than manual PBPK parameterization.

3. **Multi-tier validation framework.** We validate predictions at four levels of fidelity: PK accuracy on 20 drugs (Gold), half-life on 39 drugs (Silver), ADME properties on 151 compounds (Bronze), and temporal holdout on 5 post-2022 drugs the model has never seen.

4. **Patient-specific prediction.** A pragmatic Level 3 module provides weight-adjusted and genotype-adjusted PK predictions via allometric scaling and Bayesian individual parameter estimation from sparse observations.

5. **Full reproducibility.** Omega is MIT-licensed, pip-installable, and requires a single line of code: `pipeline.simulate(SimulationRequest(smiles="...", dose_mg=100))`.

The remainder of this paper describes the architecture (Section 2), presents validation results (Section 3), discusses strengths and limitations in context of prior work (Section 4), and outlines future directions (Section 5).
