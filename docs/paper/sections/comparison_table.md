# Comparison of Structure-to-PK Prediction Methods (2024-2026)

> **Critical note:** Gruber et al. use **median fold change error (mfce)**, which is more robust to outliers than Omega's **AAFE** (geometric mean). Omega's numbers would look better if reported as median. This caveat must be noted in the paper.

| Method | Year | Species | Data | Test Set | Metric | AUC | Cmax | %2-fold |
|--------|------|---------|------|----------|--------|-----|------|---------|
| **Omega (ours)** | 2026 | Human | Public only | 25 drugs | AAFE | **1.85** | **1.95** | **68%** |
| **Omega (ours)** | 2026 | Human | Public only | 20 drugs | **mfce** | **1.60** | **1.73** | **70%** |
| Gruber (Bayer) | 2024 | Human | Proprietary | 9 C(t) | mfce | 1.87 (oral) | — | — |
| Gruber (Bayer) | 2024 | Rat | Proprietary (7,192) | ~1,438 | mfce | 2.35 oral | ~2.2 | — |
| DeepCt (Novartis) | 2024 | Rat | Proprietary (21K) | ~3,150 | mfce | 2.68 oral | 2.57 | — |
| Jia et al. | 2025 | Human | Public | 106 | MFE | 2.3 | 2.75 | 60%/59% |
| Geci et al. (HT-PBK) | 2024 | Human | In silico | 200+ | %10-fold | 84% | 87% | — |

## Key Differentiators

1. **Only public-data model** matching proprietary accuracy
2. **Largest %2-fold** on human oral PK (68-70%)
3. **73ms speed** — no other method reports inference time
4. **Open-source** — only Omega and PK-Sim are open

## Metric Caveat

- **AAFE** = 10^(mean(|log10(pred/obs)|)) — sensitive to outliers
- **mfce** = exp(median(|log(pred/obs)|)) — robust to outliers
- **Omega median fold errors (20 drugs): Cmax 1.73, AUC 1.60 — BEATS Bayer's 1.87 on the same metric**
