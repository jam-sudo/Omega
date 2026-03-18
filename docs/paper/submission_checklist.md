# Omega PBPK Paper — Submission Checklist

Last updated: 2026-03-17

## Target Journals

- **Primary:** CPT: Pharmacometrics & Systems Pharmacology (CPT:PSP)
  - Word limit: ~4000–5000 main text + unlimited supplementary
  - Abstract: 250 words max
  - Figures: 6 max in main text (supplementary unlimited)
  - Format: Research Article or Methods Report

- **Backup:** Journal of Pharmaceutical Sciences (J. Pharm. Sci.)
  - Word limit: ~5000–6000 main text
  - Abstract: 250 words max
  - Format: Research Article

---

## Word Count

| Section | Count | Limit (CPT:PSP) | Status |
|---------|-------|-----------------|--------|
| Abstract | 273 words | 250 words | OVER by 23 words — trim needed |
| Main text (total) | ~11,065 words | ~5,000 words | OVER — significant trimming needed |
| Supplementary | Unlimited | Unlimited | OK |

**Action required:** The main text is approximately 2x the CPT:PSP limit. Sections to trim:
- Methods: Condense CYP genotype tables (move entirely to supplementary)
- Methods: Consolidate ADME predictor subsections
- Results: Reduce per-drug narrative text, rely on tables/figures
- Discussion: Tighten Bayer comparison (currently ~600 words)

For J. Pharm. Sci. (6000-word limit), trimming ~5000 words is needed — considerable work.

---

## Figures

All 8 figures have corresponding PDF files confirmed present:

| Figure | File | Status | Caption Current? |
|--------|------|--------|-----------------|
| Fig 1: Pred vs obs Cmax scatter | `fig1_pred_vs_obs_cmax.pdf` | PDF exists | Yes (24 drugs, AAFE 1.79) |
| Fig 2: Pred vs obs AUC scatter | `fig2_pred_vs_obs_auc.pdf` | PDF exists | Yes |
| Fig 3: Fold error bars | `fig3_fold_error_bars.pdf` | PDF exists | Needs regeneration (verapamil no longer outlier) |
| Fig 4: Silver tier t-half | `fig4_silver_thalf.pdf` | PDF exists | Yes |
| Fig 5: Bronze ADME | `fig5_bronze_adme.pdf` | PDF exists | Yes |
| Fig 6: Multi-tier summary | `fig6_multi_tier_summary.pdf` | PDF exists | Updated caption (3.40 not 2.88) |
| Fig 7: Expanded validation | `fig_expanded_validation.pdf` | PDF exists | Updated caption (3.40 not 2.88) |
| Fig 8: Ablation | `fig_ablation.pdf` | PDF exists | Yes |

**Note:** Fig 3 (fold_error_bars) was generated from the old benchmark (verapamil 8.83x, warfarin 1.46x). With the 2026-03-17 results, warfarin is the largest outlier (6.73x) and verapamil is within 2-fold (1.48x). Fig 3 and Fig 1 **must be regenerated** before final submission.

Fig 6 (multi-tier summary) and Fig 7 (expanded validation) have stale embedded AAFE values (2.88 shown vs. correct 3.40). These **must be regenerated** before submission.

---

## Supplementary Materials

| File | Status | Description |
|------|--------|-------------|
| `supplementary_table_S1.tex` | Created | Per-drug gold tier Cmax results (24 drugs, 2026-03-17 data, sorted by FE) |
| `supplementary_table_S2.tex` | Created | Ablation study (8 configs, bootstrap CI, Phase 0.1 data) |
| Table S2 (in main tex) | Existing | Silver tier half-life (39 drugs) |
| Table S3 (in main tex) | Existing | Temporal holdout (20 drugs) |
| Table S4 (in main tex) | Existing | CYP genotype scaling factors |
| Table S5 (in main tex) | Updated | Benchmark drug SMILES (now 24 drugs, warfarin 10mg, methanol removed) |
| Table S6 (in main tex) | Existing | Conformal calibration report |
| Table S7 (in main tex) | Existing | Bronze ADME accuracy |
| Table S8 (in main tex) | Existing | Ablation full table |
| Table S9 (in main tex) | Existing | Error cancellation per-drug |

---

## Key Inconsistencies Fixed in This Session

1. **"Methanol" benchmark drug removed** — SMILES `CO` (methanol) was listed as a benchmark drug in Table S5 and main Table. Replaced with Metformin (CN(C)C(=N)NC(=N)N, 500mg) which is the correct entry.

2. **Warfarin dose corrected: 5mg → 10mg** — Warfarin was updated to PK-DB data (10mg single dose, 2026-03-17). All references updated including patient-specific section and Table S5.

3. **Per-drug Cmax table updated** — Main Table 1 was from the 2026-03-15 benchmark (verapamil 8.83x, phenytoin 1.03). Updated to 2026-03-17 benchmark (warfarin 6.73x, ibuprofen 5.39x, verapamil 1.48x).

4. **4 missing drugs added to benchmark table** — Atenolol (50mg), Fluconazole (200mg), Furosemide (40mg), and Gabapentin (300mg) added to Table S5.

5. **Expanded gold AAFE corrected: 2.88 → 3.40** — Abstract, results section, figure captions, and discussion all updated.

6. **25-drug reference corrected to 24** — Validation design section erroneously listed 25 drugs for the gold tier.

7. **Failure analysis rewritten** — Old analysis (verapamil 8.83x as worst, flutamide as benchmark drug) replaced with current benchmark failures: warfarin 6.73x, ibuprofen 5.39x, fluconazole 3.25x.

---

## Outstanding Issues for Next Revision

### Critical (must fix before submission)
1. **Regenerate Fig 1, Fig 3, Fig 6, Fig 7** from the 2026-03-17 benchmark data. The PDF figures still show old numbers (verapamil as outlier in Fig 3, 2.88 in Fig 6/7).
2. **Trim word count to ~5000 words** for CPT:PSP. Suggest: move CYP genotype tables entirely to supplementary, condense ADME subsections, shorten Bayer comparison.
3. **Abstract trimming**: 273 words exceeds CPT:PSP 250-word limit by 23 words.

### Important (affects credibility)
4. **Data leakage disclosure**: 34% of gold-tier drugs overlap with ADME training set — must be disclosed in limitations. Currently only mentioned in CLAUDE.md, not in the paper.
5. **Benchmark data provenance**: Clarify in Methods that benchmark CSV data is synthetically generated (1-cpt model with 20% SD), not real digitized clinical curves. The current methods section implies real digitized data ("digitized concentration-time points with standard deviations").

### Nice-to-have
6. **Author affiliations**: Currently "[Authors TBD]" / "[Affiliations TBD]" — fill before submission.
7. **Acknowledgments**: Currently "[To be added.]"
8. **Data availability statement**: Add "Code and data available at https://github.com/jam-sudo/Omega under MIT license."
9. **Author contributions statement**: e.g., "J.M.: conceptualization, methodology, software, validation, writing."
10. **External validation section**: Consider adding 8-drug external validation (AAFE 2.95, 62% 2-fold) as a paragraph — currently only mentioned in MEMORY.md, not in the paper.

---

## Author Contributions Statement Template

```
Conceptualization: [Author]; Methodology: [Author]; Software: [Author];
Validation: [Author]; Formal Analysis: [Author]; Writing -- Original Draft: [Author];
Writing -- Review & Editing: [Author]; Visualization: [Author];
Supervision: [Author]; Funding Acquisition: [Author]
```

## Data Availability Statement Template

```
The Omega codebase, trained models, benchmark datasets, and all scripts
required to reproduce the results in this paper are freely available at
https://github.com/jam-sudo/Omega under the MIT open-source license.
No proprietary data were used in this study.
```

## Competing Interests Statement Template

```
The authors declare no competing interests.
```
