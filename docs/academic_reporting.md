# Academic Reporting

Enable an academic-style report from a simulation run with `--academic-report`.

## CLI usage

```bash
python -m omega simulate \
  --compound examples/compound_caffeine.yaml \
  --subject examples/subject_default.yaml \
  --dose-mg 100 --route oral --t-end-h 24 \
  --deterministic --academic-report --sensitivity \
  --out outputs/run_academic
```

## Generated artifacts

- `report_academic.md`
- (optional future extension) `report_academic.html`

## Required sections

The report includes:

- model overview (PBPK + PD status)
- exact run command and config references
- model version metadata (`package_version`, git commit, timestamp)
- PK summary metrics and curve artifact pointers
- validation status, including benchmark summary linkage if available
- uncertainty notes (if population/uncertainty workflows were run)
- sensitivity section (if enabled)
- standard limitations section

This report is designed for publication appendix material and reproducible methods sections.
