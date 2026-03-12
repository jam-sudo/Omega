# Summary

I made the two scoped test-only changes from `ai/plan.md`. `tests/test_auth.py` now skips at
module import time when `passlib` or `jose` are not installed, and
`tests/test_novel_molecule_validation.py` now allows the broader AUC/Cmax/t_half ratio range
needed for high-Vd multi-compartment cases.

# Files Changed

- `tests/test_auth.py`
- `tests/test_novel_molecule_validation.py`
- `ai/codex_report.md`

# Verification Commands Run

- `source .venv/bin/activate && pytest tests/test_auth.py -v -q 2>&1 | tail -20` (exit 0)
- `source .venv/bin/activate && pytest tests/test_novel_molecule_validation.py::TestConsistency::test_auc_cmax_thalf_relationship -v -q 2>&1 | tail -20` (exit 0)

# Results

The auth test module collected as skipped instead of failing when optional auth dependencies were
missing. The targeted novel-molecule consistency test passed all 5 parametrized cases after the
tolerance update. One existing warning appeared during the novel-molecule test: `joblib` fell back
to serial mode because of a local permission issue.

# Risks

The wider ratio bound is intentionally permissive and could mask some future AUC/Cmax/t_half
regressions that would previously have failed this single assertion. The auth tests now skip
entirely when optional dependencies are absent, so failures in that area still require an
environment with the `auth` extras installed.

# Open Questions

None.
