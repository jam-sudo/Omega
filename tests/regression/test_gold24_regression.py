# tests/regression/test_gold24_regression.py
"""Full 24-drug gold-tier AAFE regression gate + prediction latency benchmark.

Thresholds derived from Phase 0 all-clinical baseline (2026-03-18):
  AAFE  = 1.502 [1.32, 1.74],  83% 2-fold

Alert thresholds (with buffer to prevent brittle failures from CI noise):
  AAFE_THRESHOLD   = 1.70    (current 1.502 + ~13% headroom)
  PCT_2FOLD_MIN    = 75.0    (current 83% - 8 pp headroom)
  MAX_SINGLE_FE    = 6.0     (current max midazolam ~4x)
  LATENCY_LIMIT_MS = 500     (well above observed ~73ms; guards catastrophic slowdowns)

Run:
  pytest tests/regression/test_gold24_regression.py -v -m benchmark
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # for run_l1_benchmarks import

GOLD_REF_PATH = REPO_ROOT / "data" / "clinical" / "gold24_reference_cmax.json"

# -- Thresholds ----------------------------------------------------------------
AAFE_THRESHOLD = 1.70  # max acceptable AAFE
PCT_2FOLD_MIN = 75.0  # min acceptable %2-fold
MAX_SINGLE_FE = 6.0  # max acceptable single-drug fold error
LATENCY_LIMIT_MS = 500  # max acceptable prediction latency (ms)


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline():
    from omega_pbpk.pipeline import OmegaPipeline

    return OmegaPipeline()


@pytest.fixture(scope="module")
def gold_reference() -> dict:
    assert GOLD_REF_PATH.exists(), f"Clinical reference not found: {GOLD_REF_PATH}"
    return json.loads(GOLD_REF_PATH.read_text())


@pytest.fixture(scope="module")
def benchmark_drugs() -> dict:
    """Load canonical drug SMILES/doses from run_l1_benchmarks.py."""
    from run_l1_benchmarks import BENCHMARK_DRUGS  # noqa: PLC0415

    return BENCHMARK_DRUGS


@pytest.fixture(scope="module")
def fold_errors(pipeline, gold_reference, benchmark_drugs) -> dict[str, float]:
    """Run all 24 drugs and return {name: fold_error}."""
    from omega_pbpk.pipeline import SimulationRequest

    results = {}
    for name, info in benchmark_drugs.items():
        ref = gold_reference.get(name)
        if ref is None:
            continue  # drug not in clinical reference -- skip
        obs_cmax = ref["cmax_mg_L"]
        result = pipeline.simulate(
            SimulationRequest(
                smiles=info["smiles"],
                dose_mg=info["dose_mg"],
                route="oral",
            )
        )
        pred_cmax = result.cmax_mg_L
        fe = max(pred_cmax / obs_cmax, obs_cmax / pred_cmax)
        results[name] = fe
    return results


# -- Tests ---------------------------------------------------------------------


@pytest.mark.benchmark
def test_gold24_aafe(fold_errors):
    """AAFE over all 24 gold-tier drugs must stay <= AAFE_THRESHOLD."""
    fes = list(fold_errors.values())
    assert len(fes) >= 20, f"Too few drugs evaluated: {len(fes)} (expected >=20)"
    aafe = math.exp(sum(math.log(fe) for fe in fes) / len(fes))
    assert aafe <= AAFE_THRESHOLD, (
        f"AAFE {aafe:.3f} exceeds threshold {AAFE_THRESHOLD}  (N={len(fes)} drugs)"
    )


@pytest.mark.benchmark
def test_gold24_pct_2fold(fold_errors):
    """Fraction of drugs within 2-fold must stay >= PCT_2FOLD_MIN."""
    fes = list(fold_errors.values())
    pct = 100.0 * sum(1 for fe in fes if fe <= 2.0) / len(fes)
    assert pct >= PCT_2FOLD_MIN, (
        f"%2-fold {pct:.1f}% < threshold {PCT_2FOLD_MIN}%  (N={len(fes)} drugs)"
    )


@pytest.mark.benchmark
def test_gold24_no_catastrophic_error(fold_errors):
    """No single drug fold error may exceed MAX_SINGLE_FE."""
    bad = [(name, fe) for name, fe in fold_errors.items() if fe > MAX_SINGLE_FE]
    if bad:
        detail = ", ".join(f"{n}:{fe:.1f}x" for n, fe in sorted(bad, key=lambda x: -x[1]))
        pytest.fail(f"Drugs exceeding {MAX_SINGLE_FE}x: {detail}")


@pytest.mark.benchmark
def test_gold24_per_drug_report(fold_errors, capsys):
    """Print per-drug fold errors for debugging (always passes)."""
    lines = [f"\n{'Drug':<20} {'FE':>6}  {'Status'}"]
    lines.append("-" * 40)
    for name in sorted(fold_errors):
        fe = fold_errors[name]
        status = "OK" if fe <= 2.0 else ("WARN" if fe <= MAX_SINGLE_FE else "FAIL")
        lines.append(f"{name:<20} {fe:>6.2f}x  {status}")
    fes = list(fold_errors.values())
    aafe = math.exp(sum(math.log(fe) for fe in fes) / len(fes))
    pct_2f = 100.0 * sum(1 for fe in fes if fe <= 2.0) / len(fes)
    lines.append("-" * 40)
    lines.append(f"{'AAFE':<20} {aafe:>6.3f}")
    lines.append(f"{'%2-fold':<20} {pct_2f:>5.1f}%")
    print("\n".join(lines))  # visible with pytest -s


@pytest.mark.benchmark
def test_gold24_latency(benchmark, pipeline):
    """Per-drug prediction latency must stay < LATENCY_LIMIT_MS."""
    from omega_pbpk.pipeline import SimulationRequest

    # Use midazolam: highest-load drug (gut wall + VDss correction active)
    req = SimulationRequest(
        smiles="Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2",
        dose_mg=2.0,
        route="oral",
    )
    result = benchmark(pipeline.simulate, req)
    assert result.cmax_mg_L > 0, "Simulation returned non-positive Cmax"
    # pytest-benchmark reports mean/median/stddev automatically
    # Threshold check: benchmark.stats.mean is in seconds
    mean_ms = benchmark.stats["mean"] * 1000
    assert mean_ms < LATENCY_LIMIT_MS, (
        f"Prediction latency {mean_ms:.0f}ms exceeds {LATENCY_LIMIT_MS}ms limit"
    )
