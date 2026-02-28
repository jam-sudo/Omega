"""Regression: ODE output must not drift from golden snapshot > ±10%."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from omega_pbpk.adapters.yaml_loader import load_drug_spec_by_name
from omega_pbpk.contracts import PatientSpec, Regimen
from omega_pbpk.engine import WholeBodyPBPKEngine

GOLDEN_DIR = Path(__file__).parent.parent.parent / "benchmarks" / "golden"
DRUGS = ["caffeine", "warfarin", "midazolam", "propranolol"]
REL_TOL = 0.10  # ±10%


@pytest.fixture(scope="module")
def engine():
    return WholeBodyPBPKEngine()


@pytest.fixture(scope="module")
def patient():
    return PatientSpec()


@pytest.fixture(scope="module")
def regimen():
    return Regimen(dose_mg=100.0, route="oral")


@pytest.mark.regression
@pytest.mark.parametrize("drug_name", DRUGS)
def test_golden_snapshot(drug_name, engine, patient, regimen):
    golden_path = GOLDEN_DIR / f"{drug_name}_oral_100mg.json"
    if not golden_path.exists():
        pytest.skip(f"No golden snapshot for {drug_name}")

    with open(golden_path) as f:
        golden = json.load(f)

    spec = load_drug_spec_by_name(drug_name)
    result = engine.run(spec, patient, regimen)
    summary = engine.pk_summary(result)

    gold_cmax = golden["metrics"]["Cmax"]
    gold_auc = golden["metrics"]["AUC"]

    cmax_re = abs(summary["Cmax"] - gold_cmax) / (gold_cmax + 1e-12)
    auc_re = abs(summary["AUC"] - gold_auc) / (gold_auc + 1e-12)

    assert cmax_re <= REL_TOL, (
        f"[{drug_name}] Cmax drifted: current={summary['Cmax']:.4f}, "
        f"golden={gold_cmax:.4f}, RE={cmax_re:.2%} > {REL_TOL:.0%}"
    )
    assert auc_re <= REL_TOL, (
        f"[{drug_name}] AUC drifted: current={summary['AUC']:.4f}, "
        f"golden={gold_auc:.4f}, RE={auc_re:.2%} > {REL_TOL:.0%}"
    )


@pytest.mark.regression
def test_deterministic(engine, patient, regimen):
    """동일 입력 → 동일 출력 (결정론적 확인)."""
    spec = load_drug_spec_by_name("caffeine")
    r1 = engine.run(spec, patient, regimen)
    r2 = engine.run(spec, patient, regimen)
    s1 = engine.pk_summary(r1)
    s2 = engine.pk_summary(r2)
    assert math.isclose(s1["Cmax"], s2["Cmax"], rel_tol=1e-9), "비결정론적 Cmax"
    assert math.isclose(s1["AUC"], s2["AUC"], rel_tol=1e-9), "비결정론적 AUC"
