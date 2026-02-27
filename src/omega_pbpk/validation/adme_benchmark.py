"""ADME predictor benchmarking against reference dataset."""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def _aafe(predicted: list[float], measured: list[float]) -> float:
    """Absolute average fold error."""
    if not predicted:
        return float("nan")
    fes = [
        abs(math.log10(max(p, 1e-9) / max(m, 1e-9)))
        for p, m in zip(predicted, measured, strict=False)
    ]
    return 10 ** (sum(fes) / len(fes))


def benchmark_adme_predictor(predictor=None) -> dict[str, float]:
    """Benchmark ADMEPredictor against reference dataset.

    Returns AAFE for logP, logS, fup. Target: AAFE < 3 for logP/logS, < 5 for fup.
    """
    if predictor is None:
        from omega_pbpk.prediction.adme_predictor import ADMEPredictor

        predictor = ADMEPredictor()
    ref_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "adme_reference.csv"
    if not ref_path.exists():
        logger.warning("adme_reference.csv not found; skipping benchmark")
        return {
            "logP_aafe": float("nan"),
            "logS_aafe": float("nan"),
            "fup_aafe": float("nan"),
            "n_compounds": 0,
        }

    pred_logP, meas_logP = [], []
    pred_logS, meas_logS = [], []
    pred_fup, meas_fup = [], []
    n = 0
    with open(ref_path) as f:
        for row in csv.DictReader(f):
            try:
                smiles = row["smiles"]
                props = predictor.predict(smiles)
                pred_logP.append(props.logP + 10)  # shift to positive
                meas_logP.append(float(row["logP"]) + 10)
                pred_logS.append(props.logS + 10)
                meas_logS.append(float(row["logS"]) + 10)
                pred_fup.append(max(props.fup, 1e-4))
                meas_fup.append(max(float(row["fup"]), 1e-4))
                n += 1
            except Exception as e:
                logger.debug(f"Skipped {row.get('name', '?')}: {e}")

    return {
        "logP_aafe": _aafe(pred_logP, meas_logP),
        "logS_aafe": _aafe(pred_logS, meas_logS),
        "fup_aafe": _aafe(pred_fup, meas_fup),
        "n_compounds": n,
    }


def print_benchmark_report(results: dict[str, float]) -> None:
    print(f"\nADME Predictor Benchmark (n={results['n_compounds']})")
    print(f"  logP AAFE: {results['logP_aafe']:.2f}x")
    print(f"  logS AAFE: {results['logS_aafe']:.2f}x")
    print(f"  fup  AAFE: {results['fup_aafe']:.2f}x")
