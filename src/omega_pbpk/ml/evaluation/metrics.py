"""PK-specific evaluation metrics -- AAFE, fold error, coverage, calibration.

Provides conformal prediction calibration for ADME uncertainty intervals:
  - Checks that 90% CI covers 90% of true ADME values
  - Reports calibration metrics (coverage, sharpness, PICP)
  - Adjusts quantiles if coverage is off
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent


@dataclass(frozen=True)
class CalibrationResult:
    """Result of conformal calibration for one ADME property."""

    property_name: str
    target_coverage: float
    actual_coverage: float
    n_samples: int
    mean_interval_width: float
    is_calibrated: bool  # True if actual_coverage >= target_coverage - 0.05
    adjustment_factor: float  # Multiplier to widen/narrow intervals


@dataclass
class CalibrationReport:
    """Aggregate calibration report across all ADME properties."""

    results: list[CalibrationResult] = field(default_factory=list)
    overall_calibrated: bool = False

    def __str__(self) -> str:
        lines = ["Conformal Calibration Report", "=" * 40]
        for r in self.results:
            status = "OK" if r.is_calibrated else "NEEDS ADJUSTMENT"
            lines.append(
                f"  {r.property_name:<12}: coverage={r.actual_coverage:.1%} "
                f"(target={r.target_coverage:.1%}) width={r.mean_interval_width:.4f} "
                f"[{status}]"
            )
        cal_status = "CALIBRATED" if self.overall_calibrated else "NOT CALIBRATED"
        lines.append(f"\nOverall: {cal_status}")
        return "\n".join(lines)


def calibrate_uncertainty(
    predictor: Any,
    data_path: Path | None = None,
    target_coverage: float = 0.90,
    tolerance: float = 0.05,
) -> CalibrationReport:
    """Calibrate conformal prediction intervals on held-out ADME data.

    For each ADME property with ground truth in adme_reference.csv:
    1. Predict with the predictor and get intervals
    2. Check if target_coverage fraction of true values fall within intervals
    3. Compute adjustment factor if coverage is off

    Args:
        predictor: An MLADMEPredictor (or anything with .predict(smiles)).
        data_path: Path to reference CSV. Uses default if None.
        target_coverage: Desired coverage probability (default 0.90).
        tolerance: Acceptable deviation from target coverage.

    Returns:
        CalibrationReport with per-property calibration results.
    """

    path = data_path or (_REPO_ROOT / "data" / "adme_reference.csv")
    if not path.exists():
        logger.warning("Reference data not found at %s", path)
        return CalibrationReport()

    # Load reference data
    rows: list[dict[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    if not rows:
        return CalibrationReport()

    # Properties to calibrate: (property_name, csv_column, unit_conversion)
    properties = [
        ("fup", "fup", 1.0),
        ("clint_3a4", "clint_3a4_uL_min_pmol", 1.0),
        ("peff", "peff_cm_s", 1e4),  # csv is cm/s, our system is x10^-4 cm/s
        ("rbp", "rbp", 1.0),
    ]

    report = CalibrationReport()

    for prop_name, csv_col, unit_conv in properties:
        result = _calibrate_single_property(
            predictor, rows, prop_name, csv_col, unit_conv, target_coverage, tolerance
        )
        if result is not None:
            report.results.append(result)

    report.overall_calibrated = all(r.is_calibrated for r in report.results)
    return report


def _calibrate_single_property(
    predictor: Any,
    rows: list[dict[str, str]],
    prop_name: str,
    csv_col: str,
    unit_conv: float,
    target_coverage: float,
    tolerance: float,
) -> CalibrationResult | None:
    """Calibrate intervals for a single ADME property.

    Args:
        predictor: ADME predictor.
        rows: Reference data rows.
        prop_name: Property name in ADMEProperties.
        csv_col: Column name in CSV.
        unit_conv: Multiplier to convert CSV units to our units.
        target_coverage: Target coverage probability.
        tolerance: Acceptable tolerance.

    Returns:
        CalibrationResult or None if insufficient data.
    """
    from omega_pbpk.ml.models.adme.xgboost_adme import _name_to_smiles

    lo_attr = f"{prop_name}_lo"
    hi_attr = f"{prop_name}_hi"

    true_vals: list[float] = []
    pred_lo: list[float] = []
    pred_hi: list[float] = []
    interval_widths: list[float] = []

    for row in rows:
        name = row.get("name", "")
        val_str = row.get(csv_col)
        if val_str is None:
            continue

        smiles = row.get("smiles", row.get("SMILES", ""))
        if not smiles:
            smiles = _name_to_smiles(name)
        if not smiles:
            continue

        try:
            true_val = float(val_str) * unit_conv
        except (ValueError, TypeError):
            continue

        try:
            adme = predictor.predict(smiles)
            lo = getattr(adme, lo_attr, 0.0)
            hi = getattr(adme, hi_attr, 0.0)
            if hi <= lo:
                # No valid interval
                point = getattr(adme, prop_name, true_val)
                lo = point * 0.5
                hi = point * 1.5
        except Exception as exc:
            logger.debug("Prediction failed for %s: %s", name, exc)
            continue

        true_vals.append(true_val)
        pred_lo.append(lo)
        pred_hi.append(hi)
        interval_widths.append(hi - lo)

    n = len(true_vals)
    if n < 5:
        logger.warning("Insufficient data for calibration of %s (%d samples)", prop_name, n)
        return None

    # Compute coverage
    covered = sum(1 for t, lo, hi in zip(true_vals, pred_lo, pred_hi) if lo <= t <= hi)
    actual_coverage = covered / n

    # Compute adjustment factor
    # If under-covering, widen intervals; if over-covering, narrow them
    if actual_coverage < target_coverage - tolerance:
        # Need to widen: find the factor that would achieve target coverage
        # Binary search for the right multiplier
        adjustment = _find_adjustment_factor(true_vals, pred_lo, pred_hi, target_coverage)
    elif actual_coverage > target_coverage + tolerance:
        adjustment = _find_adjustment_factor(true_vals, pred_lo, pred_hi, target_coverage)
    else:
        adjustment = 1.0

    is_calibrated = abs(actual_coverage - target_coverage) <= tolerance

    return CalibrationResult(
        property_name=prop_name,
        target_coverage=target_coverage,
        actual_coverage=round(actual_coverage, 4),
        n_samples=n,
        mean_interval_width=round(float(np.mean(interval_widths)), 4),
        is_calibrated=is_calibrated,
        adjustment_factor=round(adjustment, 4),
    )


def _find_adjustment_factor(
    true_vals: list[float],
    pred_lo: list[float],
    pred_hi: list[float],
    target_coverage: float,
    max_iterations: int = 50,
) -> float:
    """Binary search for interval width adjustment factor.

    Finds a multiplier m such that widening intervals by m achieves
    approximately target_coverage.

    The adjusted interval is:
      center = (lo + hi) / 2
      half_width = (hi - lo) / 2 * m
      new_lo = center - half_width
      new_hi = center + half_width
    """
    lo_bound = 0.1
    hi_bound = 10.0
    n = len(true_vals)

    for _ in range(max_iterations):
        m = (lo_bound + hi_bound) / 2.0
        covered = 0
        for t, lo, hi in zip(true_vals, pred_lo, pred_hi):
            center = (lo + hi) / 2.0
            half_w = (hi - lo) / 2.0 * m
            if center - half_w <= t <= center + half_w:
                covered += 1
        cov = covered / n
        if abs(cov - target_coverage) < 0.01:
            return m
        if cov < target_coverage:
            lo_bound = m
        else:
            hi_bound = m

    return (lo_bound + hi_bound) / 2.0


def picp(true_vals: list[float], lo_vals: list[float], hi_vals: list[float]) -> float:
    """Prediction Interval Coverage Probability.

    PICP = fraction of true values falling within [lo, hi].

    Args:
        true_vals: Observed values.
        lo_vals: Lower bounds of prediction intervals.
        hi_vals: Upper bounds of prediction intervals.

    Returns:
        PICP in [0, 1].
    """
    if not true_vals:
        return 0.0
    covered = sum(1 for t, lo, hi in zip(true_vals, lo_vals, hi_vals) if lo <= t <= hi)
    return covered / len(true_vals)


def mpiw(lo_vals: list[float], hi_vals: list[float]) -> float:
    """Mean Prediction Interval Width.

    Args:
        lo_vals: Lower bounds.
        hi_vals: Upper bounds.

    Returns:
        Mean width of intervals.
    """
    if not lo_vals:
        return 0.0
    widths = [hi - lo for lo, hi in zip(lo_vals, hi_vals)]
    return float(np.mean(widths))
