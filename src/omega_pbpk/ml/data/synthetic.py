"""Generate synthetic PK profiles from the ODE engine for ML training data.

Provides two data generators:
1. Full PBPK ODE data generator (high-fidelity, ~500ms/sample) storing full C(t) curves
2. 1-compartment analytical model (low-fidelity, microseconds/sample) for pre-training

Both generators use Latin Hypercube Sampling for parameter space coverage and
store results in HDF5 format for efficient I/O.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TIMEPOINTS = 241  # 0 to 24h at 0.1h intervals
T_END_H = 24.0
DT_H = 0.1

# 6D parameter space for PBPK ODE data generation
PBPK_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "logP": (-2.0, 7.0),
    "fup": (0.005, 1.0),
    "clint_L_h": (0.0001, 10.0),  # Realistic after IVIVE: ~0.0005-5 L/h
    "mw": (100.0, 900.0),
    "rbp": (0.3, 3.0),
    "peff": (0.01, 10.0),  # Realistic: ~0.1-5 × 10⁻⁴ cm/s
}

PBPK_PARAM_NAMES: list[str] = list(PBPK_PARAM_RANGES.keys())

# 5D parameter space for 1-compartment model
ONECOMP_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "ka": (0.1, 5.0),  # absorption rate constant (/h)
    "ke": (0.01, 2.0),  # elimination rate constant (/h)
    "Vd": (5.0, 500.0),  # volume of distribution (L)
    "F": (0.1, 1.0),  # bioavailability
    "dose": (1.0, 1000.0),  # dose (mg)
}

ONECOMP_PARAM_NAMES: list[str] = list(ONECOMP_PARAM_RANGES.keys())

# PK scalar metrics extracted alongside curves
PK_METRIC_NAMES: list[str] = ["Cmax_mg_L", "AUC_mg_h_L", "Tmax_h", "half_life_h"]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class CurveDataset:
    """Dataset containing full C(t) curves and scalar PK metrics.

    Attributes:
        params: Input parameters, shape (n_samples, n_params).
        curves: Plasma concentration curves, shape (n_samples, n_timepoints).
        metrics: Scalar PK metrics, shape (n_samples, n_metrics).
        time_h: Time vector, shape (n_timepoints,).
        param_names: Names of input parameters.
        metric_names: Names of scalar PK metrics.
        metadata: Generation metadata (date, ranges, etc.).
    """

    params: NDArray[np.float64]
    curves: NDArray[np.float64]
    metrics: NDArray[np.float64]
    time_h: NDArray[np.float64]
    param_names: list[str] = field(default_factory=list)
    metric_names: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.params.shape[0]

    @property
    def n_params(self) -> int:
        return self.params.shape[1]

    @property
    def n_timepoints(self) -> int:
        return self.curves.shape[1]

    def save_hdf5(self, path: str | Path) -> None:
        """Save dataset to HDF5 file.

        Args:
            path: Output file path.
        """
        import h5py

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(path), "w") as f:
            f.create_dataset("params", data=self.params, compression="gzip")
            f.create_dataset("curves", data=self.curves, compression="gzip")
            f.create_dataset("metrics", data=self.metrics, compression="gzip")
            f.create_dataset("time_h", data=self.time_h)

            f.attrs["param_names"] = json.dumps(self.param_names)
            f.attrs["metric_names"] = json.dumps(self.metric_names)
            f.attrs["metadata"] = json.dumps(self.metadata, default=str)

        logger.info("Saved dataset (%d samples) to %s", self.n_samples, path)

    @classmethod
    def load_hdf5(cls, path: str | Path) -> CurveDataset:
        """Load dataset from HDF5 file.

        Args:
            path: Input file path.

        Returns:
            Loaded CurveDataset.
        """
        import h5py

        with h5py.File(str(path), "r") as f:
            params = f["params"][:]
            curves = f["curves"][:]
            metrics = f["metrics"][:]
            time_h = f["time_h"][:]
            param_names = json.loads(f.attrs["param_names"])
            metric_names = json.loads(f.attrs["metric_names"])
            metadata = json.loads(f.attrs["metadata"])

        return cls(
            params=params,
            curves=curves,
            metrics=metrics,
            time_h=time_h,
            param_names=param_names,
            metric_names=metric_names,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# LHS utility
# ---------------------------------------------------------------------------


def _latin_hypercube(n: int, d: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate Latin Hypercube Samples in [0, 1]^d.

    Args:
        n: Number of samples.
        d: Number of dimensions.
        rng: NumPy random generator.

    Returns:
        Array of shape (n, d) with values in [0, 1].
    """
    result = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.uniform(size=n)) / n
    return result


# ---------------------------------------------------------------------------
# Task A.1: Full PBPK ODE data generator with C(t) curves
# ---------------------------------------------------------------------------


def _pbpk_worker(
    args: tuple[dict[str, float], float, str, float, float, int],
) -> tuple[int, NDArray[np.float64] | None, list[float] | None]:
    """Worker: run one PBPK simulation, return full C(t) curve + PK metrics.

    Args:
        args: Tuple of (params, dose_mg, route, body_weight, t_end_h, index).

    Returns:
        (index, curve_array, pk_metrics) where curve/metrics are None on failure.
    """
    params, dose_mg, route, body_weight, t_end_h, idx = args
    try:
        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.drugs.drug import Drug

        drug = Drug(
            name="synthetic_sample",
            logP=params["logP"],
            fup=max(params["fup"], 1e-6),
            rbp=params["rbp"],
            peff=params["peff"],
            mw=params["mw"],
            clint_hepatic_L_per_h=params["clint_L_h"],
        )

        model = WholeBodyPBPK(drug, body_weight=body_weight)
        if route == "iv":
            model.setup_iv(dose_mg)
        else:
            model.setup_oral(dose_mg)

        result = model.simulate(t_end_h=t_end_h, dt_h=DT_H)
        cp = result.plasma_concentration()

        # Compute scalar PK metrics
        t = result.time_h
        cmax = float(np.max(cp))
        tmax = float(t[int(np.argmax(cp))])
        auc = float(np.trapz(cp, t))

        # Terminal half-life from last 50% of curve
        mid = len(t) // 2
        cp_term = cp[mid:]
        t_term = t[mid:]
        pos_mask = cp_term > 1e-12
        if np.sum(pos_mask) >= 2:
            log_cp = np.log(cp_term[pos_mask])
            t_pos = t_term[pos_mask]
            slope = np.polyfit(t_pos, log_cp, 1)[0]
            half_life = -np.log(2) / slope if slope < -1e-12 else t_end_h * 10.0
        else:
            half_life = t_end_h * 10.0

        if half_life <= 0 or half_life > 1e6:
            half_life = t_end_h * 10.0

        metrics = [cmax, auc, tmax, half_life]

        # Validate
        if any(not np.isfinite(v) or v < 0 for v in metrics):
            return idx, None, None
        if not np.all(np.isfinite(cp)):
            return idx, None, None

        return idx, cp.astype(np.float64), metrics

    except Exception as e:
        logger.debug("PBPK sample %d failed: %s", idx, e)
        return idx, None, None


def generate_pbpk_data(
    n_samples: int = 500,
    dose_mg: float = 10.0,
    route: str = "oral",
    body_weight: float = 70.0,
    t_end_h: float = T_END_H,
    seed: int = 42,
    n_workers: int | None = None,
    param_ranges: dict[str, tuple[float, float]] | None = None,
) -> CurveDataset:
    """Generate PBPK training data with full C(t) curves.

    Uses Latin Hypercube Sampling across a 6D parameter space and runs
    the full 35-state ODE engine for each sample. Stores both the complete
    plasma concentration-time curve and scalar PK metrics.

    Args:
        n_samples: Number of parameter combinations to sample.
        dose_mg: Dose (mg).
        route: 'oral' or 'iv'.
        body_weight: Subject body weight (kg).
        t_end_h: Simulation end time (h).
        seed: Random seed for reproducibility.
        n_workers: Number of worker processes. None uses all available cores.
        param_ranges: Optional custom parameter ranges (overrides defaults).

    Returns:
        CurveDataset with params, curves, and metrics.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    ranges = param_ranges or PBPK_PARAM_RANGES
    param_names = list(ranges.keys())
    n_params = len(param_names)

    rng = np.random.default_rng(seed)
    X_unit = _latin_hypercube(n_samples, n_params, rng)
    X = np.empty_like(X_unit)

    for j, name in enumerate(param_names):
        lo, hi = ranges[name]
        if name == "logP":
            # logP is already a log-scale quantity; sample linearly
            X[:, j] = lo + X_unit[:, j] * (hi - lo)
        else:
            # Log-uniform for strictly positive parameters
            lo_safe = max(lo, 1e-8)
            X[:, j] = np.exp(np.log(lo_safe) + X_unit[:, j] * (np.log(hi) - np.log(lo_safe)))

    all_params = [
        {name: float(X[i, j]) for j, name in enumerate(param_names)} for i in range(n_samples)
    ]
    worker_args = [
        (params, dose_mg, route, body_weight, t_end_h, i) for i, params in enumerate(all_params)
    ]

    # Expected timepoints
    t_eval = np.arange(0.0, t_end_h + DT_H / 2, DT_H)
    n_tp = len(t_eval)

    # Allocate storage
    curves_all: list[NDArray[np.float64] | None] = [None] * n_samples
    metrics_all: list[list[float] | None] = [None] * n_samples
    valid_mask: list[bool] = [False] * n_samples

    start_time = time.monotonic()

    if n_workers == 1 or n_samples <= 1:
        for args in worker_args:
            idx, curve, metrics = _pbpk_worker(args)
            if curve is not None and metrics is not None and len(curve) == n_tp:
                curves_all[idx] = curve
                metrics_all[idx] = metrics
                valid_mask[idx] = True
            if (idx + 1) % 50 == 0:
                elapsed = time.monotonic() - start_time
                logger.info(
                    "PBPK data: %d/%d samples (%.1fs elapsed)",
                    idx + 1,
                    n_samples,
                    elapsed,
                )
    else:
        max_workers = min(n_workers, n_samples)
        completed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(_pbpk_worker, args): args[-1] for args in worker_args}
            for future in as_completed(future_map):
                idx, curve, metrics = future.result()
                if curve is not None and metrics is not None and len(curve) == n_tp:
                    curves_all[idx] = curve
                    metrics_all[idx] = metrics
                    valid_mask[idx] = True
                completed += 1
                if completed % 50 == 0:
                    elapsed = time.monotonic() - start_time
                    logger.info(
                        "PBPK data: %d/%d samples (%.1fs elapsed)",
                        completed,
                        n_samples,
                        elapsed,
                    )

    elapsed = time.monotonic() - start_time
    mask = np.array(valid_mask)
    n_valid = int(np.sum(mask))

    X_valid = X[mask]
    curves_valid = np.array([c for c in curves_all if c is not None], dtype=np.float64)
    metrics_valid = np.array([m for m in metrics_all if m is not None], dtype=np.float64)

    logger.info(
        "PBPK data generation complete: %d/%d valid (%.1f%%) in %.1fs",
        n_valid,
        n_samples,
        100 * n_valid / max(n_samples, 1),
        elapsed,
    )

    metadata = {
        "generator": "pbpk_ode_35state",
        "n_requested": n_samples,
        "n_valid": n_valid,
        "dose_mg": dose_mg,
        "route": route,
        "body_weight_kg": body_weight,
        "t_end_h": t_end_h,
        "dt_h": DT_H,
        "n_timepoints": n_tp,
        "seed": seed,
        "param_ranges": {k: list(v) for k, v in ranges.items()},
        "generation_date": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
    }

    return CurveDataset(
        params=X_valid,
        curves=curves_valid,
        metrics=metrics_valid,
        time_h=t_eval,
        param_names=list(param_names),
        metric_names=list(PK_METRIC_NAMES),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Task A.2: 1-Compartment Analytical Data Generator
# ---------------------------------------------------------------------------


def _onecomp_oral_curve(
    t: NDArray[np.float64],
    ka: float,
    ke: float,
    Vd: float,
    F: float,
    dose: float,
) -> NDArray[np.float64]:
    """Compute 1-compartment oral PK concentration-time curve.

    C(t) = (F * dose * ka / (Vd * (ka - ke))) * (exp(-ke*t) - exp(-ka*t))

    When ka is very close to ke, uses the limiting form:
    C(t) = (F * dose / Vd) * t * ka * exp(-ke*t)

    Args:
        t: Time vector (h).
        ka: Absorption rate constant (/h).
        ke: Elimination rate constant (/h).
        Vd: Volume of distribution (L).
        F: Bioavailability (0-1).
        dose: Dose (mg).

    Returns:
        Concentration array (mg/L).
    """
    if abs(ka - ke) < 1e-10:
        # Limiting case: ka == ke
        return (F * dose / Vd) * t * ka * np.exp(-ke * t)

    coeff = F * dose * ka / (Vd * (ka - ke))
    return coeff * (np.exp(-ke * t) - np.exp(-ka * t))


def generate_1cpt_data(
    n_samples: int = 100_000,
    seed: int = 42,
    t_end_h: float = T_END_H,
) -> CurveDataset:
    """Generate 1-compartment oral PK analytical data for pre-training.

    Uses analytical solution of the 1-compartment oral model, which is
    computationally trivial (microseconds per sample). This provides a
    low-fidelity dataset for multi-fidelity pre-training before fine-tuning
    on the expensive PBPK ODE data.

    Args:
        n_samples: Number of samples to generate.
        seed: Random seed.
        t_end_h: End time (h).

    Returns:
        CurveDataset with params, curves, and metrics.
    """
    rng = np.random.default_rng(seed)
    ranges = ONECOMP_PARAM_RANGES
    param_names = list(ranges.keys())
    n_params = len(param_names)

    # LHS sampling
    X_unit = _latin_hypercube(n_samples, n_params, rng)
    X = np.empty_like(X_unit)

    for j, name in enumerate(param_names):
        lo, hi = ranges[name]
        if name == "F":
            # Bioavailability: uniform on [0.1, 1.0]
            X[:, j] = lo + X_unit[:, j] * (hi - lo)
        else:
            # Log-uniform for rate constants, volumes, doses
            X[:, j] = np.exp(np.log(lo) + X_unit[:, j] * (np.log(hi) - np.log(lo)))

    t = np.arange(0.0, t_end_h + DT_H / 2, DT_H)
    n_tp = len(t)

    curves = np.zeros((n_samples, n_tp), dtype=np.float64)
    metrics = np.zeros((n_samples, 4), dtype=np.float64)
    valid_mask = np.ones(n_samples, dtype=bool)

    start_time = time.monotonic()

    for i in range(n_samples):
        ka_val = X[i, 0]
        ke_val = X[i, 1]
        vd_val = X[i, 2]
        f_val = X[i, 3]
        dose_val = X[i, 4]

        cp = _onecomp_oral_curve(t, ka_val, ke_val, vd_val, f_val, dose_val)

        # Validate
        if not np.all(np.isfinite(cp)) or np.any(cp < -1e-12):
            valid_mask[i] = False
            continue

        cp = np.maximum(cp, 0.0)
        curves[i] = cp

        # PK metrics
        cmax = float(np.max(cp))
        tmax = float(t[int(np.argmax(cp))])
        auc = float(np.trapz(cp, t))
        half_life = np.log(2) / ke_val if ke_val > 1e-12 else t_end_h * 10.0

        metrics[i] = [cmax, auc, tmax, half_life]

    elapsed = time.monotonic() - start_time
    n_valid = int(np.sum(valid_mask))

    X_valid = X[valid_mask]
    curves_valid = curves[valid_mask]
    metrics_valid = metrics[valid_mask]

    logger.info(
        "1-cpt data generation: %d/%d valid in %.3fs (%.0f samples/sec)",
        n_valid,
        n_samples,
        elapsed,
        n_valid / max(elapsed, 1e-6),
    )

    metadata = {
        "generator": "1cpt_oral_analytical",
        "n_requested": n_samples,
        "n_valid": n_valid,
        "t_end_h": t_end_h,
        "dt_h": DT_H,
        "n_timepoints": n_tp,
        "seed": seed,
        "param_ranges": {k: list(v) for k, v in ranges.items()},
        "generation_date": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "model": "C(t) = (F*dose*ka/(Vd*(ka-ke))) * (exp(-ke*t) - exp(-ka*t))",
    }

    return CurveDataset(
        params=X_valid,
        curves=curves_valid,
        metrics=metrics_valid,
        time_h=t,
        param_names=list(param_names),
        metric_names=list(PK_METRIC_NAMES),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic PK training data for ML models.",
    )
    parser.add_argument(
        "--mode",
        choices=["pbpk", "1cpt", "both"],
        default="both",
        help="Which data generator to run (default: both).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of samples for PBPK data (default: 500).",
    )
    parser.add_argument(
        "--n-1cpt",
        type=int,
        default=100_000,
        help="Number of samples for 1-cpt data (default: 100000).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ml",
        help="Output directory (default: data/ml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of worker processes for PBPK (default: all cores).",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("1cpt", "both"):
        logger.info("Generating 1-compartment analytical data (%d samples)...", args.n_1cpt)
        ds_1cpt = generate_1cpt_data(n_samples=args.n_1cpt, seed=args.seed)
        ds_1cpt.save_hdf5(output_dir / "1cpt_curves.h5")
        logger.info(
            "1-cpt data: %d samples, %d timepoints, params shape %s",
            ds_1cpt.n_samples,
            ds_1cpt.n_timepoints,
            ds_1cpt.params.shape,
        )

    if args.mode in ("pbpk", "both"):
        logger.info("Generating PBPK ODE data (%d samples)...", args.n_samples)
        ds_pbpk = generate_pbpk_data(
            n_samples=args.n_samples,
            seed=args.seed,
            n_workers=args.n_workers,
        )
        ds_pbpk.save_hdf5(output_dir / "pbpk_curves.h5")
        logger.info(
            "PBPK data: %d samples, %d timepoints, params shape %s",
            ds_pbpk.n_samples,
            ds_pbpk.n_timepoints,
            ds_pbpk.params.shape,
        )

    logger.info("Data generation complete. Output in %s", output_dir)


if __name__ == "__main__":
    main()


__all__ = [
    "CurveDataset",
    "generate_pbpk_data",
    "generate_1cpt_data",
    "PBPK_PARAM_RANGES",
    "PBPK_PARAM_NAMES",
    "ONECOMP_PARAM_RANGES",
    "ONECOMP_PARAM_NAMES",
    "PK_METRIC_NAMES",
    "N_TIMEPOINTS",
]
