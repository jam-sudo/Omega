from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omega_pbpk.prediction.adme_predictor import ADMEProperties


@dataclass(frozen=True)
class ParameterBounds:
    fup_lo: float
    fup_hi: float
    clint_lo: float
    clint_hi: float
    peff_lo: float
    peff_hi: float
    rbp_lo: float
    rbp_hi: float


@dataclass(frozen=True)
class ConformalUQResult:
    cmax_p5: float
    cmax_p50: float
    cmax_p95: float
    auc_p5: float
    auc_p50: float
    auc_p95: float
    tmax_p5: float
    tmax_p50: float
    tmax_p95: float
    t_half_p5: float
    t_half_p50: float
    t_half_p95: float
    n_samples: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


def build_bounds_from_adme(props: ADMEProperties) -> ParameterBounds:
    """Build ParameterBounds from an ADMEProperties object."""
    return ParameterBounds(
        fup_lo=props.fup_lo,
        fup_hi=props.fup_hi if props.fup_hi > 0 else props.fup,
        clint_lo=props.clint_3a4_lo,
        clint_hi=props.clint_3a4_hi if props.clint_3a4_hi > 0 else props.clint_3a4,
        peff_lo=props.peff_lo,
        peff_hi=props.peff_hi if props.peff_hi > 0 else props.peff,
        rbp_lo=props.rbp_lo,
        rbp_hi=props.rbp_hi if props.rbp_hi > 0 else props.rbp,
    )


def propagate_conformal_intervals(
    drug_name: str,
    dose_mg: float,
    route: str,
    bounds: ParameterBounds,
    n_samples: int = 200,
    seed: int = 42,
) -> ConformalUQResult:
    """LHS sampling over [lo, hi] for 4 parameters, 1-compartment PK per sample.

    Uses a simplified 1-compartment model:
      Vd [L] = 0.7 * rbp / fup * 60   (volume of distribution scaled)
      CL [L/h] = clint * fup * 60 * 1e-3   (well-stirred hepatic clearance)
      F = peff / (peff + 1e-5)   (simple absorption fraction)
      For oral: Cmax ~ F * dose / Vd,  AUC ~ F * dose / CL
      For iv:   Cmax ~ dose / Vd,      AUC ~ dose / CL
      t_half = 0.693 * Vd / CL
      tmax (oral) ~ 1.0 / (peff * 3600 / 0.01)  [simplified]
      tmax (iv) = 0.0833  (5-min infusion approximation)
    """
    from scipy.stats.qmc import LatinHypercube

    warnings: list[str] = []

    # Clamp bounds so lo <= hi and both positive
    fup_lo = max(bounds.fup_lo, 0.001)
    fup_hi = max(bounds.fup_hi, fup_lo)
    clint_lo = max(bounds.clint_lo, 0.001)
    clint_hi = max(bounds.clint_hi, clint_lo)
    peff_lo = max(bounds.peff_lo, 1e-7)
    peff_hi = max(bounds.peff_hi, peff_lo)
    rbp_lo = max(bounds.rbp_lo, 0.55)
    rbp_hi = max(bounds.rbp_hi, rbp_lo)

    if fup_lo == fup_hi and clint_lo == clint_hi and peff_lo == peff_hi and rbp_lo == rbp_hi:
        warnings.append("Degenerate bounds: all lo==hi, returning point estimate")

    sampler = LatinHypercube(d=4, seed=seed)
    lhs = sampler.random(n=n_samples)  # shape (n_samples, 4) in [0,1]

    # Scale to parameter ranges
    fup_s = fup_lo + lhs[:, 0] * (fup_hi - fup_lo)
    clint_s = clint_lo + lhs[:, 1] * (clint_hi - clint_lo)
    peff_s = peff_lo + lhs[:, 2] * (peff_hi - peff_lo)
    rbp_s = rbp_lo + lhs[:, 3] * (rbp_hi - rbp_lo)

    # Clip to physiological ranges
    fup_s = np.clip(fup_s, 0.001, 1.0)
    clint_s = np.clip(clint_s, 0.001, 100.0)
    peff_s = np.clip(peff_s, 1e-7, 1e-3)
    rbp_s = np.clip(rbp_s, 0.55, 3.0)

    # 1-compartment PK
    Vd = 0.7 * rbp_s / fup_s * 60.0  # L
    CL = clint_s * fup_s * 60.0 * 1e-3  # L/h
    CL = np.maximum(CL, 1e-6)

    if route == "iv":
        F = np.ones(n_samples)
        tmax = np.full(n_samples, 0.0833)
    else:
        F = peff_s / (peff_s + 1e-5)
        ka = np.maximum(peff_s * 3600.0 / 0.01, 0.001)
        tmax = np.log(ka * Vd / CL) / (ka - CL / Vd)
        tmax = np.abs(tmax)
        tmax = np.clip(tmax, 0.01, 24.0)

    cmax = F * dose_mg / Vd
    auc = F * dose_mg / CL
    t_half = 0.693 * Vd / CL

    def pcts(arr):
        return (
            float(np.percentile(arr, 5)),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 95)),
        )

    c5, c50, c95 = pcts(cmax)
    a5, a50, a95 = pcts(auc)
    tm5, tm50, tm95 = pcts(tmax)
    th5, th50, th95 = pcts(t_half)

    return ConformalUQResult(
        cmax_p5=c5,
        cmax_p50=c50,
        cmax_p95=c95,
        auc_p5=a5,
        auc_p50=a50,
        auc_p95=a95,
        tmax_p5=tm5,
        tmax_p50=tm50,
        tmax_p95=tm95,
        t_half_p5=th5,
        t_half_p50=th50,
        t_half_p95=th95,
        n_samples=n_samples,
        warnings=tuple(warnings),
    )
