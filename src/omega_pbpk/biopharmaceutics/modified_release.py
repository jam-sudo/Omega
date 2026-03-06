"""Modified-release formulation modeling — OROS, matrix, osmotic systems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from omega_pbpk._compat import np_trapz


class MRType(str, Enum):
    MATRIX = "matrix"  # Higuchi matrix (sqrt-time release)
    OSMOTIC = "osmotic"  # OROS zero-order release
    MULTIPARTICULATE = "mp"  # Beads with first-order release
    EROSION = "erosion"  # Surface erosion (zero-order)
    DIFFUSION = "diffusion"  # Membrane-controlled first-order


@dataclass(frozen=True)
class ModifiedReleaseResult:
    drug_name: str
    mr_type: str
    dose_mg: float
    times_h: list[float]
    release_fraction: list[float]  # cumulative drug released (0–1)
    plasma_conc: list[float]  # plasma concentration (mg/L)
    auc_mg_h_L: float
    cmax_mg_L: float
    tmax_h: float
    t_plateau_h: float  # time of plateau (if OROS/zero-order)
    fluctuation_pct: float  # (Cmax-Cmin)/Cavg * 100
    comparison_vs_ir: str  # brief comparison to IR


def _release_profile(
    mr_type: MRType,
    times_h: np.ndarray,
    duration_h: float = 12.0,
    k_release: float = 0.1,
) -> np.ndarray:
    """Compute cumulative release fraction for each MR type."""
    t = times_h
    release = np.zeros(len(t))

    if mr_type == MRType.OSMOTIC:
        # OROS: zero-order until orifice exhausted
        rate = 1.0 / duration_h
        release = np.minimum(rate * t, 1.0)

    elif mr_type == MRType.MATRIX:
        # Higuchi: F = k * sqrt(t), scaled to reach 1 at duration_h
        k = 1.0 / math.sqrt(duration_h)
        release = np.minimum(k * np.sqrt(np.maximum(t, 0.0)), 1.0)

    elif mr_type == MRType.EROSION:
        # Zero-order erosion
        rate = 1.0 / duration_h
        release = np.minimum(rate * t, 1.0)

    elif mr_type == MRType.DIFFUSION:
        # First-order: F = 1 - exp(-k*t)
        release = 1.0 - np.exp(-k_release * t)

    elif mr_type == MRType.MULTIPARTICULATE:
        # Mix of fast/slow beads (biexponential)
        f_fast = 0.3
        k_fast = 2.0 * k_release
        k_slow = 0.5 * k_release
        release = f_fast * (1.0 - np.exp(-k_fast * t)) + (1.0 - f_fast) * (
            1.0 - np.exp(-k_slow * t)
        )

    return release


def simulate_modified_release(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    mr_type: MRType = MRType.OSMOTIC,
    release_duration_h: float = 12.0,
    k_release: float = 0.1,
    ka_absorption_per_h: float = 1.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> ModifiedReleaseResult:
    """Simulate PK of a modified-release formulation.

    Two-step process:
    1. Drug released from formulation (MR-type dependent)
    2. Released drug absorbed then eliminated (1-cpt)

    Parameters
    ----------
    release_duration_h : float
        Duration over which MR system releases drug.
    ka_absorption_per_h : float
        Absorption rate of released drug from GI.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    ke = cl_L_per_h / vd_L
    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)

    # Release profile
    release = _release_profile(mr_type, t, release_duration_h, k_release)

    # Forward Euler simulation
    # State: A_released (absorbed from formulation), A_central (in body)
    cp = np.zeros(n + 1)
    a_gut = 0.0  # released but not yet absorbed
    a_central = 0.0

    prev_rel_frac = 0.0
    for i in range(n):
        # Drug released from formulation this step
        rel_frac = float(release[i + 1])
        delta_released = max((rel_frac - prev_rel_frac) * dose_mg, 0.0)
        a_gut += delta_released
        prev_rel_frac = rel_frac

        # GI absorption
        absorbed = ka_absorption_per_h * a_gut * dt_h
        a_gut = max(a_gut - absorbed, 0.0)

        # Elimination from central
        a_central += absorbed - ke * a_central * dt_h
        a_central = max(a_central, 0.0)
        cp[i + 1] = a_central / vd_L

    auc = float(np_trapz(cp, t))
    cmax = float(np.max(cp))
    tmax_h = float(t[int(np.argmax(cp))])

    # Plateau time (for OROS/zero-order)
    if mr_type in (MRType.OSMOTIC, MRType.EROSION):
        t_plateau = float(release_duration_h)
    else:
        t_plateau = float(tmax_h)

    # Fluctuation = (Cmax-Cmin)/Cavg * 100
    cavg = auc / t_end_h if t_end_h > 0 else cmax
    cmin = float(np.min(cp[cp > 0])) if np.any(cp > 0) else 0.0
    fluctuation = 100.0 * (cmax - cmin) / max(cavg, 1e-9)

    # Comparison vs IR
    if mr_type == MRType.OSMOTIC:
        comparison = "OROS provides near-zero-order release; lower Cmax, higher Cmin vs IR"
    elif mr_type == MRType.MATRIX:
        comparison = "Matrix: sqrt-time release; reduced Cmax, sustained Cmin vs IR"
    else:
        comparison = f"{mr_type.value}: modified release extends tmax and reduces peak-trough vs IR"

    return ModifiedReleaseResult(
        drug_name=drug_name,
        mr_type=mr_type.value if hasattr(mr_type, "value") else str(mr_type),
        dose_mg=dose_mg,
        times_h=t.tolist(),
        release_fraction=release.tolist(),
        plasma_conc=cp.tolist(),
        auc_mg_h_L=auc,
        cmax_mg_L=cmax,
        tmax_h=tmax_h,
        t_plateau_h=t_plateau,
        fluctuation_pct=fluctuation,
        comparison_vs_ir=comparison,
    )


def compare_mr_types(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
) -> dict[str, ModifiedReleaseResult]:
    """Compare all MR types for the same drug."""
    return {
        mr.value: simulate_modified_release(drug_name, dose_mg, cl_L_per_h, vd_L, mr_type=mr)
        for mr in MRType
    }


__all__ = [
    "MRType",
    "ModifiedReleaseResult",
    "simulate_modified_release",
    "compare_mr_types",
]
