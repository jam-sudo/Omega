"""Phase 327 — Drug Stability in GI Fluid.

Predicts first-order degradation of a drug in gastric (pH 2.0) and
intestinal (pH 6.8) fluids, estimating fraction reaching the absorption
site and classifying overall GI stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GASTRIC_PH = 2.0
_INTESTINAL_PH = 6.8
_GASTRIC_PHASE_END_MIN = 120.0  # minutes


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class GIStabilityResult:
    """Result of a GI fluid stability simulation.

    List fields prevent frozen=True.
    """

    times_min: list
    remaining_fraction: list  # values in [0, 1]
    gastric_t50_min: float  # inf if stable
    intestinal_t50_min: float  # inf if stable
    fraction_reaching_absorption_site: float  # fraction at t = 120 min
    stability_class: str  # "stable" / "labile" / "highly_labile"
    notes: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_gi_stability(
    drug_name: str,
    pKa: float,
    logP: float,
    drug_type: str,
    k_deg_gastric_per_h: float,
    k_deg_intestinal_per_h: float,
    dose_mg: float,
    t_end_min: float = 360.0,
    dt_min: float = 1.0,
) -> GIStabilityResult:
    """Simulate drug degradation across gastric and intestinal phases.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    pKa:
        Acid dissociation constant.
    logP:
        Octanol-water partition coefficient.
    drug_type:
        One of ``"acid"``, ``"base"``, or ``"neutral"``.
    k_deg_gastric_per_h:
        First-order degradation rate constant in gastric fluid (h^-1 ≥ 0).
    k_deg_intestinal_per_h:
        First-order degradation rate constant in intestinal fluid (h^-1 ≥ 0).
    dose_mg:
        Administered dose in mg (> 0).
    t_end_min:
        End of simulation in minutes (> 0). Default 360 min.
    dt_min:
        Time step in minutes (> 0). Default 1 min.

    Returns
    -------
    GIStabilityResult
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0; got {dose_mg}")
    if t_end_min <= 0:
        raise ValueError(f"t_end_min must be > 0; got {t_end_min}")
    if dt_min <= 0:
        raise ValueError(f"dt_min must be > 0; got {dt_min}")
    if k_deg_gastric_per_h < 0:
        raise ValueError(f"k_deg_gastric_per_h must be >= 0; got {k_deg_gastric_per_h}")
    if k_deg_intestinal_per_h < 0:
        raise ValueError(f"k_deg_intestinal_per_h must be >= 0; got {k_deg_intestinal_per_h}")
    valid_types = {"acid", "base", "neutral"}
    if drug_type not in valid_types:
        raise ValueError(f"drug_type must be one of {valid_types}; got {drug_type!r}")

    # ------------------------------------------------------------------
    # Convert rates from per-hour to per-minute
    # ------------------------------------------------------------------
    k_gastric_min = k_deg_gastric_per_h / 60.0
    k_intestinal_min = k_deg_intestinal_per_h / 60.0

    # ------------------------------------------------------------------
    # Build time array with Forward-Euler stepping
    # ------------------------------------------------------------------
    times_min: list[float] = []
    remaining_fraction: list[float] = []

    t = 0.0
    frac = 1.0  # starts at 1.0 (full dose present)

    while t <= t_end_min + 1e-9:
        times_min.append(t)
        remaining_fraction.append(max(0.0, min(1.0, frac)))

        # Determine which phase we are in
        if t < _GASTRIC_PHASE_END_MIN:
            k = k_gastric_min
        else:
            k = k_intestinal_min

        # Forward Euler: dfrac/dt = -k * frac
        frac = frac - k * frac * dt_min
        frac = max(0.0, frac)
        t = round(t + dt_min, 10)

    # ------------------------------------------------------------------
    # Fraction at gastric→intestinal transition (t ≈ 120 min)
    # ------------------------------------------------------------------
    idx_120 = _nearest_index(times_min, _GASTRIC_PHASE_END_MIN)
    fraction_reaching = remaining_fraction[idx_120]

    # ------------------------------------------------------------------
    # Half-lives (analytical: t½ = ln(2)/k)
    # ------------------------------------------------------------------
    if k_deg_gastric_per_h > 0:
        gastric_t50_min = math.log(2.0) / k_gastric_min
    else:
        gastric_t50_min = math.inf

    if k_deg_intestinal_per_h > 0:
        intestinal_t50_min = math.log(2.0) / k_intestinal_min
    else:
        intestinal_t50_min = math.inf

    # ------------------------------------------------------------------
    # Stability classification based on fraction at absorption site
    # ------------------------------------------------------------------
    if fraction_reaching > 0.8:
        stability_class = "stable"
    elif fraction_reaching >= 0.5:
        stability_class = "labile"
    else:
        stability_class = "highly_labile"

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    notes = (
        f"{drug_name} ({drug_type}, pKa={pKa:.1f}, logP={logP:.1f}): "
        f"{fraction_reaching * 100:.1f}% reaches absorption site at 120 min. "
        f"Gastric t½={gastric_t50_min:.1f} min, "
        f"Intestinal t½={intestinal_t50_min:.1f} min."
    )

    return GIStabilityResult(
        times_min=times_min,
        remaining_fraction=remaining_fraction,
        gastric_t50_min=gastric_t50_min,
        intestinal_t50_min=intestinal_t50_min,
        fraction_reaching_absorption_site=fraction_reaching,
        stability_class=stability_class,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nearest_index(times: list[float], target: float) -> int:
    """Return index of time value nearest to *target*."""
    best = 0
    best_diff = abs(times[0] - target)
    for i, t in enumerate(times):
        d = abs(t - target)
        if d < best_diff:
            best_diff = d
            best = i
    return best
