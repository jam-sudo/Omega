"""Phase 572 — Biphasic Drug Dissolution Model."""

import math
from dataclasses import dataclass


@dataclass
class BiphasicDissolutionResult:
    """Result of biphasic dissolution simulation."""

    drug_name: str
    dose_mg: float
    fast_fraction: float
    slow_fraction: float
    times_min: list
    dissolved_pct: list
    dissolution_rate_fast_per_min: float
    dissolution_rate_slow_per_min: float
    t50_min: float
    t80_min: float
    t90_min: float
    biphasic_index: float
    dissolution_class: str
    notes: str


def simulate_biphasic_dissolution(
    drug_name: str,
    dose_mg: float,
    fast_fraction: float,
    k_fast_per_min: float,
    k_slow_per_min: float,
    t_end_min: float = 120.0,
    dt_min: float = 0.5,
) -> BiphasicDissolutionResult:
    """Simulate biphasic drug dissolution with fast and slow release phases.

    Uses Forward Euler integration for dissolution kinetics.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0.0 <= fast_fraction <= 1.0):
        raise ValueError("fast_fraction must be in [0, 1]")
    if k_fast_per_min <= 0:
        raise ValueError("k_fast_per_min must be > 0")
    if k_slow_per_min <= 0:
        raise ValueError("k_slow_per_min must be > 0")
    if k_slow_per_min >= k_fast_per_min:
        raise ValueError("k_slow_per_min must be < k_fast_per_min")

    slow_fraction = 1.0 - fast_fraction

    M_fast = dose_mg * fast_fraction
    M_slow = dose_mg * slow_fraction

    times_min = []
    dissolved_pct = []

    t50_min = math.inf
    t80_min = math.inf
    t90_min = math.inf

    n_steps = int(t_end_min / dt_min)

    for i in range(n_steps + 1):
        t = i * dt_min
        times_min.append(t)
        pct = (dose_mg - M_fast - M_slow) / dose_mg * 100.0
        dissolved_pct.append(pct)

        if t50_min == math.inf and pct >= 50.0:
            t50_min = t
        if t80_min == math.inf and pct >= 80.0:
            t80_min = t
        if t90_min == math.inf and pct >= 90.0:
            t90_min = t

        if i < n_steps:
            dM_fast = -k_fast_per_min * M_fast
            dM_slow = -k_slow_per_min * M_slow

            M_fast += dM_fast * dt_min
            M_slow += dM_slow * dt_min

            if M_fast < 0:
                M_fast = 0.0
            if M_slow < 0:
                M_slow = 0.0

    biphasic_index = k_slow_per_min / k_fast_per_min

    if t80_min < 30.0:
        dissolution_class = "rapid"
    elif t80_min <= 120.0:
        dissolution_class = "intermediate"
    else:
        dissolution_class = "slow"

    notes = (
        f"Biphasic dissolution: fast_fraction={fast_fraction:.2f}, "
        f"k_fast={k_fast_per_min}/min, k_slow={k_slow_per_min}/min; "
        f"Forward Euler dt={dt_min}min"
    )

    return BiphasicDissolutionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fast_fraction=fast_fraction,
        slow_fraction=slow_fraction,
        times_min=times_min,
        dissolved_pct=dissolved_pct,
        dissolution_rate_fast_per_min=k_fast_per_min,
        dissolution_rate_slow_per_min=k_slow_per_min,
        t50_min=t50_min,
        t80_min=t80_min,
        t90_min=t90_min,
        biphasic_index=biphasic_index,
        dissolution_class=dissolution_class,
        notes=notes,
    )


def compare_fast_fractions(
    drug_name: str,
    dose_mg: float,
    k_fast: float,
    k_slow: float,
    fast_fractions: list = None,
) -> list:
    """Compare dissolution profiles across different fast fractions.

    Returns list of BiphasicDissolutionResult sorted by t80_min ascending.
    """
    if fast_fractions is None:
        fast_fractions = [0.2, 0.5, 0.8]

    results = []
    for ff in fast_fractions:
        r = simulate_biphasic_dissolution(
            drug_name=drug_name,
            dose_mg=dose_mg,
            fast_fraction=ff,
            k_fast_per_min=k_fast,
            k_slow_per_min=k_slow,
        )
        results.append(r)

    results.sort(key=lambda r: r.t80_min)
    return results
