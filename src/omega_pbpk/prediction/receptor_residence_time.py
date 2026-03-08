"""Drug-receptor residence time and kinetic selectivity prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReceptorResidenceResult:
    """Result of receptor residence time prediction."""

    drug_name: str
    target_name: str
    kon_M_per_s: float
    koff_per_s: float
    kd_nM: float
    residence_time_s: float
    residence_time_min: float
    residence_time_h: float
    t50_recovery_min: float
    kinetic_selectivity_vs_ref: float
    occupancy_class: str
    notes: str


def _classify_occupancy(rt_min: float) -> str:
    """Classify receptor occupancy duration."""
    if rt_min < 1.0:
        return "transient"
    if rt_min < 30.0:
        return "short"
    if rt_min <= 240.0:
        return "moderate"
    return "long"


def predict_residence_time(
    drug_name: str,
    target_name: str,
    kon_M_per_s: float,
    koff_per_s: float,
    ref_koff_per_s: float | None = None,
) -> ReceptorResidenceResult:
    """Predict drug-receptor residence time and kinetic selectivity.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    target_name:
        Name of the receptor / target.
    kon_M_per_s:
        Association rate constant (M^-1 s^-1).
    koff_per_s:
        Dissociation rate constant (s^-1).
    ref_koff_per_s:
        Reference compound dissociation rate (s^-1).  If provided, kinetic
        selectivity = RT_drug / RT_ref = ref_koff / koff.  If None, kinetic
        selectivity is reported as 1.0.

    Returns
    -------
    ReceptorResidenceResult
    """
    if kon_M_per_s <= 0:
        raise ValueError("kon_M_per_s must be > 0")
    if koff_per_s <= 0:
        raise ValueError("koff_per_s must be > 0")
    if ref_koff_per_s is not None and ref_koff_per_s <= 0:
        raise ValueError("ref_koff_per_s must be > 0 when provided")

    # Residence time RT = 1 / koff
    rt_s = 1.0 / koff_per_s
    rt_min = rt_s / 60.0
    rt_h = rt_s / 3600.0

    # Equilibrium dissociation constant Kd = koff / kon  (M), convert to nM
    kd_M = koff_per_s / kon_M_per_s
    kd_nM = kd_M * 1.0e9

    # Time to 50% occupancy recovery: t50 = ln(2) * RT
    t50_s = math.log(2.0) * rt_s
    t50_min = t50_s / 60.0

    # Kinetic selectivity vs reference
    if ref_koff_per_s is not None:
        # KS = RT_drug / RT_ref = (1/koff_drug) / (1/koff_ref) = ref_koff / drug_koff
        kinetic_selectivity = ref_koff_per_s / koff_per_s
    else:
        kinetic_selectivity = 1.0

    occupancy_class = _classify_occupancy(rt_min)

    notes = (
        f"Kd={kd_nM:.3g} nM, RT={rt_min:.2f} min, "
        f"t50_recovery={t50_min:.2f} min, "
        f"class={occupancy_class}"
    )
    if ref_koff_per_s is not None:
        notes += f", KS_vs_ref={kinetic_selectivity:.2f}"

    return ReceptorResidenceResult(
        drug_name=drug_name,
        target_name=target_name,
        kon_M_per_s=kon_M_per_s,
        koff_per_s=koff_per_s,
        kd_nM=kd_nM,
        residence_time_s=rt_s,
        residence_time_min=rt_min,
        residence_time_h=rt_h,
        t50_recovery_min=t50_min,
        kinetic_selectivity_vs_ref=kinetic_selectivity,
        occupancy_class=occupancy_class,
        notes=notes,
    )


def compare_residence_times(drugs: list[dict]) -> list[ReceptorResidenceResult]:
    """Compare residence times for multiple drugs, sorted longest RT first.

    Each dict in `drugs` must contain:
        drug_name (str), target_name (str), kon_M_per_s (float),
        koff_per_s (float), and optionally ref_koff_per_s (float).

    Returns
    -------
    list[ReceptorResidenceResult]
        Sorted by residence_time_h descending (longest first).
    """
    results = []
    for d in drugs:
        res = predict_residence_time(
            drug_name=d["drug_name"],
            target_name=d["target_name"],
            kon_M_per_s=d["kon_M_per_s"],
            koff_per_s=d["koff_per_s"],
            ref_koff_per_s=d.get("ref_koff_per_s"),
        )
        results.append(res)
    results.sort(key=lambda r: r.residence_time_h, reverse=True)
    return results
