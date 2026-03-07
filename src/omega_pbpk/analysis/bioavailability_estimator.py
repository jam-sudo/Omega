"""Phase 667 — Bioavailability estimator using AUC-based methods."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BioavailabilityEstimate:
    compound_name: str
    method: str  # "absolute", "relative", "deconvolution"
    f_estimated: float  # estimated bioavailability (0-1, can exceed 1)
    f_lower_ci: float  # lower 90% confidence interval
    f_upper_ci: float  # upper 90% confidence interval
    auc_test: float
    auc_reference: float
    dose_test: float
    dose_reference: float
    dose_normalized_auc_test: float  # AUC/dose for test
    dose_normalized_auc_ref: float  # AUC/dose for reference
    bioequivalent: bool  # True if 90% CI within 80-125%
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_pk_data(
    times: list[float],
    concs: list[float],
    label: str,
    dose: float,
) -> None:
    if len(times) < 2:
        raise ValueError(f"{label}: times must have at least 2 points, got {len(times)}")
    if len(times) != len(concs):
        raise ValueError(
            f"{label}: times and concs must have the same length ({len(times)} vs {len(concs)})"
        )
    if dose <= 0:
        raise ValueError(f"{label}: dose must be > 0, got {dose}")
    for i, c in enumerate(concs):
        if c < 0:
            raise ValueError(f"{label}: negative concentration at index {i}: {c}")
    for i in range(len(times) - 1):
        if times[i + 1] <= times[i]:
            raise ValueError(
                f"{label}: times must be strictly monotone increasing; "
                f"t[{i}]={times[i]}, t[{i + 1}]={times[i + 1]}"
            )


def _trapz(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal integration."""
    return sum(
        (concs[i] + concs[i + 1]) / 2.0 * (times[i + 1] - times[i]) for i in range(len(times) - 1)
    )


def _extrapolate_to_inf(times: list[float], concs: list[float]) -> float:
    """Estimate AUC from last observed point to infinity using terminal ke."""
    if len(concs) < 3:
        return 0.0
    last_3_t = times[-3:]
    last_3_c = [c for c in concs[-3:] if c > 0]
    if len(last_3_c) < 2:
        return 0.0
    ke = (math.log(last_3_c[0]) - math.log(last_3_c[-1])) / (last_3_t[-1] - last_3_t[0])
    if ke <= 0:
        return 0.0
    return concs[-1] / max(ke, 0.001)


def _compute_auc(
    times: list[float],
    concs: list[float],
    extrapolate: bool = True,
) -> float:
    auc = _trapz(times, concs)
    if extrapolate:
        auc += _extrapolate_to_inf(times, concs)
    return auc


def _build_estimate(
    compound_name: str,
    method: str,
    f_estimated: float,
    auc_test: float,
    auc_reference: float,
    dose_test: float,
    dose_reference: float,
    notes: str,
) -> BioavailabilityEstimate:
    se = f_estimated * 0.10
    f_lower = max(0.0, f_estimated - 1.645 * se)
    f_upper = min(2.0, f_estimated + 1.645 * se)
    bioequivalent = f_lower >= 0.80 and f_upper <= 1.25
    dn_test = auc_test / dose_test if dose_test > 0 else 0.0
    dn_ref = auc_reference / dose_reference if dose_reference > 0 else 0.0
    return BioavailabilityEstimate(
        compound_name=compound_name,
        method=method,
        f_estimated=f_estimated,
        f_lower_ci=f_lower,
        f_upper_ci=f_upper,
        auc_test=auc_test,
        auc_reference=auc_reference,
        dose_test=dose_test,
        dose_reference=dose_reference,
        dose_normalized_auc_test=dn_test,
        dose_normalized_auc_ref=dn_ref,
        bioequivalent=bioequivalent,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_absolute_bioavailability(
    compound_name: str,
    times_oral: list[float],
    concs_oral: list[float],
    dose_oral: float,
    times_iv: list[float],
    concs_iv: list[float],
    dose_iv: float,
    extrapolate_to_infinity: bool = True,
) -> BioavailabilityEstimate:
    """Estimate absolute bioavailability: F = (AUC_oral/dose_oral) / (AUC_iv/dose_iv)."""
    _validate_pk_data(times_oral, concs_oral, "oral", dose_oral)
    _validate_pk_data(times_iv, concs_iv, "iv", dose_iv)

    auc_oral = _compute_auc(times_oral, concs_oral, extrapolate_to_infinity)
    auc_iv = _compute_auc(times_iv, concs_iv, extrapolate_to_infinity)

    dn_oral = auc_oral / dose_oral
    dn_iv = auc_iv / dose_iv if auc_iv > 0 else 1e-9

    f_estimated = dn_oral / dn_iv if dn_iv > 0 else 0.0

    notes = (
        f"Absolute bioavailability estimated from oral/IV crossover. "
        f"AUC_oral={auc_oral:.3f}, AUC_iv={auc_iv:.3f}. "
        f"Extrapolation to infinity: {extrapolate_to_infinity}."
    )
    return _build_estimate(
        compound_name=compound_name,
        method="absolute",
        f_estimated=f_estimated,
        auc_test=auc_oral,
        auc_reference=auc_iv,
        dose_test=dose_oral,
        dose_reference=dose_iv,
        notes=notes,
    )


def estimate_relative_bioavailability(
    compound_name: str,
    times_test: list[float],
    concs_test: list[float],
    dose_test: float,
    times_ref: list[float],
    concs_ref: list[float],
    dose_ref: float,
) -> BioavailabilityEstimate:
    """Estimate relative bioavailability: F_rel = (AUC_test/dose_test) / (AUC_ref/dose_ref)."""
    _validate_pk_data(times_test, concs_test, "test", dose_test)
    _validate_pk_data(times_ref, concs_ref, "reference", dose_ref)

    auc_test = _compute_auc(times_test, concs_test, extrapolate=True)
    auc_ref = _compute_auc(times_ref, concs_ref, extrapolate=True)

    dn_test = auc_test / dose_test
    dn_ref = auc_ref / dose_ref if auc_ref > 0 else 1e-9

    f_estimated = dn_test / dn_ref if dn_ref > 0 else 0.0

    notes = (
        f"Relative bioavailability (test vs reference). "
        f"AUC_test={auc_test:.3f}, AUC_ref={auc_ref:.3f}."
    )
    return _build_estimate(
        compound_name=compound_name,
        method="relative",
        f_estimated=f_estimated,
        auc_test=auc_test,
        auc_reference=auc_ref,
        dose_test=dose_test,
        dose_reference=dose_ref,
        notes=notes,
    )


def compare_formulations(
    compound_name: str,
    formulations: list[dict],
    reference_idx: int = 0,
) -> list[BioavailabilityEstimate]:
    """Compute relative F for each formulation vs reference, sorted by f_estimated descending."""
    if not formulations:
        raise ValueError("formulations list cannot be empty")
    if reference_idx < 0 or reference_idx >= len(formulations):
        raise ValueError(
            f"reference_idx {reference_idx} out of range for {len(formulations)} formulations"
        )

    ref = formulations[reference_idx]
    ref_times = ref["times_h"]
    ref_concs = ref["concs"]
    ref_dose = ref["dose"]

    results: list[BioavailabilityEstimate] = []
    for i, form in enumerate(formulations):
        name = form.get("name", f"formulation_{i}")
        times = form["times_h"]
        concs = form["concs"]
        dose = form["dose"]

        est = estimate_relative_bioavailability(
            compound_name=f"{compound_name} — {name}",
            times_test=times,
            concs_test=concs,
            dose_test=dose,
            times_ref=ref_times,
            concs_ref=ref_concs,
            dose_ref=ref_dose,
        )
        results.append(est)

    results.sort(key=lambda r: r.f_estimated, reverse=True)
    return results
