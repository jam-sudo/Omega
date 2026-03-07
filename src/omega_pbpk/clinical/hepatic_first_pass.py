"""Hepatic first-pass extraction using the well-stirred liver model."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "HepaticFirstPassResult",
    "predict_hepatic_first_pass",
    "screen_hepatic_fp",
]


@dataclass(frozen=True)
class HepaticFirstPassResult:
    """Result of hepatic first-pass extraction calculation."""

    drug_name: str
    dose_mg: float
    fa: float
    fg: float
    fh: float
    f_bioavailability: float
    e_hepatic: float
    cl_intrinsic_L_per_h: float
    cl_hepatic_L_per_h: float
    hepatic_blood_flow_L_per_h: float
    first_pass_loss_fraction: float
    bioavailability_class: str
    notes: str


def predict_hepatic_first_pass(
    drug_name: str,
    dose_mg: float = 10.0,
    cl_intrinsic_L_per_h: float = 10.0,
    fu_plasma: float = 0.1,
    fu_mic: float = 0.01,
    rbp: float = 1.0,
    fa: float = 0.95,
    fg: float = 0.9,
    q_hepatic_L_per_h: float = 90.0,
) -> HepaticFirstPassResult:
    """Predict hepatic first-pass extraction for an oral drug.

    Uses the well-stirred liver model:
        E_H = CLint / (Q + CLint)
        fh = 1 - E_H
        F = fa * fg * fh
    """
    # Validation
    if not (0.0 <= fa <= 1.0):
        raise ValueError(f"fa must be in [0, 1], got {fa}")
    if not (0.0 <= fg <= 1.0):
        raise ValueError(f"fg must be in [0, 1], got {fg}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if cl_intrinsic_L_per_h <= 0.0:
        raise ValueError(f"cl_intrinsic_L_per_h must be > 0, got {cl_intrinsic_L_per_h}")

    # Well-stirred model (simplified form as specified)
    e_hepatic = cl_intrinsic_L_per_h / (q_hepatic_L_per_h + cl_intrinsic_L_per_h)
    fh = 1.0 - e_hepatic
    cl_hepatic = q_hepatic_L_per_h * e_hepatic

    # Oral bioavailability
    f_bioavailability = max(0.0, min(1.0, fa * fg * fh))

    # Classification
    if f_bioavailability > 0.6:
        bioavailability_class = "high"
    elif f_bioavailability > 0.3:
        bioavailability_class = "intermediate"
    else:
        bioavailability_class = "low"

    first_pass_loss_fraction = 1.0 - fh

    # Notes based on extraction ratio
    if e_hepatic > 0.7:
        notes = (
            "High hepatic extraction (E_H >0.7) — extensive first-pass; "
            "IV vs oral route may differ significantly"
        )
    elif e_hepatic > 0.3:
        notes = "Moderate hepatic extraction — consider food effect and DDI on CYP"
    else:
        notes = "Low hepatic extraction — first-pass effect minimal"

    return HepaticFirstPassResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fa=fa,
        fg=fg,
        fh=fh,
        f_bioavailability=f_bioavailability,
        e_hepatic=e_hepatic,
        cl_intrinsic_L_per_h=cl_intrinsic_L_per_h,
        cl_hepatic_L_per_h=cl_hepatic,
        hepatic_blood_flow_L_per_h=q_hepatic_L_per_h,
        first_pass_loss_fraction=first_pass_loss_fraction,
        bioavailability_class=bioavailability_class,
        notes=notes,
    )


def screen_hepatic_fp(candidates: list[dict]) -> list[HepaticFirstPassResult]:
    """Screen a list of candidate compounds for hepatic first-pass extraction.

    Each dict must have "name" and "cl_intrinsic_L_per_h"; optionally "fa" and "fg".
    Returns results sorted by f_bioavailability descending.
    """
    results = []
    for candidate in candidates:
        name = candidate["name"]
        cl_int = candidate["cl_intrinsic_L_per_h"]
        fa = candidate.get("fa", 0.95)
        fg = candidate.get("fg", 0.9)
        result = predict_hepatic_first_pass(
            drug_name=name,
            cl_intrinsic_L_per_h=cl_int,
            fa=fa,
            fg=fg,
        )
        results.append(result)

    results.sort(key=lambda r: r.f_bioavailability, reverse=True)
    return results
