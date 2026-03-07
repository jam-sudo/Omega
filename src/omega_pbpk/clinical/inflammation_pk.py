"""Inflammation-altered drug pharmacokinetics — Phase 498.

Disease-state altered PK during acute/chronic inflammation (sepsis,
acute-phase response, critical illness).

Physiological basis
-------------------
Inflammation causes cytokine-mediated suppression of hepatic CYP enzymes,
altered protein binding (acute-phase proteins), increased vascular permeability
(Vd increase), and reduced hepatic/renal perfusion.

Key references
--------------
- Carcillo JA et al., AAPS J. 2007;9:E353-61
- Renton KW, Drug Metab Rev. 2005;37:147-65
- Krauss R et al., Eur J Clin Pharmacol. 2020;76:3-11
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "InflammationPKResult",
    "inflammation_pk_scaling",
    "acute_phase_protein_binding",
    "cyp_activity_in_inflammation",
    "simulate_inflamed_pk",
]

_VALID_SEVERITIES = ("mild", "moderate", "severe", "critical")

# Inflammation scaling factors by severity
# (cl_factor, vd_factor, albumin_fu_factor, aag_fu_factor)
# albumin_fu_factor: fu multiplier for acidic/neutral drugs (albumin-bound)
# aag_fu_factor: fu multiplier for basic drugs (AAG-bound; AAG increases)
_SEVERITY_FACTORS: dict[str, dict[str, float]] = {
    "mild": {
        "cl_factor": 0.90,
        "vd_factor": 1.10,
        "albumin_fu_factor": 1.00,  # no change
        "aag_fu_factor": 1.00,
    },
    "moderate": {
        "cl_factor": 0.70,
        "vd_factor": 1.20,
        "albumin_fu_factor": 0.90,  # albumin down → fu down for albumin-bound
        "aag_fu_factor": 1.20,      # AAG up → fu down for basic drugs
    },
    "severe": {
        "cl_factor": 0.50,
        "vd_factor": 1.30,
        "albumin_fu_factor": 0.70,
        "aag_fu_factor": 1.40,
    },
    "critical": {
        "cl_factor": 0.30,
        "vd_factor": 1.50,
        "albumin_fu_factor": 0.50,
        "aag_fu_factor": 1.60,
    },
}

# CYP suppression in severe inflammation (fraction of normal activity)
# Values represent activity at "severe" inflammation
_CYP_SEVERE_ACTIVITY: dict[str, float] = {
    "CYP3A4": 0.40,  # 60% reduction
    "CYP1A2": 0.30,  # 70% reduction
    "CYP2C9": 0.55,  # 45% reduction
    "CYP2C19": 0.60, # 40% reduction
    "CYP2D6": 0.75,  # 25% reduction
    "CYP2E1": 0.70,  # 30% reduction
}
_CYP_DEFAULT_SEVERE_ACTIVITY = 0.75  # default for unlisted CYPs


def _validate_severity(severity: str) -> None:
    if severity not in _VALID_SEVERITIES:
        raise ValueError(
            f"severity must be one of {_VALID_SEVERITIES}, got {severity!r}"
        )


@dataclass(frozen=True)
class InflammationPKResult:
    """Result of inflammation-altered PK scaling.

    Attributes
    ----------
    drug_name : Name of the drug.
    severity : Inflammation severity level.
    cl_inflamed_L_per_h : Inflamed clearance (L/h).
    cl_reduction_pct : Percentage reduction in CL from normal (0–100).
    vd_inflamed_L : Inflamed volume of distribution (L).
    fu_inflamed : Inflamed unbound fraction.
    t_half_inflamed_h : Inflamed terminal half-life (h).
    dose_adjustment_factor : Recommended dose scaling factor (<1 = reduce).
    notes : Summary text.
    """

    drug_name: str
    severity: str
    cl_inflamed_L_per_h: float
    cl_reduction_pct: float
    vd_inflamed_L: float
    fu_inflamed: float
    t_half_inflamed_h: float
    dose_adjustment_factor: float
    notes: str


def inflammation_pk_scaling(
    drug_name: str,
    cl_normal_L_per_h: float,
    vd_L: float,
    severity: str = "moderate",
    cyp_isoforms: list[str] | None = None,
    fu_baseline: float = 0.1,
    protein_type: str = "albumin",
) -> InflammationPKResult:
    """Scale PK parameters for inflammation-induced alterations.

    Parameters
    ----------
    drug_name : Drug name.
    cl_normal_L_per_h : Normal (healthy) clearance (L/h).
    vd_L : Normal volume of distribution (L).
    severity : Inflammation severity ('mild', 'moderate', 'severe', 'critical').
    cyp_isoforms : List of CYP isoforms involved in drug metabolism (e.g., ['CYP3A4']).
        Used to compute additional CYP-mediated CL reduction.
    fu_baseline : Baseline unbound fraction (0–1).
    protein_type : Primary binding protein ('albumin' or 'aag').

    Returns
    -------
    InflammationPKResult
    """
    _validate_severity(severity)
    if cl_normal_L_per_h <= 0:
        raise ValueError(f"cl_normal_L_per_h must be > 0, got {cl_normal_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0.0 < fu_baseline <= 1.0):
        raise ValueError(f"fu_baseline must be in (0, 1], got {fu_baseline}")
    if protein_type not in ("albumin", "aag"):
        raise ValueError(f"protein_type must be 'albumin' or 'aag', got {protein_type!r}")

    factors = _SEVERITY_FACTORS[severity]
    base_cl_factor = factors["cl_factor"]

    # CYP-mediated additional CL reduction
    cyp_cl_factor = 1.0
    if cyp_isoforms:
        cyp_activities = [cyp_activity_in_inflammation(cyp, severity) for cyp in cyp_isoforms]
        if cyp_activities:
            # Use minimum (most restrictive) CYP activity as the rate-limiting factor
            cyp_cl_factor = min(cyp_activities)

    # Combine base inflammation effect with CYP effect
    # CYP factor applies on top of base inflammation CL reduction
    combined_cl_factor = base_cl_factor * cyp_cl_factor

    cl_inflamed = cl_normal_L_per_h * combined_cl_factor
    vd_inflamed = vd_L * factors["vd_factor"]

    # Protein binding change
    if protein_type == "albumin":
        fu_factor = factors["albumin_fu_factor"]
    else:  # aag
        # AAG increases → more binding → fu decreases for basic drugs
        aag_factor = factors["aag_fu_factor"]
        fu_factor = 1.0 / aag_factor if aag_factor > 0 else 1.0

    fu_inflamed = min(1.0, max(1e-6, fu_baseline * fu_factor))

    # Terminal half-life
    t_half = math.log(2) * vd_inflamed / cl_inflamed if cl_inflamed > 0 else float("inf")

    # Dose adjustment: reduce dose proportionally to CL reduction to maintain same exposure
    dose_adjustment_factor = combined_cl_factor

    cl_reduction_pct = (1.0 - combined_cl_factor) * 100.0

    cyp_str = ", ".join(cyp_isoforms) if cyp_isoforms else "none specified"
    notes = (
        f"{drug_name} ({severity} inflammation): CL reduced by {cl_reduction_pct:.1f}%, "
        f"Vd increased to {vd_inflamed:.1f} L, t½={t_half:.1f} h. "
        f"CYP isoforms: {cyp_str}. Dose adjustment factor={dose_adjustment_factor:.2f}."
    )

    return InflammationPKResult(
        drug_name=drug_name,
        severity=severity,
        cl_inflamed_L_per_h=cl_inflamed,
        cl_reduction_pct=cl_reduction_pct,
        vd_inflamed_L=vd_inflamed,
        fu_inflamed=fu_inflamed,
        t_half_inflamed_h=t_half,
        dose_adjustment_factor=dose_adjustment_factor,
        notes=notes,
    )


def acute_phase_protein_binding(
    fu_baseline: float,
    severity: str = "moderate",
    protein_type: str = "albumin",
) -> float:
    """Estimate unbound fraction during acute-phase inflammatory response.

    During inflammation:
    - Albumin decreases (negative acute-phase protein) → fu increases for
      albumin-bound drugs (acidic/neutral)
    - AAG (alpha-1-acid glycoprotein) increases (positive acute-phase protein) →
      fu decreases for AAG-bound drugs (basic)

    Parameters
    ----------
    fu_baseline : Baseline unbound fraction (0–1).
    severity : Inflammation severity ('mild', 'moderate', 'severe', 'critical').
    protein_type : Binding protein type ('albumin' or 'aag').

    Returns
    -------
    float
        Estimated unbound fraction during inflammation.
    """
    _validate_severity(severity)
    if not (0.0 < fu_baseline <= 1.0):
        raise ValueError(f"fu_baseline must be in (0, 1], got {fu_baseline}")
    if protein_type not in ("albumin", "aag"):
        raise ValueError(f"protein_type must be 'albumin' or 'aag', got {protein_type!r}")

    factors = _SEVERITY_FACTORS[severity]

    if protein_type == "albumin":
        # Albumin decreases → bound fraction decreases → fu increases
        # We model this as: fu_inflamed = 1 - (1-fu_baseline)*albumin_factor
        albumin_factor = factors["albumin_fu_factor"]
        bound_baseline = 1.0 - fu_baseline
        bound_inflamed = bound_baseline * albumin_factor
        fu_inflamed = 1.0 - bound_inflamed
    else:
        # AAG increases → bound fraction increases → fu decreases
        aag_factor = factors["aag_fu_factor"]
        bound_baseline = 1.0 - fu_baseline
        bound_inflamed = min(1.0, bound_baseline * aag_factor)
        fu_inflamed = 1.0 - bound_inflamed

    return max(1e-6, min(1.0, fu_inflamed))


def cyp_activity_in_inflammation(
    cyp_isoform: str,
    severity: str = "moderate",
) -> float:
    """Estimate CYP enzyme activity fraction during inflammation.

    Cytokine-mediated (IL-6, TNF-alpha, IL-1beta) suppression of CYP enzymes
    varies by isoform and inflammation severity.

    Parameters
    ----------
    cyp_isoform : CYP isoform string (e.g., 'CYP3A4', 'CYP1A2').
    severity : Inflammation severity ('mild', 'moderate', 'severe', 'critical').

    Returns
    -------
    float
        Fraction of normal CYP activity (0–1). Lower = more suppressed.
    """
    _validate_severity(severity)

    # Get severe activity as the floor
    severe_activity = _CYP_SEVERE_ACTIVITY.get(cyp_isoform, _CYP_DEFAULT_SEVERE_ACTIVITY)

    # Interpolate between 1.0 (no inflammation) and severe_activity
    severity_scale = {
        "mild": 0.15,
        "moderate": 0.40,
        "severe": 0.85,
        "critical": 1.00,
    }
    scale = severity_scale[severity]
    activity = 1.0 - scale * (1.0 - severe_activity)
    return max(0.0, min(1.0, activity))


def simulate_inflamed_pk(
    drug_name: str,
    dose_mg: float,
    cl_normal_L_per_h: float,
    vd_L: float,
    severity: str = "moderate",
    cyp_isoforms: list[str] | None = None,
    route: str = "iv",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> dict[str, object]:
    """Simulate drug concentration-time profile under inflamed conditions.

    Uses a one-compartment model with inflammation-scaled PK parameters.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Administered dose (mg).
    cl_normal_L_per_h : Normal clearance (L/h).
    vd_L : Normal volume of distribution (L).
    severity : Inflammation severity.
    cyp_isoforms : CYP isoforms involved in metabolism.
    route : Route of administration ('iv' or 'oral').
    t_end_h : Simulation end time (h).
    dt_h : Integration step size (h).

    Returns
    -------
    dict with keys:
        'drug_name', 'severity', 'times_h', 'c_inflamed_mg_L',
        'c_normal_mg_L', 'cmax_inflamed', 'cmax_normal', 'auc_inflamed',
        'auc_normal', 'auc_ratio', 'pk_result'
    """
    _validate_severity(severity)
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_normal_L_per_h <= 0:
        raise ValueError(f"cl_normal_L_per_h must be > 0, got {cl_normal_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    pk_result = inflammation_pk_scaling(
        drug_name=drug_name,
        cl_normal_L_per_h=cl_normal_L_per_h,
        vd_L=vd_L,
        severity=severity,
        cyp_isoforms=cyp_isoforms,
    )

    cl_inflamed = pk_result.cl_inflamed_L_per_h
    vd_inflamed = pk_result.vd_inflamed_L

    ke_normal = cl_normal_L_per_h / vd_L
    ke_inflamed = cl_inflamed / vd_inflamed

    # IV bolus initial conditions
    c0_normal = dose_mg / vd_L
    c0_inflamed = dose_mg / vd_inflamed

    n_steps = int(round(t_end_h / dt_h))

    times: list[float] = [0.0]
    c_normal_list: list[float] = [c0_normal]
    c_inflamed_list: list[float] = [c0_inflamed]

    c_n = c0_normal
    c_i = c0_inflamed

    for _ in range(n_steps):
        c_n = max(0.0, c_n - ke_normal * c_n * dt_h)
        c_i = max(0.0, c_i - ke_inflamed * c_i * dt_h)
        times.append(times[-1] + dt_h)
        c_normal_list.append(c_n)
        c_inflamed_list.append(c_i)

    def _trapz(ts: list[float], vs: list[float]) -> float:
        auc = 0.0
        for i in range(1, len(ts)):
            auc += 0.5 * (vs[i] + vs[i - 1]) * (ts[i] - ts[i - 1])
        return auc

    auc_normal = _trapz(times, c_normal_list)
    auc_inflamed = _trapz(times, c_inflamed_list)
    cmax_normal = max(c_normal_list)
    cmax_inflamed = max(c_inflamed_list)
    auc_ratio = auc_inflamed / auc_normal if auc_normal > 0 else float("inf")

    return {
        "drug_name": drug_name,
        "severity": severity,
        "times_h": times,
        "c_inflamed_mg_L": c_inflamed_list,
        "c_normal_mg_L": c_normal_list,
        "cmax_inflamed": cmax_inflamed,
        "cmax_normal": cmax_normal,
        "auc_inflamed": auc_inflamed,
        "auc_normal": auc_normal,
        "auc_ratio": auc_ratio,
        "pk_result": pk_result,
    }
