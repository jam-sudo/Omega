"""Red Blood Cell (RBC) Drug Distribution — Phase 888.

Models drug partitioning into red blood cells and its effect on apparent
volume of distribution. High RBC binding reduces free plasma concentration
and alters PK interpretation.

The blood-to-plasma ratio (B/P) determines:
- True Vd_blood vs apparent Vd_plasma
- Clearance correction (plasma vs blood CL)
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "RBCDistributionResult",
    "predict_rbc_distribution",
    "screen_rbc_partitioning",
]


@dataclass(frozen=True)
class RBCDistributionResult:
    """Result of RBC drug distribution prediction."""

    drug_name: str
    kp_rbc: float  # RBC/plasma partition coefficient
    blood_plasma_ratio: float  # B/P ratio
    hematocrit: float  # default 0.45
    apparent_vd_plasma_L: float  # Vd measured using plasma concentration
    vd_blood_L: float  # corrected Vd for RBC distribution
    cl_plasma_L_per_h: float  # input plasma clearance
    cl_blood_L_per_h: float  # CL_blood = CL_plasma / B_P_ratio
    rbc_bound_fraction: float  # fraction of drug in RBCs at equilibrium
    fu_blood: float  # free fraction in blood (accounting for RBC + plasma protein)
    rbc_t_half_h: float  # half-life using plasma CL and blood Vd
    interpretation: str


def predict_rbc_distribution(
    drug_name: str,
    logp: float = 2.0,
    fup: float = 0.5,
    hematocrit: float = 0.45,
    vd_plasma_L: float = 50.0,
    cl_plasma_L_per_h: float = 10.0,
) -> RBCDistributionResult:
    """Predict drug partitioning into red blood cells.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Octanol-water partition coefficient.
    fup:
        Unbound fraction in plasma (0 < fup <= 1).
    hematocrit:
        Volume fraction of red blood cells in blood (0 < hct < 1).
    vd_plasma_L:
        Apparent volume of distribution measured from plasma concentrations (L).
    cl_plasma_L_per_h:
        Plasma clearance (L/h).

    Returns
    -------
    RBCDistributionResult
    """
    if not (0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if not (0 < hematocrit < 1.0):
        raise ValueError(f"hematocrit must be in (0, 1), got {hematocrit}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if cl_plasma_L_per_h <= 0:
        raise ValueError(f"cl_plasma_L_per_h must be > 0, got {cl_plasma_L_per_h}")

    # kp_rbc: sigmoid-like model based on logP and fup
    kp_rbc_raw = 1.0 + (logp / 5.0) * (1.0 - fup) * 3.0
    kp_rbc = max(0.1, min(20.0, kp_rbc_raw))

    # Blood-to-plasma ratio
    bp_ratio = (1.0 - hematocrit) + hematocrit * kp_rbc

    # RBC-bound fraction
    rbc_bound = hematocrit * kp_rbc / bp_ratio

    # Volume of distribution corrected to blood
    vd_blood = vd_plasma_L * bp_ratio

    # Blood clearance
    cl_blood = cl_plasma_L_per_h / bp_ratio

    # Free fraction in blood
    fu_blood = fup / bp_ratio

    # Half-life using blood Vd and blood CL
    rbc_t_half = 0.693 * vd_blood / cl_blood

    # Interpretation
    if kp_rbc > 5:
        interp = (
            "High RBC partitioning — blood:plasma ratio substantially >1. "
            "Use blood CL for in vivo predictions."
        )
    elif kp_rbc > 2:
        interp = "Moderate RBC partitioning — B/P correction recommended."
    else:
        interp = "Minimal RBC partitioning — plasma PK parameters adequate."

    return RBCDistributionResult(
        drug_name=drug_name,
        kp_rbc=kp_rbc,
        blood_plasma_ratio=bp_ratio,
        hematocrit=hematocrit,
        apparent_vd_plasma_L=vd_plasma_L,
        vd_blood_L=vd_blood,
        cl_plasma_L_per_h=cl_plasma_L_per_h,
        cl_blood_L_per_h=cl_blood,
        rbc_bound_fraction=rbc_bound,
        fu_blood=fu_blood,
        rbc_t_half_h=rbc_t_half,
        interpretation=interp,
    )


def screen_rbc_partitioning(
    candidates: list[dict],
) -> list[RBCDistributionResult]:
    """Screen a list of drug candidates for RBC partitioning.

    Parameters
    ----------
    candidates:
        List of dicts with keys:
        - "name" (str, required)
        - "logp" (float, required)
        - "fup" (float, optional, default 0.5)
        - "hematocrit" (float, optional, default 0.45)
        Additional keys are ignored.

    Returns
    -------
    List of RBCDistributionResult sorted by blood_plasma_ratio descending.
    """
    results = []
    for c in candidates:
        result = predict_rbc_distribution(
            drug_name=c["name"],
            logp=c["logp"],
            fup=c.get("fup", 0.5),
            hematocrit=c.get("hematocrit", 0.45),
        )
        results.append(result)
    return sorted(results, key=lambda r: r.blood_plasma_ratio, reverse=True)
