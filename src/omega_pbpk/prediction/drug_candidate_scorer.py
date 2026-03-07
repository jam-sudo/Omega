"""Drug candidate multi-dimensional development scorecard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrugCandidateScore:
    compound_name: str
    absorption_score: float  # 0-20
    distribution_score: float  # 0-20
    metabolism_score: float  # 0-20
    excretion_score: float  # 0-20
    toxicity_score: float  # 0-20
    total_score: float  # sum 0-100
    lead_likeness: bool  # total > 60
    optimization_priority: str  # dimension with lowest score
    risk_flags: list  # list of concern strings
    grade: str  # "A" (80+), "B" (60-80), "C" (40-60), "D" (<40)
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _calc_absorption(logP: float, mw: float, psa: float, n_hbd: int) -> float:
    """Calculate absorption score (0-20)."""
    # logP: 5 pts if -1<=logP<=5, else 0
    logp_pts = 5.0 if -1.0 <= logP <= 5.0 else 0.0
    # PSA: 5 pts if <60, 3 pts if 60-120, 0 if >120
    if psa < 60.0:
        psa_pts = 5.0
    elif psa <= 120.0:
        psa_pts = 3.0
    else:
        psa_pts = 0.0
    # HBD: 5 pts if <=3, 2 pts if 4-5, 0 if >5
    if n_hbd <= 3:
        hbd_pts = 5.0
    elif n_hbd <= 5:
        hbd_pts = 2.0
    else:
        hbd_pts = 0.0
    # MW: 5 pts if <350, 3 pts if 350-500, 0 if >500
    if mw < 350.0:
        mw_pts = 5.0
    elif mw <= 500.0:
        mw_pts = 3.0
    else:
        mw_pts = 0.0
    return _clamp(logp_pts + psa_pts + hbd_pts + mw_pts, 0.0, 20.0)


def _calc_distribution(logP: float, fup: float, n_chiral_centers: int) -> float:
    """Calculate distribution score (0-20)."""
    # fup: 5 pts if >0.1, 3 pts if 0.01-0.1, 0 if <0.01
    if fup > 0.1:
        fup_pts = 5.0
    elif fup >= 0.01:
        fup_pts = 3.0
    else:
        fup_pts = 0.0
    # Vd estimate
    vd = 0.7 + logP * 0.5
    if 0.5 < vd < 5.0:
        vd_pts = 5.0
    elif 5.0 <= vd <= 10.0:
        vd_pts = 3.0
    else:
        vd_pts = 0.0
    # Protein binding concern
    pb_pts = 5.0 if fup > 0.05 else 0.0
    # Chiral centers
    if n_chiral_centers <= 2:
        chiral_pts = 5.0
    elif n_chiral_centers <= 4:
        chiral_pts = 3.0
    else:
        chiral_pts = 0.0
    return _clamp(fup_pts + vd_pts + pb_pts + chiral_pts, 0.0, 20.0)


def _calc_metabolism(
    clint_uL_per_min_per_mg: float,
    n_aromatic_rings: int,
    has_structural_alerts: bool,
) -> float:
    """Calculate metabolism score (0-20)."""
    # CLint
    if clint_uL_per_min_per_mg < 10.0:
        clint_pts = 8.0
    elif clint_uL_per_min_per_mg <= 50.0:
        clint_pts = 5.0
    elif clint_uL_per_min_per_mg <= 200.0:
        clint_pts = 2.0
    else:
        clint_pts = 0.0
    # CYP flags: 7 - n_aromatic_rings * 1.5 (min 0)
    cyp_pts = max(0.0, 7.0 - n_aromatic_rings * 1.5)
    # Structural alerts
    alert_pts = 5.0 if not has_structural_alerts else 0.0
    return _clamp(clint_pts + cyp_pts + alert_pts, 0.0, 20.0)


def _calc_excretion(logP: float, solubility_mg_mL: float) -> float:
    """Calculate excretion score (0-20)."""
    base = 10.0
    route_pts = 5.0 if logP < 3.0 else 0.0
    sol_pts = 5.0 if solubility_mg_mL > 0.1 else 0.0
    return _clamp(base + route_pts + sol_pts, 0.0, 20.0)


def _calc_toxicity(has_structural_alerts: bool, herg_risk: bool) -> float:
    """Calculate toxicity score (0-20)."""
    alert_pts = 15.0 if not has_structural_alerts else max(0.0, 15.0 - 5.0)
    herg_pts = 5.0 if not herg_risk else 0.0
    return _clamp(alert_pts + herg_pts, 0.0, 20.0)


def _grade(total: float) -> str:
    if total >= 80.0:
        return "A"
    if total >= 60.0:
        return "B"
    if total >= 40.0:
        return "C"
    return "D"


def _optimization_priority(
    absorption: float,
    distribution: float,
    metabolism: float,
    excretion: float,
    toxicity: float,
) -> str:
    scores = {
        "absorption": absorption,
        "distribution": distribution,
        "metabolism": metabolism,
        "excretion": excretion,
        "toxicity": toxicity,
    }
    return min(scores, key=lambda k: scores[k])


def _collect_risk_flags(
    fup: float,
    herg_risk: bool,
    has_structural_alerts: bool,
    logP: float,
    mw: float,
    psa: float,
    clint_uL_per_min_per_mg: float,
) -> list:
    flags: list[str] = []
    if herg_risk:
        flags.append("High hERG risk")
    if fup < 0.01:
        flags.append("Low fup (<0.01)")
    if has_structural_alerts:
        flags.append("Structural alerts present")
    if logP > 5.0:
        flags.append("High logP (>5) — permeability/selectivity concern")
    if mw > 500.0:
        flags.append("High MW (>500) — oral absorption risk")
    if psa > 120.0:
        flags.append("High PSA (>120) — poor oral absorption")
    if clint_uL_per_min_per_mg > 200.0:
        flags.append("Very high CLint — rapid hepatic clearance")
    return flags


def score_drug_candidate(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int = 2,
    n_hba: int = 4,
    fup: float = 0.1,
    clint_uL_per_min_per_mg: float = 10.0,
    n_aromatic_rings: int = 1,
    has_structural_alerts: bool = False,
    herg_risk: bool = False,
    n_chiral_centers: int = 0,
    solubility_mg_mL: float = 1.0,
    measured_bioavailability: float | None = None,
) -> DrugCandidateScore:
    """Score a drug candidate on a 0-100 ADMET scorecard."""
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if n_hbd < 0:
        raise ValueError("n_hbd must be >= 0")
    if n_aromatic_rings < 0:
        raise ValueError("n_aromatic_rings must be >= 0")
    if not (0.0 < fup <= 1.0):
        raise ValueError("fup must be in (0, 1]")
    if clint_uL_per_min_per_mg < 0:
        raise ValueError("clint_uL_per_min_per_mg must be >= 0")
    if solubility_mg_mL < 0:
        raise ValueError("solubility_mg_mL must be >= 0")

    a_score = _calc_absorption(logP, mw, psa, n_hbd)
    d_score = _calc_distribution(logP, fup, n_chiral_centers)
    m_score = _calc_metabolism(clint_uL_per_min_per_mg, n_aromatic_rings, has_structural_alerts)
    e_score = _calc_excretion(logP, solubility_mg_mL)
    t_score = _calc_toxicity(has_structural_alerts, herg_risk)

    total = a_score + d_score + m_score + e_score + t_score
    grade = _grade(total)
    priority = _optimization_priority(a_score, d_score, m_score, e_score, t_score)
    flags = _collect_risk_flags(
        fup, herg_risk, has_structural_alerts, logP, mw, psa, clint_uL_per_min_per_mg
    )

    ba_note = ""
    if measured_bioavailability is not None:
        ba_note = f" Measured F={measured_bioavailability:.0%}."

    notes = (
        f"{compound_name}: total={total:.1f}/100, grade={grade}. "
        f"Priority for optimization: {priority}.{ba_note}"
    )

    return DrugCandidateScore(
        compound_name=compound_name,
        absorption_score=a_score,
        distribution_score=d_score,
        metabolism_score=m_score,
        excretion_score=e_score,
        toxicity_score=t_score,
        total_score=total,
        lead_likeness=total > 60.0,
        optimization_priority=priority,
        risk_flags=flags,
        grade=grade,
        notes=notes,
    )


def rank_drug_candidates(compounds: list[dict]) -> list[DrugCandidateScore]:
    """Score and rank candidates by total_score descending."""
    if not compounds:
        return []
    results = [score_drug_candidate(**c) for c in compounds]
    results.sort(key=lambda r: r.total_score, reverse=True)
    return results
