"""Multi-candidate drug comparison and ranking tool.

Compares 2–5 drug candidates on pharmacokinetic, safety, and composite
scoring dimensions using a 1-compartment analytical PK model and
property-based safety heuristics.  No full PBPK ODE solve is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from omega_pbpk.drugs.drug import Drug

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "safety": 0.35,
    "ddi": 0.20,
    "auc_adequacy": 0.20,
    "pk_quality": 0.15,
    "clearance": 0.10,
}

_RISK_PRIORITY = {"low": 0, "moderate": 1, "high": 2}

# Organ-toxicity thresholds (mirrors organ_toxicity.py)
_DILI_DOSE_THRESHOLD_MG = 100.0
_DILI_FUP_THRESHOLD = 0.1
_HERG_MARGIN_CAUTION = 10.0
_HERG_MARGIN_THRESHOLD = 30.0
_PHOSPHOLIPIDOSIS_LOGP = 2.0

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonCandidate:
    """One candidate in a multi-drug comparison."""

    drug: Drug
    dose_mg: float = 100.0
    route: str = "oral"
    label: str = ""
    herg_ic50_uM: float | None = None


@dataclass(frozen=True)
class CandidatePKProfile:
    """PK profile from 1-compartment analytical model for one candidate."""

    label: str
    drug_name: str
    cmax_mg_L: float
    tmax_h: float
    auc0t_mg_h_L: float
    auc0inf_mg_h_L: float
    t_half_h: float
    cl_apparent_L_per_h: float
    vz_apparent_L: float
    mrt_h: float
    time_h: tuple[float, ...]
    cp_mg_L: tuple[float, ...]


@dataclass(frozen=True)
class CandidateSafetyProfile:
    """Property-based safety profile for one candidate."""

    label: str
    overall_tox_risk: str  # "low", "moderate", "high"
    tox_risk_count: int
    hepatotox_risk: str
    nephrotox_risk: str
    cardiac_risk: str
    pulmonary_risk: str
    ddi_victim_flags: tuple[str, ...]
    therapeutic_index_proxy: float


@dataclass(frozen=True)
class ComparisonDimension:
    """Ranking along a single comparison dimension."""

    dimension: str
    metric_name: str
    values: dict[str, float]
    ranked_labels: list[str]
    direction: str  # "lower_is_better" or "higher_is_better"


@dataclass(frozen=True)
class DrugComparisonResult:
    """Complete multi-candidate comparison result."""

    n_candidates: int
    candidate_labels: list[str]
    pk_profiles: list[CandidatePKProfile]
    safety_profiles: list[CandidateSafetyProfile]
    dimension_rankings: list[ComparisonDimension]
    composite_scores: dict[str, float]
    overall_ranking: list[str]
    best_candidate: str
    decision_summary: str
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PK helpers  (1-compartment analytical — same as crossover_trial.py)
# ---------------------------------------------------------------------------


def _derive_pop_params(drug: Drug) -> tuple[float, float, float]:
    """Derive CL, Vd, ka from Drug object (same logic as crossover_trial)."""
    if drug.clint_hepatic_L_per_h > 0:
        cl = drug.clint_hepatic_L_per_h
    elif drug.clint_scaled_L_per_h > 0:
        cl = drug.clint_scaled_L_per_h
    else:
        cl = 5.0

    vd = drug.mw * 0.15
    vd = max(10.0, min(vd, 200.0))

    if drug.peff > 0:
        ka = drug.peff * 10.0
        ka = max(0.1, min(ka, 5.0))
    else:
        ka = 1.0

    return cl, vd, ka


def _simulate_candidate(
    candidate: ComparisonCandidate,
    t_end_h: float = 24.0,
) -> CandidatePKProfile:
    """Simulate a single candidate using the 1-compartment analytical model.

    Oral: C(t) = (F·D·ka)/(Vd·(ka-ke)) · (exp(-ke·t) - exp(-ka·t))
    IV:   C(t) = (D/Vd) · exp(-ke·t)
    """
    drug = candidate.drug
    dose = candidate.dose_mg
    route = candidate.route
    label = candidate.label or drug.name

    cl, vd, ka = _derive_pop_params(drug)
    ke = cl / vd
    t_half = math.log(2) / ke
    f_bio = 1.0

    if route == "iv":
        cmax = dose / vd
        tmax = 0.0
        auc_inf = dose / cl
    else:
        if abs(ka - ke) < 1e-8:
            tmax = 1.0 / ke
            cmax = (f_bio * dose / vd) * math.exp(-1.0)
        else:
            tmax = math.log(ka / ke) / (ka - ke)
            cmax = (
                (f_bio * dose * ka)
                / (vd * (ka - ke))
                * (math.exp(-ke * tmax) - math.exp(-ka * tmax))
            )
        auc_inf = f_bio * dose / cl

    frac_t = 1.0 - math.exp(-ke * t_end_h)
    auc_0t = auc_inf * frac_t

    # MRT for oral 1-cpt: 1/ke + 1/ka
    mrt = 1.0 / ke + (1.0 / ka if route != "iv" else 0.0)

    # Time-concentration profile
    n_pts = max(int(t_end_h * 10), 100)
    dt = t_end_h / n_pts
    time_pts: list[float] = []
    cp_pts: list[float] = []
    for i in range(n_pts + 1):
        t = i * dt
        if route == "iv":
            c = (dose / vd) * math.exp(-ke * t)
        else:
            if abs(ka - ke) < 1e-8:
                c = (f_bio * dose * ke / vd) * t * math.exp(-ke * t)
            else:
                c = (f_bio * dose * ka) / (vd * (ka - ke)) * (math.exp(-ke * t) - math.exp(-ka * t))
        time_pts.append(round(t, 4))
        cp_pts.append(round(max(c, 0.0), 6))

    return CandidatePKProfile(
        label=label,
        drug_name=drug.name,
        cmax_mg_L=round(cmax, 6),
        tmax_h=round(tmax, 4),
        auc0t_mg_h_L=round(auc_0t, 6),
        auc0inf_mg_h_L=round(auc_inf, 6),
        t_half_h=round(t_half, 4),
        cl_apparent_L_per_h=round(cl, 4),
        vz_apparent_L=round(vd, 4),
        mrt_h=round(mrt, 4),
        time_h=tuple(time_pts),
        cp_mg_L=tuple(cp_pts),
    )


# ---------------------------------------------------------------------------
# Safety helpers  (property-based heuristics)
# ---------------------------------------------------------------------------


def _assess_candidate_safety(
    candidate: ComparisonCandidate,
    pk: CandidatePKProfile,
) -> CandidateSafetyProfile:
    """Estimate organ risks from drug properties and PK (no full PBPK needed)."""
    drug = candidate.drug
    dose = candidate.dose_mg
    label = pk.label

    # --- Hepatotoxicity ---
    if dose > _DILI_DOSE_THRESHOLD_MG and drug.fup < _DILI_FUP_THRESHOLD:
        hepatotox = "high"
    elif dose > _DILI_DOSE_THRESHOLD_MG or drug.logP > 3.0:
        hepatotox = "moderate"
    else:
        hepatotox = "low"

    # --- Nephrotoxicity (dose & logP heuristic) ---
    if dose > 500 and drug.logP > 3.0:
        nephrotox = "high"
    elif dose > 300 or drug.logP > 4.0:
        nephrotox = "moderate"
    else:
        nephrotox = "low"

    # --- Cardiac (hERG margin) ---
    cmax_unbound_uM = pk.cmax_mg_L * drug.fup * 1000.0 / max(drug.mw, 1e-12)
    if candidate.herg_ic50_uM is not None and candidate.herg_ic50_uM > 0:
        herg_margin = candidate.herg_ic50_uM / max(cmax_unbound_uM, 1e-12)
        if herg_margin < _HERG_MARGIN_CAUTION:
            cardiac = "high"
        elif herg_margin < _HERG_MARGIN_THRESHOLD:
            cardiac = "moderate"
        else:
            cardiac = "low"
    else:
        cardiac = "low"  # cannot assess without data

    # --- Pulmonary (logP accumulation proxy) ---
    if drug.logP > 4.0:
        pulmonary = "moderate"
    elif drug.logP > _PHOSPHOLIPIDOSIS_LOGP:
        pulmonary = "low"
    else:
        pulmonary = "low"

    # --- DDI victim flags ---
    ddi_flags: list[str] = []
    fm = drug.fm or {}
    for enzyme, fraction in fm.items():
        if fraction > 0.5:
            ddi_flags.append(f"High {enzyme} dependence (fm={fraction:.2f})")

    # --- Overall risk ---
    risks = [hepatotox, nephrotox, cardiac, pulmonary]
    n_high = sum(1 for r in risks if r == "high")
    n_mod = sum(1 for r in risks if r == "moderate")
    if n_high >= 1:
        overall = "high"
    elif n_mod >= 2:
        overall = "moderate"
    elif n_mod >= 1:
        overall = "moderate"
    else:
        overall = "low"

    # Therapeutic index proxy: higher = better safety window
    # Rough heuristic: inversely proportional to dose and risk count
    ti_proxy = 100.0 / (1.0 + n_high * 30 + n_mod * 10 + dose / 100.0)

    return CandidateSafetyProfile(
        label=label,
        overall_tox_risk=overall,
        tox_risk_count=n_high + n_mod,
        hepatotox_risk=hepatotox,
        nephrotox_risk=nephrotox,
        cardiac_risk=cardiac,
        pulmonary_risk=pulmonary,
        ddi_victim_flags=tuple(ddi_flags),
        therapeutic_index_proxy=round(ti_proxy, 2),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_candidate(
    pk: CandidatePKProfile,
    safety: CandidateSafetyProfile,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute composite score 0–100 with configurable weights.

    Default weights: safety 35%, DDI 20%, AUC adequacy 20%,
                     PK quality 15%, clearance 10%.
    """
    w = dict(_DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # Normalise weights to sum to 1
    total_w = sum(w.values())
    if total_w > 0:
        w = {k: v / total_w for k, v in w.items()}

    # --- Safety sub-score (0–100, higher = safer) ---
    risk_penalty = {"low": 0, "moderate": 15, "high": 35}
    safety_deduct = (
        risk_penalty.get(safety.hepatotox_risk, 0)
        + risk_penalty.get(safety.nephrotox_risk, 0)
        + risk_penalty.get(safety.cardiac_risk, 0)
        + risk_penalty.get(safety.pulmonary_risk, 0)
    )
    safety_score = max(0.0, 100.0 - safety_deduct)

    # --- DDI sub-score (0–100, fewer flags = better) ---
    ddi_score = max(0.0, 100.0 - len(safety.ddi_victim_flags) * 25.0)

    # --- AUC adequacy (0–100) ---
    # Favour AUC in a sweet spot; penalise very low or extremely high
    auc = pk.auc0inf_mg_h_L
    if auc < 0.1:
        auc_score = 10.0
    elif auc > 1000.0:
        auc_score = max(0.0, 100.0 - (auc - 1000.0) / 50.0)
    else:
        auc_score = min(100.0, 50.0 + 50.0 * math.log10(max(auc, 0.1)) / 3.0)

    # --- PK quality (half-life in reasonable range) ---
    t_half = pk.t_half_h
    if 2.0 <= t_half <= 24.0:
        pk_score = 100.0
    elif t_half < 2.0:
        pk_score = max(0.0, 50.0 * t_half)
    else:
        pk_score = max(0.0, 100.0 - (t_half - 24.0) * 2.0)

    # --- Clearance (moderate CL preferred) ---
    cl = pk.cl_apparent_L_per_h
    if 1.0 <= cl <= 30.0:
        cl_score = 100.0
    elif cl < 1.0:
        cl_score = max(0.0, cl * 100.0)
    else:
        cl_score = max(0.0, 100.0 - (cl - 30.0) * 2.0)

    composite = (
        w.get("safety", 0) * safety_score
        + w.get("ddi", 0) * ddi_score
        + w.get("auc_adequacy", 0) * auc_score
        + w.get("pk_quality", 0) * pk_score
        + w.get("clearance", 0) * cl_score
    )
    return round(max(0.0, min(100.0, composite)), 2)


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


def _rank_dimension(
    label_values: dict[str, float],
    direction: str,
    dimension: str,
    metric_name: str,
) -> ComparisonDimension:
    """Rank labels on a single dimension."""
    reverse = direction == "higher_is_better"
    ranked = sorted(label_values, key=lambda lbl: label_values[lbl], reverse=reverse)
    return ComparisonDimension(
        dimension=dimension,
        metric_name=metric_name,
        values={k: round(v, 4) for k, v in label_values.items()},
        ranked_labels=ranked,
        direction=direction,
    )


def _generate_decision_summary(
    ranking: list[str],
    scores: dict[str, float],
    dimensions: list[ComparisonDimension],
) -> str:
    """Generate a human-readable decision summary."""
    best = ranking[0]
    lines = [f"Best candidate: {best} (composite score {scores[best]:.1f}/100)"]
    if len(ranking) > 1:
        lines.append(f"Runner-up: {ranking[1]} (score {scores[ranking[1]]:.1f})")
    lines.append("")
    lines.append("Dimension leaders:")
    for dim in dimensions:
        leader = dim.ranked_labels[0]
        val = dim.values[leader]
        lines.append(f"  {dim.dimension} ({dim.metric_name}): {leader} = {val:.4g}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compare_drugs(
    candidates: list[ComparisonCandidate],
    t_end_h: float = 24.0,
    weights: dict[str, float] | None = None,
) -> DrugComparisonResult:
    """Compare 2–5 drug candidates across PK, safety, and composite dimensions.

    Args:
        candidates: List of 2–5 ComparisonCandidate objects.
        t_end_h: Simulation duration (hours).
        weights: Optional custom scoring weights.

    Returns:
        DrugComparisonResult with per-candidate profiles, dimension rankings,
        composite scores, and overall ranking.

    Raises:
        ValueError: If fewer than 2 or more than 5 candidates provided.
    """
    if len(candidates) < 2 or len(candidates) > 5:
        raise ValueError(
            f"Expected 2–5 candidates, got {len(candidates)}. "
            "Provide between 2 and 5 candidates for comparison."
        )

    warnings: list[str] = []

    # Assign labels if missing
    labelled: list[ComparisonCandidate] = []
    seen_labels: set[str] = set()
    for i, c in enumerate(candidates):
        lbl = c.label or c.drug.name or f"Candidate-{i + 1}"
        if lbl in seen_labels:
            lbl = f"{lbl}-{i + 1}"
        seen_labels.add(lbl)
        labelled.append(
            ComparisonCandidate(
                drug=c.drug,
                dose_mg=c.dose_mg,
                route=c.route,
                label=lbl,
                herg_ic50_uM=c.herg_ic50_uM,
            )
        )

    # Simulate & assess each candidate
    pk_profiles: list[CandidatePKProfile] = []
    safety_profiles: list[CandidateSafetyProfile] = []
    scores: dict[str, float] = {}

    for cand in labelled:
        pk = _simulate_candidate(cand, t_end_h)
        safety = _assess_candidate_safety(cand, pk)
        score = _score_candidate(pk, safety, weights)

        pk_profiles.append(pk)
        safety_profiles.append(safety)
        scores[pk.label] = score

    # Build dimension rankings (8 dimensions)
    dimensions: list[ComparisonDimension] = []

    # 1. Cmax
    dimensions.append(
        _rank_dimension(
            {p.label: p.cmax_mg_L for p in pk_profiles},
            "higher_is_better",
            "exposure",
            "Cmax (mg/L)",
        )
    )

    # 2. AUC
    dimensions.append(
        _rank_dimension(
            {p.label: p.auc0inf_mg_h_L for p in pk_profiles},
            "higher_is_better",
            "total_exposure",
            "AUC0-inf (mg·h/L)",
        )
    )

    # 3. Half-life (closer to ideal ~12h is better; use absolute distance)
    ideal_t_half = 12.0
    dimensions.append(
        _rank_dimension(
            {p.label: abs(p.t_half_h - ideal_t_half) for p in pk_profiles},
            "lower_is_better",
            "half_life_optimality",
            "|t½ - 12h|",
        )
    )

    # 4. Clearance (moderate preferred — distance from 10 L/h)
    ideal_cl = 10.0
    dimensions.append(
        _rank_dimension(
            {p.label: abs(p.cl_apparent_L_per_h - ideal_cl) for p in pk_profiles},
            "lower_is_better",
            "clearance_optimality",
            "|CL - 10 L/h|",
        )
    )

    # 5. Safety score (overall tox risk count — lower is better)
    dimensions.append(
        _rank_dimension(
            {s.label: float(s.tox_risk_count) for s in safety_profiles},
            "lower_is_better",
            "safety",
            "tox_risk_count",
        )
    )

    # 6. Therapeutic index proxy (higher is better)
    dimensions.append(
        _rank_dimension(
            {s.label: s.therapeutic_index_proxy for s in safety_profiles},
            "higher_is_better",
            "therapeutic_index",
            "TI proxy",
        )
    )

    # 7. DDI vulnerability (fewer flags is better)
    dimensions.append(
        _rank_dimension(
            {s.label: float(len(s.ddi_victim_flags)) for s in safety_profiles},
            "lower_is_better",
            "ddi_vulnerability",
            "DDI flag count",
        )
    )

    # 8. Composite score (higher is better)
    dimensions.append(
        _rank_dimension(
            scores,
            "higher_is_better",
            "composite",
            "Composite score",
        )
    )

    # Overall ranking by composite score
    overall_ranking = sorted(scores, key=lambda lbl: scores[lbl], reverse=True)
    best = overall_ranking[0]

    summary = _generate_decision_summary(overall_ranking, scores, dimensions)

    # Warnings
    for s in safety_profiles:
        if s.overall_tox_risk == "high":
            warnings.append(f"{s.label}: overall toxicity risk is HIGH")
        if len(s.ddi_victim_flags) > 0:
            for flag in s.ddi_victim_flags:
                warnings.append(f"{s.label}: {flag}")

    return DrugComparisonResult(
        n_candidates=len(labelled),
        candidate_labels=[c.label for c in labelled],
        pk_profiles=pk_profiles,
        safety_profiles=safety_profiles,
        dimension_rankings=dimensions,
        composite_scores=scores,
        overall_ranking=overall_ranking,
        best_candidate=best,
        decision_summary=summary,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# ASCII table formatter
# ---------------------------------------------------------------------------


def format_comparison_table(result: DrugComparisonResult) -> str:
    """Format comparison result as a readable ASCII table."""
    labels = result.candidate_labels
    col_w = max(18, max(len(lbl) for lbl in labels) + 2)
    header_w = 22

    lines: list[str] = []
    sep = "-" * (header_w + col_w * len(labels) + 2)
    lines.append(sep)
    lines.append(f"{'Metric':<{header_w}}" + "".join(f"{lbl:>{col_w}}" for lbl in labels))
    lines.append(sep)

    # PK rows
    pk_map = {p.label: p for p in result.pk_profiles}
    for metric, attr, fmt in [
        ("Cmax (mg/L)", "cmax_mg_L", ".4f"),
        ("Tmax (h)", "tmax_h", ".2f"),
        ("AUC0-t (mg·h/L)", "auc0t_mg_h_L", ".4f"),
        ("AUC0-inf (mg·h/L)", "auc0inf_mg_h_L", ".4f"),
        ("t½ (h)", "t_half_h", ".2f"),
        ("CL_app (L/h)", "cl_apparent_L_per_h", ".2f"),
        ("Vz_app (L)", "vz_apparent_L", ".1f"),
        ("MRT (h)", "mrt_h", ".2f"),
    ]:
        row = f"{metric:<{header_w}}"
        for lbl in labels:
            val = getattr(pk_map[lbl], attr)
            row += f"{val:>{col_w}{fmt}}"
        lines.append(row)

    lines.append(sep)

    # Safety rows
    sf_map = {s.label: s for s in result.safety_profiles}
    for metric, attr in [
        ("Hepatotox risk", "hepatotox_risk"),
        ("Nephrotox risk", "nephrotox_risk"),
        ("Cardiac risk", "cardiac_risk"),
        ("Pulmonary risk", "pulmonary_risk"),
        ("Overall tox risk", "overall_tox_risk"),
        ("TI proxy", "therapeutic_index_proxy"),
    ]:
        row = f"{metric:<{header_w}}"
        for lbl in labels:
            val = getattr(sf_map[lbl], attr)
            row += f"{str(val):>{col_w}}"
        lines.append(row)

    lines.append(sep)

    # Composite scores & ranking
    row_score = f"{'Composite score':<{header_w}}"
    for lbl in labels:
        row_score += f"{result.composite_scores[lbl]:>{col_w}.1f}"
    lines.append(row_score)

    row_rank = f"{'Rank':<{header_w}}"
    for lbl in labels:
        rank = result.overall_ranking.index(lbl) + 1
        row_rank += f"{rank:>{col_w}}"
    lines.append(row_rank)

    lines.append(sep)
    lines.append(f"Best candidate: {result.best_candidate}")
    lines.append(sep)

    return "\n".join(lines)


__all__ = [
    "ComparisonCandidate",
    "CandidatePKProfile",
    "CandidateSafetyProfile",
    "ComparisonDimension",
    "DrugComparisonResult",
    "compare_drugs",
    "format_comparison_table",
]
