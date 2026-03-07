"""Peptide pharmacokinetics prediction — Phase 468."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PeptidePKResult:
    """PK prediction result for a therapeutic peptide."""

    sequence: str
    mw_Da: float
    n_aa: int
    cl_L_per_h: float
    vd_L: float
    t_half_h: float
    bioavailability_sc_pct: float
    oral_bioavailability_pct: float
    route_scores: dict
    stability_score: float
    notes: str


def estimate_peptide_clearance(
    mw_Da: float,
    is_cyclic: bool = False,
    n_disulfide: int = 0,
    pegylated: bool = False,
    peg_kDa: float = 0.0,
) -> float:
    """Estimate systemic clearance for a peptide (L/h).

    Model:
    - Base CL scales with MW^0.7 (renal + proteolytic)
    - Cyclization reduces CL by ~40%
    - Each disulfide reduces CL by ~10% (stabilisation)
    - PEGylation reduces CL dramatically (MW-dependent)

    Args:
        mw_Da: Molecular weight (Da).
        is_cyclic: Whether the peptide is cyclic.
        n_disulfide: Number of disulfide bonds.
        pegylated: Whether PEGylated.
        peg_kDa: PEG MW (kDa); relevant if pegylated=True.

    Returns:
        Clearance in L/h.
    """
    if mw_Da <= 0:
        raise ValueError("mw_Da must be positive")
    if n_disulfide < 0:
        raise ValueError("n_disulfide must be non-negative")
    if pegylated and peg_kDa < 0:
        raise ValueError("peg_kDa must be non-negative when pegylated=True")

    # Base clearance: reference 70 kg human, reference MW = 500 Da → CL ≈ 3.5 L/h
    ref_mw = 500.0
    ref_cl = 3.5  # L/h
    base_cl = ref_cl * (mw_Da / ref_mw) ** 0.7

    # Cyclization modifier
    if is_cyclic:
        base_cl *= 0.60  # 40% reduction

    # Disulfide modifier: each bond reduces CL by 10%, max 50%
    ds_factor = max(0.50, 1.0 - 0.10 * n_disulfide)
    base_cl *= ds_factor

    # PEGylation modifier
    if pegylated:
        effective_peg = max(1.0, peg_kDa)  # at least 1 kDa effect
        peg_reduction = math.exp(-0.05 * effective_peg)  # exponential decay
        base_cl *= peg_reduction

    return max(0.001, base_cl)


def estimate_peptide_half_life(cl_L_per_h: float, vd_L: float) -> float:
    """Estimate terminal half-life from CL and Vd (hours).

    t_half = (Vd * ln2) / CL
    """
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")
    return (vd_L * math.log(2)) / cl_L_per_h


def peptide_route_suitability(
    mw_Da: float,
    n_aa: int,
    is_cyclic: bool = False,
    stability_score: float = 0.5,
) -> dict:
    """Estimate route suitability scores (0-1) for different administration routes.

    Args:
        mw_Da: Molecular weight (Da).
        n_aa: Number of amino acids.
        is_cyclic: Whether cyclic.
        stability_score: Stability score 0-1 (higher = more stable).

    Returns:
        dict with keys 'iv', 'sc', 'oral', 'inhaled' (scores 0-1).
    """
    if mw_Da <= 0:
        raise ValueError("mw_Da must be positive")
    if n_aa <= 0:
        raise ValueError("n_aa must be positive")
    if not (0.0 <= stability_score <= 1.0):
        raise ValueError("stability_score must be in [0, 1]")

    # IV: always suitable for peptides
    iv_score = 0.95

    # SC: suitable for small to medium peptides; decreases with MW
    sc_score = max(0.05, 0.85 * math.exp(-mw_Da / 20000.0))
    sc_score = min(0.90, sc_score)

    # Oral: very poor for large peptides; cyclization and stability help somewhat
    # Base: max 5% for small peptides, drops sharply
    oral_base = max(0.01, 0.05 * math.exp(-n_aa / 10.0))
    oral_modifier = 1.0 + 0.5 * (1.0 if is_cyclic else 0.0)
    oral_score = min(0.30, oral_base * oral_modifier * (0.5 + stability_score))

    # Inhaled: works for small peptides with decent stability; MW cutoff ~6000 Da
    inhaled_base = max(0.0, 1.0 - mw_Da / 15000.0)
    inhaled_score = min(0.80, inhaled_base * stability_score)

    return {
        "iv": round(iv_score, 3),
        "sc": round(sc_score, 3),
        "oral": round(oral_score, 3),
        "inhaled": round(inhaled_score, 3),
    }


def _compute_stability_score(
    is_cyclic: bool,
    n_disulfide: int,
    pegylated: bool,
    n_aa: int,
) -> float:
    """Compute stability score [0, 1] from structural features."""
    score = 0.4  # baseline linear peptide

    if is_cyclic:
        score += 0.25
    score += min(0.15, n_disulfide * 0.075)
    if pegylated:
        score += 0.15

    # Longer peptides tend to be less stable (more proteolytic sites)
    length_penalty = max(0.0, (n_aa - 10) * 0.005)
    score -= length_penalty

    return max(0.0, min(1.0, score))


def _compute_vd(mw_Da: float, is_cyclic: bool, pegylated: bool, peg_kDa: float) -> float:
    """Estimate volume of distribution (L) for 70 kg human."""
    # Small peptides: Vd ≈ 0.1-0.3 L/kg; larger → approaches plasma volume
    # Reference: plasma volume ~3 L, extracellular ~14 L
    base_vd_L_per_kg = 0.2 * math.exp(-mw_Da / 30000.0) + 0.05
    base_vd_L_per_kg = max(0.05, min(0.3, base_vd_L_per_kg))

    # PEGylation reduces tissue distribution (stays in plasma longer)
    if pegylated:
        effective_peg = max(1.0, peg_kDa)
        peg_factor = math.exp(-0.02 * effective_peg)
        base_vd_L_per_kg *= peg_factor
        base_vd_L_per_kg = max(0.04, base_vd_L_per_kg)

    bw_kg = 70.0
    return base_vd_L_per_kg * bw_kg


def _compute_sc_bioavailability(mw_Da: float, stability_score: float) -> float:
    """Estimate SC bioavailability percent."""
    # Range 50-80% for small peptides, decays with MW
    base_f = 0.75 * math.exp(-mw_Da / 50000.0) + 0.05
    base_f = max(0.05, min(0.80, base_f))
    # Adjust for stability
    f = base_f * (0.6 + 0.4 * stability_score)
    return round(min(80.0, max(1.0, f * 100.0)), 2)


def _compute_oral_bioavailability(n_aa: int, is_cyclic: bool) -> float:
    """Estimate oral bioavailability percent."""
    # Typically <5%; max(0.1, 5 * exp(-n_aa/10))
    base_f = max(0.1, 5.0 * math.exp(-n_aa / 10.0))
    if is_cyclic:
        base_f *= 1.5
    return round(min(10.0, base_f), 3)


def predict_peptide_pk(
    sequence: str,
    mw_Da: float,
    n_aa: int,
    is_cyclic: bool = False,
    n_disulfide: int = 0,
    pegylated: bool = False,
    peg_kDa: float = 0.0,
) -> PeptidePKResult:
    """Predict PK properties of a therapeutic peptide.

    Args:
        sequence: Peptide sequence (one-letter codes or description).
        mw_Da: Molecular weight (Da).
        n_aa: Number of amino acids.
        is_cyclic: Whether the peptide is cyclic.
        n_disulfide: Number of disulfide bonds.
        pegylated: Whether PEGylated.
        peg_kDa: PEG MW (kDa).

    Returns:
        PeptidePKResult dataclass with predicted PK parameters.
    """
    if not sequence:
        raise ValueError("sequence must not be empty")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be positive")
    if n_aa <= 0:
        raise ValueError("n_aa must be positive")
    if n_disulfide < 0:
        raise ValueError("n_disulfide must be non-negative")
    if pegylated and peg_kDa < 0:
        raise ValueError("peg_kDa must be non-negative when pegylated=True")

    cl = estimate_peptide_clearance(
        mw_Da=mw_Da,
        is_cyclic=is_cyclic,
        n_disulfide=n_disulfide,
        pegylated=pegylated,
        peg_kDa=peg_kDa,
    )

    vd = _compute_vd(mw_Da=mw_Da, is_cyclic=is_cyclic, pegylated=pegylated, peg_kDa=peg_kDa)

    t_half = estimate_peptide_half_life(cl_L_per_h=cl, vd_L=vd)

    stability = _compute_stability_score(
        is_cyclic=is_cyclic,
        n_disulfide=n_disulfide,
        pegylated=pegylated,
        n_aa=n_aa,
    )

    route_scores = peptide_route_suitability(
        mw_Da=mw_Da,
        n_aa=n_aa,
        is_cyclic=is_cyclic,
        stability_score=stability,
    )

    f_sc = _compute_sc_bioavailability(mw_Da=mw_Da, stability_score=stability)
    f_oral = _compute_oral_bioavailability(n_aa=n_aa, is_cyclic=is_cyclic)

    notes_parts = [
        f"MW={mw_Da:.0f} Da",
        f"n_aa={n_aa}",
        "cyclic" if is_cyclic else "linear",
    ]
    if n_disulfide > 0:
        notes_parts.append(f"{n_disulfide} S-S bonds")
    if pegylated:
        notes_parts.append(f"PEG {peg_kDa:.0f} kDa")
    notes_parts.append(f"stability={stability:.2f}")

    return PeptidePKResult(
        sequence=sequence,
        mw_Da=mw_Da,
        n_aa=n_aa,
        cl_L_per_h=round(cl, 4),
        vd_L=round(vd, 4),
        t_half_h=round(t_half, 4),
        bioavailability_sc_pct=f_sc,
        oral_bioavailability_pct=f_oral,
        route_scores=route_scores,
        stability_score=round(stability, 4),
        notes="; ".join(notes_parts),
    )
