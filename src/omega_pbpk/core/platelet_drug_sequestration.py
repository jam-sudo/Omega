"""Phase 391 — Platelet-Mediated Drug Sequestration Model.

Models drug uptake and release by platelets, which can significantly alter
free drug concentration for basic/lipophilic drugs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlateletSequestrationResult:
    """Result of platelet drug sequestration calculation."""

    drug_name: str
    platelet_count_per_uL: float
    drug_type: str  # "basic_lipophilic", "acidic", "neutral"
    total_plasma_conc_mg_L: float
    free_plasma_conc_mg_L: float
    platelet_bound_fraction: float  # 0-1
    apparent_fup: float
    kp_platelet: float
    sequestration_effect_pct: float
    clinical_impact: str  # "none", "minor", "moderate", "major"
    notes: str


def _compute_kp_platelet(drug_type: str, logp: float, pka_base: float | None) -> float:
    """Compute platelet-to-free-plasma concentration ratio."""
    if drug_type == "basic_lipophilic":
        if pka_base is not None:
            # ion trapping: protonated fraction inside acidic platelet granules
            kp = 10.0 * (1.0 + 10.0 ** (pka_base - 7.4))
        else:
            kp = 10.0
        # logP factor
        logp_factor = max(1.0, logp / 2.0)
        kp = kp * logp_factor
    elif drug_type == "acidic":
        kp = 0.5
    else:  # neutral
        if logp > 0:
            kp = 1.0 + logp * 0.3
        else:
            kp = 1.0
    return kp


def _clinical_impact(effect_pct: float) -> str:
    if effect_pct < 5.0:
        return "none"
    elif effect_pct < 20.0:
        return "minor"
    elif effect_pct < 40.0:
        return "moderate"
    else:
        return "major"


def calculate_platelet_sequestration(
    drug_name: str,
    total_plasma_conc_mg_L: float,
    drug_type: str,
    logp: float,
    pka_base: float | None = None,
    platelet_count_per_uL: float = 250000.0,
    fup_baseline: float = 0.1,
) -> PlateletSequestrationResult:
    """Calculate platelet-mediated drug sequestration.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    total_plasma_conc_mg_L:
        Nominal (total) plasma concentration in mg/L.
    drug_type:
        One of "basic_lipophilic", "acidic", or "neutral".
    logp:
        Log P (octanol-water partition coefficient).
    pka_base:
        Basic pKa (only relevant for basic_lipophilic drugs).
    platelet_count_per_uL:
        Platelet count per microlitre of blood (default 250,000/uL).
    fup_baseline:
        Fraction unbound in plasma without platelets (0 < fup <= 1).

    Returns
    -------
    PlateletSequestrationResult
    """
    # --- validation ---
    if total_plasma_conc_mg_L <= 0:
        raise ValueError("total_plasma_conc_mg_L must be > 0")
    valid_types = {"basic_lipophilic", "acidic", "neutral"}
    if drug_type not in valid_types:
        raise ValueError(f"drug_type must be one of {valid_types}, got '{drug_type}'")
    if platelet_count_per_uL <= 0:
        raise ValueError("platelet_count_per_uL must be > 0")
    if not (0 < fup_baseline <= 1.0):
        raise ValueError("fup_baseline must be in (0, 1]")

    # Platelet volume fraction in blood
    # Each platelet ~8 fL = 8e-15 L; count per uL converted to per mL (multiply by 1e3? No.)
    # platelet_count_per_uL * 8e-15 L/platelet * 1e6 uL/L = dimensionless fraction per litre?
    # Formula given: platelet_volume_fraction = platelet_count_per_uL * 8e-9
    # For 250000/uL: 250000 * 8e-9 = 0.002
    platelet_volume_fraction = platelet_count_per_uL * 8e-9

    kp = _compute_kp_platelet(drug_type, logp, pka_base)

    # Platelet bound fraction
    platelet_bound_fraction = (platelet_volume_fraction * kp) / (
        1.0 + platelet_volume_fraction * kp
    )

    # Free plasma concentration
    free_plasma_conc = total_plasma_conc_mg_L * (1.0 - platelet_bound_fraction)

    # Apparent fup
    apparent_fup = fup_baseline * (1.0 - platelet_bound_fraction)
    # Clamp
    apparent_fup = max(0.001, min(apparent_fup, fup_baseline))

    sequestration_effect_pct = platelet_bound_fraction * 100.0
    impact = _clinical_impact(sequestration_effect_pct)

    notes_parts = []
    if drug_type == "basic_lipophilic":
        notes_parts.append("Basic/lipophilic drug subject to lysosomal trapping in platelets.")
    if impact in ("moderate", "major"):
        notes_parts.append(
            f"Platelet sequestration may significantly reduce free drug "
            f"(approx {sequestration_effect_pct:.1f}% bound to platelets)."
        )
    if not notes_parts:
        notes_parts.append("Minimal platelet sequestration expected.")
    notes = " ".join(notes_parts)

    return PlateletSequestrationResult(
        drug_name=drug_name,
        platelet_count_per_uL=platelet_count_per_uL,
        drug_type=drug_type,
        total_plasma_conc_mg_L=total_plasma_conc_mg_L,
        free_plasma_conc_mg_L=free_plasma_conc,
        platelet_bound_fraction=platelet_bound_fraction,
        apparent_fup=apparent_fup,
        kp_platelet=kp,
        sequestration_effect_pct=sequestration_effect_pct,
        clinical_impact=impact,
        notes=notes,
    )


def compare_platelet_counts(
    drug_name: str,
    total_plasma_conc_mg_L: float,
    drug_type: str,
    logp: float,
    pka_base: float | None = None,
    fup_baseline: float = 0.1,
    counts: list[float] | None = None,
) -> list[PlateletSequestrationResult]:
    """Compare sequestration across different platelet counts.

    Parameters
    ----------
    counts:
        List of platelet counts per uL to evaluate. Defaults to
        [50000, 150000, 250000, 500000, 1000000].

    Returns
    -------
    List of PlateletSequestrationResult sorted by free_plasma_conc_mg_L descending
    (highest free concentration first, i.e., lowest platelet count first).
    """
    if counts is None:
        counts = [50000.0, 150000.0, 250000.0, 500000.0, 1000000.0]

    results = []
    for count in counts:
        r = calculate_platelet_sequestration(
            drug_name=drug_name,
            total_plasma_conc_mg_L=total_plasma_conc_mg_L,
            drug_type=drug_type,
            logp=logp,
            pka_base=pka_base,
            platelet_count_per_uL=count,
            fup_baseline=fup_baseline,
        )
        results.append(r)

    results.sort(key=lambda r: r.free_plasma_conc_mg_L, reverse=True)
    return results
