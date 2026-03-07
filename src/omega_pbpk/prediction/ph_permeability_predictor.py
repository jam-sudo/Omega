"""Phase 755 — Drug Permeability Prediction by pH.

pH-dependent permeability for ionizable compounds using the Henderson-Hasselbalch equation.
Papp(pH) = Papp_max_neutral × f_neutral(pH)
"""

from __future__ import annotations

from dataclasses import dataclass

# GI segment pH values
GI_SEGMENT_PH: dict[str, float] = {
    "stomach": 1.2,
    "duodenum": 5.5,
    "jejunum": 6.5,
    "ileum": 7.2,
    "colon": 6.8,
}

PHYSIOLOGICAL_PH = 7.4


def _f_neutral(ph: float, pka: float, compound_type: str) -> float:
    """Compute neutral fraction at a given pH using Henderson-Hasselbalch."""
    if compound_type == "acid":
        return 1.0 / (1.0 + 10.0 ** (ph - pka))
    elif compound_type == "base":
        return 1.0 / (1.0 + 10.0 ** (pka - ph))
    else:
        return 1.0


def _logp_factor(logp: float) -> float:
    """Compute logP scaling factor for Papp_max."""
    raw = 1.0 + (logp - 1.0) * 0.3
    return max(0.1, min(3.0, raw))


@dataclass(frozen=True)
class PHPermeabilityResult:
    """Result of pH-dependent permeability prediction."""

    compound_name: str
    pka: float
    compound_type: str
    papp_max_nm_s: float
    papp_stomach_nm_s: float
    papp_duodenum_nm_s: float
    papp_jejunum_nm_s: float
    papp_ileum_nm_s: float
    papp_colon_nm_s: float
    optimal_absorption_segment: str
    f_neutral_physiological_ph: float
    permeability_classification: str
    notes: str


def predict_ph_permeability(
    compound_name: str,
    pka: float | None = None,
    compound_type: str = "neutral",
    papp_max_nm_s: float = 20.0,
    logp: float = 2.0,
) -> PHPermeabilityResult:
    """Predict pH-dependent permeability across GI segments.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    pka:
        Acid/base dissociation constant. None for neutral compounds.
    compound_type:
        One of "acid", "base", or "neutral".
    papp_max_nm_s:
        Maximum permeability for the neutral form (nm/s).
    logp:
        Partition coefficient; modulates papp_max via a logP factor.

    Returns
    -------
    PHPermeabilityResult
    """
    if papp_max_nm_s <= 0:
        raise ValueError(f"papp_max_nm_s must be > 0, got {papp_max_nm_s}")

    # Effective maximum Papp after logP correction
    factor = _logp_factor(logp)
    papp_max_actual = papp_max_nm_s * factor

    # Use -1.0 sentinel when pKa is None
    pka_stored = -1.0 if pka is None else pka

    # Determine effective compound type
    effective_type = compound_type
    if pka is None or compound_type == "neutral":
        effective_type = "neutral"

    # Compute Papp at each GI segment
    def _papp_at(seg_ph: float) -> float:
        fn = _f_neutral(seg_ph, pka_stored, effective_type)
        return papp_max_actual * fn

    papp_stomach = _papp_at(GI_SEGMENT_PH["stomach"])
    papp_duodenum = _papp_at(GI_SEGMENT_PH["duodenum"])
    papp_jejunum = _papp_at(GI_SEGMENT_PH["jejunum"])
    papp_ileum = _papp_at(GI_SEGMENT_PH["ileum"])
    papp_colon = _papp_at(GI_SEGMENT_PH["colon"])

    # Optimal absorption segment (highest Papp)
    segment_papps = {
        "stomach": papp_stomach,
        "duodenum": papp_duodenum,
        "jejunum": papp_jejunum,
        "ileum": papp_ileum,
        "colon": papp_colon,
    }
    optimal_segment = max(segment_papps, key=lambda s: segment_papps[s])

    # f_neutral at physiological pH 7.4 for comparison
    f_neutral_phys = _f_neutral(PHYSIOLOGICAL_PH, pka_stored, effective_type)

    # Permeability classification based on jejunum Papp
    classification = "high" if papp_jejunum > 10.0 else "low"

    # Build notes
    notes_parts = [
        f"logP_factor={factor:.2f}",
        f"papp_max_actual={papp_max_actual:.2f} nm/s",
        f"compound_type={effective_type}",
    ]
    if effective_type != "neutral" and pka is not None:
        notes_parts.append(f"pKa={pka:.2f}")
    notes = "; ".join(notes_parts)

    return PHPermeabilityResult(
        compound_name=compound_name,
        pka=pka_stored,
        compound_type=compound_type,
        papp_max_nm_s=papp_max_nm_s,
        papp_stomach_nm_s=papp_stomach,
        papp_duodenum_nm_s=papp_duodenum,
        papp_jejunum_nm_s=papp_jejunum,
        papp_ileum_nm_s=papp_ileum,
        papp_colon_nm_s=papp_colon,
        optimal_absorption_segment=optimal_segment,
        f_neutral_physiological_ph=f_neutral_phys,
        permeability_classification=classification,
        notes=notes,
    )


def screen_ph_permeability(
    compounds: list[dict],
) -> list[PHPermeabilityResult]:
    """Screen a list of compounds for pH-dependent permeability.

    Each dict in *compounds* is passed as keyword arguments to
    :func:`predict_ph_permeability`.  Results are sorted by jejunum Papp
    descending.

    Parameters
    ----------
    compounds:
        List of parameter dicts, each with at minimum ``compound_name``.

    Returns
    -------
    list[PHPermeabilityResult]
        Sorted by papp_jejunum_nm_s descending.
    """
    results: list[PHPermeabilityResult] = []
    for params in compounds:
        results.append(predict_ph_permeability(**params))
    results.sort(key=lambda r: r.papp_jejunum_nm_s, reverse=True)
    return results
