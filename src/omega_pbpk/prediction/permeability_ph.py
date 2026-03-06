"""Phase 219 — Drug Permeability pH Profile.

Models how ionization state (Henderson-Hasselbalch) affects membrane
permeability across different GI pH zones.  Only the un-ionized fraction
crosses lipid membranes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Core ionization helpers
# ---------------------------------------------------------------------------


def fraction_unionized(pka: float, ph: float, compound_type: str) -> float:
    """Return fraction of drug in un-ionized form at a given pH.

    Parameters
    ----------
    pka:
        Acid dissociation constant (0 < pKa < 14).
    ph:
        pH of the environment (0 < pH < 14).
    compound_type:
        ``"acid"`` or ``"base"``.

    Returns
    -------
    float in [0, 1].
    """
    compound_type = compound_type.lower()
    if compound_type not in ("acid", "base"):
        raise ValueError("compound_type must be 'acid' or 'base'")

    if compound_type == "acid":
        # fu = 1 / (1 + 10^(pH - pKa))
        return 1.0 / (1.0 + 10.0 ** (ph - pka))
    else:
        # fu = 1 / (1 + 10^(pKa - pH))
        return 1.0 / (1.0 + 10.0 ** (pka - ph))


def effective_permeability(
    papp_cm_s: float,
    pka: float,
    ph: float,
    compound_type: str,
) -> float:
    """Effective membrane permeability accounting for ionization.

    Parameters
    ----------
    papp_cm_s:
        Apparent Caco-2 permeability (cm/s).
    pka:
        pKa of the compound.
    ph:
        Environmental pH.
    compound_type:
        ``"acid"`` or ``"base"``.

    Returns
    -------
    Effective permeability (cm/s) = Papp * fraction_unionized.
    """
    if papp_cm_s <= 0:
        raise ValueError("papp_cm_s must be > 0")

    fu = fraction_unionized(pka, ph, compound_type)
    return papp_cm_s * fu


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

_DEFAULT_PH_RANGE = [1.2, 4.5, 6.5, 7.4, 8.0]


@dataclass
class PhPermeabilityResult:
    """Result of a pH-permeability profile sweep."""

    drug_name: str
    papp_cm_s: float
    pka: float
    compound_type: str

    ph_values: list[float] = field(default_factory=list)
    fraction_unionized_values: list[float] = field(default_factory=list)
    effective_permeability_cm_s: list[float] = field(default_factory=list)

    max_peff_cm_s: float = 0.0
    ph_at_max_peff: float = 0.0

    absorption_window: str = ""
    bcs_class_permeability: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Profile and optimisation functions
# ---------------------------------------------------------------------------

_SEGMENT_LABELS = {
    1.2: "stomach",
    4.5: "duodenum",
    6.5: "jejunum",
    7.4: "ileum",
    8.0: "colon",
}

_STOMACH_PH = {1.2}
_SMALL_INTESTINE_PH = {4.5, 6.5, 7.4}
_COLON_PH = {8.0}


def ph_permeability_profile(
    drug_name: str,
    papp_cm_s: float,
    pka: float,
    compound_type: str,
    ph_range: list[float] | None = None,
) -> PhPermeabilityResult:
    """Compute permeability profile across GI pH zones.

    Parameters
    ----------
    drug_name:
        Identifier string.
    papp_cm_s:
        Intrinsic (fully un-ionized) apparent permeability (cm/s).
    pka:
        Drug pKa.
    compound_type:
        ``"acid"`` or ``"base"``.
    ph_range:
        List of pH values to evaluate.  Defaults to
        ``[1.2, 4.5, 6.5, 7.4, 8.0]``.

    Returns
    -------
    :class:`PhPermeabilityResult`
    """
    # --- input validation ---
    if papp_cm_s <= 0:
        raise ValueError("papp_cm_s must be > 0")
    if not (0 < pka < 14):
        raise ValueError("pka must be in (0, 14)")
    compound_type = compound_type.lower()
    if compound_type not in ("acid", "base"):
        raise ValueError("compound_type must be 'acid' or 'base'")

    if ph_range is None:
        ph_range = list(_DEFAULT_PH_RANGE)
    else:
        ph_range = list(ph_range)

    if len(ph_range) < 2:
        raise ValueError("ph_range must contain at least 2 pH values")

    # --- compute profile ---
    fu_list: list[float] = []
    peff_list: list[float] = []
    for ph in ph_range:
        fu = fraction_unionized(pka, ph, compound_type)
        fu_list.append(fu)
        peff_list.append(papp_cm_s * fu)

    max_peff = max(peff_list)
    ph_at_max = ph_range[peff_list.index(max_peff)]

    # --- absorption window: where Peff > 50 % of max ---
    threshold = 0.5 * max_peff
    high_ph_set = {ph for ph, peff in zip(ph_range, peff_list, strict=True) if peff >= threshold}

    in_stomach = bool(high_ph_set & _STOMACH_PH)
    in_si = bool(high_ph_set & _SMALL_INTESTINE_PH)
    in_colon = bool(high_ph_set & _COLON_PH)

    if in_stomach and in_si and in_colon:
        absorption_window = "all_segments"
    elif in_stomach and not in_si and not in_colon:
        absorption_window = "stomach"
    elif in_si:
        absorption_window = "small_intestine"
    elif in_colon:
        absorption_window = "colon"
    else:
        absorption_window = "all_segments"  # fallback

    # --- BCS permeability class ---
    bcs_class = "high" if max_peff > 1e-4 else "low"

    # --- notes ---
    notes_parts = [
        f"pKa={pka:.2f} ({compound_type})",
        f"max Peff={max_peff:.3e} cm/s at pH={ph_at_max:.1f}",
        f"BCS permeability: {bcs_class}",
    ]
    notes = "; ".join(notes_parts)

    return PhPermeabilityResult(
        drug_name=drug_name,
        papp_cm_s=papp_cm_s,
        pka=pka,
        compound_type=compound_type,
        ph_values=ph_range,
        fraction_unionized_values=fu_list,
        effective_permeability_cm_s=peff_list,
        max_peff_cm_s=max_peff,
        ph_at_max_peff=ph_at_max,
        absorption_window=absorption_window,
        bcs_class_permeability=bcs_class,
        notes=notes,
    )


def optimal_absorption_ph(
    drug_name: str,
    papp_cm_s: float,
    pka: float,
    compound_type: str,
) -> float:
    """Return pH at which effective permeability is maximised.

    Sweeps pH 1.0 to 8.0 in steps of 0.1 and returns the pH giving the
    highest effective permeability.
    """
    if papp_cm_s <= 0:
        raise ValueError("papp_cm_s must be > 0")
    if not (0 < pka < 14):
        raise ValueError("pka must be in (0, 14)")
    compound_type = compound_type.lower()
    if compound_type not in ("acid", "base"):
        raise ValueError("compound_type must be 'acid' or 'base'")

    ph_grid = [round(1.0 + i * 0.1, 2) for i in range(71)]  # 1.0 … 8.0
    best_ph = ph_grid[0]
    best_peff = -1.0
    for ph in ph_grid:
        peff = effective_permeability(papp_cm_s, pka, ph, compound_type)
        if peff > best_peff:
            best_peff = peff
            best_ph = ph
    return best_ph
