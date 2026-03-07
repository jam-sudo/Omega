"""Comprehensive oral bioavailability (F) prediction for Phase 788.

F = fa * fg * fh

- fa: fraction absorbed via Caco-2 Peff correlation (Sun 2004)
- fg: gut-wall bioavailability from solubility and P-gp substrate status
- fh: hepatic bioavailability from well-stirred model (Er = fup*CLint / (Qh + fup*CLint))

BCS classification:
  - High permeability: Peff > 2e-4 cm/s
  - High solubility: solubility > dose_in_250mL (dose ~ mw_da * 250 / 1000 mg)

References:
  - Amidon et al. (1995) AAPS PharmSci -- BCS
  - Sun et al. (2004) JPharmacolExpTher -- Caco-2/Peff correlation
  - Rowland & Tozer, Clinical Pharmacokinetics (well-stirred model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PEFF_HIGH_THRESHOLD: float = 2e-4  # cm/s — BCS high permeability
_T_SI_MIN: float = 199.0  # small intestine transit time (min)
_R_SI_CM: float = 1.75  # SI radius (cm)
_QH_DEFAULT_L_PER_H: float = 75.0  # portal/hepatic flow (L/h)


# ---------------------------------------------------------------------------
# Result dataclass (frozen — all scalar/string fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OralBioavailabilityResultP788:
    """Result of comprehensive oral bioavailability prediction (Phase 788)."""

    compound_name: str
    fa: float  # fraction absorbed [0,1]
    fg: float  # gut-wall bioavailability [0,1]
    fh: float  # hepatic bioavailability [0,1]
    f_total: float  # fa * fg * fh [0,1]
    f_total_pct: float  # f_total * 100
    bcs_class: str  # "I", "II", "III", "IV"
    absorption_limited: bool
    gut_limited: bool
    hepatic_limited: bool
    main_limitation: str  # "absorption", "gut_extraction", "hepatic_extraction", "none"
    confidence: str  # "high", "medium", "low"
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _peff_from_logp_psa_mw(logp: float, psa: float, mw: float) -> float:
    """Estimate effective permeability (cm/s) from logP, PSA, MW.

    Simplified Sun 2004-inspired Caco-2 correlation:
      log10(Peff) = -5.0 + 0.5*logP - 0.008*PSA - 0.001*(MW/100)
    Clamped to [1e-7, 1e-3] cm/s.
    """
    log_peff = -5.0 + 0.5 * logp - 0.008 * psa - 0.001 * (mw / 100.0)
    peff = 10.0**log_peff
    return float(max(1e-7, min(1e-3, peff)))


def _compute_fa(peff: float) -> float:
    """Compute fraction absorbed from Peff.

    Absorption number: An = Peff * t_si_s / r_si
    fa = 1 - exp(-2 * An)
    """
    t_si_s = _T_SI_MIN * 60.0  # convert to seconds
    an = peff * t_si_s / _R_SI_CM
    fa = 1.0 - math.exp(-2.0 * an)
    return float(max(0.0, min(1.0, fa)))


def _compute_fg(solubility_mg_L: float, pgp_substrate: bool) -> float:
    """Compute gut-wall bioavailability from solubility and P-gp status.

    Simplified lookup:
      - High solubility (>= 100 mg/L): fg = 0.85
      - Medium solubility (10-100 mg/L): fg = 0.60
      - Low solubility (< 10 mg/L): fg = 0.30
    P-gp substrate reduces fg by 20%.
    """
    if solubility_mg_L >= 100.0:
        fg = 0.85
    elif solubility_mg_L >= 10.0:
        fg = 0.60
    else:
        fg = 0.30

    if pgp_substrate:
        fg = fg * 0.80

    return float(max(0.0, min(1.0, fg)))


def _compute_fh(
    cl_int_hepatic_L_per_h: float,
    fup: float,
    q_portal_L_per_h: float,
) -> float:
    """Compute hepatic bioavailability via well-stirred model.

    Er = fup * CLint / (Qh + fup * CLint)
    fh = 1 - Er
    """
    numerator = fup * cl_int_hepatic_L_per_h
    denominator = q_portal_L_per_h + fup * cl_int_hepatic_L_per_h
    if denominator <= 0.0:
        er = 0.0
    else:
        er = numerator / denominator
    er = float(max(0.0, min(1.0, er)))
    return float(1.0 - er)


def _classify_bcs(peff: float, mw_da: float, solubility_mg_L: float) -> str:
    """Classify BCS class based on permeability and solubility.

    High permeability: Peff >= 2e-4 cm/s
    High solubility: solubility >= dose / 250mL
      where dose ~ mw_da * 250 / 1000 mg (assuming max dose = ~MW-proportional)
      threshold = (mw_da * 250 / 1000 mg) / 0.250 L = mw_da mg/L
    """
    high_perm = peff >= _PEFF_HIGH_THRESHOLD
    sol_threshold_mg_L = mw_da  # simplified: mw_da mg/L
    high_sol = solubility_mg_L >= sol_threshold_mg_L

    if high_perm and high_sol:
        return "I"
    elif high_perm and not high_sol:
        return "II"
    elif not high_perm and high_sol:
        return "III"
    else:
        return "IV"


def _determine_main_limitation(fa: float, fg: float, fh: float) -> str:
    """Determine main limiting step for oral bioavailability."""
    loss_abs = 1.0 - fa
    loss_gut = 1.0 - fg
    loss_hep = 1.0 - fh

    max_loss = max(loss_abs, loss_gut, loss_hep)

    if max_loss < 0.15:
        return "none"
    elif loss_hep == max_loss:
        return "hepatic_extraction"
    elif loss_gut == max_loss:
        return "gut_extraction"
    else:
        return "absorption"


def _assess_confidence(
    logp: float,
    mw_da: float,
    psa: float,
    solubility_mg_L: float,
) -> str:
    """Assess prediction confidence from input data quality."""
    # Drug-like range checks
    in_range = (
        0.0 <= logp <= 6.0 and 100.0 <= mw_da <= 700.0 and psa <= 200.0 and solubility_mg_L > 0.0
    )
    if in_range:
        # Extra confidence for Lipinski-compliant
        if mw_da <= 500 and logp <= 5 and psa <= 120:
            return "high"
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_oral_bioavailability(
    compound_name: str,
    logp: float,
    mw_da: float,
    psa_A2: float,
    hbd_count: int = 0,
    solubility_mg_L: float = 100.0,
    cl_int_hepatic_L_per_h: float = 5.0,
    fup: float = 0.1,
    q_portal_L_per_h: float = 75.0,
    pgp_substrate: bool = False,
) -> OralBioavailabilityResultP788:
    """Predict comprehensive oral bioavailability F = fa * fg * fh.

    Parameters
    ----------
    compound_name : str
        Compound identifier.
    logp : float
        Octanol-water partition coefficient.
    mw_da : float
        Molecular weight (Da). Must be > 0.
    psa_A2 : float
        Polar surface area (Angstrom^2).
    hbd_count : int
        Hydrogen bond donor count (used for notes).
    solubility_mg_L : float
        Aqueous solubility (mg/L).
    cl_int_hepatic_L_per_h : float
        Hepatic intrinsic clearance (L/h).
    fup : float
        Unbound fraction in plasma (0, 1].
    q_portal_L_per_h : float
        Portal/hepatic blood flow (L/h).
    pgp_substrate : bool
        Whether compound is a P-gp substrate.

    Returns
    -------
    OralBioavailabilityResultP788
    """
    # --- Validation ---
    if mw_da <= 0:
        raise ValueError("mw_da must be > 0")
    if not 0.0 < fup <= 1.0:
        raise ValueError("fup must be in (0, 1]")
    if q_portal_L_per_h <= 0:
        raise ValueError("q_portal_L_per_h must be > 0")
    if solubility_mg_L < 0:
        raise ValueError("solubility_mg_L must be >= 0")

    # --- Compute Peff ---
    peff = _peff_from_logp_psa_mw(logp, psa_A2, mw_da)

    # --- fa ---
    fa = _compute_fa(peff)

    # --- fg ---
    fg = _compute_fg(solubility_mg_L, pgp_substrate)

    # --- fh ---
    fh = _compute_fh(cl_int_hepatic_L_per_h, fup, q_portal_L_per_h)

    # --- f_total ---
    f_total = float(max(0.0, min(1.0, fa * fg * fh)))
    f_total_pct = f_total * 100.0

    # --- BCS class ---
    bcs_class = _classify_bcs(peff, mw_da, solubility_mg_L)

    # --- Limitations ---
    absorption_limited = fa < 0.5
    gut_limited = fg < 0.5
    hepatic_limited = fh < 0.5
    main_limitation = _determine_main_limitation(fa, fg, fh)

    # --- Confidence ---
    confidence = _assess_confidence(logp, mw_da, psa_A2, solubility_mg_L)

    # --- Notes ---
    pgp_note = " [P-gp substrate]" if pgp_substrate else ""
    notes = (
        f"Peff={peff:.2e}cm/s, fa={fa:.3f}, fg={fg:.3f}, fh={fh:.3f}, "
        f"F={f_total:.3f} ({f_total_pct:.1f}%), BCS={bcs_class}, "
        f"main_limit={main_limitation}{pgp_note}"
    )

    return OralBioavailabilityResultP788(
        compound_name=compound_name,
        fa=fa,
        fg=fg,
        fh=fh,
        f_total=f_total,
        f_total_pct=f_total_pct,
        bcs_class=bcs_class,
        absorption_limited=absorption_limited,
        gut_limited=gut_limited,
        hepatic_limited=hepatic_limited,
        main_limitation=main_limitation,
        confidence=confidence,
        notes=notes,
    )


def screen_bioavailability(
    compounds: list[dict],
) -> list[OralBioavailabilityResultP788]:
    """Screen a list of compounds for oral bioavailability.

    Each dict must contain keys matching predict_oral_bioavailability parameters.
    Required: compound_name, logp, mw_da, psa_A2.
    Optional: hbd_count, solubility_mg_L, cl_int_hepatic_L_per_h, fup,
              q_portal_L_per_h, pgp_substrate.

    Returns list sorted by f_total descending.
    """
    results = []
    for cmpd in compounds:
        result = predict_oral_bioavailability(
            compound_name=cmpd["compound_name"],
            logp=float(cmpd["logp"]),
            mw_da=float(cmpd["mw_da"]),
            psa_A2=float(cmpd["psa_A2"]),
            hbd_count=int(cmpd.get("hbd_count", 0)),
            solubility_mg_L=float(cmpd.get("solubility_mg_L", 100.0)),
            cl_int_hepatic_L_per_h=float(cmpd.get("cl_int_hepatic_L_per_h", 5.0)),
            fup=float(cmpd.get("fup", 0.1)),
            q_portal_L_per_h=float(cmpd.get("q_portal_L_per_h", 75.0)),
            pgp_substrate=bool(cmpd.get("pgp_substrate", False)),
        )
        results.append(result)

    # Sort by f_total descending
    results.sort(key=lambda r: r.f_total, reverse=True)
    return results


__all__ = [
    "OralBioavailabilityResultP788",
    "predict_oral_bioavailability",
    "screen_bioavailability",
]
