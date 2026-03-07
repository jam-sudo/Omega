"""Drug absorption window prediction for oral dosage forms.

Predicts the absorption window and regional absorption for oral drugs using
BCS-based framework with Henderson-Hasselbalch pH-solubility and Peff-based
permeability estimates across GI segments.

References
----------
- Amidon GL et al., Pharm Res. 1995;12(3):413-20 (BCS framework)
- Sun D et al., Pharm Res. 2002;19(10):1417-23 (IVER Peff)
- Lennernas H, J Pharm Pharmacol. 1997;49(7):627-38 (intestinal permeability)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AbsorptionWindowResult:
    """Regional absorption and absorption window prediction for an oral drug.

    Attributes
    ----------
    drug_name : Compound identifier.
    absorption_window_h : Hours of significant absorption.
    primary_absorption_site : GI region with highest fa contribution.
    fa_stomach : Fraction absorbed from stomach.
    fa_duodenum : Fraction absorbed from duodenum.
    fa_jejunum : Fraction absorbed from jejunum.
    fa_ileum : Fraction absorbed from ileum.
    fa_colon : Fraction absorbed from colon.
    fa_total : Total fraction absorbed, clamped to 1.0.
    solubility_limited : True if dissolution is rate-limiting (Do > 1).
    permeability_limited : True if Peff is rate-limiting (An < 1).
    transit_dependent : True if absorbed in a limited transit window (<4h).
    dose_number : Do = dose / (250 mL * Cs).
    absorption_number : An = Peff * 3600 * 2 (jejunum simplification).
    dissolution_number : Dn based on Noyes-Whitney with r=50 um particles.
    bcs_class_prediction : BCS class I, II, III, or IV.
    notes : Human-readable summary.
    """

    drug_name: str
    absorption_window_h: float
    primary_absorption_site: str
    fa_stomach: float
    fa_duodenum: float
    fa_jejunum: float
    fa_ileum: float
    fa_colon: float
    fa_total: float
    solubility_limited: bool
    permeability_limited: bool
    transit_dependent: bool
    dose_number: float
    absorption_number: float
    dissolution_number: float
    bcs_class_prediction: str
    notes: str


# GI segment definitions: (name, pH, transit_h, length_cm, radius_cm)
_SEGMENTS: list[tuple[str, float, float, float, float]] = [
    ("stomach", 1.5, 0.5, 0.0, 3.5),
    ("duodenum", 6.0, 0.1, 25.0, 1.5),
    ("jejunum", 6.5, 1.5, 200.0, 1.5),
    ("ileum", 7.2, 2.0, 300.0, 1.5),
    ("colon", 7.4, 20.0, 150.0, 2.5),
]

_DOSE_VOLUME_ML = 250.0
_VALID_SITES = {s[0] for s in _SEGMENTS}


def _estimate_peff(logp: float, mw_da: float) -> float:
    """Estimate intestinal Peff from logP and MW (cm/s)."""
    raw = 10 ** (0.5 * logp - 0.01 * mw_da + 0.5) * 1e-4
    return max(1e-6, min(0.1, raw))


def _adjusted_solubility(s0: float, ph: float, pka: float, ionization_type: str) -> float:
    """Henderson-Hasselbalch pH-adjusted solubility (mg/mL)."""
    if ionization_type == "acid":
        return s0 * (1.0 + 10 ** (ph - pka))
    if ionization_type == "base":
        return s0 * (1.0 + 10 ** (pka - ph))
    # neutral
    return s0


def predict_absorption_window(
    drug_name: str,
    dose_mg: float,
    pka: float,
    logp: float,
    mw_da: float,
    solubility_mg_mL: float,
    ionization_type: str = "neutral",
    peff_cm_per_s: float | None = None,
) -> AbsorptionWindowResult:
    """Predict absorption window and regional fa for an oral drug.

    Parameters
    ----------
    drug_name : Compound identifier.
    dose_mg : Administered oral dose (mg). Must be > 0.
    pka : Ionisation pKa.
    logp : Octanol-water partition coefficient.
    mw_da : Molecular weight (Da).
    solubility_mg_mL : Intrinsic aqueous solubility at reference pH (mg/mL). >0.
    ionization_type : "acid", "base", or "neutral".
    peff_cm_per_s : Effective intestinal permeability (cm/s). If None, estimated.

    Returns
    -------
    AbsorptionWindowResult

    Raises
    ------
    ValueError
        If dose_mg <= 0 or solubility_mg_mL <= 0.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")

    if peff_cm_per_s is None:
        peff = _estimate_peff(logp, mw_da)
    else:
        peff = float(peff_cm_per_s)

    # Compute fa per segment
    segment_fa: dict[str, float] = {}
    for seg_name, ph, transit_h, _length, radius in _SEGMENTS:
        s_adj = _adjusted_solubility(solubility_mg_mL, ph, pka, ionization_type)
        conc_in_lumen = dose_mg / _DOSE_VOLUME_ML  # mg/mL if all dissolved
        dissolved_frac = min(1.0, s_adj / conc_in_lumen) if conc_in_lumen > 0 else 0.0
        # Permeability absorption: 1 - exp(-2*Peff*t/r) approximation
        # Peff in cm/s, transit_h in h -> convert to seconds
        perm_frac = 1.0 - math.exp(-2.0 * peff * transit_h * 3600.0 / radius)
        fa_seg = max(0.0, min(1.0, dissolved_frac * perm_frac))
        segment_fa[seg_name] = fa_seg

    fa_stomach = segment_fa["stomach"]
    fa_duodenum = segment_fa["duodenum"]
    fa_jejunum = segment_fa["jejunum"]
    fa_ileum = segment_fa["ileum"]
    fa_colon = segment_fa["colon"]
    fa_total = min(1.0, fa_stomach + fa_duodenum + fa_jejunum + fa_ileum + fa_colon)

    # Primary absorption site (highest fa_segment)
    primary = max(segment_fa, key=lambda k: segment_fa[k])

    # Absorption window: sum of transit times for segments where fa > 0.01
    seg_transit = {s[0]: s[2] for s in _SEGMENTS}
    absorption_window_h = sum(seg_transit[k] for k, v in segment_fa.items() if v > 0.01)

    # BCS dimensionless numbers
    dose_number = dose_mg / (_DOSE_VOLUME_ML * solubility_mg_mL)
    # An: jejunum simplified (radius=1.5, transit in s)
    absorption_number = peff * 3600.0 * 2.0
    # Dn: 3 * D * Cs / (rho * r^2) * t_transit  [D=1e-5 cm2/s, rho=1.2 g/mL, r=50um=50e-4 cm]
    _D = 1e-5  # cm2/s
    _RHO = 1.2  # g/mL
    _R_PARTICLE = 50e-4  # cm (50 um)
    _T_TRANSIT_JEJUNUM = 3.5 * 3600.0  # s (3.5h small intestinal transit)
    dissolution_number = 3.0 * _D * solubility_mg_mL / (_RHO * _R_PARTICLE**2) * _T_TRANSIT_JEJUNUM

    solubility_limited = dose_number > 1.0
    permeability_limited = absorption_number < 1.0
    transit_dependent = (primary in ("duodenum", "jejunum")) and absorption_window_h < 4.0

    # BCS class
    if not solubility_limited and not permeability_limited:
        bcs_class = "I"
    elif solubility_limited and not permeability_limited:
        bcs_class = "II"
    elif not solubility_limited and permeability_limited:
        bcs_class = "III"
    else:
        bcs_class = "IV"

    notes = (
        f"BCS Class {bcs_class}. Primary site: {primary}. "
        f"fa_total={fa_total:.3f}. Do={dose_number:.2f}, An={absorption_number:.2f}. "
        f"Window={absorption_window_h:.1f}h."
    )

    return AbsorptionWindowResult(
        drug_name=drug_name,
        absorption_window_h=absorption_window_h,
        primary_absorption_site=primary,
        fa_stomach=fa_stomach,
        fa_duodenum=fa_duodenum,
        fa_jejunum=fa_jejunum,
        fa_ileum=fa_ileum,
        fa_colon=fa_colon,
        fa_total=fa_total,
        solubility_limited=solubility_limited,
        permeability_limited=permeability_limited,
        transit_dependent=transit_dependent,
        dose_number=dose_number,
        absorption_number=absorption_number,
        dissolution_number=dissolution_number,
        bcs_class_prediction=bcs_class,
        notes=notes,
    )


def screen_dose_levels(
    drug_name: str,
    pka: float,
    logp: float,
    mw_da: float,
    solubility_mg_mL: float,
    ionization_type: str,
    doses_mg: list[float],
) -> list[AbsorptionWindowResult]:
    """Screen multiple dose levels and return results sorted by fa_total descending.

    Parameters
    ----------
    drug_name : Compound identifier.
    pka : Ionisation pKa.
    logp : Octanol-water partition coefficient.
    mw_da : Molecular weight (Da).
    solubility_mg_mL : Intrinsic solubility (mg/mL).
    ionization_type : "acid", "base", or "neutral".
    doses_mg : List of dose levels (mg) to evaluate.

    Returns
    -------
    List of AbsorptionWindowResult sorted by fa_total descending.
    """
    results = [
        predict_absorption_window(
            drug_name=drug_name,
            dose_mg=d,
            pka=pka,
            logp=logp,
            mw_da=mw_da,
            solubility_mg_mL=solubility_mg_mL,
            ionization_type=ionization_type,
        )
        for d in doses_mg
    ]
    results.sort(key=lambda r: r.fa_total, reverse=True)
    return results
