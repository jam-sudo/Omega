"""Phase 1059 — Drug Sublingual Bioavailability Predictor.

Predicts sublingual bioavailability from physicochemical properties only
(no ODE simulation). Distinct from core/sublingual_pbpk.py.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_DISSOLUTION = {"fast", "moderate", "slow"}
_VALID_IONIZATION = {"acid", "base", "neutral", "zwitterion"}

_DISSOLUTION_FACTORS = {
    "fast": 1.0,
    "moderate": 0.7,
    "slow": 0.4,
}

_BIO_CLASSES = ("excellent", "good", "moderate", "poor")


@dataclass(frozen=True)
class SublingualBioavailabilityResult:
    """Result of sublingual bioavailability prediction."""

    drug_name: str
    mw_Da: float
    logp: float
    fa_sublingual: float
    f_portal_bypass: float
    f_systemic: float
    onset_time_min: float
    duration_effect_h: float
    mucosal_permeability_cm_s: float
    suitable_for_sl: bool
    bioavailability_class: str
    notes: str


def _validate_inputs(
    mw_Da: float,
    dose_mg: float,
    cl_hepatic: float,
    q_hepatic: float,
    dissolution_rate: str,
    ionization_type: str,
) -> None:
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_hepatic <= 0:
        raise ValueError(f"cl_hepatic_L_per_h must be > 0, got {cl_hepatic}")
    if q_hepatic <= 0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0, got {q_hepatic}")
    if dissolution_rate not in _VALID_DISSOLUTION:
        raise ValueError(
            f"dissolution_rate must be one of {sorted(_VALID_DISSOLUTION)}, got {dissolution_rate!r}"
        )
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION)}, got {ionization_type!r}"
        )


def predict_sublingual_bioavailability(
    drug_name: str,
    mw_Da: float,
    logp: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    dose_mg: float = 1.0,
    dissolution_rate: str = "fast",
    cl_hepatic_L_per_h: float = 5.0,
    q_hepatic_L_per_h: float = 90.0,
) -> SublingualBioavailabilityResult:
    """Predict sublingual bioavailability from physicochemical properties.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    mw_Da:
        Molecular weight in Daltons.
    logp:
        Octanol-water partition coefficient.
    pka:
        Acid/base pKa value.
    ionization_type:
        One of "acid", "base", "neutral", "zwitterion".
    dose_mg:
        Dose in milligrams.
    dissolution_rate:
        Film/tablet dissolution speed: "fast", "moderate", or "slow".
    cl_hepatic_L_per_h:
        Hepatic clearance (L/h).
    q_hepatic_L_per_h:
        Hepatic blood flow (L/h).

    Returns
    -------
    SublingualBioavailabilityResult
    """
    _validate_inputs(
        mw_Da,
        dose_mg,
        cl_hepatic_L_per_h,
        q_hepatic_L_per_h,
        dissolution_rate,
        ionization_type,
    )

    # Sublingual pH = 6.8
    sl_ph = 6.8

    # Ionized fraction at SL pH
    if ionization_type == "acid":
        fi = 1.0 / (1.0 + 10 ** (pka - sl_ph))
    elif ionization_type == "base":
        fi = 1.0 / (1.0 + 10 ** (sl_ph - pka))
    else:
        # neutral or zwitterion: treat as neutral for SL model
        fi = 0.0

    unionized = 1.0 - fi

    # Mucosal permeability (buccal/SL)
    papp_log = 0.5 * logp - 0.01 * mw_Da - 2.0
    papp = 10**papp_log  # cm/s
    papp = max(1e-7, min(0.01, papp))

    # Ionization correction
    peff = papp * (unionized + fi * 0.01)  # cm/s

    # Dissolution factor
    diss_factor = _DISSOLUTION_FACTORS[dissolution_rate]

    # fa_sl: area=100 cm², MW penalty
    fa_sl = peff * 3600.0 * 100.0 * diss_factor / (1.0 + mw_Da / 500.0)
    fa_sl = max(0.01, min(0.95, fa_sl))

    # Portal bypass (SL → systemic venous)
    f_portal_bypass = 0.85

    # Hepatic first-pass for 15% via portal
    e_h = min(0.99, cl_hepatic_L_per_h / q_hepatic_L_per_h)
    f_hepatic = 1.0 - e_h

    f_systemic = fa_sl * (f_portal_bypass + (1.0 - f_portal_bypass) * f_hepatic)

    # Onset time (2-20 min range)
    onset_time_min = max(2.0, 20.0 / (peff * 1e4 + 1.0))

    # Duration of effect
    duration_effect_h = max(0.5, min(8.0, 2.0 + logp * 0.2))

    # Suitability and class
    suitable_for_sl = bool(fa_sl > 0.3 and mw_Da < 500.0)

    if f_systemic > 0.7:
        bio_class = "excellent"
    elif f_systemic > 0.5:
        bio_class = "good"
    elif f_systemic > 0.3:
        bio_class = "moderate"
    else:
        bio_class = "poor"

    # Notes
    note_parts = [
        f"Peff={peff:.2e} cm/s",
        f"ionized_fraction={fi:.2f}",
        f"E_hepatic={e_h:.2f}",
    ]
    if not suitable_for_sl:
        if mw_Da >= 500.0:
            note_parts.append("MW>=500 limits SL absorption")
        if fa_sl <= 0.3:
            note_parts.append("low fa_sl (<0.3)")
    else:
        note_parts.append("suitable for sublingual delivery")

    notes = "; ".join(note_parts)

    return SublingualBioavailabilityResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logp=logp,
        fa_sublingual=fa_sl,
        f_portal_bypass=f_portal_bypass,
        f_systemic=f_systemic,
        onset_time_min=onset_time_min,
        duration_effect_h=duration_effect_h,
        mucosal_permeability_cm_s=papp,
        suitable_for_sl=suitable_for_sl,
        bioavailability_class=bio_class,
        notes=notes,
    )
