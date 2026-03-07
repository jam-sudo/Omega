"""pH-dependent solubility profiling.

Predicts intrinsic and ionization-corrected solubility across pH using
the Henderson-Hasselbalch equation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "SolubilityPHProfile",
    "predict_solubility_at_ph",
    "generate_solubility_ph_profile",
    "predict_gi_solubility_risk",
    "screen_solubility_profiles",
]

_VALID_COMPOUND_TYPES = {"acid", "base", "neutral", "amphoteric"}

# GI segment reference pH values
_GI_SEGMENTS = {
    "fasted_stomach": 1.7,
    "fed_stomach": 4.9,
    "duodenum": 6.3,
    "jejunum": 6.8,
    "colon": 7.4,
}


@dataclass
class SolubilityPHProfile:
    """pH-dependent solubility profile for a compound."""

    compound_name: str
    s0_mg_per_mL: float
    pka: float
    compound_type: str
    ph_values: list = field(default_factory=list)
    solubility_mg_per_mL: list = field(default_factory=list)
    s_fasted_stomach: float = 0.0
    s_fed_stomach: float = 0.0
    s_duodenum: float = 0.0
    s_jejunum: float = 0.0
    s_colon: float = 0.0
    min_solubility: float = 0.0
    max_solubility: float = 0.0
    notes: str = ""


def predict_solubility_at_ph(
    s0_mg_per_mL: float,
    pka: float,
    compound_type: str,
    ph: float,
) -> float:
    """Predict solubility at a given pH using Henderson-Hasselbalch.

    Args:
        s0_mg_per_mL: Intrinsic solubility of neutral form (mg/mL).
        pka: pKa of the ionizable group.
        compound_type: One of "acid", "base", "neutral", "amphoteric".
        ph: Target pH value.

    Returns:
        Solubility at pH in mg/mL.
    """
    if s0_mg_per_mL <= 0:
        raise ValueError("s0_mg_per_mL must be > 0")
    if compound_type not in _VALID_COMPOUND_TYPES:
        raise ValueError(
            f"compound_type must be one of {_VALID_COMPOUND_TYPES}, got '{compound_type}'"
        )

    if compound_type == "acid":
        return s0_mg_per_mL * (1.0 + 10.0 ** (ph - pka))
    if compound_type == "base":
        return s0_mg_per_mL * (1.0 + 10.0 ** (pka - ph))
    # neutral or amphoteric: return intrinsic solubility
    return s0_mg_per_mL


def generate_solubility_ph_profile(
    compound_name: str,
    s0_mg_per_mL: float,
    pka: float,
    compound_type: str,
    ph_range: tuple[float, float] = (1.0, 8.0),
    n_points: int = 15,
) -> SolubilityPHProfile:
    """Generate a solubility vs pH profile.

    Args:
        compound_name: Name of the compound.
        s0_mg_per_mL: Intrinsic solubility (mg/mL).
        pka: pKa of the ionizable group.
        compound_type: "acid", "base", "neutral", or "amphoteric".
        ph_range: Tuple (ph_min, ph_max).
        n_points: Number of pH points.

    Returns:
        SolubilityPHProfile dataclass.
    """
    if s0_mg_per_mL <= 0:
        raise ValueError("s0_mg_per_mL must be > 0")
    if compound_type not in _VALID_COMPOUND_TYPES:
        raise ValueError(
            f"compound_type must be one of {_VALID_COMPOUND_TYPES}, got '{compound_type}'"
        )
    if not (0 < ph_range[0] < ph_range[1] <= 14):
        raise ValueError("ph_range must satisfy 0 < ph_range[0] < ph_range[1] <= 14")

    ph_min, ph_max = ph_range
    step = (ph_max - ph_min) / (n_points - 1) if n_points > 1 else 0.0

    ph_values = [ph_min + i * step for i in range(n_points)]
    solubility_values = [
        predict_solubility_at_ph(s0_mg_per_mL, pka, compound_type, ph) for ph in ph_values
    ]

    # GI segment solubilities
    s_fasted = predict_solubility_at_ph(
        s0_mg_per_mL, pka, compound_type, _GI_SEGMENTS["fasted_stomach"]
    )
    s_fed = predict_solubility_at_ph(s0_mg_per_mL, pka, compound_type, _GI_SEGMENTS["fed_stomach"])
    s_duod = predict_solubility_at_ph(s0_mg_per_mL, pka, compound_type, _GI_SEGMENTS["duodenum"])
    s_jej = predict_solubility_at_ph(s0_mg_per_mL, pka, compound_type, _GI_SEGMENTS["jejunum"])
    s_col = predict_solubility_at_ph(s0_mg_per_mL, pka, compound_type, _GI_SEGMENTS["colon"])

    min_sol = min(solubility_values)
    max_sol = max(solubility_values)

    notes = (
        f"Compound: {compound_name} ({compound_type}); pKa={pka}; "
        f"S0={s0_mg_per_mL:.4f} mg/mL; "
        f"pH range {ph_min:.1f}-{ph_max:.1f} ({n_points} points); "
        f"S(jejunum)={s_jej:.4f} mg/mL"
    )

    return SolubilityPHProfile(
        compound_name=compound_name,
        s0_mg_per_mL=s0_mg_per_mL,
        pka=pka,
        compound_type=compound_type,
        ph_values=ph_values,
        solubility_mg_per_mL=solubility_values,
        s_fasted_stomach=s_fasted,
        s_fed_stomach=s_fed,
        s_duodenum=s_duod,
        s_jejunum=s_jej,
        s_colon=s_col,
        min_solubility=min_sol,
        max_solubility=max_sol,
        notes=notes,
    )


def predict_gi_solubility_risk(
    compound_name: str,
    dose_mg: float,
    s0_mg_per_mL: float,
    pka: float,
    compound_type: str,
    dose_volume_mL: float = 250.0,
) -> dict:
    """Compute solubility-limited absorption risk across GI segments.

    The Dose number (Do) = dose_mg / (S_segment_mg_per_mL × volume_mL).
    Do > 1 indicates solubility-limited absorption in that segment.

    Args:
        compound_name: Compound identifier.
        dose_mg: Administered dose (mg).
        s0_mg_per_mL: Intrinsic solubility (mg/mL).
        pka: pKa.
        compound_type: "acid", "base", "neutral", or "amphoteric".
        dose_volume_mL: Volume of fluid for dissolution (default 250 mL).

    Returns:
        Dict with keys: risk_flags (list[str]), do_values (dict), overall_risk (str).
    """
    if s0_mg_per_mL <= 0:
        raise ValueError("s0_mg_per_mL must be > 0")
    if compound_type not in _VALID_COMPOUND_TYPES:
        raise ValueError(
            f"compound_type must be one of {_VALID_COMPOUND_TYPES}, got '{compound_type}'"
        )

    risk_flags: list[str] = []
    do_values: dict[str, float] = {}

    segment_labels = {
        "fasted_stomach": "fasted stomach",
        "fed_stomach": "fed stomach",
        "duodenum": "duodenum",
        "jejunum": "jejunum",
        "colon": "colon",
    }

    for seg_key, seg_ph in _GI_SEGMENTS.items():
        s_seg = predict_solubility_at_ph(s0_mg_per_mL, pka, compound_type, seg_ph)
        do = dose_mg / (s_seg * dose_volume_mL)
        do_values[seg_key] = do
        if do > 1.0:
            risk_flags.append(f"solubility-limited in {segment_labels[seg_key]}")

    # Determine overall risk from number and location of flags
    n_flags = len(risk_flags)
    if n_flags == 0:
        overall_risk = "low"
    elif n_flags <= 2:
        overall_risk = "moderate"
    else:
        overall_risk = "high"

    return {
        "compound_name": compound_name,
        "risk_flags": risk_flags,
        "do_values": do_values,
        "overall_risk": overall_risk,
    }


def screen_solubility_profiles(
    compounds: list[dict],
) -> list[SolubilityPHProfile]:
    """Screen multiple compounds, sorted by jejunum solubility ascending.

    Each compound dict should have:
        compound_name, s0_mg_per_mL, pka, compound_type
        Optionally: ph_range, n_points

    Args:
        compounds: List of compound parameter dicts.

    Returns:
        List of SolubilityPHProfile sorted by s_jejunum ascending.
    """
    profiles: list[SolubilityPHProfile] = []

    for c in compounds:
        profile = generate_solubility_ph_profile(
            compound_name=c["compound_name"],
            s0_mg_per_mL=c["s0_mg_per_mL"],
            pka=c["pka"],
            compound_type=c["compound_type"],
            ph_range=c.get("ph_range", (1.0, 8.0)),
            n_points=c.get("n_points", 15),
        )
        profiles.append(profile)

    profiles.sort(key=lambda p: p.s_jejunum)
    return profiles
