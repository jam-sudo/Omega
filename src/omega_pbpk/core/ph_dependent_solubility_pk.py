"""pH-dependent solubility and absorption along the GI tract (Phase 789)."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PHDependentSolubilityResult",
    "simulate_ph_dependent_absorption",
]

# GI segments: (name, pH, residence_h)
_GI_SEGMENTS = [
    ("Stomach", 1.5, 0.5),
    ("Duodenum", 5.5, 1.0),
    ("Jejunum", 6.5, 2.0),
    ("Ileum", 7.4, 3.0),
]


@dataclass
class PHDependentSolubilityResult:
    """Result of pH-dependent solubility and absorption simulation."""

    drug_name: str
    ionization_type: str  # "acid", "base", "neutral"
    pka: float
    logp: float
    segments: list  # list of dicts per GI segment
    times_h: list
    c_plasma_mg_L: list
    fa_overall: float  # overall fraction absorbed
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    limiting_segment: str  # segment with lowest fa
    notes: str


def _fu_neutral(ionization_type: str, ph: float, pka: float) -> float:
    """Fraction in neutral (unionised) form using Henderson-Hasselbalch."""
    if ionization_type == "acid":
        # fu = 1 / (1 + 10^(pH - pKa))
        exponent = ph - pka
        # clamp to avoid overflow
        exponent = max(-15.0, min(15.0, exponent))
        return 1.0 / (1.0 + 10.0**exponent)
    elif ionization_type == "base":
        # fu = 1 / (1 + 10^(pKa - pH))
        exponent = pka - ph
        exponent = max(-15.0, min(15.0, exponent))
        return 1.0 / (1.0 + 10.0**exponent)
    else:
        # neutral: always 1
        return 1.0


def _s0_from_logp(logp: float) -> float:
    """Intrinsic solubility from logP: S0 = 10^(-logP + 0.5) mg/mL, clamped [0.001, 100]."""
    raw = 10.0 ** (-logp + 0.5)
    return max(0.001, min(100.0, raw))


def _segment_ka(fu: float, logp: float) -> float:
    """Peff-based absorption rate constant for a segment, clamped [0.01, 2.0] /h."""
    # logP factor: moderate logP favours permeation
    logp_factor = max(0.1, min(3.0, (logp + 2.0) / 4.0))
    raw = 0.1 * fu * logp_factor
    return max(0.01, min(2.0, raw))


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def simulate_ph_dependent_absorption(
    drug_name: str,
    dose_mg: float,
    pka: float,
    logp: float,
    ionization_type: str = "acid",
    dose_volume_mL: float = 250.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> PHDependentSolubilityResult:
    """Simulate pH-dependent oral absorption across GI segments.

    Uses Henderson-Hasselbalch ionisation, logP-derived intrinsic solubility,
    and segment-specific Peff-based ka. Forward Euler 1-cpt plasma model with
    sequential gut emptying across Stomach, Duodenum, Jejunum, Ileum.

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose in mg.
        pka: Acid/base pKa value [0, 14].
        logp: Octanol-water partition coefficient.
        ionization_type: "acid", "base", or "neutral".
        dose_volume_mL: Co-administered water volume (mL).
        cl_L_per_h: Systemic clearance (L/h).
        vd_L: Volume of distribution (L).
        t_end_h: Simulation end time (h).
        dt_h: Forward Euler step size (h).

    Returns:
        PHDependentSolubilityResult with plasma PK and per-segment data.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if ionization_type not in ("acid", "base", "neutral"):
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got {ionization_type!r}"
        )

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # Build per-segment absorption profile
    # Each segment absorbs a portion of the remaining gut mass over its residence time
    segment_data = []
    remaining_dose_mg = dose_mg
    total_absorbed_mg = 0.0

    for seg_name, seg_ph, residence_h in _GI_SEGMENTS:
        fu = _fu_neutral(ionization_type, seg_ph, pka)
        s0 = _s0_from_logp(logp)

        # Segment solubility: ionised forms dissolve more readily
        # Acids: fu_neutral decreases at high pH → more ionised → more soluble
        # Bases: fu_neutral decreases at low pH → more ionised → more soluble
        # S = S0 / fu_neutral (unionised fraction reduces apparent solubility)
        fu_safe = max(1e-6, fu)
        seg_solubility = s0 / fu_safe
        seg_solubility = max(0.001, min(10000.0, seg_solubility))

        # Dissolved fraction
        max_dissolvable_mg = seg_solubility * dose_volume_mL
        dissolved_fraction = min(1.0, max_dissolvable_mg / max(1e-9, remaining_dose_mg))

        # ka for this segment
        ka = _segment_ka(fu, logp)

        # Fraction absorbed from segment using first-order kinetics over residence time
        fa_seg = 1.0 - math.exp(-ka * residence_h)
        fa_seg = fa_seg * dissolved_fraction  # only dissolved drug can be absorbed

        absorbed_mg = remaining_dose_mg * fa_seg
        total_absorbed_mg += absorbed_mg
        remaining_dose_mg -= absorbed_mg

        segment_data.append(
            {
                "name": seg_name,
                "pH": seg_ph,
                "residence_h": residence_h,
                "fu_neutral": round(fu, 4),
                "solubility_mg_mL": round(seg_solubility, 4),
                "dissolved_fraction": round(dissolved_fraction, 4),
                "ka_per_h": round(ka, 4),
                "fa_segment": round(fa_seg, 4),
                "absorbed_mg": round(absorbed_mg, 4),
            }
        )

    fa_overall = total_absorbed_mg / dose_mg

    # Identify limiting segment (lowest fa_segment)
    limiting_seg = min(segment_data, key=lambda s: s["fa_segment"])
    limiting_segment = limiting_seg["name"]

    # -----------------------------------------------------------------------
    # Forward Euler 1-cpt plasma simulation
    # Each segment contributes an absorption pulse starting at its onset time
    # Onset times cumulate as we move through GI segments
    # -----------------------------------------------------------------------
    n_steps = int(t_end_h / dt_h) + 1
    times_h = [i * dt_h for i in range(n_steps)]

    # Build absorption input rate A(t) as sum of first-order gut compartments
    # For each segment, gut mass decays exponentially during its residence window
    # and input rate to plasma = ka * gut_mass(t)

    # Segment onset/offset times (cumulative)
    segment_times = []
    t_onset = 0.0
    for seg in segment_data:
        t_offset = t_onset + seg["residence_h"]
        segment_times.append((t_onset, t_offset))
        t_onset = t_offset

    # Simulate
    c_plasma = [0.0] * n_steps
    # gut_mass per segment (mg remaining in that segment at its start)
    # Re-derive properly: remaining dose entering each segment
    entering = [0.0] * len(segment_data)
    rem = dose_mg
    for i, seg in enumerate(segment_data):
        entering[i] = rem
        rem -= seg["absorbed_mg"]

    # gut mass for each segment (dissolved portion available for absorption)
    seg_gut_mass0 = [
        entering[i] * segment_data[i]["dissolved_fraction"] for i in range(len(segment_data))
    ]
    seg_kas = [seg["ka_per_h"] for seg in segment_data]

    # Forward Euler
    for step in range(1, n_steps):
        t = times_h[step - 1]
        c_prev = c_plasma[step - 1]

        # Calculate absorption rate into plasma at time t (mg/h)
        absorption_rate_mg_per_h = 0.0
        for i, (t_on, t_off) in enumerate(segment_times):
            if t_on <= t < t_off:
                dt_in_seg = t - t_on
                gut_mass_t = seg_gut_mass0[i] * math.exp(-seg_kas[i] * dt_in_seg)
                absorption_rate_mg_per_h += seg_kas[i] * gut_mass_t

        # dC/dt = absorption_rate/Vd - ke*C
        absorption_rate_mg_per_L_per_h = absorption_rate_mg_per_h / vd_L
        dcdt = absorption_rate_mg_per_L_per_h - ke * c_prev
        c_plasma[step] = max(0.0, c_prev + dcdt * dt_h)

    cmax = max(c_plasma)
    tmax = times_h[c_plasma.index(cmax)]
    auc = _trapezoidal_auc(times_h, c_plasma)

    # Build notes
    notes_parts = [f"ionization_type={ionization_type}, pKa={pka:.1f}, logP={logp:.1f}"]
    if ionization_type == "acid":
        notes_parts.append("acids dissolve better in intestine (high pH)")
    elif ionization_type == "base":
        notes_parts.append("bases dissolve better in stomach (low pH)")
    else:
        notes_parts.append("neutral drug: pH-independent absorption")
    notes_parts.append(f"fa_overall={fa_overall:.3f}, limiting_segment={limiting_segment}")
    notes = "; ".join(notes_parts)

    return PHDependentSolubilityResult(
        drug_name=drug_name,
        ionization_type=ionization_type,
        pka=pka,
        logp=logp,
        segments=segment_data,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        fa_overall=round(fa_overall, 4),
        cmax_mg_L=round(cmax, 6),
        tmax_h=round(tmax, 4),
        auc_mg_h_per_L=round(auc, 4),
        limiting_segment=limiting_segment,
        notes=notes,
    )
