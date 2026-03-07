"""Phase 1057 — Drug Absorption Window Simulator.

Simulates the GI absorption window dynamically over time with transit.
Tracks drug bolus position along GI tract and cumulative fraction absorbed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AbsorptionWindowResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    position_cm: list = field(default_factory=list)
    fraction_absorbed: list = field(default_factory=list)
    absorption_rate_mg_h: list = field(default_factory=list)
    c_portal_mg_L: list = field(default_factory=list)
    total_fa: float = 0.0
    fa_small_intestine: float = 0.0
    fa_colon: float = 0.0
    absorption_window_h: float = 0.0
    tmax_absorption_h: float = 0.0
    notes: str = ""


_VALID_IONIZATION_TYPES = {"acid", "base", "neutral", "zwitterion"}

# Portal blood flow (L/h)
_Q_PORTAL_L_H = 72.0
# SI length (cm)
_SI_LENGTH_CM = 600.0
# Colon length (cm)
_COLON_LENGTH_CM = 150.0
# SI radius (cm)
_SI_RADIUS_CM = 1.75
# Physiological cap on ka_SI (/h) — prevents numerical blow-up
_KA_SI_MAX = 5.0


def _compute_peff(logp: float, mw_Da: float) -> float:
    """Compute effective permeability (cm/s) from physicochemical properties."""
    peff = 10 ** (-0.4 + 0.3 * logp - 0.003 * mw_Da)
    return max(1e-6, min(0.1, peff))


def _ionization_fraction(pka: float, ionization_type: str, ph: float) -> float:
    """Fraction ionized at given pH."""
    if ionization_type == "acid":
        return 1.0 / (1.0 + 10 ** (pka - ph))
    elif ionization_type == "base":
        return 1.0 / (1.0 + 10 ** (ph - pka))
    else:
        return 0.0


def simulate_absorption_window(
    drug_name: str,
    dose_mg: float,
    logp: float,
    mw_Da: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    fed_state: bool = False,
    si_transit_h: float = 3.0,
    colon_transit_h: float = 24.0,
    peff_cm_s: float = 0.0,
    dose_volume_mL: float = 240.0,
    t_end_h: float = 30.0,
    dt_h: float = 0.1,
) -> AbsorptionWindowResult:
    """Simulate the GI absorption window dynamically over time with transit.

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose in mg.
        logp: Lipophilicity (log octanol-water partition coefficient).
        mw_Da: Molecular weight in Da.
        pka: Acid/base dissociation constant.
        ionization_type: One of "acid", "base", "neutral", "zwitterion".
        fed_state: If True, SI transit time is slowed 1.5x.
        si_transit_h: Small intestine transit time (h).
        colon_transit_h: Colon transit time (h).
        peff_cm_s: Effective permeability in cm/s (0 = compute from logP/MW).
        dose_volume_mL: Volume of water taken with dose (mL).
        t_end_h: End time of simulation (h).
        dt_h: Time step (h).

    Returns:
        AbsorptionWindowResult with absorption profile over time.

    Raises:
        ValueError: If invalid parameters are provided.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if si_transit_h <= 0:
        raise ValueError(f"si_transit_h must be > 0, got {si_transit_h}")
    if colon_transit_h <= 0:
        raise ValueError(f"colon_transit_h must be > 0, got {colon_transit_h}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got {ionization_type!r}"
        )

    # Fed state: slow transit
    if fed_state:
        si_transit_h = si_transit_h * 1.5

    # Compute permeability
    if peff_cm_s <= 0:
        peff_cm_s = _compute_peff(logp, mw_Da)

    # Ionization adjustment at intestinal pH 6.8
    fi = _ionization_fraction(pka, ionization_type, ph=6.8)
    peff_effective_cm_s = peff_cm_s * (1.0 - 0.5 * fi)

    # Convert to /h and compute ka for SI
    # ka (1/h) = Peff_cm_s * 3600 * 2π * r_SI / dose_volume_L
    # Capped at physiological range to prevent ODE stiffness
    dose_volume_L = dose_volume_mL / 1000.0
    ka_si_raw = peff_effective_cm_s * 3600.0 * 2.0 * math.pi * _SI_RADIUS_CM / dose_volume_L
    ka_si = max(1e-4, min(_KA_SI_MAX, ka_si_raw))

    # Colon permeability: 20% of SI
    ka_colon = 0.2 * ka_si

    # Transit velocities
    v_si_cm_h = _SI_LENGTH_CM / si_transit_h
    v_colon_cm_h = _COLON_LENGTH_CM / colon_transit_h

    # ODE simulation (Forward Euler)
    n_steps = max(1, int(t_end_h / dt_h))

    times_h = []
    position_cm = []
    fraction_absorbed = []
    absorption_rate_mg_h = []
    c_portal_mg_L = []

    m_unabsorbed = dose_mg
    absorbed_si = 0.0
    absorbed_colon = 0.0
    peak_rate = 0.0
    tmax_absorption_h = 0.0

    for step in range(n_steps + 1):
        t = step * dt_h

        # Position
        if t * v_si_cm_h < _SI_LENGTH_CM:
            pos = t * v_si_cm_h
            in_si = True
            in_colon = False
        else:
            t_in_colon = t - si_transit_h
            pos = _SI_LENGTH_CM + max(0.0, t_in_colon) * v_colon_cm_h
            in_si = False
            in_colon = pos < (_SI_LENGTH_CM + _COLON_LENGTH_CM)

        pos = min(pos, _SI_LENGTH_CM + _COLON_LENGTH_CM)

        # Current ka
        if in_si:
            ka_current = ka_si
        elif in_colon:
            ka_current = ka_colon
        else:
            ka_current = 0.0

        # Absorption rate (mg/h)
        rate = ka_current * m_unabsorbed
        fa = (dose_mg - m_unabsorbed) / dose_mg

        # Portal concentration
        c_portal = rate / _Q_PORTAL_L_H

        times_h.append(t)
        position_cm.append(pos)
        fraction_absorbed.append(fa)
        absorption_rate_mg_h.append(rate)
        c_portal_mg_L.append(max(0.0, c_portal))

        if rate > peak_rate:
            peak_rate = rate
            tmax_absorption_h = t

        # Forward Euler ODE step
        if step < n_steps:
            d_abs_si = (ka_si * m_unabsorbed * dt_h) if in_si else 0.0
            d_abs_colon = (ka_colon * m_unabsorbed * dt_h) if in_colon else 0.0
            dm = -(ka_current * m_unabsorbed * dt_h)
            m_unabsorbed = max(0.0, m_unabsorbed + dm)
            absorbed_si += d_abs_si
            absorbed_colon += d_abs_colon

    total_fa = (dose_mg - m_unabsorbed) / dose_mg
    total_fa = min(1.0, max(0.0, total_fa))

    fa_si = min(1.0, absorbed_si / dose_mg)
    fa_colon = min(1.0, absorbed_colon / dose_mg)

    # Ensure fa_si + fa_colon doesn't exceed total_fa due to floating point
    if fa_si + fa_colon > total_fa + 1e-9:
        ratio = total_fa / (fa_si + fa_colon + 1e-15)
        fa_si *= ratio
        fa_colon *= ratio

    notes_parts = [
        f"Simulated {drug_name} GI absorption window over {t_end_h:.1f}h.",
        f"SI transit: {si_transit_h:.1f}h, Colon transit: {colon_transit_h:.1f}h.",
        f"Peff (cm/s): {peff_cm_s:.2e}, ka_SI (/h): {ka_si:.4f}.",
        f"Total fa: {total_fa:.3f}.",
    ]
    if fed_state:
        notes_parts.append("Fed state: SI transit slowed 1.5x.")

    return AbsorptionWindowResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        position_cm=position_cm,
        fraction_absorbed=fraction_absorbed,
        absorption_rate_mg_h=absorption_rate_mg_h,
        c_portal_mg_L=c_portal_mg_L,
        total_fa=total_fa,
        fa_small_intestine=fa_si,
        fa_colon=fa_colon,
        absorption_window_h=si_transit_h,
        tmax_absorption_h=tmax_absorption_h,
        notes=" ".join(notes_parts),
    )
