"""Drug absorption window simulation — Phase 473.

Simulates time-dependent drug absorption as the drug transits through GI segments,
limited by dissolution rate and permeability. Uses a forward Euler ODE integration.

GI Segments
-----------
- Stomach:   0.5 h transit, pH 1.5
- Duodenum:  0.25 h transit, pH 6.0
- Jejunum:   1.5 h transit, pH 6.5  (main absorption site)
- Ileum:     2.0 h transit, pH 7.2
- Colon:     4.0 h transit, pH 7.4  (limited absorption)

References
----------
- Yu LX & Amidon GL, Pharm Res. 1999;16:1014-18
- Jamei M et al., Curr Drug Metab. 2009;10:26-40
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Segment definitions: (name, transit_h, pH, volume_mL, radius_cm)
# ---------------------------------------------------------------------------
_SEGMENTS = [
    ("stomach", 0.50, 1.5, 250.0, 3.5),
    ("duodenum", 0.25, 6.0, 50.0, 1.5),
    ("jejunum", 1.50, 6.5, 150.0, 1.5),
    ("ileum", 2.00, 7.2, 200.0, 1.5),
    ("colon", 4.00, 7.4, 300.0, 2.0),
]

# Fraction of dose absorbed from colon relative to SI (penalised)
_COLON_ABSORPTION_FACTOR = 0.15

# Base effective permeability (cm/s) for logP reference drug (logP=2)
_PEFF_REF_CM_S = 2.0e-4

# Dissolution rate constant base (h^-1)
_KDISS_BASE = 0.5


def _fraction_un_ionized(ph: float, pka: float, drug_type: str) -> float:
    """Fraction of drug in un-ionized (neutral) form via Henderson-Hasselbalch."""
    if pka is None:
        return 1.0
    if drug_type == "neutral":
        return 1.0
    if drug_type == "acid":
        denom = 1.0 + 10.0 ** (ph - pka)
        return 1.0 / denom
    # base
    denom = 1.0 + 10.0 ** (pka - ph)
    return 1.0 / denom


def _ph_adjusted_solubility(
    sol_ph7: float,
    ph: float,
    pka: float | None,
    drug_type: str,
) -> float:
    """Estimate solubility at given pH using Henderson-Hasselbalch.

    For acids: S(pH) = S(pH7) * (1 + 10^(pH-pKa)) / (1 + 10^(7-pKa))
    For bases: S(pH) = S(pH7) * (1 + 10^(pKa-pH)) / (1 + 10^(pKa-7))
    Neutral: S constant.
    """
    if pka is None or drug_type == "neutral":
        return sol_ph7
    if drug_type == "acid":
        num = 1.0 + 10.0 ** (ph - pka)
        den = 1.0 + 10.0 ** (7.0 - pka)
    else:  # base
        num = 1.0 + 10.0 ** (pka - ph)
        den = 1.0 + 10.0 ** (pka - 7.0)
    # Avoid division by zero for extreme pKa values
    if den < 1e-12:
        return sol_ph7
    return sol_ph7 * num / den


def _peff_from_logp(logP: float, mw: float) -> float:
    """Estimate effective intestinal permeability (cm/s) from logP and MW.

    Uses a simplified QSPR: Peff ~ Pref * 10^((logP-2)/3) / (MW/300)^0.5
    Clamped to [1e-7, 5e-3] cm/s.
    """
    log_factor = (logP - 2.0) / 3.0
    mw_factor = (mw / 300.0) ** 0.5
    peff = _PEFF_REF_CM_S * (10.0**log_factor) / mw_factor
    return max(1e-7, min(peff, 5e-3))


@dataclass(frozen=True)
class AbsorptionWindowResult:
    """Results from a drug absorption window simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float]
    fraction_absorbed: list[float]
    fa_total: float
    lag_time_h: float
    window_width_h: float
    limiting_factor: str  # 'dissolution', 'permeability', 'transit'
    notes: str


def estimate_lag_time(
    absorbed_fraction_profile: list[float],
    times_h: list[float],
    threshold: float = 0.05,
) -> float:
    """Estimate absorption lag time (hours).

    Returns the time at which cumulative absorbed fraction first reaches
    ``threshold`` fraction of the total absorbed. Default threshold is 5%.

    Parameters
    ----------
    absorbed_fraction_profile:
        Cumulative fraction absorbed at each time point.
    times_h:
        Corresponding simulation time points (h).
    threshold:
        Fraction of maximum absorbed fraction that defines "onset" (default 0.05).

    Returns
    -------
    float
        Lag time in hours. Returns 0.0 if threshold never reached.
    """
    if not absorbed_fraction_profile or not times_h:
        return 0.0
    fa_max = max(absorbed_fraction_profile) if absorbed_fraction_profile else 0.0
    if fa_max <= 0.0:
        return 0.0
    cutoff = threshold * fa_max
    for t, fa in zip(times_h, absorbed_fraction_profile, strict=False):
        if fa >= cutoff:
            return float(t)
    return float(times_h[-1])


def calculate_window_width(
    absorbed_fraction_profile: list[float],
    times_h: list[float],
    threshold: float = 0.9,
) -> float:
    """Calculate absorption window width (hours).

    Returns the time elapsed between first reaching 5% of max absorption
    and reaching ``threshold`` fraction of max absorption (default 90%).

    Parameters
    ----------
    absorbed_fraction_profile:
        Cumulative fraction absorbed at each time point.
    times_h:
        Corresponding simulation time points (h).
    threshold:
        Fraction of max absorbed fraction at which the window closes (default 0.9).

    Returns
    -------
    float
        Window width in hours. Returns 0.0 if threshold never reached.
    """
    if not absorbed_fraction_profile or not times_h:
        return 0.0
    fa_max = max(absorbed_fraction_profile) if absorbed_fraction_profile else 0.0
    if fa_max <= 0.0:
        return 0.0

    onset_cutoff = 0.05 * fa_max
    close_cutoff = threshold * fa_max

    t_onset = None
    t_close = None
    for t, fa in zip(times_h, absorbed_fraction_profile, strict=False):
        if t_onset is None and fa >= onset_cutoff:
            t_onset = t
        if t_onset is not None and fa >= close_cutoff:
            t_close = t
            break

    if t_onset is None or t_close is None:
        return 0.0
    return float(t_close - t_onset)


def simulate_absorption_window(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw: float,
    pka: float | None = None,
    drug_type: str = "acid",
    solubility_pH7_mg_mL: float = 1.0,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> AbsorptionWindowResult:
    """Simulate drug absorption as it transits through GI segments.

    For each segment the model tracks:
    - Undissolved drug mass (mg)
    - Dissolved drug mass (mg)
    - Cumulative absorbed mass (mg)

    ODE (forward Euler per segment with transit):
    - d(undissolved)/dt = -k_diss * undissolved - k_transit * undissolved
    - d(dissolved)/dt  = k_diss * undissolved - k_abs * dissolved
                         - k_transit * dissolved
    - d(absorbed)/dt   = k_abs * dissolved

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Administered dose (mg).
    logP:
        Lipophilicity (log octanol-water partition coefficient).
    mw:
        Molecular weight (g/mol).
    pka:
        Acid/base dissociation constant (None for neutral drugs).
    drug_type:
        'acid', 'base', or 'neutral'.
    solubility_pH7_mg_mL:
        Aqueous solubility at pH 7 (mg/mL). Must be > 0.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    AbsorptionWindowResult
    """
    # --- Input validation ---
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0.0:
        raise ValueError("dose_mg must be > 0")
    if mw <= 0.0:
        raise ValueError("mw must be > 0")
    if solubility_pH7_mg_mL <= 0.0:
        raise ValueError("solubility_pH7_mg_mL must be > 0")
    if t_end_h <= 0.0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0.0:
        raise ValueError("dt_h must be > 0")
    if drug_type not in ("acid", "base", "neutral"):
        raise ValueError("drug_type must be 'acid', 'base', or 'neutral'")

    # --- Derived permeability ---
    peff_cm_s = _peff_from_logp(logP, mw)
    peff_h = peff_cm_s * 3600.0  # cm/h

    # --- Build time grid ---
    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = [i * dt_h for i in range(n_steps)]

    # --- State variables across all segments ---
    # Each segment has: undissolved_mg, dissolved_mg
    n_seg = len(_SEGMENTS)
    undissolved = [0.0] * n_seg
    dissolved = [0.0] * n_seg

    # Dose starts entirely undissolved in stomach
    undissolved[0] = dose_mg

    # Cumulative absorbed mass (mg)
    absorbed_total = 0.0

    # Profile of cumulative absorbed fraction over time
    fa_profile: list[float] = [0.0] * n_steps

    # Dissolution rate limitedness tracker
    total_dissolution_flux = 0.0
    total_absorption_flux = 0.0
    total_transit_undissolved = 0.0  # undissolved drug lost to transit (dissolution-limited signal)

    for step in range(1, n_steps):
        new_undissolved = [0.0] * n_seg
        new_dissolved = [0.0] * n_seg
        d_absorbed = 0.0

        for si, (seg_name, transit_h, ph, volume_mL, radius_cm) in enumerate(_SEGMENTS):
            # Skip empty segments
            if undissolved[si] < 1e-15 and dissolved[si] < 1e-15:
                continue

            # Transit rate constant
            k_transit = 1.0 / transit_h  # h^-1

            # pH-adjusted solubility (mg/mL)
            sol_ph = _ph_adjusted_solubility(solubility_pH7_mg_mL, ph, pka, drug_type)
            # Maximum dissolved drug capacity in this segment (mg)
            max_dissolved_mg = sol_ph * volume_mL

            # Dissolution rate constant (h^-1) — first order approximation
            # Enhanced by solubility (higher sol => faster dissolution)
            k_diss = _KDISS_BASE * min(sol_ph / solubility_pH7_mg_mL, 10.0)

            # Permeability-based absorption rate (h^-1)
            # ka = 2 * Peff / radius (cylinder approximation)
            f_neut = _fraction_un_ionized(ph, pka, drug_type) if pka is not None else 1.0
            ka = 2.0 * peff_h * f_neut / radius_cm

            # Colon: strongly reduced absorption
            if seg_name == "colon":
                ka *= _COLON_ABSORPTION_FACTOR

            # Dissolution limited by capacity
            capacity_fraction = max(
                0.0,
                1.0 - dissolved[si] / max_dissolved_mg if max_dissolved_mg > 0 else 0.0,
            )
            eff_k_diss = k_diss * capacity_fraction

            # Fluxes (mg/h)
            flux_diss = eff_k_diss * undissolved[si]
            flux_abs = ka * dissolved[si]
            flux_transit_u = k_transit * undissolved[si]
            flux_transit_d = k_transit * dissolved[si]

            # Accumulate limitedness diagnostics
            total_dissolution_flux += flux_diss * dt_h
            total_absorption_flux += flux_abs * dt_h
            total_transit_undissolved += flux_transit_u * dt_h

            # Forward Euler update
            du = -(flux_diss + flux_transit_u) * dt_h
            dd = (flux_diss - flux_abs - flux_transit_d) * dt_h
            new_undissolved[si] += max(undissolved[si] + du, 0.0)
            new_dissolved[si] += max(dissolved[si] + dd, 0.0)
            d_absorbed += flux_abs * dt_h

            # Transit to next segment (if not last)
            if si + 1 < n_seg:
                new_undissolved[si + 1] += flux_transit_u * dt_h
                new_dissolved[si + 1] += flux_transit_d * dt_h

        absorbed_total += d_absorbed
        undissolved = new_undissolved
        dissolved = new_dissolved
        fa_profile[step] = min(absorbed_total / dose_mg, 1.0)

    fa_total = fa_profile[-1]

    # --- Lag time and window width ---
    lag_time_h = estimate_lag_time(fa_profile, times)
    window_width_h = calculate_window_width(fa_profile, times)

    # --- Identify limiting factor ---
    # Dissolution-limited: significant undissolved drug transits out
    # (meaning dissolution couldn't keep up → undissolved exited GI)
    # Permeability-limited: dissolution faster than absorption
    # Transit-limited: even with good dissolution and permeability, drug runs out of time
    eps = 1e-12
    total_drug_processed = total_dissolution_flux + total_transit_undissolved + eps
    f_undissolved_transit = total_transit_undissolved / total_drug_processed

    if f_undissolved_transit > 0.25:
        # More than 25% of drug processed exited as undissolved → dissolution-limited
        limiting_factor = "dissolution"
    elif total_dissolution_flux > total_absorption_flux * 2.0 + eps:
        # Dissolution much faster than absorption → permeability-limited
        limiting_factor = "permeability"
    else:
        limiting_factor = "transit"

    # --- Compose notes ---
    dose_number = dose_mg / (solubility_pH7_mg_mL * 250.0)  # reference 250 mL
    notes_parts = [
        f"logP={logP:.1f}, MW={mw:.0f} g/mol, Peff={peff_cm_s:.2e} cm/s.",
        f"Dose number (250 mL)={dose_number:.2f}.",
    ]
    if pka is not None:
        notes_parts.append(f"pKa={pka}, drug_type={drug_type}.")
    if fa_total < 0.3:
        notes_parts.append("Low bioavailability predicted.")
    elif fa_total > 0.8:
        notes_parts.append("High bioavailability predicted.")
    notes = " ".join(notes_parts)

    return AbsorptionWindowResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        fraction_absorbed=fa_profile,
        fa_total=fa_total,
        lag_time_h=lag_time_h,
        window_width_h=window_width_h,
        limiting_factor=limiting_factor,
        notes=notes,
    )
