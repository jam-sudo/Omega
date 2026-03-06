"""Gastrointestinal transit and regional absorption model.

Models drug transit through 7 GI segments with region-specific
permeability, pH-dependent ionization, and solubility-limited absorption.

Segments
--------
1. Stomach       (V=0.25 L, pH=1.5)
2. Duodenum      (V=0.10 L, pH=6.0)
3. Jejunum 1     (V=0.15 L, pH=6.5)
4. Jejunum 2     (V=0.15 L, pH=6.5)
5. Ileum 1       (V=0.30 L, pH=7.5)
6. Ileum 2       (V=0.30 L, pH=7.5)
7. Colon         (V=0.50 L, pH=7.0)

References
----------
- Yu LX & Amidon GL, Pharm Res. 1999;16:1014–18
- Jamei M et al., Curr Drug Metab. 2009;10:26–40
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

# Segment definitions: (name, volume_L, fasted_transit_h, pH, peff_base, radius_cm)
_SEGMENTS = [
    ("stomach", 0.25, 0.25, 1.5, -6.0, 2.0),
    ("duodenum", 0.10, 0.25, 6.0, -3.5, 1.5),
    ("jejunum_1", 0.15, 1.0, 6.5, -3.2, 1.5),
    ("jejunum_2", 0.15, 1.0, 6.5, -3.2, 1.5),
    ("ileum_1", 0.30, 2.0, 7.5, -3.5, 1.5),
    ("ileum_2", 0.30, 2.0, 7.5, -3.5, 1.5),
    ("colon", 0.50, 20.0, 7.0, -4.0, 2.5),
]


def _classify_drug(logP: float, pka: float) -> str:
    """Classify drug as 'acid', 'base', or 'neutral'."""
    if pka < 5.0:
        return "acid"
    if logP > 2.0 and pka > 7.0:
        return "base"
    return "neutral"


def _fraction_neutral(ph: float, pka: float, drug_type: str) -> float:
    """Fraction of drug in neutral (absorbable) form via Henderson-Hasselbalch."""
    if drug_type == "neutral":
        return 1.0
    if drug_type == "acid":
        return 1.0 / (1.0 + 10.0 ** (ph - pka))
    # base
    return 1.0 / (1.0 + 10.0 ** (pka - ph))


@dataclass(frozen=True)
class GITransitResult:
    """Results from GI transit and absorption simulation.

    Attributes
    ----------
    drug_name : Drug identifier.
    dose_mg : Administered dose (mg).
    food_state : 'fasted' or 'fed'.
    logP : Lipophilicity.
    times_h : Simulation time points (h).
    cp_mg_L : Systemic plasma concentration over time (mg/L).
    fa_cumulative : Cumulative fraction absorbed over time.
    segment_absorbed_pct : Percent of dose absorbed from each segment.
    cmax_mg_L : Peak plasma concentration (mg/L).
    tmax_h : Time of Cmax (h).
    auc_0_t : AUC 0-t (mg*h/L).
    fa_total : Total fraction absorbed.
    f_abs_small_intestine : Fraction absorbed from small intestine segments.
    rate_limiting : Dominant absorption limitation.
    notes : Summary text.
    """

    drug_name: str
    dose_mg: float
    food_state: str
    logP: float
    times_h: list[float]
    cp_mg_L: list[float]
    fa_cumulative: list[float]
    segment_absorbed_pct: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_0_t: float
    fa_total: float
    f_abs_small_intestine: float
    rate_limiting: str
    notes: str


def simulate_gi_transit(
    drug_name: str,
    dose_mg: float,
    logP: float,
    pka: float,
    solubility_mg_mL: float = 1.0,
    mw: float = 300.0,
    food_state: str = "fasted",
    cl_L_per_h: float = 1.0,
    vd_L: float = 30.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> GITransitResult:
    """Simulate GI transit-limited absorption with regional permeability.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Oral dose (mg).
    logP : Partition coefficient (lipophilicity).
    pka : Acid dissociation constant.
    solubility_mg_mL : Aqueous solubility (mg/mL).
    mw : Molecular weight (g/mol).
    food_state : 'fasted' or 'fed'.
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    GITransitResult

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")

    drug_type = _classify_drug(logP, pka)
    n_seg = len(_SEGMENTS)

    # Compute effective permeability for each segment (cm/s)
    peff_cm_s = []
    for _name, _vol, _tt, ph, base, _radius in _SEGMENTS:
        log_peff = 0.4 * logP - 0.008 * mw + base
        peff = 10.0**log_peff
        fn = _fraction_neutral(ph, pka, drug_type)
        peff_cm_s.append(peff * fn)

    # Absorption rate constants: ka = Peff * 2 / radius * 3600 (cm/s → h⁻¹)
    ka_per_h = []
    for s in range(n_seg):
        radius = _SEGMENTS[s][5]
        ka_per_h.append(peff_cm_s[s] * 2.0 / radius * 3600.0)

    # Transit rate constants (fed: stomach 4x slower)
    ktr = []
    for s in range(n_seg):
        tt = _SEGMENTS[s][2]
        if s == 0 and food_state.lower() == "fed":
            tt *= 4.0
        ktr.append(1.0 / tt)

    # Solubility-limited absorption: cap dissolved drug per segment
    sol_cap_mg = [solubility_mg_mL * _SEGMENTS[s][1] * 1000.0 for s in range(n_seg)]

    ke = cl_L_per_h / vd_L
    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    # State: mass in each segment (mg), plasma conc (mg/L)
    a_seg = np.zeros((n_seg, n_steps + 1))
    cp = np.zeros(n_steps + 1)
    cum_abs = np.zeros(n_steps + 1)  # cumulative absorbed mass (mg)
    seg_absorbed = np.zeros(n_seg)  # total absorbed from each segment (mg)

    # Dose enters stomach at t=0
    a_seg[0, 0] = dose_mg

    for i in range(n_steps):
        total_abs_rate = 0.0
        for s in range(n_seg):
            mass = a_seg[s, i]
            # Dissolved fraction limited by solubility
            dissolved = min(mass, sol_cap_mg[s])

            # Absorption (only dissolved drug)
            abs_rate = ka_per_h[s] * dissolved
            # Transit out
            transit_out = ktr[s] * mass

            # Transit in from previous segment
            transit_in = ktr[s - 1] * a_seg[s - 1, i] if s > 0 else 0.0

            da = (transit_in - transit_out - abs_rate) * dt_h
            a_seg[s, i + 1] = max(0.0, mass + da)

            seg_absorbed[s] += abs_rate * dt_h
            total_abs_rate += abs_rate

        dcp = (total_abs_rate / vd_L - ke * cp[i]) * dt_h
        cp[i + 1] = max(0.0, cp[i] + dcp)
        cum_abs[i + 1] = cum_abs[i] + total_abs_rate * dt_h

    # Fraction absorbed
    fa_cum = cum_abs / dose_mg
    fa_cum = np.clip(fa_cum, 0.0, 1.0)
    fa_total = float(fa_cum[-1])

    # Segment absorbed percentages
    seg_abs_pct = (seg_absorbed / dose_mg * 100.0).tolist()

    # SI = segments 1-5 (duodenum, jejunum1, jejunum2, ileum1, ileum2)
    si_absorbed = sum(seg_absorbed[1:6])
    f_abs_si = si_absorbed / dose_mg

    # PK metrics
    cmax = float(np.max(cp))
    tmax = float(t[int(np.argmax(cp))])
    auc = float(np_trapz(cp, t))

    # Rate-limiting determination
    max_peff = max(peff_cm_s)
    if solubility_mg_mL < 0.1:
        rate_lim = "solubility"
    elif max_peff < 1e-5:
        rate_lim = "permeability"
    else:
        rate_lim = "transit"

    notes = (
        f"GI transit: {drug_name}, {dose_mg}mg {food_state}. "
        f"Fa={fa_total:.1%}, Cmax={cmax:.4f} mg/L at {tmax:.2f}h. "
        f"Rate-limiting: {rate_lim}."
    )

    return GITransitResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        food_state=food_state,
        logP=logP,
        times_h=t.tolist(),
        cp_mg_L=cp.tolist(),
        fa_cumulative=fa_cum.tolist(),
        segment_absorbed_pct=seg_abs_pct,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_0_t=auc,
        fa_total=fa_total,
        f_abs_small_intestine=float(np.clip(f_abs_si, 0.0, 1.0)),
        rate_limiting=rate_lim,
        notes=notes,
    )


def compare_food_states(
    drug_name: str,
    dose_mg: float,
    logP: float,
    pka: float,
    **kwargs,
) -> tuple[GITransitResult, GITransitResult]:
    """Compare fasted vs fed GI absorption.

    Returns
    -------
    Tuple of (fasted_result, fed_result).
    """
    fasted = simulate_gi_transit(drug_name, dose_mg, logP, pka, food_state="fasted", **kwargs)
    fed = simulate_gi_transit(drug_name, dose_mg, logP, pka, food_state="fed", **kwargs)
    return fasted, fed


__all__ = [
    "GITransitResult",
    "simulate_gi_transit",
    "compare_food_states",
]
