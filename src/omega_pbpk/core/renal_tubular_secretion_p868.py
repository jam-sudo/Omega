"""Phase 868 — Renal Tubular Secretion Model.

Mechanistic model of renal drug handling:
glomerular filtration + active tubular secretion + passive reabsorption.
"""

from dataclasses import dataclass

# Default tubular blood flow
_Q_TUBULAR_DEFAULT_L_PER_H = 30.0
_DEFAULT_GFR_LEVELS = [7.5, 5.0, 2.5, 1.0]


def _sigmoid_reabsorption(logp: float) -> float:
    """Fraction reabsorbed based on logP via sigmoid, clamped to [0, 0.95]."""
    if logp <= 0.0:
        return 0.0
    fr = logp / (logp + 5.0)
    if fr > 0.95:
        fr = 0.95
    return fr


@dataclass(frozen=True)
class RenalTubularSecretionResult:
    """Result of renal tubular secretion simulation."""

    drug_name: str
    dose_mg: float
    gfr_L_per_h: float
    fup: float
    logp: float
    cl_filtration_L_per_h: float
    cl_secretion_L_per_h: float
    reabsorption_fraction: float
    cl_renal_L_per_h: float
    cl_other_L_per_h: float
    fe_urine: float
    t_half_renal_h: float
    secretion_dominant: bool
    cumulative_urine_24h_mg: float
    transporter_saturation_fraction: float
    notes: str


def simulate_renal_tubular_secretion(
    drug_name: str,
    dose_mg: float,
    fup: float = 0.1,
    logp: float = 1.0,
    gfr_L_per_h: float = 7.5,
    vmax_s_mg_per_h: float = 5.0,
    km_s_mg_per_L: float = 0.5,
    q_tubular_L_per_h: float = 30.0,
    vd_L: float = 20.0,
    cl_other_L_per_h: float = 2.0,
    t_end_h: float = 24.0,
) -> RenalTubularSecretionResult:
    """Simulate renal drug handling via filtration, secretion, and reabsorption.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in mg (must be > 0).
    fup : float
        Fraction unbound in plasma (must be in (0, 1]).
    logp : float
        Octanol-water partition coefficient.
    gfr_L_per_h : float
        Glomerular filtration rate (L/h); default 7.5 (125 mL/min).
    vmax_s_mg_per_h : float
        Maximal tubular secretion rate (mg/h).
    km_s_mg_per_L : float
        Michaelis constant for secretion (mg/L).
    q_tubular_L_per_h : float
        Tubular blood flow for secretion capacity (L/h).
    vd_L : float
        Volume of distribution (L; must be > 0).
    cl_other_L_per_h : float
        Non-renal clearance (L/h).
    t_end_h : float
        Simulation period for cumulative urine (h).

    Returns
    -------
    RenalTubularSecretionResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0.0 < fup <= 1.0):
        raise ValueError("fup must be in (0, 1]")
    if gfr_L_per_h < 0:
        raise ValueError("gfr_L_per_h must be >= 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    # Initial plasma concentration (mg/L)
    c0 = dose_mg / vd_L
    c0_free = c0 * fup

    # 1. Filtration clearance
    cl_filtration = gfr_L_per_h * fup

    # 2. Active secretion: Michaelis-Menten intrinsic secretion at initial conc
    if km_s_mg_per_L > 0:
        cl_s_intrinsic = (vmax_s_mg_per_h * fup) / (km_s_mg_per_L + c0_free)
    else:
        cl_s_intrinsic = 0.0

    # Scale to in vivo via tubular capacity
    if q_tubular_L_per_h > 0 and cl_s_intrinsic > 0:
        cl_secretion = cl_s_intrinsic / (1.0 + cl_s_intrinsic / q_tubular_L_per_h)
    else:
        cl_secretion = cl_s_intrinsic

    # 3. Reabsorption fraction based on logP
    fr = _sigmoid_reabsorption(logp)

    # 4. Net renal clearance
    cl_renal = (cl_filtration + cl_secretion) * (1.0 - fr)

    # fe_urine
    cl_total = cl_renal + cl_other_L_per_h
    fe_urine = cl_renal / cl_total if cl_total > 0.0 else 0.0

    # t_half renal
    if cl_renal > 0.0:
        t_half_renal = 0.693 * vd_L / cl_renal
    else:
        t_half_renal = float("inf")

    # Secretion dominant?
    secretion_dominant = cl_secretion > cl_filtration

    # Cumulative urine at t_end_h
    # 1-cpt IV: C(t) = (dose/Vd)*exp(-ke*t)
    # ke = cl_total / vd_L
    ke = cl_total / vd_L
    # integral 0..t_end: CLrenal * (dose/Vd) * (1/ke) * (1 - exp(-ke*t_end))
    if ke > 0.0:
        cumulative_urine_24h = (cl_renal * dose_mg / cl_total) * (1.0 - _safe_exp(-ke * t_end_h))
    else:
        cumulative_urine_24h = 0.0

    # Transporter saturation fraction
    if vmax_s_mg_per_h > 0 and km_s_mg_per_L > 0:
        _transporter_sat = (vmax_s_mg_per_h * fup) / (km_s_mg_per_L + c0_free)
        # Normalise to [0,1] — at saturation CL_s_int would be Vmax/Km
        # fraction = actual/(max possible) = CL_s_int / (Vmax/Km)
        # But spec says: Vmax_s*fup / (Km_s + C0_free)
        # so treat as dimensionless ratio directly
        transporter_sat_fraction = (vmax_s_mg_per_h * fup) / (km_s_mg_per_L + c0_free)
        # The spec computes this as is (not normalised to 0-1 explicitly)
    else:
        transporter_sat_fraction = 0.0

    notes = (
        f"CLf={cl_filtration:.3f} L/h; "
        f"CLs={cl_secretion:.3f} L/h; "
        f"fr={fr:.3f}; "
        f"CLrenal={cl_renal:.3f} L/h; "
        f"fe_urine={fe_urine:.3f}"
    )

    return RenalTubularSecretionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        gfr_L_per_h=gfr_L_per_h,
        fup=fup,
        logp=logp,
        cl_filtration_L_per_h=cl_filtration,
        cl_secretion_L_per_h=cl_secretion,
        reabsorption_fraction=fr,
        cl_renal_L_per_h=cl_renal,
        cl_other_L_per_h=cl_other_L_per_h,
        fe_urine=fe_urine,
        t_half_renal_h=t_half_renal,
        secretion_dominant=secretion_dominant,
        cumulative_urine_24h_mg=cumulative_urine_24h,
        transporter_saturation_fraction=transporter_sat_fraction,
        notes=notes,
    )


def _safe_exp(x: float) -> float:
    """Safe exponential clamped to avoid overflow/underflow."""
    if x < -700.0:
        return 0.0
    if x > 700.0:
        return float("inf")
    import math

    return math.exp(x)


def screen_renal_impairment(
    drug_name: str,
    dose_mg: float,
    fup: float = 0.1,
    logp: float = 1.0,
    gfr_levels_L_per_h: list[float] | None = None,
) -> list[RenalTubularSecretionResult]:
    """Screen drug renal handling across GFR impairment levels.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in mg.
    fup : float
        Fraction unbound in plasma.
    logp : float
        Octanol-water partition coefficient.
    gfr_levels_L_per_h : list[float], optional
        GFR levels to screen. Default: [7.5, 5.0, 2.5, 1.0].

    Returns
    -------
    list[RenalTubularSecretionResult]
        Sorted by gfr_L_per_h descending.
    """
    if gfr_levels_L_per_h is None:
        gfr_levels_L_per_h = list(_DEFAULT_GFR_LEVELS)

    results = []
    for gfr in gfr_levels_L_per_h:
        r = simulate_renal_tubular_secretion(
            drug_name=drug_name,
            dose_mg=dose_mg,
            fup=fup,
            logp=logp,
            gfr_L_per_h=gfr,
        )
        results.append(r)

    results.sort(key=lambda r: r.gfr_L_per_h, reverse=True)
    return results
