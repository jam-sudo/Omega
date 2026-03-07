"""Biliary clearance model for hepatic drug elimination via bile secretion.

Models the contribution of biliary clearance (CLbile) to total hepatic clearance,
including enterohepatic recirculation (ERC) effects on plasma concentration-time profiles.

Key equations:
- Well-stirred hepatic ER: ER = fu*(CLmet + CLbile) / (Q + fu*(CLmet + CLbile))
- Biliary fraction: f_bile = CLbile / (CLmet + CLbile)
- ERC produces secondary plasma peaks from bile reabsorption
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BiliaryERResult:
    """Result from biliary extraction ratio calculation.

    Attributes:
        q_liver: Hepatic blood flow (L/h).
        cl_uptake: Hepatic uptake clearance (L/h).
        cl_bile: Biliary clearance (L/h).
        fu_plasma: Unbound fraction in plasma.
        er_hepatic: Hepatic extraction ratio (0-1).
        f_bile: Fraction of hepatic CL via bile secretion.
        f_metabolized: Fraction of hepatic CL via metabolism.
        cl_hepatic_total: Total hepatic clearance (L/h).
    """

    q_liver: float
    cl_uptake: float
    cl_bile: float
    fu_plasma: float
    er_hepatic: float
    f_bile: float
    f_metabolized: float
    cl_hepatic_total: float


@dataclass(frozen=True)
class BiliaryPKResult:
    """Result from biliary PK simulation.

    Attributes:
        drug_name: Drug identifier.
        dose_mg: Administered dose (mg).
        cl_bile_fraction: Fraction of hepatic CL via bile.
        erc_fraction: Fraction of biliary drug undergoing ERC.
        times_h: Time points (h).
        c_plasma_mg_L: Plasma concentration at each time point (mg/L).
        cmax: Maximum plasma concentration (mg/L).
        tmax_h: Time of Cmax (h).
        auc: Area under concentration-time curve (mg*h/L), trapezoidal.
        secondary_peak_present: True if ERC secondary peak detected.
        t_secondary_peak_h: Time of secondary peak (h), or None if absent.
        notes: Descriptive notes about the simulation.
    """

    drug_name: str
    dose_mg: float
    cl_bile_fraction: float
    erc_fraction: float
    times_h: tuple[float, ...]
    c_plasma_mg_L: tuple[float, ...]
    cmax: float
    tmax_h: float
    auc: float
    secondary_peak_present: bool
    t_secondary_peak_h: float | None
    notes: str


def biliary_extraction_ratio(
    q_liver_L_per_h: float,
    cl_uptake_L_per_h: float,
    cl_bile_L_per_h: float,
    fu_plasma: float,
    cl_int_met_L_per_h: float = 0.0,
) -> BiliaryERResult:
    """Compute hepatic extraction ratio with biliary clearance contribution.

    Uses the well-stirred liver model:
        ER = fu * (CLint_met + CLbile) / (Q + fu * (CLint_met + CLbile))

    When cl_int_met_L_per_h is not supplied, it is inferred from cl_uptake such
    that cl_uptake acts as the effective intrinsic clearance (a common simplification
    when the rate-limiting step is uptake).

    Args:
        q_liver_L_per_h: Hepatic blood flow (L/h). Must be > 0.
        cl_uptake_L_per_h: Hepatic uptake CL (L/h). Must be >= 0.
        cl_bile_L_per_h: Biliary secretion CL (L/h). Must be >= 0.
        fu_plasma: Unbound plasma fraction (0 < fu <= 1).
        cl_int_met_L_per_h: Intrinsic metabolic CL (L/h). Defaults to 0; when
            0, cl_uptake is used as the total intrinsic CL.

    Returns:
        BiliaryERResult with computed hepatic extraction metrics.

    Raises:
        ValueError: If any parameter fails validation.
    """
    if q_liver_L_per_h <= 0:
        raise ValueError(f"q_liver_L_per_h must be > 0, got {q_liver_L_per_h}")
    if cl_uptake_L_per_h < 0:
        raise ValueError(f"cl_uptake_L_per_h must be >= 0, got {cl_uptake_L_per_h}")
    if cl_bile_L_per_h < 0:
        raise ValueError(f"cl_bile_L_per_h must be >= 0, got {cl_bile_L_per_h}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if cl_int_met_L_per_h < 0:
        raise ValueError(f"cl_int_met_L_per_h must be >= 0, got {cl_int_met_L_per_h}")

    # When metabolic intrinsic CL not supplied, use uptake CL as proxy
    cl_met = cl_int_met_L_per_h if cl_int_met_L_per_h > 0 else cl_uptake_L_per_h
    cl_int_total = cl_met + cl_bile_L_per_h

    # Well-stirred model ER
    numerator = fu_plasma * cl_int_total
    er_hepatic = numerator / (q_liver_L_per_h + numerator)
    er_hepatic = min(max(er_hepatic, 0.0), 1.0)

    cl_hepatic_total = q_liver_L_per_h * er_hepatic

    # Fraction routed via each pathway
    if cl_int_total > 0:
        f_bile = cl_bile_L_per_h / cl_int_total
        f_metabolized = cl_met / cl_int_total
    else:
        f_bile = 0.0
        f_metabolized = 1.0

    return BiliaryERResult(
        q_liver=q_liver_L_per_h,
        cl_uptake=cl_uptake_L_per_h,
        cl_bile=cl_bile_L_per_h,
        fu_plasma=fu_plasma,
        er_hepatic=er_hepatic,
        f_bile=f_bile,
        f_metabolized=f_metabolized,
        cl_hepatic_total=cl_hepatic_total,
    )


def _simulate_one_compartment(
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    ka_per_h: float,
    route: str,
    t_points: list[float],
) -> list[float]:
    """Simulate 1-compartment PK and return concentrations at t_points.

    Args:
        dose_mg: Dose in mg.
        vd_L: Volume of distribution (L).
        cl_L_per_h: Clearance (L/h).
        ka_per_h: Absorption rate constant (1/h); unused for iv.
        route: "iv" or "oral".
        t_points: Time points in hours.

    Returns:
        Plasma concentrations (mg/L) at each time point.
    """
    ke = cl_L_per_h / vd_L  # elimination rate (1/h)

    concs = []
    if route == "iv":
        c0 = dose_mg / vd_L
        for t in t_points:
            concs.append(c0 * math.exp(-ke * t))
    else:
        # Oral 1-compartment: C(t) = F*D*ka/(Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
        # Here dose_mg already accounts for bioavailability (caller should scale)
        if abs(ka_per_h - ke) < 1e-9:
            # Avoid division by zero — use slightly perturbed ke
            ke = ke * 1.0001
        factor = dose_mg * ka_per_h / (vd_L * (ka_per_h - ke))
        for t in t_points:
            c = factor * (math.exp(-ke * t) - math.exp(-ka_per_h * t))
            concs.append(max(c, 0.0))

    return concs


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC by linear trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def _detect_secondary_peak(
    times: list[float], concs: list[float], tmax_h: float
) -> tuple[bool, float | None]:
    """Detect secondary peak after initial tmax.

    A secondary peak exists if there is a local maximum after the primary Cmax.

    Args:
        times: Time points.
        concs: Corresponding concentrations.
        tmax_h: Time of primary peak.

    Returns:
        (secondary_peak_present, t_secondary_peak_h)
    """
    # Find index of primary peak
    primary_idx = 0
    primary_c = concs[0]
    for i, c in enumerate(concs):
        if c > primary_c:
            primary_c = c
            primary_idx = i

    # Search for local max after primary peak
    # Require: a point higher than its two neighbors, and higher than post-primary minimum
    if primary_idx + 2 >= len(concs):
        return False, None

    # Find the valley after the primary peak
    _valley_c = min(concs[primary_idx:])
    valley_threshold = primary_c * 0.05  # secondary peak must be > 5% of Cmax

    for i in range(primary_idx + 1, len(concs) - 1):
        left = concs[i - 1]
        mid = concs[i]
        right = concs[i + 1]
        if mid > left and mid > right and mid > valley_threshold:
            # Confirm it's truly a secondary peak (not noise at primary rising edge)
            if times[i] > tmax_h + 0.5:
                return True, times[i]

    return False, None


def simulate_biliary_pk(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    cl_bile_fraction: float = 0.3,
    vd_L: float = 50.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    erc_fraction: float = 0.5,
    k_reabs_per_h: float = 0.1,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> BiliaryPKResult:
    """Simulate plasma PK with biliary clearance and enterohepatic recirculation.

    Biliary clearance removes drug from plasma into bile. A fraction (erc_fraction)
    of bile-excreted drug is reabsorbed from the gut, producing secondary plasma peaks.

    Model:
        dC/dt = -ke_total*C + absorption_rate + bile_reabsorption_rate
        CLbile = cl_bile_fraction * cl_hepatic
        CLmet = (1 - cl_bile_fraction) * cl_hepatic
        ke_total = cl_hepatic / Vd

    Bile pool: amount excreted accumulates, then is released at time ~2h post-dose
    (gallbladder contraction proxy) for ERC.

    Args:
        drug_name: Drug identifier.
        dose_mg: Administered dose (mg). Must be > 0.
        cl_hepatic_L_per_h: Total hepatic clearance (L/h). Must be > 0.
        cl_bile_fraction: Fraction of hepatic CL via biliary route [0, 1].
        vd_L: Volume of distribution (L). Must be > 0.
        route: Dosing route: "iv" or "oral".
        ka_per_h: First-order absorption rate (1/h); oral only.
        erc_fraction: Fraction of bile-excreted drug reabsorbed (ERC) [0, 1].
        k_reabs_per_h: First-order rate of bile reabsorption (1/h).
        t_end_h: Simulation end time (h).
        dt_h: ODE time step (h).

    Returns:
        BiliaryPKResult with concentration-time profile and metrics.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError(f"cl_hepatic_L_per_h must be > 0, got {cl_hepatic_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0.0 <= cl_bile_fraction <= 1.0):
        raise ValueError(f"cl_bile_fraction must be in [0, 1], got {cl_bile_fraction}")
    if not (0.0 <= erc_fraction <= 1.0):
        raise ValueError(f"erc_fraction must be in [0, 1], got {erc_fraction}")
    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got {route}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    if k_reabs_per_h < 0:
        raise ValueError(f"k_reabs_per_h must be >= 0, got {k_reabs_per_h}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    ke_total = cl_hepatic_L_per_h / vd_L  # overall elimination rate (1/h)
    cl_bile = cl_bile_fraction * cl_hepatic_L_per_h
    ke_bile = cl_bile / vd_L  # rate of removal into bile from central compartment

    # Time grid
    n_steps = int(round(t_end_h / dt_h)) + 1
    times = [i * dt_h for i in range(n_steps)]

    # State variables (forward Euler ODE)
    # C: plasma concentration (mg/L)
    # A_gut: drug available for oral absorption (mg)
    # A_bile: drug accumulated in bile pool (mg)
    # A_reabs_gut: drug in gut available for ERC reabsorption (mg)

    if route == "iv":
        C = dose_mg / vd_L
        A_gut = 0.0
    else:
        C = 0.0
        A_gut = dose_mg  # entire dose in gut at t=0

    A_bile = 0.0
    A_reabs_gut = 0.0

    # Gallbladder contraction delay: bile released into gut ~2h after meal/dose
    bile_release_start_h = 2.0
    bile_release_duration_h = 1.0  # spread release over 1 h
    bile_release_rate = 0.0  # computed dynamically

    concs = [C]
    bile_pool_released = False

    for step in range(1, n_steps):
        t = times[step - 1]

        # Oral absorption
        oral_abs_rate = ka_per_h * A_gut if route == "oral" else 0.0

        # Bile reabsorption (ERC) into plasma
        reabs_rate = k_reabs_per_h * A_reabs_gut

        # Net rate for plasma
        dC = -ke_total * C + (oral_abs_rate + reabs_rate) / vd_L
        dA_gut = -oral_abs_rate if route == "oral" else 0.0

        # Bile secretion rate (amount per hour)
        bile_secretion_rate = ke_bile * C * vd_L  # mg/h

        dA_bile = bile_secretion_rate

        # Gallbladder contraction: release bile pool at ~2h
        if (
            not bile_pool_released
            and t >= bile_release_start_h
            and t < bile_release_start_h + bile_release_duration_h
        ):
            # Release accumulated bile gradually
            bile_release_rate = A_bile / bile_release_duration_h * dt_h / dt_h
            released_now = bile_release_rate * dt_h
            if released_now > A_bile:
                released_now = A_bile
            erc_amount = released_now * erc_fraction
            dA_bile -= bile_release_rate
        else:
            erc_amount = 0.0
            bile_release_rate = 0.0
            if t >= bile_release_start_h + bile_release_duration_h:
                bile_pool_released = True

        dA_reabs_gut = erc_amount / dt_h - k_reabs_per_h * A_reabs_gut

        # Euler update
        C = max(C + dC * dt_h, 0.0)
        A_gut = max(A_gut + dA_gut * dt_h, 0.0)
        A_bile = max(A_bile + dA_bile * dt_h, 0.0)
        A_reabs_gut = max(A_reabs_gut + dA_reabs_gut * dt_h, 0.0)

        concs.append(C)

    # Compute PK metrics
    cmax = max(concs)
    tmax_h = times[concs.index(cmax)]
    auc = _trapezoidal_auc(times, concs)
    secondary_peak_present, t_secondary_peak_h = _detect_secondary_peak(times, concs, tmax_h)

    notes_parts = [
        f"CLbile={cl_bile:.2f} L/h ({cl_bile_fraction * 100:.0f}% of CL_hepatic)",
        f"ERC fraction={erc_fraction:.2f}",
    ]
    if secondary_peak_present:
        notes_parts.append(f"Secondary peak at t={t_secondary_peak_h:.1f}h from ERC")
    else:
        notes_parts.append("No secondary peak detected")

    return BiliaryPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_bile_fraction=cl_bile_fraction,
        erc_fraction=erc_fraction,
        times_h=tuple(times),
        c_plasma_mg_L=tuple(concs),
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        secondary_peak_present=secondary_peak_present,
        t_secondary_peak_h=t_secondary_peak_h,
        notes="; ".join(notes_parts),
    )


def fe_vs_fm_sensitivity(
    drug_name: str,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    bile_fractions: list[float],
    dose_mg: float = 100.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    erc_fraction: float = 0.5,
    k_reabs_per_h: float = 0.1,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list[BiliaryPKResult]:
    """Sweep bile_fraction values to assess biliary CL impact on PK.

    Runs simulate_biliary_pk for each bile_fraction value in bile_fractions and
    returns results sorted by bile_fraction ascending.

    Args:
        drug_name: Drug identifier.
        cl_hepatic_L_per_h: Total hepatic clearance (L/h). Must be > 0.
        vd_L: Volume of distribution (L). Must be > 0.
        bile_fractions: List of bile CL fractions to evaluate. Each must be in [0, 1].
        dose_mg: Dose (mg). Must be > 0.
        route: Dosing route ("iv" or "oral").
        ka_per_h: Oral absorption rate (1/h).
        erc_fraction: ERC fraction for all runs [0, 1].
        k_reabs_per_h: Bile reabsorption rate (1/h).
        t_end_h: Simulation duration (h).
        dt_h: ODE time step (h).

    Returns:
        List of BiliaryPKResult sorted by cl_bile_fraction ascending.

    Raises:
        ValueError: If any bile_fraction is outside [0, 1] or other params invalid.
    """
    if not bile_fractions:
        raise ValueError("bile_fractions must not be empty")

    results = []
    for bf in sorted(bile_fractions):
        result = simulate_biliary_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_hepatic_L_per_h=cl_hepatic_L_per_h,
            cl_bile_fraction=bf,
            vd_L=vd_L,
            route=route,
            ka_per_h=ka_per_h,
            erc_fraction=erc_fraction,
            k_reabs_per_h=k_reabs_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    return results
