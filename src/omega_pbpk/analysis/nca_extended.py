"""Extended non-compartmental analysis (NCA) with additional PK metrics (Phase 410).

Provides standard NCA metrics plus AUMC, MRT, Vss, Vz, CL, MAT, swing, and
peak-trough ratio. Designed for use with IV and oral PK profiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ExtendedNCAResult",
    "extended_nca",
    "compare_nca_profiles",
    "NCAResult",
    "compute_nca",
    "compute_partial_auc",
    "estimate_lambda_z",
]

_VALID_ROUTES = {"iv", "oral"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExtendedNCAResult:
    """Result from extended NCA analysis."""

    drug_name: str
    route: str
    dose_mg: float
    cmax: float
    tmax_h: float
    auc_last: float
    auc_inf: float
    lambda_z: float
    t_half_h: float
    aumc_inf: float
    mrt_h: float
    vss_L: float
    vz_L: float
    cl_L_per_h: float
    mat_h: float | None  # None if IV
    r_squared_terminal: float
    pct_extrapolated: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_nca_inputs(
    times_h: list[float],
    c_plasma: list[float],
    dose_mg: float,
    route: str,
    n_terminal: int,
) -> None:
    if len(times_h) < 3:
        raise ValueError("At least 3 time points are required for NCA.")
    if len(times_h) != len(c_plasma):
        raise ValueError("times_h and c_plasma must have the same length.")
    for i in range(1, len(times_h)):
        if times_h[i] <= times_h[i - 1]:
            raise ValueError("times_h must be strictly monotonically increasing.")
    if any(c < 0 for c in c_plasma):
        raise ValueError("c_plasma values must be >= 0.")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive.")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {sorted(_VALID_ROUTES)}, got '{route}'.")
    if n_terminal < 3:
        raise ValueError("n_terminal must be >= 3 for reliable lambda_z estimation.")


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _trapz_auc(times: list[float], conc: list[float]) -> float:
    """Linear trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += (conc[i] + conc[i - 1]) * (times[i] - times[i - 1]) / 2.0
    return auc


def _trapz_aumc(times: list[float], conc: list[float]) -> float:
    """Trapezoidal AUMC = integral of t*C(t) dt."""
    tc = [t * c for t, c in zip(times, conc, strict=True)]
    return _trapz_auc(times, tc)


def _estimate_lambda_z(
    times: list[float],
    conc: list[float],
    n_terminal: int,
) -> tuple[float, float]:
    """Estimate terminal elimination rate constant via log-linear regression.

    Uses the last n_terminal points with positive concentration.

    Returns
    -------
    (lambda_z, r_squared)
    """
    # Filter positive concentrations
    valid = [(t, c) for t, c in zip(times, conc, strict=True) if c > 0]
    if len(valid) < n_terminal:
        raise ValueError(
            f"Fewer than {n_terminal} positive concentration points available "
            f"for terminal regression (found {len(valid)})."
        )

    seg = valid[-n_terminal:]
    t_seg = np.array([p[0] for p in seg])
    c_seg = np.array([p[1] for p in seg])
    log_c = np.log(c_seg)

    coeffs = np.polyfit(t_seg, log_c, 1)
    lz = -coeffs[0]

    if lz <= 0:
        raise ValueError("Estimated lambda_z is non-positive; terminal phase is not declining.")

    predicted = np.polyval(coeffs, t_seg)
    ss_res = float(np.sum((log_c - predicted) ** 2))
    ss_tot = float(np.sum((log_c - np.mean(log_c)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0

    return float(lz), float(r2)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


def extended_nca(
    drug_name: str,
    times_h: list[float],
    c_plasma: list[float],
    dose_mg: float,
    route: str,
    n_terminal: int = 3,
    f_oral: float = 1.0,
    cmin_mg_L: float | None = None,
) -> ExtendedNCAResult:
    """Perform extended non-compartmental analysis.

    Computes standard NCA metrics (Cmax, Tmax, AUC, lambda_z, t1/2) plus
    extended metrics: AUMC, MRT, Vss (IV only), Vz, CL, MAT (oral only),
    swing (if Cmin provided), and peak-trough ratio (if Cmin provided).

    Parameters
    ----------
    drug_name:
        Drug identifier.
    times_h:
        Sampling times in hours (strictly increasing, >= 0).
    c_plasma:
        Plasma concentrations in mg/L (must be >= 0).
    dose_mg:
        Administered dose in mg.
    route:
        "iv" or "oral".
    n_terminal:
        Number of terminal points for lambda_z regression (default 3).
    f_oral:
        Oral bioavailability fraction (0-1), used for CL calculation for oral route.
        Defaults to 1.0 (CL/F).
    cmin_mg_L:
        Minimum (trough) concentration in mg/L for swing and peak-trough ratio.
        If None, swing and peak-trough ratio are not computed.

    Returns
    -------
    ExtendedNCAResult
        Dataclass with all NCA and extended metrics.
    """
    _validate_nca_inputs(times_h, c_plasma, dose_mg, route, n_terminal)

    if not (0.0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1].")
    if cmin_mg_L is not None and cmin_mg_L < 0:
        raise ValueError("cmin_mg_L must be non-negative.")

    t = list(times_h)
    c = list(c_plasma)
    notes: list[str] = []

    # --- Basic NCA ---
    cmax = max(c)
    tmax_h = t[c.index(cmax)]

    # AUC to last observed time (linear trapezoidal)
    auc_last = _trapz_auc(t, c)

    # Terminal elimination rate constant
    try:
        lambda_z, r2 = _estimate_lambda_z(t, c, n_terminal)
    except ValueError as exc:
        notes.append(f"lambda_z estimation failed: {exc}")
        lambda_z = float("nan")
        r2 = float("nan")

    t_half_h = math.log(2.0) / lambda_z if not math.isnan(lambda_z) else float("nan")

    # Last non-zero concentration
    c_last = 0.0
    t_last = t[-1]
    for ci in reversed(c):
        if ci > 0:
            c_last = ci
            break

    # AUC extrapolation: AUC_last -> inf = C_last / lambda_z
    if not math.isnan(lambda_z) and lambda_z > 0:
        auc_extrap = c_last / lambda_z
        auc_inf = auc_last + auc_extrap
        pct_extrap = (auc_extrap / auc_inf * 100.0) if auc_inf > 0 else 0.0
        if pct_extrap > 20.0:
            notes.append(f"AUC extrapolation {pct_extrap:.1f}% > 20%; AUC_inf may be unreliable.")
    else:
        auc_inf = float("nan")
        pct_extrap = float("nan")
        notes.append("AUC_inf could not be computed (lambda_z unavailable).")

    # --- Extended NCA: AUMC ---
    # AUMC_last via trapezoidal
    aumc_last = _trapz_aumc(t, c)

    # AUMC extrapolation: AUMC_last -> inf
    # aumc_extrap = C_last * t_last / lambda_z + C_last / lambda_z^2
    if not math.isnan(lambda_z) and lambda_z > 0:
        aumc_extrap = c_last * t_last / lambda_z + c_last / (lambda_z**2)
        aumc_inf = aumc_last + aumc_extrap
    else:
        aumc_inf = float("nan")

    # --- MRT ---
    if not math.isnan(auc_inf) and auc_inf > 0 and not math.isnan(aumc_inf):
        mrt_h = aumc_inf / auc_inf
    else:
        mrt_h = float("nan")
        notes.append("MRT could not be computed.")

    # --- Vss (IV only) ---
    # Vss = Dose * AUMC_inf / AUC_inf^2
    if route == "iv" and not math.isnan(aumc_inf) and not math.isnan(auc_inf) and auc_inf > 0:
        vss_L = dose_mg * aumc_inf / (auc_inf**2)
    else:
        vss_L = float("nan")
        if route == "oral":
            notes.append("Vss is not computed for oral route (use Vz instead).")

    # --- Vz (terminal volume of distribution) ---
    # Vz = Dose / (AUC_inf * lambda_z)  for IV
    # Vz = F * Dose / (AUC_inf * lambda_z)  for oral (apparent Vz/F)
    if not math.isnan(auc_inf) and auc_inf > 0 and not math.isnan(lambda_z):
        if route == "iv":
            vz_L = dose_mg / (auc_inf * lambda_z)
        else:
            vz_L = f_oral * dose_mg / (auc_inf * lambda_z)
    else:
        vz_L = float("nan")

    # --- CL ---
    # CL = Dose / AUC_inf (IV) or F*Dose / AUC_inf (oral)
    if not math.isnan(auc_inf) and auc_inf > 0:
        if route == "iv":
            cl_L_per_h = dose_mg / auc_inf
        else:
            cl_L_per_h = f_oral * dose_mg / auc_inf
    else:
        cl_L_per_h = float("nan")

    # --- MAT (oral only) ---
    # MAT = MRT_oral - MRT_iv
    # MRT_iv = 1/ke estimated as MRT - 1/lambda_z (simplified: 1/ke ≈ 1/lambda_z for 1-cpt)
    # Best estimate: MAT = MRT - 1/lambda_z  (since MRT_iv = 1/ke = Vd/CL = 1/lambda_z for 1-cpt)
    mat_h: float | None = None
    if route == "oral" and not math.isnan(mrt_h) and not math.isnan(lambda_z) and lambda_z > 0:
        mrt_iv_estimate = 1.0 / lambda_z
        mat_h = mrt_h - mrt_iv_estimate
        if mat_h < 0:
            notes.append(f"MAT computed as {mat_h:.3f} h (negative); MRT_oral < MRT_iv estimate.")

    # --- Swing and peak-trough ratio ---
    if cmin_mg_L is not None:
        if cmin_mg_L > 0:
            swing_pct = (cmax - cmin_mg_L) / cmin_mg_L * 100.0
            peak_trough = cmax / cmin_mg_L
            notes.append(f"Swing = {swing_pct:.1f}%, Peak:Trough ratio = {peak_trough:.2f}.")
        else:
            notes.append("Cmin = 0; swing and peak-trough ratio undefined.")

    notes.append(
        f"lambda_z from last {n_terminal} terminal points (R²={r2:.4f}); t½ = {t_half_h:.2f} h."
        if not math.isnan(r2)
        else "lambda_z regression failed."
    )

    return ExtendedNCAResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        cmax=cmax,
        tmax_h=tmax_h,
        auc_last=auc_last,
        auc_inf=auc_inf,
        lambda_z=lambda_z,
        t_half_h=t_half_h,
        aumc_inf=aumc_inf,
        mrt_h=mrt_h,
        vss_L=vss_L,
        vz_L=vz_L,
        cl_L_per_h=cl_L_per_h,
        mat_h=mat_h,
        r_squared_terminal=r2,
        pct_extrapolated=pct_extrap,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_nca_profiles(profiles: list[ExtendedNCAResult]) -> dict:
    """Compare multiple NCA profiles as a table keyed by metric name.

    Parameters
    ----------
    profiles:
        List of ExtendedNCAResult objects to compare.

    Returns
    -------
    dict
        Keys are metric names, values are lists of values (one per profile).
        Also includes "drug_name" and "route" keys.
    """
    if not profiles:
        raise ValueError("profiles must not be empty.")

    table: dict[str, list] = {
        "drug_name": [],
        "route": [],
        "dose_mg": [],
        "cmax": [],
        "tmax_h": [],
        "auc_last": [],
        "auc_inf": [],
        "lambda_z": [],
        "t_half_h": [],
        "aumc_inf": [],
        "mrt_h": [],
        "vss_L": [],
        "vz_L": [],
        "cl_L_per_h": [],
        "mat_h": [],
        "r_squared_terminal": [],
        "pct_extrapolated": [],
    }

    for p in profiles:
        table["drug_name"].append(p.drug_name)
        table["route"].append(p.route)
        table["dose_mg"].append(p.dose_mg)
        table["cmax"].append(p.cmax)
        table["tmax_h"].append(p.tmax_h)
        table["auc_last"].append(p.auc_last)
        table["auc_inf"].append(p.auc_inf)
        table["lambda_z"].append(p.lambda_z)
        table["t_half_h"].append(p.t_half_h)
        table["aumc_inf"].append(p.aumc_inf)
        table["mrt_h"].append(p.mrt_h)
        table["vss_L"].append(p.vss_L)
        table["vz_L"].append(p.vz_L)
        table["cl_L_per_h"].append(p.cl_L_per_h)
        table["mat_h"].append(p.mat_h)
        table["r_squared_terminal"].append(p.r_squared_terminal)
        table["pct_extrapolated"].append(p.pct_extrapolated)

    return table


# ===========================================================================
# Phase 600 — Pure-Python extended NCA API
# ===========================================================================


@dataclass(frozen=True)
class NCAResult:
    """Result from Phase 600 compute_nca."""

    drug_name: str
    route: str
    dose_mg: float
    cmax_mg_L: float
    tmax_h: float
    auc_0_t_mg_h_L: float
    auc_0_inf_mg_h_L: float
    aumc_0_inf: float
    mrt_h: float
    cl_nca_L_per_h: float
    vss_nca_L: float
    t_half_terminal_h: float
    lambda_z_per_h: float
    r2_lambda_z: float
    partial_auc: dict
    flip_flop: bool
    notes: str


def _linreg(xs: list, ys: list) -> tuple:
    """Simple linear regression. Returns (slope, intercept, r2)."""
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-15:
        slope = 0.0
        intercept = sum_y / n
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
    return slope, intercept, r2


def estimate_lambda_z(
    times_h: list,
    concentrations_mg_L: list,
    n_points: int = 4,
) -> tuple:
    """Estimate terminal elimination rate constant lambda_z and R².

    Uses last n_points with positive concentration for log-linear regression.

    Returns
    -------
    (lambda_z, r_squared)
    """
    valid = [(t, c) for t, c in zip(times_h, concentrations_mg_L) if c > 0]
    if len(valid) < n_points:
        raise ValueError(
            f"Need at least {n_points} positive concentration points, found {len(valid)}."
        )
    seg = valid[-n_points:]
    t_seg = [p[0] for p in seg]
    log_c_seg = [math.log(p[1]) for p in seg]
    slope, _intercept, r2 = _linreg(t_seg, log_c_seg)
    lambda_z = -slope
    if lambda_z <= 0:
        raise ValueError("Estimated lambda_z is non-positive; terminal phase is not declining.")
    return lambda_z, max(0.0, r2)


def _interp(times: list, concs: list, t_target: float) -> float:
    """Linear interpolation of concentration at t_target."""
    if t_target <= times[0]:
        return concs[0]
    if t_target >= times[-1]:
        return concs[-1]
    for i in range(1, len(times)):
        if times[i] >= t_target:
            t0, t1 = times[i - 1], times[i]
            c0, c1 = concs[i - 1], concs[i]
            frac = (t_target - t0) / (t1 - t0)
            return c0 + frac * (c1 - c0)
    return concs[-1]


def compute_partial_auc(
    times_h: list,
    concentrations_mg_L: list,
    intervals: list | None = None,
) -> dict:
    """Compute AUC over specified time intervals using trapezoidal rule.

    Parameters
    ----------
    times_h:
        Sampling times in hours.
    concentrations_mg_L:
        Plasma concentrations in mg/L.
    intervals:
        List of (t_start, t_end) tuples. Defaults to [(0,4), (0,8), (0,24)].

    Returns
    -------
    dict keyed by "0-4h", "0-8h", "0-24h" (or custom keys).
    """
    if intervals is None:
        intervals = [(0.0, 4.0), (0.0, 8.0), (0.0, 24.0)]
        keys = ["0-4h", "0-8h", "0-24h"]
    else:
        keys = [f"{s}-{e}h" for s, e in intervals]

    result = {}
    for key, (t_start, t_end) in zip(keys, intervals):
        # Build subset of times/concs within [t_start, t_end]
        sub_t = []
        sub_c = []
        # Add start boundary if not exactly in data
        if t_start < times_h[0]:
            sub_t.append(t_start)
            sub_c.append(_interp(times_h, concentrations_mg_L, t_start))
        elif t_start > times_h[-1]:
            result[key] = 0.0
            continue

        for t, c in zip(times_h, concentrations_mg_L):
            if t_start < t < t_end:
                sub_t.append(t)
                sub_c.append(c)
            elif t == t_start:
                sub_t.append(t)
                sub_c.append(c)

        # Add end boundary
        if t_end <= times_h[-1]:
            sub_t.append(t_end)
            sub_c.append(_interp(times_h, concentrations_mg_L, t_end))
        else:
            # t_end beyond data; use last available point
            if not sub_t or sub_t[-1] < times_h[-1]:
                sub_t.append(times_h[-1])
                sub_c.append(concentrations_mg_L[-1])

        # Ensure sorted
        paired = sorted(zip(sub_t, sub_c))
        sub_t = [p[0] for p in paired]
        sub_c = [p[1] for p in paired]

        # Trapezoidal
        auc = 0.0
        for i in range(1, len(sub_t)):
            auc += (sub_c[i] + sub_c[i - 1]) * (sub_t[i] - sub_t[i - 1]) / 2.0
        result[key] = auc

    return result


def compute_nca(
    drug_name: str,
    times_h: list,
    concentrations_mg_L: list,
    dose_mg: float,
    route: str = "oral",
    f_oral: float = 1.0,
    n_terminal_points: int = 4,
) -> NCAResult:
    """Compute extended NCA metrics.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    times_h:
        Sampling times in hours.
    concentrations_mg_L:
        Plasma concentrations in mg/L.
    dose_mg:
        Administered dose in mg.
    route:
        "iv" or "oral".
    f_oral:
        Oral bioavailability fraction (0-1].
    n_terminal_points:
        Number of terminal points for lambda_z regression.

    Returns
    -------
    NCAResult
    """
    # Validation
    if len(times_h) != len(concentrations_mg_L):
        raise ValueError("times_h and concentrations_mg_L must have the same length.")
    if len(times_h) < 4:
        raise ValueError("At least 4 time points are required for NCA.")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive.")
    if route not in {"iv", "oral"}:
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'.")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1].")

    t = list(times_h)
    c = list(concentrations_mg_L)

    # Cmax / Tmax
    cmax = max(c)
    tmax = t[c.index(cmax)]

    # AUC 0→t (linear trapezoidal)
    auc_0t = 0.0
    for i in range(1, len(t)):
        auc_0t += (c[i] + c[i - 1]) * (t[i] - t[i - 1]) / 2.0

    # AUMC 0→t (trapezoidal of t*C)
    tc = [ti * ci for ti, ci in zip(t, c)]
    aumc_0t = 0.0
    for i in range(1, len(t)):
        aumc_0t += (tc[i] + tc[i - 1]) * (t[i] - t[i - 1]) / 2.0

    # Terminal slope
    lambda_z, r2 = estimate_lambda_z(t, c, n_points=n_terminal_points)
    t_half = math.log(2.0) / lambda_z

    # Last positive concentration and its time
    c_last = 0.0
    t_last = t[-1]
    for ti, ci in zip(reversed(t), reversed(c)):
        if ci > 0:
            c_last = ci
            t_last = ti
            break

    # AUC extrapolation
    auc_extrap = c_last / lambda_z
    auc_0inf = auc_0t + auc_extrap

    # AUMC extrapolation
    aumc_extrap = t_last * c_last / lambda_z + c_last / (lambda_z**2)
    aumc_0inf = aumc_0t + aumc_extrap

    # MRT
    mrt = aumc_0inf / auc_0inf if auc_0inf > 0 else 0.0

    # CL
    if route == "iv":
        cl = dose_mg / auc_0inf
    else:
        cl = f_oral * dose_mg / auc_0inf

    # Vss
    vss = cl * mrt

    # Flip-flop detection: t_half > 2*MRT suggests flip-flop
    flip_flop = t_half > 2.0 * mrt if mrt > 0 else False

    # Partial AUC with default intervals
    partial = compute_partial_auc(t, c)

    # Notes
    note_parts = [
        f"lambda_z={lambda_z:.4f}/h (R²={r2:.4f})",
        f"t½={t_half:.2f}h",
        f"AUC_0t={auc_0t:.3f} mg·h/L",
        f"AUC_0inf={auc_0inf:.3f} mg·h/L",
        f"MRT={mrt:.2f}h",
        f"CL={cl:.3f} L/h",
        f"Vss={vss:.2f}L",
    ]
    if flip_flop:
        note_parts.append("Flip-flop kinetics suspected")
    notes = "; ".join(note_parts)

    return NCAResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_0_t_mg_h_L=auc_0t,
        auc_0_inf_mg_h_L=auc_0inf,
        aumc_0_inf=aumc_0inf,
        mrt_h=mrt,
        cl_nca_L_per_h=cl,
        vss_nca_L=vss,
        t_half_terminal_h=t_half,
        lambda_z_per_h=lambda_z,
        r2_lambda_z=r2,
        partial_auc=partial,
        flip_flop=flip_flop,
        notes=notes,
    )
