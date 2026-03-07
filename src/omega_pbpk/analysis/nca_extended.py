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
