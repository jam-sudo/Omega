"""Advanced flip-flop kinetics: detection, modeling, and parameter estimation."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FlipFlopResult:
    drug_name: str
    ka_per_h: float
    ke_per_h: float
    flip_flop_detected: bool
    terminal_slope_per_h: float
    apparent_t_half_h: float
    true_elimination_t_half_h: float
    absorption_t_half_h: float
    flip_flop_ratio: float
    fraction_absorbed_pct: float
    notes: str


def _log_linear_slope(times: list, concs: list) -> float:
    """Compute log-linear slope from concentration-time data (last n points)."""
    n = len(times)
    if n < 2:
        return 0.0
    # Filter out non-positive concentrations
    pairs = [(t, c) for t, c in zip(times, concs) if c > 0]
    if len(pairs) < 2:
        return 0.0
    log_c = [math.log(c) for _, c in pairs]
    t_vals = [t for t, _ in pairs]
    n_pts = len(t_vals)
    sum_t = sum(t_vals)
    sum_lc = sum(log_c)
    sum_tt = sum(t * t for t in t_vals)
    sum_tlc = sum(t * lc for t, lc in zip(t_vals, log_c))
    denom = n_pts * sum_tt - sum_t * sum_t
    if abs(denom) < 1e-12:
        return 0.0
    return (n_pts * sum_tlc - sum_t * sum_lc) / denom


def simulate_flip_flop(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float,
    ke_per_h: float,
    vd_L: float,
    f_oral: float = 1.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> FlipFlopResult:
    """Simulate oral PK and detect if flip-flop occurs."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be positive")
    if ke_per_h <= 0:
        raise ValueError("ke_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be positive")

    # 1-compartment oral ODE: dA_gut/dt = -ka*A_gut; dC/dt = ka*A_gut/vd - ke*C
    a_gut = dose_mg * f_oral
    c = 0.0
    times = []
    concs = []
    t = 0.0
    steps = int(t_end_h / dt_h) + 1
    for _ in range(steps):
        times.append(t)
        concs.append(c)
        da_gut = -ka_per_h * a_gut
        dc = ka_per_h * a_gut / vd_L - ke_per_h * c
        a_gut += da_gut * dt_h
        if a_gut < 0:
            a_gut = 0.0
        c += dc * dt_h
        if c < 0:
            c = 0.0
        t += dt_h

    # Terminal slope: last 20% of profile
    n_total = len(times)
    n_terminal = max(4, int(0.2 * n_total))
    t_terminal = times[-n_terminal:]
    c_terminal = concs[-n_terminal:]
    terminal_slope = _log_linear_slope(t_terminal, c_terminal)

    # apparent_t_half from terminal slope (slope is negative in decline)
    if terminal_slope < 0:
        apparent_t_half = math.log(2) / abs(terminal_slope)
    else:
        apparent_t_half = math.log(2) / min(ka_per_h, ke_per_h)

    true_elimination_t_half = math.log(2) / ke_per_h
    absorption_t_half = math.log(2) / ka_per_h

    flip_flop_detected = ka_per_h < ke_per_h
    flip_flop_ratio = ke_per_h / ka_per_h

    # Fraction absorbed: trapezoidal AUC then compare to theoretical
    # fa = A_absorbed / dose; since f_oral accounts for bioavailability
    # Estimate via terminal AUC extrapolation
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt_h
    if concs[-1] > 0 and abs(terminal_slope) > 1e-9:
        auc += concs[-1] / abs(terminal_slope)

    # Theoretical AUC = f_oral * dose / (vd * ke)
    auc_theoretical = f_oral * dose_mg / (vd_L * ke_per_h)
    if auc_theoretical > 0:
        fraction_absorbed_pct = min(100.0, (auc / auc_theoretical) * 100.0)
    else:
        fraction_absorbed_pct = 0.0

    if flip_flop_detected:
        notes = (
            f"Flip-flop kinetics detected: ka ({ka_per_h:.3f}/h) < ke ({ke_per_h:.3f}/h). "
            f"Terminal slope reflects absorption rate, not elimination."
        )
    else:
        notes = (
            f"Normal kinetics: ka ({ka_per_h:.3f}/h) > ke ({ke_per_h:.3f}/h). "
            f"Terminal slope reflects elimination rate."
        )

    return FlipFlopResult(
        drug_name=drug_name,
        ka_per_h=ka_per_h,
        ke_per_h=ke_per_h,
        flip_flop_detected=flip_flop_detected,
        terminal_slope_per_h=terminal_slope,
        apparent_t_half_h=apparent_t_half,
        true_elimination_t_half_h=true_elimination_t_half,
        absorption_t_half_h=absorption_t_half,
        flip_flop_ratio=flip_flop_ratio,
        fraction_absorbed_pct=fraction_absorbed_pct,
        notes=notes,
    )


def detect_flip_flop_from_data(
    times_h: list,
    c_oral: list,
    c_iv: list,
    dose_oral: float,
    dose_iv: float,
) -> dict:
    """
    Compare oral and IV terminal slopes to detect flip-flop.

    If oral terminal slope < IV terminal slope (in absolute value), flip-flop is present.
    Returns: {"flip_flop": bool, "oral_slope": float, "iv_slope": float,
              "ratio": float, "notes": str}
    """
    if len(times_h) < 4 or len(c_oral) < 4 or len(c_iv) < 4:
        raise ValueError("Need at least 4 data points")

    # Use last 4 points for terminal slope estimation
    t_last = times_h[-4:]
    c_oral_last = c_oral[-4:]
    c_iv_last = c_iv[-4:]

    oral_slope_raw = _log_linear_slope(t_last, c_oral_last)
    iv_slope_raw = _log_linear_slope(t_last, c_iv_last)

    # Negative terminal slopes; we report magnitudes as positive "slopes"
    oral_slope = abs(oral_slope_raw)
    iv_slope = abs(iv_slope_raw)

    # Flip-flop: oral terminal half-life > IV terminal half-life
    # i.e., oral elimination appears slower → oral_slope < iv_slope
    flip_flop = oral_slope < iv_slope

    if iv_slope > 1e-9:
        ratio = oral_slope / iv_slope
    else:
        ratio = 1.0

    if flip_flop:
        notes = (
            f"Flip-flop detected: oral terminal slope ({oral_slope:.4f}/h) < "
            f"IV terminal slope ({iv_slope:.4f}/h). Oral t½ > IV t½."
        )
    else:
        notes = (
            f"No flip-flop: oral terminal slope ({oral_slope:.4f}/h) >= "
            f"IV terminal slope ({iv_slope:.4f}/h)."
        )

    return {
        "flip_flop": flip_flop,
        "oral_slope": oral_slope,
        "iv_slope": iv_slope,
        "ratio": ratio,
        "notes": notes,
    }


def estimate_ka_ke_from_oral(
    times_h: list,
    c_oral: list,
    dose_mg: float,
    vd_L: float,
    f_oral: float = 1.0,
) -> dict:
    """
    Estimate ka and ke from oral concentration-time data using method of residuals.

    Returns: {"ka": float, "ke": float, "flip_flop": bool}
    """
    if len(times_h) < 4 or len(c_oral) < 4:
        raise ValueError("Need at least 4 data points")
    if len(times_h) != len(c_oral):
        raise ValueError("times and concentrations must have same length")

    n = len(times_h)

    # Step 1: Fit terminal slope to get ke (last 4 points)
    n_terminal = min(4, n)
    t_terminal = times_h[-n_terminal:]
    c_terminal = c_oral[-n_terminal:]
    terminal_slope = _log_linear_slope(t_terminal, c_terminal)

    # ke from terminal slope (slope should be negative)
    if terminal_slope < 0:
        ke = abs(terminal_slope)
    else:
        # fallback
        ke = 0.1

    # Step 2: Extrapolate terminal line back to earlier times
    # log(C) = log(C_last_ref) + terminal_slope * (t - t_last_ref)
    # Find a reference point on the terminal line
    pairs_terminal = [(t, c) for t, c in zip(t_terminal, c_terminal) if c > 0]
    if pairs_terminal:
        t_ref = pairs_terminal[-1][0]
        c_ref = pairs_terminal[-1][1]
    else:
        t_ref = times_h[-1]
        c_ref = c_oral[-1] if c_oral[-1] > 0 else 1e-6

    # Step 3: Find ascending phase (Tmax roughly at peak)
    c_max_idx = max(range(n), key=lambda i: c_oral[i])
    tmax = times_h[c_max_idx]

    # Residuals for times before Tmax
    residual_times = []
    residuals = []
    for i in range(n):
        t_i = times_h[i]
        c_i = c_oral[i]
        if t_i <= tmax and c_i > 0:
            # Extrapolate terminal line
            c_extrap = c_ref * math.exp(terminal_slope * (t_i - t_ref))
            if c_extrap > c_i:
                residuals.append(c_extrap - c_i)
                residual_times.append(t_i)

    # Step 4: Fit residual line to get ka
    if len(residual_times) >= 2:
        residual_slope = _log_linear_slope(residual_times, residuals)
        if residual_slope < 0:
            ka = abs(residual_slope)
        else:
            ka = ke * 2.0  # fallback
    else:
        # Use Tmax approximation: tmax ≈ ln(ka/ke)/(ka-ke)
        # Simplified: if tmax is known, use bisection to solve
        # or a simple estimate: ka ≈ ke * exp(ke * tmax) for approximate solution
        if tmax > 0 and ke > 0:
            # Approximate: ka ≈ 2*ke for typical profiles when tmax ≈ ln(2)/ke
            ka = max(ke * 1.5, math.log(2) / tmax * 2.0)
        else:
            ka = ke * 2.0

    flip_flop = ka < ke

    return {
        "ka": ka,
        "ke": ke,
        "flip_flop": flip_flop,
    }
