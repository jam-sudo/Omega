"""
Phase 587 — PK-biomarker correlation modeling for drug response prediction.
"""

from __future__ import annotations

import math


def _pearson_r(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return num / (denom_x * denom_y)


def _trapz(y: list[float], x: list[float]) -> float:
    """Trapezoidal integration."""
    if len(y) < 2:
        return 0.0
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(y)))


_VALID_MODEL_TYPES = {
    "inhibition_kin",
    "stimulation_kin",
    "inhibition_kout",
    "stimulation_kout",
}


def indirect_response_model(
    times_h: list[float],
    c_plasma: list[float],
    kin: float,
    kout: float,
    imax: float,
    ic50: float,
    hill: float = 1.0,
    baseline: float = None,
    model_type: str = "inhibition_kin",
) -> dict:
    """
    Simulate an indirect response PK/PD model.

    Parameters
    ----------
    times_h    : time points (h), evenly spaced
    c_plasma   : plasma concentration at each time point
    kin        : zero-order production rate constant
    kout       : first-order degradation rate constant
    imax       : maximum inhibition/stimulation fraction (0–1 for inhibition)
    ic50       : concentration at half-maximum effect
    hill       : Hill coefficient
    baseline   : initial response value (default kin/kout)
    model_type : one of 'inhibition_kin', 'stimulation_kin',
                 'inhibition_kout', 'stimulation_kout'

    Returns
    -------
    dict with keys: times_h, response, r_baseline, r_min, r_max,
                    auc_response, notes
    """
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Invalid model_type '{model_type}'. Must be one of {sorted(_VALID_MODEL_TYPES)}."
        )
    if len(times_h) < 2:
        raise ValueError("times_h must have at least 2 elements.")
    if len(times_h) != len(c_plasma):
        raise ValueError("times_h and c_plasma must have the same length.")

    r0 = kin / kout if baseline is None else baseline
    dt = times_h[1] - times_h[0]

    response = [r0]
    r = r0

    for i in range(1, len(times_h)):
        c = max(c_plasma[i - 1], 0.0)
        c_hill = c**hill
        ic50_hill = ic50**hill
        modulator = c_hill / (ic50_hill + c_hill) if (ic50_hill + c_hill) > 0 else 0.0

        if model_type == "inhibition_kin":
            dRdt = kin * (1.0 - imax * modulator) - kout * r
        elif model_type == "stimulation_kin":
            dRdt = kin * (1.0 + imax * modulator) - kout * r
        elif model_type == "inhibition_kout":
            dRdt = kin - kout * (1.0 - imax * modulator) * r
        else:  # stimulation_kout
            dRdt = kin - kout * (1.0 + imax * modulator) * r

        r = r + dRdt * dt
        response.append(r)

    auc = _trapz(response, times_h)
    return {
        "times_h": list(times_h),
        "response": response,
        "r_baseline": r0,
        "r_min": min(response),
        "r_max": max(response),
        "auc_response": auc,
        "notes": f"Indirect response model: {model_type}",
    }


def hysteresis_detection(
    times_h: list[float],
    c_plasma: list[float],
    response: list[float],
) -> dict:
    """
    Detect PK/PD hysteresis by comparing C-E correlations on
    ascending vs descending plasma concentration limbs.

    Returns
    -------
    dict with keys: ascending_r, descending_r, hysteresis_detected, notes
    """
    if len(times_h) != len(c_plasma) or len(times_h) != len(response):
        raise ValueError("times_h, c_plasma, and response must all have the same length.")
    if len(times_h) < 4:
        raise ValueError("Need at least 4 time points to detect hysteresis.")

    # Find Cmax index
    cmax_idx = c_plasma.index(max(c_plasma))

    asc_c = c_plasma[: cmax_idx + 1]
    asc_e = response[: cmax_idx + 1]
    desc_c = c_plasma[cmax_idx:]
    desc_e = response[cmax_idx:]

    r_asc = _pearson_r(asc_c, asc_e) if len(asc_c) >= 2 else 0.0
    r_desc = _pearson_r(desc_c, desc_e) if len(desc_c) >= 2 else 0.0

    detected = abs(r_asc - r_desc) > 0.2

    return {
        "ascending_r": r_asc,
        "descending_r": r_desc,
        "hysteresis_detected": detected,
        "notes": (
            "Hysteresis detected: ascending and descending C-E correlations differ."
            if detected
            else "No significant hysteresis detected."
        ),
    }


def pk_pd_link_ke0(
    c_plasma: list[float],
    response: list[float],
    times_h: list[float],
    ke0_per_h: float = 0.5,
) -> dict:
    """
    Link PK to PD via an effect-site compartment model.

    dCe/dt = ke0 * (Cp - Ce)   (forward Euler)

    Parameters
    ----------
    c_plasma  : plasma concentration time series
    response  : observed PD response time series
    times_h   : time points (h), evenly spaced
    ke0_per_h : equilibration rate constant (h^-1)

    Returns
    -------
    dict with keys: times_h, ce_effect_site, correlation_r
    """
    if len(times_h) < 2:
        raise ValueError("times_h must have at least 2 elements.")
    if len(times_h) != len(c_plasma) or len(times_h) != len(response):
        raise ValueError("times_h, c_plasma, and response must all have the same length.")

    dt = times_h[1] - times_h[0]
    ce = [0.0]
    for i in range(1, len(times_h)):
        cp = c_plasma[i - 1]
        ce_prev = ce[-1]
        ce_new = ce_prev + ke0_per_h * (cp - ce_prev) * dt
        ce.append(ce_new)

    r = _pearson_r(ce, response)

    return {
        "times_h": list(times_h),
        "ce_effect_site": ce,
        "correlation_r": r,
    }
