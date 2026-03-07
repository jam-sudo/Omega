"""
Zero-order and mixed-order drug release kinetics.

Provides functions for simulating various release mechanisms:
- Zero-order (constant rate)
- First-order (concentration-dependent)
- Mixed-order (combined zero + first)
- Weibull (empirical)
"""

import math


def zero_order_release(
    dose_mg: float,
    release_duration_h: float,
    t_end_h: float | None = None,
    dt_h: float = 0.1,
) -> dict:
    """
    Simulate zero-order (constant rate) drug release.

    Release rate = dose_mg / release_duration_h (mg/h).
    Release stops at t > release_duration_h.

    Returns:
        times_h, cumulative_released_mg, release_rate_mg_per_h, f_released_at_end
    """
    if dose_mg < 0:
        raise ValueError("dose_mg must be non-negative")
    if release_duration_h <= 0:
        raise ValueError("release_duration_h must be positive")
    if dt_h <= 0:
        raise ValueError("dt_h must be positive")

    if t_end_h is None:
        t_end_h = release_duration_h * 1.5

    rate = dose_mg / release_duration_h  # mg/h

    times_h = []
    cumulative_released_mg = []
    release_rate_mg_per_h = []

    t = 0.0
    cum = 0.0
    while t <= t_end_h + 1e-9:
        times_h.append(t)
        cumulative_released_mg.append(cum)
        if t < release_duration_h:
            release_rate_mg_per_h.append(rate)
            cum_next = cum + rate * dt_h
            cum = min(cum_next, dose_mg)
        else:
            release_rate_mg_per_h.append(0.0)
        t += dt_h

    f_released_at_end = cumulative_released_mg[-1] / dose_mg if dose_mg > 0 else 0.0

    return {
        "times_h": times_h,
        "cumulative_released_mg": cumulative_released_mg,
        "release_rate_mg_per_h": release_rate_mg_per_h,
        "f_released_at_end": f_released_at_end,
    }


def first_order_release(
    dose_mg: float,
    k_release_per_h: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> dict:
    """
    Simulate first-order drug release.

    dA/dt = -k_release * A  (forward Euler)

    Returns:
        times_h, cumulative_released_mg, a_remaining_mg, f_released_at_end
    """
    if dose_mg < 0:
        raise ValueError("dose_mg must be non-negative")
    if k_release_per_h <= 0:
        raise ValueError("k_release_per_h must be positive")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be positive")
    if dt_h <= 0:
        raise ValueError("dt_h must be positive")

    times_h = []
    cumulative_released_mg = []
    a_remaining_mg = []

    t = 0.0
    a = dose_mg  # amount remaining
    while t <= t_end_h + 1e-9:
        times_h.append(t)
        a_remaining_mg.append(a)
        cumulative_released_mg.append(dose_mg - a)
        # Forward Euler
        dadt = -k_release_per_h * a
        a = max(a + dadt * dt_h, 0.0)
        t += dt_h

    f_released_at_end = cumulative_released_mg[-1] / dose_mg if dose_mg > 0 else 0.0

    return {
        "times_h": times_h,
        "cumulative_released_mg": cumulative_released_mg,
        "a_remaining_mg": a_remaining_mg,
        "f_released_at_end": f_released_at_end,
    }


def mixed_order_release(
    dose_mg: float,
    k_zero: float,
    k_first: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> dict:
    """
    Simulate mixed-order (zero + first order) drug release.

    dA/dt = -(k_zero + k_first * A)

    Forward Euler integration.

    Returns:
        times_h, cumulative_released_mg, a_remaining_mg
    """
    if dose_mg < 0:
        raise ValueError("dose_mg must be non-negative")
    if k_zero < 0:
        raise ValueError("k_zero must be non-negative")
    if k_first < 0:
        raise ValueError("k_first must be non-negative")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be positive")
    if dt_h <= 0:
        raise ValueError("dt_h must be positive")

    times_h = []
    cumulative_released_mg = []
    a_remaining_mg = []

    t = 0.0
    a = dose_mg
    while t <= t_end_h + 1e-9:
        times_h.append(t)
        a_remaining_mg.append(a)
        cumulative_released_mg.append(dose_mg - a)
        dadt = -(k_zero + k_first * a)
        a = max(a + dadt * dt_h, 0.0)
        t += dt_h

    return {
        "times_h": times_h,
        "cumulative_released_mg": cumulative_released_mg,
        "a_remaining_mg": a_remaining_mg,
    }


def weibull_release(
    dose_mg: float,
    scale_h: float,
    shape: float,
    t_end_h: float = 24.0,
    n_points: int = 100,
) -> dict:
    """
    Simulate drug release using the Weibull function.

    F(t) = 1 - exp(-(t/scale)^shape)
    Q(t) = dose * F(t)

    Returns:
        times_h, cumulative_released_mg, f_released
    """
    if dose_mg < 0:
        raise ValueError("dose_mg must be non-negative")
    if scale_h <= 0:
        raise ValueError("scale_h must be positive")
    if shape <= 0:
        raise ValueError("shape must be positive")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be positive")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    times_h = []
    cumulative_released_mg = []
    f_released = []

    step = t_end_h / (n_points - 1)
    for i in range(n_points):
        t = i * step
        if t == 0.0:
            f = 0.0
        else:
            f = 1.0 - math.exp(-((t / scale_h) ** shape))
        times_h.append(t)
        f_released.append(f)
        cumulative_released_mg.append(dose_mg * f)

    return {
        "times_h": times_h,
        "cumulative_released_mg": cumulative_released_mg,
        "f_released": f_released,
    }


def compare_release_profiles(dose_mg: float, t_end_h: float = 24.0) -> list:
    """
    Compare zero-order, first-order, and Weibull release profiles.

    Returns list of dicts: model, f_released_8h, f_released_end, profile
    """
    if dose_mg < 0:
        raise ValueError("dose_mg must be non-negative")

    key_times = [0, 2, 4, 6, 8, 12, 16, 20, 24]
    key_times = [t for t in key_times if t <= t_end_h]

    results = []

    # Zero-order: constant release over t_end_h
    zo = zero_order_release(dose_mg, release_duration_h=t_end_h, t_end_h=t_end_h, dt_h=0.1)
    # Interpolate at key times
    zo_profile = _interpolate_profile(zo["times_h"], zo["cumulative_released_mg"], key_times)
    zo_f8h = (
        _interpolate_at_t(zo["times_h"], zo["cumulative_released_mg"], 8.0) / dose_mg
        if dose_mg > 0
        else 0.0
    )
    results.append(
        {
            "model": "zero_order",
            "f_released_8h": zo_f8h,
            "f_released_end": zo["f_released_at_end"],
            "profile": zo_profile,
        }
    )

    # First-order: k=0.2/h
    fo = first_order_release(dose_mg, k_release_per_h=0.2, t_end_h=t_end_h, dt_h=0.1)
    fo_f8h = (
        _interpolate_at_t(fo["times_h"], fo["cumulative_released_mg"], 8.0) / dose_mg
        if dose_mg > 0
        else 0.0
    )
    fo_profile = _interpolate_profile(fo["times_h"], fo["cumulative_released_mg"], key_times)
    results.append(
        {
            "model": "first_order",
            "f_released_8h": fo_f8h,
            "f_released_end": fo["f_released_at_end"],
            "profile": fo_profile,
        }
    )

    # Weibull: scale=t_end_h/3, shape=1.5
    wb = weibull_release(dose_mg, scale_h=t_end_h / 3.0, shape=1.5, t_end_h=t_end_h, n_points=200)
    wb_f8h = (
        _interpolate_at_t(wb["times_h"], wb["cumulative_released_mg"], 8.0) / dose_mg
        if dose_mg > 0
        else 0.0
    )
    wb_profile = _interpolate_profile(wb["times_h"], wb["cumulative_released_mg"], key_times)
    wb_f_end = wb["f_released"][-1]
    results.append(
        {
            "model": "weibull",
            "f_released_8h": wb_f8h,
            "f_released_end": wb_f_end,
            "profile": wb_profile,
        }
    )

    return results


def _interpolate_at_t(times, values, target_t):
    """Linear interpolation to get value at target_t."""
    if target_t <= times[0]:
        return values[0]
    if target_t >= times[-1]:
        return values[-1]
    for i in range(len(times) - 1):
        if times[i] <= target_t <= times[i + 1]:
            frac = (target_t - times[i]) / (times[i + 1] - times[i])
            return values[i] + frac * (values[i + 1] - values[i])
    return values[-1]


def _interpolate_profile(times, values, key_times):
    """Get cumulative values at a list of key times."""
    return [_interpolate_at_t(times, values, t) for t in key_times]
