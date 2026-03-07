"""Enzyme kinetics simulation — Phase 455."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnzymeKineticsResult:
    """Result from enzyme kinetics simulation."""

    inhibition_type: str
    km_apparent: float
    vmax_apparent: float
    substrate_range: list[float]
    rates: list[float]
    max_rate: float
    notes: str


def michaelis_menten_rate(
    substrate_uM: float,
    vmax_uM_per_min: float,
    km_uM: float,
) -> float:
    """Compute Michaelis-Menten reaction rate.

    Args:
        substrate_uM: Substrate concentration (uM). Must be >= 0.
        vmax_uM_per_min: Maximum reaction rate (uM/min). Must be > 0.
        km_uM: Michaelis constant (uM). Must be > 0.

    Returns:
        Reaction rate in uM/min.

    Raises:
        ValueError: If any parameter is out of valid range.
    """
    if vmax_uM_per_min <= 0:
        raise ValueError(f"vmax must be > 0, got {vmax_uM_per_min}")
    if km_uM <= 0:
        raise ValueError(f"km must be > 0, got {km_uM}")
    if substrate_uM < 0:
        raise ValueError(f"substrate_uM must be >= 0, got {substrate_uM}")

    return (vmax_uM_per_min * substrate_uM) / (km_uM + substrate_uM)


def competitive_inhibition(
    substrate_uM: float,
    vmax: float,
    km: float,
    ki_uM: float,
    inhibitor_uM: float,
) -> float:
    """Compute reaction rate under competitive inhibition.

    Competitive inhibitor increases apparent Km; Vmax is unchanged.

    Args:
        substrate_uM: Substrate concentration (uM).
        vmax: Maximum reaction rate (uM/min).
        km: Michaelis constant (uM).
        ki_uM: Inhibition constant (uM). Must be > 0.
        inhibitor_uM: Inhibitor concentration (uM). Must be >= 0.

    Returns:
        Reaction rate in uM/min.

    Raises:
        ValueError: If ki_uM <= 0 or inhibitor_uM < 0.
    """
    if ki_uM <= 0:
        raise ValueError(f"ki_uM must be > 0, got {ki_uM}")
    if inhibitor_uM < 0:
        raise ValueError(f"inhibitor_uM must be >= 0, got {inhibitor_uM}")

    km_apparent = km * (1.0 + inhibitor_uM / ki_uM)
    return michaelis_menten_rate(substrate_uM, vmax, km_apparent)


def uncompetitive_inhibition(
    substrate_uM: float,
    vmax: float,
    km: float,
    ki_uM: float,
    inhibitor_uM: float,
) -> float:
    """Compute reaction rate under uncompetitive inhibition.

    Uncompetitive inhibitor decreases both Vmax and Km proportionally.

    Args:
        substrate_uM: Substrate concentration (uM).
        vmax: Maximum reaction rate (uM/min).
        km: Michaelis constant (uM).
        ki_uM: Inhibition constant (uM). Must be > 0.
        inhibitor_uM: Inhibitor concentration (uM). Must be >= 0.

    Returns:
        Reaction rate in uM/min.

    Raises:
        ValueError: If ki_uM <= 0 or inhibitor_uM < 0.
    """
    if ki_uM <= 0:
        raise ValueError(f"ki_uM must be > 0, got {ki_uM}")
    if inhibitor_uM < 0:
        raise ValueError(f"inhibitor_uM must be >= 0, got {inhibitor_uM}")

    factor = 1.0 + inhibitor_uM / ki_uM
    vmax_apparent = vmax / factor
    km_apparent = km / factor
    return michaelis_menten_rate(substrate_uM, vmax_apparent, km_apparent)


def mixed_inhibition(
    substrate_uM: float,
    vmax: float,
    km: float,
    ki_uM: float,
    ki_prime_uM: float,
    inhibitor_uM: float,
) -> float:
    """Compute reaction rate under mixed inhibition.

    Mixed inhibition changes both Km and Vmax independently.

    Args:
        substrate_uM: Substrate concentration (uM).
        vmax: Maximum reaction rate (uM/min).
        km: Michaelis constant (uM).
        ki_uM: Inhibition constant for competitive component (uM). Must be > 0.
        ki_prime_uM: Inhibition constant for uncompetitive component (uM). Must be > 0.
        inhibitor_uM: Inhibitor concentration (uM). Must be >= 0.

    Returns:
        Reaction rate in uM/min.

    Raises:
        ValueError: If ki parameters are <= 0 or inhibitor_uM < 0.
    """
    if ki_uM <= 0:
        raise ValueError(f"ki_uM must be > 0, got {ki_uM}")
    if ki_prime_uM <= 0:
        raise ValueError(f"ki_prime_uM must be > 0, got {ki_prime_uM}")
    if inhibitor_uM < 0:
        raise ValueError(f"inhibitor_uM must be >= 0, got {inhibitor_uM}")

    alpha = 1.0 + inhibitor_uM / ki_uM
    alpha_prime = 1.0 + inhibitor_uM / ki_prime_uM
    vmax_apparent = vmax / alpha_prime
    km_apparent = km * alpha / alpha_prime
    return michaelis_menten_rate(substrate_uM, vmax_apparent, km_apparent)


def simulate_enzyme_kinetics(
    substrate_conc_range: tuple[float, float],
    vmax: float,
    km: float,
    inhibitor_uM: float = 0.0,
    ki_uM: float | None = None,
    inhibition_type: str = "none",
    n_points: int = 50,
) -> EnzymeKineticsResult:
    """Simulate enzyme kinetics over a substrate concentration range.

    Args:
        substrate_conc_range: (min_uM, max_uM) substrate concentration range.
        vmax: Maximum reaction rate (uM/min).
        km: Michaelis constant (uM).
        inhibitor_uM: Inhibitor concentration (uM). Default 0.0.
        ki_uM: Inhibition constant (uM). Required when inhibition_type != 'none'.
        inhibition_type: One of 'none', 'competitive', 'uncompetitive', 'mixed'.
        n_points: Number of substrate concentration points. Default 50.

    Returns:
        EnzymeKineticsResult with rates and apparent kinetic parameters.

    Raises:
        ValueError: If inhibition_type is invalid or ki_uM missing when required.
    """
    valid_types = {"none", "competitive", "uncompetitive", "mixed"}
    if inhibition_type not in valid_types:
        raise ValueError(f"inhibition_type must be one of {valid_types}, got '{inhibition_type}'")

    if inhibition_type != "none" and ki_uM is None:
        raise ValueError(f"ki_uM must be provided when inhibition_type='{inhibition_type}'")

    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    s_min, s_max = substrate_conc_range
    if s_min < 0 or s_max <= s_min:
        raise ValueError(
            f"substrate_conc_range must satisfy 0 <= s_min < s_max, got ({s_min}, {s_max})"
        )

    # Build linearly spaced substrate concentrations
    step = (s_max - s_min) / (n_points - 1)
    substrates = [s_min + i * step for i in range(n_points)]

    # Determine apparent parameters and compute rates
    rates: list[float] = []

    if inhibition_type == "none" or inhibitor_uM == 0.0:
        km_apparent = km
        vmax_apparent = vmax
        for s in substrates:
            rates.append(michaelis_menten_rate(s, vmax, km))
        notes = "Standard Michaelis-Menten kinetics (no inhibition)."

    elif inhibition_type == "competitive":
        alpha = 1.0 + inhibitor_uM / ki_uM  # type: ignore[operator]
        km_apparent = km * alpha
        vmax_apparent = vmax
        for s in substrates:
            rates.append(competitive_inhibition(s, vmax, km, ki_uM, inhibitor_uM))  # type: ignore[arg-type]
        notes = (
            f"Competitive inhibition: Km_app={km_apparent:.3f} uM "
            f"(alpha={alpha:.3f}), Vmax unchanged."
        )

    elif inhibition_type == "uncompetitive":
        factor = 1.0 + inhibitor_uM / ki_uM  # type: ignore[operator]
        km_apparent = km / factor
        vmax_apparent = vmax / factor
        for s in substrates:
            rates.append(uncompetitive_inhibition(s, vmax, km, ki_uM, inhibitor_uM))  # type: ignore[arg-type]
        notes = (
            f"Uncompetitive inhibition: Km_app={km_apparent:.3f} uM, "
            f"Vmax_app={vmax_apparent:.3f} uM/min (factor={factor:.3f})."
        )

    else:  # mixed
        # Use ki_prime = ki_uM for symmetric mixed inhibition by default
        ki_prime = ki_uM  # type: ignore[assignment]
        alpha = 1.0 + inhibitor_uM / ki_uM  # type: ignore[operator]
        alpha_prime = 1.0 + inhibitor_uM / ki_prime  # type: ignore[operator]
        vmax_apparent = vmax / alpha_prime
        km_apparent = km * alpha / alpha_prime
        for s in substrates:
            rates.append(
                mixed_inhibition(s, vmax, km, ki_uM, ki_prime, inhibitor_uM)  # type: ignore[arg-type]
            )
        notes = (
            f"Mixed inhibition: Km_app={km_apparent:.3f} uM, Vmax_app={vmax_apparent:.3f} uM/min."
        )

    max_rate = max(rates) if rates else 0.0

    return EnzymeKineticsResult(
        inhibition_type=inhibition_type,
        km_apparent=km_apparent,
        vmax_apparent=vmax_apparent,
        substrate_range=substrates,
        rates=rates,
        max_rate=max_rate,
        notes=notes,
    )


def calculate_kinetic_params(
    substrate_uM_list: list[float],
    rate_list: list[float],
) -> dict[str, float]:
    """Estimate Vmax and Km from substrate-rate data using Lineweaver-Burk OLS.

    Performs ordinary least squares on the double-reciprocal (Lineweaver-Burk)
    transform: 1/v = (Km/Vmax) * (1/[S]) + 1/Vmax.

    Args:
        substrate_uM_list: List of substrate concentrations (uM). All must be > 0.
        rate_list: Corresponding reaction rates (uM/min). All must be > 0.

    Returns:
        Dict with keys:
            - 'vmax_estimated': Estimated maximum rate (uM/min).
            - 'km_estimated': Estimated Michaelis constant (uM).

    Raises:
        ValueError: If lists have < 2 points, lengths differ, or data contains
            non-positive values.
    """
    if len(substrate_uM_list) != len(rate_list):
        raise ValueError("substrate_uM_list and rate_list must have the same length.")
    if len(substrate_uM_list) < 2:
        raise ValueError("At least 2 data points are required.")

    for s in substrate_uM_list:
        if s <= 0:
            raise ValueError(f"All substrate concentrations must be > 0, got {s}")
    for r in rate_list:
        if r <= 0:
            raise ValueError(f"All rates must be > 0, got {r}")

    n = len(substrate_uM_list)
    xs = [1.0 / s for s in substrate_uM_list]  # 1/[S]
    ys = [1.0 / r for r in rate_list]  # 1/v

    # OLS: y = slope * x + intercept
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=True))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-15:
        raise ValueError("Cannot fit Lineweaver-Burk: degenerate data (all same 1/[S]).")

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    if intercept <= 0:
        raise ValueError(
            f"Lineweaver-Burk fit yields non-positive intercept ({intercept:.4g}); "
            "data may be inconsistent with Michaelis-Menten kinetics."
        )

    vmax_estimated = 1.0 / intercept
    km_estimated = slope * vmax_estimated

    return {
        "vmax_estimated": vmax_estimated,
        "km_estimated": km_estimated,
    }
