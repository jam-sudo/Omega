"""Liposomal drug formulation PK modeling.

Models the pharmacokinetics of conventional, PEGylated, and targeted liposome
formulations using a 3-compartment system: liposome depot → free drug → elimination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_VALID_LIPOSOME_TYPES = ("conventional", "pegylated", "targeted")


@dataclass
class LiposomePKResult:
    """Result from liposomal PK simulation."""

    drug_name: str
    dose_mg: float
    liposome_type: str
    k_release_per_h: float
    times_h: list[float] = field(default_factory=list)
    c_liposome_mg_L: list[float] = field(default_factory=list)
    c_free_mg_L: list[float] = field(default_factory=list)
    cmax_free: float = 0.0
    tmax_free_h: float = 0.0
    auc_free: float = 0.0
    auc_liposome: float = 0.0
    t_half_liposome_h: float = 0.0
    f_released: float = 0.0
    notes: str = ""


def _trapz(y: list[float], x: list[float]) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total


def simulate_liposome_pk(
    drug_name: str,
    dose_mg: float,
    liposome_type: str,
    cl_free_L_per_h: float,
    vd_free_L: float,
    k_release_per_h: float,
    cl_liposome_L_per_h: float,
    vd_liposome_L: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> LiposomePKResult:
    """Simulate PK of a liposomal drug formulation.

    Uses a 3-compartment forward Euler ODE system:
      - Liposome compartment: dC_lip/dt = -(k_release + ke_lip) * C_lip
      - Free drug: dC_free/dt = (k_release * C_lip * V_lip) / V_free - ke_free * C_free

    Args:
        drug_name: Name of the drug.
        dose_mg: Total dose in mg.
        liposome_type: One of "conventional", "pegylated", "targeted".
        cl_free_L_per_h: Clearance of free drug (L/h).
        vd_free_L: Volume of distribution of free drug (L).
        k_release_per_h: First-order release rate from liposome (1/h).
        cl_liposome_L_per_h: Clearance of liposome carrier (L/h).
        vd_liposome_L: Volume of distribution of liposome (L).
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        LiposomePKResult with time courses and summary PK parameters.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if liposome_type not in _VALID_LIPOSOME_TYPES:
        raise ValueError(
            f"liposome_type must be one of {_VALID_LIPOSOME_TYPES}, got {liposome_type!r}"
        )
    if cl_free_L_per_h <= 0:
        raise ValueError(f"cl_free_L_per_h must be positive, got {cl_free_L_per_h}")
    if vd_free_L <= 0:
        raise ValueError(f"vd_free_L must be positive, got {vd_free_L}")
    if k_release_per_h < 0:
        raise ValueError(f"k_release_per_h must be non-negative, got {k_release_per_h}")
    if cl_liposome_L_per_h <= 0:
        raise ValueError(f"cl_liposome_L_per_h must be positive, got {cl_liposome_L_per_h}")
    if vd_liposome_L <= 0:
        raise ValueError(f"vd_liposome_L must be positive, got {vd_liposome_L}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be positive, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be positive, got {dt_h}")

    # Apply liposome-type modifiers
    effective_k_release = k_release_per_h
    effective_cl_liposome = cl_liposome_L_per_h

    if liposome_type == "pegylated":
        # PEGylation dramatically reduces MPS clearance (10x longer circulation)
        effective_cl_liposome = cl_liposome_L_per_h * 0.1
    elif liposome_type == "targeted":
        # Targeted delivery → faster release at target site
        effective_k_release = k_release_per_h * 2.0

    # Rate constants
    ke_lip = effective_cl_liposome / vd_liposome_L
    ke_free = cl_free_L_per_h / vd_free_L

    # Initial conditions: all drug in liposome compartment
    c_lip = dose_mg / vd_liposome_L  # mg/L
    c_free = 0.0  # mg/L

    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = []
    c_lip_series: list[float] = []
    c_free_series: list[float] = []

    t = 0.0
    for _ in range(n_steps):
        times.append(t)
        c_lip_series.append(c_lip)
        c_free_series.append(c_free)

        # Derivatives (forward Euler)
        dc_lip = -(effective_k_release + ke_lip) * c_lip
        dc_free = (effective_k_release * c_lip * vd_liposome_L) / vd_free_L - ke_free * c_free

        c_lip = max(0.0, c_lip + dc_lip * dt_h)
        c_free = max(0.0, c_free + dc_free * dt_h)
        t += dt_h

    # Summary PK parameters for free drug
    cmax_free = max(c_free_series)
    tmax_free_h = times[c_free_series.index(cmax_free)]
    auc_free = _trapz(c_free_series, times)
    auc_liposome = _trapz(c_lip_series, times)

    # Liposome half-life from overall removal rate
    k_total_lip = effective_k_release + ke_lip
    t_half_liposome_h = 0.693 / k_total_lip if k_total_lip > 0 else float("inf")

    # Fraction of dose released into free drug compartment
    # = 1 - fraction remaining in liposome at end
    c_lip_final = c_lip_series[-1]
    amount_lip_final = c_lip_final * vd_liposome_L
    f_released = 1.0 - (amount_lip_final / dose_mg)
    f_released = max(0.0, min(1.0, f_released))

    notes_parts = [f"Liposome type: {liposome_type}"]
    if liposome_type == "pegylated":
        notes_parts.append("PEGylation reduces liposome CL by 10x for prolonged circulation")
    elif liposome_type == "targeted":
        notes_parts.append("Targeted delivery doubles release rate at target site")

    return LiposomePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        liposome_type=liposome_type,
        k_release_per_h=effective_k_release,
        times_h=times,
        c_liposome_mg_L=c_lip_series,
        c_free_mg_L=c_free_series,
        cmax_free=cmax_free,
        tmax_free_h=tmax_free_h,
        auc_free=auc_free,
        auc_liposome=auc_liposome,
        t_half_liposome_h=t_half_liposome_h,
        f_released=f_released,
        notes="; ".join(notes_parts),
    )


def compare_liposome_types(
    drug_name: str,
    dose_mg: float,
    cl_free_L_per_h: float,
    vd_free_L: float,
    k_release_per_h: float,
    cl_liposome_L_per_h: float = 1.0,
    vd_liposome_L: float = 5.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> dict[str, LiposomePKResult]:
    """Compare PK across all three liposome types.

    Args:
        drug_name: Name of the drug.
        dose_mg: Total dose in mg.
        cl_free_L_per_h: Clearance of free drug (L/h).
        vd_free_L: Volume of distribution of free drug (L).
        k_release_per_h: First-order release rate constant (1/h).
        cl_liposome_L_per_h: Clearance of liposome carrier (L/h).
        vd_liposome_L: Volume of distribution of liposome (L).
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        Dict with keys "conventional", "pegylated", "targeted" mapping to results.

    Raises:
        ValueError: If any parameter is invalid.
    """
    results: dict[str, LiposomePKResult] = {}
    for ltype in _VALID_LIPOSOME_TYPES:
        results[ltype] = simulate_liposome_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            liposome_type=ltype,
            cl_free_L_per_h=cl_free_L_per_h,
            vd_free_L=vd_free_L,
            k_release_per_h=k_release_per_h,
            cl_liposome_L_per_h=cl_liposome_L_per_h,
            vd_liposome_L=vd_liposome_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
    return results
