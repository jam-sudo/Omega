"""Phase 924 — Drug Amniotic Fluid Distribution.

Model drug distribution in amniotic fluid during pregnancy, relevant for fetal
drug therapy and maternal drug use safety assessment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AmnioticFluidPKResult:
    """Result of amniotic fluid PK simulation."""

    drug_name: str
    dose_mg: float
    gestational_week: int
    times_h: tuple
    c_amniotic_mg_L: tuple
    c_fetal_plasma_mg_L: tuple
    c_maternal_plasma_mg_L: tuple
    cmax_amniotic: float
    auc_amniotic: float
    cmax_fetal_plasma: float
    auc_fetal_plasma: float
    amniotic_to_maternal_ratio: float
    fetal_drug_reservoir_mg: float
    fetal_exposure_concern: bool
    notes: str


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return auc


def simulate_amniotic_fluid_pk(
    drug_name: str,
    dose_mg: float,
    gestational_week: int = 32,
    k_abs_per_h: float = 1.0,
    k_maternal_fetal_per_h: float = 0.05,
    k_fetal_amniotic_per_h: float = 0.1,
    k_amniotic_clearance_per_h: float = 0.05,
    cl_maternal_L_per_h: float = 10.0,
    vd_maternal_L: float = 40.0,
    vd_fetal_L: float = 2.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> AmnioticFluidPKResult:
    """Simulate drug PK in amniotic fluid during pregnancy.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    gestational_week:
        Gestational age in weeks (20–40).
    k_abs_per_h:
        First-order absorption rate constant (h⁻¹).
    k_maternal_fetal_per_h:
        Maternal-to-fetal transfer rate constant (h⁻¹).
    k_fetal_amniotic_per_h:
        Fetal-to-amniotic transfer rate constant (h⁻¹).
    k_amniotic_clearance_per_h:
        Amniotic fluid clearance rate constant (h⁻¹).
    cl_maternal_L_per_h:
        Maternal clearance (L/h).
    vd_maternal_L:
        Maternal volume of distribution (L).
    vd_fetal_L:
        Fetal volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration time step (h).

    Returns
    -------
    AmnioticFluidPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if gestational_week not in range(20, 41):
        raise ValueError(f"gestational_week must be in range(20, 41), got {gestational_week}")
    if cl_maternal_L_per_h <= 0:
        raise ValueError(f"cl_maternal_L_per_h must be > 0, got {cl_maternal_L_per_h}")

    # Amniotic volume increases with gestational age
    v_amniotic_L = 0.5 + (gestational_week - 20) * 0.03
    v_amniotic_L = max(0.2, min(1.5, v_amniotic_L))

    # Derived rate constants
    ke_maternal = cl_maternal_L_per_h / vd_maternal_L
    ke_fetal = 0.1  # simplified constant (h⁻¹)

    # Initial conditions (amounts in mg)
    a_gut = dose_mg
    a_maternal = 0.0
    a_fetal = 0.0
    a_amniotic = 0.0

    # Storage
    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = []
    c_maternal_list: list[float] = []
    c_fetal_list: list[float] = []
    c_amniotic_list: list[float] = []

    t = 0.0
    for _ in range(n_steps):
        c_mat = a_maternal / vd_maternal_L
        c_fet = a_fetal / vd_fetal_L
        c_amn = a_amniotic / v_amniotic_L

        times.append(t)
        c_maternal_list.append(c_mat)
        c_fetal_list.append(c_fet)
        c_amniotic_list.append(c_amn)

        # ODEs (Forward Euler)
        d_gut = -k_abs_per_h * a_gut
        d_maternal = (
            k_abs_per_h * a_gut - ke_maternal * a_maternal - k_maternal_fetal_per_h * a_maternal
        )
        d_fetal = (
            k_maternal_fetal_per_h * a_maternal
            - ke_fetal * a_fetal
            - k_fetal_amniotic_per_h * a_fetal
        )
        d_amniotic = k_fetal_amniotic_per_h * a_fetal - k_amniotic_clearance_per_h * a_amniotic

        a_gut += d_gut * dt_h
        a_maternal += d_maternal * dt_h
        a_fetal += d_fetal * dt_h
        a_amniotic += d_amniotic * dt_h

        # Clamp to non-negative
        a_gut = max(0.0, a_gut)
        a_maternal = max(0.0, a_maternal)
        a_fetal = max(0.0, a_fetal)
        a_amniotic = max(0.0, a_amniotic)

        t += dt_h

    cmax_amniotic = max(c_amniotic_list)
    cmax_fetal_plasma = max(c_fetal_list)
    auc_amniotic = _trapz(times, c_amniotic_list)
    auc_fetal_plasma = _trapz(times, c_fetal_list)
    auc_maternal = _trapz(times, c_maternal_list)

    amniotic_to_maternal_ratio = auc_amniotic / auc_maternal if auc_maternal > 0 else 0.0

    fetal_drug_reservoir_mg = c_amniotic_list[-1] * v_amniotic_L
    fetal_exposure_concern = amniotic_to_maternal_ratio > 0.3

    # Notes (priority order)
    if fetal_exposure_concern:
        notes = (
            "Significant amniotic fluid accumulation — fetal exposure via amniotic swallowing"
            " is a concern"
        )
    elif gestational_week > 34:
        notes = (
            "Late gestation: larger amniotic volume dilutes drug but increases total fetal"
            " reservoir"
        )
    else:
        notes = (
            "Limited amniotic fluid distribution — fetal drug exposure primarily via placental"
            " transfer"
        )

    return AmnioticFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        gestational_week=gestational_week,
        times_h=tuple(times),
        c_amniotic_mg_L=tuple(c_amniotic_list),
        c_fetal_plasma_mg_L=tuple(c_fetal_list),
        c_maternal_plasma_mg_L=tuple(c_maternal_list),
        cmax_amniotic=cmax_amniotic,
        auc_amniotic=auc_amniotic,
        cmax_fetal_plasma=cmax_fetal_plasma,
        auc_fetal_plasma=auc_fetal_plasma,
        amniotic_to_maternal_ratio=amniotic_to_maternal_ratio,
        fetal_drug_reservoir_mg=fetal_drug_reservoir_mg,
        fetal_exposure_concern=fetal_exposure_concern,
        notes=notes,
    )


def compare_gestational_ages(
    drug_name: str,
    dose_mg: float,
    weeks: list[int] | None = None,
    **kwargs,
) -> list[AmnioticFluidPKResult]:
    """Compare amniotic fluid PK across gestational ages.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    weeks:
        List of gestational weeks to simulate (default: [20, 24, 28, 32, 36, 40]).
    **kwargs:
        Additional keyword arguments passed to :func:`simulate_amniotic_fluid_pk`.

    Returns
    -------
    list[AmnioticFluidPKResult]
        Results sorted by gestational_week ascending.
    """
    if weeks is None:
        weeks = [20, 24, 28, 32, 36, 40]

    results = []
    for week in weeks:
        result = simulate_amniotic_fluid_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            gestational_week=week,
            **kwargs,
        )
        results.append(result)

    results.sort(key=lambda r: r.gestational_week)
    return results
