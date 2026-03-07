"""Receptor binding kinetics simulation — Phase 467."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReceptorBindingResult:
    """Result of a receptor binding kinetics simulation."""

    drug_name: str
    kd_nM: float
    residence_time_h: float
    times_h: list[float] = field(default_factory=list)
    drug_free_nM: list[float] = field(default_factory=list)
    receptor_free_nM: list[float] = field(default_factory=list)
    complex_nM: list[float] = field(default_factory=list)
    occupancy_pct: list[float] = field(default_factory=list)
    max_occupancy_pct: float = 0.0
    time_to_50pct_occupancy_h: float = 0.0
    notes: str = ""


def calculate_kd(kon_per_nM_per_h: float, koff_per_h: float) -> float:
    """Calculate equilibrium dissociation constant Kd = koff / kon (nM)."""
    if kon_per_nM_per_h <= 0:
        raise ValueError("kon must be positive")
    if koff_per_h < 0:
        raise ValueError("koff must be non-negative")
    return koff_per_h / kon_per_nM_per_h


def residence_time(koff_per_h: float) -> float:
    """Calculate mean residence time MRT = 1/koff (hours)."""
    if koff_per_h <= 0:
        raise ValueError("koff must be positive")
    return 1.0 / koff_per_h


def occupancy_at_steady_state(drug_conc_nM: float, kd_nM: float) -> float:
    """Calculate fractional receptor occupancy at steady state (Langmuir isotherm).

    Returns fraction (0-1): occupancy = [D] / (Kd + [D])
    """
    if drug_conc_nM < 0:
        raise ValueError("drug_conc_nM must be non-negative")
    if kd_nM <= 0:
        raise ValueError("kd_nM must be positive")
    return drug_conc_nM / (kd_nM + drug_conc_nM)


def simulate_receptor_binding(
    drug_name: str,
    dose_conc_nM: float,
    kon_per_nM_per_h: float,
    koff_per_h: float,
    receptor_conc_nM: float = 10.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> ReceptorBindingResult:
    """Simulate drug-receptor binding ODE using forward Euler integration.

    ODE system:
        d[DR]/dt = kon*[D]*[R] - koff*[DR]
        d[D]/dt  = -kon*[D]*[R] + koff*[DR]
        d[R]/dt  = -kon*[D]*[R] + koff*[DR]

    Args:
        drug_name: Name of the drug.
        dose_conc_nM: Initial free drug concentration (nM).
        kon_per_nM_per_h: Association rate constant (1/nM/h).
        koff_per_h: Dissociation rate constant (1/h).
        receptor_conc_nM: Total receptor concentration (nM).
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        ReceptorBindingResult with time-course data.
    """
    if dose_conc_nM < 0:
        raise ValueError("dose_conc_nM must be non-negative")
    if kon_per_nM_per_h <= 0:
        raise ValueError("kon must be positive")
    if koff_per_h < 0:
        raise ValueError("koff must be non-negative")
    if receptor_conc_nM <= 0:
        raise ValueError("receptor_conc_nM must be positive")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be positive")
    if dt_h <= 0:
        raise ValueError("dt_h must be positive")

    kd_nM = calculate_kd(kon_per_nM_per_h, koff_per_h)
    mrt_h = 1.0 / koff_per_h if koff_per_h > 0 else float("inf")

    # Initial conditions
    D = dose_conc_nM
    R = receptor_conc_nM
    DR = 0.0

    times: list[float] = []
    drug_free: list[float] = []
    receptor_free: list[float] = []
    complex_vals: list[float] = []
    occupancy: list[float] = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1
    time_to_50pct = float("inf")
    half_occ_reached = False

    for _ in range(n_steps):
        # Record state
        times.append(round(t, 6))
        drug_free.append(max(0.0, D))
        receptor_free.append(max(0.0, R))
        complex_vals.append(max(0.0, DR))
        occ_pct = (DR / receptor_conc_nM) * 100.0 if receptor_conc_nM > 0 else 0.0
        occupancy.append(max(0.0, occ_pct))

        if not half_occ_reached and occ_pct >= 50.0:
            time_to_50pct = t
            half_occ_reached = True

        # Forward Euler step
        association = kon_per_nM_per_h * D * R
        dissociation = koff_per_h * DR

        dDR_dt = association - dissociation
        dD_dt = -association + dissociation
        dR_dt = -association + dissociation

        D = max(0.0, D + dD_dt * dt_h)
        R = max(0.0, R + dR_dt * dt_h)
        DR = max(0.0, DR + dDR_dt * dt_h)

        t += dt_h
        if t > t_end_h + dt_h * 0.5:
            break

    max_occ = max(occupancy) if occupancy else 0.0

    return ReceptorBindingResult(
        drug_name=drug_name,
        kd_nM=kd_nM,
        residence_time_h=mrt_h,
        times_h=times,
        drug_free_nM=drug_free,
        receptor_free_nM=receptor_free,
        complex_nM=complex_vals,
        occupancy_pct=occupancy,
        max_occupancy_pct=max_occ,
        time_to_50pct_occupancy_h=time_to_50pct,
        notes=f"Forward Euler ODE; Kd={kd_nM:.3f} nM; MRT={mrt_h:.2f} h",
    )


def simulate_with_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    kon_per_nM_per_h: float,
    koff_per_h: float,
    receptor_conc_nM: float = 10.0,
    mw_g_mol: float = 500.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> ReceptorBindingResult:
    """Simulate receptor binding coupled to one-compartment PK elimination.

    Drug is eliminated via CL/Vd from the central compartment while
    simultaneously binding to and dissociating from the receptor.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose (mg).
        cl_L_per_h: Systemic clearance (L/h).
        vd_L: Volume of distribution (L).
        kon_per_nM_per_h: Association rate constant (1/nM/h).
        koff_per_h: Dissociation rate constant (1/h).
        receptor_conc_nM: Total receptor concentration (nM).
        mw_g_mol: Molecular weight (g/mol) for dose unit conversion.
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        ReceptorBindingResult with time-course data.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_L_per_h < 0:
        raise ValueError("cl_L_per_h must be non-negative")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")
    if mw_g_mol <= 0:
        raise ValueError("mw_g_mol must be positive")
    if kon_per_nM_per_h <= 0:
        raise ValueError("kon must be positive")
    if koff_per_h < 0:
        raise ValueError("koff must be non-negative")
    if receptor_conc_nM <= 0:
        raise ValueError("receptor_conc_nM must be positive")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be positive")
    if dt_h <= 0:
        raise ValueError("dt_h must be positive")

    # dose_mg / mw_g_mol * 1e6 = nmol  (mg / (g/mol) * 1e6 nmol/mol * 1e-3 g/mg)
    dose_nmol = dose_mg * 1e6 / mw_g_mol  # nmol
    # initial concentration in nM = nmol / L
    initial_conc_nM = dose_nmol / vd_L

    kd_nM = calculate_kd(kon_per_nM_per_h, koff_per_h)
    mrt_h = 1.0 / koff_per_h if koff_per_h > 0 else float("inf")
    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    D = initial_conc_nM
    R = receptor_conc_nM
    DR = 0.0

    times: list[float] = []
    drug_free: list[float] = []
    receptor_free: list[float] = []
    complex_vals: list[float] = []
    occupancy: list[float] = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1
    time_to_50pct = float("inf")
    half_occ_reached = False

    for _ in range(n_steps):
        times.append(round(t, 6))
        drug_free.append(max(0.0, D))
        receptor_free.append(max(0.0, R))
        complex_vals.append(max(0.0, DR))
        occ_pct = min(100.0, (DR / receptor_conc_nM) * 100.0) if receptor_conc_nM > 0 else 0.0
        occupancy.append(max(0.0, occ_pct))

        if not half_occ_reached and occ_pct >= 50.0:
            time_to_50pct = t
            half_occ_reached = True

        # ODE with PK elimination of free drug
        association = kon_per_nM_per_h * D * R
        dissociation = koff_per_h * DR

        dD_dt = -association + dissociation - ke * D
        dR_dt = -association + dissociation
        dDR_dt = association - dissociation

        D = max(0.0, D + dD_dt * dt_h)
        R = max(0.0, R + dR_dt * dt_h)
        DR = max(0.0, DR + dDR_dt * dt_h)

        t += dt_h
        if t > t_end_h + dt_h * 0.5:
            break

    max_occ = max(occupancy) if occupancy else 0.0

    return ReceptorBindingResult(
        drug_name=drug_name,
        kd_nM=kd_nM,
        residence_time_h=mrt_h,
        times_h=times,
        drug_free_nM=drug_free,
        receptor_free_nM=receptor_free,
        complex_nM=complex_vals,
        occupancy_pct=occupancy,
        max_occupancy_pct=max_occ,
        time_to_50pct_occupancy_h=time_to_50pct,
        notes=(
            f"PK-coupled binding; ke={ke:.4f}/h; Kd={kd_nM:.3f} nM; "
            f"MRT={mrt_h:.2f} h; initial [D]={initial_conc_nM:.2f} nM"
        ),
    )
