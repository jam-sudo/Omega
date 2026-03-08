"""Phase 419 — Drug PK in Bone Marrow.

Simulates drug distribution into bone marrow using a 2-compartment model
(plasma + bone marrow). Relevant for chemotherapy, antibiotics, and stem
cell transplant drugs targeting marrow.

Bone marrow receives ~5% of cardiac output (~0.3 L/h).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoneMarrowPKResult:
    """Result of bone marrow PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list
    c_plasma_mg_L: list
    c_marrow_mg_L: list
    cmax_plasma_mg_L: float
    cmax_marrow_mg_L: float
    tmax_marrow_h: float
    auc_plasma_mg_h_per_L: float
    auc_marrow_mg_h_per_L: float
    marrow_to_plasma_auc_ratio: float
    t_half_marrow_h: float
    marrow_penetration_pct: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC calculation."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_bone_marrow_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 5.0,
    vd_marrow_L: float = 1.5,
    q_marrow_L_per_h: float = 0.3,
    clint_marrow_L_per_h: float = 0.0,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> BoneMarrowPKResult:
    """Simulate drug PK in bone marrow.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    route:
        Administration route: 'iv' or 'oral'.
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_plasma_L:
        Volume of distribution of plasma/central compartment (L).
    vd_marrow_L:
        Volume of distribution of bone marrow compartment (L).
    q_marrow_L_per_h:
        Blood flow to bone marrow (L/h).
    clint_marrow_L_per_h:
        Intrinsic clearance in bone marrow (L/h). Default 0.
    ka_per_h:
        Oral absorption rate constant (1/h).
    f_oral:
        Oral bioavailability fraction (0 < f_oral <= 1).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step for forward Euler (h).

    Returns
    -------
    BoneMarrowPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if vd_marrow_L <= 0:
        raise ValueError(f"vd_marrow_L must be > 0, got {vd_marrow_L}")
    if q_marrow_L_per_h < 0:
        raise ValueError(f"q_marrow_L_per_h must be >= 0, got {q_marrow_L_per_h}")
    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")
    if route == "oral":
        if not (0 < f_oral <= 1):
            raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")

    # Transfer rate constants
    k12 = q_marrow_L_per_h / vd_plasma_L  # plasma -> marrow (1/h)
    k21 = q_marrow_L_per_h / vd_marrow_L  # marrow -> plasma (1/h)
    k_el_plasma = cl_sys_L_per_h / vd_plasma_L  # elimination from plasma (1/h)
    k_el_marrow = clint_marrow_L_per_h / vd_marrow_L  # local marrow elimination (1/h)

    # Initial conditions
    if route == "iv":
        c_plasma = dose_mg / vd_plasma_L
        c_marrow = 0.0
        gut_mg = 0.0
    else:
        c_plasma = 0.0
        c_marrow = 0.0
        gut_mg = dose_mg * f_oral

    # Time vector
    n_steps = int(t_end_h / dt_h) + 1
    times_h: list = []
    c_plasma_list: list = []
    c_marrow_list: list = []

    t = 0.0
    for _ in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_marrow_list.append(c_marrow)

        # Absorption input rate (mg/h into plasma) for oral
        if route == "oral":
            absorption_rate = ka_per_h * gut_mg  # mg/h
            input_rate = absorption_rate / vd_plasma_L  # mg/L/h
        else:
            absorption_rate = 0.0
            input_rate = 0.0

        # ODEs (Forward Euler)
        # dC_plasma/dt = input - k_el_plasma*C_plasma - k12*C_plasma
        #                + k21*C_marrow*(Vd_marrow/Vd_plasma)
        dc_plasma = (
            input_rate
            - k_el_plasma * c_plasma
            - k12 * c_plasma
            + k21 * c_marrow * (vd_marrow_L / vd_plasma_L)
        )

        # dC_marrow/dt = k12*(Vd_plasma/Vd_marrow)*C_plasma - k21*C_marrow
        #                - k_el_marrow*C_marrow
        dc_marrow = (
            k12 * (vd_plasma_L / vd_marrow_L) * c_plasma - k21 * c_marrow - k_el_marrow * c_marrow
        )

        # Update states
        c_plasma = max(c_plasma + dc_plasma * dt_h, 0.0)
        c_marrow = max(c_marrow + dc_marrow * dt_h, 0.0)

        if route == "oral":
            gut_mg = max(gut_mg - absorption_rate * dt_h, 0.0)

        t += dt_h

    # Derived metrics
    cmax_plasma = max(c_plasma_list)
    cmax_marrow = max(c_marrow_list)

    # tmax_marrow: time at which marrow concentration peaks
    idx_tmax_marrow = c_marrow_list.index(cmax_marrow)
    tmax_marrow = times_h[idx_tmax_marrow]

    auc_plasma = _trapezoidal_auc(times_h, c_plasma_list)
    auc_marrow = _trapezoidal_auc(times_h, c_marrow_list)

    marrow_to_plasma_auc_ratio = auc_marrow / max(auc_plasma, 1e-12)
    marrow_penetration_pct = cmax_marrow / max(cmax_plasma, 1e-12) * 100.0

    # t_half_marrow: effective half-life for marrow elimination
    effective_k_out = k21 + k_el_marrow
    if effective_k_out > 0:
        t_half_marrow = 0.693 / effective_k_out
    else:
        t_half_marrow = float("inf")

    notes = (
        f"Route: {route}. "
        f"Marrow penetration: {marrow_penetration_pct:.1f}% (Cmax ratio). "
        f"AUC marrow/plasma ratio: {marrow_to_plasma_auc_ratio:.3f}. "
        f"Q_marrow = {q_marrow_L_per_h:.2f} L/h."
    )

    return BoneMarrowPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_marrow_mg_L=c_marrow_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_marrow_mg_L=cmax_marrow,
        tmax_marrow_h=tmax_marrow,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_marrow_mg_h_per_L=auc_marrow,
        marrow_to_plasma_auc_ratio=marrow_to_plasma_auc_ratio,
        t_half_marrow_h=t_half_marrow,
        marrow_penetration_pct=marrow_penetration_pct,
        notes=notes,
    )


def compare_marrow_penetration(
    drug_name: str,
    dose_mg: float,
    penetration_coefficients: list,
    cl_sys_L_per_h: float,
    vd_plasma_L: float,
) -> list:
    """Compare bone marrow penetration across different blood flow values.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Dose in milligrams.
    penetration_coefficients:
        List of q_marrow_L_per_h values to compare.
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_plasma_L:
        Volume of distribution of plasma compartment (L).

    Returns
    -------
    List of BoneMarrowPKResult sorted by marrow_to_plasma_auc_ratio descending.
    """
    results = []
    for q_val in penetration_coefficients:
        result = simulate_bone_marrow_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route="iv",
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_plasma_L=vd_plasma_L,
            q_marrow_L_per_h=q_val,
        )
        results.append(result)

    results.sort(key=lambda r: r.marrow_to_plasma_auc_ratio, reverse=True)
    return results
