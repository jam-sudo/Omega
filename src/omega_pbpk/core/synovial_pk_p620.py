"""Phase 620 — Drug Synovial Fluid Distribution.

Models drug distribution into synovial fluid (joint fluid) for arthritis
drug assessment. Synovial fluid is filtered from plasma through synovial
membrane with size-selective permeability.

References
----------
- Wallis WJ et al., Arthritis Rheum. 1987 (synovial pharmacokinetics)
- Owen SG et al., Ann Rheum Dis. 1984 (synovial fluid drug levels)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SynovialPKResult:
    """Result of synovial fluid PK simulation (Phase 620)."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_synovial_mg_L: list
    synovial_plasma_ratio: list
    cmax_plasma_mg_L: float
    cmax_synovial_mg_L: float
    tmax_synovial_h: float
    auc_plasma_mg_h_per_L: float
    auc_synovial_mg_h_per_L: float
    synovial_auc_ratio: float
    lag_to_synovial_h: float
    efficacy_coverage_h: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC calculation."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_synovial_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    fu_plasma: float = 0.1,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> SynovialPKResult:
    """Simulate drug distribution into synovial fluid.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Oral dose in milligrams.
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons.
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    fu_plasma:
        Unbound fraction in plasma (0, 1]. Default 0.1.
    t_end_h:
        Simulation end time (h). Default 24 h.
    dt_h:
        Forward Euler step size (h). Default 0.05 h.

    Returns
    -------
    SynovialPKResult
        Simulation results including plasma and synovial concentration profiles.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")

    # --- Derived parameters ---
    Vd_syn_L = 0.5  # effective synovial distribution volume (multiple joints, L)

    # Permeability into synovial fluid: reduced for large MW (size exclusion)
    k_in_syn_raw = 0.3 / max(1.0, mw_Da / 500.0) * max(0.1, fu_plasma * 10.0)
    k_in_syn = max(0.01, min(1.0, k_in_syn_raw))
    k_out_syn = k_in_syn * 2.0  # drainage back to plasma

    # Oral absorption parameters
    ka = 1.2  # /h
    F = 0.85  # bioavailability factor
    ke = cl_sys_L_per_h / vd_sys_L  # elimination rate constant (/h)

    # --- Forward Euler simulation ---
    n_steps = int(t_end_h / dt_h) + 1
    times_h: list = []
    c_plasma_mg_L: list = []
    c_synovial_mg_L: list = []
    synovial_plasma_ratio: list = []

    # Initial conditions
    A_gut = dose_mg * F  # drug amount in gut (mg)
    C_plasma = 0.0  # plasma concentration (mg/L)
    C_syn = 0.0  # synovial concentration (mg/L)

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)
        c_plasma_mg_L.append(C_plasma)
        c_synovial_mg_L.append(C_syn)
        ratio = C_syn / max(1e-10, C_plasma)
        synovial_plasma_ratio.append(ratio)

        # ODE right-hand sides
        dA_gut = -ka * A_gut
        dC_plasma = (
            ka * A_gut / vd_sys_L
            - ke * C_plasma
            - k_in_syn * C_plasma
            + k_out_syn * C_syn * (Vd_syn_L / vd_sys_L)
        )
        dC_syn = k_in_syn * C_plasma * vd_sys_L / Vd_syn_L - k_out_syn * C_syn

        # Forward Euler update
        A_gut = max(0.0, A_gut + dt_h * dA_gut)
        C_plasma = max(0.0, C_plasma + dt_h * dC_plasma)
        C_syn = max(0.0, C_syn + dt_h * dC_syn)

    # --- Summary statistics ---
    cmax_plasma_mg_L = max(c_plasma_mg_L)
    cmax_synovial_mg_L = max(c_synovial_mg_L)

    # Time of maximum synovial concentration
    tmax_synovial_h = times_h[c_synovial_mg_L.index(cmax_synovial_mg_L)]

    auc_plasma = _trapezoidal_auc(times_h, c_plasma_mg_L)
    auc_synovial = _trapezoidal_auc(times_h, c_synovial_mg_L)

    synovial_auc_ratio = auc_synovial / auc_plasma if auc_plasma > 0 else 0.0

    # Lag time: first time C_syn >= 10% of Cmax_plasma
    threshold_lag = 0.1 * cmax_plasma_mg_L
    lag_to_synovial_h = t_end_h  # default if never reached
    for _i, (t, c) in enumerate(zip(times_h, c_synovial_mg_L)):
        if c >= threshold_lag:
            lag_to_synovial_h = t
            break

    # Efficacy coverage: total time C_syn > 10% of Cmax_plasma
    efficacy_coverage_h = sum(dt_h for c in c_synovial_mg_L if c > threshold_lag)

    notes = (
        f"k_in_syn={k_in_syn:.4f}/h; k_out_syn={k_out_syn:.4f}/h; "
        f"Vd_syn={Vd_syn_L:.2f} L; "
        f"lag={lag_to_synovial_h:.2f} h; "
        f"coverage={efficacy_coverage_h:.2f} h"
    )

    return SynovialPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        c_synovial_mg_L=c_synovial_mg_L,
        synovial_plasma_ratio=synovial_plasma_ratio,
        cmax_plasma_mg_L=cmax_plasma_mg_L,
        cmax_synovial_mg_L=cmax_synovial_mg_L,
        tmax_synovial_h=tmax_synovial_h,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_synovial_mg_h_per_L=auc_synovial,
        synovial_auc_ratio=synovial_auc_ratio,
        lag_to_synovial_h=lag_to_synovial_h,
        efficacy_coverage_h=efficacy_coverage_h,
        notes=notes,
    )
