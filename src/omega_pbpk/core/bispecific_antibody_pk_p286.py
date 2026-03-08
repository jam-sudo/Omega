"""Bispecific antibody (bsAb) PK model with dual target-mediated disposition.

Simulates PK of bispecific antibodies engaging two independent targets
simultaneously (e.g., tumor antigen + T-cell CD3). Uses quasi-steady-state
(QSS) TMDD approximation (Gibiansky 2012) to avoid ODE stiffness.

MW assumed 150 kDa for nM conversion (standard IgG-like bsAb).

References
----------
- Gibiansky L et al., J Pharmacokinet Pharmacodyn. 2012;39:5-16 (QSS TMDD)
- Krippendorff BF et al., J Pharmacokinet Pharmacodyn. 2009;36:239-60
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Molecular weight (kDa) — standard IgG-like bsAb
_MW_KDA = 150.0
# Conversion: 1 mg/L / MW_kDa * 1000 = nM
# 1 mg/L = 1 ug/mL; MW=150000 g/mol => 1 mg/L = 1000/150 nM
_MG_L_TO_NM = 1000.0 / _MW_KDA


def _mg_L_to_nM(c_mg_L: float) -> float:
    return c_mg_L * _MG_L_TO_NM


def _nM_to_mg_L(c_nM: float) -> float:
    return c_nM / _MG_L_TO_NM


@dataclass
class BispecificPKResult:
    """Results from bispecific antibody PK simulation.

    Attributes
    ----------
    drug_name : Drug identifier.
    dose_mg : Dose administered (mg).
    route : Administration route ('iv_bolus' or 'sc').
    times_h : Simulation time points (h).
    c_free_mg_L : Free antibody concentration in central (mg/L).
    c_bound1_mg_L : Target-1 bound complex concentration (mg/L).
    c_bound2_mg_L : Target-2 bound complex concentration (mg/L).
    auc_free_mg_h_per_L : AUC of free antibody (mg*h/L), trapezoidal.
    cmax_free_mg_L : Peak free antibody concentration (mg/L).
    tmax_free_h : Time of peak free concentration (h).
    receptor_occupancy_target1_pct : RO of target 1 at Cmax (%).
    receptor_occupancy_target2_pct : RO of target 2 at Cmax (%).
    notes : Summary note string.
    """

    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float] = field(default_factory=list)
    c_free_mg_L: list[float] = field(default_factory=list)
    c_bound1_mg_L: list[float] = field(default_factory=list)
    c_bound2_mg_L: list[float] = field(default_factory=list)
    auc_free_mg_h_per_L: float = 0.0
    cmax_free_mg_L: float = 0.0
    tmax_free_h: float = 0.0
    receptor_occupancy_target1_pct: float = 0.0
    receptor_occupancy_target2_pct: float = 0.0
    notes: str = ""


def _validate_inputs(
    dose_mg: float,
    route: str,
    cl_L_per_day: float,
    vd_L: float,
    target1_conc_nM: float,
    target2_conc_nM: float,
    kon1_per_nM_per_h: float,
    koff1_per_h: float,
    kint1_per_h: float,
    kon2_per_nM_per_h: float,
    koff2_per_h: float,
    kint2_per_h: float,
    t_end_h: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if route not in {"iv_bolus", "sc"}:
        raise ValueError(f"route must be 'iv_bolus' or 'sc', got {route!r}")
    if cl_L_per_day <= 0:
        raise ValueError(f"cl_L_per_day must be > 0, got {cl_L_per_day}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if target1_conc_nM < 0:
        raise ValueError(f"target1_conc_nM must be >= 0, got {target1_conc_nM}")
    if target2_conc_nM < 0:
        raise ValueError(f"target2_conc_nM must be >= 0, got {target2_conc_nM}")
    if kon1_per_nM_per_h <= 0:
        raise ValueError(f"kon1_per_nM_per_h must be > 0, got {kon1_per_nM_per_h}")
    if koff1_per_h <= 0:
        raise ValueError(f"koff1_per_h must be > 0, got {koff1_per_h}")
    if kint1_per_h <= 0:
        raise ValueError(f"kint1_per_h must be > 0, got {kint1_per_h}")
    if kon2_per_nM_per_h <= 0:
        raise ValueError(f"kon2_per_nM_per_h must be > 0, got {kon2_per_nM_per_h}")
    if koff2_per_h <= 0:
        raise ValueError(f"koff2_per_h must be > 0, got {koff2_per_h}")
    if kint2_per_h <= 0:
        raise ValueError(f"kint2_per_h must be > 0, got {kint2_per_h}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")


def simulate_bispecific_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_L_per_day: float,
    vd_L: float,
    target1_conc_nM: float,
    target2_conc_nM: float,
    kon1_per_nM_per_h: float,
    koff1_per_h: float,
    kint1_per_h: float,
    kon2_per_nM_per_h: float,
    koff2_per_h: float,
    kint2_per_h: float,
    t_end_h: float = 672.0,
    dt_h: float = 0.5,
) -> BispecificPKResult:
    """Simulate bispecific antibody PK with dual QSS-TMDD.

    Uses quasi-steady-state TMDD (Gibiansky approximation) for both targets.
    Forward Euler integration.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Dose (mg).
    route : 'iv_bolus' or 'sc'.
    cl_L_per_day : Linear clearance (L/day).
    vd_L : Central volume of distribution (L).
    target1_conc_nM : Total target 1 concentration (nM, assumed constant).
    target2_conc_nM : Total target 2 concentration (nM, assumed constant).
    kon1_per_nM_per_h : Target 1 association rate constant (1/(nM*h)).
    koff1_per_h : Target 1 dissociation rate constant (1/h).
    kint1_per_h : Target 1 complex internalization rate (1/h).
    kon2_per_nM_per_h : Target 2 association rate constant (1/(nM*h)).
    koff2_per_h : Target 2 dissociation rate constant (1/h).
    kint2_per_h : Target 2 complex internalization rate (1/h).
    t_end_h : Simulation end time (h). Default 672 h (28 days).
    dt_h : Forward Euler step size (h). Default 0.5 h.

    Returns
    -------
    BispecificPKResult with time course and derived PK metrics.
    """
    _validate_inputs(
        dose_mg,
        route,
        cl_L_per_day,
        vd_L,
        target1_conc_nM,
        target2_conc_nM,
        kon1_per_nM_per_h,
        koff1_per_h,
        kint1_per_h,
        kon2_per_nM_per_h,
        koff2_per_h,
        kint2_per_h,
        t_end_h,
    )

    # Convert CL to per-hour
    cl_h = cl_L_per_day / 24.0
    ke = cl_h / vd_L  # elimination rate constant (1/h)

    # QSS Michaelis-like constants (nM)
    # Ksm = (koff + kint) / kon  [nM]
    ksm1 = (koff1_per_h + kint1_per_h) / kon1_per_nM_per_h
    ksm2 = (koff2_per_h + kint2_per_h) / kon2_per_nM_per_h

    # SC parameters
    sc_f = 0.7
    ka = 0.02  # absorption rate constant (1/h)

    # Initial conditions
    if route == "iv_bolus":
        c_free = dose_mg / vd_L  # mg/L
        a_depot = 0.0
    else:
        c_free = 0.0
        a_depot = dose_mg * sc_f  # mg

    n_steps = max(1, int(t_end_h / dt_h))
    times_h: list[float] = []
    c_free_list: list[float] = []
    c_bound1_list: list[float] = []
    c_bound2_list: list[float] = []

    t = 0.0
    for _ in range(n_steps + 1):
        # Convert free concentration to nM for QSS bound computation
        c_nM = _mg_L_to_nM(c_free)

        # QSS bound fractions (nM)
        if c_nM > 0:
            rc1_nM = target1_conc_nM * c_nM / (ksm1 + c_nM)
            rc2_nM = target2_conc_nM * c_nM / (ksm2 + c_nM)
        else:
            rc1_nM = 0.0
            rc2_nM = 0.0

        times_h.append(t)
        c_free_list.append(c_free)
        c_bound1_list.append(_nM_to_mg_L(rc1_nM))
        c_bound2_list.append(_nM_to_mg_L(rc2_nM))

        if t >= t_end_h:
            break

        # Effective elimination rate: linear + TMDD contributions
        # TMDD effective kout contribution: kint * R_total / (Ksm + C_nM)
        # Units: (1/h) * nM / nM = 1/h  (dimensionless fraction saturated)
        if c_nM > 0:
            kout_tmdd1 = kint1_per_h * target1_conc_nM / (ksm1 + c_nM)
            kout_tmdd2 = kint2_per_h * target2_conc_nM / (ksm2 + c_nM)
        else:
            kout_tmdd1 = 0.0
            kout_tmdd2 = 0.0

        effective_kout = ke + kout_tmdd1 + kout_tmdd2

        # Input rate (mg/L/h)
        if route == "sc":
            input_rate_mg_h = ka * a_depot
            da_depot = -ka * a_depot * dt_h
        else:
            input_rate_mg_h = 0.0
            da_depot = 0.0

        input_rate_conc = input_rate_mg_h / vd_L  # mg/L/h

        # Forward Euler
        dc_free = (input_rate_conc - effective_kout * c_free) * dt_h
        c_free = max(0.0, c_free + dc_free)
        a_depot = max(0.0, a_depot + da_depot)

        t = min(t + dt_h, t_end_h)

    # AUC via manual trapezoidal rule
    auc = sum(
        0.5 * (c_free_list[i] + c_free_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    cmax = max(c_free_list) if c_free_list else 0.0
    tmax = times_h[c_free_list.index(cmax)] if c_free_list else 0.0

    # Receptor occupancy at Cmax
    c_nM_cmax = _mg_L_to_nM(cmax)
    if c_nM_cmax > 0 and target1_conc_nM > 0:
        rc1_at_cmax = target1_conc_nM * c_nM_cmax / (ksm1 + c_nM_cmax)
        ro1 = min(100.0, rc1_at_cmax / target1_conc_nM * 100.0)
    elif target1_conc_nM == 0:
        ro1 = 0.0
    else:
        ro1 = 0.0

    if c_nM_cmax > 0 and target2_conc_nM > 0:
        rc2_at_cmax = target2_conc_nM * c_nM_cmax / (ksm2 + c_nM_cmax)
        ro2 = min(100.0, rc2_at_cmax / target2_conc_nM * 100.0)
    elif target2_conc_nM == 0:
        ro2 = 0.0
    else:
        ro2 = 0.0

    notes = (
        f"{drug_name}: {route} {dose_mg}mg | "
        f"Ksm1={ksm1:.2f}nM Ksm2={ksm2:.2f}nM | "
        f"Cmax={cmax:.4f}mg/L tmax={tmax:.1f}h AUC={auc:.2f}mg*h/L | "
        f"RO1={ro1:.1f}% RO2={ro2:.1f}%"
    )

    return BispecificPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times_h,
        c_free_mg_L=c_free_list,
        c_bound1_mg_L=c_bound1_list,
        c_bound2_mg_L=c_bound2_list,
        auc_free_mg_h_per_L=auc,
        cmax_free_mg_L=cmax,
        tmax_free_h=tmax,
        receptor_occupancy_target1_pct=ro1,
        receptor_occupancy_target2_pct=ro2,
        notes=notes,
    )


def compare_dose_levels(
    drug_name: str,
    dose_list_mg: list[float],
    cl_L_per_day: float,
    vd_L: float,
    target1_conc_nM: float,
    target2_conc_nM: float,
    kon1_per_nM_per_h: float,
    koff1_per_h: float,
    kint1_per_h: float,
    kon2_per_nM_per_h: float,
    koff2_per_h: float,
    kint2_per_h: float,
    t_end_h: float = 672.0,
    dt_h: float = 0.5,
    route: str = "iv_bolus",
) -> list[BispecificPKResult]:
    """Compare bispecific antibody PK across multiple dose levels.

    Results are sorted descending by auc_free_mg_h_per_L (highest AUC first).

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_list_mg : List of doses to evaluate (mg).
    Other parameters : same as simulate_bispecific_pk.

    Returns
    -------
    List of BispecificPKResult sorted by auc_free_mg_h_per_L descending.
    """
    results = [
        simulate_bispecific_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            cl_L_per_day=cl_L_per_day,
            vd_L=vd_L,
            target1_conc_nM=target1_conc_nM,
            target2_conc_nM=target2_conc_nM,
            kon1_per_nM_per_h=kon1_per_nM_per_h,
            koff1_per_h=koff1_per_h,
            kint1_per_h=kint1_per_h,
            kon2_per_nM_per_h=kon2_per_nM_per_h,
            koff2_per_h=koff2_per_h,
            kint2_per_h=kint2_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        for dose_mg in dose_list_mg
    ]
    return sorted(results, key=lambda r: r.auc_free_mg_h_per_L, reverse=True)
