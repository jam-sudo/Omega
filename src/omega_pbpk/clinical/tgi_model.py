"""Simeoni tumor growth inhibition (TGI) model for oncology PK/PD.

Reference: Simeoni M et al. (2004) Predictive pharmacokinetic-pharmacodynamic
modeling of tumor growth kinetics in xenograft models after administration of
anticancer agents. Cancer Research 64(3):1094-1101.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TGIResult:
    """Result of Simeoni TGI model simulation."""

    drug_name: str
    dose_mg: float
    n_doses: int
    route: str
    times_h: list[float]
    c_plasma_mg_L: list[float]  # plasma concentration (mg/L)
    tumor_mass_g: list[float]  # total tumor mass X_total = X1+X2+X3+X4 (g)
    tumor_mass_initial: float
    tumor_mass_final: float
    min_tumor_mass_g: float
    time_to_min_h: float
    tumor_growth_inhibition_pct: float  # TGI% vs vehicle control (0 if no control)
    regression: bool  # True if tumor_mass_final < tumor_mass_initial
    notes: str


def _simulate_pk(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str,
    ka_per_h: float,
    n_doses: int,
    dose_interval_h: float,
    times: np.ndarray,
    dt_h: float,
) -> np.ndarray:
    """Simulate plasma concentration profile using 1-cpt forward Euler."""
    n_steps = len(times) - 1
    ke = cl_L_per_h / vd_L
    c_plasma = np.zeros(len(times))
    is_oral = route.lower() == "oral"

    a_central = 0.0
    a_gut = 0.0

    # Schedule dose times
    dose_times = [i * dose_interval_h for i in range(n_doses)]
    next_dose_idx = 0

    # Initial dose at t=0
    if next_dose_idx < len(dose_times) and dose_times[next_dose_idx] == 0.0:
        if is_oral:
            a_gut += dose_mg
        else:
            a_central += dose_mg
        next_dose_idx += 1

    if not is_oral:
        c_plasma[0] = a_central / vd_L

    for i in range(n_steps):
        t_now = times[i]

        # Administer subsequent doses
        if next_dose_idx < len(dose_times):
            if t_now >= dose_times[next_dose_idx] - dt_h / 2:
                if is_oral:
                    a_gut += dose_mg
                else:
                    a_central += dose_mg
                next_dose_idx += 1

        if is_oral and ka_per_h > 0:
            absorbed = ka_per_h * a_gut * dt_h
            a_gut = max(a_gut - absorbed, 0.0)
            a_central += absorbed

        a_central = max(a_central - ke * a_central * dt_h, 0.0)
        c_plasma[i + 1] = a_central / vd_L

    return c_plasma


def simulate_tgi(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    k2_per_mg_per_L_per_h: float,
    lambda0_per_h: float = 0.005,
    lambda1_g_per_h: float = 0.05,
    k1_per_h: float = 0.1,
    psi: float = 20.0,
    x0_g: float = 0.2,
    route: str = "iv",
    ka_per_h: float = 1.0,
    n_doses: int = 1,
    dose_interval_h: float = 168.0,
    t_end_h: float = 672.0,
    dt_h: float = 0.5,
    _control_final: float | None = None,
) -> TGIResult:
    """Simulate tumor growth inhibition using the Simeoni (2004) model.

    The Simeoni model uses a 4-state damage cascade:
    - X1: proliferating tumor cells (g)
    - X2, X3, X4: sequentially damaged cells (damaged cells transit to death)
    - X_total = X1 + X2 + X3 + X4

    Tumor growth ODE:
      dX1/dt = [λ0*X1 / (1 + (λ0/λ1 * X_total)^ψ)^(1/ψ)] - k2*C*X1

    Damage cascade:
      dX2/dt = k2*C*X1 - k1*X2
      dX3/dt = k1*X2  - k1*X3
      dX4/dt = k1*X3  - k1*X4

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Dose per administration (mg).
    cl_L_per_h : float
        Clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    k2_per_mg_per_L_per_h : float
        Drug kill rate constant (L/mg/h). Set to 0 for vehicle control.
    lambda0_per_h : float
        Exponential tumor growth rate (1/h).
    lambda1_g_per_h : float
        Linear growth rate limit (g/h).
    k1_per_h : float
        Transit rate constant through damage cascade (1/h).
    psi : float
        Shape parameter for growth rate switch (~20 gives sharp transition).
    x0_g : float
        Initial tumor mass (g).
    route : str
        "iv" or "oral".
    ka_per_h : float
        Oral absorption rate constant (1/h). Ignored for IV.
    n_doses : int
        Number of doses to administer.
    dose_interval_h : float
        Time between doses (h).
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Time step (h).
    _control_final : float | None
        Internal: vehicle control final tumor mass for TGI% calculation.

    Returns
    -------
    TGIResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if k2_per_mg_per_L_per_h < 0:
        raise ValueError("k2_per_mg_per_L_per_h must be >= 0")
    if lambda0_per_h <= 0:
        raise ValueError("lambda0_per_h must be > 0")
    if lambda1_g_per_h <= 0:
        raise ValueError("lambda1_g_per_h must be > 0")
    if x0_g <= 0:
        raise ValueError("x0_g must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")

    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    # Simulate PK
    c_plasma = _simulate_pk(
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        route=route,
        ka_per_h=ka_per_h,
        n_doses=n_doses,
        dose_interval_h=dose_interval_h,
        times=times,
        dt_h=dt_h,
    )

    # PD simulation: Simeoni tumor model
    x1 = np.zeros(n_steps + 1)
    x2 = np.zeros(n_steps + 1)
    x3 = np.zeros(n_steps + 1)
    x4 = np.zeros(n_steps + 1)

    x1[0] = x0_g
    x2[0] = 0.0
    x3[0] = 0.0
    x4[0] = 0.0

    for i in range(n_steps):
        x_total = x1[i] + x2[i] + x3[i] + x4[i]
        c = c_plasma[i]

        # Simeoni growth term: logistic-like switch from exponential to linear
        ratio = lambda0_per_h / lambda1_g_per_h * x_total
        denom = (1.0 + ratio**psi) ** (1.0 / psi)
        growth_rate = lambda0_per_h * x1[i] / denom

        # Drug kill
        kill_rate = k2_per_mg_per_L_per_h * c * x1[i]

        dx1 = growth_rate - kill_rate
        dx2 = kill_rate - k1_per_h * x2[i]
        dx3 = k1_per_h * x2[i] - k1_per_h * x3[i]
        dx4 = k1_per_h * x3[i] - k1_per_h * x4[i]

        x1[i + 1] = max(x1[i] + dx1 * dt_h, 0.0)
        x2[i + 1] = max(x2[i] + dx2 * dt_h, 0.0)
        x3[i + 1] = max(x3[i] + dx3 * dt_h, 0.0)
        x4[i + 1] = max(x4[i] + dx4 * dt_h, 0.0)

    tumor_total = x1 + x2 + x3 + x4

    tumor_initial = float(tumor_total[0])
    tumor_final = float(tumor_total[-1])
    min_tumor = float(np.min(tumor_total))
    time_to_min = float(times[int(np.argmin(tumor_total))])

    regression = tumor_final < tumor_initial

    # TGI%: if control final provided, compute; else 0
    if _control_final is not None and _control_final > 0:
        tgi_pct = (1.0 - tumor_final / _control_final) * 100.0
    else:
        tgi_pct = 0.0

    notes = (
        f"Simeoni TGI model for {drug_name}. "
        f"Dose: {dose_mg} mg x {n_doses} ({route}). "
        f"Regression: {regression}. "
        f"Min tumor: {min_tumor:.3f} g at {time_to_min:.1f} h."
    )

    return TGIResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        n_doses=n_doses,
        route=route,
        times_h=times.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        tumor_mass_g=tumor_total.tolist(),
        tumor_mass_initial=tumor_initial,
        tumor_mass_final=tumor_final,
        min_tumor_mass_g=min_tumor,
        time_to_min_h=time_to_min,
        tumor_growth_inhibition_pct=tgi_pct,
        regression=regression,
        notes=notes,
    )


def tgi_dose_response(
    drug_name: str,
    doses_mg: list[float],
    cl_L_per_h: float,
    vd_L: float,
    k2: float,
    **kwargs,
) -> list[TGIResult]:
    """Compute TGI dose-response: simulate each dose, sort by min_tumor_mass ascending.

    Also computes vehicle control (dose_mg used as IV bolus with k2=0) to derive TGI%.

    Parameters
    ----------
    drug_name : str
        Drug name.
    doses_mg : list[float]
        List of doses to evaluate (mg).
    cl_L_per_h : float
        Clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    k2 : float
        Drug kill rate constant (L/mg/h).
    **kwargs
        Additional arguments passed to simulate_tgi.

    Returns
    -------
    list[TGIResult]
        Sorted by min_tumor_mass_g ascending (most efficacious first).
    """
    # Vehicle control: use first dose but k2=0 to get untreated growth
    first_dose = doses_mg[0] if doses_mg else 1.0
    control = simulate_tgi(
        drug_name=drug_name,
        dose_mg=first_dose,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        k2_per_mg_per_L_per_h=0.0,
        **kwargs,
    )
    control_final = control.tumor_mass_final

    results = []
    for dose in doses_mg:
        r = simulate_tgi(
            drug_name=drug_name,
            dose_mg=dose,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            k2_per_mg_per_L_per_h=k2,
            _control_final=control_final,
            **kwargs,
        )
        results.append(r)

    return sorted(results, key=lambda r: r.min_tumor_mass_g)


__all__ = [
    "TGIResult",
    "simulate_tgi",
    "tgi_dose_response",
]
