"""
Phase 935 — Multi-tissue extravascular distribution PBPK model.

5-compartment Forward Euler ODE:
  Compartments: plasma, muscle, fat, liver, brain
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ExtravascularDistributionResult",
    "simulate_extravascular_distribution",
    "screen_tissue_distribution",
]

# Default physiology (typical human)
_V_PLASMA_L = 4.0

_TISSUES = {
    "muscle": {"Q_L_per_h": 75.0, "V_L": 35.0, "Kp_default": 0.5},
    "fat": {"Q_L_per_h": 5.0, "V_L": 10.0, "Kp_default": 1.0},
    "liver": {"Q_L_per_h": 90.0, "V_L": 1.5, "Kp_default": 5.0},
    "brain": {"Q_L_per_h": 50.0, "V_L": 1.5, "Kp_default": 0.1},
}


def _kp_from_logp(logp: float) -> dict:
    """Estimate tissue/plasma Kp values from logP."""
    kp_muscle = max(0.1, 0.2 * (10 ** (0.3 * logp)))
    kp_fat = max(0.5, 10 ** (0.6 * logp))
    kp_liver = max(1.0, 10 ** (0.4 * logp))
    kp_brain = max(0.01, 0.05 * (10 ** (0.2 * logp)))
    return {
        "muscle": kp_muscle,
        "fat": kp_fat,
        "liver": kp_liver,
        "brain": kp_brain,
    }


@dataclass
class ExtravascularDistributionResult:
    drug_name: str
    dose_mg: float
    logp: float
    times_h: list
    c_plasma_mg_L: list
    c_muscle_mg_L: list
    c_fat_mg_L: list
    c_liver_mg_L: list
    c_brain_mg_L: list
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    kp_muscle_used: float
    kp_fat_used: float
    kp_liver_used: float
    kp_brain_used: float
    brain_exposure_index: float
    liver_muscle_ratio: float
    notes: str


def _trapz(xs: list, ys: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(xs)):
        total += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return total


def simulate_extravascular_distribution(
    drug_name: str,
    dose_mg: float,
    logp: float = 1.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    cl_hepatic_L_per_h: float = 10.0,
    v_plasma_L: float = 4.0,
    kp_muscle: float | None = None,
    kp_fat: float | None = None,
    kp_liver: float | None = None,
    kp_brain: float | None = None,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> ExtravascularDistributionResult:
    """Simulate multi-tissue extravascular distribution via Forward Euler ODE."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError("cl_hepatic_L_per_h must be positive")
    if v_plasma_L <= 0:
        raise ValueError("v_plasma_L must be positive")
    if route not in {"iv", "oral"}:
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")

    # Resolve Kp values
    kp_est = _kp_from_logp(logp)
    kp_used = {
        "muscle": kp_muscle if kp_muscle is not None else kp_est["muscle"],
        "fat": kp_fat if kp_fat is not None else kp_est["fat"],
        "liver": kp_liver if kp_liver is not None else kp_est["liver"],
        "brain": kp_brain if kp_brain is not None else kp_est["brain"],
    }

    # Tissue parameters
    tissue_names = ["muscle", "fat", "liver", "brain"]
    Q = {t: _TISSUES[t]["Q_L_per_h"] for t in tissue_names}
    V_t = {t: _TISSUES[t]["V_L"] for t in tissue_names}
    Kp = kp_used

    # Rate constants per tissue
    # k_in_i  = Q_i / V_plasma   (delivery into tissue from plasma)
    # k_out_i = Q_i / (V_tissue * Kp_i)  (return from tissue to plasma)
    k_in = {t: Q[t] / v_plasma_L for t in tissue_names}
    k_out = {t: Q[t] / (V_t[t] * Kp[t]) for t in tissue_names}

    k_elim = cl_hepatic_L_per_h / v_plasma_L  # plasma elimination rate constant

    # Initial conditions
    # State: C_plasma (mg/L), A_muscle (mg), A_fat (mg), A_liver (mg), A_brain (mg)
    # + gut_dose (mg) for oral
    if route == "iv":
        C_plasma = dose_mg / v_plasma_L
        A_gut = 0.0
    else:  # oral
        C_plasma = 0.0
        A_gut = dose_mg

    A_tissue = {t: 0.0 for t in tissue_names}

    times = []
    c_plasma_list = []
    c_tissue = {t: [] for t in tissue_names}

    n_steps = max(1, int(round(t_end_h / dt_h)))

    for step in range(n_steps + 1):
        t = step * dt_h
        times.append(t)
        c_plasma_list.append(C_plasma)
        for t_name in tissue_names:
            c_tissue[t_name].append(A_tissue[t_name] / V_t[t_name])

        if step < n_steps:
            # dC_plasma/dt:
            # gains: sum(k_out_i * A_tissue_i / V_plasma)
            # losses: k_elim * C_plasma + sum(k_in_i * C_plasma)
            gain_from_tissues = sum(k_out[t] * A_tissue[t] / v_plasma_L for t in tissue_names)
            loss_to_tissues = sum(k_in[t] * C_plasma for t in tissue_names)

            # oral absorption input
            if route == "oral":
                absorption_rate = ka_per_h * A_gut  # mg/h
                dA_gut = -absorption_rate
            else:
                absorption_rate = 0.0
                dA_gut = 0.0

            dC_plasma = (
                -k_elim * C_plasma
                - loss_to_tissues
                + gain_from_tissues
                + absorption_rate / v_plasma_L
            )

            # dA_tissue_i/dt = k_in_i * C_plasma * V_plasma - k_out_i * A_tissue_i
            dA_tissue = {}
            for t_name in tissue_names:
                dA_tissue[t_name] = (
                    k_in[t_name] * C_plasma * v_plasma_L - k_out[t_name] * A_tissue[t_name]
                )

            # Euler update
            C_plasma = max(0.0, C_plasma + dC_plasma * dt_h)
            A_gut = max(0.0, A_gut + dA_gut * dt_h)
            for t_name in tissue_names:
                A_tissue[t_name] = max(0.0, A_tissue[t_name] + dA_tissue[t_name] * dt_h)

    # PK metrics
    cmax_plasma = max(c_plasma_list)
    tmax_plasma_h = times[c_plasma_list.index(cmax_plasma)]
    auc_plasma = _trapz(times, c_plasma_list)

    # Brain exposure index
    auc_brain = _trapz(times, c_tissue["brain"])
    brain_exposure_index = (auc_brain / auc_plasma) if auc_plasma > 0 else 0.0

    # Liver/muscle ratio
    cmax_liver = max(c_tissue["liver"]) if c_tissue["liver"] else 0.0
    cmax_muscle = max(c_tissue["muscle"]) if c_tissue["muscle"] else 0.0
    liver_muscle_ratio = (cmax_liver / cmax_muscle) if cmax_muscle > 0 else 0.0

    notes = (
        f"logP={logp:.2f}; route={route}; "
        f"Kp(muscle={kp_used['muscle']:.3f}, fat={kp_used['fat']:.3f}, "
        f"liver={kp_used['liver']:.3f}, brain={kp_used['brain']:.4f}); "
        f"CL_hepatic={cl_hepatic_L_per_h} L/h"
    )

    return ExtravascularDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_muscle_mg_L=c_tissue["muscle"],
        c_fat_mg_L=c_tissue["fat"],
        c_liver_mg_L=c_tissue["liver"],
        c_brain_mg_L=c_tissue["brain"],
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        kp_muscle_used=kp_used["muscle"],
        kp_fat_used=kp_used["fat"],
        kp_liver_used=kp_used["liver"],
        kp_brain_used=kp_used["brain"],
        brain_exposure_index=brain_exposure_index,
        liver_muscle_ratio=liver_muscle_ratio,
        notes=notes,
    )


def screen_tissue_distribution(
    drug_name: str,
    logp_values: list,
    dose_mg: float = 10.0,
    route: str = "iv",
    cl_hepatic_L_per_h: float = 10.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list:
    """Screen tissue distribution across multiple logP values.

    Returns list[ExtravascularDistributionResult] sorted by brain_exposure_index descending.
    """
    results = []
    for logp in logp_values:
        res = simulate_extravascular_distribution(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logp=logp,
            route=route,
            cl_hepatic_L_per_h=cl_hepatic_L_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(res)
    results.sort(key=lambda r: r.brain_exposure_index, reverse=True)
    return results
