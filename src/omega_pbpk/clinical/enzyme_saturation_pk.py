"""Phase 640 — Michaelis-Menten enzyme saturation kinetics for nonlinear PK.

Models drugs with saturable elimination (phenytoin, ethanol, warfarin at high doses)
where Cmax >> Km causes disproportionate increases in plasma concentration.
Uses Forward Euler ODE integration.
"""

from dataclasses import dataclass


@dataclass
class EnzymeSaturationResult:
    drug_name: str
    dose_mg: float
    vmax_mg_per_h: float
    km_mg_L: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_effective_h: float  # effective t½ at Cmax: Vd*(Km+Cmax)*(Km)/(Vmax*Cmax) simplified
    dose_proportionality_index: float  # auc/dose ratio vs reference; 1.0 = linear
    saturation_fraction: float  # Cmax/(Cmax+Km)
    elimination_mode: str  # "linear" (<20% saturation), "mixed", "saturable" (>80%)
    notes: str


def _validate_mm_params(
    dose_mg: float,
    vmax_mg_per_h: float,
    km_mg_L: float,
    vd_L: float,
    route: str,
    f_oral: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vmax_mg_per_h <= 0:
        raise ValueError(f"vmax_mg_per_h must be > 0, got {vmax_mg_per_h}")
    if km_mg_L <= 0:
        raise ValueError(f"km_mg_L must be > 0, got {km_mg_L}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")
    if not (0 < f_oral <= 1):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")


def simulate_mm_pk(
    drug_name: str,
    dose_mg: float,
    vmax_mg_per_h: float,
    km_mg_L: float,
    vd_L: float = 50.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
    reference_dose_mg: float | None = None,
) -> EnzymeSaturationResult:
    """Simulate Michaelis-Menten PK using Forward Euler ODE."""
    _validate_mm_params(dose_mg, vmax_mg_per_h, km_mg_L, vd_L, route, f_oral)

    # Initial conditions
    if route == "iv":
        c_plasma = dose_mg / vd_L
        a_gut = 0.0
    else:
        c_plasma = 0.0
        a_gut = dose_mg  # bioavailability applied during absorption

    times_h = []
    c_plasma_list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)

        if t >= t_end_h:
            break

        # MM elimination rate (mg/L/h)
        mm_rate = vmax_mg_per_h * c_plasma / (km_mg_L + c_plasma) / vd_L

        if route == "oral":
            # Gut absorption
            dA_gut = -ka_per_h * a_gut
            absorption_rate = ka_per_h * a_gut * f_oral / vd_L
            dC_plasma = absorption_rate - mm_rate
            a_gut = max(0.0, a_gut + dA_gut * dt_h)
        else:
            dC_plasma = -mm_rate

        c_plasma = max(0.0, c_plasma + dC_plasma * dt_h)
        t = round(t + dt_h, 10)

    # Summary statistics
    cmax_mg_L = max(c_plasma_list)
    tmax_h = times_h[c_plasma_list.index(cmax_mg_L)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(len(times_h) - 1):
        auc += (c_plasma_list[i] + c_plasma_list[i + 1]) / 2.0 * (times_h[i + 1] - times_h[i])

    # Effective t½ at Cmax: dC/dt = -Vmax*C/((Km+C)*Vd)
    # t½ = ln(2) / (Vmax / (Vd*(Km+Cmax)))
    if cmax_mg_L > 1e-12:
        ke_eff_at_cmax = vmax_mg_per_h / (vd_L * (km_mg_L + cmax_mg_L))
        t_half_effective_h = 0.693 / ke_eff_at_cmax if ke_eff_at_cmax > 0 else 0.0
    else:
        t_half_effective_h = 0.693 * vd_L * km_mg_L / vmax_mg_per_h  # linear approx

    # Saturation fraction
    saturation_fraction = cmax_mg_L / (cmax_mg_L + km_mg_L) if (cmax_mg_L + km_mg_L) > 0 else 0.0

    # Elimination mode
    if saturation_fraction < 0.2:
        elimination_mode = "linear"
    elif saturation_fraction > 0.8:
        elimination_mode = "saturable"
    else:
        elimination_mode = "mixed"

    # Dose proportionality index
    if reference_dose_mg is not None and reference_dose_mg > 0:
        ref_result = simulate_mm_pk(
            drug_name=drug_name,
            dose_mg=reference_dose_mg,
            vmax_mg_per_h=vmax_mg_per_h,
            km_mg_L=km_mg_L,
            vd_L=vd_L,
            route=route,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
            t_end_h=t_end_h,
            dt_h=dt_h,
            reference_dose_mg=None,
        )
        ref_auc = ref_result.auc_mg_h_per_L
        if ref_auc > 1e-12 and reference_dose_mg > 1e-12:
            dose_proportionality_index = (auc / dose_mg) / (ref_auc / reference_dose_mg)
        else:
            dose_proportionality_index = 1.0
    else:
        dose_proportionality_index = 1.0

    notes_parts = [
        f"Route: {route}",
        f"Saturation fraction at Cmax: {saturation_fraction:.2%}",
        f"Elimination mode: {elimination_mode}",
        f"Km={km_mg_L:.2f} mg/L, Vmax={vmax_mg_per_h:.2f} mg/h",
    ]
    if saturation_fraction > 0.8:
        notes_parts.append(
            "WARNING: Near-zero-order elimination — small dose increases may cause large Cmax rises"
        )

    return EnzymeSaturationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        vmax_mg_per_h=vmax_mg_per_h,
        km_mg_L=km_mg_L,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        t_half_effective_h=t_half_effective_h,
        dose_proportionality_index=dose_proportionality_index,
        saturation_fraction=saturation_fraction,
        elimination_mode=elimination_mode,
        notes="; ".join(notes_parts),
    )


def dose_escalation_mm(
    drug_name: str,
    doses_mg: list,
    vmax_mg_per_h: float,
    km_mg_L: float,
    vd_L: float = 50.0,
) -> list:
    """Simulate each dose and return list sorted by dose ascending."""
    sorted_doses = sorted(doses_mg)
    results = []
    for dose in sorted_doses:
        result = simulate_mm_pk(
            drug_name=drug_name,
            dose_mg=dose,
            vmax_mg_per_h=vmax_mg_per_h,
            km_mg_L=km_mg_L,
            vd_L=vd_L,
        )
        results.append(result)
    return results
