"""Drug toxicokinetics model (Phase 569).

Models toxicokinetics of drugs/toxins: time above NOAEL, LOAEL,
and LD50-related thresholds using a 1-compartment oral PK model.
"""

from dataclasses import dataclass


@dataclass
class ToxicokineticsResult:
    """Result of a toxicokinetics simulation."""

    substance_name: str
    dose_mg_per_kg: float
    weight_kg: float
    times_h: list
    c_systemic_mg_L: list
    noael_mg_L: float
    loael_mg_L: float
    tnoael_h: float
    tloael_h: float
    cmax_mg_L: float
    auc_mg_h_per_L: float
    margin_of_safety: float
    toxicity_concern: str
    notes: str


def simulate_toxicokinetics(
    substance_name: str,
    dose_mg_per_kg: float,
    weight_kg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    noael_mg_L: float = 0.1,
    loael_mg_L: float = 0.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> ToxicokineticsResult:
    """Simulate toxicokinetics for a substance.

    Uses a 1-compartment oral PK model with Forward Euler integration.
    """
    if dose_mg_per_kg <= 0:
        raise ValueError("dose_mg_per_kg must be > 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if noael_mg_L <= 0:
        raise ValueError("noael_mg_L must be > 0")
    if loael_mg_L <= noael_mg_L:
        raise ValueError("loael_mg_L must be > noael_mg_L")

    dose_mg = dose_mg_per_kg * weight_kg
    a_gi = dose_mg * f_oral
    c_sys = 0.0

    times_h = []
    c_systemic_mg_L = []

    t = 0.0
    n_steps = int(t_end_h / dt_h)

    for _ in range(n_steps + 1):
        times_h.append(t)
        c_systemic_mg_L.append(c_sys)

        da_gi = -ka_per_h * a_gi
        dc_sys = ka_per_h * a_gi / vd_L - (cl_L_per_h / vd_L) * c_sys

        a_gi += da_gi * dt_h
        c_sys += dc_sys * dt_h
        t += dt_h

    # Cmax
    cmax = max(c_systemic_mg_L)

    # Time above thresholds
    tnoael_h = sum(1 for c in c_systemic_mg_L if c > noael_mg_L) * dt_h
    tloael_h = sum(1 for c in c_systemic_mg_L if c > loael_mg_L) * dt_h

    # AUC via trapezoidal rule
    auc = sum(
        0.5 * (c_systemic_mg_L[i] + c_systemic_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Margin of safety
    margin_of_safety = noael_mg_L / cmax if cmax > 0 else float("inf")

    # Toxicity concern classification
    if cmax < noael_mg_L:
        toxicity_concern = "none"
    elif cmax < loael_mg_L:
        toxicity_concern = "possible"
    elif cmax < 2 * loael_mg_L:
        toxicity_concern = "likely"
    else:
        toxicity_concern = "severe"

    return ToxicokineticsResult(
        substance_name=substance_name,
        dose_mg_per_kg=dose_mg_per_kg,
        weight_kg=weight_kg,
        times_h=times_h,
        c_systemic_mg_L=c_systemic_mg_L,
        noael_mg_L=noael_mg_L,
        loael_mg_L=loael_mg_L,
        tnoael_h=tnoael_h,
        tloael_h=tloael_h,
        cmax_mg_L=cmax,
        auc_mg_h_per_L=auc,
        margin_of_safety=margin_of_safety,
        toxicity_concern=toxicity_concern,
        notes=f"1-cpt oral PK, ka={ka_per_h}, F={f_oral}, dt={dt_h}h",
    )


def screen_doses(
    substance_name: str,
    weight_kg: float,
    cl_L_per_h: float,
    vd_L: float,
    noael_mg_L: float,
    loael_mg_L: float,
    dose_mg_per_kg_values: list,
) -> list:
    """Screen multiple doses and return results sorted by dose ascending."""
    results = [
        simulate_toxicokinetics(
            substance_name=substance_name,
            dose_mg_per_kg=d,
            weight_kg=weight_kg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            noael_mg_L=noael_mg_L,
            loael_mg_L=loael_mg_L,
        )
        for d in dose_mg_per_kg_values
    ]
    results.sort(key=lambda r: r.dose_mg_per_kg)
    return results
