"""Phase 592 — Drug Crystallization Kinetics in GI Fluid."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrystallizationKineticsResult:
    """Result of a crystallization kinetics simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_dissolved_mg_mL: list
    c_crystalline_mg_mL: list
    supersaturation_ratio: list
    mass_fraction_crystalline: list
    induction_time_h: float
    t_90pct_crystallized_h: float
    peak_supersaturation: float
    auc_dissolved_mg_h_per_mL: float
    absorption_efficiency_pct: float
    notes: str


def _trapz(x: list, y: list) -> float:
    """Manual trapezoidal integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_crystallization_kinetics(
    drug_name: str,
    dose_mg: float,
    cs_mg_mL: float,
    ka_crystallization_per_h: float,
    volume_fluid_mL: float = 250.0,
    polymer_conc_mg_mL: float = 0.0,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> CrystallizationKineticsResult:
    """Simulate crystallization kinetics of a supersaturated drug in GI fluid.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        Total drug dose (mg).
    cs_mg_mL : float
        Equilibrium (crystalline) solubility (mg/mL).
    ka_crystallization_per_h : float
        Second-order crystallization rate constant (per h).
    volume_fluid_mL : float
        Volume of GI fluid (mL), default 250.
    polymer_conc_mg_mL : float
        Polymer (stabilizer) concentration (mg/mL); reduces crystallization rate.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration step (h).
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cs_mg_mL <= 0:
        raise ValueError("cs_mg_mL must be > 0")
    if ka_crystallization_per_h < 0:
        raise ValueError("ka_crystallization_per_h must be >= 0")
    if volume_fluid_mL <= 0:
        raise ValueError("volume_fluid_mL must be > 0")

    # Effective crystallization rate (polymer inhibition)
    ka_eff = ka_crystallization_per_h / (1.0 + polymer_conc_mg_mL * 0.5)

    n = max(int(round(t_end_h / dt_h)), 1)
    times = [i * dt_h for i in range(n + 1)]

    # Initial dissolved concentration (cap at 20× solubility)
    c_diss_init = dose_mg / volume_fluid_mL
    if c_diss_init > cs_mg_mL * 20.0:
        c_diss_init = cs_mg_mL * 20.0

    c_dissolved = [0.0] * (n + 1)
    c_crystalline = [0.0] * (n + 1)
    c_dissolved[0] = c_diss_init
    c_crystalline[0] = 0.0

    for i in range(n):
        cd = c_dissolved[i]
        cc = c_crystalline[i]

        if cd > cs_mg_mL:
            excess = cd - cs_mg_mL
            dc_cryst_dt = ka_eff * (excess**2) / cs_mg_mL
        else:
            dc_cryst_dt = 0.0

        dc = dc_cryst_dt * dt_h

        # Prevent dissolved going below equilibrium solubility
        new_cd = cd - dc
        if new_cd < cs_mg_mL and cd > cs_mg_mL:
            # Crystallize only the excess
            dc = cd - cs_mg_mL
            new_cd = cs_mg_mL

        c_dissolved[i + 1] = max(new_cd, 0.0)
        c_crystalline[i + 1] = max(cc + dc, 0.0)

    # Derived quantities
    supersaturation_ratio = [c_dissolved[i] / cs_mg_mL for i in range(n + 1)]

    mass_fraction_crystalline = []
    for i in range(n + 1):
        total = c_dissolved[i] + c_crystalline[i]
        if total > 0:
            mass_fraction_crystalline.append(c_crystalline[i] / total)
        else:
            mass_fraction_crystalline.append(0.0)

    # Induction time: first index where mass_fraction_crystalline >= 0.01
    induction_time_h = t_end_h  # default
    for i in range(n + 1):
        if mass_fraction_crystalline[i] >= 0.01:
            induction_time_h = times[i]
            break

    # t_90pct: first index where mass_fraction_crystalline >= 0.90
    t_90pct = t_end_h
    for i in range(n + 1):
        if mass_fraction_crystalline[i] >= 0.90:
            t_90pct = times[i]
            break

    peak_supersaturation = max(supersaturation_ratio)
    auc_dissolved = _trapz(times, c_dissolved)

    # Absorption efficiency: fraction dissolved at t=2h (or t_end if shorter)
    ref_time = min(2.0, t_end_h)
    ref_idx = min(int(round(ref_time / dt_h)), n)
    abs_efficiency = (c_dissolved[ref_idx] * volume_fluid_mL / dose_mg) * 100.0

    notes = (
        f"Polymer conc: {polymer_conc_mg_mL:.2f} mg/mL, "
        f"ka_eff: {ka_eff:.4f}/h. "
        f"Peak SR: {peak_supersaturation:.2f}. "
        f"Induction time: {induction_time_h:.3f} h."
    )

    return CrystallizationKineticsResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_dissolved_mg_mL=c_dissolved,
        c_crystalline_mg_mL=c_crystalline,
        supersaturation_ratio=supersaturation_ratio,
        mass_fraction_crystalline=mass_fraction_crystalline,
        induction_time_h=induction_time_h,
        t_90pct_crystallized_h=t_90pct,
        peak_supersaturation=peak_supersaturation,
        auc_dissolved_mg_h_per_mL=auc_dissolved,
        absorption_efficiency_pct=abs_efficiency,
        notes=notes,
    )


__all__ = [
    "CrystallizationKineticsResult",
    "simulate_crystallization_kinetics",
]
