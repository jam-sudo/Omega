"""Cardiac output-dependent drug distribution (Phase 252).

Models how changes in cardiac output (heart failure, exercise, critical illness)
affect drug distribution and clearance. Based on Patterson 1997, Greil 2004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CardiacPKResult:
    """Result of a cardiac output-dependent PK simulation."""

    drug_name: str
    dose_mg: float
    cardiac_output_fraction: float
    route: str  # "iv" or "oral"
    cl_effective_L_per_h: float
    vd_effective_L: float

    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)

    cmax: float = 0.0
    tmax_h: float = 0.0
    auc: float = 0.0
    t_half_h: float = 0.0
    notes: str = ""


@dataclass
class HFDoseResult:
    """Result of heart failure dose adjustment."""

    drug_name: str
    normal_dose_mg: float
    nyha_class: int
    cardiac_output_fraction: float
    cl_effective_L_per_h: float
    adjusted_dose_mg: float
    dose_reduction_pct: float
    t_half_h: float
    rationale: str


def scale_organ_flows(cardiac_output_fraction: float) -> dict[str, float]:
    """Scale organ blood flows based on cardiac output fraction.

    In heart failure, hepatic and renal flows are partially preserved
    through autoregulation, while muscle and splanchnic flows are
    preferentially reduced.

    In exercise, all flows increase proportionally to cardiac output.

    Parameters
    ----------
    cardiac_output_fraction:
        Relative cardiac output (1.0 = normal, 0.3 = severe HF, 2.0 = exercise).

    Returns
    -------
    dict with scaled flow fractions for: hepatic, renal, muscle, fat,
    brain, splanchnic. Values represent fraction of normal flow.
    """
    if cardiac_output_fraction <= 0 or cardiac_output_fraction > 3:
        raise ValueError("cardiac_output_fraction must be in (0, 3]")

    co = cardiac_output_fraction

    if co <= 1.0:
        # Heart failure or reduced CO: autoregulation preserves hepatic and renal
        # Muscle and splanchnic are preferentially reduced
        hepatic = 0.8 * co + 0.2  # More preserved: 0.2 at CO=0, 1.0 at CO=1
        renal = 0.9 * co + 0.1  # Most preserved: 0.1 at CO=0, 1.0 at CO=1
        muscle = 0.4 * co  # Most reduced in HF
        brain = 0.95 * co + 0.05  # Nearly fully preserved
        fat = 0.6 * co + 0.1  # Moderately preserved
        splanchnic = 0.5 * co + 0.1  # Moderately reduced in HF
    else:
        # Exercise or elevated CO: all flows increase proportionally
        hepatic = co * 0.9 + 0.1  # Slight dampening (splanchnic steal)
        renal = co  # Proportional
        muscle = co * 1.3  # Preferentially increased during exercise
        brain = co * 0.7 + 0.3  # Modest increase (autoregulation)
        fat = co * 0.5 + 0.5  # Modest increase
        splanchnic = co * 0.6 + 0.4  # Moderate increase

    return {
        "hepatic": hepatic,
        "renal": renal,
        "muscle": muscle,
        "fat": fat,
        "brain": brain,
        "splanchnic": splanchnic,
    }


def simulate_cardiac_pk(
    drug_name: str,
    dose_mg: float,
    cl_normal_L_per_h: float,
    vd_L: float,
    cardiac_output_fraction: float = 0.5,
    route: str = "iv",
    ka_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> CardiacPKResult:
    """Simulate PK with altered cardiac output.

    Scales CL using the well-stirred hepatic extraction model and accounts
    for increased Vd due to edema in heart failure.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose administered (mg).
    cl_normal_L_per_h:
        Normal systemic clearance (L/h).
    vd_L:
        Normal volume of distribution (L).
    cardiac_output_fraction:
        Relative cardiac output (1.0 = normal, 0.3 = severe HF, 2.0 = exercise).
    route:
        "iv" or "oral".
    ka_per_h:
        First-order absorption rate constant for oral dosing (h⁻¹).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration time step (h).

    Returns
    -------
    CardiacPKResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_normal_L_per_h <= 0:
        raise ValueError("cl_normal_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cardiac_output_fraction <= 0 or cardiac_output_fraction > 3:
        raise ValueError("cardiac_output_fraction must be in (0, 3]")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")

    # Scale CL based on hepatic flow (well-stirred model approximation)
    # CL_effective = CL_normal * CO_fraction * 0.8 (conservative; hepatic preserved)
    cl_effective = cl_normal_L_per_h * cardiac_output_fraction * 0.8
    # Apply minimum: CL can't be less than 5% of normal
    cl_effective = max(cl_normal_L_per_h * 0.05, cl_effective)

    # Vd increases in HF due to edema and third spacing
    # Vd_effective = Vd_normal * (1 + (1 - CO_fraction) * 0.3)
    if cardiac_output_fraction < 1.0:
        vd_effective = vd_L * (1.0 + (1.0 - cardiac_output_fraction) * 0.3)
    else:
        # Exercise: slight decrease in Vd (increased perfusion)
        vd_effective = vd_L

    ke = cl_effective / vd_effective  # Elimination rate constant

    # Simulation (forward Euler)
    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]
    c_plasma = [0.0] * n_steps

    if route == "iv":
        c_plasma[0] = dose_mg / vd_effective
        for i in range(1, n_steps):
            cp = c_plasma[i - 1]
            dc = -ke * cp * dt_h
            c_plasma[i] = max(0.0, cp + dc)
    else:
        # Oral: absorption compartment
        m_gut = [0.0] * n_steps
        m_gut[0] = dose_mg
        c_plasma[0] = 0.0
        for i in range(1, n_steps):
            mg = m_gut[i - 1]
            cp = c_plasma[i - 1]
            absorption_rate = ka_per_h * mg
            elimination_rate = ke * cp * vd_effective
            m_gut[i] = max(0.0, mg - absorption_rate * dt_h)
            c_plasma[i] = max(0.0, cp + (absorption_rate - elimination_rate) * dt_h / vd_effective)

    # Summary statistics
    cmax = max(c_plasma)
    tmax = times[c_plasma.index(cmax)]

    # AUC via trapezoidal rule
    auc = 0.0
    for i in range(1, n_steps):
        auc += 0.5 * (c_plasma[i - 1] + c_plasma[i]) * dt_h

    t_half = math.log(2) / ke

    notes = (
        f"CO_fraction={cardiac_output_fraction:.2f}: "
        f"CL_eff={cl_effective:.2f} L/h, Vd_eff={vd_effective:.1f} L, "
        f"t½={t_half:.1f} h"
    )

    return CardiacPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cardiac_output_fraction=cardiac_output_fraction,
        route=route,
        cl_effective_L_per_h=cl_effective,
        vd_effective_L=vd_effective,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        cmax=cmax,
        tmax_h=tmax,
        auc=auc,
        t_half_h=t_half,
        notes=notes,
    )


# NYHA class -> CO fraction mapping
_NYHA_CO_FRACTION: dict[int, float] = {
    1: 0.9,  # NYHA I: mild, near-normal
    2: 0.7,  # NYHA II: moderate impairment
    3: 0.5,  # NYHA III: significant impairment
    4: 0.3,  # NYHA IV: severe impairment
}


def heart_failure_dose_adjustment(
    drug_name: str,
    normal_dose_mg: float,
    cl_normal_L_per_h: float,
    vd_L: float,
    nyha_class: int,
) -> HFDoseResult:
    """Calculate dose adjustment for heart failure by NYHA class.

    Uses NYHA functional class to determine cardiac output fraction,
    then scales dose proportionally to maintain equivalent exposure.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    normal_dose_mg:
        Normal dose in healthy subjects (mg).
    cl_normal_L_per_h:
        Normal systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    nyha_class:
        NYHA functional class (1, 2, 3, or 4).

    Returns
    -------
    HFDoseResult
    """
    # --- Input validation ---
    if normal_dose_mg <= 0:
        raise ValueError("normal_dose_mg must be > 0")
    if cl_normal_L_per_h <= 0:
        raise ValueError("cl_normal_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if nyha_class not in (1, 2, 3, 4):
        raise ValueError("nyha_class must be 1, 2, 3, or 4")

    co_frac = _NYHA_CO_FRACTION[nyha_class]

    # Scale CL (same formula as simulate_cardiac_pk)
    cl_effective = cl_normal_L_per_h * co_frac * 0.8
    cl_effective = max(cl_normal_L_per_h * 0.05, cl_effective)

    # Scale Vd for edema
    vd_effective = vd_L * (1.0 + (1.0 - co_frac) * 0.3)

    ke_effective = cl_effective / vd_effective
    t_half = math.log(2) / ke_effective

    # Adjusted dose: proportional to CL reduction (maintain same AUC)
    cl_ratio = cl_effective / cl_normal_L_per_h
    adjusted_dose = normal_dose_mg * cl_ratio
    dose_reduction_pct = max(0.0, (1.0 - cl_ratio) * 100.0)

    nyha_labels = {1: "I (mild)", 2: "II (moderate)", 3: "III (significant)", 4: "IV (severe)"}
    rationale = (
        f"NYHA class {nyha_labels[nyha_class]}: CO reduced to {co_frac * 100:.0f}% of normal. "
        f"CL reduced from {cl_normal_L_per_h:.1f} to {cl_effective:.2f} L/h "
        f"({dose_reduction_pct:.1f}% reduction). "
        f"Dose reduced proportionally to maintain target AUC. "
        f"t½ extended to {t_half:.1f} h."
    )

    return HFDoseResult(
        drug_name=drug_name,
        normal_dose_mg=normal_dose_mg,
        nyha_class=nyha_class,
        cardiac_output_fraction=co_frac,
        cl_effective_L_per_h=cl_effective,
        adjusted_dose_mg=adjusted_dose,
        dose_reduction_pct=dose_reduction_pct,
        t_half_h=t_half,
        rationale=rationale,
    )
