"""Intravitreal (IVT) injection PK model — 3-compartment Forward Euler.

Models drug distribution from vitreous humor → retina/choroid and
vitreous → aqueous humor drainage after intravitreal injection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class IntravitreaPKResult:
    """Result of intravitreal PK simulation."""

    drug_name: str
    dose_mcg: float
    mw_kDa: float
    times_h: list = field(default_factory=list)
    c_vitreous_mg_L: list = field(default_factory=list)
    c_retina_mg_L: list = field(default_factory=list)
    c_aqueous_mg_L: list = field(default_factory=list)
    cmax_vitreous_mg_L: float = 0.0
    cmax_retina_mg_L: float = 0.0
    tmax_retina_h: float = 0.0
    auc_vitreous_mg_h_per_L: float = 0.0
    auc_retina_mg_h_per_L: float = 0.0
    t_half_vitreous_h: float = 0.0
    time_above_therapeutic_h: float = 0.0
    notes: str = ""


def simulate_intravitreal_pk(
    drug_name: str,
    dose_mcg: float,
    mw_kDa: float = 48.0,
    logp: float = -2.0,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> IntravitreaPKResult:
    """Simulate intravitreal injection PK using 3-compartment Forward Euler.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mcg:
        Dose in micrograms (µg).
    mw_kDa:
        Molecular weight in kDa.
    logp:
        Lipophilicity (octanol-water partition coefficient).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Time step in hours.

    Returns
    -------
    IntravitreaPKResult
    """
    if dose_mcg <= 0:
        raise ValueError(f"dose_mcg must be > 0, got {dose_mcg}")
    if mw_kDa <= 0:
        raise ValueError(f"mw_kDa must be > 0, got {mw_kDa}")

    # Compartment volumes (L)
    v_vit = 0.004  # vitreous: 4.0 mL
    v_ret = 0.0005  # retina: 0.5 mL
    v_aq = 0.00025  # aqueous: 0.25 mL

    # Elimination rates depend on molecular size
    if mw_kDa > 5.0:
        # Large proteins (e.g., anti-VEGF biologics)
        k_vit_ant = 0.007  # /h  anterior route (slow)
        k_vit_post = 0.001  # /h  posterior route
    else:
        # Small molecules
        k_vit_ant = 0.05  # /h
        k_vit_post = 0.01  # /h

    # Vitreous → retina transfer rate (hydrophilic better)
    k_ret = max(0.001, 0.02 - 0.003 * max(0.0, logp))
    k_ret_back = k_ret * 0.1  # slow back diffusion

    k_vit_total = k_vit_ant + k_vit_post + k_ret

    # Initial conditions (mg/L)
    c_vit = dose_mcg / 1000.0 / v_vit  # µg → mg, then /V_vit
    c_ret = 0.0
    c_aq = 0.0

    # Half-life from analytical rate
    t_half = math.log(2.0) / k_vit_total

    # Determine number of steps
    n_steps = max(1, int(round(t_end_h / dt_h)))

    times: list[float] = []
    cv_list: list[float] = []
    cr_list: list[float] = []
    ca_list: list[float] = []

    cmax_ret = 0.0
    tmax_ret = 0.0
    time_above = 0.0
    auc_vit = 0.0
    auc_ret = 0.0

    for step in range(n_steps + 1):
        t = step * dt_h
        times.append(t)
        cv_list.append(c_vit)
        cr_list.append(c_ret)
        ca_list.append(c_aq)

        if c_ret > cmax_ret:
            cmax_ret = c_ret
            tmax_ret = t

        if c_vit > 0.1:
            time_above += dt_h

        # Trapezoidal AUC accumulation (add trapezoid at end of interval)
        if step < n_steps:
            # Forward Euler derivatives
            dc_vit = -k_vit_total * c_vit + k_ret_back * c_ret * v_ret / v_vit
            dc_ret = k_ret * c_vit * v_vit / v_ret - k_ret_back * c_ret - 0.02 * c_ret
            dc_aq = k_vit_ant * c_vit * v_vit / v_aq - 0.1 * c_aq

            c_vit_next = c_vit + dc_vit * dt_h
            c_ret_next = c_ret + dc_ret * dt_h
            c_aq_next = c_aq + dc_aq * dt_h

            # Clamp negative concentrations
            c_vit_next = max(0.0, c_vit_next)
            c_ret_next = max(0.0, c_ret_next)
            c_aq_next = max(0.0, c_aq_next)

            # Trapezoidal AUC
            auc_vit += 0.5 * (c_vit + c_vit_next) * dt_h
            auc_ret += 0.5 * (c_ret + c_ret_next) * dt_h

            c_vit = c_vit_next
            c_ret = c_ret_next
            c_aq = c_aq_next

    cmax_vit = cv_list[0]  # initial concentration is maximum for vitreous

    # Correct time_above: subtract one extra dt_h added for step 0
    # The loop adds dt_h for each step where c_vit > 0.1, including step 0.
    # Since step=0 is included and we want the integral, keep as-is but
    # clamp to simulation duration.
    time_above = min(time_above, t_end_h)

    notes = (
        f"{'Large molecule' if mw_kDa > 5 else 'Small molecule'} "
        f"(MW={mw_kDa} kDa). "
        f"k_vit_total={k_vit_total:.4f}/h, k_ret={k_ret:.4f}/h. "
        f"Vitreous t½={t_half:.1f} h."
    )

    return IntravitreaPKResult(
        drug_name=drug_name,
        dose_mcg=dose_mcg,
        mw_kDa=mw_kDa,
        times_h=times,
        c_vitreous_mg_L=cv_list,
        c_retina_mg_L=cr_list,
        c_aqueous_mg_L=ca_list,
        cmax_vitreous_mg_L=cmax_vit,
        cmax_retina_mg_L=cmax_ret,
        tmax_retina_h=tmax_ret,
        auc_vitreous_mg_h_per_L=auc_vit,
        auc_retina_mg_h_per_L=auc_ret,
        t_half_vitreous_h=t_half,
        time_above_therapeutic_h=time_above,
        notes=notes,
    )
