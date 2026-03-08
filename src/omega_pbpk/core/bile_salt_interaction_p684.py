"""Phase 684 — Bile Salt Interaction model."""

from dataclasses import dataclass, field


@dataclass
class BileSaltInteractionResult:
    drug_name: str
    dose_mg: float
    bile_salt_conc_mM: float = 0.0
    times_h: list = field(default_factory=list)
    c_free_gut_mg_L: list = field(default_factory=list)
    c_micellar_mg_L: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    f_micellar: float = 0.0
    effective_solubility_mg_L: float = 0.0
    cmax_mg_L: float = 0.0
    auc_mg_h_per_L: float = 0.0
    absorption_enhancement_fold: float = 1.0
    notes: str = ""


def _run_bile_sim(
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    bile_salt_conc_mM: float,
    v_gut_L: float,
    t_end_h: float,
    dt_h: float,
):
    """Run the ODE simulation and return (times, c_free_gut, c_micellar, c_plasma, f_micellar, R)."""
    R = 1.0 + bile_salt_conc_mM * max(0.0, logP) * 0.1
    R = max(1.0, min(50.0, R))
    f_mic = (R - 1.0) / R if R > 1.0 else 0.0

    f_oral = 0.85
    ka_free = 0.6
    k_exchange = 2.0
    ke = cl_sys_L_per_h / vd_sys_L

    # Solubility-limited dissolution: bile salts increase effective solubility
    base_solubility = 100.0 / (1.0 + max(0.0, logP))
    eff_sol = base_solubility * R
    max_dissolved = eff_sol * v_gut_L  # mg that can be in solution

    total_gut_drug = dose_mg * f_oral
    dissolved = min(total_gut_drug, max_dissolved)
    undissolved = total_gut_drug - dissolved

    a_gut_free = dissolved * (1.0 - f_mic)
    a_gut_mic = dissolved * f_mic
    a_solid = undissolved
    c_plasma = 0.0

    n_steps = int(t_end_h / dt_h) + 1
    times = []
    c_free_list = []
    c_mic_list = []
    c_plasma_list = []

    k_dissolve = 1.0  # /h dissolution rate from solid

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)
        c_free_list.append(a_gut_free / v_gut_L)
        c_mic_list.append(a_gut_mic / v_gut_L)
        c_plasma_list.append(c_plasma)

        # Dissolution from solid into solution (up to solubility limit)
        current_dissolved = a_gut_free + a_gut_mic
        room = max(0.0, max_dissolved - current_dissolved)
        dissolve_amount = min(a_solid, room) * k_dissolve

        total_gut = a_gut_free + a_gut_mic
        eq_mic = f_mic * total_gut
        eq_free = total_gut - eq_mic

        d_a_solid = -dissolve_amount
        d_a_gut_free = (
            k_exchange * (eq_free - a_gut_free)
            - ka_free * a_gut_free
            + dissolve_amount * (1.0 - f_mic)
        )
        d_a_gut_mic = k_exchange * (eq_mic - a_gut_mic) + dissolve_amount * f_mic
        d_c_plasma = ka_free * a_gut_free / vd_sys_L - ke * c_plasma

        a_solid += d_a_solid * dt_h
        a_gut_free += d_a_gut_free * dt_h
        a_gut_mic += d_a_gut_mic * dt_h
        c_plasma += d_c_plasma * dt_h

        a_solid = max(0.0, a_solid)
        a_gut_free = max(0.0, a_gut_free)
        a_gut_mic = max(0.0, a_gut_mic)
        c_plasma = max(0.0, c_plasma)

    return times, c_free_list, c_mic_list, c_plasma_list, f_mic, R


def simulate_bile_salt_interaction(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    bile_salt_conc_mM: float = 10.0,
    v_gut_L: float = 0.25,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> BileSaltInteractionResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if bile_salt_conc_mM < 0:
        raise ValueError("bile_salt_conc_mM must be >= 0")

    # Main simulation
    times, c_free_list, c_mic_list, c_plasma_list, f_mic, R = _run_bile_sim(
        dose_mg,
        logP,
        cl_sys_L_per_h,
        vd_sys_L,
        bile_salt_conc_mM,
        v_gut_L,
        t_end_h,
        dt_h,
    )

    # Effective solubility
    base_solubility = 100.0 / (1.0 + max(0.0, logP))
    effective_solubility = base_solubility * R

    # AUC (trapezoidal)
    auc = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # Reference AUC (no bile salts)
    _, _, _, c_plasma_ref, _, _ = _run_bile_sim(
        dose_mg,
        logP,
        cl_sys_L_per_h,
        vd_sys_L,
        0.0,
        v_gut_L,
        t_end_h,
        dt_h,
    )
    ref_auc = sum(
        0.5 * (c_plasma_ref[i] + c_plasma_ref[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    absorption_enhancement_fold = auc / ref_auc if ref_auc > 0 else 1.0

    cmax = max(c_plasma_list)

    notes = (
        f"Bile salt interaction simulation for {drug_name}. "
        f"Bile salt conc={bile_salt_conc_mM} mM, R={R:.2f}, "
        f"f_micellar={f_mic:.4f}."
    )

    return BileSaltInteractionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        bile_salt_conc_mM=bile_salt_conc_mM,
        times_h=times,
        c_free_gut_mg_L=c_free_list,
        c_micellar_mg_L=c_mic_list,
        c_plasma_mg_L=c_plasma_list,
        f_micellar=f_mic,
        effective_solubility_mg_L=effective_solubility,
        cmax_mg_L=cmax,
        auc_mg_h_per_L=auc,
        absorption_enhancement_fold=absorption_enhancement_fold,
        notes=notes,
    )
