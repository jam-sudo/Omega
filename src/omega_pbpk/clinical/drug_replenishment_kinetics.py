"""Drug replenishment kinetics — models substrate depletion and recovery.

Models drugs that deplete endogenous substrates (e.g., coenzyme Q10 by statins,
folate by methotrexate, potassium by diuretics). Substrate replenishment occurs via
endogenous synthesis and dietary intake.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReplenishmentResult:
    drug_name: str
    substrate_name: str
    drug_dose_mg: float
    times_h: list
    c_drug_mg_L: list  # drug concentration over time
    s_substrate_units: list  # substrate level (100 = normal baseline)
    drug_cmax_mg_L: float
    drug_auc: float
    substrate_nadir: float  # minimum substrate level achieved
    time_to_nadir_h: float
    depletion_pct: float  # % depletion from baseline: (100 - nadir)/100 * 100
    time_to_recovery_h: float  # time after drug clearance for substrate to reach 90% of baseline
    recovery_complete: bool  # True if substrate >= 90% baseline by t_end
    clinical_significance: str  # "none"(<10%), "minor"(10-30%), "moderate"(30-60%), "severe"(>60%)
    notes: str


def _classify_significance(depletion_pct: float) -> str:
    if depletion_pct < 10.0:
        return "none"
    elif depletion_pct < 30.0:
        return "minor"
    elif depletion_pct < 60.0:
        return "moderate"
    else:
        return "severe"


def simulate_replenishment_kinetics(
    drug_name: str,
    substrate_name: str,
    drug_dose_mg: float,
    drug_cl_L_per_h: float = 5.0,
    drug_vd_L: float = 50.0,
    drug_route: str = "iv",  # "iv" or "oral"
    drug_ka_per_h: float = 1.0,
    drug_f_oral: float = 0.8,
    k_depletion_per_h: float = 0.05,  # rate of substrate depletion per unit drug conc
    k_synthesis_per_h: float = 0.1,  # endogenous synthesis rate (restores to baseline)
    substrate_baseline: float = 100.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> ReplenishmentResult:
    """Simulate drug replenishment kinetics for substrate-depleting drugs.

    Uses Forward Euler integration:
    - IV: dC/dt = -ke*C
    - Oral: dA_gut/dt = -ka*A_gut, dC/dt = ka*f*A_gut/Vd - ke*C
    - dS/dt = k_syn*(baseline - S) - k_dep*C*S/baseline
    """
    # Validation
    if drug_dose_mg <= 0:
        raise ValueError("drug_dose_mg must be > 0")
    if drug_cl_L_per_h <= 0:
        raise ValueError("drug_cl_L_per_h must be > 0")
    if drug_vd_L <= 0:
        raise ValueError("drug_vd_L must be > 0")
    if not (0 < drug_f_oral <= 1.0):
        raise ValueError("drug_f_oral must be in (0, 1]")
    if k_depletion_per_h < 0:
        raise ValueError("k_depletion_per_h must be >= 0")
    if k_synthesis_per_h <= 0:
        raise ValueError("k_synthesis_per_h must be > 0")
    if substrate_baseline <= 0:
        raise ValueError("substrate_baseline must be > 0")
    if drug_route not in ("iv", "oral"):
        raise ValueError("drug_route must be 'iv' or 'oral'")

    ke = drug_cl_L_per_h / drug_vd_L

    # Initial conditions
    if drug_route == "iv":
        c = drug_dose_mg / drug_vd_L
        a_gut = 0.0
    else:
        c = 0.0
        a_gut = drug_dose_mg  # full dose in gut at t=0

    s = substrate_baseline

    times_h = []
    c_drug_list = []
    s_substrate_list = []

    n_steps = int(round(t_end_h / dt_h))

    for i in range(n_steps + 1):
        t = i * dt_h
        times_h.append(t)
        c_drug_list.append(c)
        s_substrate_list.append(s)

        if i < n_steps:
            # Forward Euler
            if drug_route == "iv":
                dc = -ke * c
                da_gut = 0.0
            else:
                da_gut = -drug_ka_per_h * a_gut
                dc = drug_ka_per_h * drug_f_oral * a_gut / drug_vd_L - ke * c

            ds = (
                k_synthesis_per_h * (substrate_baseline - s)
                - k_depletion_per_h * c * s / substrate_baseline
            )

            a_gut = max(0.0, a_gut + da_gut * dt_h)
            c = max(0.0, c + dc * dt_h)
            s = max(0.0, s + ds * dt_h)

    # Compute drug Cmax and AUC
    drug_cmax = max(c_drug_list)
    drug_auc = sum(
        (c_drug_list[i] + c_drug_list[i + 1]) / 2.0 * (times_h[i + 1] - times_h[i])
        for i in range(len(times_h) - 1)
    )

    # Substrate nadir and time to nadir
    substrate_nadir = min(s_substrate_list)
    nadir_idx = s_substrate_list.index(substrate_nadir)
    time_to_nadir_h = times_h[nadir_idx]

    depletion_pct = (substrate_baseline - substrate_nadir) / substrate_baseline * 100.0

    # Time to recovery: find when substrate first reaches 90% of baseline AFTER nadir
    recovery_threshold = 0.9 * substrate_baseline
    time_to_recovery_h = 0.0
    recovery_complete = False

    for i in range(nadir_idx, len(s_substrate_list)):
        if s_substrate_list[i] >= recovery_threshold:
            time_to_recovery_h = times_h[i] - times_h[nadir_idx]
            recovery_complete = True
            break

    clinical_significance = _classify_significance(depletion_pct)

    notes_parts = [
        f"Drug {drug_name} depletes {substrate_name} via concentration-dependent mechanism.",
        f"Nadir at {time_to_nadir_h:.1f}h ({depletion_pct:.1f}% depletion).",
        f"Clinical significance: {clinical_significance}.",
    ]
    if recovery_complete:
        notes_parts.append(
            f"Recovery to 90% baseline achieved at {time_to_nadir_h + time_to_recovery_h:.1f}h."
        )
    else:
        notes_parts.append(
            "Substrate did not fully recover to 90% baseline within simulation period."
        )

    notes = " ".join(notes_parts)

    return ReplenishmentResult(
        drug_name=drug_name,
        substrate_name=substrate_name,
        drug_dose_mg=drug_dose_mg,
        times_h=times_h,
        c_drug_mg_L=c_drug_list,
        s_substrate_units=s_substrate_list,
        drug_cmax_mg_L=drug_cmax,
        drug_auc=drug_auc,
        substrate_nadir=substrate_nadir,
        time_to_nadir_h=time_to_nadir_h,
        depletion_pct=depletion_pct,
        time_to_recovery_h=time_to_recovery_h,
        recovery_complete=recovery_complete,
        clinical_significance=clinical_significance,
        notes=notes,
    )


def compare_depletion_recovery(
    drug_name: str,
    substrate_name: str,
    drug_doses_mg: list,
    drug_cl_L_per_h: float = 5.0,
    drug_vd_L: float = 50.0,
) -> list:
    """Run simulation for each dose and return list sorted by depletion_pct descending."""
    results = []
    for dose in drug_doses_mg:
        result = simulate_replenishment_kinetics(
            drug_name=drug_name,
            substrate_name=substrate_name,
            drug_dose_mg=dose,
            drug_cl_L_per_h=drug_cl_L_per_h,
            drug_vd_L=drug_vd_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.depletion_pct, reverse=True)
    return results
