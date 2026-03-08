"""Phase 484 — Inhaled Drug Deposition Model.

Models regional deposition of inhaled drugs in the respiratory tract and
subsequent systemic absorption using a simplified ICRP-based model.
"""

from dataclasses import dataclass


@dataclass
class InhaledDepositionResult:
    """Result of inhaled deposition simulation."""

    drug_name: str
    total_dose_mg: float
    MMAD_um: float
    times_h: list
    c_systemic_mg_L: list
    f_oropharyngeal: float
    f_tracheobronchial: float
    f_alveolar: float
    f_exhaled: float
    dose_oropharyngeal_mg: float
    dose_tb_mg: float
    dose_alveolar_mg: float
    cmax_systemic_mg_L: float
    auc_systemic_mg_h_per_L: float
    lung_bioavailability_pct: float
    notes: str


def _compute_deposition_fractions(mmad: float) -> tuple:
    """Compute regional deposition fractions (oropharyngeal, TB, alveolar, exhaled).

    Based on simplified ICRP model.
    """
    # Oropharyngeal
    if mmad > 1.0:
        fo = min(0.9, 0.5 + 0.04 * (mmad - 1.0) ** 2)
    else:
        fo = 0.1

    # Tracheobronchial
    ftb = 0.2 * max(0.0, 1.0 - abs(mmad - 3.0) / 3.0)

    # Alveolar
    if mmad < 5.0:
        fa = max(0.0, 0.3 * (1.0 - abs(mmad - 2.0) / 3.0))
    else:
        fa = 0.0

    # Exhaled fraction (remainder)
    fex = max(0.0, 1.0 - fo - ftb - fa)

    # Normalize so fractions sum to 1
    total = fo + ftb + fa + fex
    if total > 0.0:
        fo /= total
        ftb /= total
        fa /= total
        fex /= total

    return fo, ftb, fa, fex


def simulate_inhaled_deposition(
    drug_name: str,
    total_dose_mg: float,
    MMAD_um: float,
    f_oral: float = 0.3,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> InhaledDepositionResult:
    """Simulate regional inhaled drug deposition and systemic PK.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    total_dose_mg:
        Total inhaled dose in mg.
    MMAD_um:
        Mass median aerodynamic diameter in micrometres.
    f_oral:
        Oral bioavailability fraction for oropharyngeally deposited drug.
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vd_sys_L:
        Volume of distribution in L.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        ODE integration time step in hours.

    Returns
    -------
    InhaledDepositionResult
    """
    if total_dose_mg <= 0:
        raise ValueError(f"total_dose_mg must be > 0, got {total_dose_mg}")
    if MMAD_um <= 0:
        raise ValueError(f"MMAD_um must be > 0, got {MMAD_um}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if not (0 <= f_oral <= 1):
        raise ValueError(f"f_oral must be in [0, 1], got {f_oral}")

    # Regional deposition fractions
    fo, ftb, fa, fex = _compute_deposition_fractions(MMAD_um)

    dose_op_mg = fo * total_dose_mg
    dose_tb_mg = ftb * total_dose_mg
    dose_alv_mg = fa * total_dose_mg

    # Absorption rates per region (1/h)
    # oropharynx (k_abs_op=0.2/h) → swallowed; handled via GI depot below
    k_abs_tb = 1.0  # tracheobronchial: mucociliary
    k_abs_alv = 5.0  # alveolar: fast

    # Depot compartments (mass remaining, mg)
    # ODE states: [depot_tb, depot_alv, depot_op, depot_op_gi, c_sys]
    # Oropharynx dose goes to GI → absorbed via f_oral
    # TB and alveolar → direct systemic absorption

    depot_tb = dose_tb_mg
    depot_alv = dose_alv_mg
    depot_op_gi = dose_op_mg  # majority swallowed → GI tract
    c_sys = 0.0  # systemic concentration mg/L

    # GI absorption rate (assume first-order, k = 0.5/h for oral)
    k_abs_gi = 0.5

    times = []
    concentrations = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _ in range(n_steps):
        times.append(t)
        concentrations.append(c_sys)

        # Rates
        rate_tb = k_abs_tb * depot_tb
        rate_alv = k_abs_alv * depot_alv
        rate_op_gi = k_abs_gi * depot_op_gi * f_oral  # oral-absorbed fraction only

        # Total systemic input rate (mg/h)
        input_rate = rate_tb + rate_alv + rate_op_gi

        # Systemic elimination rate
        elim_rate_mg_per_h = cl_sys_L_per_h * c_sys

        # Forward Euler updates
        dc_sys = (input_rate - elim_rate_mg_per_h) / vd_sys_L
        c_sys = max(0.0, c_sys + dc_sys * dt_h)

        depot_tb = max(0.0, depot_tb - rate_tb * dt_h)
        depot_alv = max(0.0, depot_alv - rate_alv * dt_h)
        depot_op_gi = max(0.0, depot_op_gi - k_abs_gi * depot_op_gi * dt_h)

        t += dt_h

    # AUC via manual trapezoidal rule
    auc = sum(
        0.5 * (concentrations[i] + concentrations[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    cmax = max(concentrations) if concentrations else 0.0

    # Lung bioavailability: fraction absorbed from lung (TB + alveolar) vs total dose
    lung_dose_mg = dose_tb_mg + dose_alv_mg
    lung_bav_pct = 100.0 * lung_dose_mg / total_dose_mg if total_dose_mg > 0 else 0.0

    notes_parts = [
        f"MMAD={MMAD_um} um",
        f"f_alv={fa:.3f}",
        f"f_tb={ftb:.3f}",
        f"f_op={fo:.3f}",
        f"f_ex={fex:.3f}",
    ]
    if fa > 0.25:
        notes_parts.append("Good alveolar deposition; optimal for systemic delivery.")
    elif fo > 0.5:
        notes_parts.append("High oropharyngeal deposition; consider smaller particle size.")
    else:
        notes_parts.append("Predominantly tracheobronchial deposition.")

    return InhaledDepositionResult(
        drug_name=drug_name,
        total_dose_mg=total_dose_mg,
        MMAD_um=MMAD_um,
        times_h=times,
        c_systemic_mg_L=concentrations,
        f_oropharyngeal=fo,
        f_tracheobronchial=ftb,
        f_alveolar=fa,
        f_exhaled=fex,
        dose_oropharyngeal_mg=dose_op_mg,
        dose_tb_mg=dose_tb_mg,
        dose_alveolar_mg=dose_alv_mg,
        cmax_systemic_mg_L=cmax,
        auc_systemic_mg_h_per_L=auc,
        lung_bioavailability_pct=lung_bav_pct,
        notes="; ".join(notes_parts),
    )


def optimize_particle_size(
    drug_name: str,
    total_dose_mg: float,
    MMAD_values: list = None,
) -> list:
    """Screen MMAD values and return results sorted by AUC descending.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    total_dose_mg:
        Total inhaled dose in mg.
    MMAD_values:
        List of MMAD values (µm) to screen. Defaults to [0.5, 1, 2, 3, 5, 10].

    Returns
    -------
    List of InhaledDepositionResult sorted by auc_systemic_mg_h_per_L descending.
    """
    if MMAD_values is None:
        MMAD_values = [0.5, 1, 2, 3, 5, 10]

    results = []
    for mmad in MMAD_values:
        result = simulate_inhaled_deposition(
            drug_name=drug_name,
            total_dose_mg=total_dose_mg,
            MMAD_um=mmad,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_systemic_mg_h_per_L, reverse=True)
    return results
