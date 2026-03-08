"""Phase 1088 — Inhalation Anesthetic PK.

Pharmacokinetics of volatile inhalation anesthetics (sevoflurane, isoflurane,
desflurane, halothane) using a 4-compartment model with alveolar equilibration.
"""

from dataclasses import dataclass, field

# Agent-specific partition coefficients
_AGENTS: dict[str, dict[str, float]] = {
    "sevoflurane": {"lambda_blood_gas": 0.65, "lambda_oil_gas": 47.0},
    "isoflurane": {"lambda_blood_gas": 1.4, "lambda_oil_gas": 91.0},
    "desflurane": {"lambda_blood_gas": 0.42, "lambda_oil_gas": 19.0},
    "halothane": {"lambda_blood_gas": 2.5, "lambda_oil_gas": 224.0},
}

# Tissue compartment parameters
# (volume_L, fractional_CO, tissue_lipid_fraction)
_TISSUES: dict[str, tuple[float, float, float]] = {
    "vrg": (6.0, 0.75, 0.04),
    "muscle": (33.0, 0.19, 0.15),
    "fat": (14.0, 0.05, 0.80),
    "vpg": (22.0, 0.01, 0.05),
}


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal AUC integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


@dataclass
class InhalationAnestheticResult:
    """Result of inhalation anesthetic PK simulation."""

    agent: str
    lambda_blood_gas: float
    t_induction_h: float
    t_recovery_h: float
    times_h: list = field(default_factory=list)
    c_vrg: list = field(default_factory=list)
    c_muscle: list = field(default_factory=list)
    c_fat: list = field(default_factory=list)
    peak_fat_accumulation: float = 0.0
    fat_equilibration_pct_at_2h: float = 0.0
    notes: str = ""


def simulate_inhalation_anesthetic(
    agent: str,
    f_inspired: float = 0.05,
    maintenance_h: float = 2.0,
    co_L_per_h: float = 300.0,
    alveolar_ventilation_L_per_h: float = 300.0,
    t_end_h: float = 6.0,
    dt_h: float = 0.01,
) -> InhalationAnestheticResult:
    """Simulate inhalation anesthetic pharmacokinetics.

    Parameters
    ----------
    agent : str
        Anesthetic agent: 'sevoflurane', 'isoflurane', 'desflurane', or 'halothane'.
    f_inspired : float
        Inspired fraction [0, 1] (e.g., 0.05 = 5%).
    maintenance_h : float
        Duration of maintenance phase in hours.
    co_L_per_h : float
        Cardiac output in L/h.
    alveolar_ventilation_L_per_h : float
        Alveolar ventilation in L/h.
    t_end_h : float
        Total simulation time in hours.
    dt_h : float
        Forward Euler time step in hours.

    Returns
    -------
    InhalationAnestheticResult
    """
    if agent not in _AGENTS:
        raise ValueError(f"Unknown agent '{agent}'. Choose from: {sorted(_AGENTS.keys())}")
    if not (0.0 < f_inspired < 1.0):
        raise ValueError(f"f_inspired must be in (0, 1), got {f_inspired}")
    if maintenance_h <= 0.0:
        raise ValueError(f"maintenance_h must be > 0, got {maintenance_h}")
    if co_L_per_h <= 0.0:
        raise ValueError(f"co_L_per_h must be > 0, got {co_L_per_h}")
    if alveolar_ventilation_L_per_h <= 0.0:
        raise ValueError(
            f"alveolar_ventilation_L_per_h must be > 0, got {alveolar_ventilation_L_per_h}"
        )

    params = _AGENTS[agent]
    lambda_bg = params["lambda_blood_gas"]
    lambda_og = params["lambda_oil_gas"]

    # Tissue blood-gas partition: Kp_t = lambda_tissue_blood / lambda_bg
    # lambda_tissue_blood = lambda_oil_gas / lambda_blood_gas * tissue_lipid_fraction
    tissue_kp: dict[str, float] = {}
    tissue_volumes: dict[str, float] = {}
    tissue_flows: dict[str, float] = {}
    for tissue, (vol, fco, lipid) in _TISSUES.items():
        lambda_tb = lambda_og / lambda_bg * lipid
        # Kp_tissue = lambda_tb (ratio of tissue to blood conc at equilibrium)
        tissue_kp[tissue] = lambda_tb
        tissue_volumes[tissue] = vol
        tissue_flows[tissue] = fco * co_L_per_h  # L/h

    # State: alveolar concentration (MAC-equivalent, dimensionless = fraction of inspired)
    # and tissue concentrations (MAC-equivalent units)
    c_alv = 0.0
    c_tissue: dict[str, float] = {t: 0.0 for t in _TISSUES}

    times = []
    c_vrg_list = []
    c_muscle_list = []
    c_fat_list = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    # Track induction time: first time VRG reaches 50% of c_alveolar_target
    # During maintenance c_alv_target = f_inspired
    c_alv_target = f_inspired
    t_induction_h = float("nan")
    t_recovery_h = float("nan")

    # After washout begins (t > maintenance_h), track when VRG falls to 10% of maintenance level
    vrg_at_end_maintenance = None
    washout_started = False

    for step in range(n_steps):
        t = step * dt_h

        # Record state
        times.append(t)
        c_vrg_list.append(c_tissue["vrg"])
        c_muscle_list.append(c_tissue["muscle"])
        c_fat_list.append(c_tissue["fat"])

        # Check induction (VRG reaches 50% of steady-state during maintenance)
        if t <= maintenance_h and float("nan") == float("nan"):  # always True — check nan
            # Steady-state VRG = c_alv_target * lambda_bg * Kp_vrg / Kp_vrg
            # At equilibrium: c_tissue = c_blood * Kp_t = c_alv * lambda_bg * Kp_t
            # But we track in MAC-equivalent so c_blood = c_alv * lambda_bg
            # Actually tissue conc at equilibrium = c_alv * lambda_bg (blood-equivalent)
            # We store tissue conc as MAC-fraction units where equilibrium = f_inspired * Kp_t
            # Actually let's define: C_tissue in "blood-equivalent" units = MAC fraction * lambda_bg
            # Steady-state: C_tissue_eq = C_blood = C_alv * lambda_bg = f_inspired * lambda_bg
            # But tissue equilibrium considering Kp: C_tissue_eq = C_blood * Kp_t = f_inspired * lambda_bg * Kp_t
            # Induction = VRG reaches 50% of C_tissue_eq_vrg
            c_vrg_eq = c_alv_target * lambda_bg * tissue_kp["vrg"]
            if float("nan") != float("nan"):  # placeholder to structure the if below
                pass

        # Determine alveolar input
        if t <= maintenance_h:
            # Maintenance: alveolar concentration is driven toward f_inspired
            # Use first-order approach: d_c_alv/dt = Va/FRC * (f_inspired - c_alv / lambda_bg * lambda_bg)
            # Simplified: alveolar equilibrates via ventilation
            # FRC functional residual capacity ≈ 3L → very fast → quasi-steady-state
            # For simplicity, set c_alv as the result of alveolar ventilation balance:
            # Va*(F_i - C_A) = Q_blood*lambda_bg*(C_A - C_venous_mean/lambda_bg)
            # Approximate: c_alv = f_inspired (driven by high ventilation)
            c_alv = f_inspired
        else:
            # Washout: inspired = 0; alveolar washes out exponentially
            # d_C_alv/dt = -Va/FRC * C_alv + Q*C_venous / (Va + Q*lambda_bg)
            # Simplified approach: alveolar concentration decays via ventilation
            # Using: Va*(0 - C_alv) = net elimination; time constant = lambda_bg / (Va/FRC_volume)
            # FRC ~ 2.5 L = 0.0417 * 60 = 2.5 L
            # Quasi-steady-state alveolar: C_alv = (Va*F_i + Q*C_venous/lambda_bg) / (Va + Q)
            # During washout F_i = 0; back-diffusion from venous blood sustains C_alv
            c_venous = 0.0
            for tissue in _TISSUES:
                q_t = tissue_flows[tissue]
                c_ven_t = c_tissue[tissue] / tissue_kp[tissue]
                c_venous += q_t * c_ven_t
            c_venous /= co_L_per_h

            c_alv = (alveolar_ventilation_L_per_h * 0.0 + co_L_per_h * c_venous / lambda_bg) / (
                alveolar_ventilation_L_per_h + co_L_per_h
            )

            if not washout_started:
                washout_started = True
                vrg_at_end_maintenance = c_tissue["vrg"]

        # Update tissue concentrations via forward Euler
        c_blood = c_alv * lambda_bg  # blood concentration (MAC-fraction * lambda_bg)
        new_c_tissue: dict[str, float] = {}
        for tissue, (vol, _fco, _lipid) in _TISSUES.items():
            q_t = tissue_flows[tissue]
            kp = tissue_kp[tissue]
            ct = c_tissue[tissue]
            # dC_tissue/dt = Q_t/V_t * (C_blood_in - C_tissue/Kp_t)
            dc_dt = q_t / vol * (c_blood - ct / kp)
            new_c_tissue[tissue] = ct + dc_dt * dt_h
        c_tissue = new_c_tissue

        # Check induction: VRG reaches 50% of equilibrium during maintenance
        if t <= maintenance_h and float("nan") != t_induction_h:
            pass  # not yet found
        if t <= maintenance_h and (t_induction_h != t_induction_h):  # nan check
            c_vrg_eq = c_alv_target * lambda_bg * tissue_kp["vrg"]
            if c_tissue["vrg"] >= 0.5 * c_vrg_eq:
                t_induction_h = t

        # Check recovery: VRG falls to 10% of maintenance-end level after washout
        if (
            washout_started
            and (t_recovery_h != t_recovery_h)
            and vrg_at_end_maintenance is not None
        ):
            threshold = 0.1 * vrg_at_end_maintenance
            if c_tissue["vrg"] <= threshold and c_tissue["vrg"] >= 0.0:
                t_recovery_h = t

    # Handle cases where thresholds were never crossed
    if t_induction_h != t_induction_h:  # still nan
        t_induction_h = maintenance_h  # never reached 50%, set to end of maintenance
    if t_recovery_h != t_recovery_h:  # still nan
        t_recovery_h = t_end_h  # never recovered, set to end

    # Peak fat accumulation
    peak_fat = max(c_fat_list) if c_fat_list else 0.0

    # Fat equilibration at 2h
    # Equilibrium fat concentration = f_inspired * lambda_bg * tissue_kp["fat"]
    c_fat_eq = f_inspired * lambda_bg * tissue_kp["fat"]
    idx_2h = min(int(2.0 / dt_h), len(c_fat_list) - 1)
    c_fat_at_2h = c_fat_list[idx_2h] if idx_2h < len(c_fat_list) else 0.0
    if c_fat_eq > 0.0:
        fat_equil_pct = min(100.0, c_fat_at_2h / c_fat_eq * 100.0)
    else:
        fat_equil_pct = 0.0

    notes = (
        f"Agent: {agent}. "
        f"lambda_blood_gas={lambda_bg}. "
        f"Induction time={t_induction_h:.3f}h, "
        f"Recovery time={t_recovery_h:.3f}h. "
        f"Fat equilibration at 2h: {fat_equil_pct:.1f}%."
    )

    return InhalationAnestheticResult(
        agent=agent,
        lambda_blood_gas=lambda_bg,
        t_induction_h=t_induction_h,
        t_recovery_h=t_recovery_h,
        times_h=times,
        c_vrg=c_vrg_list,
        c_muscle=c_muscle_list,
        c_fat=c_fat_list,
        peak_fat_accumulation=peak_fat,
        fat_equilibration_pct_at_2h=fat_equil_pct,
        notes=notes,
    )


def compare_agents(
    f_inspired: float = 0.05,
    maintenance_h: float = 2.0,
) -> list[InhalationAnestheticResult]:
    """Simulate all four inhalation anesthetic agents and return sorted by recovery time.

    Returns results sorted by t_recovery_h ascending (fastest recovery first).
    """
    results = []
    for agent in _AGENTS:
        result = simulate_inhalation_anesthetic(
            agent=agent,
            f_inspired=f_inspired,
            maintenance_h=maintenance_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.t_recovery_h)
    return results
