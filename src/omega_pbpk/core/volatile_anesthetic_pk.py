"""
Phase 411 — Volatile Anesthetic PK/PD Model

Lowe-Ernst / Eger first-order FA/FI equilibration model for inhaled
volatile anesthetics.  Pure Python, stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Agent physico-chemical properties
# ---------------------------------------------------------------------------
_AGENT_PROPS = {
    "sevoflurane": {"blood_gas": 0.65, "mac_pct": 2.0, "oil_gas": 47},
    "isoflurane": {"blood_gas": 1.4, "mac_pct": 1.15, "oil_gas": 98},
    "desflurane": {"blood_gas": 0.45, "mac_pct": 6.0, "oil_gas": 18},
    "halothane": {"blood_gas": 2.54, "mac_pct": 0.77, "oil_gas": 224},
}

_VALID_AGENTS = set(_AGENT_PROPS.keys())

# Functional residual capacity (L)
_FRC_L = 2.5


# ---------------------------------------------------------------------------
# Result dataclass  (plain — list fields present)
# ---------------------------------------------------------------------------
@dataclass
class VolatileAnestheticResult:
    drug_name: str
    agent: str
    inspired_conc_pct: float
    times_min: list
    fa_fi_ratio: list  # alveolar/inspired concentration ratio (0-1)
    c_blood_mL_per_100mL: list  # arterial blood concentration
    c_brain_mL_per_100mL: list  # brain/effect-site concentration
    mac_fraction: list  # multiples of MAC achieved
    time_to_50pct_mac_min: float
    time_to_induction_min: float  # time to reach 1.0 MAC
    time_to_awakening_min: float  # time for FA to drop to 0.3 MAC after discontinuation
    blood_gas_coefficient: float
    mac_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate_volatile_anesthetic(
    drug_name: str,
    agent: str,
    inspired_conc_pct: float,
    fresh_gas_flow_L_per_min: float = 4.0,
    alveolar_ventilation_L_per_min: float = 4.0,
    cardiac_output_L_per_min: float = 5.0,
    t_induction_min: float = 30.0,
    t_end_min: float = 60.0,
    dt_min: float = 0.1,
) -> VolatileAnestheticResult:
    """Simulate volatile anesthetic wash-in and wash-out.

    Parameters
    ----------
    drug_name : str
        Label for the drug/simulation.
    agent : str
        One of "sevoflurane", "isoflurane", "desflurane", "halothane".
    inspired_conc_pct : float
        Inspired concentration in % (e.g. 2.0 for 2 %).
    fresh_gas_flow_L_per_min : float
        Fresh gas flow (L/min).
    alveolar_ventilation_L_per_min : float
        Effective alveolar ventilation (L/min).
    cardiac_output_L_per_min : float
        Cardiac output (L/min).
    t_induction_min : float
        Duration of administration before discontinuation (min).
    t_end_min : float
        Total simulation length (min).
    dt_min : float
        Forward-Euler time step (min).

    Returns
    -------
    VolatileAnestheticResult
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if inspired_conc_pct <= 0:
        raise ValueError("inspired_conc_pct must be > 0")
    if agent not in _VALID_AGENTS:
        raise ValueError(f"agent must be one of {sorted(_VALID_AGENTS)}, got '{agent}'")
    if alveolar_ventilation_L_per_min <= 0:
        raise ValueError("alveolar_ventilation_L_per_min must be > 0")
    if cardiac_output_L_per_min <= 0:
        raise ValueError("cardiac_output_L_per_min must be > 0")
    if t_end_min <= 0:
        raise ValueError("t_end_min must be > 0")
    if dt_min <= 0:
        raise ValueError("dt_min must be > 0")
    if t_induction_min < 0:
        raise ValueError("t_induction_min must be >= 0")

    props = _AGENT_PROPS[agent]
    blood_gas = props["blood_gas"]
    mac_pct = props["mac_pct"]

    # Effective distribution volume for anesthetic uptake
    # VD_eff scales with cardiac output relative to reference 5 L/min
    VD_eff = 10.0 * (cardiac_output_L_per_min / 5.0)  # L

    # Time constant (min) for FA equilibration
    tau = (_FRC_L + blood_gas * VD_eff) / alveolar_ventilation_L_per_min

    # Brain equilibration is slower than alveolar
    tau_brain = 2.0 * tau

    # Inspired concentration as fraction (0-1) — FI
    FI = inspired_conc_pct / 100.0

    # ------------------------------------------------------------------
    # Forward Euler integration
    # ------------------------------------------------------------------
    FA = 0.0  # alveolar fraction
    c_brain = 0.0  # brain concentration (mL/100 mL)

    times_min: list[float] = []
    fa_fi_ratio: list[float] = []
    c_blood_list: list[float] = []
    c_brain_list: list[float] = []
    mac_fraction_list: list[float] = []

    time_to_50pct_mac_min = -1.0
    time_to_induction_min_ = -1.0

    n_steps = int(t_end_min / dt_min) + 1
    t = 0.0

    for _step in range(n_steps):
        # Record state
        times_min.append(t)

        fi_ratio = FA / FI if FI > 0 else 0.0
        fa_fi_ratio.append(min(1.0, max(0.0, fi_ratio)))

        c_blood = FA * blood_gas * 10.0  # mL / 100 mL
        c_blood_list.append(c_blood)
        c_brain_list.append(c_brain)

        mf = (FA * 100.0) / mac_pct  # FA as % / MAC%
        mac_fraction_list.append(mf)

        # Milestone detection (first crossing)
        if time_to_50pct_mac_min < 0 and mf >= 0.5:
            time_to_50pct_mac_min = t
        if time_to_induction_min_ < 0 and mf >= 1.0:
            time_to_induction_min_ = t

        # ------------------------------------------------------------------
        # ODE  — Forward Euler
        # ------------------------------------------------------------------
        if t < t_induction_min:
            # Wash-in: FI drives FA toward FI
            dFA_dt = (FI - FA) / tau
        else:
            # Wash-out: no inspired gas, FA decays
            dFA_dt = -FA / tau

        c_blood_now = FA * blood_gas * 10.0
        dc_brain_dt = (c_blood_now - c_brain) / tau_brain

        FA = max(0.0, FA + dFA_dt * dt_min)
        c_brain = max(0.0, c_brain + dc_brain_dt * dt_min)

        t += dt_min

    # ------------------------------------------------------------------
    # Awakening: first time (post-discontinuation) FA <= 0.3*MAC
    # ------------------------------------------------------------------
    threshold_FA = 0.3 * mac_pct / 100.0  # FA fraction corresponding to 0.3 MAC

    time_to_awakening_min = -1.0
    for i, t_i in enumerate(times_min):
        if t_i >= t_induction_min:
            fa_val = fa_fi_ratio[i] * FI  # recover FA from ratio
            if fa_val <= threshold_FA:
                time_to_awakening_min = t_i - t_induction_min
                break

    # If still negative (never reached milestone), report end of simulation
    if time_to_50pct_mac_min < 0:
        time_to_50pct_mac_min = t_end_min
    if time_to_induction_min_ < 0:
        time_to_induction_min_ = t_end_min
    if time_to_awakening_min < 0:
        time_to_awakening_min = t_end_min - t_induction_min

    notes = (
        f"Lowe-Ernst first-order model; tau={tau:.2f} min; "
        f"blood:gas={blood_gas}; MAC={mac_pct}%; "
        f"VD_eff={VD_eff:.1f} L"
    )

    return VolatileAnestheticResult(
        drug_name=drug_name,
        agent=agent,
        inspired_conc_pct=inspired_conc_pct,
        times_min=times_min,
        fa_fi_ratio=fa_fi_ratio,
        c_blood_mL_per_100mL=c_blood_list,
        c_brain_mL_per_100mL=c_brain_list,
        mac_fraction=mac_fraction_list,
        time_to_50pct_mac_min=time_to_50pct_mac_min,
        time_to_induction_min=time_to_induction_min_,
        time_to_awakening_min=time_to_awakening_min,
        blood_gas_coefficient=blood_gas,
        mac_pct=mac_pct,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------
def compare_agents(
    inspired_conc_pct: float = 2.0,
    t_induction_min: float = 30.0,
) -> list:
    """Compare all four volatile agents at the same inspired concentration.

    Returns a list of VolatileAnestheticResult sorted by
    time_to_induction_min ascending (fastest induction first).
    """
    results = []
    for agent_name in _VALID_AGENTS:
        r = simulate_volatile_anesthetic(
            drug_name=agent_name,
            agent=agent_name,
            inspired_conc_pct=inspired_conc_pct,
            t_induction_min=t_induction_min,
            t_end_min=t_induction_min + 30.0,
        )
        results.append(r)

    results.sort(key=lambda r: r.time_to_induction_min)
    return results
