"""Phase 375 — Drug Crystallization Thermodynamics.

Models thermodynamics of drug crystallization including supersaturation,
nucleation kinetics, and crystal growth using classical nucleation theory.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CrystallizationThermodynamicsResult:
    """Result of crystallization thermodynamics assessment."""

    drug_name: str
    temperature_K: float
    initial_concentration_mg_mL: float
    equilibrium_solubility_mg_mL: float
    supersaturation_ratio: float
    nucleation_rate_per_s: float
    critical_nucleus_size_molecules: float
    induction_time_s: float
    crystallization_tendency: str
    amorphous_advantage_fold: float
    delta_g_nucleation_kJ_per_mol: float
    notes: str


# Physical constants
_R = 8.314  # J / (mol * K)
_K_B = 1.38e-23  # J / K
_N_A = 6.022e23  # 1 / mol


def _classify_tendency(supersaturation_ratio: float) -> str:
    """Classify crystallization tendency from supersaturation ratio."""
    if supersaturation_ratio < 1.5:
        return "low"
    if supersaturation_ratio < 3.0:
        return "moderate"
    if supersaturation_ratio < 10.0:
        return "high"
    return "very_high"


def assess_crystallization(
    drug_name: str,
    initial_concentration_mg_mL: float,
    equilibrium_solubility_mg_mL: float,
    mw_g_per_mol: float,
    logp: float,
    temperature_C: float = 25.0,
    surface_tension_mN_per_m: float = 30.0,
) -> CrystallizationThermodynamicsResult:
    """Assess crystallization thermodynamics for a drug.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    initial_concentration_mg_mL:
        Actual drug concentration in solution (mg/mL).
    equilibrium_solubility_mg_mL:
        Equilibrium (crystalline) solubility (mg/mL).
    mw_g_per_mol:
        Molecular weight (g/mol).
    logp:
        Octanol-water partition coefficient (log scale).
    temperature_C:
        Temperature in Celsius (default 25 C).
    surface_tension_mN_per_m:
        Crystal-solution surface tension (mN/m), default 30 mN/m.

    Returns
    -------
    CrystallizationThermodynamicsResult
    """
    # --- Validation ---
    if initial_concentration_mg_mL <= 0:
        raise ValueError("initial_concentration_mg_mL must be positive")
    if equilibrium_solubility_mg_mL <= 0:
        raise ValueError("equilibrium_solubility_mg_mL must be positive")
    if mw_g_per_mol <= 0:
        raise ValueError("mw_g_per_mol must be positive")
    if not (-100.0 <= temperature_C <= 200.0):
        raise ValueError("temperature_C must be between -100 and 200")

    T_K = temperature_C + 273.15
    S = initial_concentration_mg_mL / equilibrium_solubility_mg_mL

    notes_parts = []

    if S <= 1.0:
        notes_parts.append("Solution is undersaturated; no nucleation expected.")
        return CrystallizationThermodynamicsResult(
            drug_name=drug_name,
            temperature_K=T_K,
            initial_concentration_mg_mL=initial_concentration_mg_mL,
            equilibrium_solubility_mg_mL=equilibrium_solubility_mg_mL,
            supersaturation_ratio=S,
            nucleation_rate_per_s=0.0,
            critical_nucleus_size_molecules=float("inf"),
            induction_time_s=float("inf"),
            crystallization_tendency="low",
            amorphous_advantage_fold=max(1.0, 10 ** (logp * 0.15)),
            delta_g_nucleation_kJ_per_mol=0.0,
            notes="; ".join(notes_parts),
        )

    # --- Thermodynamic quantities ---
    # Surface tension: gamma in J/m^2
    gamma = surface_tension_mN_per_m * 1e-3  # N/m = J/m^2

    # Molar volume approximation (density ~ 1.2 g/cm^3)
    Vm_cm3 = mw_g_per_mol / 1.2
    Vm_m3 = Vm_cm3 * 1e-6  # cm^3 -> m^3

    # Molecular volume
    v_m3 = Vm_m3 / _N_A

    # log(S) for nucleation expressions — guard against S very close to 1
    ln_S = math.log(max(S, 1.0001))

    # Critical nucleus radius (m)
    r_star = 2.0 * gamma * v_m3 / (_K_B * T_K * ln_S)
    r_star = max(r_star, 1e-10)

    # Critical nucleus size (number of molecules)
    n_star = (4.0 / 3.0 * math.pi * r_star**3) / v_m3

    # Free energy barrier (J)
    delta_G_star = 16.0 * math.pi * gamma**3 * v_m3**2 / (3.0 * (_K_B * T_K * ln_S) ** 2)

    # Nucleation rate (classical, pre-factor 1e30)
    exponent = -delta_G_star / (_K_B * T_K)
    # Clamp to avoid overflow / underflow
    exponent = max(-690.0, min(0.0, exponent))
    nucleation_rate = 1e30 * math.exp(exponent)
    nucleation_rate = max(1e-300, min(1e30, nucleation_rate))

    # Induction time
    induction_time_s = 1.0 / max(nucleation_rate, 1e-300)

    # Crystallization tendency
    tendency = _classify_tendency(S)

    # Amorphous advantage (logP-based estimate)
    amorphous_advantage_fold = max(1.0, 10 ** (logp * 0.15))

    # Free energy barrier in kJ/mol
    delta_g_nucleation_kJ_per_mol = delta_G_star * _N_A / 1000.0

    # Build notes
    if S >= 10.0:
        notes_parts.append("Very high supersaturation; rapid nucleation likely.")
    elif S >= 3.0:
        notes_parts.append("High supersaturation; consider stabilizing excipients.")
    elif S >= 1.5:
        notes_parts.append("Moderate supersaturation; monitor for crystal growth.")
    else:
        notes_parts.append("Low supersaturation; crystallization unlikely in short term.")

    if logp > 3.0:
        notes_parts.append(
            "Lipophilic drug; amorphous form may offer significant solubility advantage."
        )

    return CrystallizationThermodynamicsResult(
        drug_name=drug_name,
        temperature_K=T_K,
        initial_concentration_mg_mL=initial_concentration_mg_mL,
        equilibrium_solubility_mg_mL=equilibrium_solubility_mg_mL,
        supersaturation_ratio=S,
        nucleation_rate_per_s=nucleation_rate,
        critical_nucleus_size_molecules=n_star,
        induction_time_s=induction_time_s,
        crystallization_tendency=tendency,
        amorphous_advantage_fold=amorphous_advantage_fold,
        delta_g_nucleation_kJ_per_mol=delta_g_nucleation_kJ_per_mol,
        notes="; ".join(notes_parts) if notes_parts else "Assessment complete.",
    )


def compare_temperatures(
    drug_name: str,
    initial_concentration_mg_mL: float,
    equilibrium_solubility_mg_mL: float,
    mw_g_per_mol: float,
    logp: float,
    temperatures_C: list[float] | None = None,
) -> list[CrystallizationThermodynamicsResult]:
    """Compare crystallization thermodynamics across multiple temperatures.

    Parameters
    ----------
    drug_name:
        Drug compound name.
    initial_concentration_mg_mL:
        Drug concentration in solution (mg/mL).
    equilibrium_solubility_mg_mL:
        Equilibrium solubility (mg/mL).
    mw_g_per_mol:
        Molecular weight (g/mol).
    logp:
        Octanol-water partition coefficient.
    temperatures_C:
        List of temperatures in Celsius. Defaults to [4.0, 25.0, 37.0, 60.0].

    Returns
    -------
    List of CrystallizationThermodynamicsResult sorted by nucleation_rate ascending.
    """
    if temperatures_C is None:
        temperatures_C = [4.0, 25.0, 37.0, 60.0]

    results = [
        assess_crystallization(
            drug_name=drug_name,
            initial_concentration_mg_mL=initial_concentration_mg_mL,
            equilibrium_solubility_mg_mL=equilibrium_solubility_mg_mL,
            mw_g_per_mol=mw_g_per_mol,
            logp=logp,
            temperature_C=t,
        )
        for t in temperatures_C
    ]

    return sorted(results, key=lambda r: r.nucleation_rate_per_s)
