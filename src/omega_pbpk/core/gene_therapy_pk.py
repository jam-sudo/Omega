"""Gene therapy PK/PD modeling — AAV vector distribution and transgene expression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AAVPKResult:
    """Result of AAV gene therapy PK/PD simulation."""

    drug_name: str
    dose_vg_per_kg: float
    bw_kg: float
    times_days: list[float]
    v_blood: list[float]  # vector genome copies/mL in blood
    v_liver_copies: list[float]  # transduced genome copies in liver
    protein_expression: list[float]  # therapeutic protein (arbitrary units)
    peak_protein: float
    t_peak_protein_days: float
    protein_at_1year: float  # protein at day 365 or t_end
    t_half_blood_days: float  # vector clearance half-life in blood (days)
    liver_transduction_efficiency: float  # v_liver_max / dose_total (fraction)
    notes: str


def simulate_aav_pk(
    drug_name: str,
    dose_vg_per_kg: float,
    bw_kg: float = 70.0,
    k_blood_clear_per_h: float = 0.5,
    f_liver_tropic: float = 0.7,
    k_transduce_per_h: float = 0.1,
    k_express_per_day: float = 100.0,
    k_protein_deg_per_day: float = 0.2,
    liver_volume_L: float = 1.7,
    t_end_days: float = 365.0,
    dt_days: float = 0.1,
) -> AAVPKResult:
    """Simulate AAV gene therapy vector PK and transgene expression.

    A simplified 3-state model:
    - V_blood: vector genome copies/mL in blood — cleared by immune/non-specific
      clearance and hepatic transduction
    - V_liver: transduced genome copies in liver — produced from blood, very slow
      loss (episomal dilution; hepatocytes rarely divide)
    - Protein: therapeutic protein (AU) — synthesized from V_liver, degraded

    Parameters
    ----------
    drug_name : str
        Name of the gene therapy product.
    dose_vg_per_kg : float
        Dose in vector genomes per kg body weight (vg/kg). Typical: 1e13-1e14 vg/kg.
    bw_kg : float
        Body weight (kg).
    k_blood_clear_per_h : float
        First-order blood clearance rate constant (1/h) — immune + non-specific.
    f_liver_tropic : float
        Fraction of dose that will ultimately transduce liver (0-1).
    k_transduce_per_h : float
        Rate of transduction from blood to liver (1/h).
    k_express_per_day : float
        Transgene expression rate constant (protein AU per copy per day).
    k_protein_deg_per_day : float
        Protein degradation rate constant (1/day).
    liver_volume_L : float
        Liver volume (L), used to convert blood pool to liver copies.
    t_end_days : float
        Simulation duration (days).
    dt_days : float
        Time step (days).

    Returns
    -------
    AAVPKResult
    """
    if dose_vg_per_kg <= 0:
        raise ValueError("dose_vg_per_kg must be > 0")
    if bw_kg <= 0:
        raise ValueError("bw_kg must be > 0")
    if k_blood_clear_per_h <= 0:
        raise ValueError("k_blood_clear_per_h must be > 0")
    if k_transduce_per_h <= 0:
        raise ValueError("k_transduce_per_h must be > 0")
    if k_express_per_day <= 0:
        raise ValueError("k_express_per_day must be > 0")
    if k_protein_deg_per_day <= 0:
        raise ValueError("k_protein_deg_per_day must be > 0")
    if t_end_days <= 0:
        raise ValueError("t_end_days must be > 0")
    if not (0.0 < f_liver_tropic <= 1.0):
        raise ValueError("f_liver_tropic must be in (0, 1]")

    # Total dose in vector genomes
    dose_total_vg = dose_vg_per_kg * bw_kg

    # Blood volume approximation (mL): ~70 mL/kg
    blood_volume_mL = 70.0 * bw_kg

    # Initial concentration in blood (vg/mL)
    v_blood_0 = dose_total_vg / blood_volume_mL

    # Convert rate constants: k_blood_clear and k_transduce from per_h to per_day
    k_clear_per_day = k_blood_clear_per_h * 24.0
    k_trans_per_day = k_transduce_per_h * 24.0

    # Effective transduction rate from blood (scaled by liver tropism)
    k_trans_effective = k_trans_per_day * f_liver_tropic

    # Total blood elimination = clearance + transduction
    k_blood_total = k_clear_per_day + k_trans_effective

    # Episomal dilution rate (very slow; hepatocytes divide ~once per year)
    # Half-life ~1 year → k_dilution ~ ln(2)/365 per day
    k_episomal_dilution = np.log(2.0) / 365.0  # per day

    # Time array
    n_steps = max(int(t_end_days / dt_days), 1)
    times = np.linspace(0.0, t_end_days, n_steps + 1)

    # State arrays
    v_blood = np.zeros(n_steps + 1)
    v_liver = np.zeros(n_steps + 1)
    protein = np.zeros(n_steps + 1)

    # Initial conditions
    v_blood[0] = v_blood_0
    v_liver[0] = 0.0
    protein[0] = 0.0

    # Forward Euler integration
    for i in range(n_steps):
        # Flux: blood → liver transduction (copies entering liver per mL per day * volume)
        transduction_flux = k_trans_effective * v_blood[i] * blood_volume_mL

        # dV_blood/dt = -k_blood_total * V_blood  (copies/mL)
        dv_blood = -k_blood_total * v_blood[i]

        # dV_liver/dt = transduction_flux - k_episomal_dilution * V_liver
        # (V_liver in total copies in liver)
        dv_liver = transduction_flux - k_episomal_dilution * v_liver[i]

        # Protein production proportional to transduced copies in liver
        # dProtein/dt = k_express * V_liver - k_deg * Protein
        dprot = k_express_per_day * v_liver[i] - k_protein_deg_per_day * protein[i]

        v_blood[i + 1] = max(v_blood[i] + dv_blood * dt_days, 0.0)
        v_liver[i + 1] = max(v_liver[i] + dv_liver * dt_days, 0.0)
        protein[i + 1] = max(protein[i] + dprot * dt_days, 0.0)

    # Derived metrics
    peak_protein = float(np.max(protein))
    t_peak_idx = int(np.argmax(protein))
    t_peak_protein_days = float(times[t_peak_idx])

    # protein_at_1year: value at day 365 or last time point
    target_day = min(365.0, t_end_days)
    idx_1year = int(np.argmin(np.abs(times - target_day)))
    protein_at_1year = float(protein[idx_1year])

    # Blood half-life from exponential decay: t½ = ln(2) / k_blood_total (days)
    t_half_blood_days = np.log(2.0) / k_blood_total if k_blood_total > 0 else float("inf")

    # Liver transduction efficiency: max copies in liver / total dose
    v_liver_max = float(np.max(v_liver))
    liver_transduction_efficiency = float(np.clip(v_liver_max / dose_total_vg, 0.0, 1.0))

    notes = (
        f"AAV PK simulation for {drug_name}. "
        f"Total dose: {dose_total_vg:.2e} vg. "
        f"Blood t½: {t_half_blood_days * 24:.1f} h. "
        f"Peak protein at day {t_peak_protein_days:.1f}."
    )

    return AAVPKResult(
        drug_name=drug_name,
        dose_vg_per_kg=dose_vg_per_kg,
        bw_kg=bw_kg,
        times_days=times.tolist(),
        v_blood=v_blood.tolist(),
        v_liver_copies=v_liver.tolist(),
        protein_expression=protein.tolist(),
        peak_protein=peak_protein,
        t_peak_protein_days=t_peak_protein_days,
        protein_at_1year=protein_at_1year,
        t_half_blood_days=float(t_half_blood_days),
        liver_transduction_efficiency=liver_transduction_efficiency,
        notes=notes,
    )


def compare_doses(
    drug_name: str,
    doses_vg_per_kg: list[float],
    **kwargs,
) -> list[AAVPKResult]:
    """Simulate AAV PK for multiple doses and return sorted by peak_protein descending.

    Parameters
    ----------
    drug_name : str
        Gene therapy product name.
    doses_vg_per_kg : list[float]
        List of doses to compare (vg/kg).
    **kwargs
        Additional keyword arguments passed to simulate_aav_pk.

    Returns
    -------
    list[AAVPKResult]
        Results sorted by peak_protein descending (highest efficacy first).
    """
    results = [simulate_aav_pk(drug_name, dose, **kwargs) for dose in doses_vg_per_kg]
    return sorted(results, key=lambda r: r.peak_protein, reverse=True)


__all__ = [
    "AAVPKResult",
    "simulate_aav_pk",
    "compare_doses",
]
