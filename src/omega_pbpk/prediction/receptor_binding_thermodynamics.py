"""Phase 856 — Receptor Binding Thermodynamics.

Compute thermodynamic parameters of receptor-drug binding:
delta_G, delta_H, delta_S from Kd and temperature.
Apply van't Hoff analysis across temperatures.

Pure Python only: stdlib math. No numpy/scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Gas constant J/(mol·K)
R: float = 8.314
# Standard temperature K (25°C)
T_STANDARD: float = 298.15


@dataclass(frozen=True)
class ReceptorBindingThermodynamicsResult:
    """Thermodynamic result of receptor-drug binding analysis."""

    drug_name: str
    receptor_name: str
    kd_M: float
    temperature_K: float
    delta_g_kJ_per_mol: float
    delta_h_kJ_per_mol: float
    delta_s_J_per_mol_K: float
    ligand_efficiency_kJ_per_mol_per_HA: float
    enthalpic_efficiency_kJ_per_mol_per_HA: float
    selectivity_ratio: float
    binding_class: str
    entropy_driven: bool
    notes: str


def _classify_binding(kd_M: float) -> str:
    """Return binding class string based on Kd."""
    if kd_M < 1e-9:
        return "picomolar"
    if kd_M < 1e-6:
        return "nanomolar"
    if kd_M < 1e-3:
        return "micromolar"
    return "millimolar"


def _vant_hoff_fit(kd_at_temperatures: list[tuple[float, float]]) -> tuple[float, float]:
    """Fit delta_H and delta_S from van't Hoff analysis.

    ln(Kd) = delta_H/(R*T) - delta_S/R
    => ln(Kd) = (delta_H/R) * (1/T) - delta_S/R

    Linear regression of ln(Kd) vs 1/T:
      slope = delta_H / R  => delta_H = slope * R
      intercept = -delta_S / R => delta_S = -intercept * R

    For exactly 2 points, uses 2-point formula directly.

    Parameters
    ----------
    kd_at_temperatures : list of (T_K, Kd_M) tuples

    Returns
    -------
    (delta_H_J_per_mol, delta_S_J_per_mol_K)
    """
    n = len(kd_at_temperatures)

    # Build x = 1/T, y = ln(Kd)
    xs = []
    ys = []
    for t_k, kd in kd_at_temperatures:
        xs.append(1.0 / t_k)
        ys.append(math.log(kd))

    if n == 2:
        # 2-point linear: slope = (y2-y1)/(x2-x1)
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        intercept = ys[0] - slope * xs[0]
    else:
        # Ordinary least squares
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        ss_xx = sum((x - mean_x) ** 2 for x in xs)
        ss_xy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
        intercept = mean_y - slope * mean_x

    delta_h_J = slope * R
    delta_s_J = -intercept * R
    return delta_h_J, delta_s_J


def calculate_binding_thermodynamics(
    drug_name: str,
    receptor_name: str,
    kd_M: float,
    n_heavy_atoms: int,
    temperature_K: float = 298.15,
    kd_at_temperatures: list[tuple[float, float]] | None = None,
    reference_kd_M: float | None = None,
) -> ReceptorBindingThermodynamicsResult:
    """Calculate thermodynamic parameters of receptor-drug binding.

    Parameters
    ----------
    drug_name : str
    receptor_name : str
    kd_M : float
        Dissociation constant in molar (must be > 0).
    n_heavy_atoms : int
        Number of heavy atoms in the ligand (must be > 0).
    temperature_K : float
        Temperature in Kelvin (default 298.15 K = 25°C).
    kd_at_temperatures : list of (T_K, Kd_M) tuples, optional
        For van't Hoff analysis; must have >= 2 points to compute delta_H/delta_S.
    reference_kd_M : float, optional
        Reference compound Kd for selectivity ratio calculation.

    Returns
    -------
    ReceptorBindingThermodynamicsResult
    """
    if kd_M <= 0:
        raise ValueError("kd_M must be > 0")
    if n_heavy_atoms <= 0:
        raise ValueError("n_heavy_atoms must be > 0")
    if temperature_K <= 0:
        raise ValueError("temperature_K must be > 0")

    # delta_G = RT * ln(Kd) [J/mol]
    delta_g_J = R * temperature_K * math.log(kd_M)
    delta_g_kJ = delta_g_J / 1000.0

    # Van't Hoff for delta_H and delta_S
    if kd_at_temperatures is not None and len(kd_at_temperatures) >= 2:
        delta_h_J, delta_s_J = _vant_hoff_fit(kd_at_temperatures)
        delta_h_kJ = delta_h_J / 1000.0
        delta_s_J_per_K = delta_s_J
    else:
        # Unknown delta_H; derive delta_S from delta_G = delta_H - T*delta_S
        delta_h_J = 0.0
        delta_h_kJ = 0.0
        # delta_G = -T*delta_S => delta_S = -delta_G / T
        delta_s_J_per_K = -delta_g_J / temperature_K

    # Ligand efficiency: LE = -delta_G / n_heavy_atoms (kJ/mol/HA)
    ligand_efficiency = -delta_g_kJ / n_heavy_atoms

    # Enthalpic efficiency: LE_enthalpic = -delta_H / n_heavy_atoms (kJ/mol/HA)
    enthalpic_efficiency = -delta_h_kJ / n_heavy_atoms

    # Selectivity ratio
    if reference_kd_M is not None:
        selectivity_ratio = reference_kd_M / kd_M
    else:
        selectivity_ratio = 1.0

    # Binding class
    binding_class = _classify_binding(kd_M)

    # Entropy driven: entropic contribution to delta_G
    # delta_G = delta_H - T*delta_S
    # Entropic term = -T * delta_S [J/mol]; sign convention: if -T*delta_S is more negative than delta_H
    t_delta_s_J = temperature_K * delta_s_J_per_K
    entropic_contribution_J = -t_delta_s_J  # this is the -TdS term

    # Entropy driven if entropic contribution (|-TdeltaS|) > 50% of |delta_G|
    if abs(delta_g_J) > 0:
        entropic_fraction = abs(entropic_contribution_J) / abs(delta_g_J)
        entropy_driven = entropic_fraction > 0.5
    else:
        entropy_driven = False

    notes_parts = [
        f"Kd={kd_M:.2e} M ({binding_class})",
        f"delta_G={delta_g_kJ:.2f} kJ/mol",
        f"delta_H={delta_h_kJ:.2f} kJ/mol",
        f"delta_S={delta_s_J_per_K:.2f} J/(mol*K)",
        f"LE={ligand_efficiency:.3f} kJ/mol/HA",
    ]
    if reference_kd_M is not None:
        notes_parts.append(f"selectivity_ratio={selectivity_ratio:.2f}x")
    if entropy_driven:
        notes_parts.append("entropy-driven binding")
    else:
        notes_parts.append("enthalpy-driven binding")

    notes = "; ".join(notes_parts)

    return ReceptorBindingThermodynamicsResult(
        drug_name=drug_name,
        receptor_name=receptor_name,
        kd_M=kd_M,
        temperature_K=temperature_K,
        delta_g_kJ_per_mol=delta_g_kJ,
        delta_h_kJ_per_mol=delta_h_kJ,
        delta_s_J_per_mol_K=delta_s_J_per_K,
        ligand_efficiency_kJ_per_mol_per_HA=ligand_efficiency,
        enthalpic_efficiency_kJ_per_mol_per_HA=enthalpic_efficiency,
        selectivity_ratio=selectivity_ratio,
        binding_class=binding_class,
        entropy_driven=entropy_driven,
        notes=notes,
    )


def screen_binding_affinities(
    receptor_name: str,
    compounds: list[dict],
) -> list[ReceptorBindingThermodynamicsResult]:
    """Screen a list of compounds for binding affinity to a receptor.

    Parameters
    ----------
    receptor_name : str
    compounds : list of dict
        Each dict must have keys: "drug_name" (str), "kd_M" (float), "n_heavy_atoms" (int).

    Returns
    -------
    List of ReceptorBindingThermodynamicsResult sorted by kd_M ascending (tightest first).
    """
    results = []
    for c in compounds:
        result = calculate_binding_thermodynamics(
            drug_name=c["drug_name"],
            receptor_name=receptor_name,
            kd_M=c["kd_M"],
            n_heavy_atoms=c["n_heavy_atoms"],
        )
        results.append(result)

    results.sort(key=lambda r: r.kd_M)
    return results
