"""Nanoparticle protein adsorption model (Langmuir corona formation).

Models the equilibrium adsorption of proteins onto nanoparticle surfaces using
the Langmuir adsorption isotherm.  The protein corona thickness and stability
are estimated from the equilibrium surface coverage.

References
----------
- Langmuir I, J Am Chem Soc. 1918;40(9):1361-1403 (adsorption isotherm)
- Monopoli MP et al., Nat Nanotechnol. 2012;7(12):779-786 (protein corona)
- Walczyk D et al., J Am Chem Soc. 2010;132(16):5761-5768 (corona dynamics)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NanoProteinAdsorptionResult:
    """Result of Langmuir protein adsorption onto a nanoparticle surface.

    Attributes
    ----------
    nanoparticle_name : Nanoparticle identifier.
    protein_name : Protein identifier.
    surface_area_nm2 : Total nanoparticle surface area (nm^2).
    ka_L_per_mol : Association rate constant (L/mol).
    kd_mol_per_L : Dissociation constant Kd (mol/L).
    max_adsorption_mol_per_nm2 : Langmuir Gamma_max (mol/nm^2).
    equilibrium_coverage : Fractional surface coverage theta [0-1].
    adsorbed_molecules_per_np : Number of adsorbed protein molecules per NP.
    corona_thickness_nm : Estimated corona shell thickness (nm).
    binding_energy_kJ_per_mol : Gibbs free energy of binding (kJ/mol).
    desorption_rate_per_h : Effective first-order desorption rate (h^-1).
    stability_classification : "stable", "moderate", or "unstable".
    notes : Summary text.
    """

    nanoparticle_name: str
    protein_name: str
    surface_area_nm2: float
    ka_L_per_mol: float
    kd_mol_per_L: float
    max_adsorption_mol_per_nm2: float
    equilibrium_coverage: float
    adsorbed_molecules_per_np: float
    corona_thickness_nm: float
    binding_energy_kJ_per_mol: float
    desorption_rate_per_h: float
    stability_classification: str
    notes: str


def predict_adsorption(
    nanoparticle_name: str,
    protein_name: str,
    nanoparticle_diameter_nm: float,
    protein_conc_mol_per_L: float,
    ka_L_per_mol: float,
    kd_mol_per_L: float,
    protein_mw_kDa: float,
    protein_diameter_nm: float,
) -> NanoProteinAdsorptionResult:
    """Predict Langmuir equilibrium protein adsorption on a nanoparticle.

    Parameters
    ----------
    nanoparticle_name:
        Identifier for the nanoparticle formulation.
    protein_name:
        Identifier for the adsorbing protein.
    nanoparticle_diameter_nm:
        Hydrodynamic diameter of the nanoparticle (nm).  Must be > 0.
    protein_conc_mol_per_L:
        Bulk protein concentration (mol/L).  Must be >= 0.
    ka_L_per_mol:
        Association constant (L/mol).  Must be > 0.
    kd_mol_per_L:
        Dissociation constant Kd (mol/L).  Must be > 0.
    protein_mw_kDa:
        Molecular weight of the protein (kDa).
    protein_diameter_nm:
        Effective diameter of one protein molecule (nm), used for close-packing.

    Returns
    -------
    NanoProteinAdsorptionResult
    """
    if nanoparticle_diameter_nm <= 0:
        raise ValueError("nanoparticle_diameter_nm must be > 0")
    if ka_L_per_mol <= 0:
        raise ValueError("ka_L_per_mol must be > 0")
    if kd_mol_per_L <= 0:
        raise ValueError("kd_mol_per_L must be > 0")
    if protein_conc_mol_per_L < 0:
        raise ValueError("protein_conc_mol_per_L must be >= 0")
    if protein_mw_kDa <= 0:
        raise ValueError("protein_mw_kDa must be > 0")
    if protein_diameter_nm <= 0:
        raise ValueError("protein_diameter_nm must be > 0")

    # Surface area of spherical nanoparticle (nm^2)
    radius_nm = nanoparticle_diameter_nm / 2.0
    surface_area_nm2 = 4.0 * math.pi * radius_nm**2

    # Langmuir equilibrium coverage: theta = C/Kd / (1 + C/Kd)
    c_over_kd = protein_conc_mol_per_L / kd_mol_per_L
    theta = c_over_kd / (1.0 + c_over_kd)

    # Maximum adsorption: monolayer close-packing (proteins modeled as disks)
    protein_radius_nm = protein_diameter_nm / 2.0
    gamma_max_per_nm2 = 1.0 / (math.pi * protein_radius_nm**2)  # mol/nm^2 equivalent

    # Adsorbed molecules per nanoparticle
    adsorbed_molecules = theta * gamma_max_per_nm2 * surface_area_nm2

    # Corona thickness (simplified: protein diameter scaled by coverage)
    corona_thickness_nm = protein_diameter_nm * theta

    # Binding energy: dG = -RT * ln(ka / kd)
    # R = 8.314e-3 kJ/mol/K, T = 310 K (physiological)
    R_kJ = 8.314e-3  # kJ/mol/K
    T_K = 310.0
    # ka has units L/mol; kd has units mol/L — ratio ka/kd is dimensionless
    # (association equilibrium constant Ka = ka/kd or 1/Kd)
    # Here ka is in L/mol and kd is in mol/L, so Ka = ka (treated as 1/Kd for thermodynamics)
    # binding_energy = -R*T * ln(Ka) where Ka = 1/kd in L/mol equivalent
    ka_over_kd = ka_L_per_mol / kd_mol_per_L
    if ka_over_kd > 0:
        binding_energy_kJ_per_mol = -R_kJ * T_K * math.log(ka_over_kd)
    else:
        binding_energy_kJ_per_mol = 0.0

    # Desorption rate: koff = kon * Kd (s^-1), then convert to h^-1
    # ka is association constant (L/mol), not kon (L/mol/s);
    # treat koff = ka * kd as a pseudo-rate in h^-1 (per spec)
    koff = ka_L_per_mol * kd_mol_per_L
    desorption_rate_per_h = koff * 3600.0

    # Stability classification based on equilibrium coverage
    if theta > 0.7:
        stability = "stable"
    elif theta > 0.3:
        stability = "moderate"
    else:
        stability = "unstable"

    notes = (
        f"{protein_name} adsorption on {nanoparticle_name}: "
        f"theta={theta:.3f}, corona={corona_thickness_nm:.2f} nm, "
        f"dG={binding_energy_kJ_per_mol:.1f} kJ/mol, {stability}"
    )

    return NanoProteinAdsorptionResult(
        nanoparticle_name=nanoparticle_name,
        protein_name=protein_name,
        surface_area_nm2=surface_area_nm2,
        ka_L_per_mol=ka_L_per_mol,
        kd_mol_per_L=kd_mol_per_L,
        max_adsorption_mol_per_nm2=gamma_max_per_nm2,
        equilibrium_coverage=theta,
        adsorbed_molecules_per_np=adsorbed_molecules,
        corona_thickness_nm=corona_thickness_nm,
        binding_energy_kJ_per_mol=binding_energy_kJ_per_mol,
        desorption_rate_per_h=desorption_rate_per_h,
        stability_classification=stability,
        notes=notes,
    )


def screen_proteins(
    nanoparticle_name: str,
    nanoparticle_diameter_nm: float,
    protein_library: list[dict[str, Any]],
) -> list[NanoProteinAdsorptionResult]:
    """Screen a library of proteins for adsorption onto a nanoparticle.

    Parameters
    ----------
    nanoparticle_name:
        Identifier for the nanoparticle formulation.
    nanoparticle_diameter_nm:
        Hydrodynamic diameter of the nanoparticle (nm).
    protein_library:
        List of protein parameter dicts.  Each dict must contain keys:
        ``name``, ``conc_mol_per_L``, ``ka_L_per_mol``, ``kd_mol_per_L``,
        ``mw_kDa``, ``diameter_nm``.

    Returns
    -------
    list[NanoProteinAdsorptionResult]
        Sorted descending by equilibrium_coverage (strongest binders first).
    """
    results: list[NanoProteinAdsorptionResult] = []
    for entry in protein_library:
        result = predict_adsorption(
            nanoparticle_name=nanoparticle_name,
            protein_name=entry["name"],
            nanoparticle_diameter_nm=nanoparticle_diameter_nm,
            protein_conc_mol_per_L=entry["conc_mol_per_L"],
            ka_L_per_mol=entry["ka_L_per_mol"],
            kd_mol_per_L=entry["kd_mol_per_L"],
            protein_mw_kDa=entry["mw_kDa"],
            protein_diameter_nm=entry["diameter_nm"],
        )
        results.append(result)

    results.sort(key=lambda r: r.equilibrium_coverage, reverse=True)
    return results
