"""Gut microbiome drug metabolism model.

Models drug biotransformation by colonic microbiota including:
- Azo-reduction: prodrug activation (e.g., sulfasalazine → 5-ASA + sulfapyridine)
- Beta-glucuronidase: deconjugation of glucuronide metabolites
- Nitro-reduction: nitro-compounds → amine
- Ester hydrolysis: hydrolysis of ester prodrugs

References
----------
- Haiser HJ & Turnbaugh PJ. Science 2012;336:1253-55
- Zimmermann M et al. Nature 2019;570:462-67
- Spanogiannopoulos P et al. Nat Rev Microbiol 2016;14:273-87
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class MicrobiomeMetabResult:
    """Result of microbiome metabolism assessment.

    Attributes
    ----------
    drug_name : Drug name assessed.
    reactions_detected : List of microbial reactions detected from SMILES.
    f_micro : Estimated fraction metabolized by microbiome (0-0.8).
    n_reactions : Number of distinct microbial reactions detected.
    notes : Summary of assessment.
    """

    drug_name: str
    reactions_detected: list[str]
    f_micro: float
    n_reactions: int
    notes: str


@dataclass(frozen=True)
class ColonicPKResult:
    """Result of colonic PK simulation with microbiome metabolism.

    Attributes
    ----------
    drug_name : Drug name.
    dose_mg : Dose administered (mg).
    f_micro : Fraction of dose subject to microbial metabolism.
    times_h : Simulation time points (h).
    c_parent_mg_L : Parent drug plasma concentration (mg/L).
    c_active_mg_L : Active metabolite plasma concentration (mg/L).
    cmax_parent : Peak parent plasma concentration (mg/L).
    cmax_active : Peak active metabolite plasma concentration (mg/L).
    auc_parent : AUC of parent from 0 to t_end (mg*h/L).
    auc_active : AUC of active metabolite from 0 to t_end (mg*h/L).
    colonic_conversion_fraction : Fraction of dose converted in colon.
    notes : Summary of simulation.
    """

    drug_name: str
    dose_mg: float
    f_micro: float
    times_h: list[float]
    c_parent_mg_L: list[float]
    c_active_mg_L: list[float]
    cmax_parent: float
    cmax_active: float
    auc_parent: float
    auc_active: float
    colonic_conversion_fraction: float
    notes: str


def _detect_reactions(smiles: str, mw: float) -> list[str]:
    """Detect susceptible microbial metabolic reactions from SMILES.

    Parameters
    ----------
    smiles : SMILES string of the drug.
    mw : Molecular weight (Da); used to infer glucuronide conjugates.

    Returns
    -------
    List of detected reaction names.
    """
    reactions: list[str] = []

    # Azo-reduction: azo group N=N
    if "N=N" in smiles:
        reactions.append("azo_reduction")

    # Beta-glucuronidase: glucuronide conjugates are large (MW > 400)
    # and contain an ester/carbamate linkage
    if "C(=O)O" in smiles and mw > 400:
        reactions.append("glucuronidase")

    # Nitro-reduction: nitro group
    if "[N+](=O)[O-]" in smiles or "N(=O)=O" in smiles:
        reactions.append("nitro_reduction")

    # Ester hydrolysis: ester group (broad pattern, includes above)
    # Only add separately if glucuronidase not already detected
    if "C(=O)O" in smiles and "glucuronidase" not in reactions:
        reactions.append("ester_hydrolysis")

    return reactions


def assess_microbiome_metabolism(
    drug_name: str,
    smiles: str,
    daily_dose_mg: float,
    route: str = "oral",
    microbiome_density_relative: float = 1.0,
    mw: float = 300.0,
) -> MicrobiomeMetabResult:
    """Assess susceptibility to gut microbiome metabolism.

    Detects functional groups in SMILES and estimates the fraction
    metabolized by colonic microbiota.

    Parameters
    ----------
    drug_name : Name of the drug.
    smiles : SMILES string of the drug molecule.
    daily_dose_mg : Daily dose (mg); used for context.
    route : Administration route ('oral' or 'iv'). Non-oral routes have
        reduced colonic exposure.
    microbiome_density_relative : Relative microbiome density (1.0 = normal,
        <1.0 = dysbiosis/antibiotic-reduced, >1.0 = enriched).
    mw : Molecular weight (Da); used to help detect glucuronide conjugates.
        Default 300.0.

    Returns
    -------
    MicrobiomeMetabResult with detected reactions and estimated f_micro.

    Raises
    ------
    ValueError
        If daily_dose_mg <= 0 or microbiome_density_relative < 0.
    """
    if daily_dose_mg <= 0:
        raise ValueError(f"daily_dose_mg must be > 0, got {daily_dose_mg}")
    if microbiome_density_relative < 0:
        raise ValueError(
            f"microbiome_density_relative must be >= 0, got {microbiome_density_relative}"
        )

    reactions = _detect_reactions(smiles, mw)
    n_reactions = len(reactions)

    # Estimate fraction metabolized by microbiome
    # Non-oral routes have reduced colonic exposure (IV bypasses GI tract)
    route_factor = 1.0 if route.lower() == "oral" else 0.1
    f_micro_raw = 0.2 * n_reactions * microbiome_density_relative * route_factor
    f_micro = min(0.8, f_micro_raw)

    if n_reactions == 0:
        notes = (
            f"{drug_name}: No susceptible microbial functional groups detected. "
            "Low likelihood of microbiome-mediated metabolism."
        )
    else:
        rxn_str = ", ".join(reactions)
        notes = (
            f"{drug_name}: Detected {n_reactions} microbial reaction(s): {rxn_str}. "
            f"Estimated f_micro = {f_micro:.2f} "
            f"(microbiome_density_relative={microbiome_density_relative:.2f})."
        )

    return MicrobiomeMetabResult(
        drug_name=drug_name,
        reactions_detected=reactions,
        f_micro=f_micro,
        n_reactions=n_reactions,
        notes=notes,
    )


def simulate_colonic_pk(
    drug_name: str,
    dose_mg: float,
    f_micro: float,
    k_micro_per_h: float = 0.1,
    cl_parent_L_per_h: float = 5.0,
    cl_active_L_per_h: float = 3.0,
    vd_L: float = 30.0,
    ka_per_h: float = 0.5,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> ColonicPKResult:
    """Simulate plasma PK with colonic microbiome-mediated drug activation.

    Model structure:
    - SI absorption: fraction (1 - f_micro) absorbed quickly from small intestine
    - Colonic depot: fraction f_micro reaches colon where microbiome converts
      parent to active metabolite
    - Active metabolite absorbed from colon (slower ka_colonic)
    - Parent plasma: 1-cpt with systemic CL
    - Active metabolite plasma: 1-cpt with its own CL

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Oral dose (mg).
    f_micro : Fraction of dose subject to microbial activation (0-1).
    k_micro_per_h : Microbial conversion rate constant (h^-1).
    cl_parent_L_per_h : Parent drug systemic clearance (L/h).
    cl_active_L_per_h : Active metabolite systemic clearance (L/h).
    vd_L : Volume of distribution for both species (L).
    ka_per_h : SI absorption rate constant for parent (h^-1).
    t_end_h : Simulation end time (h).
    dt_h : Time step (h).

    Returns
    -------
    ColonicPKResult with time-concentration profiles.

    Raises
    ------
    ValueError
        If parameters violate constraints.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0.0 <= f_micro <= 1.0):
        raise ValueError(f"f_micro must be in [0, 1], got {f_micro}")
    if k_micro_per_h < 0:
        raise ValueError(f"k_micro_per_h must be >= 0, got {k_micro_per_h}")
    if cl_parent_L_per_h <= 0:
        raise ValueError(f"cl_parent_L_per_h must be > 0, got {cl_parent_L_per_h}")
    if cl_active_L_per_h <= 0:
        raise ValueError(f"cl_active_L_per_h must be > 0, got {cl_active_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")

    # Derived rate constants
    ke_parent = cl_parent_L_per_h / vd_L  # elimination rate parent
    ke_active = cl_active_L_per_h / vd_L  # elimination rate active
    # Colonic absorption is slower than SI absorption
    ka_colonic = ka_per_h * 0.3

    # Initial conditions (masses in mg)
    # SI depot: fraction not going to colon
    depot_si = dose_mg * (1.0 - f_micro)
    # Colonic depot: fraction going to colon
    depot_colon = dose_mg * f_micro
    # Plasma concentrations (mg/L)
    cp_parent = 0.0
    cp_active = 0.0

    n_steps = int(t_end_h / dt_h) + 1
    times = np.linspace(0.0, t_end_h, n_steps)

    c_parent_arr = np.zeros(n_steps)
    c_active_arr = np.zeros(n_steps)

    for i, _t in enumerate(times):
        c_parent_arr[i] = cp_parent
        c_active_arr[i] = cp_active

        if i == n_steps - 1:
            break

        # Fluxes
        # SI absorption of parent (fast)
        flux_si_abs = ka_per_h * depot_si  # mg/h

        # Microbial conversion in colon: parent → active metabolite
        flux_micro = k_micro_per_h * depot_colon  # mg/h

        # Colonic absorption of active metabolite into plasma
        # We track converted metabolite in a "colon active" depot
        # For simplicity: converted material immediately available
        # in a colonic active depot, absorbed at ka_colonic
        # We integrate depot_colon_active implicitly:
        # d/dt depot_colon_active = flux_micro - ka_colonic * depot_colon_active
        # But since we update sequentially, use forward Euler accumulation
        # Simplified: active formed in colon is absorbed at ka_colonic from
        # a colonic active depot. We need to track this explicitly.
        # (Integrated as a separate state below)
        flux_parent_elim = ke_parent * cp_parent * vd_L  # mg/h
        flux_active_elim = ke_active * cp_active * vd_L  # mg/h

        # Parent plasma: gains from SI absorption
        dcp_parent = (flux_si_abs - flux_parent_elim) / vd_L
        dcp_active = (0.0 - flux_active_elim) / vd_L  # no SI source for active

        # Update depots and plasma
        depot_si = max(0.0, depot_si - flux_si_abs * dt_h)
        depot_colon = max(0.0, depot_colon - flux_micro * dt_h)

        cp_parent = max(0.0, cp_parent + dcp_parent * dt_h)
        cp_active = max(0.0, cp_active + dcp_active * dt_h)

    # We need to properly track colonic active metabolite absorption.
    # Redo with explicit depot_colon_active state.
    depot_si = dose_mg * (1.0 - f_micro)
    depot_colon = dose_mg * f_micro
    depot_colon_active = 0.0  # active metabolite formed in colon, awaiting absorption
    cp_parent = 0.0
    cp_active = 0.0

    for i, _t in enumerate(times):
        c_parent_arr[i] = cp_parent
        c_active_arr[i] = cp_active

        if i == n_steps - 1:
            break

        flux_si_abs = ka_per_h * depot_si
        flux_micro = k_micro_per_h * depot_colon
        flux_colon_active_abs = ka_colonic * depot_colon_active

        flux_parent_elim = ke_parent * cp_parent * vd_L
        flux_active_elim = ke_active * cp_active * vd_L

        dcp_parent = (flux_si_abs - flux_parent_elim) / vd_L
        dcp_active = (flux_colon_active_abs - flux_active_elim) / vd_L

        depot_si = max(0.0, depot_si - flux_si_abs * dt_h)
        depot_colon = max(0.0, depot_colon - flux_micro * dt_h)
        depot_colon_active = max(
            0.0, depot_colon_active + (flux_micro - flux_colon_active_abs) * dt_h
        )

        cp_parent = max(0.0, cp_parent + dcp_parent * dt_h)
        cp_active = max(0.0, cp_active + dcp_active * dt_h)

    cmax_parent = float(np.max(c_parent_arr))
    cmax_active = float(np.max(c_active_arr))
    auc_parent = float(np_trapz(c_parent_arr, times))
    auc_active = float(np_trapz(c_active_arr, times))

    # Colonic conversion fraction: fraction of dose actually converted
    # = initial colon depot - remaining colon depot (absorbed as active)
    colon_converted = dose_mg * f_micro - max(0.0, depot_colon)
    colonic_conversion_fraction = min(1.0, colon_converted / dose_mg) if dose_mg > 0 else 0.0

    notes = (
        f"{drug_name}: dose={dose_mg}mg, f_micro={f_micro:.2f}, "
        f"k_micro={k_micro_per_h}/h. "
        f"Cmax_parent={cmax_parent:.3f}mg/L, Cmax_active={cmax_active:.3f}mg/L, "
        f"colonic_conversion={colonic_conversion_fraction:.2f}."
    )

    return ColonicPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_micro=f_micro,
        times_h=times.tolist(),
        c_parent_mg_L=c_parent_arr.tolist(),
        c_active_mg_L=c_active_arr.tolist(),
        cmax_parent=cmax_parent,
        cmax_active=cmax_active,
        auc_parent=auc_parent,
        auc_active=auc_active,
        colonic_conversion_fraction=colonic_conversion_fraction,
        notes=notes,
    )


def dysbiosis_impact(
    result_normal: ColonicPKResult,
    dysbiosis_factor: float = 0.3,
) -> ColonicPKResult:
    """Simulate the PK impact of dysbiosis (reduced microbiome activity).

    Re-runs the colonic PK simulation with the microbial conversion rate
    scaled by dysbiosis_factor, representing antibiotic treatment or
    disease-related microbiome depletion.

    Parameters
    ----------
    result_normal : ColonicPKResult from a normal microbiome simulation.
    dysbiosis_factor : Fraction of normal microbial activity retained (0-1).
        Default 0.3 represents severe dysbiosis (~70% reduction).

    Returns
    -------
    ColonicPKResult simulated under dysbiotic conditions.

    Raises
    ------
    ValueError
        If dysbiosis_factor not in (0, 1].
    """
    if not (0.0 < dysbiosis_factor <= 1.0):
        raise ValueError(f"dysbiosis_factor must be in (0, 1], got {dysbiosis_factor}")

    # Extract parameters from original result
    dose_mg = result_normal.dose_mg
    f_micro = result_normal.f_micro
    t_end_h = result_normal.times_h[-1] if result_normal.times_h else 48.0
    dt_h = (
        result_normal.times_h[1] - result_normal.times_h[0]
        if len(result_normal.times_h) > 1
        else 0.1
    )

    # We'll use a default k_micro and scale it
    # Infer approximate k_micro from original simulation context
    # Default: 0.1 /h (the default parameter value)
    k_micro_default = 0.1
    k_micro_dysbiosis = k_micro_default * dysbiosis_factor

    return simulate_colonic_pk(
        drug_name=result_normal.drug_name,
        dose_mg=dose_mg,
        f_micro=f_micro,
        k_micro_per_h=k_micro_dysbiosis,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
