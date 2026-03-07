"""Bile acid-drug interactions in the GI tract — Phase 311.

Models interactions between bile acids and drugs:
1. Micellar solubilization of lipophilic drugs in bile acid mixed micelles.
2. Cholestyramine binding (ionic interaction with drugs).
3. Bile acid transporter DDI (ASBT, OATP).
4. Full 1-compartment PK simulation with bile-enhanced absorption.

References
----------
- Lennernas H (2007) Intestinal permeability and its relevance for absorption
  and elimination. Xenobiotica 37:1015-1051.
- Hofmann AF (1999) The continuing importance of bile acids in liver and
  intestinal disease. Arch Intern Med 159:2647-2658.
- Dawson PA (2011) Role of the intestinal bile acid transporters in bile acid
  and drug absorption. Handb Exp Pharmacol 201:169-203.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CholesBoundResult",
    "BileTransporterDDIResult",
    "BileAbsorptionResult",
    "micellar_solubilization_capacity",
    "cholestyramine_binding",
    "bile_acid_transporter_ddi",
    "simulate_bile_acid_effect_on_absorption",
]

# ---------------------------------------------------------------------------
# Result dataclasses (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CholesBoundResult:
    """Result of cholestyramine-drug binding interaction."""

    drug_dose_mg: float
    cholestyramine_dose_g: float
    fraction_bound: float
    dose_available_mg: float
    notes: str


@dataclass(frozen=True)
class BileTransporterDDIResult:
    """Result of bile acid transporter DDI assessment."""

    drug_is_oatp_substrate: bool
    drug_is_asbt_substrate: bool
    bile_level: str
    oatp_inhibition_pct: float
    absorption_reduction_pct: float
    notes: str


@dataclass(frozen=True)
class BileAbsorptionResult:
    """Result of bile-acid-modulated oral PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    meal_state: str
    bile_conc_mM: float
    solubility_enhancement: float
    times_h: list[float]
    c_plasma: list[float]
    cmax: float
    auc: float
    notes: str


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def micellar_solubilization_capacity(
    c_bile_mM: float,
    logP: float,
    mw: float,
) -> float:
    """Compute bile acid micellar solubilization enhancement factor.

    Bile acids form mixed micelles that solubilize lipophilic drugs.
    The total solubility relative to aqueous solubility is given by:

        S_m / S_water = 1 + K_bile * c_bile  (dimensionless fold increase)

    where K_bile is an empirical partition coefficient (L/mol) that increases
    with logP:

        K_bile = 0.1 * exp(0.5 * logP)  [L/mol]

    and c_bile is converted from mM to mol/L before use.

    Parameters
    ----------
    c_bile_mM : float
        Bile acid concentration in the intestinal lumen (mM). Must be >= 0.
    logP : float
        Octanol-water partition coefficient of the drug.
    mw : float
        Molecular weight of the drug (g/mol). Used for contextual scaling
        but does not modify the core equation in this simplified model.

    Returns
    -------
    float
        Solubilization enhancement factor (fold increase; >= 1.0).

    Raises
    ------
    ValueError
        If c_bile_mM < 0.
    """
    if c_bile_mM < 0:
        raise ValueError(f"c_bile_mM must be >= 0, got {c_bile_mM}")

    # Empirical bile acid-lipophilicity micellar partition coefficient (L/mol)
    k_bile_L_per_mol = 0.1 * math.exp(0.5 * logP)

    # Convert c_bile from mM to mol/L
    c_bile_mol_per_L = c_bile_mM * 1e-3

    enhancement = 1.0 + k_bile_L_per_mol * c_bile_mol_per_L
    return enhancement


def cholestyramine_binding(
    drug_dose_mg: float,
    drug_pka: float,
    drug_logP: float,
    cholestyramine_dose_g: float = 4.0,
) -> CholesBoundResult:
    """Model cholestyramine-drug binding in the GI tract.

    Cholestyramine is an anionic bile acid sequestrant resin that binds
    cationic (basic) drugs via ionic interaction. Drug binding capacity
    is ~50 mg drug per gram of cholestyramine for fully ionized drugs,
    adjusted by the actual ionization fraction at intestinal pH = 6.5.

    ionization_factor = 1 / (1 + 10^(pH_gut - pKa))

    fraction_bound = min(1.0, cholestyramine_dose_g * 50 * ionization_factor / drug_dose_mg)

    Parameters
    ----------
    drug_dose_mg : float
        Drug dose in mg. Must be > 0.
    drug_pka : float
        pKa of the drug (basic drugs have pKa > 7).
    drug_logP : float
        logP of the drug (hydrophobic drugs may have additional non-ionic binding).
    cholestyramine_dose_g : float
        Cholestyramine dose in grams (default 4.0 g per sachet).
        Must be >= 0.

    Returns
    -------
    CholesBoundResult

    Raises
    ------
    ValueError
        If drug_dose_mg <= 0 or cholestyramine_dose_g < 0.
    """
    if drug_dose_mg <= 0:
        raise ValueError(f"drug_dose_mg must be > 0, got {drug_dose_mg}")
    if cholestyramine_dose_g < 0:
        raise ValueError(f"cholestyramine_dose_g must be >= 0, got {cholestyramine_dose_g}")

    ph_gut = 6.5

    # Fraction of drug in ionized (cationic) form at intestinal pH
    # For basic drug: ionized fraction = 1 / (1 + 10^(pH - pKa))
    ionization_factor = 1.0 / (1.0 + 10.0 ** (ph_gut - drug_pka))

    # Binding capacity (mg drug bound per g resin), adjusted for ionization
    binding_capacity_mg_per_g = 50.0 * ionization_factor

    # Total drug available to be bound by the resin
    drug_bound_capacity = cholestyramine_dose_g * binding_capacity_mg_per_g

    fraction_bound = min(1.0, drug_bound_capacity / drug_dose_mg)
    dose_available_mg = drug_dose_mg * (1.0 - fraction_bound)

    # Build notes
    notes_parts = [
        f"pKa={drug_pka:.1f}, pH_gut={ph_gut}",
        f"ionization_factor={ionization_factor:.3f}",
        f"binding_capacity={binding_capacity_mg_per_g:.1f} mg/g resin",
        f"fraction_bound={fraction_bound:.3f}",
    ]
    if ionization_factor < 0.1:
        notes_parts.append("Low ionization: minimal cholestyramine binding expected.")
    elif fraction_bound > 0.5:
        notes_parts.append("Significant binding: administer drugs 1h before or 4-6h after resin.")

    return CholesBoundResult(
        drug_dose_mg=drug_dose_mg,
        cholestyramine_dose_g=cholestyramine_dose_g,
        fraction_bound=round(fraction_bound, 6),
        dose_available_mg=round(dose_available_mg, 4),
        notes="; ".join(notes_parts),
    )


def bile_acid_transporter_ddi(
    drug_is_oatp_substrate: bool,
    drug_is_asbt_substrate: bool,
    bile_acid_level: str = "normal",
) -> BileTransporterDDIResult:
    """Assess bile acid transporter-mediated drug-drug interactions.

    Bile acids are endogenous substrates for OATP1B1/1B3 (hepatic uptake)
    and ASBT (ileal absorption). Elevated bile acid concentrations can
    competitively inhibit drug transport.

    - OATP inhibition by bile acids: relevant for OATP substrate drugs
      (e.g., statins, fexofenadine). Elevated bile acids reduce hepatic
      uptake → increased systemic exposure.
    - ASBT competition: if drug is an ASBT substrate and bile acid level
      is elevated, competition reduces drug absorption.

    bile_acid_level:
        "normal"     → 1x concentration baseline
        "after_meal" → 3x (postprandial bile acid surge)
        "cholestasis"→ 10x (pathological elevation)

    Parameters
    ----------
    drug_is_oatp_substrate : bool
        Whether the drug is a substrate of OATP1B1/1B3.
    drug_is_asbt_substrate : bool
        Whether the drug is a substrate of ASBT (ileal bile acid transporter).
    bile_acid_level : str
        Bile acid concentration category. One of "normal", "after_meal",
        "cholestasis".

    Returns
    -------
    BileTransporterDDIResult

    Raises
    ------
    ValueError
        If bile_acid_level is not one of the valid options.
    """
    valid_levels = {"normal", "after_meal", "cholestasis"}
    if bile_acid_level not in valid_levels:
        raise ValueError(f"bile_acid_level must be one of {valid_levels}, got {bile_acid_level!r}")

    # Bile acid level multipliers
    level_multiplier = {"normal": 1.0, "after_meal": 3.0, "cholestasis": 10.0}
    multiplier = level_multiplier[bile_acid_level]

    # OATP inhibition: endogenous bile acids compete with drug uptake
    # Baseline inhibition ~5% at normal; scales with bile acid level
    # Max inhibition ~50% at cholestasis (Ki ~100 µM; normal ~10 µM)
    oatp_inhibition_pct = 0.0
    if drug_is_oatp_substrate:
        # Sigmoid inhibition model: Imax=50%, IC50=3x normal
        imax = 50.0
        ic50_multiplier = 3.0
        oatp_inhibition_pct = imax * multiplier / (ic50_multiplier + multiplier)

    # ASBT absorption reduction: competition at ileal transporter
    absorption_reduction_pct = 0.0
    if drug_is_asbt_substrate and multiplier > 1.0:
        # Linear model: 10% reduction per 3x bile acid increase
        absorption_reduction_pct = min(40.0, (multiplier - 1.0) / 9.0 * 40.0)

    # Build notes
    notes_parts: list[str] = [f"bile_level={bile_acid_level} ({multiplier}x normal)"]
    if drug_is_oatp_substrate:
        notes_parts.append(
            f"OATP substrate: {oatp_inhibition_pct:.1f}% hepatic uptake inhibition"
        )
    if drug_is_asbt_substrate:
        if multiplier > 1.0:
            notes_parts.append(
                f"ASBT substrate: {absorption_reduction_pct:.1f}% absorption reduction"
            )
        else:
            notes_parts.append("ASBT substrate: no inhibition at normal bile acid levels")
    if not drug_is_oatp_substrate and not drug_is_asbt_substrate:
        notes_parts.append("No bile acid transporter interaction expected.")

    return BileTransporterDDIResult(
        drug_is_oatp_substrate=drug_is_oatp_substrate,
        drug_is_asbt_substrate=drug_is_asbt_substrate,
        bile_level=bile_acid_level,
        oatp_inhibition_pct=round(oatp_inhibition_pct, 2),
        absorption_reduction_pct=round(absorption_reduction_pct, 2),
        notes="; ".join(notes_parts),
    )


def simulate_bile_acid_effect_on_absorption(
    drug_name: str,
    dose_mg: float,
    logP: float,
    pka: float,
    solubility_mg_mL: float,
    cl_L_per_h: float,
    vd_L: float,
    meal_state: str = "fed",
) -> BileAbsorptionResult:
    """Simulate 1-compartment oral PK with bile-acid-enhanced solubility.

    Fed state bile acid concentration ~5 mM; fasted ~1 mM. Enhanced solubility
    increases the effective dissolved dose fraction and thereby ka. The
    simulation uses Forward Euler integration over 48 h.

    Absorption rate constant ka is adjusted by the solubility enhancement
    factor (improved dissolution → faster/more complete absorption):
        ka_eff = ka_base * sqrt(solubility_enhancement)

    Bioavailability approximation from BCS perspective:
        F_eff = min(1.0, F_base * solubility_enhancement)

    Simple 1-cpt model:
        dA_gut/dt = -ka * A_gut
        dC/dt    = ka * A_gut / Vd - (CL/Vd) * C

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    dose_mg : float
        Oral dose in mg. Must be > 0.
    logP : float
        Drug octanol-water logP.
    pka : float
        Drug pKa.
    solubility_mg_mL : float
        Intrinsic aqueous solubility (mg/mL). Must be > 0.
    cl_L_per_h : float
        Clearance in L/h. Must be > 0.
    vd_L : float
        Volume of distribution in L. Must be > 0.
    meal_state : str
        "fed" (5 mM bile acids) or "fasted" (1 mM).

    Returns
    -------
    BileAbsorptionResult

    Raises
    ------
    ValueError
        If any required parameter is out of valid range or meal_state invalid.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if meal_state not in {"fed", "fasted"}:
        raise ValueError(f"meal_state must be 'fed' or 'fasted', got {meal_state!r}")

    # Bile acid concentration by meal state
    bile_conc_mM = 5.0 if meal_state == "fed" else 1.0

    # Solubility enhancement factor (dimensionless)
    # mw placeholder 300 g/mol for the formula (mw doesn't affect the result here)
    enhancement = micellar_solubilization_capacity(bile_conc_mM, logP, mw=300.0)

    # Base absorption rate constant (1/h): typical ka = 1.0/h
    ka_base = 1.0  # 1/h
    ka_eff = ka_base * math.sqrt(enhancement)

    # Elimination rate constant
    ke = cl_L_per_h / vd_L  # 1/h

    # Forward Euler simulation
    dt = 0.01  # h
    t_end = 48.0
    n_steps = int(t_end / dt) + 1

    # State: gut amount (mg), plasma concentration (mg/L)
    a_gut = dose_mg
    c_plasma_val = 0.0

    times: list[float] = []
    concentrations: list[float] = []

    record_interval = 10  # record every 10 steps → 0.1h resolution
    for i in range(n_steps):
        t = i * dt
        if i % record_interval == 0:
            times.append(round(t, 4))
            concentrations.append(max(0.0, c_plasma_val))

        # Forward Euler derivatives
        da_gut = -ka_eff * a_gut
        dc_dt = (ka_eff * a_gut) / vd_L - ke * c_plasma_val

        a_gut = max(0.0, a_gut + da_gut * dt)
        c_plasma_val = max(0.0, c_plasma_val + dc_dt * dt)

    # Capture final point
    times.append(round(t_end, 4))
    concentrations.append(max(0.0, c_plasma_val))

    # Pharmacokinetic metrics
    cmax = max(concentrations)
    # Trapezoidal AUC (mg/L * h)
    auc = 0.0
    for j in range(len(times) - 1):
        auc += 0.5 * (concentrations[j] + concentrations[j + 1]) * (times[j + 1] - times[j])

    notes = (
        f"meal_state={meal_state}; bile_conc={bile_conc_mM} mM; "
        f"enhancement={enhancement:.2f}x; ka_eff={ka_eff:.3f}/h; ke={ke:.3f}/h"
    )

    return BileAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        meal_state=meal_state,
        bile_conc_mM=bile_conc_mM,
        solubility_enhancement=round(enhancement, 4),
        times_h=times,
        c_plasma=concentrations,
        cmax=round(cmax, 6),
        auc=round(auc, 4),
        notes=notes,
    )
