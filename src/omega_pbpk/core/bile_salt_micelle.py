"""Bile salt micelle solubilization model for GI tract drug absorption.

Models how bile salt micelles enhance effective solubility for BCS II/IV drugs
using the Yalkowsky (1999) partition coefficient approach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MicelleResult:
    """Result from bile salt micelle solubility calculation."""

    drug_name: str
    logP: float
    bile_salt_conc_mM: float
    solubility_aqueous_mg_mL: float  # intrinsic aqueous solubility
    solubility_micellar_mg_mL: float  # effective solubility with micelles
    enhancement_ratio: float  # micellar/aqueous
    absorption_fraction_increase_pct: float  # estimated increase in fa
    notes: str


def calculate_micelle_solubility(
    drug_name: str,
    logP: float,
    mw: float,
    solubility_aqueous_mg_mL: float,
    bile_salt_conc_mM: float = 3.0,
    temperature_C: float = 37.0,
) -> MicelleResult:
    """Calculate effective drug solubility with bile salt micelles present.

    Uses Yalkowsky (1999) micellar partition coefficient:
        log(Km) = 0.69 * logP - 0.32
        Km = 10^(0.69*logP - 0.32)

    Volume fraction of micelles:
        phi_m = bile_salt_conc_mM * 0.001 * 0.6   (molar volume ~600 mL/mmol)

    Effective solubility:
        S_eff = S_aq * (1 + Km * phi_m)

    Parameters
    ----------
    drug_name:
        Identifier string for the drug.
    logP:
        Octanol-water partition coefficient (can be any float).
    mw:
        Molecular weight in g/mol (must be > 0).
    solubility_aqueous_mg_mL:
        Intrinsic aqueous solubility in mg/mL (must be > 0).
    bile_salt_conc_mM:
        Bile salt concentration in mM (must be >= 0). Default 3.0 (fasted state).
    temperature_C:
        Temperature in Celsius (informational only, not used in calculation).

    Returns
    -------
    MicelleResult
        Frozen dataclass with all calculated values.
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if solubility_aqueous_mg_mL <= 0:
        raise ValueError(f"solubility_aqueous_mg_mL must be > 0, got {solubility_aqueous_mg_mL}")
    if bile_salt_conc_mM < 0:
        raise ValueError(f"bile_salt_conc_mM must be >= 0, got {bile_salt_conc_mM}")

    # Micellar partition coefficient (Yalkowsky 1999)
    km = 10.0 ** (0.69 * logP - 0.32)

    # Volume fraction of micelles (approximate molar volume = 600 mL/mmol)
    phi_m = bile_salt_conc_mM * 0.001 * 0.6

    # Effective solubility
    s_eff = solubility_aqueous_mg_mL * (1.0 + km * phi_m)

    enhancement_ratio = s_eff / solubility_aqueous_mg_mL

    # Estimate absorption fraction increase: ~30% * log(enhancement_ratio), capped at 50%
    if enhancement_ratio > 1.0:
        fa_increase = min(50.0, 30.0 * math.log(enhancement_ratio))
    else:
        fa_increase = 0.0

    notes = (
        f"{drug_name}: logP={logP:.2f}, Km={km:.3f}, phi_m={phi_m:.4f}, "
        f"T={temperature_C:.1f}C, S_aq={solubility_aqueous_mg_mL:.4f} mg/mL"
    )

    return MicelleResult(
        drug_name=drug_name,
        logP=logP,
        bile_salt_conc_mM=bile_salt_conc_mM,
        solubility_aqueous_mg_mL=solubility_aqueous_mg_mL,
        solubility_micellar_mg_mL=s_eff,
        enhancement_ratio=enhancement_ratio,
        absorption_fraction_increase_pct=fa_increase,
        notes=notes,
    )


def compare_fed_fasted(
    drug_name: str,
    logP: float,
    mw: float,
    solubility_aqueous_mg_mL: float,
) -> dict:
    """Compare bile salt solubilization in fasted vs fed state.

    Fasted state: bile salt concentration = 3.0 mM
    Fed state:    bile salt concentration = 15.0 mM

    Parameters
    ----------
    drug_name:
        Identifier string for the drug.
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight in g/mol (must be > 0).
    solubility_aqueous_mg_mL:
        Intrinsic aqueous solubility in mg/mL (must be > 0).

    Returns
    -------
    dict with keys:
        "fasted" : MicelleResult for fasted state (3.0 mM bile salts)
        "fed"    : MicelleResult for fed state (15.0 mM bile salts)
        "enhancement_ratio_fed_vs_fasted" : ratio of fed to fasted effective solubility
    """
    fasted = calculate_micelle_solubility(
        drug_name=drug_name,
        logP=logP,
        mw=mw,
        solubility_aqueous_mg_mL=solubility_aqueous_mg_mL,
        bile_salt_conc_mM=3.0,
    )
    fed = calculate_micelle_solubility(
        drug_name=drug_name,
        logP=logP,
        mw=mw,
        solubility_aqueous_mg_mL=solubility_aqueous_mg_mL,
        bile_salt_conc_mM=15.0,
    )
    enhancement_ratio_fed_vs_fasted = (
        fed.solubility_micellar_mg_mL / fasted.solubility_micellar_mg_mL
    )

    return {
        "fasted": fasted,
        "fed": fed,
        "enhancement_ratio_fed_vs_fasted": enhancement_ratio_fed_vs_fasted,
    }
