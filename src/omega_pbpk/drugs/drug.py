"""Drug dataclass — central compound parameter container.

All parameters use internal standard units:
  Concentration: mg/L, Volume: L, Flow: L/h, Time: h, Clearance: L/h, Dose: mg
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# IVIVE (In Vitro -> In Vivo Extrapolation) physical constants
# Units encoded in names to prevent silent conversion errors.
# Reference: Houston 1994; Davies & Morris 1993 (70 kg reference human)
# ---------------------------------------------------------------------------
_CYP_CONTENT_pmol_per_mg: float = 40.0  # pmol CYP per mg microsomal protein
_MPPGL_mg_per_g: float = 45.0  # mg microsomal protein per g liver (MPPGL)
_LIVER_WEIGHT_g: float = 1800.0  # g (70 kg adult reference liver)
_GUT_WEIGHT_g: float = 18.0  # g CYP3A4-rich intestinal epithelium (~1% liver)
_uL_PER_L: float = 1e6  # uL -> L unit conversion
_MIN_PER_H: float = 60.0  # min -> h unit conversion

# Derived IVIVE scaling factors
# Usage: CLint [uL/min/pmol] x _IVIVE_HEPATIC -> whole-liver CLint [L/h]
_IVIVE_HEPATIC: float = (
    _CYP_CONTENT_pmol_per_mg * _MPPGL_mg_per_g * _LIVER_WEIGHT_g / _uL_PER_L / _MIN_PER_H
)  # = 40 x 45 x 1800 / 1e6 / 60 = 0.054

# Usage: CYP3A4_CLint [uL/min/pmol] x gut_multiplier x _IVIVE_GUT -> gut-wall CLint [L/h]
_IVIVE_GUT: float = (
    _CYP_CONTENT_pmol_per_mg * _MPPGL_mg_per_g * _GUT_WEIGHT_g / _uL_PER_L / _MIN_PER_H
)  # = 40 x 45 x 18 / 1e6 / 60 = 0.00054


@dataclass(frozen=True)
class Drug:
    """Immutable drug compound specification.

    Attributes:
        name: Compound name.
        mw: Molecular weight (g/mol).
        logP: Octanol-water partition coefficient.
        pka: pKa values.
        drug_type: Ionization class (neutral, monoprotic_acid, monoprotic_base, diprotic).
        fup: Fraction unbound in plasma (0–1).
        rbp: Blood-to-plasma ratio.
        smiles: Optional SMILES string.
        clint: Intrinsic clearance by CYP enzyme (µL/min/pmol CYP).
        fm: Fraction metabolized by each enzyme (should sum to 1.0).
        peff: Effective intestinal permeability (×10⁻⁴ cm/s).
        solubility_mg_mL: Aqueous solubility at pH 6.5 (mg/mL).
        particle_radius_um: Particle radius for dissolution (µm).
        particle_density: Particle density (g/cm³).
        kp: Tissue:plasma partition coefficients for perfusion-limited organs.
        permeability_limited: Organs with permeability-surface area limitation.
            Maps organ name → {kp: float, ps: float (L/h)}.
        gut_clint_multiplier: Enterocyte CYP3A4 activity relative to hepatic.
        ka_sc: SC absorption rate constant (h^-1), first-order from depot.
        f_sc: SC bioavailability fraction (0-1); defaults to 1.0 (no first-pass).
        dose_mg: Dose (mg).
        route: Administration route ('oral', 'iv', or 'sc').
    """

    name: str = "Unknown"
    mw: float = 300.0
    logP: float = 2.0
    pka: list[float] = field(default_factory=lambda: [7.0])
    drug_type: str = "neutral"
    fup: float = 0.5
    rbp: float = 1.0
    smiles: str | None = None

    # Clearance — in vitro (IVIVE)
    clint: dict[str, float] = field(default_factory=dict)
    fm: dict[str, float] = field(default_factory=dict)

    # Clearance — calibrated in vivo (L/h); when > 0, overrides IVIVE scaling
    clint_hepatic_L_per_h: float = 0.0
    clint_gut_L_per_h: float = 0.0

    # Renal clearance on total plasma concentration basis (L/h)
    clr_L_per_h: float = 0.0

    # Absorption
    peff: float = 1.0
    solubility_mg_mL: float = 1.0
    particle_radius_um: float = 25.0
    particle_density: float = 1.2

    # Distribution — perfusion-limited Kp
    kp: dict[str, float] = field(default_factory=dict)

    # Distribution — permeability-limited organs
    permeability_limited: dict[str, dict[str, float]] = field(default_factory=dict)

    # Kp estimation method ('heuristic' or 'rodgers_rowland')
    partition_method: str = "heuristic"
    # Ionisation class for Rodgers & Rowland ('neutral', 'acid', 'base', 'zwitterion')
    compound_type: str = "neutral"

    # Gut metabolism
    gut_clint_multiplier: float = 1.0

    # SC (subcutaneous) absorption parameters
    ka_sc: float = 0.2  # SC absorption rate constant (h^-1), first-order
    f_sc: float = 1.0  # SC bioavailability fraction (no first-pass effect)

    # Dosing (set at simulation time)
    dose_mg: float = 0.0
    route: str = "oral"

    @property
    def total_clint(self) -> float:
        """Total intrinsic clearance across all enzymes (µL/min/pmol)."""
        return sum(self.clint.values())

    @property
    def clint_scaled_L_per_h(self) -> float:
        """Whole-liver scaled CLint (L/h).

        If clint_hepatic_L_per_h > 0 (calibrated in vivo value), returns that
        directly.  Otherwise applies IVIVE scaling:
          µL/min/pmol × MPPGL(40) × mg_protein/g(45) × liver_wt(1800g) / 1e6 / 60
          ≈ CLint × 0.054
        """
        if self.clint_hepatic_L_per_h > 0:
            return self.clint_hepatic_L_per_h
        return self.total_clint * _IVIVE_HEPATIC

    @property
    def gut_clint_scaled_L_per_h(self) -> float:
        """Gut wall scaled CLint (L/h) for CYP3A4 gut metabolism.

        If clint_gut_L_per_h > 0 (calibrated in vivo), returns that directly.
        Otherwise applies IVIVE scaling using ~1% of liver CYP3A4 content.
        """
        if self.clint_gut_L_per_h > 0:
            return self.clint_gut_L_per_h
        cyp3a4_clint = self.clint.get("CYP3A4", 0.0)
        return cyp3a4_clint * self.gut_clint_multiplier * _IVIVE_GUT

    def default_kp(self, organ: str) -> float:
        """Return Kp for an organ.

        Uses the stored Kp value if available, otherwise falls back to
        the selected partition_method ('heuristic' or 'rodgers_rowland').
        """
        if organ in self.kp:
            return self.kp[organ]
        from omega_pbpk.core.heuristics import get_partition_method

        method = get_partition_method(self.partition_method)
        pka_val = self.pka[0] if self.pka else None
        return method(
            logP=self.logP,
            pka=pka_val,
            compound_type=self.compound_type,
            tissue_name=organ,
            fup=self.fup,
        )
