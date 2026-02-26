"""Drug dataclass — central compound parameter container.

All parameters use internal standard units:
  Concentration: mg/L, Volume: L, Flow: L/h, Time: h, Clearance: L/h, Dose: mg
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
        dose_mg: Dose (mg).
        route: Administration route ('oral' or 'iv').
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

    # Absorption
    peff: float = 1.0
    solubility_mg_mL: float = 1.0
    particle_radius_um: float = 25.0
    particle_density: float = 1.2

    # Distribution — perfusion-limited Kp
    kp: dict[str, float] = field(default_factory=dict)

    # Distribution — permeability-limited organs
    permeability_limited: dict[str, dict[str, float]] = field(default_factory=dict)

    # Gut metabolism
    gut_clint_multiplier: float = 1.0

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
        return self.total_clint * 40.0 * 45.0 * 1800.0 / 1e6 / 60.0

    @property
    def gut_clint_scaled_L_per_h(self) -> float:
        """Gut wall scaled CLint (L/h) for CYP3A4 gut metabolism.

        If clint_gut_L_per_h > 0 (calibrated in vivo), returns that directly.
        Otherwise applies IVIVE scaling using ~1% of liver CYP3A4 content.
        """
        if self.clint_gut_L_per_h > 0:
            return self.clint_gut_L_per_h
        cyp3a4_clint = self.clint.get("CYP3A4", 0.0)
        return cyp3a4_clint * self.gut_clint_multiplier * 40.0 * 45.0 * 18.0 / 1e6 / 60.0

    def default_kp(self, organ: str) -> float:
        """Return Kp for an organ.

        Uses the stored Kp value if available, otherwise falls back to
        heuristic estimation from physicochemical properties (logP, pKa,
        drug_type, fup). This replaces the previous hard-coded 1.0 default.
        """
        if organ in self.kp:
            return self.kp[organ]
        from omega_pbpk.core.heuristics import heuristic_kp

        pka_val = self.pka[0] if self.pka else None
        return heuristic_kp(
            logP=self.logP,
            pka=pka_val,
            drug_type=self.drug_type,
            tissue_name=organ,
            fup=self.fup,
        )
