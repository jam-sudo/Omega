"""Advanced oral absorption model for the ACAT PBPK framework.

Improvements over the naive first-order ACAT model:

1. pH-dependent solubility (Henderson-Hasselbalch)
   Weak acids: S_pH = S0 × (1 + 10^(pH - pKa))
   Weak bases: S_pH = S0 × (1 + 10^(pKa - pH))
   Neutral:    S_pH = S0

2. Ionization-corrected effective permeability
   Peff_pH = Peff × f_unionized(pH)
   where f_unionized is the Henderson-Hasselbalch fraction at segment pH.

3. Noyes-Whitney dissolution factor
   If drug mass in segment exceeds solubility × volume (dose > dose-number threshold),
   the effective ka is reduced proportionally:
   ka_eff = ka_base × min(1, S_pH × V_lumen / mass_in_lumen)
   This prevents absorption of more than the dissolved fraction.

4. Supersaturation
   Amorphous or nano-formulations may achieve temporary supersaturation:
   S_eff = supersaturation_ratio × S_pH
   Rain-out (precipitation) is modelled as a first-order process.

5. Food effect (fasted vs. fed state)
   Fed state modifiers per segment:
   - Higher gastric pH (1.5 → 4.5) reduces ionisation of weak acids
   - Delayed gastric emptying (t_transit_stomach × gastric_delay_factor)
   - Increased bile-salt concentration → micellar solubilisation of lipophilic drugs
     ΔS_bile = bile_enhancement × logP_factor × S0

Reference physiology:
  Yu & Amidon (1999) Eur J Pharm Biopharm 48:157-167
  Willmann et al. (2003) J Med Chem 46:1625-1629
  Jamei et al. (2009) Expert Opin Drug Metab Toxicol 5:211-227
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omega_pbpk.drugs.drug import Drug

__all__ = [
    "GISegment",
    "GI_SEGMENTS",
    "AbsorptionModel",
    "FoodEffect",
]

# ---------------------------------------------------------------------------
# GI segment physiological data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GISegment:
    """Physiology of a single GI compartment.

    Args:
        name: Segment name (matches ACAT state index label in body.py).
        ph_fasted: Luminal pH in fasted state.
        ph_fed: Luminal pH in fed state.
        volume_fasted_L: Luminal fluid volume — fasted (L).
        volume_fed_L: Luminal fluid volume — fed (L).
        length_cm: Segment length (cm).
        radius_cm: Luminal radius (cm).
        surface_area_cm2: Effective absorptive surface area including villi (cm²).
        ka_fraction: Fractional absorption activity vs. jejunum 1 reference (0–1).
        bile_mM_fasted: Bile salt concentration — fasted (mM).
        bile_mM_fed: Bile salt concentration — fed (mM).
    """

    name: str
    ph_fasted: float
    ph_fed: float
    volume_fasted_L: float
    volume_fed_L: float
    length_cm: float
    radius_cm: float
    surface_area_cm2: float
    ka_fraction: float
    bile_mM_fasted: float = 3.0
    bile_mM_fed: float = 15.0


# Reference GI segment physiology (Yu & Amidon 1999; Jamei 2009)
GI_SEGMENTS: list[GISegment] = [
    GISegment(
        name="stomach",
        ph_fasted=1.5,   ph_fed=4.5,
        volume_fasted_L=0.050, volume_fed_L=0.200,
        length_cm=20.0,  radius_cm=5.0,
        surface_area_cm2=600.0,
        ka_fraction=0.0,         # no absorption from stomach
        bile_mM_fasted=0.0, bile_mM_fed=0.0,
    ),
    GISegment(
        name="duodenum",
        ph_fasted=5.5,   ph_fed=5.8,
        volume_fasted_L=0.030, volume_fed_L=0.050,
        length_cm=25.0,  radius_cm=1.75,
        surface_area_cm2=7_680.0,
        ka_fraction=1.0,
        bile_mM_fasted=3.0, bile_mM_fed=15.0,
    ),
    GISegment(
        name="jejunum1",
        ph_fasted=6.0,   ph_fed=6.2,
        volume_fasted_L=0.085, volume_fed_L=0.120,
        length_cm=150.0, radius_cm=1.60,
        surface_area_cm2=30_000.0,
        ka_fraction=1.0,
        bile_mM_fasted=3.0, bile_mM_fed=15.0,
    ),
    GISegment(
        name="jejunum2",
        ph_fasted=6.2,   ph_fed=6.4,
        volume_fasted_L=0.085, volume_fed_L=0.120,
        length_cm=150.0, radius_cm=1.60,
        surface_area_cm2=30_000.0,
        ka_fraction=1.0,
        bile_mM_fasted=3.0, bile_mM_fed=15.0,
    ),
    GISegment(
        name="ileum1",
        ph_fasted=6.8,   ph_fed=7.0,
        volume_fasted_L=0.060, volume_fed_L=0.080,
        length_cm=100.0, radius_cm=1.50,
        surface_area_cm2=21_000.0,
        ka_fraction=0.8,
        bile_mM_fasted=2.0, bile_mM_fed=10.0,
    ),
    GISegment(
        name="ileum2",
        ph_fasted=7.0,   ph_fed=7.2,
        volume_fasted_L=0.060, volume_fed_L=0.080,
        length_cm=100.0, radius_cm=1.50,
        surface_area_cm2=21_000.0,
        ka_fraction=0.6,
        bile_mM_fasted=2.0, bile_mM_fed=8.0,
    ),
    GISegment(
        name="ileum3",
        ph_fasted=7.0,   ph_fed=7.4,
        volume_fasted_L=0.060, volume_fed_L=0.080,
        length_cm=100.0, radius_cm=1.50,
        surface_area_cm2=21_000.0,
        ka_fraction=0.3,
        bile_mM_fasted=1.0, bile_mM_fed=5.0,
    ),
    GISegment(
        name="colon",
        ph_fasted=6.8,   ph_fed=6.8,
        volume_fasted_L=0.270, volume_fed_L=0.270,
        length_cm=100.0, radius_cm=2.50,
        surface_area_cm2=4_500.0,
        ka_fraction=0.05,
        bile_mM_fasted=0.5, bile_mM_fed=0.5,
    ),
]

# Build lookup by name
_GI_SEGMENT_MAP: dict[str, GISegment] = {s.name: s for s in GI_SEGMENTS}


# ---------------------------------------------------------------------------
# Food effect parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FoodEffect:
    """Fed-state modifiers for GI absorption.

    Args:
        gastric_delay_factor: Multiplier on stomach transit time (fed > 1 → slower
            gastric emptying, e.g. 2.0 for high-fat meal).
        intestinal_transit_factor: Multiplier on small-intestine transit times.
        solubility_enhancement_per_logP: Slope for micellar solubilisation;
            ΔS (mg/L) = bile_mM × enhancement × logP_factor × S0.
            Empirical: ~0.05 per mM bile salt per logP unit.
        supersaturation_ratio: Drug concentration achievable above equilibrium
            solubility (dimensionless, ≥ 1); relevant for amorphous/nanoformulations.
        precipitation_rate_per_h: First-order rate for supersaturated drug to
            precipitate back to the equilibrium solubility (h⁻¹).
    """

    gastric_delay_factor: float = 2.0           # fed: 2× longer gastric residence
    intestinal_transit_factor: float = 1.0      # typically no change for SI
    solubility_enhancement_per_logP: float = 0.05  # (mM bile)⁻¹ · logP⁻¹
    supersaturation_ratio: float = 1.0          # 1 = no supersaturation
    precipitation_rate_per_h: float = 0.0       # 0 = no precipitation modelled


# ---------------------------------------------------------------------------
# Main absorption model
# ---------------------------------------------------------------------------

class AbsorptionModel:
    """Pre-compute and provide per-segment absorption kinetics for a drug.

    Used by ``WholeBodyPBPK._rhs`` to replace the single hard-coded ``ka``
    formula with mechanistic, concentration-dependent absorption rates.

    The model distinguishes between:

    * **Dissolution-limited** absorption (BCS class II/IV): The effective ka
      is scaled by the dissolved fraction, preventing over-absorption of
      undissolved drug.
    * **Permeability-limited** absorption (BCS class I/III): pH-dependent
      ionisation correction applied to Peff.
    * **Food effect**: pH shift, micellar solubilisation, gastric delay.
    * **Supersaturation**: Temporary excess solubility from amorphous or
      nano-formulations with optional precipitation.

    Args:
        drug: Drug parameter container.
        fed_state: Whether to use fed-state GI physiology.
        food_effect: Custom food-effect parameters (uses defaults if None).
    """

    PEFF_SCALING = 1e-4 * 3600.0  # cm/s → cm/h; factor in ka formula
    # Empirical scaling to match Caco-2 → in vivo absorption:
    KA_GLOBAL_SCALE = 0.01

    def __init__(
        self,
        drug: Drug,
        fed_state: bool = False,
        food_effect: FoodEffect | None = None,
    ) -> None:
        self.drug = drug
        self.fed_state = fed_state
        self.food_effect = food_effect or (FoodEffect() if fed_state else None)
        self._segments = _GI_SEGMENT_MAP

    # ------------------------------------------------------------------
    # pH-dependent solubility
    # ------------------------------------------------------------------

    def solubility_at_ph(self, ph: float, segment: GISegment | None = None) -> float:
        """Compute pH-dependent solubility (mg/L) using Henderson-Hasselbalch.

        For lipophilic drugs (logP > 2) in the presence of bile salts, adds a
        micellar solubilisation term.

        Args:
            ph: Luminal pH.
            segment: GI segment (used for bile-salt solubilisation lookup).

        Returns:
            Effective solubility (mg/L) ≥ intrinsic S0.
        """
        s0_mg_L = self.drug.solubility_mg_mL * 1000.0  # mg/mL → mg/L
        pka = self.drug.pka[0] if self.drug.pka else None
        drug_type = self.drug.drug_type

        # Henderson-Hasselbalch ionisation factor
        if pka is not None and pka > 0:
            if drug_type in ("monoprotic_acid", "acid"):
                # S_pH = S0 × (1 + 10^(pH - pKa))
                s_ph = s0_mg_L * (1.0 + 10.0 ** (ph - pka))
            elif drug_type in ("monoprotic_base", "base"):
                # S_pH = S0 × (1 + 10^(pKa - pH))
                s_ph = s0_mg_L * (1.0 + 10.0 ** (pka - ph))
            else:
                s_ph = s0_mg_L
        else:
            s_ph = s0_mg_L

        # Micellar solubilisation by bile salts (lipophilic drugs)
        if segment is not None and self.drug.logP > 2.0:
            bile_mM = segment.bile_mM_fed if self.fed_state else segment.bile_mM_fasted
            if bile_mM > 0 and self.food_effect is not None:
                # ΔS ≈ bile_mM × Km_bile × logP (empirical, Pouton 2006)
                logP_factor = max(self.drug.logP - 2.0, 0.0)  # only above logP=2
                bile_enhancement = bile_mM * self.food_effect.solubility_enhancement_per_logP
                s_ph += s_ph * bile_enhancement * logP_factor

        return max(s_ph, s0_mg_L)  # never below intrinsic solubility

    # ------------------------------------------------------------------
    # Ionisation-corrected Peff
    # ------------------------------------------------------------------

    def peff_at_ph(self, ph: float) -> float:
        """Ionisation-corrected effective permeability (×10⁻⁴ cm/s) at pH.

        Only the unionised fraction is membrane-permeable. For highly charged
        species at extreme pH the effective Peff is reduced, limiting absorption.

        Args:
            ph: Luminal pH.

        Returns:
            Effective Peff (×10⁻⁴ cm/s) ≥ 0.
        """
        pka = self.drug.pka[0] if self.drug.pka else None
        drug_type = self.drug.drug_type

        if pka is None:
            return self.drug.peff

        # Fraction unionised at segment pH
        if drug_type in ("monoprotic_acid", "acid"):
            f_un = 1.0 / (1.0 + 10.0 ** (ph - pka))
        elif drug_type in ("monoprotic_base", "base"):
            f_un = 1.0 / (1.0 + 10.0 ** (pka - ph))
        else:
            f_un = 1.0  # neutral: fully permeable at all pH

        # Minimum Peff for ionised fraction (membrane carries ~10% even charged)
        f_un_corrected = max(f_un, 0.10)
        return self.drug.peff * f_un_corrected

    # ------------------------------------------------------------------
    # Per-segment effective ka
    # ------------------------------------------------------------------

    def segment_ka(self, segment_name: str, mass_mg: float = 1.0) -> float:
        """Compute effective absorption rate constant (h⁻¹) for a segment.

        Incorporates:
        1. pH-corrected Peff (ionisation at segment pH)
        2. Noyes-Whitney dissolution factor (if mass > solubility × volume)
        3. Supersaturation enhancement (from FoodEffect.supersaturation_ratio)

        Args:
            segment_name: GI segment name (must be in GI_SEGMENTS).
            mass_mg: Current drug mass in segment (mg). Used for dissolution
                     factor calculation. Pass 1.0 or any positive value to
                     get the maximum (non-dissolution-limited) ka.

        Returns:
            Effective ka (h⁻¹) for the segment. Returns 0 for stomach
            (no absorption) and colon (reduced).
        """
        seg = self._segments.get(segment_name)
        if seg is None or seg.ka_fraction == 0.0:
            return 0.0

        # Segment pH
        ph = seg.ph_fed if self.fed_state else seg.ph_fasted

        # pH-corrected Peff (×10⁻⁴ cm/s)
        peff_ph = self.peff_at_ph(ph)

        # Base ka from permeability and particle radius
        # ka (h⁻¹) = 2 × Peff / R_particle
        # Peff in (cm/s×10⁻⁴), convert: × 1e-4 × 3600 → cm/h
        radius_cm = max(self.drug.particle_radius_um * 1e-4, 1e-8)
        ka_base = 2.0 * peff_ph * self.PEFF_SCALING / radius_cm

        # Apply segment-specific activity fraction and global empirical scaling
        ka = ka_base * seg.ka_fraction * self.KA_GLOBAL_SCALE

        # Dissolution factor (Noyes-Whitney)
        # If undissolved drug exceeds local solubility × volume, ka is reduced
        if mass_mg > 1e-9:
            sigma = 1.0
            if self.food_effect is not None:
                sigma = max(self.food_effect.supersaturation_ratio, 1.0)
            v_lumen = seg.volume_fed_L if self.fed_state else seg.volume_fasted_L
            s_eff = self.solubility_at_ph(ph, seg) * sigma  # mg/L
            max_dissolved_mg = s_eff * v_lumen  # mg that can be dissolved

            if mass_mg > max_dissolved_mg > 0:
                ka *= max_dissolved_mg / mass_mg  # reduce ka proportionally

        return max(ka, 0.0)

    def transit_time_multiplier(self) -> dict[str, float]:
        """Return per-segment transit time multipliers for fed-state correction.

        In fed state, gastric emptying is delayed (multiplier > 1 for stomach).
        Small intestinal and colonic transit times are typically unchanged.

        Returns:
            Dict mapping segment name → transit time multiplier (≥ 1).
        """
        multipliers: dict[str, float] = {seg.name: 1.0 for seg in GI_SEGMENTS}
        if self.fed_state and self.food_effect is not None:
            multipliers["stomach"] = self.food_effect.gastric_delay_factor
            for s in ("duodenum", "jejunum1", "jejunum2", "ileum1", "ileum2", "ileum3", "colon"):
                multipliers[s] = self.food_effect.intestinal_transit_factor
        return multipliers

    def all_ka(self, amounts_mg: dict[str, float] | None = None) -> dict[str, float]:
        """Compute ka (h⁻¹) for all 8 GI segments.

        Args:
            amounts_mg: Dict of segment_name → current drug mass (mg).
                        If None, dissolution factor is not applied.

        Returns:
            Dict of segment_name → ka (h⁻¹).
        """
        result: dict[str, float] = {}
        for seg in GI_SEGMENTS:
            mass = (amounts_mg or {}).get(seg.name, 1.0)
            result[seg.name] = self.segment_ka(seg.name, mass_mg=max(mass, 1e-12))
        return result
