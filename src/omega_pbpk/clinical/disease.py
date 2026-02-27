"""Disease state physiological scaling for PBPK simulations.

Implements population-specific physiological adjustments per FDA/EMA guidance
for special populations:

1. **Renal Impairment** (RI)
   GFR-based CKD staging (NKF KDIGO): Normal → Mild → Moderate → Severe → ESRD
   - Reduced GFR → reduced CLr (filtration + secretion)
   - Hypoalbuminaemia → altered fup (reduced protein binding)
   - Altered renal blood flow and volume
   Reference: Nolin et al. (2009) Semin Nephrol 29:496-503; FDA (2010) RI guidance

2. **Hepatic Impairment** (HI)
   Child-Pugh scoring (A/B/C) → reduced CYP activity, protein synthesis, hepatic blood flow
   - Reduced CLint (enzyme activity)
   - Reduced albumin → increased fup
   - Reduced hepatic blood flow and liver volume
   - Portal hypertension (shunting, reduces first-pass)
   Reference: Child & Turcotte (1964); FDA (2003) HI guidance

3. **Heart Failure** (HF)
   NYHA class I–IV
   - Reduced cardiac output → prolonged organ residence times
   - Reduced hepatic blood flow (splanchnic vasoconstriction) → reduced CLh
   - Peripheral oedema → expanded Vd
   Reference: Verbeeck (2008) Eur J Clin Pharmacol 64:1147-1161

4. **Obesity** (BW-based; BMI ≥ 30 kg/m²)
   - Increased adipose volume, total body water
   - Increased cardiac output
   - Altered fup (elevated triglycerides, fatty acids)
   - Reference body weight: IBW (Devine, 1974); ABW for dose calculation
   Reference: Hanley et al. (2010) Clin Pharmacokinet 49:71-87

5. **Pregnancy** (trimester I–III)
   - Expanded plasma volume (+40-50% in T3)
   - Increased cardiac output (+40%)
   - Induced CYP3A4/2C9 → reduced AUC of substrates
   - Reduced CYP1A2 → increased AUC of substrates
   - Increased GFR (+50% in T2/T3)
   - Increased albumin dilution → altered fup
   Reference: Hodge & Tracy (2007) Expert Opin Drug Metab Toxicol 3:557-571
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

__all__ = [
    "RenalImpairmentStage",
    "ChildPughClass",
    "NyhaClass",
    "TrimesterStage",
    "DiseaseState",
    "RenalImpairment",
    "HepaticImpairment",
    "HeartFailure",
    "ObesityState",
    "PregnancyState",
    "apply_disease_scaling",
    "ScaledParameters",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RenalImpairmentStage(IntEnum):
    """NKF KDIGO CKD staging based on eGFR (mL/min/1.73m²)."""

    NORMAL = 0  # eGFR ≥ 90
    MILD = 1  # eGFR 60–89
    MODERATE = 2  # eGFR 30–59
    SEVERE = 3  # eGFR 15–29
    ESRD = 4  # eGFR < 15 (or dialysis)


class ChildPughClass(IntEnum):
    """Child-Pugh classification of hepatic impairment."""

    NORMAL = 0
    A = 1  # Mild:   score 5-6
    B = 2  # Moderate: score 7-9
    C = 3  # Severe:   score 10-15


class NyhaClass(IntEnum):
    """NYHA functional classification of heart failure."""

    NORMAL = 0
    CLASS_I = 1  # No limitation
    CLASS_II = 2  # Slight limitation
    CLASS_III = 3  # Marked limitation
    CLASS_IV = 4  # Symptoms at rest


class TrimesterStage(IntEnum):
    """Pregnancy trimester."""

    NON_PREGNANT = 0
    T1 = 1  # First trimester (weeks 1-13)
    T2 = 2  # Second trimester (weeks 14-26)
    T3 = 3  # Third trimester (weeks 27-40)


# ---------------------------------------------------------------------------
# Disease state dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenalImpairment:
    """Renal impairment specification.

    Args:
        stage: CKD stage (RenalImpairmentStage enum).
        egfr_mL_min: Estimated GFR (mL/min/1.73 m²). If provided, overrides
                     stage-based lookup for CLr scaling.
        albumin_g_dL: Plasma albumin (g/dL). Normal ~4.0 g/dL; CKD ~3.0 g/dL
                      (stage 4-5). Used to correct fup.
        on_dialysis: Whether patient is on haemodialysis (affects CLr).
    """

    stage: RenalImpairmentStage = RenalImpairmentStage.NORMAL
    egfr_mL_min: float | None = None  # if None, use stage-based value
    albumin_g_dL: float = 4.0
    on_dialysis: bool = False

    # Reference GFR by stage (mL/min, midpoint; FDA 2010)
    _STAGE_EGFR: dict[RenalImpairmentStage, float] = field(
        default_factory=lambda: {
            RenalImpairmentStage.NORMAL: 110.0,
            RenalImpairmentStage.MILD: 75.0,
            RenalImpairmentStage.MODERATE: 44.0,
            RenalImpairmentStage.SEVERE: 21.0,
            RenalImpairmentStage.ESRD: 8.0,
        },
        repr=False,
        compare=False,
    )

    @property
    def effective_egfr(self) -> float:
        """eGFR used for clearance scaling."""
        if self.egfr_mL_min is not None:
            return max(self.egfr_mL_min, 0.0)
        return self._STAGE_EGFR[self.stage]

    @property
    def clr_fraction(self) -> float:
        """Fraction of normal renal clearance remaining (0–1).

        Assumes CLr scales linearly with GFR.
        Reference GFR for normal = 110 mL/min.
        """
        if self.on_dialysis:
            return 0.0
        return min(self.effective_egfr / 110.0, 1.0)

    @property
    def albumin_fraction(self) -> float:
        """Albumin as fraction of normal (4.0 g/dL)."""
        return min(self.albumin_g_dL / 4.0, 1.0)


@dataclass(frozen=True)
class HepaticImpairment:
    """Hepatic impairment specification.

    Args:
        cp_class: Child-Pugh class.
        albumin_g_dL: Plasma albumin (g/dL). Normal ~4.5; severe HI ~2.0.
        portal_shunt_fraction: Fraction of portal blood bypassing hepatocytes.
            Portal hypertension in Child-Pugh B/C leads to portosystemic shunting.
    """

    cp_class: ChildPughClass = ChildPughClass.NORMAL
    albumin_g_dL: float = 4.5
    portal_shunt_fraction: float | None = None  # if None, use cp_class defaults

    # CYP activity retention by CP class (Edginton & Willmann, 2008)
    _CYP_FRACTION: dict[ChildPughClass, float] = field(
        default_factory=lambda: {
            ChildPughClass.NORMAL: 1.00,
            ChildPughClass.A: 0.75,
            ChildPughClass.B: 0.50,
            ChildPughClass.C: 0.25,
        },
        repr=False,
        compare=False,
    )

    # Hepatic blood flow fraction by CP class
    _QH_FRACTION: dict[ChildPughClass, float] = field(
        default_factory=lambda: {
            ChildPughClass.NORMAL: 1.00,
            ChildPughClass.A: 0.90,
            ChildPughClass.B: 0.75,
            ChildPughClass.C: 0.60,
        },
        repr=False,
        compare=False,
    )

    # Default portal shunt fractions by CP class (Lam et al., 1997)
    _SHUNT_FRACTION: dict[ChildPughClass, float] = field(
        default_factory=lambda: {
            ChildPughClass.NORMAL: 0.00,
            ChildPughClass.A: 0.05,
            ChildPughClass.B: 0.20,
            ChildPughClass.C: 0.40,
        },
        repr=False,
        compare=False,
    )

    @property
    def cyp_activity_fraction(self) -> float:
        """Fraction of hepatic CYP enzyme activity remaining (0–1)."""
        return self._CYP_FRACTION[self.cp_class]

    @property
    def hepatic_flow_fraction(self) -> float:
        """Fraction of normal hepatic blood flow (0–1)."""
        return self._QH_FRACTION[self.cp_class]

    @property
    def effective_shunt_fraction(self) -> float:
        """Portal shunt fraction (0–1)."""
        if self.portal_shunt_fraction is not None:
            return max(0.0, min(self.portal_shunt_fraction, 1.0))
        return self._SHUNT_FRACTION[self.cp_class]

    @property
    def albumin_fraction(self) -> float:
        """Albumin as fraction of normal (4.5 g/dL)."""
        return min(self.albumin_g_dL / 4.5, 1.0)


@dataclass(frozen=True)
class HeartFailure:
    """Heart failure specification.

    Args:
        nyha_class: NYHA class (I–IV).
        ejection_fraction: Left ventricular ejection fraction (0–1, normal ~0.60).
    """

    nyha_class: NyhaClass = NyhaClass.NORMAL
    ejection_fraction: float = 0.60

    # Cardiac output fraction by NYHA class (relative to normal CO at 390 L/h)
    _CO_FRACTION: dict[NyhaClass, float] = field(
        default_factory=lambda: {
            NyhaClass.NORMAL: 1.00,
            NyhaClass.CLASS_I: 0.90,
            NyhaClass.CLASS_II: 0.80,
            NyhaClass.CLASS_III: 0.65,
            NyhaClass.CLASS_IV: 0.50,
        },
        repr=False,
        compare=False,
    )

    # Hepatic flow fraction (splanchnic vasoconstriction)
    _QH_FRACTION: dict[NyhaClass, float] = field(
        default_factory=lambda: {
            NyhaClass.NORMAL: 1.00,
            NyhaClass.CLASS_I: 0.90,
            NyhaClass.CLASS_II: 0.75,
            NyhaClass.CLASS_III: 0.60,
            NyhaClass.CLASS_IV: 0.45,
        },
        repr=False,
        compare=False,
    )

    @property
    def cardiac_output_fraction(self) -> float:
        """Fraction of normal cardiac output (0–1)."""
        return self._CO_FRACTION[self.nyha_class]

    @property
    def hepatic_flow_fraction(self) -> float:
        """Fraction of normal hepatic blood flow (0–1)."""
        return self._QH_FRACTION[self.nyha_class]


@dataclass(frozen=True)
class ObesityState:
    """Obesity physiological scaling.

    Args:
        body_weight_kg: Actual body weight (kg).
        height_cm: Height (cm). Used for IBW and BMI calculation.
        sex: 'male' or 'female' (for Devine IBW formula).
    """

    body_weight_kg: float = 70.0
    height_cm: float = 170.0
    sex: str = "male"

    @property
    def bmi(self) -> float:
        """Body mass index (kg/m²)."""
        h_m = max(self.height_cm / 100.0, 0.1)
        return self.body_weight_kg / h_m**2

    @property
    def is_obese(self) -> bool:
        return self.bmi >= 30.0

    @property
    def ibw_kg(self) -> float:
        """Ideal body weight (Devine formula, 1974)."""
        h_over_152 = max(self.height_cm - 152.4, 0.0)
        if self.sex.lower() == "female":
            return 45.5 + 0.906 * h_over_152
        return 50.0 + 0.906 * h_over_152

    @property
    def adjusted_bw_kg(self) -> float:
        """Adjusted body weight (ABW) for dosing in obese patients.

        ABW = IBW + 0.4 × (TBW - IBW)  [Bauer et al., 1983]
        """
        ibw = self.ibw_kg
        tbw = self.body_weight_kg
        if tbw <= ibw:
            return tbw
        return ibw + 0.4 * (tbw - ibw)

    @property
    def adipose_volume_scale(self) -> float:
        """Scale factor for adipose tissue volume relative to 70 kg reference.

        Adipose mass scales approximately linearly with excess body weight.
        """
        reference_adipose_kg = 14.5  # ICRP reference 70 kg
        # Excess fat approximated from excess BW (rough; more rigorous uses BF%)
        excess_bw = max(self.body_weight_kg - self.ibw_kg, 0.0)
        adipose_kg = reference_adipose_kg + 0.8 * excess_bw  # ~80% of excess is fat
        return adipose_kg / reference_adipose_kg


@dataclass(frozen=True)
class PregnancyState:
    """Pregnancy physiological changes.

    Args:
        trimester: Pregnancy trimester (T1, T2, T3, or NON_PREGNANT).
        gestational_week: Exact gestational week (used for continuous scaling
            within trimester if specified; otherwise uses trimester defaults).
    """

    trimester: TrimesterStage = TrimesterStage.NON_PREGNANT
    gestational_week: int | None = None

    # Plasma volume expansion (fraction above normal)
    _PLASMA_EXPANSION: dict[TrimesterStage, float] = field(
        default_factory=lambda: {
            TrimesterStage.NON_PREGNANT: 0.00,
            TrimesterStage.T1: 0.10,
            TrimesterStage.T2: 0.30,
            TrimesterStage.T3: 0.45,
        },
        repr=False,
        compare=False,
    )

    # Cardiac output expansion (fraction above normal)
    _CO_EXPANSION: dict[TrimesterStage, float] = field(
        default_factory=lambda: {
            TrimesterStage.NON_PREGNANT: 0.00,
            TrimesterStage.T1: 0.15,
            TrimesterStage.T2: 0.35,
            TrimesterStage.T3: 0.40,
        },
        repr=False,
        compare=False,
    )

    # GFR expansion (fraction above normal)
    _GFR_EXPANSION: dict[TrimesterStage, float] = field(
        default_factory=lambda: {
            TrimesterStage.NON_PREGNANT: 0.00,
            TrimesterStage.T1: 0.20,
            TrimesterStage.T2: 0.50,
            TrimesterStage.T3: 0.50,
        },
        repr=False,
        compare=False,
    )

    # CYP activity changes (fold-change from normal)
    _CYP3A4_FACTOR: dict[TrimesterStage, float] = field(
        default_factory=lambda: {
            TrimesterStage.NON_PREGNANT: 1.00,
            TrimesterStage.T1: 1.30,
            TrimesterStage.T2: 1.70,
            TrimesterStage.T3: 2.00,
        },
        repr=False,
        compare=False,
    )
    _CYP1A2_FACTOR: dict[TrimesterStage, float] = field(
        default_factory=lambda: {
            TrimesterStage.NON_PREGNANT: 1.00,
            TrimesterStage.T1: 0.90,
            TrimesterStage.T2: 0.70,
            TrimesterStage.T3: 0.65,
        },
        repr=False,
        compare=False,
    )
    _CYP2D6_FACTOR: dict[TrimesterStage, float] = field(
        default_factory=lambda: {
            TrimesterStage.NON_PREGNANT: 1.00,
            TrimesterStage.T1: 1.20,
            TrimesterStage.T2: 1.50,
            TrimesterStage.T3: 1.50,
        },
        repr=False,
        compare=False,
    )

    @property
    def plasma_volume_factor(self) -> float:
        """Multiplicative factor on plasma volume (> 1 in pregnancy)."""
        return 1.0 + self._PLASMA_EXPANSION[self.trimester]

    @property
    def cardiac_output_factor(self) -> float:
        """Multiplicative factor on cardiac output."""
        return 1.0 + self._CO_EXPANSION[self.trimester]

    @property
    def gfr_factor(self) -> float:
        """Multiplicative factor on GFR → scales CLr."""
        return 1.0 + self._GFR_EXPANSION[self.trimester]

    @property
    def cyp3a4_factor(self) -> float:
        return self._CYP3A4_FACTOR[self.trimester]

    @property
    def cyp1a2_factor(self) -> float:
        return self._CYP1A2_FACTOR[self.trimester]

    @property
    def cyp2d6_factor(self) -> float:
        return self._CYP2D6_FACTOR[self.trimester]


# ---------------------------------------------------------------------------
# Composite disease state
# ---------------------------------------------------------------------------


@dataclass
class DiseaseState:
    """Container for all active disease-state modifications.

    All fields default to None / normal state, meaning no modification is applied.
    Multiple conditions can be combined.

    Example::

        state = DiseaseState(
            renal=RenalImpairment(stage=RenalImpairmentStage.MODERATE),
            hepatic=HepaticImpairment(cp_class=ChildPughClass.B),
        )
        params = apply_disease_scaling(drug, body_weight=70.0, disease=state)
    """

    renal: RenalImpairment | None = None
    hepatic: HepaticImpairment | None = None
    heart_failure: HeartFailure | None = None
    obesity: ObesityState | None = None
    pregnancy: PregnancyState | None = None


# ---------------------------------------------------------------------------
# Scaled parameter container
# ---------------------------------------------------------------------------


@dataclass
class ScaledParameters:
    """Drug and physiological parameters after disease-state scaling.

    These values should be passed to WholeBodyPBPK to run the diseased
    population simulation.

    Attributes:
        body_weight_kg: Effective body weight for organ scaling (may use ABW
                        for obese patients).
        cardiac_output_scale: Fraction of normal CO applied to all organ flows.
        hepatic_flow_fraction: Additional scale on liver blood flow (independent
                               of CO scale; from HF or hepatic impairment).
        clint_scale: Fraction of normal CLint (CYP activity × co-induction/etc.).
        clr_scale: Fraction of normal CLr.
        fup_scale: Ratio of disease-state fup to reference fup (1.0 = no change).
        clint_per_enzyme: Per-enzyme CYP activity scale dict (e.g. pregnancy).
        notes: Human-readable description of applied modifications.
    """

    body_weight_kg: float = 70.0
    cardiac_output_scale: float = 1.0
    hepatic_flow_fraction: float = 1.0
    clint_scale: float = 1.0
    clr_scale: float = 1.0
    fup_scale: float = 1.0
    clint_per_enzyme: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scaling function
# ---------------------------------------------------------------------------


def apply_disease_scaling(
    body_weight: float = 70.0,
    disease: DiseaseState | None = None,
) -> ScaledParameters:
    """Compute scaled physiological parameters for a given disease state.

    Combines all active conditions into a single ScaledParameters object
    which can be used to modify Drug fields and WholeBodyPBPK inputs.

    Args:
        body_weight: Reference body weight (kg; before obesity adjustment).
        disease: Active disease state (or None for healthy population).

    Returns:
        ScaledParameters with all scalings applied.
    """
    params = ScaledParameters(body_weight_kg=body_weight)

    if disease is None:
        return params

    # ------------------------------------------------------------------
    # Renal impairment
    # ------------------------------------------------------------------
    if disease.renal is not None:
        ri = disease.renal
        params.clr_scale *= ri.clr_fraction
        # Hypoalbuminaemia: fup ≈ 1 − (1 − fup_ref) × albumin_fraction
        # Simplified: scale fup inversely with albumin
        if ri.albumin_fraction < 1.0:
            # Fup increases as albumin decreases (less binding)
            params.fup_scale *= (1.0 / max(ri.albumin_fraction, 0.1)) ** 0.5
        params.notes.append(
            f"Renal impairment {ri.stage.name}: CLr×{ri.clr_fraction:.2f}, "
            f"fup×{params.fup_scale:.2f}"
        )

    # ------------------------------------------------------------------
    # Hepatic impairment
    # ------------------------------------------------------------------
    if disease.hepatic is not None:
        hi = disease.hepatic
        params.clint_scale *= hi.cyp_activity_fraction
        params.hepatic_flow_fraction *= hi.hepatic_flow_fraction
        # Hypoalbuminaemia effect on fup (independent direction from RI above)
        if hi.albumin_fraction < 1.0:
            params.fup_scale *= (1.0 / max(hi.albumin_fraction, 0.1)) ** 0.5
        params.notes.append(
            f"Hepatic impairment {hi.cp_class.name}: CLint×{hi.cyp_activity_fraction:.2f}, "
            f"Qh×{hi.hepatic_flow_fraction:.2f}, shunt={hi.effective_shunt_fraction:.0%}"
        )

    # ------------------------------------------------------------------
    # Heart failure
    # ------------------------------------------------------------------
    if disease.heart_failure is not None:
        hf = disease.heart_failure
        params.cardiac_output_scale *= hf.cardiac_output_fraction
        params.hepatic_flow_fraction *= hf.hepatic_flow_fraction
        params.notes.append(
            f"Heart failure NYHA {hf.nyha_class.name}: CO×{hf.cardiac_output_fraction:.2f}, "
            f"Qh×{hf.hepatic_flow_fraction:.2f}"
        )

    # ------------------------------------------------------------------
    # Obesity
    # ------------------------------------------------------------------
    if disease.obesity is not None:
        ob = disease.obesity
        # Use adjusted body weight for organ volume scaling
        params.body_weight_kg = ob.adjusted_bw_kg
        # Slightly elevated CO in obesity
        co_scale = 1.0 + max(ob.bmi - 30.0, 0.0) * 0.01  # ~1% per BMI unit
        params.cardiac_output_scale *= min(co_scale, 1.5)
        params.notes.append(
            f"Obesity BMI={ob.bmi:.1f}: ABW={ob.adjusted_bw_kg:.1f} kg, CO×{co_scale:.2f}"
        )

    # ------------------------------------------------------------------
    # Pregnancy
    # ------------------------------------------------------------------
    if disease.pregnancy is not None:
        pg = disease.pregnancy
        if pg.trimester != TrimesterStage.NON_PREGNANT:
            params.cardiac_output_scale *= pg.cardiac_output_factor
            params.clr_scale *= pg.gfr_factor
            # Per-enzyme CYP scaling
            params.clint_per_enzyme["CYP3A4"] = (
                params.clint_per_enzyme.get("CYP3A4", 1.0) * pg.cyp3a4_factor
            )
            params.clint_per_enzyme["CYP1A2"] = (
                params.clint_per_enzyme.get("CYP1A2", 1.0) * pg.cyp1a2_factor
            )
            params.clint_per_enzyme["CYP2D6"] = (
                params.clint_per_enzyme.get("CYP2D6", 1.0) * pg.cyp2d6_factor
            )
            params.notes.append(
                f"Pregnancy {pg.trimester.name}: CO×{pg.cardiac_output_factor:.2f}, "
                f"CLr×{pg.gfr_factor:.2f}, CYP3A4×{pg.cyp3a4_factor:.2f}, "
                f"CYP1A2×{pg.cyp1a2_factor:.2f}"
            )

    return params
