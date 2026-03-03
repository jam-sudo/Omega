"""Dynamic DDI simulation — victim PK with perpetrator-driven enzyme inhibition/induction.

Uses WholeBodyPBPK with DDIInhibitor to simulate perpetrator effects on victim PK.
Reports AUCR, Cmax ratio, and FDA 2020 AUCR-based severity classification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from omega_pbpk.core.body import DDIInhibitor, WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug


@dataclass(frozen=True)
class PerpetratorSpec:
    """Perpetrator drug specification for dynamic DDI simulation."""

    name: str
    ki_uM: float
    target_enzyme: str = "CYP3A4"
    mechanism: str = "competitive"  # competitive, mbi, induction
    kinact_per_h: float = 0.0
    kdeg_per_h: float = 0.02
    fold_induction: float = 1.0
    cmax_uM: float | None = None  # perpetrator plasma Cmax; None → default 1.0 µM


@dataclass(frozen=True)
class DDISimulationResult:
    """Dynamic DDI simulation result."""

    victim_name: str
    perpetrator_name: str
    target_enzyme: str
    mechanism: str
    auc_alone_mg_h_L: float
    cmax_alone_mg_L: float
    t_half_alone_h: float
    auc_ddi_mg_h_L: float
    cmax_ddi_mg_L: float
    t_half_ddi_h: float
    auc_ratio: float
    cmax_ratio: float
    classification: str  # no_interaction, weak, moderate, strong / induction variants
    perpetrator_cmax_uM: float
    static_r1: float
    time_h: tuple[float, ...]
    cp_alone_mg_L: tuple[float, ...]
    cp_ddi_mg_L: tuple[float, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def classify_ddi(auc_ratio: float) -> str:
    """Classify DDI severity per FDA 2020 AUCR thresholds.

    Inhibition:  <1.25 = no_interaction, 1.25–2 = weak, 2–5 = moderate, >=5 = strong
    Induction:   >0.8  = no_interaction, 0.5–0.8 = weak_induction,
                 0.2–0.5 = moderate_induction, <=0.2 = strong_induction
    """
    if auc_ratio >= 5.0:
        return "strong"
    elif auc_ratio >= 2.0:
        return "moderate"
    elif auc_ratio >= 1.25:
        return "weak"
    elif auc_ratio > 0.8:
        return "no_interaction"
    elif auc_ratio > 0.5:
        return "weak_induction"
    elif auc_ratio > 0.2:
        return "moderate_induction"
    else:
        return "strong_induction"


def simulate_ddi(
    victim_drug: Drug,
    perpetrator: PerpetratorSpec,
    dose_mg: float = 100.0,
    route: str = "oral",
    body_weight: float = 70.0,
    t_end_h: float = 24.0,
) -> DDISimulationResult:
    """Run dynamic DDI simulation comparing victim PK alone vs with perpetrator.

    Steps:
      1. Resolve perpetrator Cmax (from spec or default 1.0 µM).
      2. Simulate victim alone (baseline PK).
      3. Simulate victim + DDIInhibitor (interaction PK).
      4. Compute AUCR and Cmax ratio; classify per FDA 2020 thresholds.
      5. Compute static R1 for cross-validation.

    Args:
        victim_drug: Drug object for the victim compound.
        perpetrator: PerpetratorSpec describing inhibitor/inducer properties.
        dose_mg: Victim drug dose (mg).
        route: Administration route ('oral', 'iv', 'sc').
        body_weight: Subject body weight (kg).
        t_end_h: Simulation duration (h).

    Returns:
        DDISimulationResult with full PK profiles and DDI metrics.
    """
    warnings: list[str] = []

    # --- Resolve perpetrator Cmax ---
    cmax_uM = perpetrator.cmax_uM
    if cmax_uM is None:
        warnings.append("No perpetrator Cmax provided; using default 1.0 µM")
        cmax_uM = 1.0

    # --- Baseline simulation (victim alone) ---
    model_alone = WholeBodyPBPK(victim_drug, body_weight=body_weight)
    if route == "oral":
        model_alone.setup_oral(dose_mg)
    elif route == "iv":
        model_alone.setup_iv(dose_mg)
    else:
        model_alone.setup_sc(dose_mg)
    result_alone = model_alone.simulate(t_end_h=t_end_h)
    pk_alone = result_alone.pk_summary()

    # --- DDI simulation (victim + perpetrator) ---
    model_ddi = WholeBodyPBPK(victim_drug, body_weight=body_weight)
    if route == "oral":
        model_ddi.setup_oral(dose_mg)
    elif route == "iv":
        model_ddi.setup_iv(dose_mg)
    else:
        model_ddi.setup_sc(dose_mg)

    inhibitor = DDIInhibitor(
        name=perpetrator.name,
        ki_uM=perpetrator.ki_uM,
        concentration_uM=cmax_uM,
        target_enzyme=perpetrator.target_enzyme,
        mechanism=perpetrator.mechanism,
        kinact_per_h=perpetrator.kinact_per_h,
        kdeg_per_h=perpetrator.kdeg_per_h,
        fold_induction=perpetrator.fold_induction,
    )
    model_ddi.add_inhibitor(inhibitor)
    result_ddi = model_ddi.simulate(t_end_h=t_end_h)
    pk_ddi = result_ddi.pk_summary()

    # --- Extract PK parameters (use actual pk_summary keys) ---
    auc_alone = pk_alone.get("AUC_mg_h_L", 0.001)
    cmax_alone = pk_alone.get("Cmax_mg_L", 0.001)
    t_half_alone = pk_alone.get("half_life_h", 0.0)

    auc_ddi = pk_ddi.get("AUC_mg_h_L", 0.001)
    cmax_ddi = pk_ddi.get("Cmax_mg_L", 0.001)
    t_half_ddi = pk_ddi.get("half_life_h", 0.0)

    # Clip infinite half-lives (un-estimable terminal phase) to 0.0
    if math.isinf(t_half_alone) or t_half_alone > 1e6:
        t_half_alone = 0.0
    if math.isinf(t_half_ddi) or t_half_ddi > 1e6:
        t_half_ddi = 0.0

    # --- Ratios ---
    auc_ratio = auc_ddi / max(auc_alone, 1e-12)
    cmax_ratio = cmax_ddi / max(cmax_alone, 1e-12)

    classification = classify_ddi(auc_ratio)

    # Static R1 for cross-validation: R1 = 1 + [I]/Ki
    if perpetrator.ki_uM > 0:
        static_r1 = 1.0 + cmax_uM / max(perpetrator.ki_uM, 1e-12)
    else:
        static_r1 = float("inf")

    if perpetrator.mechanism == "mbi":
        warnings.append(
            "MBI uses static Cmax approximation; time-varying enzyme inactivation not modeled"
        )

    # --- Concentration-time profiles ---
    time_h = tuple(float(t) for t in result_alone.time_h)
    cp_alone_arr = tuple(float(c) for c in result_alone.plasma_concentration())
    cp_ddi_arr = tuple(float(c) for c in result_ddi.plasma_concentration())

    return DDISimulationResult(
        victim_name=victim_drug.name,
        perpetrator_name=perpetrator.name,
        target_enzyme=perpetrator.target_enzyme,
        mechanism=perpetrator.mechanism,
        auc_alone_mg_h_L=float(auc_alone),
        cmax_alone_mg_L=float(cmax_alone),
        t_half_alone_h=float(t_half_alone),
        auc_ddi_mg_h_L=float(auc_ddi),
        cmax_ddi_mg_L=float(cmax_ddi),
        t_half_ddi_h=float(t_half_ddi),
        auc_ratio=float(auc_ratio),
        cmax_ratio=float(cmax_ratio),
        classification=classification,
        perpetrator_cmax_uM=float(cmax_uM),
        static_r1=float(static_r1) if not math.isinf(static_r1) else 999999.0,
        time_h=time_h,
        cp_alone_mg_L=cp_alone_arr,
        cp_ddi_mg_L=cp_ddi_arr,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class PolypharmacyDDIResult:
    """Multi-perpetrator (polypharmacy) DDI simulation result."""

    victim_name: str
    perpetrators: tuple[str, ...]
    n_perpetrators: int
    auc_alone_mg_h_L: float
    cmax_alone_mg_L: float
    t_half_alone_h: float
    auc_combined_mg_h_L: float
    cmax_combined_mg_L: float
    t_half_combined_h: float
    auc_ratio_combined: float
    cmax_ratio_combined: float
    classification_combined: str
    pairwise: tuple[DDISimulationResult, ...]
    synergy_flag: bool
    time_h: tuple[float, ...]
    cp_alone_mg_L: tuple[float, ...]
    cp_combined_mg_L: tuple[float, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def simulate_polypharmacy_ddi(
    victim_drug: Drug,
    perpetrators: list[PerpetratorSpec],
    dose_mg: float = 100.0,
    route: str = "oral",
    body_weight: float = 70.0,
    t_end_h: float = 24.0,
) -> PolypharmacyDDIResult:
    """Multi-perpetrator DDI simulation.

    Simulates victim PK alone, N pairwise (victim + each perpetrator individually),
    and one combined run (victim + all perpetrators simultaneously).

    Synergy is flagged when the combined AUCR exceeds the maximum pairwise AUCR by >10%.

    Args:
        victim_drug: Drug object for the victim compound.
        perpetrators: List of 1–10 PerpetratorSpec objects.
        dose_mg: Victim drug dose (mg).
        route: Administration route ('oral', 'iv', 'sc').
        body_weight: Subject body weight (kg).
        t_end_h: Simulation duration (h).

    Returns:
        PolypharmacyDDIResult with pairwise results and combined interaction metrics.
    """
    if not (1 <= len(perpetrators) <= 10):
        raise ValueError(f"Expected 1–10 perpetrators, got {len(perpetrators)}")

    warnings: list[str] = []

    # --- Baseline simulation (victim alone) ---
    model_alone = WholeBodyPBPK(victim_drug, body_weight=body_weight)
    if route == "oral":
        model_alone.setup_oral(dose_mg)
    elif route == "iv":
        model_alone.setup_iv(dose_mg)
    else:
        model_alone.setup_sc(dose_mg)
    result_alone = model_alone.simulate(t_end_h=t_end_h)
    pk_alone = result_alone.pk_summary()

    auc_alone = pk_alone.get("AUC_mg_h_L", 0.001)
    cmax_alone = pk_alone.get("Cmax_mg_L", 0.001)
    t_half_alone = pk_alone.get("half_life_h", 0.0)
    if math.isinf(t_half_alone) or t_half_alone > 1e6:
        t_half_alone = 0.0

    # --- Pairwise simulations (one per perpetrator) ---
    pairwise_results: list[DDISimulationResult] = []
    for perp in perpetrators:
        pair = simulate_ddi(
            victim_drug=victim_drug,
            perpetrator=perp,
            dose_mg=dose_mg,
            route=route,
            body_weight=body_weight,
            t_end_h=t_end_h,
        )
        pairwise_results.append(pair)

    # --- Combined simulation (all perpetrators simultaneously) ---
    model_combined = WholeBodyPBPK(victim_drug, body_weight=body_weight)
    if route == "oral":
        model_combined.setup_oral(dose_mg)
    elif route == "iv":
        model_combined.setup_iv(dose_mg)
    else:
        model_combined.setup_sc(dose_mg)

    for perp in perpetrators:
        cmax_uM = perp.cmax_uM
        if cmax_uM is None:
            warnings.append(f"No Cmax provided for '{perp.name}'; using default 1.0 µM")
            cmax_uM = 1.0
        inhibitor = DDIInhibitor(
            name=perp.name,
            ki_uM=perp.ki_uM,
            concentration_uM=cmax_uM,
            target_enzyme=perp.target_enzyme,
            mechanism=perp.mechanism,
            kinact_per_h=perp.kinact_per_h,
            kdeg_per_h=perp.kdeg_per_h,
            fold_induction=perp.fold_induction,
        )
        model_combined.add_inhibitor(inhibitor)

    result_combined = model_combined.simulate(t_end_h=t_end_h)
    pk_combined = result_combined.pk_summary()

    auc_combined = pk_combined.get("AUC_mg_h_L", 0.001)
    cmax_combined = pk_combined.get("Cmax_mg_L", 0.001)
    t_half_combined = pk_combined.get("half_life_h", 0.0)
    if math.isinf(t_half_combined) or t_half_combined > 1e6:
        t_half_combined = 0.0

    # --- Ratios ---
    auc_ratio_combined = auc_combined / max(auc_alone, 1e-12)
    cmax_ratio_combined = cmax_combined / max(cmax_alone, 1e-12)

    # --- Synergy: combined AUCR > 110% of max pairwise AUCR ---
    max_pairwise_aucr = max(r.auc_ratio for r in pairwise_results)
    synergy_flag = auc_ratio_combined > max_pairwise_aucr * 1.1

    classification_combined = classify_ddi(auc_ratio_combined)

    # --- Concentration-time profiles ---
    time_h = tuple(float(t) for t in result_alone.time_h)
    cp_alone_arr = tuple(float(c) for c in result_alone.plasma_concentration())
    cp_combined_arr = tuple(float(c) for c in result_combined.plasma_concentration())

    return PolypharmacyDDIResult(
        victim_name=victim_drug.name,
        perpetrators=tuple(p.name for p in perpetrators),
        n_perpetrators=len(perpetrators),
        auc_alone_mg_h_L=float(auc_alone),
        cmax_alone_mg_L=float(cmax_alone),
        t_half_alone_h=float(t_half_alone),
        auc_combined_mg_h_L=float(auc_combined),
        cmax_combined_mg_L=float(cmax_combined),
        t_half_combined_h=float(t_half_combined),
        auc_ratio_combined=float(auc_ratio_combined),
        cmax_ratio_combined=float(cmax_ratio_combined),
        classification_combined=classification_combined,
        pairwise=tuple(pairwise_results),
        synergy_flag=synergy_flag,
        time_h=time_h,
        cp_alone_mg_L=cp_alone_arr,
        cp_combined_mg_L=cp_combined_arr,
        warnings=tuple(warnings),
    )
