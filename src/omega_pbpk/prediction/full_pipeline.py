"""End-to-end full prediction pipeline: SMILES → PK assessment.

Orchestrates ADME prediction, transporter classification, PBPK simulation,
conformal UQ propagation, and risk flagging into a single call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FullPredictionResult:
    """Complete pharmacokinetic assessment result from SMILES input."""

    drug_name: str
    smiles: str
    # ADME
    adme: dict[str, Any]
    adme_confidence: str
    # Transporters
    transporter_substrates: tuple[str, ...]
    transporter_inhibitors: tuple[str, ...]
    transporter_ddi_flags: tuple[str, ...]
    transporter_confidence: str
    # PK simulation
    cmax_mg_L: float
    tmax_h: float
    auc0t_mg_h_L: float
    t_half_h: float
    time_h: tuple[float, ...]
    cp_mg_L: tuple[float, ...]
    # UQ bands
    cmax_p5: float
    cmax_p50: float
    cmax_p95: float
    auc_p5: float
    auc_p50: float
    auc_p95: float
    # Risk
    risk_flags: dict[str, bool]
    risk_count: int
    overall_risk_level: str  # "low", "moderate", "high"
    # Meta
    confidence: str
    pubchem_enriched: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)


def run_full_prediction(
    smiles: str,
    dose_mg: float = 100.0,
    route: str = "oral",
    duration_h: float = 24.0,
    n_uq_samples: int = 50,
) -> FullPredictionResult:
    """Chain all prediction subsystems for a complete PK assessment.

    Flow: ADME prediction → transporter classification → PBPK simulation
          → UQ propagation → risk flags → assemble result.

    Graceful degradation: each subsystem failure adds a warning and uses
    safe defaults rather than crashing the pipeline.

    Args:
        smiles: Valid SMILES string for the compound.
        dose_mg: Dose in milligrams.
        route: Administration route ('oral' or 'iv').
        duration_h: Simulation duration in hours.
        n_uq_samples: Number of LHS samples for UQ propagation.

    Returns:
        FullPredictionResult with all subsystem outputs.
    """
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. ADME prediction
    # ------------------------------------------------------------------
    adme_props = None
    adme_confidence = "low"
    adme_dict: dict[str, Any] = {}

    try:
        from omega_pbpk.prediction.adme_predictor import ADMEPredictor

        predictor = ADMEPredictor()
        adme_props = predictor.predict(smiles)
        adme_confidence = adme_props.confidence
        adme_dict = {
            "mw": adme_props.mw,
            "logP": adme_props.logP,
            "logS": adme_props.logS,
            "peff": adme_props.peff,
            "fup": adme_props.fup,
            "rbp": adme_props.rbp,
            "clint_3a4": adme_props.clint_3a4,
            "clint_2d6": adme_props.clint_2d6,
            "herg_ic50_uM": adme_props.herg_ic50_uM,
        }
    except Exception as e:  # noqa: BLE001
        warnings.append(f"ADME prediction failed: {e}")

    # ------------------------------------------------------------------
    # 1b. PubChem enrichment — override featurizer values when available
    # ------------------------------------------------------------------
    pubchem_enriched = False
    try:
        from omega_pbpk.data.pubchem_client import lookup_by_smiles

        pc = lookup_by_smiles(smiles)
        if pc is not None and adme_props is not None:
            # Override mw and logP in adme_dict with authoritative PubChem values
            if pc.molecular_weight > 0:
                adme_dict["mw"] = pc.molecular_weight
            if pc.xlogp != 0.0:
                adme_dict["logP"] = pc.xlogp
            if pc.tpsa > 0:
                adme_dict["tpsa"] = pc.tpsa
            pubchem_enriched = True
    except Exception as e:  # noqa: BLE001
        warnings.append(f"PubChem enrichment failed: {e}")

    # ------------------------------------------------------------------
    # 2. Transporter classification
    # ------------------------------------------------------------------
    substrates: tuple[str, ...] = ()
    inhibitors: tuple[str, ...] = ()
    ddi_flags: tuple[str, ...] = ()
    transporter_confidence = "low"

    try:
        from omega_pbpk.prediction.transporter_classifier import TransporterClassifier

        tc = TransporterClassifier()
        tp = tc.predict(smiles)
        substrates = tuple(tp.active_substrates())
        inhibitors = tuple(tp.active_inhibitors())
        ddi_flags = tp.ddi_risk_flags
        transporter_confidence = tp.confidence
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Transporter classification failed: {e}")

    # ------------------------------------------------------------------
    # 3. PBPK simulation via OmegaPipeline
    # ------------------------------------------------------------------
    cmax_mg_L = 0.0
    tmax_h = 0.0
    auc0t_mg_h_L = 0.0
    t_half_h = 0.0
    time_h_tup: tuple[float, ...] = ()
    cp_mg_L_tup: tuple[float, ...] = ()
    sim_confidence = "low"

    try:
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        sim_req = SimulationRequest(
            smiles=smiles,
            dose_mg=dose_mg,
            route=route,
            duration_h=duration_h,
        )
        sim_result = pipeline.simulate(sim_req)
        cmax_mg_L = sim_result.cmax_mg_L
        tmax_h = sim_result.tmax_h
        auc0t_mg_h_L = sim_result.auc0t_mg_h_L
        raw_t_half = sim_result.t_half_h
        t_half_h = raw_t_half if not math.isnan(raw_t_half) else 0.0
        time_h_tup = tuple(float(x) for x in sim_result.time_h)
        cp_mg_L_tup = tuple(float(x) for x in sim_result.cp_mg_L)
        sim_confidence = sim_result.confidence
        warnings.extend(sim_result.warnings)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"PBPK simulation failed: {e}")

    # ------------------------------------------------------------------
    # 4. Conformal UQ propagation
    # ------------------------------------------------------------------
    cmax_p5 = cmax_p50 = cmax_p95 = 0.0
    auc_p5 = auc_p50 = auc_p95 = 0.0

    try:
        if adme_props is not None:
            from omega_pbpk.uncertainty.conformal_uq import (
                build_bounds_from_adme,
                propagate_conformal_intervals,
            )

            bounds = build_bounds_from_adme(adme_props)
            uq = propagate_conformal_intervals(
                drug_name=smiles[:20],
                dose_mg=dose_mg,
                route=route,
                bounds=bounds,
                n_samples=n_uq_samples,
            )
            cmax_p5 = uq.cmax_p5
            cmax_p50 = uq.cmax_p50
            cmax_p95 = uq.cmax_p95
            auc_p5 = uq.auc_p5
            auc_p50 = uq.auc_p50
            auc_p95 = uq.auc_p95
        else:
            warnings.append("UQ skipped: no ADME properties available")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"UQ propagation failed: {e}")

    # ------------------------------------------------------------------
    # 5. Risk flags
    # ------------------------------------------------------------------
    risk_flags_dict: dict[str, bool] = {}
    risk_count = 0

    try:
        from omega_pbpk.risk import compute_risk_flags

        exposure_cv = (auc_p95 - auc_p5) / (2.0 * auc_p50) if auc_p50 > 1e-12 else 0.0
        ddi_score = min(1.0, len(inhibitors) * 0.25)
        risk = compute_risk_flags(
            cmax_mg_per_l=cmax_mg_L,
            half_life_h=t_half_h,
            exposure_cv=exposure_cv,
            ddi_risk_score=ddi_score,
        )
        risk_flags_dict = risk.summary()
        risk_count = risk.risk_count
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Risk assessment failed: {e}")

    overall_risk = "low" if risk_count == 0 else ("moderate" if risk_count <= 2 else "high")

    # ------------------------------------------------------------------
    # 6. Combine confidence: minimum across subsystems
    # ------------------------------------------------------------------
    _conf_order = {"high": 3, "medium": 2, "low": 1}
    min_conf = min(
        _conf_order.get(adme_confidence, 1),
        _conf_order.get(transporter_confidence, 1),
        _conf_order.get(sim_confidence, 1),
    )
    confidence = {3: "high", 2: "medium", 1: "low"}[min_conf]

    drug_name = f"compound_{smiles[:8]}"

    return FullPredictionResult(
        drug_name=drug_name,
        smiles=smiles,
        adme=adme_dict,
        adme_confidence=adme_confidence,
        transporter_substrates=substrates,
        transporter_inhibitors=inhibitors,
        transporter_ddi_flags=ddi_flags,
        transporter_confidence=transporter_confidence,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc0t_mg_h_L=auc0t_mg_h_L,
        t_half_h=t_half_h,
        time_h=time_h_tup,
        cp_mg_L=cp_mg_L_tup,
        cmax_p5=cmax_p5,
        cmax_p50=cmax_p50,
        cmax_p95=cmax_p95,
        auc_p5=auc_p5,
        auc_p50=auc_p50,
        auc_p95=auc_p95,
        risk_flags=risk_flags_dict,
        risk_count=risk_count,
        overall_risk_level=overall_risk,
        confidence=confidence,
        pubchem_enriched=pubchem_enriched,
        warnings=tuple(warnings),
    )


__all__ = ["FullPredictionResult", "run_full_prediction"]
