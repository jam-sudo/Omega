"""FastAPI server with 10 REST API endpoints for Omega PBPK.

Endpoints:
  GET  /health              — Server health check
  POST /predict             — SMILES → ADME properties
  POST /simulate/iv         — IV bolus simulation
  POST /simulate/oral       — Oral simulation
  POST /simulate/ddi        — DDI simulation
  POST /simulate/multidose  — Multi-dose steady state
  POST /safety              — Off-target safety panel
  POST /pgx                 — Pharmacogenomics analysis
  POST /optimize            — Dose optimization
  POST /population          — Population PK simulation

Requires: pip install fastapi uvicorn
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.info("FastAPI not installed. API server is scaffold-only.")


def create_app() -> Any:
    """Create and configure the FastAPI application."""
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Omega PBPK API",
        description="Whole-body PBPK simulation platform",
        version="0.7.0",
    )

    # --- Request/Response models ---

    class HealthResponse(BaseModel):
        status: str = "ok"
        version: str = "0.7.0"

    class PredictRequest(BaseModel):
        smiles: str

    class SimulateRequest(BaseModel):
        compound_yaml: str | None = None
        smiles: str | None = None
        dose_mg: float = 10.0
        t_end_h: float = 24.0
        body_weight_kg: float = 70.0

    class DDIRequest(BaseModel):
        compound_yaml: str | None = None
        smiles: str | None = None
        dose_mg: float = 10.0
        inhibitor_name: str = ""
        inhibitor_ki_uM: float = 1.0
        inhibitor_conc_uM: float = 1.0
        target_enzyme: str = "CYP3A4"
        mechanism: str = "competitive"

    class MultiDoseRequest(BaseModel):
        compound_yaml: str | None = None
        dose_mg: float = 10.0
        interval_h: float = 12.0
        n_days: int = 7

    class SafetyRequest(BaseModel):
        smiles: str | None = None
        logP: float = 2.0
        mw: float = 300.0
        cmax_uM: float = 1.0
        fup: float = 0.5

    class PGxRequest(BaseModel):
        gene: str = "CYP2D6"
        population: str = "Global"

    class OptimizeRequest(BaseModel):
        compound_yaml: str | None = None
        mec_mg_L: float = 1.0
        mtc_mg_L: float = 100.0

    class PopulationRequest(BaseModel):
        compound_yaml: str | None = None
        dose_mg: float = 10.0
        route: str = "oral"
        n_subjects: int = 50

    # --- Endpoints ---

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict[str, Any]:
        from omega_pbpk.prediction.adme_predictor import ADMEPredictor

        predictor = ADMEPredictor()
        props = predictor.predict(req.smiles)
        return {k: v for k, v in props.__dict__.items()}

    @app.post("/simulate/iv")
    def simulate_iv(req: SimulateRequest) -> dict[str, Any]:
        drug = _load_drug(req.compound_yaml, req.smiles)
        from omega_pbpk.core.body import WholeBodyPBPK

        model = WholeBodyPBPK(drug, body_weight=req.body_weight_kg)
        model.setup_iv(req.dose_mg)
        result = model.simulate(t_end_h=req.t_end_h)
        return result.pk_summary()

    @app.post("/simulate/oral")
    def simulate_oral(req: SimulateRequest) -> dict[str, Any]:
        drug = _load_drug(req.compound_yaml, req.smiles)
        from omega_pbpk.core.body import WholeBodyPBPK

        model = WholeBodyPBPK(drug, body_weight=req.body_weight_kg)
        model.setup_oral(req.dose_mg)
        result = model.simulate(t_end_h=req.t_end_h)
        return result.pk_summary()

    @app.post("/simulate/ddi")
    def simulate_ddi(req: DDIRequest) -> dict[str, Any]:
        drug = _load_drug(req.compound_yaml, req.smiles)
        from omega_pbpk.core.body import DDIInhibitor, WholeBodyPBPK

        model = WholeBodyPBPK(drug)
        model.setup_oral(req.dose_mg)
        model.add_inhibitor(
            DDIInhibitor(
                name=req.inhibitor_name,
                ki_uM=req.inhibitor_ki_uM,
                concentration_uM=req.inhibitor_conc_uM,
                target_enzyme=req.target_enzyme,
                mechanism=req.mechanism,
            )
        )
        result = model.simulate(t_end_h=24.0)
        return result.pk_summary()

    @app.post("/simulate/multidose")
    def simulate_multidose(req: MultiDoseRequest) -> dict[str, Any]:
        drug = _load_drug(req.compound_yaml)
        from omega_pbpk.clinical.dose_optimization import MultiDoseSimulator

        sim = MultiDoseSimulator()
        result = sim.simulate(drug, req.dose_mg, req.interval_h, req.n_days)
        return {
            "css_max": result.css_max,
            "css_min": result.css_min,
            "css_avg": result.css_avg,
            "accumulation_ratio": result.accumulation_ratio,
            "fluctuation_percent": result.fluctuation_percent,
        }

    @app.post("/safety")
    def safety(req: SafetyRequest) -> dict[str, Any]:
        from omega_pbpk.docking.off_target import SafetyPanel

        panel = SafetyPanel()
        report = panel.assess(
            compound_name=req.smiles or "Unknown",
            logP=req.logP,
            mw=req.mw,
            cmax_uM=req.cmax_uM,
            fup=req.fup,
        )
        return {
            "overall_risk": report.overall_risk,
            "flags": report.flags,
            "n_targets_assessed": len(report.target_results),
            "n_cyp_assessed": len(report.cyp_results),
        }

    @app.post("/pgx")
    def pgx(req: PGxRequest) -> dict[str, Any]:
        from omega_pbpk.pharmacogenomics.cyp_polymorphism import PGxAnalyzer

        analyzer = PGxAnalyzer()
        return analyzer.population_summary(req.gene, req.population)

    @app.post("/optimize")
    def optimize(req: OptimizeRequest) -> dict[str, Any]:
        drug = _load_drug(req.compound_yaml)
        from omega_pbpk.clinical.dose_optimization import DoseOptimizer

        optimizer = DoseOptimizer()
        return optimizer.optimize_dose(drug, req.mec_mg_L, req.mtc_mg_L)

    @app.post("/population")
    def population(req: PopulationRequest) -> dict[str, Any]:
        drug = _load_drug(req.compound_yaml)
        import numpy as np

        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.population.physiology import VirtualPopulation

        pop = VirtualPopulation(n=req.n_subjects)
        subjects = pop.generate()
        cmax_values = []

        for subj in subjects:
            model = WholeBodyPBPK(drug, body_weight=subj.body_weight_kg)
            if req.route == "iv":
                model.setup_iv(req.dose_mg)
            else:
                model.setup_oral(req.dose_mg)
            result = model.simulate(t_end_h=24.0)
            pk = result.pk_summary()
            cmax_values.append(pk["Cmax_mg_L"])

        arr = np.array(cmax_values)
        return {
            "n_subjects": req.n_subjects,
            "Cmax_median": round(float(np.median(arr)), 6),
            "Cmax_mean": round(float(np.mean(arr)), 6),
            "Cmax_cv_pct": round(float(np.std(arr) / np.mean(arr) * 100), 1),
            "Cmax_5th_pct": round(float(np.percentile(arr, 5)), 6),
            "Cmax_95th_pct": round(float(np.percentile(arr, 95)), 6),
        }

    def _load_drug(
        compound_yaml: str | None = None,
        smiles: str | None = None,
    ) -> Any:
        """Load drug from YAML path or SMILES string."""
        if compound_yaml:
            from omega_pbpk.config import load_compound

            return load_compound(compound_yaml)
        elif smiles:
            from omega_pbpk.drugs.drug import Drug
            from omega_pbpk.prediction.adme_predictor import ADMEPredictor

            predictor = ADMEPredictor()
            props = predictor.predict(smiles)
            return Drug(
                name="Predicted",
                mw=props.mw,
                logP=props.logP,
                fup=props.fup,
                rbp=props.rbp,
                peff=props.peff,
                clint={"CYP3A4": props.clint_3a4, "CYP2D6": props.clint_2d6},
            )
        else:
            raise HTTPException(400, "Either compound_yaml or smiles must be provided")

    return app
