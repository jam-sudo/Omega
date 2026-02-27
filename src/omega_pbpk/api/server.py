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
        version="0.9.0",
    )

    # --- Request/Response models ---

    class HealthResponse(BaseModel):
        status: str = "ok"
        version: str = "0.9.0"

    class PredictRequest(BaseModel):
        smiles: str

    class SimulateRequest(BaseModel):
        compound_yaml: str | None = None
        smiles: str | None = None
        dose_mg: float = 10.0
        t_end_h: float = 24.0
        body_weight_kg: float = 70.0
        partition_method: str = "heuristic"  # "heuristic" or "rodgers_rowland"

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


    class NewMoleculeRequest(BaseModel):
        smiles: str
        dose_mg: float = 100.0
        route: str = "oral"
        duration_h: float = 24.0
        species: str = "human"

    class NewMoleculeResponse(BaseModel):
        cmax_mg_L: float
        tmax_h: float
        auc0t_mg_h_L: float
        t_half_h: float
        confidence: str
        warnings: list[str]
        adme_properties: dict[str, float]

    class UncertaintyRequest(BaseModel):
        smiles: str
        dose_mg: float = 100.0
        route: str = "oral"
        duration_h: float = 24.0
        species: str = "human"
        n_samples: int = 100
        adme_cv: float = 0.3

    class UncertaintyResponse(BaseModel):
        cmax_p5: float
        cmax_median: float
        cmax_p95: float
        auc_p5: float
        auc_median: float
        auc_p95: float
        n_samples: int

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
        drug = _load_drug(req.compound_yaml, req.smiles, req.partition_method)
        from omega_pbpk.core.body import WholeBodyPBPK

        model = WholeBodyPBPK(drug, body_weight=req.body_weight_kg)
        model.setup_iv(req.dose_mg)
        result = model.simulate(t_end_h=req.t_end_h)
        return result.pk_summary()

    @app.post("/simulate/oral")
    def simulate_oral(req: SimulateRequest) -> dict[str, Any]:
        drug = _load_drug(req.compound_yaml, req.smiles, req.partition_method)
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


    @app.post("/predict/new-molecule")
    async def predict_new_molecule(body: NewMoleculeRequest) -> NewMoleculeResponse:
        """Predict PK profile for any drug given by SMILES string."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest
        pipeline = _get_pipeline()
        req = SimulationRequest(
            smiles=body.smiles,
            dose_mg=body.dose_mg,
            route=body.route,
            duration_h=body.duration_h,
            species=body.species,
        )
        result = pipeline.simulate(req)
        adme_float = {k: float(v) for k, v in result.adme_properties.items() if isinstance(v, (int, float))}
        return NewMoleculeResponse(
            cmax_mg_L=result.cmax_mg_L,
            tmax_h=result.tmax_h,
            auc0t_mg_h_L=result.auc0t_mg_h_L,
            t_half_h=float("nan") if result.t_half_h != result.t_half_h else result.t_half_h,
            confidence=result.confidence,
            warnings=result.warnings,
            adme_properties=adme_float,
        )

    @app.post("/predict/uncertainty")
    async def predict_with_uncertainty(body: UncertaintyRequest) -> UncertaintyResponse:
        """Monte Carlo uncertainty quantification for PK predictions."""
        import numpy as np
        from omega_pbpk.pipeline import SimulationRequest, simulate_with_uncertainty
        req = SimulationRequest(
            smiles=body.smiles,
            dose_mg=body.dose_mg,
            route=body.route,
            duration_h=body.duration_h,
            species=body.species,
        )
        result = simulate_with_uncertainty(req, n_samples=body.n_samples, adme_cv=body.adme_cv)
        cmax = result["cmax_samples"]
        auc = result["auc_samples"]
        return UncertaintyResponse(
            cmax_p5=float(np.percentile(cmax, 5)),
            cmax_median=float(np.median(cmax)),
            cmax_p95=float(np.percentile(cmax, 95)),
            auc_p5=float(np.percentile(auc, 5)),
            auc_median=float(np.median(auc)),
            auc_p95=float(np.percentile(auc, 95)),
            n_samples=len(cmax),
        )

    @app.get("/pipeline/health")
    async def pipeline_health() -> dict:
        """Check if the full pipeline (ADME predictor + PBPK) is operational."""
        from omega_pbpk.pipeline import OmegaPipeline
        pipeline = _get_pipeline()
        adme_ok = pipeline._adme_predictor is not None
        return {
            "status": "ok",
            "adme_predictor": "available" if adme_ok else "unavailable (using defaults)",
            "pbpk_model": "available",
            "pipeline": "operational",
        }


    # Pipeline singleton for reuse
    _pipeline_instance = None

    def _get_pipeline():
        nonlocal _pipeline_instance
        if _pipeline_instance is None:
            from omega_pbpk.pipeline import OmegaPipeline
            _pipeline_instance = OmegaPipeline()
            _pipeline_instance._ensure_initialized()
        return _pipeline_instance

    @app.on_event("startup")
    async def startup_event():
        """Initialize pipeline at startup."""
        _get_pipeline()

    def _load_drug(
        compound_yaml: str | None = None,
        smiles: str | None = None,
        partition_method: str = "heuristic",
    ) -> Any:
        """Load drug from YAML path or SMILES string.

        Args:
            compound_yaml: Path to compound YAML file.
            smiles: SMILES string for ADME prediction.
            partition_method: Kp estimation method — 'heuristic' (default,
                simplified Poulin & Theil 2002) or 'rodgers_rowland'
                (mechanistic, Rodgers & Rowland 2006).
        """
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
                partition_method=partition_method,
            )
        else:
            raise HTTPException(400, "Either compound_yaml or smiles must be provided")

    return app
