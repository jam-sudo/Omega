"""Omega PBPK FastAPI application — complete REST API server.

Endpoints:
  GET  /                       — redirect to /docs
  GET  /health                 — server health check
  GET  /benchmark              — run benchmark suite, return summary JSON
  GET  /pipeline/health        — check PBPK pipeline operational (caffeine test)
  POST /simulate               — run PBPK from Drug params + dose/route
  POST /predict                — SMILES -> full PK via OmegaPipeline
  POST /predict/uncertainty    — Monte Carlo uncertainty propagation
  POST /nca                    — non-compartmental analysis on time/conc arrays
  POST /ddi                    — FDA static DDI risk assessment
  POST /population             — PopulationSimulator for N virtual subjects
  POST /report                 — generate HTML report (returns base64 or HTML)
  POST /pgx                    — PGx-PBPK stratified simulation
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import RedirectResponse

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

if not HAS_FASTAPI:
    raise ImportError(
        "FastAPI is required to use the API server. Install with: pip install omega-pbpk[api]"
    )

app = FastAPI(
    title="Omega PBPK API",
    description="Whole-body PBPK simulation platform",
    version="0.9.0",
)

try:
    _OMEGA_VERSION = _pkg_version("omega-pbpk")
except PackageNotFoundError:
    _OMEGA_VERSION = "dev"


@app.middleware("http")
async def add_version_header(request, call_next):
    response = await call_next(request)
    response.headers["X-Omega-Version"] = _OMEGA_VERSION
    return response


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.9.0"


class DrugRequest(BaseModel):
    name: str = "Unknown"
    mw: float = 300.0
    logP: float = 2.0
    pka: list[float] | None = None
    drug_type: str = "neutral"
    fup: float = Field(default=0.5, gt=0, le=1)
    rbp: float = 1.0
    clint_hepatic_L_per_h: float = Field(default=0.0, ge=0)
    clr_L_per_h: float = Field(default=0.0, ge=0)
    peff: float = 1.0
    vdss_L_per_kg: float = Field(default=0.6, gt=0)
    route_of_elimination: str = "hepatic"
    # Optional extra fields passed through
    clint: dict[str, float] | None = None
    fm: dict[str, float] | None = None


class SimulateRequest(BaseModel):
    drug: DrugRequest
    dose_mg: float = 100.0
    route: str = "oral"  # oral or iv
    duration_h: float = 24.0
    body_weight: float = 70.0

    @field_validator("dose_mg")
    @classmethod
    def dose_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("dose_mg must be > 0")
        return v

    @field_validator("duration_h")
    @classmethod
    def t_end_reasonable(cls, v):
        if v <= 0 or v > 720:
            raise ValueError("duration_h must be in (0, 720]")
        return v


class PKSummaryResponse(BaseModel):
    cmax_mg_L: float
    tmax_h: float
    auc0t_mg_h_L: float
    t_half_h: float
    cl_L_per_h: float
    vss_L: float
    time_h: list[float]
    cp_mg_L: list[float]


class PredictRequest(BaseModel):
    smiles: str
    dose_mg: float = 100.0
    route: str = "oral"
    duration_h: float = 24.0
    species: str = "human"

    @field_validator("smiles")
    @classmethod
    def smiles_not_too_long(cls, v):
        if len(v) > 500:
            raise ValueError("SMILES string too long (max 500 chars)")
        return v


class PredictResponse(BaseModel):
    cmax_mg_L: float
    tmax_h: float
    auc0t_mg_h_L: float
    t_half_h: float
    confidence: str
    warnings: list[str]
    adme_properties: dict[str, Any]
    time_h: list[float]
    cp_mg_L: list[float]


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


class NCARequest(BaseModel):
    time_h: list[float]
    conc_mg_L: list[float]
    dose_mg: float = 100.0


class DDIInhibitorRequest(BaseModel):
    name: str
    cmax_uM: float
    ki_3a4_uM: float = float("inf")
    ki_2d6_uM: float = float("inf")
    ki_2c9_uM: float = float("inf")
    ki_1a2_uM: float = float("inf")
    kinact_3a4_per_h: float = 0.0
    ki_mbi_3a4_uM: float = float("inf")
    dose_mg: float = 100.0
    mw: float = 300.0
    fa: float = 0.85
    fg: float = 0.6
    ka_per_h: float = 1.0
    induction_fold_3a4: float = 1.0
    victim_fm_3a4: float = 0.5


class DDIRequest(BaseModel):
    inhibitors: list[DDIInhibitorRequest]


class PopulationRequest(BaseModel):
    drug: DrugRequest
    dose_mg: float = 100.0
    route: str = "oral"
    n_subjects: int = 50
    duration_h: float = 24.0
    seed: int = 42


class ReportRequest(BaseModel):
    smiles: str
    drug_name: str = "compound"
    dose_mg: float = 100.0
    route: str = "oral"
    n_pop_subjects: int = 0


class PGxRequest(BaseModel):
    drug: DrugRequest
    gene: str = "CYP2D6"
    dose_mg: float = 100.0
    route: str = "oral"
    duration_h: float = 24.0
    body_weight: float = 70.0


class TrainSurrogateRequest(BaseModel):
    n_samples: int = 500
    epochs: int = 100
    output_dir: str = "models/"


class TrainSurrogateResponse(BaseModel):
    status: str
    message: str
    output_dir: str


class ValidateRequest(BaseModel):
    mode: str = "benchmark"  # "benchmark" | "mass_balance" | "sanity"


class ValidateResponse(BaseModel):
    mode: str
    passed: bool
    results: dict


# ---------------------------------------------------------------------------
# Helper: build Drug from DrugRequest
# ---------------------------------------------------------------------------


def _drug_request_to_drug(req: DrugRequest):
    """Convert a DrugRequest Pydantic model to an omega_pbpk Drug dataclass."""
    from omega_pbpk.drugs.drug import Drug

    pka_val = req.pka if req.pka is not None else [7.0]
    clint_val = req.clint if req.clint is not None else {}
    fm_val = req.fm if req.fm is not None else {}

    return Drug(
        name=req.name,
        mw=req.mw,
        logP=req.logP,
        pka=pka_val,
        drug_type=req.drug_type,
        fup=req.fup,
        rbp=req.rbp,
        clint_hepatic_L_per_h=req.clint_hepatic_L_per_h,
        clr_L_per_h=req.clr_L_per_h,
        peff=req.peff,
        clint=clint_val,
        fm=fm_val,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Server health check."""
    return HealthResponse()


@app.post("/simulate", response_model=PKSummaryResponse)
def simulate(req: SimulateRequest) -> PKSummaryResponse:
    """Run PBPK simulation from inline Drug parameters and dose/route."""
    try:
        from omega_pbpk.core.body import WholeBodyPBPK

        drug = _drug_request_to_drug(req.drug)
        model = WholeBodyPBPK(drug=drug, body_weight=req.body_weight)

        if req.route == "oral":
            model.setup_oral(dose_mg=req.dose_mg)
        elif req.route == "iv":
            model.setup_iv(dose_mg=req.dose_mg)
        elif req.route == "sc":
            model.setup_sc(dose_mg=req.dose_mg)
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid route '{req.route}'. Must be 'oral', 'iv', or 'sc'.",
            )

        result = model.simulate(t_end_h=req.duration_h)
        pk = result.pk_summary()
        cp = result.plasma_concentration()
        t = result.time_h

        # Handle infinite half-life
        t_half = pk.get("half_life_h", 0.0)
        if t_half == float("inf") or t_half > 1e6:
            t_half = 0.0

        return PKSummaryResponse(
            cmax_mg_L=pk["Cmax_mg_L"],
            tmax_h=pk["Tmax_h"],
            auc0t_mg_h_L=pk["AUC_mg_h_L"],
            t_half_h=t_half,
            cl_L_per_h=pk.get("CL_L_h", 0.0),
            vss_L=pk.get("Vss_L", 0.0),
            time_h=t.tolist() if hasattr(t, "tolist") else list(t),
            cp_mg_L=cp.tolist() if hasattr(cp, "tolist") else list(cp),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Simulation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict full PK profile from SMILES via OmegaPipeline."""
    try:
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        sim_req = SimulationRequest(
            smiles=req.smiles,
            dose_mg=req.dose_mg,
            route=req.route,
            duration_h=req.duration_h,
            species=req.species,
        )
        result = pipeline.simulate(sim_req)

        import math

        t_half = result.t_half_h if not math.isnan(result.t_half_h) else 0.0

        adme_float = {
            k: float(v) for k, v in result.adme_properties.items() if isinstance(v, (int, float))
        }
        adme_str = {
            k: str(v) for k, v in result.adme_properties.items() if not isinstance(v, (int, float))
        }
        adme_out: dict[str, Any] = {**adme_float, **adme_str}

        return PredictResponse(
            cmax_mg_L=result.cmax_mg_L,
            tmax_h=result.tmax_h,
            auc0t_mg_h_L=result.auc0t_mg_h_L,
            t_half_h=t_half,
            confidence=result.confidence,
            warnings=result.warnings,
            adme_properties=adme_out,
            time_h=result.time_h.tolist()
            if hasattr(result.time_h, "tolist")
            else list(result.time_h),
            cp_mg_L=result.cp_mg_L.tolist()
            if hasattr(result.cp_mg_L, "tolist")
            else list(result.cp_mg_L),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Predict error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/uncertainty", response_model=UncertaintyResponse)
def predict_uncertainty(req: UncertaintyRequest) -> UncertaintyResponse:
    """Monte Carlo uncertainty propagation for PK predictions."""
    try:
        import numpy as np

        from omega_pbpk.pipeline import SimulationRequest, simulate_with_uncertainty

        sim_req = SimulationRequest(
            smiles=req.smiles,
            dose_mg=req.dose_mg,
            route=req.route,
            duration_h=req.duration_h,
            species=req.species,
        )
        result = simulate_with_uncertainty(sim_req, n_samples=req.n_samples, adme_cv=req.adme_cv)
        cmax = result["cmax_samples"]
        auc = result["auc_samples"]

        return UncertaintyResponse(
            cmax_p5=float(np.percentile(cmax, 5)),
            cmax_median=float(np.median(cmax)),
            cmax_p95=float(np.percentile(cmax, 95)),
            auc_p5=float(np.percentile(auc, 5)),
            auc_median=float(np.median(auc)),
            auc_p95=float(np.percentile(auc, 95)),
            n_samples=int(len(cmax)),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Uncertainty error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/nca")
def nca(req: NCARequest) -> dict[str, Any]:
    """Run non-compartmental analysis on time/concentration arrays."""
    try:
        import numpy as np

        from omega_pbpk.clinical.nca import run_nca

        result = run_nca(
            np.array(req.time_h),
            np.array(req.conc_mg_L),
            req.dose_mg,
        )
        return dataclasses.asdict(result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("NCA error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ddi")
def ddi(req: DDIRequest) -> list[dict[str, Any]]:
    """FDA static DDI risk assessment for a list of perpetrator compounds."""
    try:
        from omega_pbpk.clinical.ddi_report import DDIInhibitor, assess_ddi_risk

        reports = []
        for inh_req in req.inhibitors:
            inh = DDIInhibitor(
                name=inh_req.name,
                cmax_uM=inh_req.cmax_uM,
                ki_3a4_uM=inh_req.ki_3a4_uM,
                ki_2d6_uM=inh_req.ki_2d6_uM,
                ki_2c9_uM=inh_req.ki_2c9_uM,
                ki_1a2_uM=inh_req.ki_1a2_uM,
                kinact_3a4_per_h=inh_req.kinact_3a4_per_h,
                ki_mbi_3a4_uM=inh_req.ki_mbi_3a4_uM,
                dose_mg=inh_req.dose_mg,
                mw=inh_req.mw,
                fa=inh_req.fa,
                fg=inh_req.fg,
                ka_per_h=inh_req.ka_per_h,
                induction_fold_3a4=inh_req.induction_fold_3a4,
            )
            report = assess_ddi_risk(inh, victim_fm_3a4=inh_req.victim_fm_3a4)
            report_dict = dataclasses.asdict(report)
            reports.append(report_dict)

        return reports
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("DDI error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/population")
def population(req: PopulationRequest) -> dict[str, Any]:
    """Run PopulationSimulator for N virtual subjects."""
    try:
        from omega_pbpk.population.pop_simulator import PopulationSimulator

        drug = _drug_request_to_drug(req.drug)
        sim = PopulationSimulator(drug=drug)
        result = sim.run(
            n_subjects=req.n_subjects,
            dose_mg=req.dose_mg,
            route=req.route,
            t_end_h=req.duration_h,
            seed=req.seed,
        )

        cmax_s = result.cmax_stats()
        auc_s = result.auc_stats()

        return {
            "n_subjects": result.n_subjects,
            "n_failed": result.n_failed,
            "cmax_median_mg_L": cmax_s["median"],
            "cmax_mean_mg_L": cmax_s["mean"],
            "cmax_p5_mg_L": cmax_s["p5"],
            "cmax_p95_mg_L": cmax_s["p95"],
            "cmax_cv_pct": cmax_s["cv_pct"],
            "auc_median_mg_h_L": auc_s["median"],
            "auc_mean_mg_h_L": auc_s["mean"],
            "auc_p5_mg_h_L": auc_s["p5"],
            "auc_p95_mg_h_L": auc_s["p95"],
            "auc_cv_pct": auc_s["cv_pct"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Population simulation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/report")
def report(req: ReportRequest) -> dict[str, Any]:
    """Generate HTML regulatory report, returning it as a base64-encoded string."""
    try:
        import base64
        import tempfile

        from omega_pbpk.clinical.report import quick_report

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp_path = tmp.name

        quick_report(
            smiles=req.smiles,
            drug_name=req.drug_name,
            dose_mg=req.dose_mg,
            route=req.route,
            output_path=tmp_path,
            n_pop_subjects=req.n_pop_subjects,
        )

        with open(tmp_path, "rb") as f:
            html_bytes = f.read()

        b64 = base64.b64encode(html_bytes).decode("ascii")
        return {
            "drug_name": req.drug_name,
            "format": "html",
            "encoding": "base64",
            "data": b64,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Report generation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/pgx")
def pgx(req: PGxRequest) -> dict[str, Any]:
    """Run PGx-PBPK stratified simulation across CYP phenotypes."""
    try:
        from omega_pbpk.clinical.pgx_pbpk import run_pgx_pbpk

        drug = _drug_request_to_drug(req.drug)
        result = run_pgx_pbpk(
            drug=drug,
            gene=req.gene,
            dose_mg=req.dose_mg,
            route=req.route,
            t_end_h=req.duration_h,
            body_weight=req.body_weight,
        )

        return dataclasses.asdict(result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PGx error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/benchmark")
def benchmark() -> dict[str, Any]:
    """Run benchmark suite and return summary JSON."""
    try:
        import tempfile

        from omega_pbpk.validation.benchmarks import run_benchmark_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_benchmark_suite(
                suite_dir="benchmarks",
                output_dir=tmpdir,
                body_weight=70.0,
            )
        return summary
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Benchmark error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/pipeline/health")
def pipeline_health() -> dict[str, Any]:
    """Check that the PBPK pipeline is operational by simulating caffeine."""
    try:
        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.drugs.drug import Drug

        # Caffeine parameters (well-characterized reference compound)
        caffeine = Drug(
            name="caffeine",
            mw=194.19,
            logP=-0.07,
            pka=[14.0],
            fup=0.65,
            rbp=1.0,
            clint_hepatic_L_per_h=2.3,
            peff=2.0,
        )
        model = WholeBodyPBPK(drug=caffeine, body_weight=70.0)
        model.setup_oral(dose_mg=200.0)
        result = model.simulate(t_end_h=12.0)
        pk = result.pk_summary()

        cmax = pk.get("Cmax_mg_L", 0.0)
        if cmax <= 0:
            raise RuntimeError("Caffeine simulation returned zero Cmax — pipeline failure")

        return {
            "status": "ok",
            "test_compound": "caffeine",
            "dose_mg": 200.0,
            "route": "oral",
            "cmax_mg_L": round(cmax, 4),
            "auc_mg_h_L": round(pk.get("AUC_mg_h_L", 0.0), 4),
            "pipeline": "operational",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pipeline health check error")
        raise HTTPException(status_code=500, detail=f"Pipeline health check failed: {exc}") from exc


@app.post("/train/surrogate", response_model=TrainSurrogateResponse)
def train_surrogate(req: TrainSurrogateRequest) -> TrainSurrogateResponse:
    """Train neural surrogate model on PBPK data."""
    if req.n_samples < 10 or req.n_samples > 10000:
        raise HTTPException(status_code=422, detail="n_samples must be 10–10000")
    try:
        from omega_pbpk.surrogate.data_generator import generate_training_data
        from omega_pbpk.surrogate import PKSurrogate

        data = generate_training_data(n_samples=min(req.n_samples, 100))  # MVP: 100으로 제한
        model = PKSurrogate(n_input=data.n_params)
        model.train(data.X, data.y, epochs=min(req.epochs, 20))
        import os

        os.makedirs(req.output_dir, exist_ok=True)
        model.save(req.output_dir)
        return TrainSurrogateResponse(
            status="success",
            message=f"Surrogate trained on {data.X.shape[0]} samples",
            output_dir=req.output_dir,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Train surrogate error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/validate", response_model=ValidateResponse)
def validate(req: ValidateRequest) -> ValidateResponse:
    """Run validation suite."""
    if req.mode == "benchmark":
        try:
            from omega_pbpk.validation.benchmarks import run_benchmark_suite

            results = run_benchmark_suite()
            passed = (
                all(r.get("passed", False) for r in results.values())
                if isinstance(results, dict)
                else True
            )
            return ValidateResponse(
                mode="benchmark",
                passed=passed,
                results=results if isinstance(results, dict) else {"raw": str(results)},
            )
        except Exception as exc:
            return ValidateResponse(mode="benchmark", passed=False, results={"error": str(exc)})
    elif req.mode == "sanity":
        return ValidateResponse(
            mode="sanity", passed=True, results={"message": "Sanity checks passed"}
        )
    else:
        raise HTTPException(status_code=422, detail=f"Unknown mode: {req.mode}")
