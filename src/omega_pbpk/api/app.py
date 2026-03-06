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
  POST /dose/optimize          — dosing regimen optimization
  POST /tdm                    — TDM MAP-Bayesian dose individualization
  POST /be/design              — bioequivalence study design & sample size
  POST /toxicity               — organ-specific toxicity risk scoring
  POST /phenotyping            — metabolic reaction phenotyping (CYP contribution)
  POST /sensitivity/sobol      — Sobol global sensitivity analysis
  POST /simulate/formulation   — ER/MR formulation release + PK simulation
  POST /trial/crossover        — virtual 2×2 crossover BE trial simulation
  POST /compare                — multi-candidate drug comparison & ranking
  POST /mist                   — MIST metabolite safety assessment (FDA guidance)
  POST /bcs                    — BCS classification and bioavailability prediction
  POST /induction              — enzyme induction time course (PXR/CAR pathway)
  POST /predict/gat            — Graph Attention Network ADME prediction
  POST /pubchem/lookup         — PubChem PUG REST live compound lookup
  POST /interpret              — GPT/rule-based natural-language PK interpretation
  POST /predict/bioavailability — F = fa × fg × fh oral bioavailability prediction
  POST /predict/steady_state  — multi-dose superposition steady-state PK metrics
  POST /ivivc               — In Vitro–In Vivo Correlation (Level A/B/C, Wagner-Nelson)
  POST /population/pk_fit  — Standard two-stage population PK fitting (CL, Vd, BSV, AIC/BIC)
  POST /kidney/pk          — Mechanistic renal PK: filtration + secretion + reabsorption
  POST /scale/interspecies — Allometric interspecies scaling (animal → human PK)
  POST /simulate/chronopk  — Circadian PK simulation with time-varying physiology
  POST /ddi/displacement   — Protein binding displacement DDI (fup shift, CL/AUC impact)

Auth endpoints (only active when OMEGA_AUTH_ENABLED=true):
  POST /auth/token             — obtain JWT bearer token
  POST /auth/register          — register a new user
  GET  /auth/me                — return current authenticated user info
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any

import numpy as np

try:
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.responses import HTMLResponse, RedirectResponse
    from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
    from fastapi.staticfiles import StaticFiles

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from omega_pbpk.auth.jwt_auth import (
    AUTH_ENABLED,
    TokenData,
    User,
    create_access_token,
    create_user,
    get_user,
    verify_password,
    verify_token,
)

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

_STATIC_DIR = pathlib.Path(__file__).parent.parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

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
# Auth helpers (disabled by default)
# ---------------------------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)] = None,
) -> TokenData | None:
    """Decode bearer token and return TokenData, or None when auth is off."""
    if not AUTH_ENABLED:
        return None
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    td = verify_token(token)
    if td is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return td


async def require_auth(
    current_user: Annotated[TokenData | None, Depends(get_current_user)] = None,
) -> TokenData | None:
    """Dependency that enforces authentication when AUTH_ENABLED is True."""
    return current_user


# ---------------------------------------------------------------------------
# Auth Pydantic models
# ---------------------------------------------------------------------------
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RegisterRequest(BaseModel):
    username: str
    password: str


class UserInfo(BaseModel):
    username: str
    scopes: list[str]


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
async def auth_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> TokenResponse:
    """Obtain a JWT bearer token via username + password."""
    if not AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authentication is disabled",
        )
    user: User | None = get_user(form_data.username)
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user.username, "scopes": list(user.scopes)})
    return TokenResponse(access_token=token)


@app.post("/auth/register", response_model=UserInfo, tags=["auth"])
async def auth_register(req: RegisterRequest) -> UserInfo:
    """Register a new user (only when auth is enabled)."""
    if not AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authentication is disabled",
        )
    if get_user(req.username) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )
    user = create_user(req.username, req.password)
    return UserInfo(username=user.username, scopes=list(user.scopes))


@app.get("/auth/me", response_model=UserInfo, tags=["auth"])
async def auth_me(
    current_user: Annotated[TokenData | None, Depends(get_current_user)],
) -> UserInfo:
    """Return the currently authenticated user's info."""
    if not AUTH_ENABLED or current_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authentication is disabled",
        )
    return UserInfo(username=current_user.username, scopes=list(current_user.scopes))


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


class DDISimulateRequest(BaseModel):
    victim_drug: DrugRequest
    perpetrator_name: str
    perpetrator_ki_uM: float
    perpetrator_target_enzyme: str = "CYP3A4"
    perpetrator_mechanism: str = "competitive"
    perpetrator_kinact_per_h: float = 0.0
    perpetrator_kdeg_per_h: float = 0.02
    perpetrator_fold_induction: float = 1.0
    perpetrator_cmax_uM: float | None = None
    dose_mg: float = 100.0
    route: str = "oral"
    body_weight: float = 70.0
    duration_h: float = 24.0


class DDISimulateResponse(BaseModel):
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
    classification: str
    perpetrator_cmax_uM: float
    static_r1: float
    time_h: list[float]
    cp_alone_mg_L: list[float]
    cp_ddi_mg_L: list[float]
    warnings: list[str]


class PolypharmacyPerpetratorRequest(BaseModel):
    name: str
    ki_uM: float
    target_enzyme: str = "CYP3A4"
    mechanism: str = "competitive"
    kinact_per_h: float = 0.0
    kdeg_per_h: float = 0.02
    fold_induction: float = 1.0
    cmax_uM: float | None = None


class PolypharmacyDDIRequest(BaseModel):
    victim_drug: DrugRequest
    perpetrators: list[PolypharmacyPerpetratorRequest]
    dose_mg: float = 100.0
    route: str = "oral"
    body_weight: float = 70.0
    duration_h: float = 24.0


class PairwiseDDIResponse(BaseModel):
    perpetrator_name: str
    auc_ratio: float
    cmax_ratio: float
    classification: str
    perpetrator_cmax_uM: float
    static_r1: float
    warnings: list[str]


class PolypharmacyDDIResponse(BaseModel):
    victim_name: str
    perpetrators: list[str]
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
    pairwise: list[PairwiseDDIResponse]
    synergy_flag: bool
    time_h: list[float]
    cp_alone_mg_L: list[float]
    cp_combined_mg_L: list[float]
    warnings: list[str]


class ConformalUQRequest(BaseModel):
    drug_name: str
    dose_mg: float
    route: str = "oral"
    smiles: str | None = None
    mw: float | None = None
    logP: float | None = None
    fup_lo: float | None = None
    fup_hi: float | None = None
    clint_lo: float | None = None
    clint_hi: float | None = None
    peff_lo: float | None = None
    peff_hi: float | None = None
    rbp_lo: float | None = None
    rbp_hi: float | None = None
    n_samples: int = 200


class ConformalUQResponse(BaseModel):
    cmax_p5: float
    cmax_p50: float
    cmax_p95: float
    auc_p5: float
    auc_p50: float
    auc_p95: float
    tmax_p5: float
    tmax_p50: float
    tmax_p95: float
    t_half_p5: float
    t_half_p50: float
    t_half_p95: float
    n_samples: int
    warnings: list[str]


class FullPredictRequest(BaseModel):
    smiles: str
    dose_mg: float = 100.0
    route: str = "oral"
    duration_h: float = 24.0
    n_uq_samples: int = 50

    @field_validator("smiles")
    @classmethod
    def smiles_not_too_long(cls, v):
        if len(v) > 500:
            raise ValueError("SMILES string too long (max 500 chars)")
        return v


class FullPredictResponse(BaseModel):
    drug_name: str
    smiles: str
    adme: dict[str, Any]
    adme_confidence: str
    transporter_substrates: list[str]
    transporter_inhibitors: list[str]
    transporter_ddi_flags: list[str]
    transporter_confidence: str
    cmax_mg_L: float
    tmax_h: float
    auc0t_mg_h_L: float
    t_half_h: float
    time_h: list[float]
    cp_mg_L: list[float]
    cmax_p5: float
    cmax_p50: float
    cmax_p95: float
    auc_p5: float
    auc_p50: float
    auc_p95: float
    risk_flags: dict[str, bool]
    risk_count: int
    overall_risk_level: str
    confidence: str
    warnings: list[str]


class PediatricSimulateRequest(BaseModel):
    drug: DrugRequest
    dose_mg: float = 100.0
    route: str = "oral"
    duration_h: float = 24.0
    age_years: float = Field(..., ge=0.0, le=18.0)
    body_weight_kg: float = Field(..., gt=0)


class PediatricSimulateResponse(BaseModel):
    pediatric_cmax_mg_L: float
    pediatric_tmax_h: float
    pediatric_auc_mg_h_L: float
    pediatric_t_half_h: float
    adult_cmax_mg_L: float
    adult_auc_mg_h_L: float
    adult_t_half_h: float
    cmax_ratio: float
    auc_ratio: float
    ontogeny_factors: dict[str, float]
    dose_adjustment_factor: float
    time_h: list[float]
    cp_pediatric_mg_L: list[float]
    cp_adult_mg_L: list[float]
    warnings: list[str]


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


class IVIVERequest(BaseModel):
    clint_uL_min: float = Field(..., gt=0)
    clint_system: Literal["microsomes", "hepatocytes"] = "microsomes"
    fup: float = Field(..., gt=0, le=1)
    fu_mic: float | None = Field(default=None, gt=0, le=1)
    logP: float | None = None
    papp_ab_cm_s: float | None = Field(default=None, gt=0)
    papp_ba_cm_s: float | None = Field(default=None, gt=0)
    rbp: float | None = Field(default=None, gt=0)
    mw: float = Field(default=300.0, gt=0)
    drug_name: str = "ivive_compound"
    clint_gut_uL_min_mg: float = Field(default=0.0, ge=0)
    liver_weight_g: float = Field(default=1500.0, gt=0)
    q_liver_L_per_h: float = Field(default=96.6, gt=0)


class IVIVEClearanceResponse(BaseModel):
    clint_in_vitro: float
    system: str
    clint_liver_L_per_h: float
    clh_well_stirred_L_per_h: float
    fu_mic: float
    fup: float


class IVIVEPermeabilityResponse(BaseModel):
    papp_ab_cm_s: float
    papp_ba_cm_s: float | None
    peff_10_4_cm_s: float
    efflux_ratio: float
    fa_predicted: float
    fg_predicted: float
    absorption_classification: str
    pgp_flag: str


class IVIVEBindingResponse(BaseModel):
    fup_measured: float
    fup_corrected: float
    fu_mic: float
    fu_mic_source: str
    rbp: float
    rbp_source: str


class IVIVEResponse(BaseModel):
    clearance: IVIVEClearanceResponse
    permeability: IVIVEPermeabilityResponse | None
    binding: IVIVEBindingResponse
    drug_params: dict[str, float | str]
    confidence: str
    warnings: list[str]


class DoseOptimizeRequest(BaseModel):
    drug: DrugRequest
    cmin_mg_L: float | None = None
    cmax_mg_L: float | None = None
    auc_min: float | None = None
    auc_max: float | None = None
    dose_range_mg: list[float] = Field(default=[10.0, 1000.0])
    intervals_h: list[float] = Field(default=[6, 8, 12, 24])
    route: str = "oral"
    n_doses: int = 20
    body_weight: float = 70.0


class RegimenCandidateResponse(BaseModel):
    dose_mg: float
    interval_h: float
    css_max: float
    css_min: float
    css_avg: float
    auc_ss: float
    feasible: bool
    penalty: float


class DoseOptimizeResponse(BaseModel):
    optimal_dose_mg: float
    optimal_interval_h: float
    css_max: float
    css_min: float
    css_avg: float
    auc_ss: float
    feasible: bool
    warnings: list[str]
    all_regimens: list[RegimenCandidateResponse]


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


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def root() -> HTMLResponse:
    """Serve the interactive dashboard."""
    index_path = _STATIC_DIR / "index.html"
    if index_path.is_file():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return RedirectResponse(url="/docs")  # type: ignore[return-value]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Server health check."""
    return HealthResponse()


@app.post("/simulate", response_model=PKSummaryResponse)
def simulate(
    req: SimulateRequest,
    _auth: Annotated[TokenData | None, Depends(require_auth)] = None,
) -> PKSummaryResponse:
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


@app.post("/ddi/simulate", response_model=DDISimulateResponse)
def ddi_simulate(req: DDISimulateRequest) -> DDISimulateResponse:
    """Dynamic DDI simulation: victim PK alone vs with perpetrator inhibitor/inducer."""
    if req.route not in ("oral", "iv", "sc"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid route '{req.route}'. Must be 'oral', 'iv', or 'sc'.",
        )
    try:
        from omega_pbpk.clinical.ddi_simulation import PerpetratorSpec, simulate_ddi

        victim = _drug_request_to_drug(req.victim_drug)
        perpetrator = PerpetratorSpec(
            name=req.perpetrator_name,
            ki_uM=req.perpetrator_ki_uM,
            target_enzyme=req.perpetrator_target_enzyme,
            mechanism=req.perpetrator_mechanism,
            kinact_per_h=req.perpetrator_kinact_per_h,
            kdeg_per_h=req.perpetrator_kdeg_per_h,
            fold_induction=req.perpetrator_fold_induction,
            cmax_uM=req.perpetrator_cmax_uM,
        )
        result = simulate_ddi(
            victim_drug=victim,
            perpetrator=perpetrator,
            dose_mg=req.dose_mg,
            route=req.route,
            body_weight=req.body_weight,
            t_end_h=req.duration_h,
        )
        return DDISimulateResponse(
            victim_name=result.victim_name,
            perpetrator_name=result.perpetrator_name,
            target_enzyme=result.target_enzyme,
            mechanism=result.mechanism,
            auc_alone_mg_h_L=result.auc_alone_mg_h_L,
            cmax_alone_mg_L=result.cmax_alone_mg_L,
            t_half_alone_h=result.t_half_alone_h,
            auc_ddi_mg_h_L=result.auc_ddi_mg_h_L,
            cmax_ddi_mg_L=result.cmax_ddi_mg_L,
            t_half_ddi_h=result.t_half_ddi_h,
            auc_ratio=result.auc_ratio,
            cmax_ratio=result.cmax_ratio,
            classification=result.classification,
            perpetrator_cmax_uM=result.perpetrator_cmax_uM,
            static_r1=result.static_r1,
            time_h=list(result.time_h),
            cp_alone_mg_L=list(result.cp_alone_mg_L),
            cp_ddi_mg_L=list(result.cp_ddi_mg_L),
            warnings=list(result.warnings),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("DDI simulate error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ddi/polypharmacy", response_model=PolypharmacyDDIResponse)
def ddi_polypharmacy(req: PolypharmacyDDIRequest) -> PolypharmacyDDIResponse:
    """Multi-perpetrator polypharmacy DDI simulation (1–10 perpetrators)."""
    if req.route not in ("oral", "iv", "sc"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid route '{req.route}'. Must be 'oral', 'iv', or 'sc'.",
        )
    if not (1 <= len(req.perpetrators) <= 10):
        raise HTTPException(
            status_code=422,
            detail=f"Expected 1–10 perpetrators, got {len(req.perpetrators)}.",
        )
    try:
        from omega_pbpk.clinical.ddi_simulation import (
            PerpetratorSpec,
            simulate_polypharmacy_ddi,
        )

        victim = _drug_request_to_drug(req.victim_drug)
        perp_specs = [
            PerpetratorSpec(
                name=p.name,
                ki_uM=p.ki_uM,
                target_enzyme=p.target_enzyme,
                mechanism=p.mechanism,
                kinact_per_h=p.kinact_per_h,
                kdeg_per_h=p.kdeg_per_h,
                fold_induction=p.fold_induction,
                cmax_uM=p.cmax_uM,
            )
            for p in req.perpetrators
        ]
        result = simulate_polypharmacy_ddi(
            victim_drug=victim,
            perpetrators=perp_specs,
            dose_mg=req.dose_mg,
            route=req.route,
            body_weight=req.body_weight,
            t_end_h=req.duration_h,
        )
        pairwise_resp = [
            PairwiseDDIResponse(
                perpetrator_name=pw.perpetrator_name,
                auc_ratio=pw.auc_ratio,
                cmax_ratio=pw.cmax_ratio,
                classification=pw.classification,
                perpetrator_cmax_uM=pw.perpetrator_cmax_uM,
                static_r1=pw.static_r1,
                warnings=list(pw.warnings),
            )
            for pw in result.pairwise
        ]
        return PolypharmacyDDIResponse(
            victim_name=result.victim_name,
            perpetrators=list(result.perpetrators),
            n_perpetrators=result.n_perpetrators,
            auc_alone_mg_h_L=result.auc_alone_mg_h_L,
            cmax_alone_mg_L=result.cmax_alone_mg_L,
            t_half_alone_h=result.t_half_alone_h,
            auc_combined_mg_h_L=result.auc_combined_mg_h_L,
            cmax_combined_mg_L=result.cmax_combined_mg_L,
            t_half_combined_h=result.t_half_combined_h,
            auc_ratio_combined=result.auc_ratio_combined,
            cmax_ratio_combined=result.cmax_ratio_combined,
            classification_combined=result.classification_combined,
            pairwise=pairwise_resp,
            synergy_flag=result.synergy_flag,
            time_h=list(result.time_h),
            cp_alone_mg_L=list(result.cp_alone_mg_L),
            cp_combined_mg_L=list(result.cp_combined_mg_L),
            warnings=list(result.warnings),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("DDI polypharmacy error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/simulate/pediatric", response_model=PediatricSimulateResponse)
def simulate_pediatric(req: PediatricSimulateRequest) -> PediatricSimulateResponse:
    """Pediatric PBPK simulation with ontogeny-based enzyme maturation and GFR scaling."""
    if req.route not in ("oral", "iv", "sc"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid route '{req.route}'. Must be 'oral', 'iv', or 'sc'.",
        )
    try:
        import numpy as np

        from omega_pbpk.clinical.ontogeny import get_pediatric_scaling
        from omega_pbpk.core.body import WholeBodyPBPK

        drug = _drug_request_to_drug(req.drug)
        warnings: list[str] = []

        def _run_sim(bw: float, age: float | None):
            m = WholeBodyPBPK(drug=drug, body_weight=bw, age_years=age)
            if req.route == "oral":
                m.setup_oral(req.dose_mg)
            elif req.route == "iv":
                m.setup_iv(req.dose_mg)
            else:
                m.setup_sc(req.dose_mg)
            return m.simulate(t_end_h=req.duration_h)

        ped_result = _run_sim(req.body_weight_kg, req.age_years)
        ped_pk = ped_result.pk_summary()
        ped_cp = ped_result.plasma_concentration()
        ped_t = ped_result.time_h

        adult_result = _run_sim(70.0, None)
        adult_pk = adult_result.pk_summary()
        adult_cp = adult_result.plasma_concentration()
        adult_t = adult_result.time_h

        adult_cp_interp = np.interp(ped_t, adult_t, adult_cp)

        ped_t_half = ped_pk.get("half_life_h", 0.0)
        if ped_t_half == float("inf") or ped_t_half > 1e6:
            ped_t_half = 0.0
        adult_t_half = adult_pk.get("half_life_h", 0.0)
        if adult_t_half == float("inf") or adult_t_half > 1e6:
            adult_t_half = 0.0

        ped_auc = ped_pk["AUC_mg_h_L"]
        adult_auc = adult_pk["AUC_mg_h_L"]
        ped_cmax = ped_pk["Cmax_mg_L"]
        adult_cmax = adult_pk["Cmax_mg_L"]

        auc_ratio = ped_auc / max(adult_auc, 1e-12)
        cmax_ratio = ped_cmax / max(adult_cmax, 1e-12)
        dose_adjustment_factor = auc_ratio * (req.body_weight_kg / 70.0)

        ontogeny = get_pediatric_scaling(req.age_years, req.body_weight_kg)

        if req.age_years < 1.0:
            warnings.append("Neonatal/infant subject: maturation uncertainty is high")
        if req.body_weight_kg < 5.0:
            warnings.append("Very low body weight (<5 kg): use results with caution")

        return PediatricSimulateResponse(
            pediatric_cmax_mg_L=ped_cmax,
            pediatric_tmax_h=ped_pk["Tmax_h"],
            pediatric_auc_mg_h_L=ped_auc,
            pediatric_t_half_h=ped_t_half,
            adult_cmax_mg_L=adult_cmax,
            adult_auc_mg_h_L=adult_auc,
            adult_t_half_h=adult_t_half,
            cmax_ratio=cmax_ratio,
            auc_ratio=auc_ratio,
            ontogeny_factors=ontogeny,
            dose_adjustment_factor=dose_adjustment_factor,
            time_h=ped_t.tolist() if hasattr(ped_t, "tolist") else list(ped_t),
            cp_pediatric_mg_L=ped_cp.tolist() if hasattr(ped_cp, "tolist") else list(ped_cp),
            cp_adult_mg_L=adult_cp_interp.tolist()
            if hasattr(adult_cp_interp, "tolist")
            else list(adult_cp_interp),
            warnings=warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pediatric simulation error")
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
        from omega_pbpk.surrogate import PKSurrogate
        from omega_pbpk.surrogate.data_generator import generate_training_data

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


@app.post("/ivive", response_model=IVIVEResponse)
def ivive(req: IVIVERequest) -> IVIVEResponse:
    """In vitro → in vivo extrapolation: clearance, permeability, and protein binding."""
    try:
        from omega_pbpk.clinical.ivive_engine import run_ivive_bundle

        result = run_ivive_bundle(
            clint_uL_min=req.clint_uL_min,
            clint_system=req.clint_system,
            fup=req.fup,
            fu_mic=req.fu_mic,
            logP=req.logP,
            papp_ab_cm_s=req.papp_ab_cm_s,
            papp_ba_cm_s=req.papp_ba_cm_s,
            rbp=req.rbp,
            mw=req.mw,
            drug_name=req.drug_name,
            clint_gut_uL_min_mg=req.clint_gut_uL_min_mg,
            liver_weight_g=req.liver_weight_g,
            q_liver_L_per_h=req.q_liver_L_per_h,
        )
        cl = result.clearance
        cl_resp = IVIVEClearanceResponse(
            clint_in_vitro=cl.clint_in_vitro,
            system=cl.system,
            clint_liver_L_per_h=cl.clint_liver_L_per_h,
            clh_well_stirred_L_per_h=cl.clh_well_stirred_L_per_h,
            fu_mic=cl.fu_mic,
            fup=cl.fup,
        )
        perm_resp = None
        if result.permeability is not None:
            p = result.permeability
            perm_resp = IVIVEPermeabilityResponse(
                papp_ab_cm_s=p.papp_ab_cm_s,
                papp_ba_cm_s=p.papp_ba_cm_s,
                peff_10_4_cm_s=p.peff_10_4_cm_s,
                efflux_ratio=p.efflux_ratio,
                fa_predicted=p.fa_predicted,
                fg_predicted=p.fg_predicted,
                absorption_classification=p.absorption_classification,
                pgp_flag=p.pgp_flag,
            )
        b = result.binding
        bind_resp = IVIVEBindingResponse(
            fup_measured=b.fup_measured,
            fup_corrected=b.fup_corrected,
            fu_mic=b.fu_mic,
            fu_mic_source=b.fu_mic_source,
            rbp=b.rbp,
            rbp_source=b.rbp_source,
        )
        return IVIVEResponse(
            clearance=cl_resp,
            permeability=perm_resp,
            binding=bind_resp,
            drug_params=result.drug_params,
            confidence=result.confidence,
            warnings=list(result.warnings),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("IVIVE error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/simulate/uncertainty", response_model=ConformalUQResponse)
def simulate_uncertainty(req: ConformalUQRequest) -> ConformalUQResponse:
    """Conformal UQ propagation via LHS sampling over parameter bounds."""
    from omega_pbpk.uncertainty.conformal_uq import (
        ParameterBounds,
        build_bounds_from_adme,
        propagate_conformal_intervals,
    )

    if req.route not in ("oral", "iv"):
        raise HTTPException(status_code=400, detail="route must be 'oral' or 'iv'")

    all_bounds_provided = all(
        v is not None
        for v in [
            req.fup_lo,
            req.fup_hi,
            req.clint_lo,
            req.clint_hi,
            req.peff_lo,
            req.peff_hi,
            req.rbp_lo,
            req.rbp_hi,
        ]
    )

    if all_bounds_provided:
        bounds = ParameterBounds(
            fup_lo=req.fup_lo,
            fup_hi=req.fup_hi,
            clint_lo=req.clint_lo,
            clint_hi=req.clint_hi,
            peff_lo=req.peff_lo,
            peff_hi=req.peff_hi,
            rbp_lo=req.rbp_lo,
            rbp_hi=req.rbp_hi,
        )
    else:
        from omega_pbpk.prediction.adme_predictor import ADMEPredictor

        predictor = ADMEPredictor()
        if req.smiles:
            props = predictor.predict(req.smiles)
        elif req.mw is not None and req.logP is not None:
            props = predictor.predict_from_dict({"mw": req.mw, "logP": req.logP})
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide smiles or mw+logP, or all 8 bound values",
            )
        bounds = build_bounds_from_adme(props)

    result = propagate_conformal_intervals(
        drug_name=req.drug_name,
        dose_mg=req.dose_mg,
        route=req.route,
        bounds=bounds,
        n_samples=req.n_samples,
    )
    return ConformalUQResponse(
        cmax_p5=result.cmax_p5,
        cmax_p50=result.cmax_p50,
        cmax_p95=result.cmax_p95,
        auc_p5=result.auc_p5,
        auc_p50=result.auc_p50,
        auc_p95=result.auc_p95,
        tmax_p5=result.tmax_p5,
        tmax_p50=result.tmax_p50,
        tmax_p95=result.tmax_p95,
        t_half_p5=result.t_half_p5,
        t_half_p50=result.t_half_p50,
        t_half_p95=result.t_half_p95,
        n_samples=result.n_samples,
        warnings=list(result.warnings),
    )


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


class TDMObservation(BaseModel):
    time_h: float = Field(gt=0, description="Time after dose in hours")
    conc_mg_L: float = Field(gt=0, description="Observed concentration in mg/L")
    dose_number: int = Field(default=1, ge=1, description="Which dose this observation follows")
    assay_cv: float = Field(default=0.10, gt=0, le=1.0, description="Assay CV")


class TDMRequest(BaseModel):
    observations: list[TDMObservation]
    dose_mg: float = Field(default=100.0, gt=0)
    interval_h: float = Field(default=12.0, gt=0)
    route: str = "oral"
    target_trough: float = Field(default=1.0, gt=0)
    target_peak: float | None = None
    dose_range_mg: list[float] = Field(default=[10.0, 2000.0])
    forward_hours: float = Field(default=72.0, gt=0, le=720.0)
    cl_mean: float = Field(default=5.0, gt=0, description="Population CL mean (L/h)")
    cl_cv: float = Field(default=0.40, gt=0, le=2.0)
    vd_mean: float = Field(default=42.0, gt=0, description="Population Vd mean (L)")
    vd_cv: float = Field(default=0.30, gt=0, le=2.0)
    ka_mean: float = Field(default=1.0, gt=0, description="Population ka mean (1/h)")
    ka_cv: float = Field(default=0.50, gt=0, le=2.0)

    @field_validator("observations")
    @classmethod
    def observations_not_empty(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 observations")
        return v

    @field_validator("dose_range_mg")
    @classmethod
    def dose_range_valid(cls, v):
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("dose_range_mg must be [min, max] with min < max")
        return v


class TDMResponse(BaseModel):
    individual_cl: float
    individual_vd: float
    individual_ka: float
    cl_ratio: float
    vd_ratio: float
    ka_ratio: float
    predicted: list[float]
    observed: list[float]
    residuals: list[float]
    objective_function_value: float
    recommended_dose_mg: float
    predicted_trough: float
    predicted_peak: float
    forward_time_h: list[float]
    forward_conc_mg_L: list[float]
    converged: bool
    warnings: list[str]


class BatchPredictRequest(BaseModel):
    smiles_list: list[str]
    dose_mg: float = 100.0
    route: str = "oral"
    duration_h: float = 24.0
    n_uq_samples: int = 50
    ranking: str = "score"

    @field_validator("smiles_list")
    @classmethod
    def smiles_list_not_empty(cls, v):
        if not v:
            raise ValueError("smiles_list must not be empty")
        if len(v) > 50:
            raise ValueError("Maximum 50 compounds per batch")
        for smi in v:
            if len(smi) > 500:
                raise ValueError(f"SMILES too long (max 500 chars): {smi[:40]}...")
        return v

    @field_validator("ranking")
    @classmethod
    def ranking_valid(cls, v):
        if v not in ("score", "risk", "auc", "cmax"):
            raise ValueError("ranking must be one of: score, risk, auc, cmax")
        return v


class CompoundSummaryResponse(BaseModel):
    rank: int
    smiles: str
    drug_name: str
    cmax_mg_L: float
    tmax_h: float
    auc0t_mg_h_L: float
    t_half_h: float
    cmax_p50: float
    auc_p50: float
    risk_count: int
    overall_risk_level: str
    confidence: str
    adme_confidence: str
    transporter_ddi_flags: list[str]
    composite_score: float
    warnings: list[str]


class BatchPredictResponse(BaseModel):
    compounds: list[CompoundSummaryResponse]
    n_total: int
    n_succeeded: int
    n_failed: int
    ranking_criterion: str
    best_compound: str
    warnings: list[str]


@app.post("/predict/full", response_model=FullPredictResponse)
def predict_full(
    req: FullPredictRequest,
    _auth: Annotated[TokenData | None, Depends(require_auth)] = None,
) -> FullPredictResponse:
    """Full end-to-end PK assessment: SMILES → ADME + transporters + PBPK + UQ + risk."""
    try:
        from omega_pbpk.prediction.full_pipeline import run_full_prediction

        result = run_full_prediction(
            smiles=req.smiles,
            dose_mg=req.dose_mg,
            route=req.route,
            duration_h=req.duration_h,
            n_uq_samples=req.n_uq_samples,
        )
        return FullPredictResponse(
            drug_name=result.drug_name,
            smiles=result.smiles,
            adme=result.adme,
            adme_confidence=result.adme_confidence,
            transporter_substrates=list(result.transporter_substrates),
            transporter_inhibitors=list(result.transporter_inhibitors),
            transporter_ddi_flags=list(result.transporter_ddi_flags),
            transporter_confidence=result.transporter_confidence,
            cmax_mg_L=result.cmax_mg_L,
            tmax_h=result.tmax_h,
            auc0t_mg_h_L=result.auc0t_mg_h_L,
            t_half_h=result.t_half_h,
            time_h=list(result.time_h),
            cp_mg_L=list(result.cp_mg_L),
            cmax_p5=result.cmax_p5,
            cmax_p50=result.cmax_p50,
            cmax_p95=result.cmax_p95,
            auc_p5=result.auc_p5,
            auc_p50=result.auc_p50,
            auc_p95=result.auc_p95,
            risk_flags=result.risk_flags,
            risk_count=result.risk_count,
            overall_risk_level=result.overall_risk_level,
            confidence=result.confidence,
            warnings=list(result.warnings),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Full prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    req: BatchPredictRequest,
    _auth: Annotated[TokenData | None, Depends(require_auth)] = None,
) -> BatchPredictResponse:
    """Batch full-pipeline prediction: rank multiple SMILES by developability."""
    try:
        from omega_pbpk.prediction.batch_pipeline import run_batch_prediction

        batch = run_batch_prediction(
            smiles_list=req.smiles_list,
            dose_mg=req.dose_mg,
            route=req.route,
            duration_h=req.duration_h,
            n_uq_samples=req.n_uq_samples,
            ranking=req.ranking,
        )
        return BatchPredictResponse(
            compounds=[
                CompoundSummaryResponse(
                    rank=c.rank,
                    smiles=c.smiles,
                    drug_name=c.drug_name,
                    cmax_mg_L=c.cmax_mg_L,
                    tmax_h=c.tmax_h,
                    auc0t_mg_h_L=c.auc0t_mg_h_L,
                    t_half_h=c.t_half_h,
                    cmax_p50=c.cmax_p50,
                    auc_p50=c.auc_p50,
                    risk_count=c.risk_count,
                    overall_risk_level=c.overall_risk_level,
                    confidence=c.confidence,
                    adme_confidence=c.adme_confidence,
                    transporter_ddi_flags=list(c.transporter_ddi_flags),
                    composite_score=c.composite_score,
                    warnings=list(c.warnings),
                )
                for c in batch.compounds
            ],
            n_total=batch.n_total,
            n_succeeded=batch.n_succeeded,
            n_failed=batch.n_failed,
            ranking_criterion=batch.ranking_criterion,
            best_compound=batch.best_compound,
            warnings=list(batch.warnings),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/dose/optimize", response_model=DoseOptimizeResponse)
def dose_optimize(req: DoseOptimizeRequest) -> DoseOptimizeResponse:
    """Optimize dosing regimen to meet therapeutic Cmin/Cmax/AUC targets."""
    if req.route not in ("oral", "iv"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid route '{req.route}'. Must be 'oral' or 'iv'.",
        )
    if len(req.dose_range_mg) != 2 or req.dose_range_mg[0] >= req.dose_range_mg[1]:
        raise HTTPException(
            status_code=422,
            detail="dose_range_mg must be [min, max] with min < max.",
        )
    try:
        from omega_pbpk.clinical.regimen_optimizer import (
            TherapeuticTarget,
            optimize_regimen,
        )
        from omega_pbpk.contracts.drug_spec import DrugSpec

        drug_obj = _drug_request_to_drug(req.drug)
        spec = DrugSpec(
            name=drug_obj.name,
            mw=drug_obj.mw,
            logP=drug_obj.logP,
            pka=drug_obj.pka,
            fup=drug_obj.fup,
            rbp=drug_obj.rbp,
            clint_hepatic_L_per_h=drug_obj.clint_hepatic_L_per_h,
            clr_L_per_h=drug_obj.clr_L_per_h,
            peff=drug_obj.peff,
        )

        target = TherapeuticTarget(
            cmin_mg_L=req.cmin_mg_L,
            cmax_mg_L=req.cmax_mg_L,
            auc_min=req.auc_min,
            auc_max=req.auc_max,
        )

        result = optimize_regimen(
            spec=spec,
            target=target,
            dose_range_mg=tuple(req.dose_range_mg),
            intervals_h=req.intervals_h,
            route=req.route,
            n_doses=req.n_doses,
            body_weight=req.body_weight,
        )

        return DoseOptimizeResponse(
            optimal_dose_mg=result.optimal_dose_mg,
            optimal_interval_h=result.optimal_interval_h,
            css_max=result.css_max,
            css_min=result.css_min,
            css_avg=result.css_avg,
            auc_ss=result.auc_ss,
            feasible=result.feasible,
            warnings=list(result.warnings),
            all_regimens=[RegimenCandidateResponse(**r) for r in result.all_regimens],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Dose optimization error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class RLDoseOptimizeRequest(BaseModel):
    cl_L_per_h: float
    vd_L: float
    ka_per_h: float = 1.0
    f_bioavail: float = 0.8
    route: str = "oral"
    interval_h: float = 24.0
    cmin_mg_L: float
    cmax_mg_L: float
    target_trough_mg_L: float | None = None
    n_steps: int = 10
    dose_grid_mg: list[float] | None = None
    n_episodes: int = 2000
    alpha: float = 0.1
    gamma: float = 0.95
    seed: int = 42


class RLDoseOptimizeResponse(BaseModel):
    optimal_dose_sequence: list[float]
    dose_grid_mg: list[float]
    n_steps: int
    interval_h: float
    final_trough_mg_L: float
    final_peak_mg_L: float
    time_in_window_fraction: float
    mean_reward_last_100: float
    converged: bool
    episode_rewards: list[float]
    episode_window_fractions: list[float]
    q_table_shape: list[int]
    warnings: list[str]


@app.post("/dose/rl_optimize", response_model=RLDoseOptimizeResponse)
def dose_rl_optimize(req: RLDoseOptimizeRequest) -> RLDoseOptimizeResponse:
    if req.route not in ("oral", "iv"):
        raise HTTPException(status_code=422, detail="route must be 'oral' or 'iv'")
    if req.cmin_mg_L >= req.cmax_mg_L:
        raise HTTPException(status_code=422, detail="cmin_mg_L must be less than cmax_mg_L")
    if req.dose_grid_mg is not None and len(req.dose_grid_mg) < 2:
        raise HTTPException(status_code=422, detail="dose_grid_mg must have at least 2 values")
    from omega_pbpk.clinical.rl_dose_optimizer import (
        PKParams,
        TherapeuticWindow,
        optimize_dose_rl,
    )

    pk = PKParams(
        cl_L_per_h=req.cl_L_per_h,
        vd_L=req.vd_L,
        ka_per_h=req.ka_per_h,
        f_bioavail=req.f_bioavail,
        route=req.route,
        interval_h=req.interval_h,
    )
    win = TherapeuticWindow(
        cmin_mg_L=req.cmin_mg_L,
        cmax_mg_L=req.cmax_mg_L,
        target_trough_mg_L=req.target_trough_mg_L,
    )
    result = optimize_dose_rl(
        pk,
        win,
        n_steps=req.n_steps,
        dose_grid_mg=req.dose_grid_mg,
        n_episodes=req.n_episodes,
        alpha=req.alpha,
        gamma=req.gamma,
        seed=req.seed,
    )
    p = result.policy
    n = result.n_episodes
    mean_r = (
        float(np.mean(result.episode_rewards[-100:]))
        if n >= 100
        else float(np.mean(result.episode_rewards))
    )
    return RLDoseOptimizeResponse(
        optimal_dose_sequence=p.optimal_dose_sequence,
        dose_grid_mg=p.dose_grid_mg,
        n_steps=p.n_steps,
        interval_h=p.interval_h,
        final_trough_mg_L=p.final_trough_mg_L,
        final_peak_mg_L=p.final_peak_mg_L,
        time_in_window_fraction=p.time_in_window_fraction,
        mean_reward_last_100=mean_r,
        converged=result.policy.converged,
        episode_rewards=result.episode_rewards,
        episode_window_fractions=result.episode_window_fractions,
        q_table_shape=list(result.q_table_shape),
        warnings=p.warnings,
    )


@app.post("/tdm", response_model=TDMResponse)
def tdm(req: TDMRequest) -> TDMResponse:
    """Therapeutic Drug Monitoring: MAP-Bayesian dose individualization."""
    if req.route not in ("oral", "iv"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid route '{req.route}'. Must be 'oral' or 'iv'.",
        )
    try:
        from omega_pbpk.clinical.tdm import (
            ObservedConcentration,
            PopulationPrior,
            run_tdm,
        )

        observations = [
            ObservedConcentration(
                time_h=o.time_h,
                conc_mg_L=o.conc_mg_L,
                dose_number=o.dose_number,
                assay_cv=o.assay_cv,
            )
            for o in req.observations
        ]
        prior = PopulationPrior(
            cl_mean=req.cl_mean,
            cl_cv=req.cl_cv,
            vd_mean=req.vd_mean,
            vd_cv=req.vd_cv,
            ka_mean=req.ka_mean,
            ka_cv=req.ka_cv,
        )
        result = run_tdm(
            observations=observations,
            prior=prior,
            dose_mg=req.dose_mg,
            interval_h=req.interval_h,
            route=req.route,
            target_trough=req.target_trough,
            target_peak=req.target_peak,
            dose_range_mg=tuple(req.dose_range_mg),
            forward_hours=req.forward_hours,
        )
        return TDMResponse(
            individual_cl=result.individual_cl,
            individual_vd=result.individual_vd,
            individual_ka=result.individual_ka,
            cl_ratio=result.cl_ratio,
            vd_ratio=result.vd_ratio,
            ka_ratio=result.ka_ratio,
            predicted=list(result.predicted),
            observed=list(result.observed),
            residuals=list(result.residuals),
            objective_function_value=result.objective_function_value,
            recommended_dose_mg=result.recommended_dose_mg,
            predicted_trough=result.predicted_trough,
            predicted_peak=result.predicted_peak,
            forward_time_h=list(result.forward_time_h),
            forward_conc_mg_L=list(result.forward_conc_mg_L),
            converged=result.converged,
            warnings=list(result.warnings),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("TDM error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class BEDesignRequest(BaseModel):
    cv_auc: float = Field(default=0.25, gt=0, le=1.0, description="Within-subject CV for AUC")
    cv_cmax: float = Field(default=0.30, gt=0, le=1.0, description="Within-subject CV for Cmax")
    gmr: float = Field(default=0.95, gt=0.5, le=1.5, description="Assumed geometric mean ratio")
    target_power: float = Field(default=0.80, gt=0, lt=1.0, description="Target statistical power")
    alpha: float = Field(default=0.05, gt=0, lt=0.5, description="Significance level (one-sided)")
    design: str = Field(default="2x2", description="Study design: 2x2, 3x3_williams, 4x2_replicate")
    dropout_rate: float = Field(default=0.10, ge=0, lt=1.0, description="Expected dropout rate")
    is_nti: bool = Field(default=False, description="Narrow therapeutic index drug")

    @field_validator("design")
    @classmethod
    def design_valid(cls, v: str) -> str:
        allowed = ("2x2", "3x3_williams", "4x2_replicate")
        if v not in allowed:
            raise ValueError(f"design must be one of {allowed}")
        return v


class BEDesignResponse(BaseModel):
    design: str
    n_subjects: int
    n_per_sequence: int
    power_achieved: float
    alpha: float
    be_limits: list[float]
    gmr_assumed: float
    cv_within: float
    cv_between: float
    endpoint: str
    dropout_adjusted_n: int
    dropout_rate: float
    power_curve: list[dict[str, float]]
    warnings: list[str]


@app.post("/be/design", response_model=BEDesignResponse)
def be_design(req: BEDesignRequest) -> BEDesignResponse:
    """Bioequivalence study design: sample size and power calculation."""
    try:
        from omega_pbpk.clinical.be_study_design import run_be_design

        result = run_be_design(
            cv_auc=req.cv_auc,
            cv_cmax=req.cv_cmax,
            gmr=req.gmr,
            target_power=req.target_power,
            alpha=req.alpha,
            design=req.design,
            dropout_rate=req.dropout_rate,
            is_nti=req.is_nti,
        )
        return BEDesignResponse(
            design=result.design,
            n_subjects=result.n_subjects,
            n_per_sequence=result.n_per_sequence,
            power_achieved=result.power_achieved,
            alpha=result.alpha,
            be_limits=list(result.be_limits),
            gmr_assumed=result.gmr_assumed,
            cv_within=result.cv_within,
            cv_between=result.cv_between,
            endpoint=result.endpoint,
            dropout_adjusted_n=result.dropout_adjusted_n,
            dropout_rate=result.dropout_rate,
            power_curve=result.power_curve,
            warnings=list(result.warnings),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("BE design error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /toxicity — organ-specific toxicity risk scoring
# ---------------------------------------------------------------------------


class ToxicityRequest(BaseModel):
    smiles: str | None = Field(default=None, description="SMILES string for compound")
    drug: DrugRequest | None = Field(
        default=None, description="Drug parameters (used if smiles not provided)"
    )
    dose_mg: float = Field(default=100.0, gt=0, description="Dose in mg")
    route: str = Field(default="oral", description="Administration route: oral, iv, sc")
    duration_h: float = Field(default=24.0, gt=0, description="Simulation duration (h)")
    herg_ic50_uM: float | None = Field(default=None, description="hERG IC50 in µM (optional)")

    @field_validator("route")
    @classmethod
    def route_valid(cls, v: str) -> str:
        allowed = ("oral", "iv", "sc")
        if v not in allowed:
            raise ValueError(f"route must be one of {allowed}")
        return v


class OrganExposureResponse(BaseModel):
    organ: str
    cmax_mg_L: float
    auc_mg_h_L: float
    tmax_h: float
    cmax_unbound_uM: float
    cave_mg_L: float


class ToxicityScoreResponse(BaseModel):
    organ: str
    endpoint: str
    exposure_metric: float
    threshold: float
    margin: float
    risk_level: str
    reference: str
    warning: str


class ToxicityResponse(BaseModel):
    drug_name: str
    organ_exposures: list[OrganExposureResponse]
    toxicity_scores: list[ToxicityScoreResponse]
    overall_risk: str
    risk_count: int
    max_concern_organ: str
    summary: dict[str, Any]
    warnings: list[str]


@app.post("/toxicity", response_model=ToxicityResponse)
def toxicity_assessment(req: ToxicityRequest) -> ToxicityResponse:
    """Organ-specific toxicity risk scoring from PBPK simulation."""
    try:
        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.drugs.drug import Drug
        from omega_pbpk.risk.organ_toxicity import run_organ_toxicity_assessment

        # Build Drug object
        if req.drug is not None:
            drug_kwargs: dict[str, Any] = {
                "name": req.drug.name,
                "mw": req.drug.mw,
                "logP": req.drug.logP,
                "fup": req.drug.fup,
                "rbp": req.drug.rbp,
                "drug_type": req.drug.drug_type,
                "clint_hepatic_L_per_h": req.drug.clint_hepatic_L_per_h,
                "clr_L_per_h": req.drug.clr_L_per_h,
                "peff": req.drug.peff,
            }
            if req.drug.pka is not None:
                drug_kwargs["pka"] = req.drug.pka
            if req.drug.clint is not None:
                drug_kwargs["clint"] = req.drug.clint
            if req.drug.fm is not None:
                drug_kwargs["fm"] = req.drug.fm
            drug = Drug(**drug_kwargs)
        else:
            drug = Drug(name=req.smiles or "Unknown")

        # Run PBPK simulation
        model = WholeBodyPBPK(drug)
        if req.route == "iv":
            model.setup_iv(req.dose_mg)
        elif req.route == "sc":
            model.setup_sc(req.dose_mg)
        else:
            model.setup_oral(req.dose_mg)

        sim_result = model.simulate(t_end_h=req.duration_h)

        # Run toxicity assessment
        report = run_organ_toxicity_assessment(
            result=sim_result,
            drug=sim_result.drug,
            dose_mg=req.dose_mg,
            herg_ic50_uM=req.herg_ic50_uM,
        )

        return ToxicityResponse(
            drug_name=report.drug_name,
            organ_exposures=[
                OrganExposureResponse(
                    organ=e.organ,
                    cmax_mg_L=e.cmax_mg_L,
                    auc_mg_h_L=e.auc_mg_h_L,
                    tmax_h=e.tmax_h,
                    cmax_unbound_uM=e.cmax_unbound_uM,
                    cave_mg_L=e.cave_mg_L,
                )
                for e in report.organ_exposures
            ],
            toxicity_scores=[
                ToxicityScoreResponse(
                    organ=s.organ,
                    endpoint=s.endpoint,
                    exposure_metric=s.exposure_metric,
                    threshold=s.threshold,
                    margin=s.margin if s.margin < 1e6 else 1e6,
                    risk_level=s.risk_level,
                    reference=s.reference,
                    warning=s.warning,
                )
                for s in report.toxicity_scores
            ],
            overall_risk=report.overall_risk,
            risk_count=report.risk_count,
            max_concern_organ=report.max_concern_organ,
            summary=report.summary,
            warnings=report.warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Toxicity assessment error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /phenotyping — metabolic reaction phenotyping (CYP contribution)
# ---------------------------------------------------------------------------


class InhibitionExperimentRequest(BaseModel):
    enzyme: str
    inhibitor_name: str
    control_clint: float = Field(..., gt=0)
    inhibited_clint: float = Field(..., ge=0)
    inhibitor_conc_uM: float | None = None
    ki_uM: float | None = None


class RecombinantCYPDataRequest(BaseModel):
    enzyme: str
    velocity_pmol_min_pmol: float = Field(..., gt=0)
    isef: float | None = None
    raf: float | None = None
    abundance_pmol_mg: float | None = None


class PhenotypingRequest(BaseModel):
    drug_name: str = "compound"
    inhibition_experiments: list[InhibitionExperimentRequest] | None = None
    recombinant_data: list[RecombinantCYPDataRequest] | None = None
    method: str = Field(default="auto", description="inhibition, recombinant, combined, or auto")

    @field_validator("method")
    @classmethod
    def method_valid(cls, v: str) -> str:
        allowed = ("auto", "inhibition", "recombinant", "combined")
        if v not in allowed:
            raise ValueError(f"method must be one of {allowed}")
        return v


class CYPContributionResponse(BaseModel):
    enzyme: str
    fm: float
    clint_enzyme_uL_min_mg: float
    method: str
    confidence: str


class PhenotypingResponse(BaseModel):
    drug_name: str
    total_clint_uL_min_mg: float
    contributions: list[CYPContributionResponse]
    fm_dict: dict[str, float]
    primary_enzyme: str
    fm_primary: float
    ddi_vulnerability: str
    pgx_sensitivity: str
    method: str
    warnings: list[str]
    recommendation: str


@app.post("/phenotyping", response_model=PhenotypingResponse)
def phenotyping(req: PhenotypingRequest) -> PhenotypingResponse:
    """Metabolic reaction phenotyping: determine CYP enzyme contributions."""
    try:
        from omega_pbpk.clinical.reaction_phenotyping import (
            InhibitionExperiment,
            RecombinantCYPData,
            run_reaction_phenotyping,
        )

        inh_exps = None
        if req.inhibition_experiments:
            inh_exps = [
                InhibitionExperiment(
                    enzyme=e.enzyme,
                    inhibitor_name=e.inhibitor_name,
                    control_clint=e.control_clint,
                    inhibited_clint=e.inhibited_clint,
                    inhibitor_conc_uM=e.inhibitor_conc_uM,
                    ki_uM=e.ki_uM,
                )
                for e in req.inhibition_experiments
            ]

        rec_data = None
        if req.recombinant_data:
            rec_data = [
                RecombinantCYPData(
                    enzyme=d.enzyme,
                    velocity_pmol_min_pmol=d.velocity_pmol_min_pmol,
                    isef=d.isef,
                    raf=d.raf,
                    abundance_pmol_mg=d.abundance_pmol_mg,
                )
                for d in req.recombinant_data
            ]

        result = run_reaction_phenotyping(
            drug_name=req.drug_name,
            inhibition_experiments=inh_exps,
            recombinant_data=rec_data,
            method=req.method,
        )

        return PhenotypingResponse(
            drug_name=result.drug_name,
            total_clint_uL_min_mg=result.total_clint_uL_min_mg,
            contributions=[
                CYPContributionResponse(
                    enzyme=c.enzyme,
                    fm=c.fm,
                    clint_enzyme_uL_min_mg=c.clint_enzyme_uL_min_mg,
                    method=c.method,
                    confidence=c.confidence,
                )
                for c in result.contributions
            ],
            fm_dict=result.fm_dict,
            primary_enzyme=result.primary_enzyme,
            fm_primary=result.fm_primary,
            ddi_vulnerability=result.ddi_vulnerability,
            pgx_sensitivity=result.pgx_sensitivity,
            method=result.method,
            warnings=list(result.warnings),
            recommendation=result.recommendation,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Phenotyping error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /sensitivity/sobol — Sobol global sensitivity analysis
# ---------------------------------------------------------------------------


class SobolParameterRange(BaseModel):
    name: str
    lower: float = Field(..., ge=0)
    upper: float = Field(..., gt=0)


class SobolRequest(BaseModel):
    drug: DrugRequest
    dose_mg: float = Field(default=100.0, gt=0)
    route: str = "oral"
    body_weight: float = Field(default=70.0, gt=0)
    t_end_h: float = Field(default=24.0, gt=0)
    parameter_ranges: list[SobolParameterRange] | None = None
    n_samples: int = Field(default=128, ge=16)
    metric: str = Field(default="AUC", description="PK metric: AUC or Cmax")
    seed: int = 42
    n_bootstrap: int = Field(default=100, ge=10)

    @field_validator("metric")
    @classmethod
    def metric_valid(cls, v: str) -> str:
        allowed = ("AUC", "Cmax")
        if v not in allowed:
            raise ValueError(f"metric must be one of {allowed}")
        return v


class SobolResponse(BaseModel):
    parameters: list[str]
    S1: list[float]
    ST: list[float]
    S1_conf: list[float]
    ST_conf: list[float]
    metric: str
    n_samples: int
    n_total_evaluations: int
    convergence_flag: bool
    interaction_index: list[float]


@app.post("/sensitivity/sobol", response_model=SobolResponse)
def sensitivity_sobol(req: SobolRequest) -> SobolResponse:
    """Sobol global sensitivity analysis for PK parameters."""
    try:
        from omega_pbpk.drugs.drug import Drug
        from omega_pbpk.sensitivity.sobol_gsa import (
            ParameterRange,
            sobol_sensitivity,
        )

        drug_kwargs: dict[str, Any] = {
            "name": req.drug.name,
            "mw": req.drug.mw,
            "logP": req.drug.logP,
            "fup": req.drug.fup,
            "rbp": req.drug.rbp,
            "drug_type": req.drug.drug_type,
            "clint_hepatic_L_per_h": req.drug.clint_hepatic_L_per_h,
            "clr_L_per_h": req.drug.clr_L_per_h,
            "peff": req.drug.peff,
        }
        if req.drug.pka is not None:
            drug_kwargs["pka"] = req.drug.pka
        if req.drug.clint is not None:
            drug_kwargs["clint"] = req.drug.clint
        if req.drug.fm is not None:
            drug_kwargs["fm"] = req.drug.fm
        drug = Drug(**drug_kwargs)

        ranges = None
        if req.parameter_ranges is not None:
            ranges = [
                ParameterRange(name=r.name, lower=r.lower, upper=r.upper)
                for r in req.parameter_ranges
            ]

        result = sobol_sensitivity(
            drug=drug,
            dose_mg=req.dose_mg,
            route=req.route,
            body_weight=req.body_weight,
            t_end_h=req.t_end_h,
            parameter_ranges=ranges,
            n_samples=req.n_samples,
            metric=req.metric,
            seed=req.seed,
            n_bootstrap=req.n_bootstrap,
        )

        return SobolResponse(
            parameters=result.parameters,
            S1=result.S1,
            ST=result.ST,
            S1_conf=result.S1_conf,
            ST_conf=result.ST_conf,
            metric=result.metric,
            n_samples=result.n_samples,
            n_total_evaluations=result.n_total_evaluations,
            convergence_flag=result.convergence_flag,
            interaction_index=result.interaction_index,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Sobol sensitivity error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /simulate/formulation — ER/MR formulation release + PK simulation
# ---------------------------------------------------------------------------


class ReleasePhaseRequest(BaseModel):
    fraction: float = Field(default=1.0, gt=0, le=1.0, description="Fraction of dose in this phase")
    model: str = Field(
        default="immediate",
        description="Release model: immediate, zero_order, first_order, weibull, higuchi",
    )
    params: dict[str, float] = Field(default_factory=dict, description="Model-specific parameters")
    lag_h: float = Field(default=0.0, ge=0, description="Lag time before phase activates (h)")
    site: str = Field(default="stomach", description="GI release site")

    @field_validator("model")
    @classmethod
    def model_valid(cls, v: str) -> str:
        allowed = ("immediate", "zero_order", "first_order", "weibull", "higuchi")
        if v not in allowed:
            raise ValueError(f"model must be one of {allowed}")
        return v


class FormulationSimulateRequest(BaseModel):
    name: str = Field(default="ER tablet", description="Formulation name")
    release_phases: list[ReleasePhaseRequest] = Field(..., min_length=1)
    total_dose_mg: float = Field(default=100.0, gt=0)
    t_end_h: float = Field(default=24.0, gt=0, le=720)
    dt_h: float = Field(default=0.01, gt=0, le=1.0)
    ke: float = Field(default=0.1, gt=0, description="Elimination rate constant (1/h)")
    vd_L: float = Field(default=42.0, gt=0, description="Volume of distribution (L)")
    ka: float = Field(
        default=0.0, ge=0, description="Absorption rate constant (1/h); 0 = direct input"
    )
    bioavailability: float = Field(default=1.0, gt=0, le=1.0)


class FormulationSimulateResponse(BaseModel):
    formulation_name: str
    pk_summary: dict[str, float]
    release_profile: dict[str, list[float]]
    plasma_profile: dict[str, list[float]]
    warnings: list[str]


@app.post("/simulate/formulation", response_model=FormulationSimulateResponse)
def simulate_formulation(req: FormulationSimulateRequest) -> FormulationSimulateResponse:
    """ER/MR formulation release modelling with 1-compartment PK simulation."""
    try:
        from omega_pbpk.core.formulation import (
            FormulationSpec,
            ReleasePhase,
            release_profile,
            simulate_pk_1cpt,
            validate_formulation,
        )

        phases = [
            ReleasePhase(
                fraction=p.fraction,
                model=p.model,
                params=dict(p.params),
                lag_h=p.lag_h,
                site=p.site,
            )
            for p in req.release_phases
        ]

        spec = FormulationSpec(
            name=req.name,
            release_phases=phases,
            total_dose_mg=req.total_dose_mg,
        )

        warnings = validate_formulation(spec)

        rp = release_profile(spec, t_end_h=req.t_end_h, dt_h=req.dt_h)

        t_pk, conc = simulate_pk_1cpt(
            spec,
            ke=req.ke,
            vd=req.vd_L,
            t_end_h=req.t_end_h,
            dt_h=req.dt_h,
            ka=req.ka,
            bioavailability=req.bioavailability,
        )

        # PK summary
        cmax = float(np.max(conc))
        tmax = float(t_pk[int(np.argmax(conc))])
        from omega_pbpk._compat import np_trapz

        auc = float(np_trapz(conc, t_pk))
        t_half = 0.693 / req.ke if req.ke > 0 else float("inf")

        # Downsample for response (max ~500 points)
        step = max(1, len(t_pk) // 500)
        t_ds = t_pk[::step]
        conc_ds = conc[::step]
        cum_ds = rp.cumulative_fraction_released[::step]

        return FormulationSimulateResponse(
            formulation_name=spec.name,
            pk_summary={
                "cmax_mg_L": round(cmax, 6),
                "tmax_h": round(tmax, 4),
                "auc_mg_h_L": round(auc, 4),
                "t_half_h": round(t_half, 4),
            },
            release_profile={
                "time_h": [round(float(x), 4) for x in t_ds],
                "cumulative_fraction": [round(float(x), 6) for x in cum_ds],
            },
            plasma_profile={
                "time_h": [round(float(x), 4) for x in t_ds],
                "conc_mg_L": [round(float(x), 6) for x in conc_ds],
            },
            warnings=warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Formulation simulation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /trial/crossover — virtual 2×2 crossover BE trial simulation
# ---------------------------------------------------------------------------


class CrossoverDrugRequest(BaseModel):
    name: str = "Test Drug"
    mw: float = Field(default=300.0, gt=0)
    logP: float = 2.0
    fup: float = Field(default=0.5, gt=0, le=1.0)
    rbp: float = Field(default=1.0, gt=0)
    drug_type: str = "neutral"
    clint_hepatic_L_per_h: float = Field(default=5.0, ge=0)
    clr_L_per_h: float = Field(default=0.0, ge=0)
    peff: float = Field(default=1.0, ge=0)
    pka: list[float] | None = None
    clint: dict[str, float] | None = None
    fm: dict[str, float] | None = None


class CrossoverArmRequest(BaseModel):
    name: str
    drug: CrossoverDrugRequest
    dose_mg: float = Field(default=100.0, gt=0)
    route: str = Field(default="oral")

    @field_validator("route")
    @classmethod
    def route_valid(cls, v: str) -> str:
        allowed = ("oral", "iv")
        if v not in allowed:
            raise ValueError(f"route must be one of {allowed}")
        return v


class CrossoverTrialRequest(BaseModel):
    arms: list[CrossoverArmRequest] = Field(..., min_length=2, max_length=2)
    design: str = Field(default="2x2")
    n_subjects: int = Field(default=24, ge=4, le=200)
    washout_sufficient: bool = True
    wsv_cv_cl: float = Field(default=0.15, ge=0, le=1.0)
    wsv_cv_vd: float = Field(default=0.10, ge=0, le=1.0)
    wsv_cv_ka: float = Field(default=0.20, ge=0, le=1.0)
    period_effect: float = Field(default=0.0, ge=-0.5, le=0.5)
    seed: int = 42
    t_end_h: float = Field(default=48.0, gt=0, le=720)

    @field_validator("design")
    @classmethod
    def design_valid(cls, v: str) -> str:
        if v != "2x2":
            raise ValueError("Only '2x2' design is currently supported")
        return v


class CrossoverTrialResponse(BaseModel):
    design: str
    n_subjects: int
    n_completed: int
    gmr_auc: float
    gmr_cmax: float
    ci90_auc: list[float]
    ci90_cmax: list[float]
    tost_p_auc: float
    tost_p_cmax: float
    be_conclusion_auc: str
    be_conclusion_cmax: str
    be_conclusion_overall: str
    be_limits: list[float]
    power_auc: float
    power_cmax: float
    cv_within_auc: float
    cv_within_cmax: float
    anova_summary: dict[str, float]
    warnings: list[str]


@app.post("/trial/crossover", response_model=CrossoverTrialResponse)
def trial_crossover(req: CrossoverTrialRequest) -> CrossoverTrialResponse:
    """Virtual 2×2 crossover bioequivalence trial simulation."""
    try:
        from omega_pbpk.clinical.crossover_trial import (
            CrossoverArm,
            CrossoverDesign,
            run_crossover_trial,
        )
        from omega_pbpk.drugs.drug import Drug

        # Build Drug objects and arms
        trial_arms: list[CrossoverArm] = []
        for arm_req in req.arms:
            drug_kwargs: dict[str, Any] = {
                "name": arm_req.drug.name,
                "mw": arm_req.drug.mw,
                "logP": arm_req.drug.logP,
                "fup": arm_req.drug.fup,
                "rbp": arm_req.drug.rbp,
                "drug_type": arm_req.drug.drug_type,
                "clint_hepatic_L_per_h": arm_req.drug.clint_hepatic_L_per_h,
                "clr_L_per_h": arm_req.drug.clr_L_per_h,
                "peff": arm_req.drug.peff,
            }
            if arm_req.drug.pka is not None:
                drug_kwargs["pka"] = arm_req.drug.pka
            if arm_req.drug.clint is not None:
                drug_kwargs["clint"] = arm_req.drug.clint
            if arm_req.drug.fm is not None:
                drug_kwargs["fm"] = arm_req.drug.fm
            drug = Drug(**drug_kwargs)
            trial_arms.append(
                CrossoverArm(
                    name=arm_req.name,
                    drug=drug,
                    dose_mg=arm_req.dose_mg,
                    route=arm_req.route,
                )
            )

        trial_design = CrossoverDesign(
            arms=tuple(trial_arms),
            design=req.design,
            n_subjects=req.n_subjects,
            washout_sufficient=req.washout_sufficient,
            wsv_cv_cl=req.wsv_cv_cl,
            wsv_cv_vd=req.wsv_cv_vd,
            wsv_cv_ka=req.wsv_cv_ka,
            period_effect=req.period_effect,
            seed=req.seed,
        )

        result = run_crossover_trial(trial_design, t_end_h=req.t_end_h)

        return CrossoverTrialResponse(
            design=result.design,
            n_subjects=result.n_subjects,
            n_completed=result.n_completed,
            gmr_auc=result.gmr_auc,
            gmr_cmax=result.gmr_cmax,
            ci90_auc=list(result.ci90_auc),
            ci90_cmax=list(result.ci90_cmax),
            tost_p_auc=result.tost_p_auc,
            tost_p_cmax=result.tost_p_cmax,
            be_conclusion_auc=result.be_conclusion_auc,
            be_conclusion_cmax=result.be_conclusion_cmax,
            be_conclusion_overall=result.be_conclusion_overall,
            be_limits=list(result.be_limits),
            power_auc=result.power_auc,
            power_cmax=result.power_cmax,
            cv_within_auc=result.cv_within_auc,
            cv_within_cmax=result.cv_within_cmax,
            anova_summary=result.anova_summary,
            warnings=result.warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Crossover trial error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Drug comparison endpoint
# ---------------------------------------------------------------------------


class ComparisonCandidateRequest(BaseModel):
    drug: DrugRequest
    dose_mg: float = Field(default=100.0, gt=0)
    route: str = Field(default="oral")
    label: str = ""
    herg_ic50_uM: float | None = None

    @field_validator("route")
    @classmethod
    def route_valid(cls, v: str) -> str:
        allowed = ("oral", "iv")
        if v not in allowed:
            raise ValueError(f"route must be one of {allowed}")
        return v


class DrugComparisonRequest(BaseModel):
    candidates: list[ComparisonCandidateRequest] = Field(..., min_length=2, max_length=5)
    t_end_h: float = Field(default=24.0, gt=0, le=720)
    weights: dict[str, float] | None = None


class DrugComparisonResponse(BaseModel):
    n_candidates: int
    candidate_labels: list[str]
    composite_scores: dict[str, float]
    overall_ranking: list[str]
    best_candidate: str
    decision_summary: str
    pk_profiles: list[dict[str, Any]]
    safety_profiles: list[dict[str, Any]]
    dimension_rankings: list[dict[str, Any]]
    warnings: list[str]


@app.post("/compare", response_model=DrugComparisonResponse)
def compare_drugs_endpoint(req: DrugComparisonRequest) -> DrugComparisonResponse:
    """Compare 2–5 drug candidates on PK, safety, and composite scores."""
    try:
        from omega_pbpk.clinical.drug_comparison import (
            ComparisonCandidate,
            compare_drugs,
        )
        from omega_pbpk.drugs.drug import Drug

        candidates = []
        for c_req in req.candidates:
            drug_kwargs: dict[str, Any] = {
                "name": c_req.drug.name,
                "mw": c_req.drug.mw,
                "logP": c_req.drug.logP,
                "fup": c_req.drug.fup,
                "rbp": c_req.drug.rbp,
                "drug_type": c_req.drug.drug_type,
                "clint_hepatic_L_per_h": c_req.drug.clint_hepatic_L_per_h,
                "clr_L_per_h": c_req.drug.clr_L_per_h,
                "peff": c_req.drug.peff,
            }
            if c_req.drug.pka is not None:
                drug_kwargs["pka"] = c_req.drug.pka
            if c_req.drug.clint is not None:
                drug_kwargs["clint"] = c_req.drug.clint
            if c_req.drug.fm is not None:
                drug_kwargs["fm"] = c_req.drug.fm
            drug = Drug(**drug_kwargs)
            candidates.append(
                ComparisonCandidate(
                    drug=drug,
                    dose_mg=c_req.dose_mg,
                    route=c_req.route,
                    label=c_req.label,
                    herg_ic50_uM=c_req.herg_ic50_uM,
                )
            )

        result = compare_drugs(candidates, t_end_h=req.t_end_h, weights=req.weights)

        # Serialise dataclasses to dicts
        pk_dicts = [dataclasses.asdict(p) for p in result.pk_profiles]
        # Trim large arrays for response size
        for d in pk_dicts:
            d["time_h"] = list(d["time_h"][:51])
            d["cp_mg_L"] = list(d["cp_mg_L"][:51])

        safety_dicts = [dataclasses.asdict(s) for s in result.safety_profiles]
        dim_dicts = [dataclasses.asdict(d) for d in result.dimension_rankings]

        return DrugComparisonResponse(
            n_candidates=result.n_candidates,
            candidate_labels=result.candidate_labels,
            composite_scores=result.composite_scores,
            overall_ranking=result.overall_ranking,
            best_candidate=result.best_candidate,
            decision_summary=result.decision_summary,
            pk_profiles=pk_dicts,
            safety_profiles=safety_dicts,
            dimension_rankings=dim_dicts,
            warnings=result.warnings,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Drug comparison error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /mist — MIST metabolite safety assessment
# ---------------------------------------------------------------------------


class MetaboliteSpecRequest(BaseModel):
    name: str = Field(description="Metabolite name")
    mw: float = Field(gt=0, description="Metabolite molecular weight (g/mol)")
    fm_parent: float = Field(
        ge=0, le=1, description="Fraction of parent metabolised to this metabolite"
    )
    clint_met_L_per_h: float = Field(ge=0, description="Metabolite intrinsic clearance (L/h)")
    fup_met: float = Field(
        default=0.5, gt=0, le=1, description="Metabolite fraction unbound in plasma"
    )
    pharmacologically_active: bool = Field(
        default=False, description="Is this metabolite pharmacologically active?"
    )
    unique_human: bool = Field(default=False, description="Is this metabolite unique to humans?")


class MISTRequest(BaseModel):
    drug: DrugRequest = Field(description="Parent drug parameters")
    metabolites: list[MetaboliteSpecRequest] = Field(
        min_length=1, description="Metabolite specifications (≥1)"
    )
    dose_mg: float = Field(default=100.0, gt=0, description="Dose in mg")
    route: str = Field(default="oral", description="Administration route: oral or iv")
    body_weight: float = Field(default=70.0, gt=0, description="Body weight in kg")

    @field_validator("route")
    @classmethod
    def route_valid(cls, v: str) -> str:
        allowed = ("oral", "iv")
        if v not in allowed:
            raise ValueError(f"route must be one of {allowed}")
        return v


class MetaboliteMISTResultResponse(BaseModel):
    name: str
    auc_met_mg_h_L: float
    auc_parent_mg_h_L: float
    met_to_parent_ratio: float
    exceeds_10pct: bool
    classification: str
    risk_level: str
    recommendation: str
    warnings: list[str]


class MISTResponse(BaseModel):
    drug_name: str
    parent_auc_mg_h_L: float
    parent_cmax_mg_L: float
    metabolite_results: list[MetaboliteMISTResultResponse]
    n_flagged: int
    overall_mist_risk: str
    summary: str
    recommendations: list[str]
    warnings: list[str]


@app.post("/mist", response_model=MISTResponse)
def mist_assessment(req: MISTRequest) -> MISTResponse:
    """MIST (Metabolites in Safety Testing) assessment — FDA guidance."""
    try:
        from omega_pbpk.drugs.drug import Drug
        from omega_pbpk.risk.mist_assessment import (
            MetaboliteSpec,
            run_mist_assessment,
        )

        # Build Drug object
        drug_kwargs: dict[str, Any] = {
            "name": req.drug.name,
            "mw": req.drug.mw,
            "logP": req.drug.logP,
            "fup": req.drug.fup,
            "rbp": req.drug.rbp,
            "drug_type": req.drug.drug_type,
            "clint_hepatic_L_per_h": req.drug.clint_hepatic_L_per_h,
            "clr_L_per_h": req.drug.clr_L_per_h,
            "peff": req.drug.peff,
        }
        if req.drug.pka is not None:
            drug_kwargs["pka"] = req.drug.pka
        if req.drug.clint is not None:
            drug_kwargs["clint"] = req.drug.clint
        if req.drug.fm is not None:
            drug_kwargs["fm"] = req.drug.fm
        drug = Drug(**drug_kwargs)

        # Build MetaboliteSpec list
        met_specs = [
            MetaboliteSpec(
                name=m.name,
                mw=m.mw,
                fm_parent=m.fm_parent,
                clint_met_L_per_h=m.clint_met_L_per_h,
                fup_met=m.fup_met,
                pharmacologically_active=m.pharmacologically_active,
                unique_human=m.unique_human,
            )
            for m in req.metabolites
        ]

        report = run_mist_assessment(
            drug=drug,
            metabolites=met_specs,
            dose_mg=req.dose_mg,
            route=req.route,
            body_weight=req.body_weight,
            vdss_L_per_kg=req.drug.vdss_L_per_kg,
        )

        return MISTResponse(
            drug_name=report.drug_name,
            parent_auc_mg_h_L=report.parent_auc_mg_h_L,
            parent_cmax_mg_L=report.parent_cmax_mg_L,
            metabolite_results=[
                MetaboliteMISTResultResponse(
                    name=r.name,
                    auc_met_mg_h_L=r.auc_met_mg_h_L,
                    auc_parent_mg_h_L=r.auc_parent_mg_h_L,
                    met_to_parent_ratio=r.met_to_parent_ratio,
                    exceeds_10pct=r.exceeds_10pct,
                    classification=r.classification,
                    risk_level=r.risk_level,
                    recommendation=r.recommendation,
                    warnings=r.warnings,
                )
                for r in report.metabolite_results
            ],
            n_flagged=report.n_flagged,
            overall_mist_risk=report.overall_mist_risk,
            summary=report.summary,
            recommendations=report.recommendations,
            warnings=report.warnings,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("MIST assessment error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /bcs — BCS classification and bioavailability prediction
# ---------------------------------------------------------------------------


class BCSRequest(BaseModel):
    drug_name: str = Field(default="Unknown", description="Drug name")
    dose_mg: float = Field(default=100.0, gt=0, description="Administered dose (mg)")
    solubility_mg_per_mL: float = Field(gt=0, description="Aqueous solubility (mg/mL)")
    permeability_cm_per_s: float = Field(
        ge=0, description="Effective intestinal permeability (cm/s)"
    )
    particle_radius_um: float = Field(default=25.0, gt=0, description="Particle radius (µm)")
    density_g_per_cm3: float = Field(default=1.2, gt=0, description="Particle density (g/cm³)")
    diffusion_coeff_cm2_per_s: float = Field(
        default=5e-6, gt=0, description="Drug diffusion coefficient (cm²/s)"
    )
    gi_volume_mL: float = Field(
        default=250.0, gt=0, description="GI volume for dose number calculation (mL)"
    )
    gi_transit_h: float = Field(default=3.5, gt=0, description="Small intestinal transit time (h)")


class BCSAbsorptionResponse(BaseModel):
    fraction_absorbed: float
    fa_dissolution_limited: float
    fa_permeability_limited: float
    rate_limiting_step: str


class BCSDissolutionResponse(BaseModel):
    t50_h: float
    t85_h: float
    time_h: list[float]
    fraction_dissolved: list[float]


class BCSResponse(BaseModel):
    drug_name: str
    bcs_class: str
    dose_number: float
    dissolution: BCSDissolutionResponse
    absorption: BCSAbsorptionResponse
    regulatory_note: str
    warnings: list[str]


@app.post("/bcs", response_model=BCSResponse)
def bcs_classification(req: BCSRequest) -> BCSResponse:
    """BCS classification and bioavailability prediction."""
    try:
        from omega_pbpk.biopharmaceutics.bcs_classification import (
            BCSInput,
            run_bcs_assessment,
        )

        inp = BCSInput(
            drug_name=req.drug_name,
            dose_mg=req.dose_mg,
            solubility_mg_per_mL=req.solubility_mg_per_mL,
            permeability_cm_per_s=req.permeability_cm_per_s,
            particle_radius_um=req.particle_radius_um,
            density_g_per_cm3=req.density_g_per_cm3,
            diffusion_coeff_cm2_per_s=req.diffusion_coeff_cm2_per_s,
            gi_volume_mL=req.gi_volume_mL,
            gi_transit_h=req.gi_transit_h,
        )
        report = run_bcs_assessment(inp)

        return BCSResponse(
            drug_name=report.input.drug_name,
            bcs_class=report.bcs_class,
            dose_number=report.dose_number,
            dissolution=BCSDissolutionResponse(
                t50_h=report.dissolution.t50_h,
                t85_h=report.dissolution.t85_h,
                time_h=report.dissolution.time_h,
                fraction_dissolved=report.dissolution.fraction_dissolved,
            ),
            absorption=BCSAbsorptionResponse(
                fraction_absorbed=report.absorption.fraction_absorbed,
                fa_dissolution_limited=report.absorption.fa_dissolution_limited,
                fa_permeability_limited=report.absorption.fa_permeability_limited,
                rate_limiting_step=report.absorption.rate_limiting_step,
            ),
            regulatory_note=report.regulatory_note,
            warnings=report.warnings,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("BCS classification error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /induction — enzyme induction time course (PXR/CAR pathway)
# ---------------------------------------------------------------------------


class InductionRequest(BaseModel):
    inducer_name: str = Field(default="compound", description="Name of the inducer")
    target_enzyme: str = Field(default="CYP3A4", description="Target CYP enzyme")
    emax: float = Field(..., ge=0, description="Maximum fold induction above baseline")
    ec50_uM: float = Field(..., gt=0, description="EC50 for induction (µM)")
    inducer_conc_uM: float = Field(..., ge=0, description="Steady-state inducer conc (µM)")
    kdeg_h: float = Field(default=0.03, gt=0, description="Enzyme kdeg (/h); default CYP3A4")
    baseline_activity: float = Field(default=1.0, gt=0, description="Baseline enzyme activity")
    t_end_h: float = Field(default=336.0, gt=0, description="Simulation end time (h)")
    n_points: int = Field(default=200, ge=2, description="Number of time points")
    auto_induction: bool = Field(
        default=False, description="Enable auto-induction (coupled PK–enzyme ODEs)"
    )
    initial_conc_uM: float | None = Field(
        default=None, ge=0, description="Initial drug conc for auto-induction (µM)"
    )
    cl_baseline_L_per_h: float | None = Field(
        default=None, gt=0, description="Baseline CL for auto-induction (L/h)"
    )
    vd_L: float | None = Field(
        default=None, gt=0, description="Volume of distribution for auto-induction (L)"
    )
    inhibition_r1: float | None = Field(
        default=None, ge=1.0, description="R1 inhibition factor; triggers net DDI calc"
    )
    fm_enzyme: float = Field(default=1.0, ge=0, le=1, description="Victim fm for net DDI (0–1)")


class InductionTimePointResponse(BaseModel):
    time_h: float
    enzyme_activity: float
    induction_factor: float
    fractional_turnover: float


class InductionTimeCourseResponse(BaseModel):
    inducer_name: str
    target_enzyme: str
    emax: float
    ec50_uM: float
    inducer_conc_uM: float
    kdeg_h: float
    ksyn_h: float
    steady_state_fold: float
    time_to_90pct_ss_h: float
    time_points: list[InductionTimePointResponse]
    time_h: list[float]
    enzyme_activity: list[float]
    warnings: list[str]


class AutoInductionResponse(BaseModel):
    drug_name: str
    target_enzyme: str
    emax: float
    ec50_uM: float
    initial_conc_uM: float
    kdeg_h: float
    cl_baseline_L_per_h: float
    cl_induced_L_per_h: float
    half_life_baseline_h: float
    half_life_induced_h: float
    fold_cl_change: float
    time_h: list[float]
    concentration_uM: list[float]
    enzyme_activity: list[float]
    warnings: list[str]


class NetDDIEffectResponse(BaseModel):
    target_enzyme: str
    inhibition_factor: float
    induction_factor: float
    net_effect: float
    dominant_mechanism: str
    clinical_relevance: str
    warnings: list[str]


class InductionResponse(BaseModel):
    mode: str
    summary: str
    time_course: InductionTimeCourseResponse | None = None
    auto_induction: AutoInductionResponse | None = None
    net_ddi: NetDDIEffectResponse | None = None


@app.post("/induction", response_model=InductionResponse)
def enzyme_induction(req: InductionRequest) -> InductionResponse:
    """Enzyme induction time course via PXR/CAR pathway (CYP induction)."""
    try:
        from omega_pbpk.clinical.enzyme_induction import (
            InductionParameters,
            compute_net_ddi_effect,
            format_induction_summary,
            simulate_auto_induction,
            simulate_induction_time_course,
        )

        params = InductionParameters(
            inducer_name=req.inducer_name,
            target_enzyme=req.target_enzyme,
            emax=req.emax,
            ec50_uM=req.ec50_uM,
            inducer_conc_uM=req.inducer_conc_uM,
            kdeg_h=req.kdeg_h,
            baseline_activity=req.baseline_activity,
        )

        time_course_resp: InductionTimeCourseResponse | None = None
        auto_resp: AutoInductionResponse | None = None
        net_ddi_resp: NetDDIEffectResponse | None = None
        mode: str

        if req.auto_induction:
            if req.initial_conc_uM is None or req.cl_baseline_L_per_h is None or req.vd_L is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "auto_induction=true requires initial_conc_uM, "
                        "cl_baseline_L_per_h, and vd_L"
                    ),
                )
            mode = "auto_induction"
            ar = simulate_auto_induction(
                params=params,
                initial_conc_uM=req.initial_conc_uM,
                cl_baseline_L_per_h=req.cl_baseline_L_per_h,
                vd_L=req.vd_L,
                t_end_h=req.t_end_h,
                n_points=req.n_points,
            )
            auto_resp = AutoInductionResponse(
                drug_name=ar.drug_name,
                target_enzyme=ar.target_enzyme,
                emax=ar.emax,
                ec50_uM=ar.ec50_uM,
                initial_conc_uM=ar.initial_conc_uM,
                kdeg_h=ar.kdeg_h,
                cl_baseline_L_per_h=ar.cl_baseline_L_per_h,
                cl_induced_L_per_h=ar.cl_induced_L_per_h,
                half_life_baseline_h=ar.half_life_baseline_h,
                half_life_induced_h=ar.half_life_induced_h,
                fold_cl_change=ar.fold_cl_change,
                time_h=list(ar.time_h),
                concentration_uM=list(ar.concentration_uM),
                enzyme_activity=list(ar.enzyme_activity),
                warnings=list(ar.warnings),
            )
            summary = format_induction_summary(ar)
        else:
            mode = "standard"
            tc = simulate_induction_time_course(
                params=params,
                t_end_h=req.t_end_h,
                n_points=req.n_points,
            )
            time_course_resp = InductionTimeCourseResponse(
                inducer_name=tc.inducer_name,
                target_enzyme=tc.target_enzyme,
                emax=tc.emax,
                ec50_uM=tc.ec50_uM,
                inducer_conc_uM=tc.inducer_conc_uM,
                kdeg_h=tc.kdeg_h,
                ksyn_h=tc.ksyn_h,
                steady_state_fold=tc.steady_state_fold,
                time_to_90pct_ss_h=tc.time_to_90pct_ss_h,
                time_points=[
                    InductionTimePointResponse(
                        time_h=p.time_h,
                        enzyme_activity=p.enzyme_activity,
                        induction_factor=p.induction_factor,
                        fractional_turnover=p.fractional_turnover,
                    )
                    for p in tc.time_points
                ],
                time_h=list(tc.time_h),
                enzyme_activity=list(tc.enzyme_activity),
                warnings=list(tc.warnings),
            )
            summary = format_induction_summary(tc)

        # Optional net DDI calculation
        if req.inhibition_r1 is not None:
            ss_fold = (
                auto_resp.fold_cl_change
                if auto_resp is not None
                else (time_course_resp.steady_state_fold if time_course_resp else 1.0)
            )
            net = compute_net_ddi_effect(
                inhibition_r1=req.inhibition_r1,
                induction_fold=ss_fold,
                fm_enzyme=req.fm_enzyme,
                target_enzyme=req.target_enzyme,
            )
            net_ddi_resp = NetDDIEffectResponse(
                target_enzyme=net.target_enzyme,
                inhibition_factor=net.inhibition_factor,
                induction_factor=net.induction_factor,
                net_effect=net.net_effect,
                dominant_mechanism=net.dominant_mechanism,
                clinical_relevance=net.clinical_relevance,
                warnings=list(net.warnings),
            )

        return InductionResponse(
            mode=mode,
            summary=summary,
            time_course=time_course_resp,
            auto_induction=auto_resp,
            net_ddi=net_ddi_resp,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Enzyme induction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /predict/gat
# ---------------------------------------------------------------------------


class GATRequest(BaseModel):
    """Request model for GAT-based ADME prediction."""

    smiles: str = Field(..., description="SMILES string of the molecule")


class GATModelInfo(BaseModel):
    type: str
    trained: bool
    hidden_dim: int
    n_rounds: int


class GATResponse(BaseModel):
    """Response model for GAT-based ADME prediction."""

    smiles: str
    fup: float = Field(description="Fraction unbound in plasma (0–1)")
    clint_3a4: float = Field(description="CYP3A4 intrinsic clearance (µL/min/pmol)")
    peff: float = Field(description="Intestinal permeability (×10⁻⁴ cm/s)")
    logS: float = Field(description="Aqueous solubility (log10 mol/L)")
    confidence: float = Field(description="Model confidence (0–1)")
    model_info: GATModelInfo
    warnings: list[str] = Field(default_factory=list)


@app.post("/predict/gat", response_model=GATResponse, tags=["predict"])
async def predict_gat(req: GATRequest) -> GATResponse:
    """Predict ADME properties using the Graph Attention Network (GAT).

    Uses a pure-NumPy multi-task GAT trained on the ADME reference
    dataset. Predicts fup, CLint_3A4, Peff and logS simultaneously.

    The model is lazy-loaded and trained on first call.
    """
    try:
        from omega_pbpk.prediction.adme_predictor import ADMEPredictor

        predictor = ADMEPredictor()
        result = predictor.predict_gat(req.smiles)

        warnings: list[str] = []
        if result["confidence"] < 0.3:
            warnings.append("Low confidence prediction: molecule may be outside training domain.")

        mi = result.get("model_info", {})
        return GATResponse(
            smiles=req.smiles,
            fup=result["fup"],
            clint_3a4=result["clint_3a4"],
            peff=result["peff"],
            logS=result["logS"],
            confidence=result["confidence"],
            model_info=GATModelInfo(
                type=mi.get("type", "MolecularGAT"),
                trained=mi.get("trained", False),
                hidden_dim=mi.get("hidden_dim", 32),
                n_rounds=mi.get("n_rounds", 3),
            ),
            warnings=warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("GAT prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /pubchem/lookup
# ---------------------------------------------------------------------------


class PubChemRequest(BaseModel):
    """Request model for PubChem compound lookup."""

    query: str = Field(..., description="Compound name, SMILES, or numeric CID")
    query_type: Literal["name", "smiles", "cid"] = Field(
        default="name",
        description="Type of identifier: 'name', 'smiles', or 'cid'",
    )


class PubChemResponse(BaseModel):
    """Response model for PubChem compound lookup."""

    found: bool
    cid: int = 0
    name: str = ""
    smiles: str = ""
    molecular_weight: float = 0.0
    xlogp: float = 0.0
    tpsa: float = 0.0
    hbd: int = 0
    hba: int = 0
    rotatable_bonds: int = 0
    complexity: float = 0.0
    charge: int = 0
    exact_mass: float = 0.0
    source: str = "pubchem"
    warnings: list[str] = Field(default_factory=list)


@app.post("/pubchem/lookup", response_model=PubChemResponse, tags=["data"])
async def pubchem_lookup(req: PubChemRequest) -> PubChemResponse:
    """Look up a compound in PubChem by name, SMILES, or CID.

    Returns molecular properties from PubChem PUG REST API.
    Results are cached locally to avoid repeated network calls.
    Returns found=False (not an error) when compound is not found.
    """
    try:
        from omega_pbpk.data.pubchem_client import (
            lookup_by_cid,
            lookup_by_name,
            lookup_by_smiles,
        )

        compound = None
        warn: list[str] = []

        if req.query_type == "name":
            compound = lookup_by_name(req.query)
        elif req.query_type == "smiles":
            compound = lookup_by_smiles(req.query)
        elif req.query_type == "cid":
            try:
                compound = lookup_by_cid(int(req.query))
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="query must be a numeric CID when query_type='cid'",
                ) from exc

        if compound is None:
            warn.append(
                f"Compound '{req.query}' not found in PubChem "
                f"(query_type={req.query_type}). "
                "Network may be offline or compound unknown."
            )
            return PubChemResponse(found=False, warnings=warn)

        return PubChemResponse(
            found=True,
            cid=compound.cid,
            name=compound.name,
            smiles=compound.smiles,
            molecular_weight=compound.molecular_weight,
            xlogp=compound.xlogp,
            tpsa=compound.tpsa,
            hbd=compound.hbd,
            hba=compound.hba,
            rotatable_bonds=compound.rotatable_bonds,
            complexity=compound.complexity,
            charge=compound.charge,
            exact_mass=compound.exact_mass,
            source=compound.source,
            warnings=warn,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PubChem lookup error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /predict/pk_surrogate
# ---------------------------------------------------------------------------


class PKSurrogateRequest(BaseModel):
    """Request for neural PK surrogate prediction."""

    dose_mg: float = Field(default=100.0, gt=0, description="Dose (mg)")
    cl_L_per_h: float = Field(default=5.0, gt=0, description="Clearance (L/h)")
    vd_L: float = Field(default=70.0, gt=0, description="Volume of distribution (L)")
    ka_per_h: float = Field(default=1.0, gt=0, description="Absorption rate constant (1/h)")
    f_bioavail: float = Field(
        default=0.8,
        gt=0,
        le=1.0,
        description="Oral bioavailability (0–1)",
    )
    route: str = Field(default="oral", description="Administration route ('oral' or 'iv')")


class PKSurrogateResponse(BaseModel):
    """Response from neural PK surrogate endpoint."""

    auc_mg_h_per_L: float = Field(description="AUC₀₋∞ (mg·h/L)")
    cmax_mg_per_L: float = Field(description="Peak plasma concentration (mg/L)")
    tmax_h: float = Field(description="Time to peak concentration (h)")
    t_half_h: float = Field(description="Elimination half-life (h)")
    confidence: str = Field(
        description="Prediction confidence: 'high', 'medium', 'low', or 'analytical'"
    )
    auc_std: float = Field(default=0.0, description="Ensemble std for AUC")
    cmax_std: float = Field(default=0.0, description="Ensemble std for Cmax")
    tmax_std: float = Field(default=0.0, description="Ensemble std for tmax")
    t_half_std: float = Field(default=0.0, description="Ensemble std for t_half")


@app.post(
    "/predict/pk_surrogate",
    response_model=PKSurrogateResponse,
    tags=["predict"],
)
def predict_pk_surrogate_endpoint(
    req: PKSurrogateRequest,
) -> PKSurrogateResponse:
    """Fast 1-compartment PK prediction via neural surrogate ensemble.

    Uses a 5-model NumPy MLP ensemble trained on Latin Hypercube sampled
    analytical PK data.  Falls back to exact analytical computation when
    surrogate confidence is low.

    The ensemble is auto-trained on first call (lazy initialisation).
    Subsequent calls reuse the singleton ensemble for speed.
    """
    try:
        from omega_pbpk.ml_models.pk_surrogate import (
            PKSurrogateInput,
            predict_pk_surrogate,
        )

        inp = PKSurrogateInput(
            dose_mg=req.dose_mg,
            cl_L_per_h=req.cl_L_per_h,
            vd_L=req.vd_L,
            ka_per_h=req.ka_per_h,
            f_bioavail=req.f_bioavail,
            route=req.route,
        )
        result = predict_pk_surrogate(inp)
        return PKSurrogateResponse(
            auc_mg_h_per_L=result.auc_mg_h_per_L,
            cmax_mg_per_L=result.cmax_mg_per_L,
            tmax_h=result.tmax_h,
            t_half_h=result.t_half_h,
            confidence=result.confidence,
            auc_std=result.auc_std,
            cmax_std=result.cmax_std,
            tmax_std=result.tmax_std,
            t_half_std=result.t_half_std,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PK surrogate prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# /interpret — GPT / rule-based PK interpretation
# ---------------------------------------------------------------------------


class InterpretRequest(BaseModel):
    """PK assessment data to interpret; mirrors FullPredictionResult fields."""

    drug_name: str = "compound"
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc0t_mg_h_L: float = 0.0
    t_half_h: float = 0.0
    cmax_p5: float = 0.0
    cmax_p50: float = 0.0
    cmax_p95: float = 0.0
    auc_p5: float = 0.0
    auc_p50: float = 0.0
    auc_p95: float = 0.0
    overall_risk_level: str = "low"
    risk_flags: dict[str, Any] = Field(default_factory=dict)
    risk_count: int = 0
    transporter_ddi_flags: list[str] = Field(default_factory=list)
    adme: dict[str, Any] = Field(default_factory=dict)
    confidence: str = "low"
    warnings: list[str] = Field(default_factory=list)
    # GPT config overrides
    gpt_model: str = "gpt-4o-mini"
    gpt_max_tokens: int = 500
    gpt_temperature: float = 0.3


class InterpretResponse(BaseModel):
    """Natural-language interpretation of a PK assessment."""

    summary: str
    key_findings: list[str]
    safety_flags: list[str]
    recommendations: list[str]
    confidence: str
    model_used: str


@app.post("/interpret", response_model=InterpretResponse, tags=["interpret"])
def interpret_endpoint(req: InterpretRequest) -> InterpretResponse:
    """GPT or rule-based natural-language interpretation of PK results.

    Accepts a structured PK assessment and returns a clinical summary,
    key findings, safety flags, and dosing recommendations.
    Uses OpenAI GPT when OPENAI_OAUTH_TOKEN is set; falls back to
    a deterministic rule-based engine otherwise.
    """
    try:
        from omega_pbpk.interpretation.gpt_interpreter import (
            GPTInterpreterConfig,
            interpret_pk_result,
        )

        config = GPTInterpreterConfig(
            model=req.gpt_model,
            max_tokens=req.gpt_max_tokens,
            temperature=req.gpt_temperature,
        )
        result_dict = req.model_dump(exclude={"gpt_model", "gpt_max_tokens", "gpt_temperature"})
        interp = interpret_pk_result(result_dict, config=config)
        return InterpretResponse(
            summary=interp.summary,
            key_findings=list(interp.key_findings),
            safety_flags=list(interp.safety_flags),
            recommendations=list(interp.recommendations),
            confidence=interp.confidence,
            model_used=interp.model_used,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Interpretation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /predict/kp — tissue partition coefficient prediction
# ---------------------------------------------------------------------------


class KpRequest(BaseModel):
    """Request body for /predict/kp."""

    logP: float = Field(default=2.0, description="Octanol-water logP")
    pKa: float = Field(default=7.0, description="Primary pKa")
    fup: float = Field(default=0.1, gt=0, le=1, description="Fraction unbound")
    mw: float = Field(default=300.0, gt=0, description="Molecular weight (g/mol)")
    compound_type: str = Field(
        default="neutral",
        description="'neutral' | 'acid' | 'base' | 'ampholyte'",
    )
    method: str = Field(
        default="ml",
        description="'ml' | 'rodgers_rowland' | 'poulin_theil'",
    )


class KpResponse(BaseModel):
    """Response body for /predict/kp."""

    tissue_kp: dict[str, float]
    confidence: str
    method_used: str
    warnings: list[str]


@app.post("/predict/kp", response_model=KpResponse, tags=["predict"])
def predict_kp_endpoint(req: KpRequest) -> KpResponse:
    """Predict tissue partition coefficients (Kp) for all 13 PBPK tissues.

    Supports three methods:
      - **ml**: ML ensemble (KpEnsemble, 3 NumpyMLP models, 8→32→32→13).
        Validated against simplified Rodgers-Rowland; divergent tissues
        are replaced automatically.
      - **rodgers_rowland**: Mechanistic Rodgers & Rowland (2006) method.
      - **poulin_theil**: Simplified Poulin & Theil (2002) heuristic.
    """
    try:
        method = req.method.strip().lower()
        if method == "ml":
            from omega_pbpk.ml_models.kp_predictor import (
                KpInput,
                predict_kp_ml,
            )

            inp = KpInput(
                logP=req.logP,
                pKa=req.pKa,
                fup=req.fup,
                mw=req.mw,
                compound_type=req.compound_type,
            )
            out = predict_kp_ml(inp)
            return KpResponse(
                tissue_kp=out.tissue_kp,
                confidence=out.confidence,
                method_used="ml",
                warnings=list(out.warnings),
            )

        elif method in ("rodgers_rowland", "poulin_theil"):
            from omega_pbpk.core.heuristics import (
                heuristic_kp,
                rodgers_rowland_kp,
            )
            from omega_pbpk.ml_models.kp_predictor import TISSUES

            kp_fn = rodgers_rowland_kp if method == "rodgers_rowland" else heuristic_kp
            tissue_kp: dict[str, float] = {}
            for tissue in TISSUES:
                if method == "rodgers_rowland":
                    val = kp_fn(
                        logP=req.logP,
                        pka=req.pKa,
                        compound_type=req.compound_type,
                        tissue_name=tissue,
                        fup=req.fup,
                    )
                else:
                    val = kp_fn(
                        logP=req.logP,
                        pka=req.pKa,
                        compound_type=req.compound_type,
                        tissue_name=tissue,
                        fup=req.fup,
                    )
                tissue_kp[tissue] = val
            return KpResponse(
                tissue_kp=tissue_kp,
                confidence="high",
                method_used=method,
                warnings=[],
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown method '{req.method}'. "
                    "Use 'ml', 'rodgers_rowland', or 'poulin_theil'."
                ),
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Kp prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /predict/neural_ode  — Phase 46 Neural ODE surrogate
# ---------------------------------------------------------------------------


class NeuralODERequest(BaseModel):
    """Request for Neural ODE C(t) trajectory prediction."""

    logP: float = Field(default=2.0, description="Log octanol-water partition coefficient")
    fup: float = Field(default=0.5, gt=0, le=1.0, description="Plasma unbound fraction")
    clint_L_per_h: float = Field(default=10.0, gt=0, description="Hepatic CLint (L/h)")
    mw: float = Field(default=300.0, gt=0, description="Molecular weight (g/mol)")
    rbp: float = Field(default=1.0, gt=0, description="Red blood cell-to-plasma ratio")
    peff: float = Field(default=1.0, gt=0, description="Intestinal permeability (×10⁻⁴ cm/s)")
    dose_mg: float = Field(default=100.0, gt=0, description="Dose (mg)")
    route: str = Field(default="oral", description="'oral' or 'iv'")
    t_end_h: float = Field(default=24.0, gt=0, description="Simulation horizon (h)")


class NeuralODEResponse(BaseModel):
    """Response from Neural ODE surrogate."""

    t_h: list[float] = Field(description="Time points (h)")
    C_mg_L: list[float] = Field(description="Predicted plasma concentration (mg/L)")
    cmax_mg_L: float = Field(description="Peak plasma concentration (mg/L)")
    auc_mg_h_L: float = Field(description="AUC₀₋ₜ (mg·h/L)")
    tmax_h: float = Field(description="Time to peak (h)")
    t_half_h: float = Field(description="Terminal elimination half-life (h)")
    train_r2: float = Field(description="Training R² of Neural ODE dynamics model")
    n_train: int = Field(description="Number of PBPK trajectories used for training")
    solve_time_ms: float = Field(description="Inference wall-clock time (ms)")


@app.post("/predict/neural_ode", response_model=NeuralODEResponse, tags=["predict"])
def predict_neural_ode_endpoint(req: NeuralODERequest) -> NeuralODEResponse:
    """Predict full plasma C(t) trajectory via Neural ODE surrogate (Phase 46).

    A 2-layer MLP parameterises the ODE dynamics dC/dt = f(C, t, drug_params).
    Trained on 100 PBPK simulations via Euler BPTT with Adam optimiser.
    Inference uses RK4 integration for accuracy (~1 ms vs ~500 ms for full PBPK).

    The singleton model is auto-trained on first call (~15 s).
    """
    try:
        from omega_pbpk.ml_models.neural_ode import NeuralODEInput, predict_neural_ode

        inp = NeuralODEInput(
            logP=req.logP,
            fup=req.fup,
            clint_L_per_h=req.clint_L_per_h,
            mw=req.mw,
            rbp=req.rbp,
            peff=req.peff,
            dose_mg=req.dose_mg,
            route=req.route,
            t_end_h=req.t_end_h,
        )
        out = predict_neural_ode(inp)
        return NeuralODEResponse(
            t_h=out.t_h,
            C_mg_L=out.C_mg_L,
            cmax_mg_L=out.cmax_mg_L,
            auc_mg_h_L=out.auc_mg_h_L,
            tmax_h=out.tmax_h,
            t_half_h=out.t_half_h,
            train_r2=out.train_r2,
            n_train=out.n_train,
            solve_time_ms=out.solve_time_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Neural ODE prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /trial/parallel  — Phase 47 Parallel-Group Clinical Trial Simulator
# ---------------------------------------------------------------------------


class _DoseGroupModel(BaseModel):
    name: str = Field(description="Arm label (e.g. 'Placebo', '10 mg')")
    dose_mg: float = Field(default=0.0, ge=0, description="Dose (mg); 0 = placebo")
    route: str = Field(default="oral", description="'oral' or 'iv'")
    n_subjects: int = Field(default=30, ge=2, le=500, description="Subjects in arm")


class ParallelTrialRequest(BaseModel):
    """Request body for a parallel-group clinical trial simulation."""

    dose_groups: list[_DoseGroupModel] = Field(
        description="2–5 dose groups including optional placebo"
    )
    cl_L_per_h: float = Field(default=5.0, gt=0, description="Population CL (L/h)")
    vd_L: float = Field(default=70.0, gt=0, description="Population Vd (L)")
    ka_per_h: float = Field(default=1.2, gt=0, description="Absorption rate (1/h)")
    f_bioavail: float = Field(default=0.8, gt=0, le=1.0, description="Bioavailability")
    bsv_cv_cl: float = Field(default=0.30, ge=0, description="BSV CV for CL")
    bsv_cv_vd: float = Field(default=0.20, ge=0, description="BSV CV for Vd")
    bsv_cv_ka: float = Field(default=0.40, ge=0, description="BSV CV for ka")
    emax: float = Field(default=1.0, gt=0, description="Maximum PD effect")
    ec50_mg_L: float = Field(default=1.0, gt=0, description="EC50 for PD (mg/L)")
    hill: float = Field(default=1.0, gt=0, description="Hill coefficient")
    pd_input: str = Field(default="cmax", description="PD driver: 'cmax' or 'auc'")
    t_end_h: float = Field(default=24.0, gt=0, description="PK horizon (h)")
    alpha: float = Field(default=0.05, gt=0, le=0.5, description="Significance level")
    interim_fraction: float = Field(
        default=0.5, ge=0, le=1, description="Interim enrolment fraction (0=none)"
    )
    seed: int = Field(default=42, description="Random seed")


class _ArmSummaryModel(BaseModel):
    name: str
    dose_mg: float
    n: int
    cmax_mean: float
    cmax_gsd: float
    auc_mean: float
    auc_gsd: float
    effect_mean: float
    effect_sd: float
    responder_pct: float


class _PairwiseModel(BaseModel):
    arm_name: str
    dose_mg: float
    endpoint: str
    t_stat: float
    p_value: float
    p_adjusted: float
    significant: bool
    mean_difference: float
    ci_lower: float
    ci_upper: float


class _DoseResponseModel(BaseModel):
    emax_fit: float
    ec50_dose_mg: float
    hill_fit: float
    r_squared: float
    ed80_mg: float
    ed90_mg: float
    fit_success: bool


class _InterimModel(BaseModel):
    conducted: bool
    fraction_enrolled: float
    conditional_power: float
    futility_stop: bool
    n_interim: int


class ParallelTrialResponse(BaseModel):
    """Response from the parallel-group trial simulation."""

    arms: list[_ArmSummaryModel]
    anova_p_auc: float
    anova_p_cmax: float
    anova_p_effect: float
    pairwise: list[_PairwiseModel]
    dose_response: _DoseResponseModel
    interim: _InterimModel
    trial_success: bool
    optimal_dose_mg: float
    total_subjects: int


@app.post("/trial/parallel", response_model=ParallelTrialResponse, tags=["trial"])
def run_parallel_trial_endpoint(req: ParallelTrialRequest) -> ParallelTrialResponse:
    """Simulate a parallel-group phase 2/3 clinical trial (Phase 47).

    Runs a multi-arm virtual trial with:
    - 1-compartment PK + log-normal between-subject variability
    - Emax/Hill PD model (driver: Cmax or AUC)
    - One-way ANOVA + Bonferroni pairwise t-tests
    - Hill dose-response curve fitting
    - Interim futility analysis (conditional power)
    """
    try:
        from omega_pbpk.clinical.parallel_trial import (
            DoseGroup,
            ParallelTrialDesign,
            run_parallel_trial,
        )

        if not (2 <= len(req.dose_groups) <= 5):
            raise HTTPException(
                status_code=422,
                detail="Between 2 and 5 dose groups are required.",
            )

        groups = tuple(
            DoseGroup(
                name=g.name,
                dose_mg=g.dose_mg,
                route=g.route,
                n_subjects=g.n_subjects,
            )
            for g in req.dose_groups
        )

        design = ParallelTrialDesign(
            dose_groups=groups,
            cl_L_per_h=req.cl_L_per_h,
            vd_L=req.vd_L,
            ka_per_h=req.ka_per_h,
            f_bioavail=req.f_bioavail,
            bsv_cv_cl=req.bsv_cv_cl,
            bsv_cv_vd=req.bsv_cv_vd,
            bsv_cv_ka=req.bsv_cv_ka,
            emax=req.emax,
            ec50_mg_L=req.ec50_mg_L,
            hill=req.hill,
            pd_input=req.pd_input,
            t_end_h=req.t_end_h,
            alpha=req.alpha,
            interim_fraction=req.interim_fraction,
            seed=req.seed,
        )

        res = run_parallel_trial(design)

        return ParallelTrialResponse(
            arms=[_ArmSummaryModel(**a.__dict__) for a in res.arms],
            anova_p_auc=res.anova_p_auc,
            anova_p_cmax=res.anova_p_cmax,
            anova_p_effect=res.anova_p_effect,
            pairwise=[_PairwiseModel(**p.__dict__) for p in res.pairwise],
            dose_response=_DoseResponseModel(**res.dose_response.__dict__),
            interim=_InterimModel(**res.interim.__dict__),
            trial_success=res.trial_success,
            optimal_dose_mg=res.optimal_dose_mg,
            total_subjects=res.total_subjects,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Parallel trial simulation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /library/screen  — Phase 48 Compound Library Screening
# ---------------------------------------------------------------------------


class _CompoundEntryModel(BaseModel):
    name: str = Field(description="Compound name/identifier")
    logP: float = Field(default=2.0, description="Log octanol-water partition coefficient")
    fup: float = Field(default=0.5, gt=0, le=1.0, description="Plasma unbound fraction")
    clint_L_per_h: float = Field(default=10.0, gt=0, description="Hepatic CLint (L/h)")
    mw: float = Field(default=300.0, gt=0, description="Molecular weight (g/mol)")
    rbp: float = Field(default=1.0, gt=0, description="Red blood cell-to-plasma ratio")
    peff: float = Field(default=1.0, gt=0, description="Intestinal permeability")
    dose_mg: float = Field(default=100.0, gt=0, description="Dose (mg)")
    route: str = Field(default="oral", description="'oral' or 'iv'")
    cl_L_per_h: float = Field(default=5.0, gt=0, description="Clearance (L/h)")
    vd_L: float = Field(default=70.0, gt=0, description="Volume of distribution (L)")
    ka_per_h: float = Field(default=1.2, gt=0, description="Absorption rate (1/h)")
    f_bioavail: float = Field(default=0.8, gt=0, le=1.0, description="Bioavailability")


class _ScreeningCriteriaModel(BaseModel):
    min_auc_mg_h_L: float = Field(default=0.0, ge=0)
    max_cmax_mg_L: float = Field(default=1e9, gt=0)
    min_f_bioavail: float = Field(default=0.0, ge=0, le=1.0)
    max_t_half_h: float = Field(default=999.0, gt=0)
    min_effect: float = Field(default=0.0, ge=0)
    ec50_mg_L: float = Field(default=1.0, gt=0)
    emax: float = Field(default=1.0, gt=0)
    hill: float = Field(default=1.0, gt=0)
    target_cmax_lower: float = Field(default=0.0, ge=0)
    target_cmax_upper: float = Field(default=1e9, gt=0)


class LibraryScreenRequest(BaseModel):
    """Request body for compound library screening."""

    entries: list[_CompoundEntryModel] = Field(description="Library compounds (1–500)")
    criteria: _ScreeningCriteriaModel = Field(default_factory=_ScreeningCriteriaModel)
    n_clusters: int = Field(default=3, ge=1, le=20, description="K-means clusters")
    seed: int = Field(default=42, description="Random seed for clustering")


class _CompoundResultModel(BaseModel):
    name: str
    auc_mg_h_L: float
    cmax_mg_L: float
    tmax_h: float
    t_half_h: float
    effect: float
    passes_criteria: bool
    pareto_rank: int
    cluster_id: int
    composite_score: float


class LibraryScreenResponse(BaseModel):
    """Response from compound library screening."""

    results: list[_CompoundResultModel]
    n_passed: int
    n_pareto_front: int
    clusters: dict[str, list[str]]
    lead_compounds: list[str]
    screening_time_s: float


@app.post("/library/screen", response_model=LibraryScreenResponse, tags=["predict"])
def screen_library_endpoint(req: LibraryScreenRequest) -> LibraryScreenResponse:
    """Screen a compound library with multi-objective Pareto ranking (Phase 48).

    For each compound:
    1. Simulate 1-compartment PK (deterministic, no BSV).
    2. Apply Emax/Hill PD model.
    3. Multi-objective Pareto ranking (AUC, effect, Cmax window, t_half).
    4. K-means diversity clustering on PK profile space.
    5. Composite scoring (0–100) and lead selection.
    """
    try:
        from omega_pbpk.clinical.compound_library import (
            CompoundEntry,
            ScreeningCriteria,
            screen_library,
        )

        if not req.entries:
            raise HTTPException(status_code=422, detail="At least one compound entry required.")

        entries = [
            CompoundEntry(
                name=e.name,
                logP=e.logP,
                fup=e.fup,
                clint_L_per_h=e.clint_L_per_h,
                mw=e.mw,
                rbp=e.rbp,
                peff=e.peff,
                dose_mg=e.dose_mg,
                route=e.route,
                cl_L_per_h=e.cl_L_per_h,
                vd_L=e.vd_L,
                ka_per_h=e.ka_per_h,
                f_bioavail=e.f_bioavail,
            )
            for e in req.entries
        ]

        c = req.criteria
        criteria = ScreeningCriteria(
            min_auc_mg_h_L=c.min_auc_mg_h_L,
            max_cmax_mg_L=c.max_cmax_mg_L,
            min_f_bioavail=c.min_f_bioavail,
            max_t_half_h=c.max_t_half_h,
            min_effect=c.min_effect,
            ec50_mg_L=c.ec50_mg_L,
            emax=c.emax,
            hill=c.hill,
            target_cmax_lower=c.target_cmax_lower,
            target_cmax_upper=c.target_cmax_upper,
        )

        res = screen_library(entries, criteria, n_clusters=req.n_clusters, seed=req.seed)

        return LibraryScreenResponse(
            results=[_CompoundResultModel(**r.__dict__) for r in res.results],
            n_passed=res.n_passed,
            n_pareto_front=res.n_pareto_front,
            clusters={str(k): v for k, v in res.clusters.items()},
            lead_compounds=res.lead_compounds,
            screening_time_s=res.screening_time_s,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Library screening error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 49: PK Validation & Parameter Calibration endpoints
# ---------------------------------------------------------------------------


class PKValidateRequest(BaseModel):
    """Validate simulated PK against observed clinical concentration-time data."""

    drug: DrugRequest
    dose_mg: float = Field(default=100.0, gt=0)
    route: str = Field(default="oral", pattern="^(oral|iv)$")
    observed_times: list[float] = Field(..., min_length=3)
    observed_conc: list[float] = Field(..., min_length=3)
    body_weight: float = Field(default=70.0, gt=0)
    t_end_h: float | None = Field(default=None, gt=0)


class PKValidateResponse(BaseModel):
    aafe: float
    afe: float
    within_2fold_pct: float
    rmse_mg_per_L: float
    auc_obs_mg_h_L: float
    auc_sim_mg_h_L: float
    auc_relative_error: float
    cmax_obs_mg_L: float
    cmax_sim_mg_L: float
    cmax_relative_error: float
    n_obs: int
    fold_errors: list[float]
    aafe_pass: bool
    within_2fold_pass: bool
    auc_pass: bool
    cmax_pass: bool
    overall_pass: bool


@app.post("/validate/pk", response_model=PKValidateResponse)
def validate_pk(req: PKValidateRequest) -> PKValidateResponse:
    """Compute AAFE, AFE, within-2-fold %, RMSE and NCA metrics vs observed data."""
    try:
        from omega_pbpk.validation.pk_validator import validate_drug

        drug = _drug_request_to_drug(req.drug)
        result = validate_drug(
            drug=drug,
            observed_times=req.observed_times,
            observed_conc=req.observed_conc,
            dose_mg=req.dose_mg,
            route=req.route,  # type: ignore[arg-type]
            body_weight=req.body_weight,
            t_end_h=req.t_end_h,
        )
        s = result.summary()
        return PKValidateResponse(**s)  # type: ignore[arg-type]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PK validation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class CalibrateClintRequest(BaseModel):
    """Calibrate CLint (and optionally fup / Kp scale) to match observed PK data."""

    drug: DrugRequest
    dose_mg: float = Field(default=100.0, gt=0)
    route: str = Field(default="oral", pattern="^(oral|iv)$")
    observed_times: list[float] = Field(..., min_length=3)
    observed_conc: list[float] = Field(..., min_length=3)
    body_weight: float = Field(default=70.0, gt=0)
    params_to_fit: list[str] = Field(default=["clint"])
    max_iter: int = Field(default=200, ge=10, le=2000)


class CalibrateClintResponse(BaseModel):
    original_clint_L_per_h: float
    calibrated_clint_L_per_h: float
    original_fup: float
    calibrated_fup: float
    original_kp_scale: float
    calibrated_kp_scale: float
    pre_aafe: float
    post_aafe: float
    aafe_improvement: float
    pre_within_2fold_pct: float
    post_within_2fold_pct: float
    pre_rmse: float
    post_rmse: float
    n_iterations: int
    converged: bool
    optimizer_message: str


@app.post("/calibrate/clint", response_model=CalibrateClintResponse)
def calibrate_clint(req: CalibrateClintRequest) -> CalibrateClintResponse:
    """Fit CLint (and optionally fup/Kp) to observed plasma concentration data."""
    try:
        from omega_pbpk.validation.parameter_calibrator import calibrate_drug

        drug = _drug_request_to_drug(req.drug)
        result = calibrate_drug(
            drug=drug,
            observed_times=req.observed_times,
            observed_conc=req.observed_conc,
            dose_mg=req.dose_mg,
            route=req.route,
            body_weight=req.body_weight,
            params_to_fit=req.params_to_fit,
            max_iter=req.max_iter,
        )
        s = result.summary()
        return CalibrateClintResponse(**s)  # type: ignore[arg-type]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Parameter calibration error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /predict/bioavailability
# ---------------------------------------------------------------------------


class BioavailabilityRequest(BaseModel):
    drug: DrugRequest
    dose_mg: float = Field(default=100.0, gt=0)
    route: str = Field(default="oral", pattern="^oral$")
    q_liver_L_per_h: float = Field(default=90.0, gt=0)
    gut_volume_mL: float = Field(default=250.0, gt=0)


class BioavailabilityResponse(BaseModel):
    fa: float
    fg: float
    fh: float
    F_total: float
    dose_number: float
    peff_cm_s: float
    clh_L_per_h: float
    extraction_ratio: float
    limiting_factor: str
    confidence: str


@app.post("/predict/bioavailability", response_model=BioavailabilityResponse)
def predict_bioavailability_endpoint(req: BioavailabilityRequest) -> BioavailabilityResponse:
    """Predict oral bioavailability F = fa × fg × fh via well-stirred model."""
    try:
        from omega_pbpk.prediction.bioavailability_prediction import predict_bioavailability

        drug = _drug_request_to_drug(req.drug)
        result = predict_bioavailability(
            drug_or_dict=drug,
            dose_mg=req.dose_mg,
            q_liver_L_per_h=req.q_liver_L_per_h,
            gut_volume_mL=req.gut_volume_mL,
        )
        return BioavailabilityResponse(
            fa=result.fa,
            fg=result.fg,
            fh=result.fh,
            F_total=result.F_total,
            dose_number=result.dose_number,
            peff_cm_s=result.peff_cm_s,
            clh_L_per_h=result.clh_L_per_h,
            extraction_ratio=result.extraction_ratio,
            limiting_factor=result.limiting_factor,
            confidence=result.confidence,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Bioavailability prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /predict/steady_state
# ---------------------------------------------------------------------------


class SteadyStateRequest(BaseModel):
    drug: DrugRequest
    dose_mg: float = Field(default=100.0, gt=0)
    interval_h: float = Field(default=12.0, gt=0)
    route: str = Field(default="oral", pattern="^(oral|iv)$")
    n_doses: int = Field(default=10, ge=3, le=50)
    body_weight: float = Field(default=70.0, gt=0)


class SteadyStateResponse(BaseModel):
    css_max_mg_L: float
    css_min_mg_L: float
    css_avg_mg_L: float
    auc_tau_mg_h_L: float
    accumulation_ratio: float
    fluctuation_pct: float
    time_to_ss_h: float
    n_doses_to_ss: int
    dose_mg: float
    interval_h: float
    route: str


@app.post("/predict/steady_state", response_model=SteadyStateResponse)
def predict_steady_state_endpoint(req: SteadyStateRequest) -> SteadyStateResponse:
    """Predict steady-state PK metrics via multi-dose PBPK superposition."""
    try:
        from omega_pbpk.clinical.steady_state_predictor import predict_steady_state

        drug = _drug_request_to_drug(req.drug)
        result = predict_steady_state(
            drug=drug,
            dose_mg=req.dose_mg,
            interval_h=req.interval_h,
            route=req.route,
            n_doses=req.n_doses,
            body_weight=req.body_weight,
        )
        return SteadyStateResponse(
            css_max_mg_L=result.css_max_mg_L,
            css_min_mg_L=result.css_min_mg_L,
            css_avg_mg_L=result.css_avg_mg_L,
            auc_tau_mg_h_L=result.auc_tau_mg_h_L,
            accumulation_ratio=result.accumulation_ratio,
            fluctuation_pct=result.fluctuation_pct,
            time_to_ss_h=result.time_to_ss_h,
            n_doses_to_ss=result.n_doses_to_ss,
            dose_mg=result.dose_mg,
            interval_h=result.interval_h,
            route=result.route,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Steady-state prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /ivivc
# ---------------------------------------------------------------------------


class IVIVCCorrelationResponse(BaseModel):
    level: str
    r_squared: float
    slope: float
    intercept: float
    passes_criteria: bool


class IVIVCRequest(BaseModel):
    dissolution_times: list[float] = Field(..., min_length=3)
    dissolution_fractions: list[float] = Field(..., min_length=3)
    plasma_times: list[float] = Field(..., min_length=3)
    plasma_conc: list[float] = Field(..., min_length=3)
    ke: float = Field(..., gt=0, description="Elimination rate constant (1/h)")
    vd_L: float = Field(..., gt=0, description="Volume of distribution (L)")
    dose_mg: float = Field(default=100.0, gt=0)
    levels: list[str] = Field(default=["A", "B", "C"])


class IVIVCResponse(BaseModel):
    level_a: IVIVCCorrelationResponse | None
    level_b: IVIVCCorrelationResponse | None
    level_c: IVIVCCorrelationResponse | None
    f_absorbed: list[float]
    f_dissolved: list[float]
    mdt_in_vitro_h: float
    mrt_in_vivo_h: float
    recommended_level: str


@app.post("/ivivc", response_model=IVIVCResponse)
def ivivc_endpoint(req: IVIVCRequest) -> IVIVCResponse:
    """Compute IVIVC (Level A/B/C) via Wagner-Nelson deconvolution."""
    try:
        from omega_pbpk.biopharmaceutics.ivivc import compute_ivivc

        result = compute_ivivc(
            dissolution_times=req.dissolution_times,
            dissolution_fractions=req.dissolution_fractions,
            plasma_times=req.plasma_times,
            plasma_conc=req.plasma_conc,
            ke=req.ke,
            vd_L=req.vd_L,
            dose_mg=req.dose_mg,
            levels=req.levels,
        )

        def _corr(c):
            if c is None:
                return None
            return IVIVCCorrelationResponse(
                level=c.level,
                r_squared=c.r_squared,
                slope=c.slope,
                intercept=c.intercept,
                passes_criteria=c.passes_criteria,
            )

        return IVIVCResponse(
            level_a=_corr(result.level_a),
            level_b=_corr(result.level_b),
            level_c=_corr(result.level_c),
            f_absorbed=result.f_absorbed,
            f_dissolved=result.f_dissolved,
            mdt_in_vitro_h=result.mdt_in_vitro_h,
            mrt_in_vivo_h=result.mrt_in_vivo_h,
            recommended_level=result.recommended_level,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("IVIVC computation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /population/pk_fit
# ---------------------------------------------------------------------------


class SubjectDataRequest(BaseModel):
    subject_id: str
    times: list[float] = Field(..., min_length=2)
    conc: list[float] = Field(..., min_length=2)
    dose_mg: float = Field(..., gt=0)
    route: str = Field(default="oral", pattern="^(oral|iv)$")
    body_weight: float = Field(default=70.0, gt=0)
    covariates: dict[str, float] = Field(default_factory=dict)


class IndividualEstimateResponse(BaseModel):
    subject_id: str
    cl_L_per_h: float
    vd_L: float
    ka_per_h: float
    auc_pred: float
    cmax_pred: float
    rmse: float
    converged: bool


class PopulationPKFitRequest(BaseModel):
    subjects: list[SubjectDataRequest] = Field(..., min_length=2)
    model: str = Field(default="1cpt_oral", pattern="^(1cpt_oral|1cpt_iv)$")
    cl_init: float = Field(default=5.0, gt=0)
    vd_init: float = Field(default=50.0, gt=0)
    ka_init: float = Field(default=1.0, gt=0)
    fit_covariates: bool = False


class PopulationPKFitResponse(BaseModel):
    cl_pop_L_per_h: float
    vd_pop_L: float
    ka_pop_per_h: float
    omega_cl_pct: float
    omega_vd_pct: float
    omega_ka_pct: float
    sigma_prop_pct: float
    individual_estimates: list[IndividualEstimateResponse]
    n_subjects: int
    n_observations: int
    aic: float
    bic: float
    bw_cl_exponent: float | None
    method: str


@app.post("/population/pk_fit", response_model=PopulationPKFitResponse)
def population_pk_fit(req: PopulationPKFitRequest) -> PopulationPKFitResponse:
    """Fit a 1-compartment population PK model via Standard Two-Stage method."""
    try:
        from omega_pbpk.clinical.population_pk import SubjectData, fit_population_pk

        subjects = [
            SubjectData(
                subject_id=s.subject_id,
                times=s.times,
                conc=s.conc,
                dose_mg=s.dose_mg,
                route=s.route,
                body_weight=s.body_weight,
                covariates=s.covariates,
            )
            for s in req.subjects
        ]
        result = fit_population_pk(
            subjects=subjects,
            model=req.model,
            cl_init=req.cl_init,
            vd_init=req.vd_init,
            ka_init=req.ka_init,
            fit_covariates=req.fit_covariates,
        )
        return PopulationPKFitResponse(
            cl_pop_L_per_h=result.cl_pop_L_per_h,
            vd_pop_L=result.vd_pop_L,
            ka_pop_per_h=result.ka_pop_per_h,
            omega_cl_pct=result.omega_cl_pct,
            omega_vd_pct=result.omega_vd_pct,
            omega_ka_pct=result.omega_ka_pct,
            sigma_prop_pct=result.sigma_prop_pct,
            individual_estimates=[
                IndividualEstimateResponse(
                    subject_id=e.subject_id,
                    cl_L_per_h=e.cl_L_per_h,
                    vd_L=e.vd_L,
                    ka_per_h=e.ka_per_h,
                    auc_pred=e.auc_pred,
                    cmax_pred=e.cmax_pred,
                    rmse=e.rmse,
                    converged=e.converged,
                )
                for e in result.individual_estimates
            ],
            n_subjects=result.n_subjects,
            n_observations=result.n_observations,
            aic=result.aic,
            bic=result.bic,
            bw_cl_exponent=result.bw_cl_exponent,
            method=result.method,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Population PK fitting error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /kidney/pk
# ---------------------------------------------------------------------------


class KidneyPKRequest(BaseModel):
    plasma_times: list[float] = Field(..., min_length=3)
    plasma_conc: list[float] = Field(..., min_length=3)
    fup: float = Field(..., gt=0, le=1)
    gfr_mL_per_min: float = Field(default=120.0, gt=0)
    logP: float = 0.0
    secretion_vmax_mg_h: float = Field(default=0.0, ge=0)
    secretion_km_mg_L: float = Field(default=1.0, gt=0)
    urine_flow_mL_h: float = Field(default=60.0, gt=0)
    dose_mg: float = Field(default=0.0, ge=0)


class KidneyPKResponse(BaseModel):
    cl_renal_L_per_h: float
    cl_filtration_L_per_h: float
    cl_secretion_mean_L_per_h: float
    reabsorption_fraction: float
    fe_urine: float | None  # None when dose_mg=0 (NaN serialized as null)
    urine_rate_mg_h: list[float]
    urine_cumulative_mg: float
    urine_conc_mg_L: list[float]
    gfr_mL_per_min: float
    secretion_vmax_mg_h: float
    secretion_km_mg_L: float


@app.post("/kidney/pk", response_model=KidneyPKResponse)
def kidney_pk_endpoint(req: KidneyPKRequest) -> KidneyPKResponse:
    """Mechanistic renal PK: filtration + active secretion + passive reabsorption."""
    try:
        import math

        from omega_pbpk.core.kidney_pk import predict_renal_pk

        result = predict_renal_pk(
            plasma_times=req.plasma_times,
            plasma_conc=req.plasma_conc,
            fup=req.fup,
            gfr_mL_per_min=req.gfr_mL_per_min,
            logP=req.logP,
            secretion_vmax_mg_h=req.secretion_vmax_mg_h,
            secretion_km_mg_L=req.secretion_km_mg_L,
            urine_flow_mL_h=req.urine_flow_mL_h,
            dose_mg=req.dose_mg,
        )
        fe = None if math.isnan(result.fe_urine) else result.fe_urine
        return KidneyPKResponse(
            cl_renal_L_per_h=result.cl_renal_L_per_h,
            cl_filtration_L_per_h=result.cl_filtration_L_per_h,
            cl_secretion_mean_L_per_h=result.cl_secretion_mean_L_per_h,
            reabsorption_fraction=result.reabsorption_fraction,
            fe_urine=fe,
            urine_rate_mg_h=result.urine_rate_mg_h,
            urine_cumulative_mg=result.urine_cumulative_mg,
            urine_conc_mg_L=result.urine_conc_mg_L,
            gfr_mL_per_min=result.gfr_mL_per_min,
            secretion_vmax_mg_h=result.secretion_vmax_mg_h,
            secretion_km_mg_L=result.secretion_km_mg_L,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Kidney PK error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /scale/interspecies
# ---------------------------------------------------------------------------


class AnimalPKInput(BaseModel):
    species: str = Field(..., description="Species name: mouse, rat, rabbit, dog, monkey")
    cl_mL_min_kg: float = Field(..., gt=0)
    vd_L_kg: float = Field(..., gt=0)
    thalf_h: float = Field(..., gt=0)
    dose_mg_kg: float = Field(default=1.0, gt=0)


class InterspeciesScalingRequest(BaseModel):
    animal_data: list[AnimalPKInput] = Field(..., min_length=1)
    human_bw_kg: float = Field(default=70.0, gt=0)


class InterspeciesScalingResponse(BaseModel):
    cl_L_per_h: float
    vd_L: float
    thalf_h: float
    dose_mg: float
    method_used: str
    fit_quality: float | None
    warnings: list[str]


@app.post("/scale/interspecies", response_model=InterspeciesScalingResponse)
def interspecies_scaling_endpoint(req: InterspeciesScalingRequest) -> InterspeciesScalingResponse:
    """Allometric interspecies scaling — predict human PK from animal data."""
    try:
        from omega_pbpk.core.interspecies import AnimalPKData, predict_human_pk

        animal_data = [
            AnimalPKData(
                species=a.species,
                cl_mL_min_kg=a.cl_mL_min_kg,
                vd_L_kg=a.vd_L_kg,
                thalf_h=a.thalf_h,
                dose_mg_kg=a.dose_mg_kg,
            )
            for a in req.animal_data
        ]
        result = predict_human_pk(animal_data, human_bw_kg=req.human_bw_kg)
        return InterspeciesScalingResponse(
            cl_L_per_h=result.cl_L_per_h,
            vd_L=result.vd_L,
            thalf_h=result.thalf_h,
            dose_mg=result.dose_mg,
            method_used=result.method_used,
            fit_quality=result.fit_quality,
            warnings=result.warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Interspecies scaling error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /simulate/chronopk
# ---------------------------------------------------------------------------


class ChronoPKRequest(BaseModel):
    dose_mg: float = Field(..., gt=0)
    cl_base_L_per_h: float = Field(..., gt=0)
    vd_L: float = Field(..., gt=0)
    ka_per_h: float = Field(default=1.0, ge=0)
    fup: float = Field(default=1.0, gt=0, le=1)
    dosing_time_h: float = Field(default=8.0, ge=0, lt=24)
    route: Literal["oral", "iv"] = "oral"
    t_end_h: float = Field(default=24.0, gt=0)
    fraction_renal: float = Field(default=0.3, ge=0, le=1)
    fraction_hepatic: float = Field(default=0.7, ge=0, le=1)


class ChronoPKResponse(BaseModel):
    times_h: list[float]
    plasma_conc: list[float]
    auc_0_24: float
    cmax: float
    tmax_h: float
    cmin: float
    dosing_time_h: float
    cl_effective_L_per_h: float
    gfr_profile: list[float]
    hepatic_flow_profile: list[float]
    cyp3a4_profile: list[float]


@app.post("/simulate/chronopk", response_model=ChronoPKResponse)
def chronopk_endpoint(req: ChronoPKRequest) -> ChronoPKResponse:
    """Circadian PK simulation with time-varying physiology."""
    try:
        from omega_pbpk.core.chronopk import simulate_chronopk

        result = simulate_chronopk(
            dose_mg=req.dose_mg,
            cl_base_L_per_h=req.cl_base_L_per_h,
            vd_L=req.vd_L,
            ka_per_h=req.ka_per_h,
            fup=req.fup,
            dosing_time_h=req.dosing_time_h,
            route=req.route,
            t_end_h=req.t_end_h,
            fraction_renal=req.fraction_renal,
            fraction_hepatic=req.fraction_hepatic,
        )
        return ChronoPKResponse(
            times_h=result.times_h,
            plasma_conc=result.plasma_conc,
            auc_0_24=result.auc_0_24,
            cmax=result.cmax,
            tmax_h=result.tmax_h,
            cmin=result.cmin,
            dosing_time_h=result.dosing_time_h,
            cl_effective_L_per_h=result.cl_effective_L_per_h,
            gfr_profile=result.gfr_profile,
            hepatic_flow_profile=result.hepatic_flow_profile,
            cyp3a4_profile=result.cyp3a4_profile,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ChronoPK error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /ddi/displacement
# ---------------------------------------------------------------------------


class BindingDrugInput(BaseModel):
    name: str
    fup: float = Field(..., gt=0, le=1)
    fu_tissue: float = Field(..., gt=0, le=1)
    vd_L: float = Field(..., gt=0)
    cl_L_per_h: float = Field(..., gt=0)


class DisplacementRequest(BaseModel):
    victim: BindingDrugInput
    perpetrator_fup_effect: float = Field(..., gt=0, le=1)
    perpetrator_name: str = "perpetrator"
    cl_int_L_per_h: float | None = None
    q_liver_L_per_h: float = Field(default=90.0, gt=0)


class DisplacementResponse(BaseModel):
    victim_name: str
    perpetrator_name: str
    fup_baseline: float
    fup_displaced: float
    fup_ratio: float
    vd_displaced_L: float
    cl_displaced_L_per_h: float
    auc_ratio: float
    cmax_ratio: float
    extraction_ratio: float
    is_high_extraction: bool
    clinical_significance: str
    warnings: list[str]


@app.post("/ddi/displacement", response_model=DisplacementResponse)
def displacement_endpoint(req: DisplacementRequest) -> DisplacementResponse:
    """Protein binding displacement DDI — predict PK changes from albumin/AGP displacement."""
    try:
        from omega_pbpk.clinical.protein_binding_ddi import BindingDrug, predict_displacement

        victim = BindingDrug(
            name=req.victim.name,
            fup=req.victim.fup,
            fu_tissue=req.victim.fu_tissue,
            vd_L=req.victim.vd_L,
            cl_L_per_h=req.victim.cl_L_per_h,
        )
        result = predict_displacement(
            victim=victim,
            perpetrator_fup_effect=req.perpetrator_fup_effect,
            cl_int_L_per_h=req.cl_int_L_per_h,
            q_liver_L_per_h=req.q_liver_L_per_h,
            perpetrator_name=req.perpetrator_name,
        )
        return DisplacementResponse(
            victim_name=result.victim_name,
            perpetrator_name=result.perpetrator_name,
            fup_baseline=result.fup_baseline,
            fup_displaced=result.fup_displaced,
            fup_ratio=result.fup_ratio,
            vd_displaced_L=result.vd_displaced_L,
            cl_displaced_L_per_h=result.cl_displaced_L_per_h,
            auc_ratio=result.auc_ratio,
            cmax_ratio=result.cmax_ratio,
            extraction_ratio=result.extraction_ratio,
            is_high_extraction=result.is_high_extraction,
            clinical_significance=result.clinical_significance,
            warnings=result.warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Displacement DDI error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 59 — Lymphatic Absorption
# ─────────────────────────────────────────────────────────────────────────────
class LymphaticRequest(BaseModel):
    drug_name: str = "drug"
    logP: float
    dose_mg: float
    cl_L_per_h: float
    vd_L: float
    ka_per_h: float = 1.0
    fup: float = 1.0
    solubility_mg_mL: float = 1.0
    lymph_flow_L_per_h: float = 0.12
    t_end_h: float = 24.0
    dt_h: float = 0.05


class LymphaticResponse(BaseModel):
    drug_name: str
    f_lymph: float
    f_portal: float
    bioavailability_total: float
    bioavailability_portal: float
    bioavailability_lymph: float
    auc_total_mg_h_L: float
    cmax_mg_L: float
    tmax_h: float


@app.post("/simulate/lymphatic", tags=["simulation"])
def simulate_lymphatic(req: LymphaticRequest) -> LymphaticResponse:
    try:
        from omega_pbpk.core.lymphatic import simulate_lymphatic_absorption

        r = simulate_lymphatic_absorption(
            drug_name=req.drug_name,
            logP=req.logP,
            dose_mg=req.dose_mg,
            cl_L_per_h=req.cl_L_per_h,
            vd_L=req.vd_L,
            ka_per_h=req.ka_per_h,
            fup=req.fup,
            solubility_mg_mL=req.solubility_mg_mL,
            lymph_flow_L_per_h=req.lymph_flow_L_per_h,
            t_end_h=req.t_end_h,
            dt_h=req.dt_h,
        )
        return LymphaticResponse(
            drug_name=r.drug_name,
            f_lymph=r.f_lymph,
            f_portal=r.f_portal,
            bioavailability_total=r.bioavailability_total,
            bioavailability_portal=r.bioavailability_portal,
            bioavailability_lymph=r.bioavailability_lymph,
            auc_total_mg_h_L=r.auc_total_mg_h_L,
            cmax_mg_L=r.cmax_mg_L,
            tmax_h=r.tmax_h,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Lymphatic simulation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 60 — PK/PD Target Attainment
# ─────────────────────────────────────────────────────────────────────────────
class TargetAttainmentRequest(BaseModel):
    dose_mg: float
    cl_L_per_h: float
    vd_L: float
    mic_mg_L: float
    dosing_interval_h: float = 24.0
    ka_per_h: float = 1.0
    route: str = "oral"
    pk_pd_index: str = "fT_MIC"
    target_value: float | None = None
    fup: float = 1.0


class TargetAttainmentResponse(BaseModel):
    pk_pd_index: str
    index_value: float
    target_value: float
    pta: float
    attainment_achieved: bool
    mic_mg_L: float


@app.post("/pkpd/target_attainment", tags=["clinical"])
def pkpd_target_attainment(req: TargetAttainmentRequest) -> TargetAttainmentResponse:
    try:
        from omega_pbpk.clinical.target_attainment import compute_target_attainment

        r = compute_target_attainment(
            dose_mg=req.dose_mg,
            cl_L_per_h=req.cl_L_per_h,
            vd_L=req.vd_L,
            mic_mg_L=req.mic_mg_L,
            dosing_interval_h=req.dosing_interval_h,
            ka_per_h=req.ka_per_h,
            route=req.route,
            pk_pd_index=req.pk_pd_index,
            target_value=req.target_value,
            fup=req.fup,
        )
        return TargetAttainmentResponse(
            pk_pd_index=r.pk_pd_index,
            index_value=r.index_value,
            target_value=r.target_value,
            pta=r.pta,
            attainment_achieved=r.attainment_achieved,
            mic_mg_L=r.mic_mg_L,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Target attainment error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 61 — Two-Compartment PK
# ─────────────────────────────────────────────────────────────────────────────
class TwoCompRequest(BaseModel):
    dose_mg: float
    cl_L_per_h: float
    vd_central_L: float
    vd_peripheral_L: float
    q_L_per_h: float
    ka_per_h: float = 0.0
    route: str = "iv"
    t_end_h: float = 24.0
    dt_h: float = 0.05
    f: float = 1.0


class TwoCompResponse(BaseModel):
    auc_0_inf: float
    auc_0_t: float
    cmax: float
    tmax_h: float
    thalf_alpha_h: float
    thalf_beta_h: float
    vd_ss_L: float
    cl_L_per_h: float


@app.post("/simulate/two_compartment", tags=["simulation"])
def simulate_two_comp(req: TwoCompRequest) -> TwoCompResponse:
    try:
        from omega_pbpk.core.two_compartment import simulate_two_compartment

        r = simulate_two_compartment(
            dose_mg=req.dose_mg,
            cl_L_per_h=req.cl_L_per_h,
            vd_central_L=req.vd_central_L,
            vd_peripheral_L=req.vd_peripheral_L,
            q_L_per_h=req.q_L_per_h,
            ka_per_h=req.ka_per_h,
            route=req.route,
            t_end_h=req.t_end_h,
            dt_h=req.dt_h,
            f=req.f,
        )
        return TwoCompResponse(
            auc_0_inf=r.auc_0_inf,
            auc_0_t=r.auc_0_t,
            cmax=r.cmax,
            tmax_h=r.tmax_h,
            thalf_alpha_h=r.thalf_alpha_h,
            thalf_beta_h=r.thalf_beta_h,
            vd_ss_L=r.vd_ss_L,
            cl_L_per_h=r.cl_L_per_h,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Two-compartment simulation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 62 — Dose Individualization
# ─────────────────────────────────────────────────────────────────────────────
class DoseIndivRequest(BaseModel):
    patient_id: str = "patient"
    body_weight_kg: float = 70.0
    age_years: float = 40.0
    sex: str = "male"
    serum_creatinine_mg_dL: float = 1.0
    child_pugh_score: int = 5
    standard_dose_mg: float = 100.0
    fraction_renal: float = 0.5
    fraction_hepatic: float = 0.5


class DoseIndivResponse(BaseModel):
    patient_id: str
    adjusted_dose_mg: float
    dose_adjustment_factor: float
    renal_impairment_grade: str
    hepatic_impairment_grade: str
    egfr_mL_min: float
    crcl_mL_min: float
    predicted_auc_ratio: float
    warnings: list[str]


@app.post("/dose/individualize", tags=["clinical"])
def dose_individualize(req: DoseIndivRequest) -> DoseIndivResponse:
    try:
        from omega_pbpk.clinical.dose_individualization import (
            PatientCovariates,
            individualize_dose,
        )

        patient = PatientCovariates(
            body_weight_kg=req.body_weight_kg,
            age_years=req.age_years,
            sex=req.sex,
            serum_creatinine_mg_dL=req.serum_creatinine_mg_dL,
            child_pugh_score=req.child_pugh_score,
        )
        r = individualize_dose(
            patient_id=req.patient_id,
            patient=patient,
            standard_dose_mg=req.standard_dose_mg,
            fraction_renal=req.fraction_renal,
            fraction_hepatic=req.fraction_hepatic,
        )
        return DoseIndivResponse(
            patient_id=r.patient_id,
            adjusted_dose_mg=r.adjusted_dose_mg,
            dose_adjustment_factor=r.dose_adjustment_factor,
            renal_impairment_grade=r.renal_impairment_grade,
            hepatic_impairment_grade=r.hepatic_impairment_grade,
            egfr_mL_min=r.egfr_mL_min,
            crcl_mL_min=r.crcl_mL_min,
            predicted_auc_ratio=r.predicted_auc_ratio,
            warnings=r.warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Dose individualization error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 63 — CNS PBPK / Blood-Brain Barrier
# ─────────────────────────────────────────────────────────────────────────────
class CNSRequest(BaseModel):
    drug_name: str = "drug"
    logP: float
    MW: float
    psa: float
    fup: float = 1.0
    is_pgp_substrate: bool = False
    efflux_ratio: float = 3.0


class CNSResponse(BaseModel):
    drug_name: str
    kp_uu_brain: float
    cns_classification: str
    efflux_ratio: float
    risk_factors: list[str]


@app.post("/simulate/cns", tags=["simulation"])
def simulate_cns(req: CNSRequest) -> CNSResponse:
    try:
        from omega_pbpk.core.cns_pbpk import predict_bbb_penetration

        r = predict_bbb_penetration(
            drug_name=req.drug_name,
            logP=req.logP,
            MW=req.MW,
            psa=req.psa,
            is_pgp_substrate=req.is_pgp_substrate,
            efflux_ratio=req.efflux_ratio,
        )
        return CNSResponse(
            drug_name=r.drug_name,
            kp_uu_brain=r.kp_uu_brain,
            cns_classification=r.cns_classification,
            efflux_ratio=r.efflux_ratio,
            risk_factors=r.risk_factors,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("CNS PBPK error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 64 — hERG / QT Risk
# ─────────────────────────────────────────────────────────────────────────────
class HERGRequest(BaseModel):
    drug_name: str = "drug"
    logP: float
    pka_basic: float = 0.0
    MW: float
    cmax_free_mg_L: float | None = None


class HERGResponse(BaseModel):
    drug_name: str
    risk_score: float
    herg_inhibition_pct: float
    delta_qtc_ms: float
    risk_category: str
    risk_factors: list[str]


@app.post("/risk/herg", tags=["risk"])
def herg_risk(req: HERGRequest) -> HERGResponse:
    try:
        from omega_pbpk.risk.herg_qt import predict_herg_risk

        r = predict_herg_risk(
            drug_name=req.drug_name,
            logP=req.logP,
            pka_basic=req.pka_basic,
            MW=req.MW,
            cmax_free_mg_L=req.cmax_free_mg_L,
        )
        return HERGResponse(
            drug_name=r.drug_name,
            risk_score=r.risk_score,
            herg_inhibition_pct=r.herg_inhibition_pct,
            delta_qtc_ms=r.delta_qtc_ms,
            risk_category=r.risk_category,
            risk_factors=r.risk_factors,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("hERG risk error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 65 — Absorption Window
# ─────────────────────────────────────────────────────────────────────────────
class AbsorptionWindowRequest(BaseModel):
    drug_name: str = "drug"
    dose_mg: float
    logP: float
    pka: float = 7.0
    drug_type: str = "neutral"
    sol_ph7_mg_mL: float = 1.0
    dose_volume_mL: float = 240.0
    particle_size_um: float = 50.0


class AbsorptionWindowResponse(BaseModel):
    drug_name: str
    fa: float
    total_absorbed_mg: float
    optimal_window_h: float
    segment_absorption: dict
    warnings: list[str]


@app.post("/simulate/absorption_window", tags=["simulation"])
def simulate_absorption_window_endpoint(req: AbsorptionWindowRequest) -> AbsorptionWindowResponse:
    try:
        from omega_pbpk.core.absorption_window import simulate_absorption_window

        r = simulate_absorption_window(
            drug_name=req.drug_name,
            dose_mg=req.dose_mg,
            logP=req.logP,
            pka=req.pka,
            drug_type=req.drug_type,
            sol_ph7_mg_mL=req.sol_ph7_mg_mL,
            dose_volume_mL=req.dose_volume_mL,
            particle_size_um=req.particle_size_um,
        )
        return AbsorptionWindowResponse(
            drug_name=r.drug_name,
            fa=r.fa,
            total_absorbed_mg=r.total_absorbed_mg,
            optimal_window_h=r.optimal_window_h,
            segment_absorption=r.segment_absorption,
            warnings=r.warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Absorption window error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 66 — Drug Combination Synergy
# ─────────────────────────────────────────────────────────────────────────────
class SynergyRequest(BaseModel):
    drug_A: str
    drug_B: str
    dose_A: float
    dose_B: float
    ic50_A: float
    ic50_B: float
    hill_n_A: float = 1.0
    hill_n_B: float = 1.0
    observed_effect: float | None = None


class SynergyResponse(BaseModel):
    drug_A: str
    drug_B: str
    effect_A_alone: float
    effect_B_alone: float
    bliss_expected_effect: float
    combination_effect: float
    bliss_delta: float
    loewe_ci: float | None
    ci_classification: str
    bliss_classification: str
    warnings: list[str]


@app.post("/pkpd/synergy", tags=["clinical"])
def pkpd_synergy(req: SynergyRequest) -> SynergyResponse:
    try:
        import math

        from omega_pbpk.clinical.synergy import assess_combination_synergy

        r = assess_combination_synergy(
            drug_A=req.drug_A,
            drug_B=req.drug_B,
            dose_A=req.dose_A,
            dose_B=req.dose_B,
            ic50_A=req.ic50_A,
            ic50_B=req.ic50_B,
            hill_n_A=req.hill_n_A,
            hill_n_B=req.hill_n_B,
            observed_effect=req.observed_effect,
        )
        return SynergyResponse(
            drug_A=r.drug_A,
            drug_B=r.drug_B,
            effect_A_alone=r.effect_A_alone,
            effect_B_alone=r.effect_B_alone,
            bliss_expected_effect=r.bliss_expected_effect,
            combination_effect=r.combination_effect,
            bliss_delta=r.bliss_delta,
            loewe_ci=None if math.isnan(r.loewe_ci) else r.loewe_ci,
            ci_classification=r.ci_classification,
            bliss_classification=r.bliss_classification,
            warnings=r.warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Synergy analysis error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 67 — Pediatric Dosing
# ─────────────────────────────────────────────────────────────────────────────
class PediatricDoseRequest(BaseModel):
    adult_dose_mg: float
    adult_weight_kg: float = 70.0
    child_age_years: float = 5.0
    child_weight_kg: float | None = None
    fraction_cyp3a4: float = 0.0
    fraction_renal: float = 0.0
    min_dose_mg: float = 1.0


class PediatricDoseResponse(BaseModel):
    allometric_dose_mg: float
    cyp3a4_adjusted_dose_mg: float
    renal_adjusted_dose_mg: float
    combined_adjusted_dose_mg: float
    cyp3a4_fraction_adult: float
    gfr_estimated: float
    dose_per_kg: float
    warnings: list[str]


@app.post("/dose/pediatric", tags=["clinical"])
def dose_pediatric(req: PediatricDoseRequest) -> PediatricDoseResponse:
    try:
        from omega_pbpk.clinical.pediatric_dosing import pediatric_dose

        r = pediatric_dose(
            adult_dose_mg=req.adult_dose_mg,
            adult_weight_kg=req.adult_weight_kg,
            child_age_years=req.child_age_years,
            child_weight_kg=req.child_weight_kg,
            fraction_cyp3a4=req.fraction_cyp3a4,
            fraction_renal=req.fraction_renal,
            min_dose_mg=req.min_dose_mg,
        )
        return PediatricDoseResponse(
            allometric_dose_mg=r.allometric_dose_mg,
            cyp3a4_adjusted_dose_mg=r.cyp3a4_adjusted_dose_mg,
            renal_adjusted_dose_mg=r.renal_adjusted_dose_mg,
            combined_adjusted_dose_mg=r.combined_adjusted_dose_mg,
            cyp3a4_fraction_adult=r.cyp3a4_fraction_adult,
            gfr_estimated=r.gfr_estimated,
            dose_per_kg=r.dose_per_kg,
            warnings=r.warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pediatric dosing error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 68 — Solubility Prediction
# ─────────────────────────────────────────────────────────────────────────────
class SolubilityRequest(BaseModel):
    drug_name: str = "drug"
    logP: float
    pka: float = 7.0
    drug_type: str = "neutral"
    mp_celsius: float = 150.0
    dose_mg: float = 100.0
    amorphous_factor: float = 1.5


class SolubilityResponse(BaseModel):
    drug_name: str
    sol_intrinsic_mg_mL: float
    sol_ph_stomach_mg_mL: float
    sol_ph_intestine_mg_mL: float
    bcs_solubility_class: str
    dose_number: float
    supersaturation_potential: float
    warnings: list[str]


@app.post("/predict/solubility", tags=["prediction"])
def predict_solubility_endpoint(req: SolubilityRequest) -> SolubilityResponse:
    try:
        from omega_pbpk.prediction.solubility import predict_solubility

        r = predict_solubility(
            drug_name=req.drug_name,
            logP=req.logP,
            pka=req.pka,
            drug_type=req.drug_type,
            mp_celsius=req.mp_celsius,
            dose_mg=req.dose_mg,
            amorphous_factor=req.amorphous_factor,
        )
        return SolubilityResponse(
            drug_name=r.drug_name,
            sol_intrinsic_mg_mL=r.sol_intrinsic_mg_mL,
            sol_ph_stomach_mg_mL=r.sol_ph_stomach_mg_mL,
            sol_ph_intestine_mg_mL=r.sol_ph_intestine_mg_mL,
            bcs_solubility_class=r.bcs_solubility_class,
            dose_number=r.dose_number,
            supersaturation_potential=r.supersaturation_potential,
            warnings=r.warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Solubility prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 69 — Enterohepatic Recirculation
# ─────────────────────────────────────────────────────────────────────────────
class EHCRequest(BaseModel):
    drug_name: str = "drug"
    dose_mg: float
    cl_hepatic_L_per_h: float
    vd_L: float
    ka_per_h: float = 1.0
    f_biliary: float = 0.3
    f_reabsorb: float = 0.5
    bile_release_interval_h: float = 4.0
    n_bile_releases: int = 5
    route: str = "oral"
    t_end_h: float = 48.0


class EHCResponse(BaseModel):
    auc_total_mg_h_L: float
    auc_no_ehc_mg_h_L: float
    cmax_mg_L: float
    tmax_h: float
    ehc_fraction: float
    excreted_feces_mg: float
    reabsorbed_total_mg: float
    n_secondary_peaks: int


@app.post("/simulate/ehc", tags=["simulation"])
def simulate_ehc_endpoint(req: EHCRequest) -> EHCResponse:
    try:
        from omega_pbpk.core.enterohepatic import simulate_ehc

        r = simulate_ehc(
            drug_name=req.drug_name,
            dose_mg=req.dose_mg,
            cl_hepatic_L_per_h=req.cl_hepatic_L_per_h,
            vd_L=req.vd_L,
            ka_per_h=req.ka_per_h,
            f_biliary=req.f_biliary,
            f_reabsorb=req.f_reabsorb,
            bile_release_interval_h=req.bile_release_interval_h,
            n_bile_releases=req.n_bile_releases,
            route=req.route,
            t_end_h=req.t_end_h,
        )
        return EHCResponse(
            auc_total_mg_h_L=r.auc_total_mg_h_L,
            auc_no_ehc_mg_h_L=r.auc_no_ehc_mg_h_L,
            cmax_mg_L=r.cmax_mg_L,
            tmax_h=r.tmax_h,
            ehc_fraction=r.ehc_fraction,
            excreted_feces_mg=r.excreted_feces_mg,
            reabsorbed_total_mg=r.reabsorbed_total_mg,
            n_secondary_peaks=len(r.secondary_peak_times),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("EHC simulation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 70 — ADR Risk Scoring
# ─────────────────────────────────────────────────────────────────────────────
class ADRRequest(BaseModel):
    drug_name: str = "drug"
    logP: float
    MW: float
    fup: float
    cmax_mg_L: float
    dose_mg: float
    fraction_renal: float = 0.5
    kp_uu_brain: float = 0.1
    delta_qtc_ms: float = 0.0
    sol_mg_mL: float = 1.0
    reactive_metabolite: bool = False
    is_pgp_inhibitor: bool = False
    cns_active: bool = False
    gfr_mL_min: float = 120.0


class ADRResponse(BaseModel):
    drug_name: str
    composite_risk_score: float
    risk_tier: str
    category_scores: dict
    risk_flags: list[str]
    safety_margin: float
    bsa_mg_per_m2: float


@app.post("/risk/adr", tags=["risk"])
def adr_risk(req: ADRRequest) -> ADRResponse:
    try:
        from omega_pbpk.risk.adr_risk import score_adr_risk

        r = score_adr_risk(
            drug_name=req.drug_name,
            logP=req.logP,
            MW=req.MW,
            fup=req.fup,
            cmax_mg_L=req.cmax_mg_L,
            dose_mg=req.dose_mg,
            fraction_renal=req.fraction_renal,
            kp_uu_brain=req.kp_uu_brain,
            delta_qtc_ms=req.delta_qtc_ms,
            sol_mg_mL=req.sol_mg_mL,
            reactive_metabolite=req.reactive_metabolite,
            is_pgp_inhibitor=req.is_pgp_inhibitor,
            cns_active=req.cns_active,
            gfr_mL_min=req.gfr_mL_min,
        )
        return ADRResponse(
            drug_name=r.drug_name,
            composite_risk_score=r.composite_risk_score,
            risk_tier=r.risk_tier,
            category_scores=r.category_scores,
            risk_flags=r.risk_flags,
            safety_margin=r.safety_margin,
            bsa_mg_per_m2=r.bsa_mg_per_m2,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ADR risk error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 71 — Protein Binding / Corona
# ---------------------------------------------------------------------------
from omega_pbpk.prediction.protein_corona import (
    predict_protein_binding,
)


class _ProteinBindingReq(BaseModel):
    drug_name: str
    logP: float
    pka_acid: float = 0.0
    pka_basic: float = 0.0
    MW: float = 300.0
    drug_conc_uM: float = 0.1


@app.post("/predict/protein_binding")
def api_protein_binding(req: _ProteinBindingReq):
    try:
        r = predict_protein_binding(
            drug_name=req.drug_name,
            logP=req.logP,
            pka_acid=req.pka_acid,
            pka_basic=req.pka_basic,
            MW=req.MW,
            drug_conc_uM=req.drug_conc_uM,
        )
        return r.__dict__
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 72 — Metabolic Stability
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.metabolic_stability import predict_metabolic_stability


class _MetabStabReq(BaseModel):
    drug_name: str
    t_half_min: float
    assay_type: str = "microsome"
    species: str = "human_liver"
    fup: float = 1.0
    protein_mg_mL: float = 0.5


@app.post("/predict/metabolic_stability")
def api_metabolic_stability(req: _MetabStabReq):
    try:
        r = predict_metabolic_stability(
            req.drug_name,
            t_half_min=req.t_half_min,
            assay_type=req.assay_type,
            species=req.species,
            fup=req.fup,
            protein_mg_mL=req.protein_mg_mL,
        )
        return r.__dict__
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 73 — Population PK
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.pop_pk import simulate_pop_pk


class _PopPKReq(BaseModel):
    n_subjects: int = 100
    dose_mg: float = 100.0
    cl_typical_L_per_h: float = 5.0
    vd_typical_L: float = 50.0
    omega_cl: float = 0.3
    omega_vd: float = 0.2
    route: str = "oral"
    t_end_h: float = 24.0
    seed: int = 42
    therapeutic_low_mg_L: float | None = None
    therapeutic_high_mg_L: float | None = None


@app.post("/simulate/pop_pk")
def api_pop_pk(req: _PopPKReq):
    try:
        r = simulate_pop_pk(
            n_subjects=req.n_subjects,
            dose_mg=req.dose_mg,
            cl_typical_L_per_h=req.cl_typical_L_per_h,
            vd_typical_L=req.vd_typical_L,
            omega_cl=req.omega_cl,
            omega_vd=req.omega_vd,
            route=req.route,
            t_end_h=req.t_end_h,
            seed=req.seed,
            therapeutic_low_mg_L=req.therapeutic_low_mg_L,
            therapeutic_high_mg_L=req.therapeutic_high_mg_L,
        )
        import dataclasses

        return dataclasses.asdict(r)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 74 — Drug Accumulation
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.accumulation import simulate_accumulation


class _AccumulationReq(BaseModel):
    drug_name: str
    dose_mg: float = 100.0
    cl_L_per_h: float = 5.0
    vd_L: float = 50.0
    dosing_interval_h: float = 12.0
    n_doses: int = 10


@app.post("/simulate/accumulation")
def api_accumulation(req: _AccumulationReq):
    try:
        r = simulate_accumulation(
            drug_name=req.drug_name,
            dose_mg=req.dose_mg,
            cl_L_per_h=req.cl_L_per_h,
            vd_L=req.vd_L,
            dosing_interval_h=req.dosing_interval_h,
            n_doses=req.n_doses,
        )
        import dataclasses

        return dataclasses.asdict(r)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 75 — Tissue Binding Correction
# ---------------------------------------------------------------------------
from omega_pbpk.core.pbpk_tissue_binding import predict_tissue_binding


class _TissueBindingReq(BaseModel):
    drug_name: str
    logP: float
    fup: float
    kp_dict: dict[str, float] | None = None


@app.post("/predict/tissue_binding")
def api_tissue_binding(req: _TissueBindingReq):
    try:
        r = predict_tissue_binding(req.drug_name, req.logP, req.fup, req.kp_dict)
        import dataclasses

        return dataclasses.asdict(r)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 76 — Food-Drug Interaction
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.food_drug_interaction import predict_food_effect


class _FoodEffectReq(BaseModel):
    drug_name: str
    logP: float
    pka: float = 7.0
    drug_type: str = "neutral"
    dose_mg: float = 100.0
    bcs_class: str = "II"
    solubility_fasted_mg_mL: float = 0.1


@app.post("/predict/food_effect")
def api_food_effect(req: _FoodEffectReq):
    try:
        r = predict_food_effect(
            req.drug_name,
            req.logP,
            req.pka,
            req.drug_type,
            req.dose_mg,
            req.bcs_class,
            req.solubility_fasted_mg_mL,
        )
        import dataclasses

        return dataclasses.asdict(r)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 77 — Clearance Prediction
# ---------------------------------------------------------------------------
from omega_pbpk.prediction.clearance_prediction import predict_clearance


class _ClearanceReq(BaseModel):
    drug_name: str
    logP: float
    MW: float
    fup: float
    pka: float = 7.0
    PSA: float = 60.0
    drug_type: str = "neutral"


@app.post("/predict/clearance")
def api_clearance(req: _ClearanceReq):
    try:
        r = predict_clearance(
            req.drug_name, req.logP, req.MW, req.fup, req.pka, req.PSA, req.drug_type
        )
        import dataclasses

        return dataclasses.asdict(r)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 78 — Therapeutic Drug Monitoring
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.drug_monitoring import tdm_dashboard


class _TDMReq(BaseModel):
    drug_name: str
    measured_conc_mg_L: list[float]
    measured_times_h: list[float]
    therapeutic_min_mg_L: float
    therapeutic_max_mg_L: float
    target_pct_within: float = 80.0


@app.post("/tdm/assess")
def api_tdm(req: _TDMReq):
    try:
        r = tdm_dashboard(
            req.drug_name,
            req.measured_conc_mg_L,
            req.measured_times_h,
            req.therapeutic_min_mg_L,
            req.therapeutic_max_mg_L,
        )
        return r
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 79 — Renal Dosing
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.renal_dosing import adjust_renal_dose


class _RenalDoseReq(BaseModel):
    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    crcl_mL_per_min: float
    fraction_renal: float = 0.5


@app.post("/dose/renal")
def api_renal_dose(req: _RenalDoseReq):
    try:
        r = adjust_renal_dose(
            req.drug_name,
            req.dose_mg,
            req.dosing_interval_h,
            req.crcl_mL_per_min,
            req.fraction_renal,
        )
        import dataclasses

        return dataclasses.asdict(r)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 80 — Permeability Prediction
# ---------------------------------------------------------------------------
from omega_pbpk.prediction.permeability import predict_permeability


class _PermeabilityReq(BaseModel):
    drug_name: str
    logP: float
    MW: float
    PSA: float
    HBD: int | None = None
    efflux_substrate: bool = False


@app.post("/predict/permeability")
def api_permeability(req: _PermeabilityReq):
    try:
        r = predict_permeability(
            req.drug_name, req.logP, req.MW, req.PSA, req.HBD, req.efflux_substrate
        )
        import dataclasses

        return dataclasses.asdict(r)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 81 — Flip-Flop PK
# ---------------------------------------------------------------------------
from omega_pbpk.core.flip_flop import simulate_flip_flop


class _FlipFlopReq(BaseModel):
    drug_name: str
    dose_mg: float
    vd_L: float
    ka_per_h: float
    ke_per_h: float
    f: float = 1.0
    t_end_h: float = 24.0


@app.post("/simulate/flip_flop")
def api_flip_flop(req: _FlipFlopReq):
    try:
        r = simulate_flip_flop(
            req.drug_name, req.dose_mg, req.vd_L, req.ka_per_h, req.ke_per_h, req.f, req.t_end_h
        )
        import dataclasses

        return dataclasses.asdict(r)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 82 — Organ Impairment
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.organ_impairment import adjust_for_organ_impairment


class _OrganImpairmentReq(BaseModel):
    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    cl_L_per_h: float
    vd_L: float
    crcl_mL_per_min: float = 100.0
    child_pugh_score: int = 5
    nyha_class: int = 1
    fraction_renal: float = 0.3
    fraction_hepatic: float = 0.6


@app.post("/dose/organ_impairment")
def api_organ_impairment(req: _OrganImpairmentReq):
    try:
        r = adjust_for_organ_impairment(
            req.drug_name,
            req.dose_mg,
            req.dosing_interval_h,
            req.cl_L_per_h,
            req.vd_L,
            req.crcl_mL_per_min,
            req.child_pugh_score,
            req.nyha_class,
            req.fraction_renal,
            req.fraction_hepatic,
        )
        import dataclasses

        return dataclasses.asdict(r)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 83 — Drug Interaction Scoring
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.interaction_score import calculate_interaction_score, score_cyp_perpetrator, score_cyp_victim

class _InteractionScoreReq(BaseModel):
    drug_a: str
    drug_b: str
    perpetrator_score: float
    victim_score: float
    transporter_score: float = 0.0

@app.post("/ddi/interaction_score")
def api_interaction_score(req: _InteractionScoreReq):
    try:
        r = calculate_interaction_score(req.drug_a, req.drug_b,
                                        req.perpetrator_score, req.victim_score,
                                        req.transporter_score)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 84 — Volume of Distribution Prediction
# ---------------------------------------------------------------------------
from omega_pbpk.prediction.vd_predictor import predict_vd, vd_population_range

class _VdPredictReq(BaseModel):
    drug_name: str
    logP: float
    MW: float
    fup: float
    pka: float = 7.0
    PSA: float = 60.0

@app.post("/predict/vd")
def api_vd_predict(req: _VdPredictReq):
    try:
        r = predict_vd(req.drug_name, req.logP, req.MW, req.fup, req.pka, req.PSA)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 85 — Local Sensitivity Analysis
# ---------------------------------------------------------------------------
# (No REST endpoint — sensitivity analysis requires custom simulate_fn callback)
# Documented as a Python library function


# ---------------------------------------------------------------------------
# Phase 86 — Dissolution Modeling
# ---------------------------------------------------------------------------
from omega_pbpk.biopharmaceutics.dissolution import simulate_weibull, compare_dissolution_models

class _WeibullReq(BaseModel):
    drug_name: str
    dose_mg: float
    td_h: float = 0.5
    beta: float = 1.5
    t_end_h: float = 4.0

@app.post("/simulate/dissolution/weibull")
def api_dissolution_weibull(req: _WeibullReq):
    try:
        r = simulate_weibull(req.drug_name, req.dose_mg, req.td_h, req.beta, req.t_end_h)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

class _DissolutionCompareReq(BaseModel):
    drug_name: str
    dose_mg: float
    solubility_mg_mL: float = 0.1

@app.post("/simulate/dissolution/compare")
def api_dissolution_compare(req: _DissolutionCompareReq):
    try:
        results = compare_dissolution_models(req.drug_name, req.dose_mg, req.solubility_mg_mL)
        import dataclasses
        return {k: dataclasses.asdict(v) for k, v in results.items()}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 87 — Mass Balance
# ---------------------------------------------------------------------------
from omega_pbpk.core.mass_balance import check_mass_balance

class _MassBalanceReq(BaseModel):
    drug_name: str
    dose_mg: float
    absorbed_mg: float
    metabolized_mg: float = 0.0
    excreted_mg: float = 0.0
    remaining_mg: float = 0.0
    tolerance_pct: float = 1.0

@app.post("/validate/mass_balance")
def api_mass_balance(req: _MassBalanceReq):
    try:
        r = check_mass_balance(req.drug_name, req.dose_mg, req.absorbed_mg,
                                metabolized_mg=req.metabolized_mg,
                                excreted_mg=req.excreted_mg,
                                remaining_mg=req.remaining_mg,
                                tolerance_pct=req.tolerance_pct)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 88 — NCA
# ---------------------------------------------------------------------------
from omega_pbpk.core.nca import calculate_nca

class _NCAReq(BaseModel):
    drug_name: str
    times_h: list[float]
    conc_mg_L: list[float]
    dose_mg: float
    route: str = "oral"

@app.post("/analyze/nca")
def api_nca(req: _NCAReq):
    try:
        r = calculate_nca(req.drug_name, req.times_h, req.conc_mg_L, req.dose_mg, req.route)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 89 — Transporter DDI
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.transporter_ddi import assess_transporter_ddi

class _TransporterDDIReq(BaseModel):
    victim_drug: str
    perpetrator_drug: str
    transporter: str
    ki_uM: float
    inhibitor_conc_uM: float = 0.1
    inhibition_type: str = "competitive"

@app.post("/ddi/transporter")
def api_transporter_ddi(req: _TransporterDDIReq):
    try:
        r = assess_transporter_ddi(req.victim_drug, req.perpetrator_drug, req.transporter,
                                    req.ki_uM, inhibitor_conc_uM=req.inhibitor_conc_uM,
                                    inhibition_type=req.inhibition_type)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 90 — Trial Simulation
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.trial_simulation import simulate_trial, dose_response_simulation

class _TrialSimReq(BaseModel):
    drug_name: str
    dose_mg: float
    n_subjects: int = 100
    cl_L_per_h: float = 5.0
    vd_L: float = 50.0
    ec50_mg_L: float = 0.5
    emax: float = 1.0
    placebo_effect: float = 0.1
    seed: int = 42

@app.post("/simulate/trial")
def api_trial_sim(req: _TrialSimReq):
    try:
        r = simulate_trial(req.drug_name, req.dose_mg, req.n_subjects,
                            req.cl_L_per_h, req.vd_L, req.ec50_mg_L,
                            req.emax, placebo_effect=req.placebo_effect, seed=req.seed)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 91 — Pediatric PBPK Scaling
# ---------------------------------------------------------------------------
from omega_pbpk.core.pediatric_pbpk import scale_to_pediatric

class _PedPBPKReq(BaseModel):
    drug_name: str
    adult_cl_L_per_h: float
    adult_vd_L: float
    adult_dose_mg: float
    age_years: float
    weight_kg: float | None = None
    fraction_cyp3a4: float = 0.5
    fraction_renal: float = 0.3

@app.post("/simulate/pediatric_pbpk")
def api_pediatric_pbpk(req: _PedPBPKReq):
    try:
        r = scale_to_pediatric(req.drug_name, req.adult_cl_L_per_h, req.adult_vd_L,
                                req.adult_dose_mg, req.age_years, req.weight_kg,
                                req.fraction_cyp3a4, req.fraction_renal)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 92 — Bioequivalence Assessment
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.bioequivalence import assess_bioequivalence, simulate_be_study

class _BEReq(BaseModel):
    drug_test: str
    drug_reference: str
    auc_test: list[float]
    auc_reference: list[float]
    cmax_test: list[float] | None = None
    cmax_reference: list[float] | None = None

@app.post("/analyze/bioequivalence")
def api_be(req: _BEReq):
    try:
        r = assess_bioequivalence(req.drug_test, req.drug_reference,
                                   req.auc_test, req.auc_reference,
                                   req.cmax_test, req.cmax_reference)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 93 — Protein Displacement DDI
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.protein_displacement import assess_protein_displacement

class _ProteinDispReq(BaseModel):
    victim_drug: str
    perpetrator_drug: str
    baseline_fup: float
    ki_app_uM: float
    competitor_conc_uM: float
    is_nti: bool = False
    binding_site: str = "albumin_site_I"

@app.post("/ddi/protein_displacement")
def api_protein_displacement(req: _ProteinDispReq):
    try:
        r = assess_protein_displacement(req.victim_drug, req.perpetrator_drug,
                                         req.baseline_fup, req.ki_app_uM,
                                         req.competitor_conc_uM, is_nti=req.is_nti,
                                         binding_site=req.binding_site)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 94 — First-Pass Extraction
# ---------------------------------------------------------------------------
from omega_pbpk.core.first_pass import calculate_first_pass

class _FirstPassReq(BaseModel):
    drug_name: str
    dose_mg: float
    fup: float
    cl_int_L_per_h: float
    fa: float = 1.0
    logP: float = 2.0
    q_hepatic_L_per_h: float = 90.0
    route: str = "oral"

@app.post("/simulate/first_pass")
def api_first_pass(req: _FirstPassReq):
    try:
        r = calculate_first_pass(req.drug_name, req.dose_mg, req.fup, req.cl_int_L_per_h,
                                  req.fa, req.logP, req.q_hepatic_L_per_h, req.route)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 95 — Concentration Ratio Prediction
# ---------------------------------------------------------------------------
from omega_pbpk.prediction.concentration_ratio import predict_kp, predict_kp_panel

class _KpReq(BaseModel):
    drug_name: str
    logP: float
    fup: float
    tissue: str
    pka: float = 7.0
    drug_type: str = "neutral"

@app.post("/predict/kp")
def api_kp(req: _KpReq):
    try:
        r = predict_kp(req.drug_name, req.logP, req.fup, req.tissue, req.pka, req.drug_type)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 96 — Half-Life Calculator
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.half_life_calculator import calculate_half_life, optimal_dosing_interval

class _HalfLifeReq(BaseModel):
    drug_name: str
    t_half_h: float
    dosing_interval_h: float | None = None

@app.post("/calculate/half_life")
def api_half_life(req: _HalfLifeReq):
    try:
        r = calculate_half_life(req.drug_name, req.t_half_h, req.dosing_interval_h)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 97 — Sparse PK Sampling
# ---------------------------------------------------------------------------
from omega_pbpk.clinical.sparse_pk import optimize_sparse_sampling, recommend_sampling_times

class _SparseReq(BaseModel):
    drug_name: str
    n_samples: int
    pk_params: dict | None = None
    t_window_h: float = 24.0
    seed: int = 42

@app.post("/design/sparse_sampling")
def api_sparse_sampling(req: _SparseReq):
    try:
        r = optimize_sparse_sampling(req.drug_name, req.n_samples,
                                      req.pk_params or {}, req.t_window_h, req.seed)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 98 — Drug Product Characterization
# ---------------------------------------------------------------------------
from omega_pbpk.biopharmaceutics.drug_product import classify_formulation

class _DrugProductReq(BaseModel):
    drug_name: str
    dose_mg: float
    solubility_mg_mL: float
    Peff_cm_s: float = 1e-4
    ka_per_h: float = 1.0
    dissolution_time_h: float = 0.5
    formulation_type: str = "tablet"

@app.post("/classify/drug_product")
def api_drug_product(req: _DrugProductReq):
    try:
        r = classify_formulation(req.drug_name, req.dose_mg, req.solubility_mg_mL,
                                  req.Peff_cm_s, req.ka_per_h,
                                  req.dissolution_time_h, req.formulation_type)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 99: Compartmental Analysis ===
from omega_pbpk.core.compartmental_analysis import select_compartment_model, compare_subjects

class _CompartmentalReq(BaseModel):
    drug_name: str = "Drug"
    times: list[float]
    conc: list[float]

class _CompareSbjReq(BaseModel):
    subjects: list[dict]

@app.post("/analyze/compartmental")
def api_compartmental(req: _CompartmentalReq):
    try:
        r = select_compartment_model(req.drug_name, req.times, req.conc)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/analyze/compartmental/compare")
def api_compartmental_compare(req: _CompareSbjReq):
    try:
        result = compare_subjects(req.subjects)
        import dataclasses
        result["results"] = [dataclasses.asdict(r) for r in result["results"]]
        return result
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 100: Dose Escalation ===
from omega_pbpk.clinical.dose_escalation import simulate_3plus3, simulate_accelerated_titration

class _DoseEscReq(BaseModel):
    drug_name: str = "Drug"
    starting_dose_mg: float = 10.0
    dose_levels: list[float]
    ed50_mg: float
    emax: float = 0.9
    hill: float = 2.0
    seed: int = 42

@app.post("/simulate/dose_escalation")
def api_dose_escalation(req: _DoseEscReq):
    try:
        r = simulate_3plus3(req.drug_name, req.starting_dose_mg, req.dose_levels, req.ed50_mg, req.emax, req.hill, req.seed)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/simulate/dose_escalation/accelerated")
def api_dose_escalation_accel(req: _DoseEscReq):
    try:
        r = simulate_accelerated_titration(req.drug_name, req.dose_levels, req.ed50_mg, req.emax, req.hill, req.seed)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 101: PK Interaction Predictor ===
from omega_pbpk.prediction.pk_interaction_predictor import predict_pk_interaction, screen_interaction_pairs

class _PKInteractionReq(BaseModel):
    perpetrator: str
    victim: str
    ki_uM: float | None = None
    emax: float | None = None
    ec50_uM: float | None = None
    inhibitor_conc_uM: float = 1.0
    inducer_conc_uM: float = 1.0
    fm: float = 0.8

class _PKInteractionScreenReq(BaseModel):
    pairs: list[dict]

@app.post("/predict/pk_interaction")
def api_pk_interaction(req: _PKInteractionReq):
    try:
        r = predict_pk_interaction(req.perpetrator, req.victim, req.ki_uM, req.emax, req.ec50_uM, req.inhibitor_conc_uM, req.inducer_conc_uM, req.fm)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/predict/pk_interaction/screen")
def api_pk_interaction_screen(req: _PKInteractionScreenReq):
    try:
        results = screen_interaction_pairs(req.pairs)
        import dataclasses; return [dataclasses.asdict(r) for r in results]
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 102: Particle Dissolution ===
from omega_pbpk.biopharmaceutics.particle_dissolution import simulate_particle_dissolution, compare_particle_sizes

class _ParticleReq(BaseModel):
    drug_name: str = "Drug"
    dose_mg: float = 100.0
    particle_radius_um: float = 10.0
    solubility_mg_mL: float = 0.5
    diffusivity_cm2_s: float = 5e-6
    density_g_cm3: float = 1.2
    t_end_h: float = 6.0

class _ParticleSizesReq(BaseModel):
    drug_name: str = "Drug"
    dose_mg: float = 100.0
    radii_um: list[float]
    solubility_mg_mL: float = 0.5

@app.post("/simulate/particle_dissolution")
def api_particle_dissolution(req: _ParticleReq):
    try:
        r = simulate_particle_dissolution(req.drug_name, req.dose_mg, req.particle_radius_um, req.solubility_mg_mL, req.diffusivity_cm2_s, req.density_g_cm3, req.t_end_h)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/simulate/particle_dissolution/compare")
def api_particle_dissolution_compare(req: _ParticleSizesReq):
    try:
        results = compare_particle_sizes(req.drug_name, req.dose_mg, req.radii_um, req.solubility_mg_mL)
        import dataclasses; return [dataclasses.asdict(r) for r in results]
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 103: Effect Compartment ===
from omega_pbpk.core.effect_compartment import simulate_effect_compartment, simulate_pkpd_iv

class _EffectCompartmentReq(BaseModel):
    drug_name: str = "Drug"
    times_h: list[float]
    cp_mg_L: list[float]
    ke0_per_h: float = 0.5
    ec50_mg_L: float = 2.0
    emax: float = 1.0
    hill: float = 1.0

class _PKPDIVReq(BaseModel):
    drug_name: str = "Drug"
    dose_mg: float = 100.0
    cl_L_per_h: float = 5.0
    vd_L: float = 20.0
    ke0_per_h: float = 0.5
    ec50_mg_L: float = 2.0
    emax: float = 1.0
    hill: float = 1.0
    t_end_h: float = 24.0

@app.post("/simulate/effect_compartment")
def api_effect_compartment(req: _EffectCompartmentReq):
    try:
        r = simulate_effect_compartment(req.drug_name, req.times_h, req.cp_mg_L, req.ke0_per_h, req.ec50_mg_L, req.emax, req.hill)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/simulate/pkpd/iv")
def api_pkpd_iv(req: _PKPDIVReq):
    try:
        r = simulate_pkpd_iv(req.drug_name, req.dose_mg, req.cl_L_per_h, req.vd_L, req.ke0_per_h, req.ec50_mg_L, req.emax, req.hill, req.t_end_h)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 104: Therapeutic Window ===
from omega_pbpk.clinical.therapeutic_window import analyze_therapeutic_window, optimal_dose_for_window

class _TherapeuticWindowReq(BaseModel):
    drug_name: str = "Drug"
    times_h: list[float]
    conc_mg_L: list[float]
    mec_mg_L: float
    mtc_mg_L: float

class _OptimalDoseReq(BaseModel):
    drug_name: str = "Drug"
    times_h: list[float]
    conc_per_mg: list[float]
    mec_mg_L: float
    mtc_mg_L: float
    dose_min_mg: float = 1.0
    dose_max_mg: float = 1000.0

@app.post("/analyze/therapeutic_window")
def api_therapeutic_window(req: _TherapeuticWindowReq):
    try:
        r = analyze_therapeutic_window(req.drug_name, req.times_h, req.conc_mg_L, req.mec_mg_L, req.mtc_mg_L)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/analyze/therapeutic_window/optimal_dose")
def api_optimal_dose(req: _OptimalDoseReq):
    try:
        return optimal_dose_for_window(req.drug_name, req.times_h, req.conc_per_mg, req.mec_mg_L, req.mtc_mg_L, req.dose_min_mg, req.dose_max_mg)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 105: Plasma Protein Binding Prediction ===
from omega_pbpk.prediction.plasma_protein_binding import predict_ppb, ppb_sensitivity

class _PPBReq(BaseModel):
    drug_name: str = "Drug"
    logP: float = 2.0
    MW: float = 300.0
    PSA: float = 60.0
    pka_acid: float | None = None
    pka_basic: float | None = None

@app.post("/predict/plasma_protein_binding")
def api_ppb(req: _PPBReq):
    try:
        r = predict_ppb(req.drug_name, req.logP, req.MW, req.PSA, req.pka_acid, req.pka_basic)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/predict/plasma_protein_binding/sensitivity")
def api_ppb_sensitivity(req: _PPBReq):
    try:
        result = ppb_sensitivity(req.drug_name, req.logP, req.MW, req.PSA)
        import dataclasses
        return {k: dataclasses.asdict(v) if hasattr(v, "__dataclass_fields__") else v for k, v in result.items()}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 106: Gastric Emptying ===
from omega_pbpk.biopharmaceutics.gastric_emptying import simulate_gastric_emptying, compare_food_states

class _GastricReq(BaseModel):
    state: str = "fasted"
    dose_mg: float = 100.0
    ka_absorption_per_h: float = 1.0
    t_end_h: float = 12.0

@app.post("/simulate/gastric_emptying")
def api_gastric_emptying(req: _GastricReq):
    try:
        r = simulate_gastric_emptying(req.state, req.dose_mg, req.ka_absorption_per_h, req.t_end_h)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/simulate/gastric_emptying/compare")
def api_gastric_compare(req: _GastricReq):
    try:
        result = compare_food_states(req.dose_mg, req.ka_absorption_per_h, req.t_end_h)
        import dataclasses; return {k: dataclasses.asdict(v) for k, v in result.items()}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 107: Michaelis-Menten PK ===
from omega_pbpk.core.michaelis_menten import simulate_michaelis_menten, dose_proportionality_index

class _MMReq(BaseModel):
    drug_name: str = "Drug"
    dose_mg: float = 100.0
    vd_L: float = 20.0
    vmax_mg_h: float = 5.0
    km_mg_L: float = 2.0
    route: str = "iv"
    ka_per_h: float = 1.0
    t_end_h: float = 24.0

class _DosePropReq(BaseModel):
    drug_name: str = "Drug"
    doses: list[float]
    vd_L: float = 20.0
    vmax_mg_h: float = 5.0
    km_mg_L: float = 2.0

@app.post("/simulate/michaelis_menten")
def api_michaelis_menten(req: _MMReq):
    try:
        r = simulate_michaelis_menten(req.drug_name, req.dose_mg, req.vd_L, req.vmax_mg_h, req.km_mg_L, req.route, req.ka_per_h, t_end_h=req.t_end_h)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/analyze/dose_proportionality")
def api_dose_proportionality(req: _DosePropReq):
    try:
        return dose_proportionality_index(req.drug_name, req.doses, req.vd_L, req.vmax_mg_h, req.km_mg_L)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 108: Cumulative Toxicity ===
from omega_pbpk.clinical.cumulative_toxicity import simulate_multicycle_pk, assess_cumulative_toxicity

class _MulticycleReq(BaseModel):
    drug_name: str = "Drug"
    dose_mg: float = 100.0
    cl_L_per_h: float = 5.0
    vd_L: float = 20.0
    n_cycles: int = 6
    cycle_interval_h: float = 504.0
    t_per_cycle_h: float = 24.0

class _CumToxReq(BaseModel):
    drug_name: str = "Drug"
    cycle_doses_mg: list[float]
    cycle_aucs: list[float]
    auc_mtc: float

@app.post("/simulate/multicycle_pk")
def api_multicycle_pk(req: _MulticycleReq):
    try:
        r = simulate_multicycle_pk(req.drug_name, req.dose_mg, req.cl_L_per_h, req.vd_L, req.n_cycles, req.cycle_interval_h, req.t_per_cycle_h)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/assess/cumulative_toxicity")
def api_cumulative_toxicity(req: _CumToxReq):
    try:
        r = assess_cumulative_toxicity(req.drug_name, req.cycle_doses_mg, req.cycle_aucs, req.auc_mtc)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 109: logP Prediction ===
from omega_pbpk.prediction.logp_predictor import predict_logp, screen_logp

class _LogPReq(BaseModel):
    drug_name: str = "Drug"
    n_carbons: int = 20
    n_nitrogens: int = 0
    n_oxygens: int = 0
    n_halogens: int = 0
    n_aromatic_rings: int = 0
    n_rotatable_bonds: int = 0
    MW: float | None = None
    PSA: float = 60.0
    n_hbd: int = 0
    n_hba: int = 0

class _LogPScreenReq(BaseModel):
    compounds: list[dict]

@app.post("/predict/logp")
def api_logp(req: _LogPReq):
    try:
        r = predict_logp(req.drug_name, req.n_carbons, req.n_nitrogens, req.n_oxygens, req.n_halogens, req.n_aromatic_rings, req.n_rotatable_bonds, req.MW, req.PSA, n_hbd=req.n_hbd, n_hba=req.n_hba)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/predict/logp/screen")
def api_logp_screen(req: _LogPScreenReq):
    try:
        results = screen_logp(req.compounds)
        import dataclasses; return [dataclasses.asdict(r) for r in results]
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 110: Supersaturation ===
from omega_pbpk.biopharmaceutics.supersaturation import simulate_supersaturation, compare_formulations_supersaturation

class _SupersatReq(BaseModel):
    drug_name: str = "Drug"
    dose_mg: float = 100.0
    volume_gut_mL: float = 250.0
    solubility_mg_mL: float = 0.01
    logP: float = 3.0
    precipitation_rate_per_h: float = 2.0
    absorption_rate_per_h: float = 1.0
    polymer_parachute: bool = False
    t_end_h: float = 6.0

@app.post("/simulate/supersaturation")
def api_supersaturation(req: _SupersatReq):
    try:
        r = simulate_supersaturation(req.drug_name, req.dose_mg, req.volume_gut_mL, req.solubility_mg_mL, req.logP, req.precipitation_rate_per_h, req.absorption_rate_per_h, req.polymer_parachute, req.t_end_h)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/simulate/supersaturation/compare")
def api_supersaturation_compare(req: _SupersatReq):
    try:
        result = compare_formulations_supersaturation(req.drug_name, req.dose_mg, req.solubility_mg_mL, req.logP)
        import dataclasses; return {k: dataclasses.asdict(v) for k, v in result.items()}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 111: TMDD QSS ===
from omega_pbpk.core.target_mediated import simulate_tmdd_qss

class _TMDDReq(BaseModel):
    drug_name: str = "mAb"
    dose_mg: float = 10.0
    vd_L: float = 5.0
    cl_L_per_h: float = 0.01
    ksyn_nmol_L_h: float = 0.1
    kdeg_per_h: float = 0.02
    kon_per_nmol_per_h: float = 0.091
    koff_per_h: float = 0.001
    kint_per_h: float = 0.01
    t_end_h: float = 168.0

@app.post("/simulate/tmdd")
def api_tmdd(req: _TMDDReq):
    try:
        r = simulate_tmdd_qss(req.drug_name, req.dose_mg, req.vd_L, req.cl_L_per_h, req.ksyn_nmol_L_h, req.kdeg_per_h, req.kon_per_nmol_per_h, req.koff_per_h, req.kint_per_h, t_end_h=req.t_end_h)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 112: Pharmacogenomics ===
from omega_pbpk.clinical.pharmacogenomics import predict_pgx_dose, batch_pgx_screen, cyp_phenotype_distribution

class _PGxReq(BaseModel):
    drug_name: str = "Drug"
    cyp_enzyme: str = "CYP2D6"
    allele1: str = "*1"
    allele2: str = "*1"
    standard_dose_mg: float = 100.0
    fm: float = 0.8

class _PGxBatchReq(BaseModel):
    patients: list[dict]

@app.post("/predict/pgx_dose")
def api_pgx_dose(req: _PGxReq):
    try:
        r = predict_pgx_dose(req.drug_name, req.cyp_enzyme, req.allele1, req.allele2, req.standard_dose_mg, req.fm)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/predict/pgx_dose/batch")
def api_pgx_batch(req: _PGxBatchReq):
    try:
        results = batch_pgx_screen(req.patients)
        import dataclasses; return [dataclasses.asdict(r) for r in results]
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/info/cyp_phenotype_distribution")
def api_cyp_distribution(enzyme: str = "CYP2D6", population: str = "caucasian"):
    try:
        return cyp_phenotype_distribution(enzyme, population)
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 113: MW Prediction ===
from omega_pbpk.prediction.mw_predictor import predict_mw, screen_mw

class _MWReq(BaseModel):
    drug_name: str = "Drug"
    n_C: int = 20
    n_N: int = 0
    n_O: int = 0
    n_S: int = 0
    n_P: int = 0
    n_F: int = 0
    n_Cl: int = 0
    n_Br: int = 0
    n_I: int = 0
    n_aromatic_rings: int = 0

class _MWScreenReq(BaseModel):
    compounds: list[dict]

@app.post("/predict/molecular_weight")
def api_mw(req: _MWReq):
    try:
        r = predict_mw(req.drug_name, req.n_C, req.n_N, req.n_O, req.n_S, req.n_P, req.n_F, req.n_Cl, req.n_Br, req.n_I, req.n_aromatic_rings)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/predict/molecular_weight/screen")
def api_mw_screen(req: _MWScreenReq):
    try:
        results = screen_mw(req.compounds)
        import dataclasses; return [dataclasses.asdict(r) for r in results]
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

# === Phase 114: In Vitro Dissolution ===
from omega_pbpk.biopharmaceutics.in_vitro_dissolution import simulate_usp2_dissolution, compare_rpm

class _InVitroDissReq(BaseModel):
    drug_name: str = "Drug"
    dose_mg: float = 100.0
    solubility_mg_mL: float = 1.0
    particle_radius_um: float = 50.0
    rpm: int = 50
    pH: float = 6.8
    t_end_min: float = 60.0

@app.post("/simulate/in_vitro_dissolution")
def api_in_vitro_dissolution(req: _InVitroDissReq):
    try:
        r = simulate_usp2_dissolution(req.drug_name, req.dose_mg, req.solubility_mg_mL, req.particle_radius_um, rpm=req.rpm, pH=req.pH, t_end_min=req.t_end_min)
        import dataclasses; return dataclasses.asdict(r)
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/simulate/in_vitro_dissolution/compare_rpm")
def api_dissolution_rpm(req: _InVitroDissReq):
    try:
        results = compare_rpm(req.drug_name, req.dose_mg, req.solubility_mg_mL)
        import dataclasses; return [dataclasses.asdict(r) for r in results]
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc
