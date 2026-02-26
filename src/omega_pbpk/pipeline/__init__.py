"""Integrated drug candidate evaluation pipeline.

Combines PBPK simulation, sensitivity analysis, uncertainty propagation,
safety assessment, pharmacogenomics, and risk flagging into a single
end-to-end evaluation workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug
from omega_pbpk.risk import RiskFlags, compute_risk_flags
from omega_pbpk.uncertainty import DistributionSpec, monte_carlo_propagation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateReport:
    """Comprehensive drug candidate evaluation report.

    Attributes:
        drug_name: Compound name.
        pk_summary: Standard PK parameters.
        risk_flags: PK-based risk assessment.
        exposure_cv: Coefficient of variation of AUC.
        pk_stability_score: 1 / (1 + AUC_CV) ∈ (0, 1].
        clearance_risk_score: CLint-weighted risk.
        ddi_risk_score: DDI susceptibility from fm CYP3A4.
        genotype_auc_ratio: AUC ratio for worst-case CYP PM genotype.
        overall_score: Composite candidate score (0–100).
        details: Additional detail dict.
    """

    drug_name: str
    pk_summary: dict[str, float]
    risk_flags: RiskFlags
    exposure_cv: float
    pk_stability_score: float
    clearance_risk_score: float
    ddi_risk_score: float
    genotype_auc_ratio: float
    overall_score: float
    details: dict[str, Any] = field(default_factory=dict)


def evaluate_candidate(
    drug: Drug,
    dose_mg: float = 10.0,
    route: str = "oral",
    body_weight: float = 70.0,
    t_end_h: float = 24.0,
    n_mc_samples: int = 200,
    seed: int = 42,
    use_surrogate: Any | None = None,
) -> CandidateReport:
    """Run end-to-end drug candidate evaluation.

    Pipeline stages:
    1. Baseline PK simulation
    2. Monte Carlo uncertainty propagation
    3. DDI susceptibility scoring
    4. Pharmacogenomics impact (CYP PM genotype)
    5. PK risk flag computation
    6. Composite scoring

    Args:
        drug: Drug compound to evaluate.
        dose_mg: Dose (mg).
        route: 'oral' or 'iv'.
        body_weight: Subject body weight (kg).
        t_end_h: Simulation end time (h).
        n_mc_samples: Monte Carlo samples for uncertainty.
        seed: Random seed.
        use_surrogate: Optional PKSurrogate for fast evaluation.

    Returns:
        CandidateReport with all assessment metrics.
    """
    # 1. Baseline PK simulation
    model = WholeBodyPBPK(drug, body_weight=body_weight)
    if route == "iv":
        model.setup_iv(dose_mg)
    else:
        model.setup_oral(dose_mg)
    result = model.simulate(t_end_h=t_end_h)
    pk = result.pk_summary()

    # 2. Uncertainty propagation
    base_params = {
        "clint_hepatic_L_per_h": drug.clint_scaled_L_per_h,
        "clint_gut_L_per_h": drug.gut_clint_scaled_L_per_h,
        "fup": drug.fup,
        "rbp": drug.rbp,
        "peff": drug.peff,
        "logP": drug.logP,
    }
    clint_val = base_params["clint_hepatic_L_per_h"]
    uncertainty_specs = [
        DistributionSpec("clint_hepatic_L_per_h", "lognormal", clint_val, 0.3),
        DistributionSpec("fup", "lognormal", max(base_params["fup"], 0.001), 0.2),
    ]

    mc_result = monte_carlo_propagation(
        drug_params=base_params,
        uncertainty_specs=uncertainty_specs,
        n_samples=n_mc_samples,
        dose_mg=dose_mg,
        route=route,
        body_weight=body_weight,
        t_end_h=t_end_h,
        seed=seed,
        use_surrogate=use_surrogate,
    )

    exposure_cv = mc_result.auc_cv
    pk_stability_score = 1.0 / (1.0 + exposure_cv)

    # 3. DDI susceptibility
    fm_3a4 = drug.fm.get("CYP3A4", 0.0)
    ddi_risk_score = fm_3a4  # Simple: high fm CYP3A4 = high DDI risk

    # 4. Pharmacogenomics impact
    # Simulate worst-case PM genotype (CLint reduced to near-zero for major enzyme)
    max_fm_enzyme = max(drug.fm.items(), key=lambda x: x[1], default=("", 0.0))
    if max_fm_enzyme[1] > 0.1:
        # PM genotype: reduce CLint by the fm fraction of the major enzyme
        pm_clint = drug.clint_scaled_L_per_h * (1.0 - max_fm_enzyme[1])
        pm_params = dict(base_params)
        pm_params["clint_hepatic_L_per_h"] = max(pm_clint, 0.01)

        if use_surrogate is not None:
            pm_pk = use_surrogate.predict_dict(pm_params)
        else:
            pm_overrides = {**drug.__dict__}
            pm_overrides["clint_hepatic_L_per_h"] = pm_params["clint_hepatic_L_per_h"]
            pm_drug = Drug(**pm_overrides)
            pm_model = WholeBodyPBPK(pm_drug, body_weight=body_weight)
            if route == "iv":
                pm_model.setup_iv(dose_mg)
            else:
                pm_model.setup_oral(dose_mg)
            pm_pk = pm_model.simulate(t_end_h=t_end_h).pk_summary()

        genotype_auc_ratio = pm_pk["AUC_mg_h_L"] / max(pk["AUC_mg_h_L"], 1e-12)
    else:
        genotype_auc_ratio = 1.0

    # 5. Clearance risk score (high CLint = rapid clearance = may need high dose)
    clearance_risk_score = min(drug.clint_scaled_L_per_h / 100.0, 1.0)

    # 6. Risk flags
    half_life = pk.get("half_life_h", 0.0)
    if half_life == float("inf"):
        half_life = 0.0
    risk_flags = compute_risk_flags(
        cmax_mg_per_l=pk["Cmax_mg_L"],
        half_life_h=half_life,
        exposure_cv=exposure_cv,
        ddi_risk_score=ddi_risk_score,
    )

    # 7. Composite score (0-100)
    score = 100.0
    score -= risk_flags.risk_count * 15.0  # -15 per risk flag
    score -= max(0, (exposure_cv - 0.3)) * 50  # Penalize high variability
    score -= max(0, (genotype_auc_ratio - 2.0)) * 10  # Penalize large genotype effect
    score -= max(0, (ddi_risk_score - 0.3)) * 20  # Penalize DDI vulnerability
    score = max(0.0, min(100.0, score))

    return CandidateReport(
        drug_name=drug.name,
        pk_summary=pk,
        risk_flags=risk_flags,
        exposure_cv=round(exposure_cv, 4),
        pk_stability_score=round(pk_stability_score, 4),
        clearance_risk_score=round(clearance_risk_score, 4),
        ddi_risk_score=round(ddi_risk_score, 4),
        genotype_auc_ratio=round(genotype_auc_ratio, 4),
        overall_score=round(score, 1),
        details={
            "mc_cmax_cv": mc_result.cmax_cv,
            "mc_auc_percentiles": mc_result.auc_percentiles,
            "mc_cmax_percentiles": mc_result.cmax_percentiles,
            "n_mc_samples": mc_result.n_samples,
            "major_enzyme": max_fm_enzyme[0] if max_fm_enzyme[1] > 0 else "none",
        },
    )



# ---------------------------------------------------------------------------
# New SMILES-based pipeline (Task 1)
# ---------------------------------------------------------------------------


@dataclass
class SimulationRequest:
    smiles: str
    dose_mg: float = 100.0
    route: str = "oral"
    duration_h: float = 24.0
    n_timepoints: int = 241
    species: str = "human"
    subject_age_years: float | None = None
    subject_weight_kg: float | None = None


@dataclass
class SimulationResult:
    time_h: "NDArray[np.float64]"
    cp_mg_L: "NDArray[np.float64]"
    cmax_mg_L: float
    tmax_h: float
    auc0t_mg_h_L: float
    t_half_h: float
    adme_properties: "dict[str, Any]"
    confidence: str
    warnings: "list[str]"


class OmegaPipeline:
    def __init__(self) -> None:
        self._adme_predictor = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            from omega_pbpk.prediction.adme_predictor import ADMEPredictor
            self._adme_predictor = ADMEPredictor()
        except Exception:
            self._adme_predictor = None
        self._initialized = True

    def simulate(self, request: "SimulationRequest") -> "SimulationResult":
        self._ensure_initialized()
        warnings_list: list = []
        adme_props = self._predict_adme(request.smiles, warnings_list)
        drug = self._build_drug(request.smiles, adme_props, warnings_list)
        time_h, cp = self._run_simulation(drug, request, warnings_list)
        cmax = float(np.max(cp))
        tmax = float(time_h[np.argmax(cp)])
        auc = float(np.trapz(cp, time_h))
        n = len(time_h)
        tail_start = int(0.7 * n)
        if np.all(cp[tail_start:] > 0):
            log_cp = np.log(cp[tail_start:] + 1e-12)
            t_tail = time_h[tail_start:]
            slope, _ = np.polyfit(t_tail, log_cp, 1)
            t_half = float(-np.log(2) / slope) if slope < 0 else float("nan")
        else:
            t_half = float("nan")
        confidence = adme_props.get("confidence", "low")
        return SimulationResult(
            time_h=time_h, cp_mg_L=cp, cmax_mg_L=cmax, tmax_h=tmax,
            auc0t_mg_h_L=auc, t_half_h=t_half, adme_properties=adme_props,
            confidence=confidence, warnings=warnings_list,
        )

    def _predict_adme(self, smiles: str, warnings_list: list) -> dict:
        if self._adme_predictor is not None:
            try:
                props = self._adme_predictor.predict(smiles)
                return {
                    "mw": props.mw, "logP": props.logP, "logS": props.logS,
                    "fup": props.fup, "rbp": props.rbp, "clint_3a4": props.clint_3a4,
                    "herg_ic50_uM": props.herg_ic50_uM, "confidence": props.confidence,
                }
            except Exception as e:
                warnings_list.append(f"ADME prediction failed: {e}; using defaults")
        else:
            warnings_list.append("ADMEPredictor not available; using default ADME values")
        return {
            "mw": 300.0, "logP": 2.0, "logS": -3.0, "fup": 0.1, "rbp": 0.55,
            "clint_3a4": 5.0, "herg_ic50_uM": 10.0, "confidence": "low",
        }

    def _build_drug(self, smiles: str, adme: dict, warnings_list: list):
        from omega_pbpk.drugs.drug import Drug
        fup = max(float(adme.get("fup", 0.1)), 0.001)
        logP = float(adme.get("logP", 2.0))
        clint_in_vitro = float(adme.get("clint_3a4", 5.0))
        clint_L_per_h = clint_in_vitro * 3.6
        herg = float(adme.get("herg_ic50_uM", 100.0))
        if herg < 1.0:
            warnings_list.append(f"hERG IC50 = {herg:.2f} uM -- potential cardiac safety concern")
        return Drug(
            name=f"compound_{smiles[:8]}",
            mw=float(adme.get("mw", 300.0)),
            logP=logP, fup=fup,
            rbp=float(adme.get("rbp", 0.55)),
            clint_hepatic_L_per_h=clint_L_per_h,
            peff=1.0,
        )

    def _run_simulation(self, drug, request, warnings_list):
        from omega_pbpk.core.body import WholeBodyPBPK
        time_h = np.linspace(0, request.duration_h, request.n_timepoints)
        try:
            model = WholeBodyPBPK(drug=drug)
            if request.route == "iv":
                model.setup_iv(request.dose_mg)
            elif request.route == "sc":
                if hasattr(model, "setup_sc"):
                    model.setup_sc(request.dose_mg)
                else:
                    model.setup_oral(request.dose_mg)
            else:
                model.setup_oral(request.dose_mg)
            result = model.simulate(t_end_h=request.duration_h)
            sim_time = result.time_h
            sim_cp = result.plasma_concentration()
            cp = np.interp(time_h, sim_time, sim_cp)
            return time_h, cp
        except Exception as e:
            warnings_list.append(f"PBPK simulation failed: {e}; returning simplified 1-compartment")
            ke = 0.1
            vd = 50.0
            if request.route == "iv":
                cp = (request.dose_mg / vd) * np.exp(-ke * time_h)
            else:
                ka = 1.0
                cp = (request.dose_mg / vd) * (ka / (ka - ke)) * (np.exp(-ke * time_h) - np.exp(-ka * time_h))
            return time_h, np.maximum(cp, 0.0)


def simulate_with_uncertainty(
    request: "SimulationRequest",
    n_samples: int = 100,
    adme_cv: float = 0.3,
    seed: "int | None" = None,
) -> dict:
    rng = np.random.default_rng(seed)
    pipeline = OmegaPipeline()
    pipeline._ensure_initialized()
    nominal_warnings: list = []
    adme_nominal = pipeline._predict_adme(request.smiles, nominal_warnings)
    time_h = np.linspace(0, request.duration_h, request.n_timepoints)
    cp_matrix = []
    cmax_samples = []
    auc_samples = []
    sigma = float(np.sqrt(np.log(1.0 + adme_cv**2)))
    for _ in range(n_samples):
        perturbed_adme = dict(adme_nominal)
        perturbed_adme["fup"] = float(adme_nominal.get("fup", 0.1)) * float(rng.lognormal(mean=0.0, sigma=sigma))
        perturbed_adme["clint_3a4"] = float(adme_nominal.get("clint_3a4", 5.0)) * float(rng.lognormal(mean=0.0, sigma=sigma))
        perturbed_adme["fup"] = float(np.clip(perturbed_adme["fup"], 0.001, 1.0))
        perturbed_adme["clint_3a4"] = float(np.clip(perturbed_adme["clint_3a4"], 0.001, 1000.0))
        sample_warnings: list = []
        drug = pipeline._build_drug(request.smiles, perturbed_adme, sample_warnings)
        sim_time, cp = pipeline._run_simulation(drug, request, sample_warnings)
        cp_interp = np.interp(time_h, sim_time, cp)
        cp_matrix.append(cp_interp)
        cmax_samples.append(float(np.max(cp_interp)))
        auc_samples.append(float(np.trapz(cp_interp, time_h)))
    cp_arr = np.array(cp_matrix)
    cmax_arr = np.array(cmax_samples)
    auc_arr = np.array(auc_samples)
    return {
        "median_cp": np.median(cp_arr, axis=0),
        "p5_cp": np.percentile(cp_arr, 5, axis=0),
        "p95_cp": np.percentile(cp_arr, 95, axis=0),
        "cmax_samples": cmax_arr,
        "auc_samples": auc_arr,
        "time_h": time_h,
    }


__all__ = [
    "CandidateReport",
    "evaluate_candidate",
    "SimulationRequest",
    "SimulationResult",
    "OmegaPipeline",
    "simulate_with_uncertainty",
]
