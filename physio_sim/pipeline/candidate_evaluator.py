from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from physio_sim.config import CompoundConfig, SubjectConfig, load_subject
from physio_sim.pbpk.solver import simulate
from physio_sim.risk.flags import RiskFlags, compute_risk_flags
from physio_sim.uncertainty import monte_carlo_propagation
from physio_sim.utils.metrics import auc_trapezoid, cmax_tmax


@dataclass(frozen=True)
class CandidateEvaluation:
    scores: dict[str, float]
    risk_flags: RiskFlags


def _load_candidate(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = "compound.yaml root must be a mapping"
        raise ValueError(msg)
    return cast(dict[str, Any], data)


def evaluate_candidate(
    candidate_path: Path,
    default_subject_path: Path,
    dose_mg: float,
    route: str,
    t_end_h: float,
    dt_out_h: float,
    seed: int | None,
    deterministic: bool,
) -> CandidateEvaluation:
    candidate = _load_candidate(candidate_path)
    compound = CompoundConfig.model_validate(candidate.get("compound", candidate))
    subject = load_subject(candidate.get("subject", default_subject_path))
    if not isinstance(subject, SubjectConfig):
        msg = "Invalid subject configuration"
        raise ValueError(msg)

    run = simulate(
        subject, compound, dose_mg, route, t_end_h, dt_out_h, deterministic=deterministic
    )
    tc = run.timecourse
    times = tc["time_h"].to_numpy()
    conc = tc["C_plasma_mg_per_L"].to_numpy()
    cmax, _ = cmax_tmax(times, conc)
    auc = auc_trapezoid(times, conc)

    cmax_idx = int(np.argmax(conc))
    cmax_time = times[cmax_idx]
    terminal = float(tc["C_plasma_mg_per_L"].iloc[-1])
    tail = conc[cmax_idx:]
    if cmax <= 0.0 or len(tail) < 2 or tail[-1] >= cmax:
        half_life = float("inf")
    else:
        decay_delta = np.log(cmax / max(terminal, 1e-12))
        dt = t_end_h - cmax_time
        half_life = float(
            np.inf if decay_delta <= 0.0 or dt <= 0.0 else np.log(2.0) * (dt / decay_delta)
        )

    uncertainty_cfg = cast(dict[str, Any], candidate.get("uncertainty", {}))
    specs = cast(
        dict[str, Any],
        uncertainty_cfg.get(
            "parameters",
            {
                "clint_L_per_h": {
                    "dist": "lognormal",
                    "mean": np.log(max(compound.clint_L_per_h, 1e-9)),
                    "sd": 0.2,
                },
                "fu_plasma": {
                    "dist": "lognormal",
                    "mean": np.log(max(compound.fu_plasma, 1e-9)),
                    "sd": 0.15,
                },
                "ka_per_h": {
                    "dist": "lognormal",
                    "mean": np.log(max(compound.ka_per_h, 1e-9)),
                    "sd": 0.2,
                },
            },
        ),
    )
    unc = monte_carlo_propagation(
        subject=subject,
        compound=compound,
        dose_mg=dose_mg,
        route=route,
        t_end_h=t_end_h,
        dt_out_h=dt_out_h,
        n_samples=int(uncertainty_cfg.get("n_samples", 64)),
        parameter_specs=specs,
        seed=seed,
        n_workers=1 if deterministic else int(uncertainty_cfg.get("n_workers", 1)),
    ).sample_metrics

    auc_cv = float(
        unc["AUC0_tend_mg_h_per_L"].std(ddof=1) / max(unc["AUC0_tend_mg_h_per_L"].mean(), 1e-9)
    )

    genotype_cfg = cast(dict[str, Any], candidate.get("genotype_scenario", {}))
    genotype_factor = float(genotype_cfg.get("clint_factor", 1.0))
    genotype_compound = compound.model_copy(
        update={"clint_L_per_h": max(1e-12, compound.clint_L_per_h * genotype_factor)}
    )
    geno_run = simulate(
        subject, genotype_compound, dose_mg, route, t_end_h, dt_out_h, deterministic=deterministic
    )
    geno_auc = auc_trapezoid(
        geno_run.timecourse["time_h"].to_numpy(),
        geno_run.timecourse["C_plasma_mg_per_L"].to_numpy(),
    )

    ddi_score = 1.0 - min(
        1.0,
        (compound.ddi.ki_mg_per_L if compound.ddi and compound.ddi.ki_mg_per_L else 10.0) / 10.0,
    )
    scores = {
        "pk_stability_score": float(1.0 / (1.0 + auc_cv)),
        "clearance_risk": float(max(0.0, min(1.0, 1.0 - compound.clint_L_per_h / 30.0))),
        "dd_interaction_risk": float(max(0.0, min(1.0, ddi_score))),
        "exposure_variability": float(auc_cv),
        "genotype_auc_ratio": float(geno_auc / max(auc, 1e-9)),
        "estimated_half_life_h": half_life,
        "reference_cmax_mg_per_L": float(cmax),
    }

    flags = compute_risk_flags(cmax, half_life, auc_cv, scores["dd_interaction_risk"])
    return CandidateEvaluation(scores=scores, risk_flags=flags)
