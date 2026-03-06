"""Tissue-to-plasma concentration ratio prediction (Kp).

Uses simplified Rodgers-Rowland approach.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ConcentrationRatioResult",
    "predict_kp",
    "predict_kp_panel",
]

_DEFAULT_TISSUES = ["muscle", "liver", "brain", "adipose", "kidney", "lung", "heart"]

_KP_METHOD = "rodgers_rowland_simplified"


@dataclass
class ConcentrationRatioResult:
    drug_name: str
    logP: float
    fup: float
    pka: float
    tissue: str
    kp_predicted: float
    kp_method: str
    tissue_conc_free_uM: float
    plasma_conc_total_mg_L: float | None
    tissue_binding_factor: float


def _estimate_fut(logP: float, fup: float, tissue: str) -> float:
    lp = max(logP, 0.0)
    t = tissue.lower()
    if t == "muscle":
        fut = fup / (1.0 + 0.3 * lp)
    elif t == "liver":
        fut = fup * 0.7
    elif t == "brain":
        fut = fup / (1.0 + 0.25 * lp)
    elif t == "adipose":
        fut = fup / (1.0 + 0.6 * lp)
    elif t == "kidney":
        fut = fup / (1.0 + 0.15 * lp)
    else:
        fut = fup / (1.0 + 0.2 * lp)
    return max(0.001, min(fut, 1.0))


def predict_kp(
    drug_name: str,
    logP: float,
    fup: float,
    tissue: str,
    pka: float = 7.0,
    drug_type: str = "neutral",
    plasma_conc_total_mg_L: float | None = None,
) -> ConcentrationRatioResult:
    if fup <= 0 or fup > 1:
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if not tissue or not tissue.strip():
        raise ValueError("tissue must be a non-empty string")

    fut = _estimate_fut(logP, fup, tissue)
    kp = fup / fut

    dt = drug_type.lower()
    if dt == "acid":
        kp *= 0.8
    elif dt == "base":
        kp *= 1.3

    tissue_binding_factor = 1.0 / fut

    # tissue_conc_free_uM: free tissue concentration in µM
    # Using plasma_conc_total_mg_L if provided; assume MW=400 g/mol as placeholder
    # when no plasma_conc given, report 0.0
    if plasma_conc_total_mg_L is not None:
        # tissue_total_mg_L = kp * plasma_conc_total_mg_L
        # tissue_free_mg_L = fut * tissue_total_mg_L
        # tissue_conc_free_uM = (tissue_free_mg_L / MW_g_per_mol) * 1000
        # MW unknown here; compute proportional value in (mg/L) units normalised by 1 g/mol → µmol/g
        # Store as (fut * kp * plasma_conc_total_mg_L) in mg/L for caller to scale by MW
        tissue_free_mg_L = fut * kp * plasma_conc_total_mg_L
        # Convert mg/L → µM using approximate MW=400 g/mol (standard small molecule)
        _mw_approx = 400.0
        tissue_conc_free_uM = (tissue_free_mg_L / _mw_approx) * 1000.0
    else:
        tissue_conc_free_uM = 0.0

    return ConcentrationRatioResult(
        drug_name=drug_name,
        logP=logP,
        fup=fup,
        pka=pka,
        tissue=tissue,
        kp_predicted=kp,
        kp_method=_KP_METHOD,
        tissue_conc_free_uM=tissue_conc_free_uM,
        plasma_conc_total_mg_L=plasma_conc_total_mg_L,
        tissue_binding_factor=tissue_binding_factor,
    )


def predict_kp_panel(
    drug_name: str,
    logP: float,
    fup: float,
    tissues: list[str] | None = None,
    pka: float = 7.0,
    drug_type: str = "neutral",
    plasma_conc_total_mg_L: float | None = None,
) -> list[ConcentrationRatioResult]:
    if tissues is None:
        tissues = _DEFAULT_TISSUES

    results = [
        predict_kp(
            drug_name=drug_name,
            logP=logP,
            fup=fup,
            tissue=t,
            pka=pka,
            drug_type=drug_type,
            plasma_conc_total_mg_L=plasma_conc_total_mg_L,
        )
        for t in tissues
    ]
    results.sort(key=lambda r: r.kp_predicted, reverse=True)
    return results
