"""Phase 1000 — Thousand-compounds ADMET benchmark.

Screens 1000 virtual compounds through key ADMET prediction endpoints using a
deterministic LCG-based compound generator (no external RNG dependency).
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate results from the 1000-compound benchmark."""

    n_compounds: int
    mean_logp: float
    mean_mw: float
    mean_solubility_mg_mL: float
    mean_papp_cm_s: float
    mean_t_half_min: float
    pct_lipinski_pass: float
    pct_bcs_class_i: float
    pct_bcs_class_ii: float
    pct_bcs_class_iii: float
    pct_bcs_class_iv: float
    best_drug_likeness_name: str
    best_drug_likeness_score: float
    worst_drug_likeness_name: str
    worst_drug_likeness_score: float
    runtime_ms: float
    notes: str


# ---------------------------------------------------------------------------
# Deterministic LCG (no random module)
# ---------------------------------------------------------------------------


def _lcg(seed: int) -> int:
    """Linear congruential generator (deterministic)."""
    a, c, m = 1664525, 1013904223, 2**32
    return (a * seed + c) % m


def _generate_compound(idx: int, seed: int) -> dict:
    """Generate a single virtual compound using the LCG chain."""
    s = seed
    for _i in range(idx + 1):
        s = _lcg(s)

    # Derive properties from successive LCG draws
    s0 = s
    s1 = _lcg(s0)
    s2 = _lcg(s1)
    s3 = _lcg(s2)
    s4 = _lcg(s3)

    logp = (s0 % 900) / 100.0 - 2.0  # -2.0 to 6.99
    mw = 100.0 + (s1 % 600)  # 100 to 699
    psa = s2 % 200  # 0 to 199 Å²
    hbd = s3 % 7  # 0 to 6
    hba = s4 % 13  # 0 to 12

    return {
        "name": f"Compound_{idx:04d}",
        "logp": logp,
        "mw": mw,
        "psa": psa,
        "hbd": hbd,
        "hba": hba,
    }


def _compute_properties(cmpd: dict) -> dict:
    """Compute ADMET-derived properties for a generated compound."""
    logp: float = cmpd["logp"]
    mw: float = cmpd["mw"]
    psa: float = cmpd["psa"]
    hbd: int = cmpd["hbd"]
    hba: int = cmpd["hba"]

    # Intrinsic solubility (mg/mL)
    so = 10.0 ** (-0.7 * logp + 0.44)

    # Apparent permeability (cm/s) — Papp MDCK/Caco-2 empirical
    papp_exp = -0.90 - 0.011 * psa - 0.24 * hbd + 0.27 * logp - 0.0015 * mw
    papp = 10.0**papp_exp

    # Lipinski Ro5
    lip_pass = int(mw < 500 and logp < 5 and hbd < 5 and hba < 10)

    # Metabolic t½ (min) — simplified rule-based
    base = 450.0
    denom = max(2.0, 5.0 + 30.0 * (1 if logp > 2 else 0) + 25.0 * (1 if mw < 300 else 0))
    t_half = base / denom

    # Drug-likeness score (0-100)
    score = 0.0
    # Lipinski components (4 × 15 pts)
    if mw < 500:
        score += 15.0
    if logp < 5:
        score += 15.0
    if hbd <= 5:
        score += 15.0
    if hba <= 10:
        score += 15.0
    # Veber: PSA <= 140, rotatable bonds ≈ modelled by PSA threshold
    if psa <= 140:
        score += 20.0
    # Solubility bonus
    if so >= 0.1:
        score += 10.0
    # Permeability bonus (Papp > 1e-6 cm/s)
    if papp > 1e-6:
        score += 10.0

    # BCS classification
    # Dimensionless dose number Do = (dose/vol) / solubility; use dose=1 mg, vol=250 mL
    dose_mg = 1.0
    vol_mL = 250.0
    sol_mg_mL = so
    perm_threshold = 1e-6  # cm/s; > threshold = high permeability

    if sol_mg_mL > 0:
        do = (dose_mg / vol_mL) / sol_mg_mL
    else:
        do = 1e9  # practically insoluble → high Do

    high_sol = do < 1.0  # Do < 1 → high solubility
    high_perm = papp > perm_threshold

    if high_sol and high_perm:
        bcs = "I"
    elif not high_sol and high_perm:
        bcs = "II"
    elif high_sol and not high_perm:
        bcs = "III"
    else:
        bcs = "IV"

    return {
        "so": so,
        "papp": papp,
        "lip_pass": lip_pass,
        "t_half": t_half,
        "score": score,
        "bcs": bcs,
    }


def run_benchmark(n_compounds: int = 1000, seed: int = 42) -> BenchmarkResult:
    """Screen *n_compounds* virtual compounds through the ADMET pipeline.

    Uses a deterministic LCG-based generator, so results are fully
    reproducible given the same *seed*.

    Parameters
    ----------
    n_compounds:
        Number of virtual compounds to screen (default 1000).
    seed:
        Initial LCG seed for reproducible compound generation.

    Returns
    -------
    BenchmarkResult
    """
    t_start = time.time()

    sum_logp = 0.0
    sum_mw = 0.0
    sum_sol = 0.0
    sum_papp = 0.0
    sum_thalf = 0.0
    n_lip = 0
    n_bcs = {"I": 0, "II": 0, "III": 0, "IV": 0}

    best_name = ""
    best_score = -1.0
    worst_name = ""
    worst_score = 101.0

    for idx in range(n_compounds):
        cmpd = _generate_compound(idx, seed)
        props = _compute_properties(cmpd)

        sum_logp += cmpd["logp"]
        sum_mw += cmpd["mw"]
        sum_sol += props["so"]
        sum_papp += props["papp"]
        sum_thalf += props["t_half"]
        n_lip += props["lip_pass"]
        n_bcs[props["bcs"]] += 1

        sc = props["score"]
        name = cmpd["name"]
        if sc > best_score:
            best_score = sc
            best_name = name
        if sc < worst_score:
            worst_score = sc
            worst_name = name

    t_end = time.time()
    runtime_ms = (t_end - t_start) * 1000.0

    n = n_compounds
    pct_lip = 100.0 * n_lip / n
    pct_i = 100.0 * n_bcs["I"] / n
    pct_ii = 100.0 * n_bcs["II"] / n
    pct_iii = 100.0 * n_bcs["III"] / n
    pct_iv = 100.0 * n_bcs["IV"] / n

    notes = (
        f"Deterministic LCG benchmark, seed={seed}, n={n_compounds}. "
        f"Lipinski pass: {pct_lip:.1f}%, BCS I/II/III/IV: "
        f"{pct_i:.1f}/{pct_ii:.1f}/{pct_iii:.1f}/{pct_iv:.1f}%"
    )

    return BenchmarkResult(
        n_compounds=n_compounds,
        mean_logp=sum_logp / n,
        mean_mw=sum_mw / n,
        mean_solubility_mg_mL=sum_sol / n,
        mean_papp_cm_s=sum_papp / n,
        mean_t_half_min=sum_thalf / n,
        pct_lipinski_pass=pct_lip,
        pct_bcs_class_i=pct_i,
        pct_bcs_class_ii=pct_ii,
        pct_bcs_class_iii=pct_iii,
        pct_bcs_class_iv=pct_iv,
        best_drug_likeness_name=best_name,
        best_drug_likeness_score=best_score,
        worst_drug_likeness_name=worst_name,
        worst_drug_likeness_score=worst_score,
        runtime_ms=runtime_ms,
        notes=notes,
    )
