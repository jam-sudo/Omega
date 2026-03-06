"""Plasma protein binding prediction — albumin, AAG, lipoprotein binding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Protein concentrations in plasma
# ---------------------------------------------------------------------------
ALBUMIN_CONC_uM: float = 600.0  # human plasma albumin ~600 μM
AGP_CONC_uM: float = 15.0  # α1-acid glycoprotein ~15 μM
LIPOPROTEIN_CONC_uM: float = 5.0  # combined lipoprotein particles ~5 μM

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProteinBindingResult:
    drug_name: str
    logP: float
    pka_acid: float  # pKa of acidic group (0 if none)
    pka_basic: float  # pKa of basic group (0 if none)
    fup: float  # fraction unbound in plasma (0-1)
    f_albumin: float  # fraction bound to albumin
    f_agp: float  # fraction bound to AGP
    f_lipoprotein: float  # fraction bound to lipoprotein
    ka_albumin_L_uM: float  # affinity constant (L/μM) for albumin
    fu_blood: float  # fraction unbound in blood (accounts for RBC binding)
    blood_plasma_ratio: float  # Cb/Cp
    disease_adjustments: dict  # {"CKD": fup_ckd, "liver_cirrhosis": fup_cirrhosis}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ka_albumin(logP: float, pka_acid: float) -> float:
    """Estimate albumin affinity constant (L/μM).

    Albumin has 2 main binding sites:
    - Site I (warfarin site): prefers acidic drugs with logP 2-4
    - Site II (diazepam site): prefers neutral/basic lipophilic drugs
    """
    ka_base = 10 ** (0.5 * logP - 2)
    acid_bonus = 2.0 if (pka_acid > 0 and pka_acid < 5) else 1.0
    ka = np.clip(ka_base * acid_bonus, 0.001, 100.0)
    return float(ka)


def _ka_agp(logP: float, pka_basic: float) -> float:
    """Estimate AGP affinity constant (L/μM).

    AGP preferentially binds basic and lipophilic drugs.
    """
    ka_base = 10 ** (0.4 * logP - 1.5)
    if pka_basic > 7.5:
        basic_bonus = 3.0
    elif pka_basic > 5:
        basic_bonus = 1.5
    else:
        basic_bonus = 1.0
    ka = np.clip(ka_base * basic_bonus, 0.001, 50.0)
    return float(ka)


def _fu_from_binding(
    ka_albumin: float,
    ka_agp: float,
    drug_conc_uM: float = 0.1,
) -> tuple[float, float, float]:
    """Return (fup, f_albumin, f_agp) using 1-site binding at low drug conc."""
    f_albumin_frac = (ka_albumin * ALBUMIN_CONC_uM) / (1 + ka_albumin * ALBUMIN_CONC_uM)
    f_agp_frac = (ka_agp * AGP_CONC_uM) / (1 + ka_agp * AGP_CONC_uM)

    # Combined: fup + f_alb + f_agp = 1 (simplified, ignore lipoprotein)
    total_bound = f_albumin_frac + f_agp_frac
    if total_bound > 0.999:
        total_bound = 0.999

    # Adjust f_alb and f_agp to sum properly
    denom = 1 + f_albumin_frac + f_agp_frac
    f_alb = f_albumin_frac / denom
    f_agp = f_agp_frac / denom
    fup = 1.0 - f_alb - f_agp

    fup = float(np.clip(fup, 0.0, 1.0))
    f_alb = float(np.clip(f_alb, 0.0, 1.0))
    f_agp = float(np.clip(f_agp, 0.0, 1.0))
    return fup, f_alb, f_agp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_protein_binding(
    drug_name: str,
    logP: float,
    pka_acid: float = 0.0,
    pka_basic: float = 0.0,
    MW: float = 300.0,
    drug_conc_uM: float = 0.1,
) -> ProteinBindingResult:
    """Predict plasma protein binding for a drug compound.

    Parameters
    ----------
    drug_name : str
        Name or identifier of the drug.
    logP : float
        Octanol-water partition coefficient (log scale).
    pka_acid : float
        pKa of the acidic group (0 if none).
    pka_basic : float
        pKa of the basic group (0 if none).
    MW : float
        Molecular weight in Da.
    drug_conc_uM : float
        Drug concentration in plasma (μM).

    Returns
    -------
    ProteinBindingResult
    """
    ka_alb = _ka_albumin(logP, pka_acid)
    ka_agp = _ka_agp(logP, pka_basic)
    fup, f_alb, f_agp = _fu_from_binding(ka_alb, ka_agp, drug_conc_uM)

    # Rough lipoprotein binding estimate
    f_lipo = float(np.clip(logP * 0.01, 0.0, 0.1))

    # Adjust fup for lipoprotein binding
    fup = max(fup - f_lipo, 0.001)

    # Blood-to-plasma ratio: RBC binding for basic lipophilic drugs
    if pka_basic > 7:
        bpr = 1.0 + max(logP - 2, 0) * 0.3
    else:
        bpr = 1.0 + logP * 0.05
    bpr = float(np.clip(bpr, 0.5, 3.0))

    fu_blood = fup / bpr

    # Disease adjustments
    disease_adjustments = {
        "CKD": float(np.clip(fup * 1.5, 0.0, 1.0)),
        "liver_cirrhosis": float(np.clip(fup * 1.8, 0.0, 1.0)),
    }

    return ProteinBindingResult(
        drug_name=drug_name,
        logP=logP,
        pka_acid=pka_acid,
        pka_basic=pka_basic,
        fup=fup,
        f_albumin=f_alb,
        f_agp=f_agp,
        f_lipoprotein=f_lipo,
        ka_albumin_L_uM=ka_alb,
        fu_blood=fu_blood,
        blood_plasma_ratio=bpr,
        disease_adjustments=disease_adjustments,
    )


def binding_sensitivity_analysis(
    drug_name: str,
    logP: float,
    pka_acid: float = 0.0,
    pka_basic: float = 0.0,
    albumin_range: list[float] | None = None,
) -> list[dict]:
    """Analyse sensitivity of fup to albumin concentration changes.

    Parameters
    ----------
    drug_name : str
        Name or identifier of the drug.
    logP : float
        Octanol-water partition coefficient (log scale).
    pka_acid : float
        pKa of the acidic group (0 if none).
    pka_basic : float
        pKa of the basic group (0 if none).
    albumin_range : list[float] | None
        Albumin concentrations (μM) to evaluate.  Defaults to
        ``[200, 400, 600, 700, 800]``.

    Returns
    -------
    list[dict]
        Each dict contains ``albumin_uM``, ``fup``, and ``fup_change_pct``
        relative to the baseline at 600 μM.
    """
    if albumin_range is None:
        albumin_range = [200.0, 400.0, 600.0, 700.0, 800.0]

    ka_alb = _ka_albumin(logP, pka_acid)
    ka_agp = _ka_agp(logP, pka_basic)

    # Baseline fup at standard albumin (600 μM)
    global ALBUMIN_CONC_uM
    original_albumin = ALBUMIN_CONC_uM

    # Compute baseline
    ALBUMIN_CONC_uM = 600.0
    _fup_base, _, _ = _fu_from_binding(ka_alb, ka_agp)
    f_lipo = float(np.clip(logP * 0.01, 0.0, 0.1))
    fup_baseline = max(_fup_base - f_lipo, 0.001)

    results: list[dict] = []
    for alb in albumin_range:
        ALBUMIN_CONC_uM = alb
        _fup, _, _ = _fu_from_binding(ka_alb, ka_agp)
        fup = max(_fup - f_lipo, 0.001)
        change_pct = ((fup - fup_baseline) / fup_baseline) * 100.0
        results.append(
            {
                "albumin_uM": alb,
                "fup": round(fup, 6),
                "fup_change_pct": round(change_pct, 2),
            }
        )

    # Restore global
    ALBUMIN_CONC_uM = original_albumin
    return results


__all__ = [
    "ProteinBindingResult",
    "predict_protein_binding",
    "binding_sensitivity_analysis",
]
