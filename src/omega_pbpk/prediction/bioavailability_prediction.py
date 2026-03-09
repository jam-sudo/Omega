"""Oral bioavailability prediction: F = fa × fg × fh.

Components:
  fa  — fraction absorbed (Peff/solubility limited)
  fg  — gut-wall first-pass survival (1 - E_gut)
  fh  — hepatic first-pass survival (1 - E_hepatic, well-stirred model)

Reference: Rowland & Tozer, Clinical Pharmacokinetics, 5th Ed., Ch. 7.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BioavailabilityResult:
    """Result of oral bioavailability prediction."""

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


def predict_bioavailability(
    drug_or_dict,
    dose_mg: float = 100.0,
    q_liver_L_per_h: float = 90.0,
    gut_volume_mL: float = 250.0,
) -> BioavailabilityResult:
    """Predict oral bioavailability F = fa × fg × fh.

    Parameters
    ----------
    drug_or_dict : Drug or dict
        Drug object or dictionary with drug properties.
    dose_mg : float
        Oral dose in mg.
    q_liver_L_per_h : float
        Hepatic blood flow in L/h.
    gut_volume_mL : float
        GI fluid volume in mL.

    Returns
    -------
    BioavailabilityResult
    """
    # Extract parameters from Drug object or dict
    if hasattr(drug_or_dict, "peff"):
        peff = drug_or_dict.peff
        solubility_mg_mL = getattr(drug_or_dict, "solubility_mg_mL", 0.1)
        fup = drug_or_dict.fup
        clint_hepatic = drug_or_dict.clint_hepatic_L_per_h
        clint_gut = getattr(drug_or_dict, "clint_gut_L_per_h", 0.0)
        has_explicit_peff = True
        has_explicit_sol = hasattr(drug_or_dict, "solubility_mg_mL")
    else:
        peff = drug_or_dict.get("peff", 1.0)
        solubility_mg_mL = drug_or_dict.get("solubility_mg_mL", 0.1)
        fup = drug_or_dict["fup"]
        clint_hepatic = drug_or_dict["clint_hepatic_L_per_h"]
        clint_gut = drug_or_dict.get("clint_gut_L_per_h", 0.0)
        has_explicit_peff = "peff" in drug_or_dict
        has_explicit_sol = "solubility_mg_mL" in drug_or_dict

    # 1. fa — fraction absorbed
    # peff is in units of ×10⁻⁴ cm/s.  Convert to cm/s for calculations.
    peff_cm_s = peff * 1e-4
    dose_number = dose_mg / (solubility_mg_mL * gut_volume_mL)

    # Permeability-limited absorption using the compartmental absorption model.
    # BCS threshold: Papp > ~1×10⁻⁶ cm/s (Caco-2) = 0.01 ×10⁻⁴ cm/s is "high"
    # peff=0.01 → low; peff=1.0 → moderate; peff=10+ → high; peff=100 → complete
    # Use effective intestinal surface model:
    #   fa_perm = 1 - exp(-2 × Peff × R_SI × T_SI)
    # where R_SI ≈ 175 cm (small intestine radius), T_SI ≈ 3.32 h mean transit
    # Simplified: fa_perm = 1 - exp(-k × peff_cm_s) with k calibrated to
    # give fa=0.90 at peff=1×10⁻⁵ cm/s (BCS I/II threshold)
    # k = -ln(0.10) / 1e-5 ≈ 230,259
    # This gives: peff=4e-5 → fa=0.9999, peff=1e-6 → fa=0.206
    import math

    k_abs = 2.3e5  # absorption rate constant (1/cm·s → dimensionless via peff)
    fa_perm = 1.0 - math.exp(-k_abs * peff_cm_s) if peff_cm_s > 0 else 0.0
    fa_sol = 1.0 / max(dose_number, 1.0)
    fa = fa_perm * fa_sol if dose_number > 1.0 else fa_perm
    fa = max(0.0, min(1.0, fa))

    # 2. fg — gut-wall survival
    Q_gut = 18.0  # mesenteric blood flow, L/h
    if clint_gut > 0:
        fg = Q_gut / (Q_gut + fup * clint_gut)
        fg = max(0.0, min(1.0, fg))
    else:
        fg = 1.0

    # 3. fh — hepatic first-pass (well-stirred model)
    clh = (fup * clint_hepatic * q_liver_L_per_h) / (q_liver_L_per_h + fup * clint_hepatic)
    extraction_ratio = clh / q_liver_L_per_h if q_liver_L_per_h > 0 else 0.0
    fh = 1.0 - extraction_ratio
    fh = max(0.0, min(1.0, fh))

    # 4. F_total
    F_total = fa * fg * fh

    # 5. limiting_factor
    losses = {"absorption": 1 - fa, "gut_metabolism": 1 - fg, "hepatic": 1 - fh}
    if all(v < 0.05 for v in losses.values()):
        limiting_factor = "none"
    else:
        limiting_factor = max(losses, key=losses.get)

    # 6. confidence
    confidence = "high" if has_explicit_peff and has_explicit_sol else "medium"

    return BioavailabilityResult(
        fa=fa,
        fg=fg,
        fh=fh,
        F_total=F_total,
        dose_number=dose_number,
        peff_cm_s=peff_cm_s,
        clh_L_per_h=clh,
        extraction_ratio=extraction_ratio,
        limiting_factor=limiting_factor,
        confidence=confidence,
    )
