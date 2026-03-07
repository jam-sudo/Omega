"""Intestinal first-pass metabolism predictor.

Predicts intestinal first-pass extraction (fg) and its contribution to oral
bioavailability using CYP3A4 and P-gp activity in enterocytes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntestinalFirstPassResult:
    drug_name: str
    logp: float
    mw: float
    psa_A2: float
    cyp3a4_substrate: bool
    pgp_substrate: bool
    fa: float
    fg: float
    eg_cyp: float
    pgp_factor: float
    f_oral_estimate: float
    first_pass_gut_pct: float
    dominant_mechanism: str
    bioavailability_class: str
    notes: str


def _validate_inputs(
    logp: float,
    mw: float,
    psa_A2: float,
    cl_int_gut_mL_per_min: float,
    qgut_L_per_h: float,
) -> None:
    if not (-5 < logp < 10):
        raise ValueError(f"logp must be in (-5, 10), got {logp}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")
    if cl_int_gut_mL_per_min < 0:
        raise ValueError(f"cl_int_gut_mL_per_min must be >= 0, got {cl_int_gut_mL_per_min}")
    if qgut_L_per_h <= 0:
        raise ValueError(f"qgut_L_per_h must be > 0, got {qgut_L_per_h}")


def predict_intestinal_first_pass(
    drug_name: str,
    logp: float,
    mw: float,
    psa_A2: float,
    pka: float,
    cyp3a4_substrate: bool,
    pgp_substrate: bool,
    cl_int_gut_mL_per_min: float = 10.0,
    qgut_L_per_h: float = 18.0,
) -> IntestinalFirstPassResult:
    """Predict intestinal first-pass extraction and oral bioavailability.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logp:
        Octanol-water partition coefficient, must be in (-5, 10).
    mw:
        Molecular weight in g/mol, must be > 0.
    psa_A2:
        Polar surface area in Angstrom^2, must be >= 0.
    pka:
        Acid dissociation constant.
    cyp3a4_substrate:
        Whether the drug is a CYP3A4 substrate.
    pgp_substrate:
        Whether the drug is a P-gp substrate.
    cl_int_gut_mL_per_min:
        Intrinsic clearance in gut wall (mL/min), default 10.0.
    qgut_L_per_h:
        Gut blood flow (L/h), default 18.0.

    Returns
    -------
    IntestinalFirstPassResult
    """
    _validate_inputs(logp, mw, psa_A2, cl_int_gut_mL_per_min, qgut_L_per_h)

    # Absorptive permeability (Peff proxy)
    peff_raw = 10 ** (0.4 * logp - 0.01 * psa_A2 - 3)
    peff_cm_s = max(1e-7, min(1e-3, peff_raw))

    # Fraction absorbed: normalized to reference Peff of 1e-5 cm/s
    fa_raw = peff_cm_s / 1e-5
    fa = max(0.01, min(1.0, fa_raw))

    # Intestinal CYP3A4 extraction
    if cyp3a4_substrate:
        cl_int_gut_L_per_h = cl_int_gut_mL_per_min * 60.0 / 1000.0
        eg_cyp = cl_int_gut_L_per_h / (qgut_L_per_h + cl_int_gut_L_per_h)
    else:
        cl_int_gut_L_per_h = 0.0
        eg_cyp = 0.0

    # P-gp efflux factor
    pgp_factor = 0.7 if pgp_substrate else 1.0

    # Gut availability
    fg_raw = (1.0 - eg_cyp) * pgp_factor
    fg = max(0.01, min(1.0, fg_raw))

    # Oral bioavailability estimate (hepatic FH = 0.7)
    fh = 0.7
    f_oral_estimate = fa * fg * fh

    # Intestinal first-pass loss
    first_pass_gut_pct = (1.0 - fg) * 100.0

    # Dominant mechanism
    pgp_loss = 1.0 - pgp_factor  # 0 or 0.3
    if eg_cyp > 0.2 and eg_cyp > pgp_loss:
        dominant_mechanism = "cyp3a4"
    elif pgp_substrate and eg_cyp <= 0.2:
        dominant_mechanism = "pgp_efflux"
    else:
        dominant_mechanism = "minimal"

    # Bioavailability class
    if f_oral_estimate >= 0.6:
        bioavailability_class = "high"
    elif f_oral_estimate >= 0.3:
        bioavailability_class = "moderate"
    else:
        bioavailability_class = "low"

    notes_parts: list[str] = []
    if fa < 0.3:
        notes_parts.append("poor permeability limits absorption")
    if eg_cyp > 0.3:
        notes_parts.append("significant CYP3A4 gut-wall extraction")
    if pgp_substrate:
        notes_parts.append("P-gp efflux reduces effective absorption")
    if not notes_parts:
        notes_parts.append("intestinal first-pass effects are minimal")
    notes = "; ".join(notes_parts)

    return IntestinalFirstPassResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        psa_A2=psa_A2,
        cyp3a4_substrate=cyp3a4_substrate,
        pgp_substrate=pgp_substrate,
        fa=fa,
        fg=fg,
        eg_cyp=eg_cyp,
        pgp_factor=pgp_factor,
        f_oral_estimate=f_oral_estimate,
        first_pass_gut_pct=first_pass_gut_pct,
        dominant_mechanism=dominant_mechanism,
        bioavailability_class=bioavailability_class,
        notes=notes,
    )


def screen_substrates(substrate_list: list[dict]) -> list[IntestinalFirstPassResult]:
    """Screen a list of drug substrates and rank by fg ascending (worst first).

    Parameters
    ----------
    substrate_list:
        List of dicts, each containing keys matching the parameters of
        predict_intestinal_first_pass (drug_name, logp, mw, psa_A2, pka,
        cyp3a4_substrate, pgp_substrate, and optionally cl_int_gut_mL_per_min
        and qgut_L_per_h).

    Returns
    -------
    List of IntestinalFirstPassResult sorted by fg ascending (worst first).
    """
    results: list[IntestinalFirstPassResult] = []
    for entry in substrate_list:
        result = predict_intestinal_first_pass(
            drug_name=entry["drug_name"],
            logp=entry["logp"],
            mw=entry["mw"],
            psa_A2=entry["psa_A2"],
            pka=entry["pka"],
            cyp3a4_substrate=entry["cyp3a4_substrate"],
            pgp_substrate=entry["pgp_substrate"],
            cl_int_gut_mL_per_min=entry.get("cl_int_gut_mL_per_min", 10.0),
            qgut_L_per_h=entry.get("qgut_L_per_h", 18.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.fg)
    return results
