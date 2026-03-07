"""Phase 1033 — Drug Nasal Bioavailability prediction.

Models systemic bioavailability via nasal route, accounting for:
- Mucociliary clearance vs. permeability-based absorption
- Ionization effects at nasal pH 7.4
- Formulation-specific deposition fractions
- Hepatic first-pass (nasal bypasses gut but NOT liver)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NasalBioavailabilityResult:
    drug_name: str
    mw_Da: float
    logp: float
    fa_nasal: float
    f_hepatic: float
    f_systemic: float
    t_max_absorption_h: float
    cl_mucociliary_per_h: float
    deposition_fraction: float
    formulation_factor: float
    bioavailability_class: str
    notes: str


_VALID_FORMULATIONS = {"solution", "spray", "powder", "gel"}
_VALID_IONIZATION = {"acid", "base", "neutral"}

_DEPOSITION = {
    "solution": 0.85,
    "spray": 0.70,
    "powder": 0.60,
    "gel": 0.80,
}

_FORMULATION_FACTOR = {
    "solution": 1.0,
    "spray": 0.95,
    "powder": 0.85,
    "gel": 0.90,
}

_K_MCC = 30.0  # mucociliary clearance rate (/h)


def _validate_inputs(
    mw_Da: float,
    dose_mg: float,
    fu_plasma: float,
    cl_hepatic_L_per_h: float,
    q_hepatic_L_per_h: float,
    formulation: str,
    ionization_type: str,
) -> None:
    if mw_Da <= 0 or mw_Da > 2000:
        raise ValueError(f"mw_Da must be in (0, 2000], got {mw_Da}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError(f"cl_hepatic_L_per_h must be > 0, got {cl_hepatic_L_per_h}")
    if q_hepatic_L_per_h <= 0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0, got {q_hepatic_L_per_h}")
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(f"formulation must be one of {_VALID_FORMULATIONS}, got '{formulation}'")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got '{ionization_type}'"
        )


def predict_nasal_bioavailability(
    drug_name: str,
    mw_Da: float,
    logp: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    formulation: str = "solution",
    particle_size_um: float = 0.0,
    dose_mg: float = 1.0,
    fu_plasma: float = 0.3,
    cl_hepatic_L_per_h: float = 10.0,
    q_hepatic_L_per_h: float = 90.0,
) -> NasalBioavailabilityResult:
    """Predict nasal bioavailability for systemic drug delivery."""
    _validate_inputs(
        mw_Da,
        dose_mg,
        fu_plasma,
        cl_hepatic_L_per_h,
        q_hepatic_L_per_h,
        formulation,
        ionization_type,
    )

    # Permeability-based absorption rate
    exponent = 0.5 * logp - 0.001 * mw_Da
    k_abs = min(30.0, max(0.01, 10.0**exponent))

    # Baseline nasal fraction absorbed (before ionization and deposition)
    fa_nasal_base = k_abs / (k_abs + _K_MCC)

    # Ionization penalty at nasal pH 7.4
    if ionization_type == "acid":
        fi = 1.0 / (1.0 + 10.0 ** (pka - 7.4))
    elif ionization_type == "base":
        fi = 1.0 / (1.0 + 10.0 ** (7.4 - pka))
    else:
        fi = 0.0

    fa_nasal_ionized = fa_nasal_base * (1.0 - 0.5 * fi)

    # Deposition fraction by formulation
    dep_frac = _DEPOSITION[formulation]

    # Actual absorbed fraction
    fa_actual = fa_nasal_ionized * dep_frac

    # Hepatic first-pass (well-stirred model)
    e_h = cl_hepatic_L_per_h / q_hepatic_L_per_h
    e_h = max(0.0, min(1.0, e_h))
    f_hepatic = 1.0 - e_h

    # Overall systemic bioavailability
    f_systemic = fa_actual * f_hepatic
    f_systemic = max(0.0, min(1.0, f_systemic))

    # Time to peak absorption
    k_total = k_abs + _K_MCC
    t_max_absorption_h = 1.0 / k_total

    # Formulation factor
    form_factor = _FORMULATION_FACTOR[formulation]

    # Bioavailability class
    if f_systemic > 0.5:
        bio_class = "high"
    elif f_systemic > 0.2:
        bio_class = "moderate"
    else:
        bio_class = "low"

    # Notes
    notes_parts = []
    if fi > 0.5:
        notes_parts.append(f"Highly ionized at nasal pH (fi={fi:.2f}), reducing absorption")
    if mw_Da > 500:
        notes_parts.append(f"High MW ({mw_Da} Da) may limit nasal permeability")
    if dep_frac < 0.75:
        notes_parts.append(f"Formulation '{formulation}' has lower deposition ({dep_frac:.0%})")
    if e_h > 0.7:
        notes_parts.append(f"High hepatic extraction (E_h={e_h:.2f}) limits systemic exposure")
    if not notes_parts:
        notes_parts.append(
            f"k_abs={k_abs:.3f}/h, fa_nasal_base={fa_nasal_base:.3f}, f_systemic={f_systemic:.3f}"
        )

    return NasalBioavailabilityResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logp=logp,
        fa_nasal=fa_nasal_ionized,
        f_hepatic=f_hepatic,
        f_systemic=f_systemic,
        t_max_absorption_h=t_max_absorption_h,
        cl_mucociliary_per_h=_K_MCC,
        deposition_fraction=dep_frac,
        formulation_factor=form_factor,
        bioavailability_class=bio_class,
        notes="; ".join(notes_parts),
    )


def compare_formulations(
    drug_name: str,
    mw_Da: float,
    logp: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
) -> list:
    """Compare all 4 formulations, sorted by f_systemic descending."""
    results = []
    for form in _VALID_FORMULATIONS:
        result = predict_nasal_bioavailability(
            drug_name=drug_name,
            mw_Da=mw_Da,
            logp=logp,
            pka=pka,
            ionization_type=ionization_type,
            formulation=form,
        )
        results.append(result)

    results.sort(key=lambda r: r.f_systemic, reverse=True)
    return results
