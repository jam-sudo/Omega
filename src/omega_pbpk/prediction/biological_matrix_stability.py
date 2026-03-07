"""Drug stability prediction in biological matrices.

Predicts apparent half-life, percent remaining, and degradation pathway
for drugs in plasma, blood, urine, liver microsomes, and brain homogenate.

Models are simplified empirical correlations suitable for early-stage
stability screening.  Matrix-specific esterase/oxidase activities and
temperature dependence are approximated with Q10-style Arrhenius correction.

References
----------
- Stringer RA et al., Drug Metab Dispos. 2009;37(2):301-5 (plasma esterase)
- Obach RS, Drug Metab Dispos. 1999;27(11):1350-9 (microsomal stability)
- Naritomi Y et al., Drug Metab Dispos. 2001;29(10):1316-24 (CLint scaling)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BiologicalMatrixStabilityResult:
    """Stability prediction in a single biological matrix.

    Attributes
    ----------
    drug_name : Compound identifier.
    matrix : Biological matrix name.
    temperature_celsius : Incubation temperature (°C).
    t_half_h : Apparent first-order half-life (h).
    percent_remaining_1h : Percent parent remaining at 1 h.
    percent_remaining_4h : Percent parent remaining at 4 h.
    percent_remaining_24h : Percent parent remaining at 24 h.
    primary_degradation_pathway : Dominant degradation route.
    matrix_binding_pct : Estimated non-specific matrix binding (%).
    analyte_stability : "stable" (>80% at 24h), "moderate" (>50%), "labile" (<50%).
    back_conversion : True if ester hydrolysis may regenerate parent.
    recommended_storage : Practical sample handling guidance.
    notes : Human-readable summary.
    """

    drug_name: str
    matrix: str
    temperature_celsius: float
    t_half_h: float
    percent_remaining_1h: float
    percent_remaining_4h: float
    percent_remaining_24h: float
    primary_degradation_pathway: str
    matrix_binding_pct: float
    analyte_stability: str
    back_conversion: bool
    recommended_storage: str
    notes: str


_VALID_MATRICES = frozenset({"plasma", "blood", "urine", "liver_microsomes", "brain_homogenate"})


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _base_kdeg_plasma(has_ester: bool, has_amide: bool) -> tuple[float, str]:
    """Return (k_base_per_h, primary_pathway) for plasma."""
    k = 0.005 * (2.0 if has_ester else 1.0) * (1.5 if has_amide else 1.0)
    if has_ester:
        pathway = "esterase"
    elif has_amide:
        pathway = "hydrolysis"
    else:
        pathway = "stable"
    return k, pathway


def _base_kdeg(
    matrix: str,
    logp: float,
    mw_da: float,
    pka: float,
    has_ester: bool,
    has_amide: bool,
    has_lactam: bool,
) -> tuple[float, str]:
    """Return (k_base_per_h, primary_pathway) for the given matrix at 37°C."""
    if matrix == "plasma":
        return _base_kdeg_plasma(has_ester, has_amide)

    if matrix == "blood":
        k_plasma, _ = _base_kdeg_plasma(has_ester, has_amide)
        k = k_plasma * 1.3
        pathway = "esterase" if has_ester else "hydrolysis" if has_amide else "stable"
        return k, pathway

    if matrix == "urine":
        k = 0.001 * (1.0 + abs(pka - 6.0) / 5.0)
        return k, "hydrolysis"

    if matrix == "liver_microsomes":
        k = 0.02 * (1.0 + logp / 10.0) * (2.0 if has_lactam else 1.0)
        return k, "oxidase"

    if matrix == "brain_homogenate":
        k = 0.01 * (1.0 + logp / 5.0) * (1.5 if has_ester else 1.0)
        if mw_da > 500:
            pathway = "protease"
        elif has_ester:
            pathway = "esterase"
        else:
            pathway = "oxidase"
        return k, pathway

    raise ValueError(f"Unknown matrix: {matrix!r}. Valid: {sorted(_VALID_MATRICES)}")


def _matrix_binding_pct(matrix: str, logp: float) -> float:
    """Estimate non-specific matrix binding (%)."""
    if matrix == "plasma":
        raw = 10.0 + 20.0 * logp * 0.3
        return _clamp(raw, 1.0, 95.0)
    if matrix == "blood":
        raw = (10.0 + 20.0 * logp * 0.3) * 1.2
        return _clamp(raw, 1.0, 95.0)
    if matrix == "urine":
        return 5.0
    if matrix == "liver_microsomes":
        raw = 30.0 + logp * 5.0
        return _clamp(raw, 1.0, 95.0)
    if matrix == "brain_homogenate":
        raw = 40.0 + logp * 10.0
        return _clamp(raw, 10.0, 95.0)
    return 10.0


def predict_matrix_stability(
    drug_name: str,
    logp: float,
    mw_da: float,
    pka: float,
    has_ester: bool,
    has_amide: bool,
    has_lactam: bool,
    matrix: str,
    temperature_celsius: float,
) -> BiologicalMatrixStabilityResult:
    """Predict drug stability in a biological matrix.

    Parameters
    ----------
    drug_name : Compound identifier.
    logp : Octanol-water partition coefficient.
    mw_da : Molecular weight (Da).
    pka : Ionisation pKa.
    has_ester : Contains an ester functional group.
    has_amide : Contains an amide functional group.
    has_lactam : Contains a lactam (cyclic amide) functional group.
    matrix : One of "plasma", "blood", "urine", "liver_microsomes", "brain_homogenate".
    temperature_celsius : Incubation temperature (°C).

    Returns
    -------
    BiologicalMatrixStabilityResult

    Raises
    ------
    ValueError
        If temperature_celsius is outside [-80, 100] or matrix is unknown.
    """
    if temperature_celsius < -80 or temperature_celsius > 100:
        raise ValueError(f"temperature_celsius must be in [-80, 100], got {temperature_celsius}")
    if matrix not in _VALID_MATRICES:
        raise ValueError(f"Unknown matrix: {matrix!r}. Valid: {sorted(_VALID_MATRICES)}")

    k_base, primary_pathway = _base_kdeg(matrix, logp, mw_da, pka, has_ester, has_amide, has_lactam)

    # Temperature correction: Q10 ~ 2 approximation
    k = k_base * math.exp(0.07 * (temperature_celsius - 37.0))
    k = _clamp(k, 1e-6, 1.0)

    t_half_h = math.log(2.0) / k

    pct_1h = math.exp(-k * 1.0) * 100.0
    pct_4h = math.exp(-k * 4.0) * 100.0
    pct_24h = math.exp(-k * 24.0) * 100.0

    binding_pct = _matrix_binding_pct(matrix, logp)

    if pct_24h > 80.0:
        analyte_stability = "stable"
    elif pct_24h > 50.0:
        analyte_stability = "moderate"
    else:
        analyte_stability = "labile"

    back_conversion = has_ester and matrix in ("plasma", "blood")

    if t_half_h > 4.0:
        recommended_storage = "stable on ice (0-4°C)"
    elif t_half_h < 1.0:
        recommended_storage = "freeze immediately (-80°C)"
    else:
        recommended_storage = "process within 2h on ice"

    notes = (
        f"Matrix: {matrix} at {temperature_celsius}°C. "
        f"t½={t_half_h:.2f}h. "
        f"Remaining: 1h={pct_1h:.1f}%, 4h={pct_4h:.1f}%, 24h={pct_24h:.1f}%. "
        f"Stability: {analyte_stability}. Pathway: {primary_pathway}."
    )

    return BiologicalMatrixStabilityResult(
        drug_name=drug_name,
        matrix=matrix,
        temperature_celsius=temperature_celsius,
        t_half_h=t_half_h,
        percent_remaining_1h=pct_1h,
        percent_remaining_4h=pct_4h,
        percent_remaining_24h=pct_24h,
        primary_degradation_pathway=primary_pathway,
        matrix_binding_pct=binding_pct,
        analyte_stability=analyte_stability,
        back_conversion=back_conversion,
        recommended_storage=recommended_storage,
        notes=notes,
    )


def compare_matrices(
    drug_name: str,
    logp: float,
    mw_da: float,
    pka: float,
    has_ester: bool,
    has_amide: bool,
    has_lactam: bool,
) -> list[BiologicalMatrixStabilityResult]:
    """Compare drug stability across all five biological matrices at 37°C.

    Parameters
    ----------
    drug_name : Compound identifier.
    logp : Octanol-water partition coefficient.
    mw_da : Molecular weight (Da).
    pka : Ionisation pKa.
    has_ester : Contains an ester group.
    has_amide : Contains an amide group.
    has_lactam : Contains a lactam group.

    Returns
    -------
    List of BiologicalMatrixStabilityResult for all matrices, sorted by
    t_half_h descending (most stable first).
    """
    results = [
        predict_matrix_stability(
            drug_name=drug_name,
            logp=logp,
            mw_da=mw_da,
            pka=pka,
            has_ester=has_ester,
            has_amide=has_amide,
            has_lactam=has_lactam,
            matrix=m,
            temperature_celsius=37.0,
        )
        for m in sorted(_VALID_MATRICES)
    ]
    results.sort(key=lambda r: r.t_half_h, reverse=True)
    return results
