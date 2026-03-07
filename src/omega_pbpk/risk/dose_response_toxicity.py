"""Phase 668 — Dose-response toxicity modeling using Hill equation and threshold models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToxicityDoseResponseResult:
    compound_name: str
    toxic_endpoint: str  # e.g., "hepatotoxicity"
    noael_mg_kg: float  # no-observed-adverse-effect level
    loael_mg_kg: float  # lowest observed adverse effect level
    td50_mg_kg: float  # dose causing 50% toxicity incidence
    safety_factor: float  # uncertainty_factor * (noael / human_dose)
    predicted_incidence_at_dose: float  # predicted % incidence at human_dose (0-1)
    hill_n: float  # Hill coefficient
    dose_margin: float  # NOAEL / human_dose_mg_kg
    benchmark_dose_10_mg_kg: float  # BMD10: dose for 10% response above background
    risk_category: str  # "acceptable","marginal","concerning","unacceptable"
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    noael_mg_kg: float,
    loael_mg_kg: float,
    td50_mg_kg: float,
    human_dose_mg_kg: float,
    background_incidence: float,
    hill_n: float,
    uncertainty_factor: float,
) -> None:
    if noael_mg_kg <= 0:
        raise ValueError(f"noael_mg_kg must be > 0, got {noael_mg_kg}")
    if loael_mg_kg <= noael_mg_kg:
        raise ValueError(
            f"loael_mg_kg must be > noael_mg_kg; got loael={loael_mg_kg}, noael={noael_mg_kg}"
        )
    if loael_mg_kg > td50_mg_kg:
        raise ValueError(
            f"loael_mg_kg must be <= td50_mg_kg; got loael={loael_mg_kg}, td50={td50_mg_kg}"
        )
    if human_dose_mg_kg <= 0:
        raise ValueError(f"human_dose_mg_kg must be > 0, got {human_dose_mg_kg}")
    if not (0 < background_incidence < 1):
        raise ValueError(f"background_incidence must be in (0, 1), got {background_incidence}")
    if hill_n <= 0:
        raise ValueError(f"hill_n must be > 0, got {hill_n}")
    if uncertainty_factor <= 0:
        raise ValueError(f"uncertainty_factor must be > 0, got {uncertainty_factor}")


def _hill_response(dose: float, td50: float, hill_n: float, background: float) -> float:
    """E(D) = background + (1 - background) * D^n / (TD50^n + D^n)."""
    if dose <= 0:
        return background
    d_n = dose**hill_n
    td50_n = td50**hill_n
    return background + (1.0 - background) * d_n / (td50_n + d_n)


def _bmd10(td50: float, hill_n: float, background: float) -> float:
    """BMD10: dose at which response is 10% above background.

    D^n / (TD50^n + D^n) = 0.1 / (1 - background)
    → D = TD50 * (r / (1 - r))^(1/n)  where r = 0.1 / (1 - background)
    """
    r = 0.1 / (1.0 - background)
    if r >= 1.0:
        return td50  # degenerate; return td50
    return td50 * (r / (1.0 - r)) ** (1.0 / hill_n)


def _risk_category(dose_margin: float) -> str:
    if dose_margin > 10:
        return "acceptable"
    elif dose_margin >= 5:
        return "marginal"
    elif dose_margin >= 2:
        return "concerning"
    else:
        return "unacceptable"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def model_toxicity_dose_response(
    compound_name: str,
    toxic_endpoint: str,
    noael_mg_kg: float,
    loael_mg_kg: float,
    td50_mg_kg: float,
    human_dose_mg_kg: float = 1.0,
    background_incidence: float = 0.01,
    hill_n: float = 2.0,
    species: str = "rodent",
    uncertainty_factor: float = 100.0,
) -> ToxicityDoseResponseResult:
    """Model toxicity dose-response using Hill equation and derive safety metrics."""
    _validate_inputs(
        noael_mg_kg,
        loael_mg_kg,
        td50_mg_kg,
        human_dose_mg_kg,
        background_incidence,
        hill_n,
        uncertainty_factor,
    )

    predicted_incidence = _hill_response(human_dose_mg_kg, td50_mg_kg, hill_n, background_incidence)
    dose_margin = noael_mg_kg / human_dose_mg_kg
    safety_factor = uncertainty_factor * dose_margin
    bmd10 = _bmd10(td50_mg_kg, hill_n, background_incidence)
    category = _risk_category(dose_margin)

    notes = (
        f"Species: {species}. Hill n={hill_n:.2f}. "
        f"NOAEL/LOAEL/TD50={noael_mg_kg}/{loael_mg_kg}/{td50_mg_kg} mg/kg. "
        f"Dose margin={dose_margin:.2f}, BMD10={bmd10:.3f} mg/kg. "
        f"Uncertainty factor={uncertainty_factor}. Risk: {category}."
    )

    return ToxicityDoseResponseResult(
        compound_name=compound_name,
        toxic_endpoint=toxic_endpoint,
        noael_mg_kg=noael_mg_kg,
        loael_mg_kg=loael_mg_kg,
        td50_mg_kg=td50_mg_kg,
        safety_factor=safety_factor,
        predicted_incidence_at_dose=predicted_incidence,
        hill_n=hill_n,
        dose_margin=dose_margin,
        benchmark_dose_10_mg_kg=bmd10,
        risk_category=category,
        notes=notes,
    )


def compute_dose_response_curve(
    td50: float,
    hill_n: float,
    doses: list[float],
    background: float = 0.01,
) -> list[float]:
    """Return Hill dose-response incidence curve (0-1) for each dose."""
    return [_hill_response(d, td50, hill_n, background) for d in doses]


def screen_toxicity_endpoints(
    compound_name: str,
    endpoints: list[dict],
    human_dose_mg_kg: float = 1.0,
) -> list[ToxicityDoseResponseResult]:
    """Screen all endpoints, sorted by dose_margin ascending (most concerning first)."""
    results: list[ToxicityDoseResponseResult] = []
    for ep in endpoints:
        result = model_toxicity_dose_response(
            compound_name=compound_name,
            toxic_endpoint=ep["endpoint"],
            noael_mg_kg=ep["noael"],
            loael_mg_kg=ep["loael"],
            td50_mg_kg=ep["td50"],
            human_dose_mg_kg=human_dose_mg_kg,
        )
        results.append(result)
    results.sort(key=lambda r: r.dose_margin)
    return results
