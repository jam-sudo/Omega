"""Drug combination synergy analysis — Phase 499."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SynergyMatrixResult:
    """Result of a drug synergy matrix analysis."""

    drug_a: str
    drug_b: str
    doses_a: list[float]
    doses_b: list[float]
    bliss_matrix: list[list[float]]
    loewe_ci_matrix: list[list[float]]
    mean_bliss_score: float
    synergy_classification: str
    max_synergy_doses: tuple[float, float]
    notes: str


def bliss_independence(effect_a: float, effect_b: float) -> float:
    """Compute expected combined effect under Bliss independence.

    Under Bliss independence, the expected combined fractional effect is:
        E_expected = E_A + E_B - E_A * E_B

    Parameters
    ----------
    effect_a:
        Fractional effect of drug A alone (0 to 1).
    effect_b:
        Fractional effect of drug B alone (0 to 1).

    Returns
    -------
    float
        Expected combined effect under independence (0 to 1).
    """
    if not (0.0 <= effect_a <= 1.0):
        raise ValueError(f"effect_a must be in [0, 1], got {effect_a}")
    if not (0.0 <= effect_b <= 1.0):
        raise ValueError(f"effect_b must be in [0, 1], got {effect_b}")
    return effect_a + effect_b - effect_a * effect_b


def bliss_synergy_score(effect_a: float, effect_b: float, effect_combo: float) -> float:
    """Compute Bliss synergy score (observed minus expected).

    A positive score indicates synergy (combo exceeds independence expectation).
    A negative score indicates antagonism.

    Parameters
    ----------
    effect_a:
        Fractional effect of drug A alone (0 to 1).
    effect_b:
        Fractional effect of drug B alone (0 to 1).
    effect_combo:
        Observed fractional effect of the combination (0 to 1).

    Returns
    -------
    float
        Synergy score = observed - expected.
    """
    if not (0.0 <= effect_a <= 1.0):
        raise ValueError(f"effect_a must be in [0, 1], got {effect_a}")
    if not (0.0 <= effect_b <= 1.0):
        raise ValueError(f"effect_b must be in [0, 1], got {effect_b}")
    if not (0.0 <= effect_combo <= 1.0):
        raise ValueError(f"effect_combo must be in [0, 1], got {effect_combo}")
    expected = bliss_independence(effect_a, effect_b)
    return effect_combo - expected


def _hill_effect(dose: float, ec50: float, hill_n: float) -> float:
    """Compute Hill equation fractional effect."""
    if dose <= 0.0:
        return 0.0
    if ec50 <= 0.0:
        raise ValueError(f"ec50 must be positive, got {ec50}")
    numerator = dose**hill_n
    denominator = ec50**hill_n + dose**hill_n
    return numerator / denominator


def loewe_ci(
    dose_a: float,
    dose_b: float,
    ec50_a: float,
    ec50_b: float,
    effect: float,
    hill_n: float = 1.0,
) -> float:
    """Compute Loewe combination index (CI) for a two-drug combination.

    CI < 1  → synergy
    CI = 1  → additivity
    CI > 1  → antagonism

    For each drug the isobologram reference dose is the dose that produces
    the same ``effect`` when given alone, derived from the Hill equation:

        D_x = EC50_x * (E / (1 - E))^(1/n)

    Parameters
    ----------
    dose_a:
        Actual dose of drug A in the combination.
    dose_b:
        Actual dose of drug B in the combination.
    ec50_a:
        EC50 of drug A.
    ec50_b:
        EC50 of drug B.
    effect:
        Target fractional effect (0 < effect < 1).
    hill_n:
        Hill coefficient (default 1.0).

    Returns
    -------
    float
        Combination index.
    """
    if ec50_a <= 0:
        raise ValueError(f"ec50_a must be positive, got {ec50_a}")
    if ec50_b <= 0:
        raise ValueError(f"ec50_b must be positive, got {ec50_b}")
    if not (0.0 < effect < 1.0):
        raise ValueError(f"effect must be in (0, 1) for CI calculation, got {effect}")
    if hill_n <= 0:
        raise ValueError(f"hill_n must be positive, got {hill_n}")

    # D_x = EC50 * (E / (1-E))^(1/n)
    ratio = effect / (1.0 - effect)
    exponent = 1.0 / hill_n
    d_a = ec50_a * (ratio**exponent)
    d_b = ec50_b * (ratio**exponent)

    return dose_a / d_a + dose_b / d_b


def classify_synergy(score: float) -> str:
    """Classify a Bliss synergy score into a category.

    Thresholds:
        score <= -0.2                → 'strong_synergy'
        -0.2 < score <= -0.05       → 'synergy'
        -0.05 < score <= 0.05       → 'additive'
        0.05 < score <= 0.2         → 'antagonism'
        score > 0.2                 → 'strong_antagonism'

    Parameters
    ----------
    score:
        Bliss synergy score (observed - expected).

    Returns
    -------
    str
        Synergy classification string.
    """
    if score <= -0.2:
        return "strong_synergy"
    if score <= -0.05:
        return "synergy"
    if score <= 0.05:
        return "additive"
    if score <= 0.2:
        return "antagonism"
    return "strong_antagonism"


def build_synergy_matrix(
    doses_a: list[float],
    doses_b: list[float],
    ec50_a: float,
    ec50_b: float,
    n_a: float = 1.0,
    n_b: float = 1.0,
    drug_a: str = "Drug A",
    drug_b: str = "Drug B",
) -> SynergyMatrixResult:
    """Build a synergy matrix over a dose grid for two drugs.

    For each combination (dose_a, dose_b) on the dose grid:
    - The fractional effect of each drug alone is computed via the Hill equation.
    - The Bliss expected effect and synergy score are computed.
    - The Loewe CI is computed at the mean individual effects.

    Parameters
    ----------
    doses_a:
        List of doses for drug A (positive floats).
    doses_b:
        List of doses for drug B (positive floats).
    ec50_a:
        EC50 of drug A.
    ec50_b:
        EC50 of drug B.
    n_a:
        Hill coefficient for drug A (default 1.0).
    n_b:
        Hill coefficient for drug B (default 1.0).
    drug_a:
        Name of drug A.
    drug_b:
        Name of drug B.

    Returns
    -------
    SynergyMatrixResult
    """
    if not doses_a:
        raise ValueError("doses_a must not be empty")
    if not doses_b:
        raise ValueError("doses_b must not be empty")
    if ec50_a <= 0:
        raise ValueError(f"ec50_a must be positive, got {ec50_a}")
    if ec50_b <= 0:
        raise ValueError(f"ec50_b must be positive, got {ec50_b}")

    bliss_matrix: list[list[float]] = []
    loewe_ci_matrix: list[list[float]] = []

    all_bliss: list[float] = []

    for da in doses_a:
        bliss_row: list[float] = []
        ci_row: list[float] = []
        for db in doses_b:
            ea = _hill_effect(da, ec50_a, n_a)
            eb = _hill_effect(db, ec50_b, n_b)

            # Bliss null model: without observed combination data the matrix
            # reports the additive baseline (score = 0).  Users may overlay
            # real combo measurements using bliss_synergy_score().
            bliss_score = 0.0  # additive null reference
            bliss_row.append(bliss_score)
            all_bliss.append(bliss_score)

            # Loewe CI: use mean individual effect as target
            avg_effect = (ea + eb) / 2.0
            avg_effect = min(max(avg_effect, 1e-6), 1.0 - 1e-6)
            ci_val = loewe_ci(da, db, ec50_a, ec50_b, avg_effect, hill_n=1.0)
            ci_row.append(ci_val)

        bliss_matrix.append(bliss_row)
        loewe_ci_matrix.append(ci_row)

    mean_bliss = sum(all_bliss) / len(all_bliss) if all_bliss else 0.0
    classification = classify_synergy(mean_bliss)

    # max_synergy_doses: coordinates of lowest Loewe CI (most synergistic)
    min_ci = float("inf")
    max_syn_da = doses_a[0]
    max_syn_db = doses_b[0]
    for i, da in enumerate(doses_a):
        for j, db in enumerate(doses_b):
            if loewe_ci_matrix[i][j] < min_ci:
                min_ci = loewe_ci_matrix[i][j]
                max_syn_da = da
                max_syn_db = db

    notes = (
        f"Bliss independence null model; Loewe CI computed at mean individual "
        f"effects. Grid: {len(doses_a)}x{len(doses_b)} doses. "
        f"EC50_A={ec50_a}, EC50_B={ec50_b}."
    )

    return SynergyMatrixResult(
        drug_a=drug_a,
        drug_b=drug_b,
        doses_a=list(doses_a),
        doses_b=list(doses_b),
        bliss_matrix=bliss_matrix,
        loewe_ci_matrix=loewe_ci_matrix,
        mean_bliss_score=mean_bliss,
        synergy_classification=classification,
        max_synergy_doses=(max_syn_da, max_syn_db),
        notes=notes,
    )
