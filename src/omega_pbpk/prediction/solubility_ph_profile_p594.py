"""Drug solubility pH profile prediction (Phase 594).

Predicts drug solubility as a function of pH based on the
Henderson-Hasselbalch equation for ionizable drugs.

Equations
---------
Acid   (pKa1):     S(pH) = S0 * (1 + 10^(pH - pKa1))
Base   (pKa1):     S(pH) = S0 * (1 + 10^(pKa1 - pH))
Amphoteric:        S(pH) = S0 * (1 + 10^(pH - pKa1) + 10^(pKa2 - pH))
Neutral:           S(pH) = S0

References
----------
- Henderson L, Am J Physiol. 1908;21:173-179 (Henderson-Hasselbalch equation)
- Avdeef A, Absorption and Drug Development. 2003, Wiley-Interscience.
- Bergstrom CAS et al., J Med Chem. 2003;46(4):558-570 (solubility-pH)
- Bergstrom CAS, Basic Clin Pharmacol. 2005;96(3):156-161 (BCS solubility)
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VALID_DRUG_TYPES = {"acid", "base", "amphoteric", "neutral"}
_SOL_CAP_MG_ML = 1000.0  # maximum solubility cap (mg/mL)
_PH_FASTED_GI = 6.5  # fasted small intestine pH
_PH_FED_GI = 5.0  # fed stomach/SI pH
_PH_GASTRIC = 1.2  # gastric pH
_BCS_HIGH_THRESHOLD = 1.0  # mg/mL threshold for BCS high solubility


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SolubilityPHProfileResult:
    """Results from solubility-pH profile prediction.

    Attributes
    ----------
    drug_name : Name of the drug.
    drug_type : Ionization type ('acid', 'base', 'amphoteric', 'neutral').
    pka_values : List of pKa values used (1 or 2 for ionizable; empty for neutral).
    ph_values : pH points evaluated.
    solubility_mg_mL : Predicted solubility at each pH (mg/mL).
    s0_mg_mL : Intrinsic solubility of unionized form (mg/mL).
    ph_max_solubility : pH at which solubility is highest.
    solubility_at_ph_max : Maximum solubility value (mg/mL).
    solubility_fasted_gi : Solubility at pH 6.5 (fasted small intestine) (mg/mL).
    solubility_fed_gi : Solubility at pH 5.0 (fed stomach/SI) (mg/mL).
    solubility_gastric : Solubility at pH 1.2 (gastric) (mg/mL).
    bcs_solubility_class : 'high' if max(fasted_gi, fed_gi) >= 1 mg/mL else 'low'.
    notes : Summary text.
    """

    drug_name: str
    drug_type: str
    pka_values: list
    ph_values: list
    solubility_mg_mL: list
    s0_mg_mL: float
    ph_max_solubility: float
    solubility_at_ph_max: float
    solubility_fasted_gi: float
    solubility_fed_gi: float
    solubility_gastric: float
    bcs_solubility_class: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _solubility_at_ph(
    ph: float,
    s0: float,
    drug_type: str,
    pka1: float | None,
    pka2: float | None,
) -> float:
    """Compute solubility at a single pH value."""
    if drug_type == "neutral":
        sol = s0
    elif drug_type == "acid":
        sol = s0 * (1.0 + 10.0 ** (ph - pka1))
    elif drug_type == "base":
        sol = s0 * (1.0 + 10.0 ** (pka1 - ph))
    elif drug_type == "amphoteric":
        sol = s0 * (1.0 + 10.0 ** (ph - pka1) + 10.0 ** (pka2 - ph))
    else:
        sol = s0

    return min(sol, _SOL_CAP_MG_ML)


def _linspace(start: float, stop: float, n: int) -> list:
    """Generate n evenly spaced points from start to stop inclusive."""
    if n <= 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------


def predict_solubility_ph_profile(
    drug_name: str,
    s0_mg_mL: float,
    drug_type: str,
    pka1: float | None = None,
    pka2: float | None = None,
    ph_range: tuple = (1.0, 10.0),
    n_points: int = 37,
) -> SolubilityPHProfileResult:
    """Predict solubility-pH profile using Henderson-Hasselbalch equations.

    Parameters
    ----------
    drug_name : Drug identifier.
    s0_mg_mL : Intrinsic solubility of unionized form (mg/mL).
    drug_type : Ionization type: 'acid', 'base', 'amphoteric', or 'neutral'.
    pka1 : First pKa value. Required for 'acid', 'base', and 'amphoteric'.
           For amphoteric: pKa of the acidic group.
    pka2 : Second pKa value. Required for 'amphoteric'.
           For amphoteric: pKa of the basic group.
    ph_range : (min_pH, max_pH) range tuple.
    n_points : Number of pH points to evaluate.

    Returns
    -------
    SolubilityPHProfileResult
    """
    # --- Validation ---
    if s0_mg_mL <= 0:
        raise ValueError(f"s0_mg_mL must be positive, got {s0_mg_mL}")
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {_VALID_DRUG_TYPES}, got '{drug_type}'")
    if drug_type in ("acid", "base") and pka1 is None:
        raise ValueError(f"pka1 is required for drug_type='{drug_type}'")
    if drug_type == "amphoteric" and (pka1 is None or pka2 is None):
        raise ValueError("Both pka1 and pka2 are required for drug_type='amphoteric'")

    # --- Build pH grid ---
    ph_step = (ph_range[1] - ph_range[0]) / (n_points - 1)
    ph_values = [ph_range[0] + i * ph_step for i in range(n_points)]

    # --- Compute solubility at each pH ---
    solubility_mg_mL = [_solubility_at_ph(ph, s0_mg_mL, drug_type, pka1, pka2) for ph in ph_values]

    # --- Key metrics ---
    max_sol = max(solubility_mg_mL)
    idx_max = solubility_mg_mL.index(max_sol)
    ph_max_solubility = ph_values[idx_max]

    # Solubility at key GI pH values (direct computation for accuracy)
    solubility_fasted_gi = _solubility_at_ph(_PH_FASTED_GI, s0_mg_mL, drug_type, pka1, pka2)
    solubility_fed_gi = _solubility_at_ph(_PH_FED_GI, s0_mg_mL, drug_type, pka1, pka2)
    solubility_gastric = _solubility_at_ph(_PH_GASTRIC, s0_mg_mL, drug_type, pka1, pka2)

    # BCS solubility class
    gi_max = max(solubility_fasted_gi, solubility_fed_gi)
    bcs_solubility_class = "high" if gi_max >= _BCS_HIGH_THRESHOLD else "low"

    # pKa summary
    if drug_type == "neutral":
        pka_values = []
        pka_note = "neutral"
    elif drug_type == "amphoteric":
        pka_values = [pka1, pka2]
        pka_note = f"pKa(acid)={pka1}, pKa(base)={pka2}"
    elif pka1 is not None:
        pka_values = [pka1]
        pka_note = f"pKa={pka1}"
    else:
        pka_values = []
        pka_note = "neutral"

    notes = (
        f"{drug_type.capitalize()} drug ({pka_note}): "
        f"S0={s0_mg_mL:.3f} mg/mL, "
        f"S_max={max_sol:.3f} mg/mL at pH={ph_max_solubility:.2f}, "
        f"BCS={bcs_solubility_class}"
    )

    return SolubilityPHProfileResult(
        drug_name=drug_name,
        drug_type=drug_type,
        pka_values=pka_values,
        ph_values=ph_values,
        solubility_mg_mL=solubility_mg_mL,
        s0_mg_mL=s0_mg_mL,
        ph_max_solubility=ph_max_solubility,
        solubility_at_ph_max=max_sol,
        solubility_fasted_gi=solubility_fasted_gi,
        solubility_fed_gi=solubility_fed_gi,
        solubility_gastric=solubility_gastric,
        bcs_solubility_class=bcs_solubility_class,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Compound screening
# ---------------------------------------------------------------------------


def screen_solubility_ph(compounds: list) -> list:
    """Screen a list of compounds for solubility-pH profiles.

    Parameters
    ----------
    compounds : List of dicts, each with keys matching
                predict_solubility_ph_profile parameters:
                - drug_name (str)
                - s0_mg_mL (float)
                - drug_type (str)
                - pka1 (float, optional)
                - pka2 (float, optional)
                - ph_range (tuple, optional)
                - n_points (int, optional)

    Returns
    -------
    list of SolubilityPHProfileResult sorted by solubility_fasted_gi descending.
    """
    results = []
    for compound in compounds:
        result = predict_solubility_ph_profile(
            drug_name=compound["drug_name"],
            s0_mg_mL=compound["s0_mg_mL"],
            drug_type=compound["drug_type"],
            pka1=compound.get("pka1"),
            pka2=compound.get("pka2"),
            ph_range=compound.get("ph_range", (1.0, 10.0)),
            n_points=compound.get("n_points", 37),
        )
        results.append(result)

    results.sort(key=lambda r: r.solubility_fasted_gi, reverse=True)
    return results
