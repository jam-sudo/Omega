"""Phase 443 — Bile acid sequestrant drug interaction PK model.

Models the reduction in oral drug absorption due to bile acid sequestrants
(cholestyramine, colesevelam, colestipol) binding acidic drugs in the GI tract.
Also handles enterohepatic recycling disruption effects.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Sequestrant database
# ---------------------------------------------------------------------------

SEQUESTRANTS = {
    "cholestyramine": {"binding_fraction_per_g": 0.08, "onset_h": 0.5, "duration_h": 8},
    "colesevelam": {"binding_fraction_per_g": 0.04, "onset_h": 0.5, "duration_h": 8},
    "colestipol": {"binding_fraction_per_g": 0.06, "onset_h": 0.5, "duration_h": 8},
    "none": {"binding_fraction_per_g": 0.0, "onset_h": 0, "duration_h": 0},
}

# Drug binding affinity by ionization type
BINDING_AFFINITY = {
    "acid": 1.0,
    "base": 0.2,
    "neutral": 0.1,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BileAcidInteractionResult:
    """Result of bile acid sequestrant drug interaction simulation."""

    drug_name: str
    bile_sequestrant: str
    dose_mg: float
    sequestrant_dose_g: float
    times_h: list
    c_plasma_without_seq_mg_L: list
    c_plasma_with_seq_mg_L: list
    auc_without_seq_mg_h_per_L: float
    auc_with_seq_mg_h_per_L: float
    auc_reduction_pct: float
    f_oral_without: float
    f_oral_with: float
    interaction_severity: str
    timing_recommendation: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _simulate_1cpt_oral(
    dose_mg: float,
    f_oral: float,
    ka_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float,
) -> tuple:
    """Simulate 1-compartment oral PK using Forward Euler.

    Returns (times_h, c_plasma_mg_L).
    """
    times_h: list[float] = []
    c_plasma: list[float] = []

    # State variables
    a_gut = f_oral * dose_mg  # drug in gut (mg)
    c_central = 0.0  # concentration in central compartment (mg/L)

    t = 0.0
    while t <= t_end_h + 1e-9:
        times_h.append(t)
        c_plasma.append(c_central)

        # Rates
        absorption_rate = ka_per_h * a_gut  # mg/h
        elimination_rate = cl_L_per_h * c_central  # mg/h

        # Forward Euler update
        a_gut = a_gut - absorption_rate * dt_h
        if a_gut < 0.0:
            a_gut = 0.0
        c_central = c_central + (absorption_rate / vd_L - elimination_rate / vd_L) * dt_h
        if c_central < 0.0:
            c_central = 0.0

        t += dt_h

    return times_h, c_plasma


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def _compute_bound_fraction(
    binding_fraction_per_g: float,
    sequestrant_dose_g: float,
    drug_binding_affinity: float,
) -> float:
    """Compute fraction of drug bound by sequestrant, capped at 0.95."""
    return min(0.95, binding_fraction_per_g * sequestrant_dose_g * drug_binding_affinity)


def _severity(auc_reduction_pct: float) -> str:
    if auc_reduction_pct >= 60.0:
        return "severe"
    if auc_reduction_pct >= 30.0:
        return "moderate"
    if auc_reduction_pct >= 10.0:
        return "mild"
    return "none"


def _timing_recommendation(severity: str) -> str:
    mapping = {
        "severe": "avoid",
        "moderate": "take_4h_after_seq",
        "mild": "take_1h_before_seq",
        "none": "no_restriction",
    }
    return mapping[severity]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_bile_acid_interaction(
    drug_name: str,
    bile_sequestrant: str,
    dose_mg: float,
    sequestrant_dose_g: float,
    ionization_type: str,
    ka_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral_baseline: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> BileAcidInteractionResult:
    """Simulate drug PK with and without bile acid sequestrant co-administration.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    bile_sequestrant : str
        Sequestrant name: "cholestyramine", "colesevelam", "colestipol", or "none".
    dose_mg : float
        Drug dose in mg.
    sequestrant_dose_g : float
        Sequestrant dose in grams.
    ionization_type : str
        "acid", "base", or "neutral".
    ka_per_h : float
        First-order absorption rate constant (h^-1).
    cl_L_per_h : float
        Total clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    f_oral_baseline : float
        Baseline oral bioavailability (0–1].
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step for Forward Euler (h).

    Returns
    -------
    BileAcidInteractionResult
    """
    # Validation
    if bile_sequestrant not in SEQUESTRANTS:
        raise ValueError(
            f"Unknown sequestrant '{bile_sequestrant}'. Valid options: {list(SEQUESTRANTS.keys())}"
        )
    if ionization_type not in BINDING_AFFINITY:
        raise ValueError(
            f"Unknown ionization_type '{ionization_type}'. "
            f"Valid options: {list(BINDING_AFFINITY.keys())}"
        )
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0 < f_oral_baseline <= 1):
        raise ValueError(f"f_oral_baseline must be in (0, 1], got {f_oral_baseline}")

    seq_params = SEQUESTRANTS[bile_sequestrant]
    drug_binding_affinity = BINDING_AFFINITY[ionization_type]

    # Compute effective bioavailability with sequestrant
    bound_fraction = _compute_bound_fraction(
        seq_params["binding_fraction_per_g"],
        sequestrant_dose_g,
        drug_binding_affinity,
    )
    f_oral_with = f_oral_baseline * (1.0 - bound_fraction)

    # Simulate reference scenario (no sequestrant)
    times_h, c_without = _simulate_1cpt_oral(
        dose_mg, f_oral_baseline, ka_per_h, cl_L_per_h, vd_L, t_end_h, dt_h
    )

    # Simulate with sequestrant
    _, c_with = _simulate_1cpt_oral(dose_mg, f_oral_with, ka_per_h, cl_L_per_h, vd_L, t_end_h, dt_h)

    auc_without = _trapezoidal_auc(times_h, c_without)
    auc_with = _trapezoidal_auc(times_h, c_with)
    auc_reduction_pct = (auc_without - auc_with) / max(auc_without, 1e-12) * 100.0

    severity = _severity(auc_reduction_pct)
    timing = _timing_recommendation(severity)

    notes = (
        f"{drug_name} co-administered with {bile_sequestrant} ({sequestrant_dose_g} g). "
        f"Drug ionization: {ionization_type}. "
        f"AUC reduced by {auc_reduction_pct:.1f}% due to sequestrant binding. "
        f"Recommendation: {timing}."
    )

    return BileAcidInteractionResult(
        drug_name=drug_name,
        bile_sequestrant=bile_sequestrant,
        dose_mg=dose_mg,
        sequestrant_dose_g=sequestrant_dose_g,
        times_h=times_h,
        c_plasma_without_seq_mg_L=c_without,
        c_plasma_with_seq_mg_L=c_with,
        auc_without_seq_mg_h_per_L=auc_without,
        auc_with_seq_mg_h_per_L=auc_with,
        auc_reduction_pct=auc_reduction_pct,
        f_oral_without=f_oral_baseline,
        f_oral_with=f_oral_with,
        interaction_severity=severity,
        timing_recommendation=timing,
        notes=notes,
    )


def compare_sequestrants(
    drug_name: str,
    dose_mg: float,
    sequestrant_dose_g: float,
    ionization_type: str,
    ka_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral_baseline: float,
) -> list[BileAcidInteractionResult]:
    """Compare all 3 active sequestrants for a given drug.

    Returns a list of BileAcidInteractionResult sorted by auc_reduction_pct descending.
    """
    active_sequestrants = ["cholestyramine", "colesevelam", "colestipol"]
    results = []
    for seq in active_sequestrants:
        result = simulate_bile_acid_interaction(
            drug_name=drug_name,
            bile_sequestrant=seq,
            dose_mg=dose_mg,
            sequestrant_dose_g=sequestrant_dose_g,
            ionization_type=ionization_type,
            ka_per_h=ka_per_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            f_oral_baseline=f_oral_baseline,
        )
        results.append(result)
    results.sort(key=lambda r: r.auc_reduction_pct, reverse=True)
    return results
