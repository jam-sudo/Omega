"""
Phase 258: Corneal Permeation PK

PK model for drug permeation through the cornea (ophthalmic drug delivery).
Pure Python (stdlib: math, dataclasses only). Forward Euler ODE integration.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Dosage form lookup table
# ---------------------------------------------------------------------------

_DOSAGE_FORMS = {
    "eye_drop": {
        "lag_min": 0.0,
        "dose_fraction": 1.0,
        "drainage_rate": 30.0,  # percent/min
    },
    "gel": {
        "lag_min": 2.0,
        "dose_fraction": 0.8,
        "drainage_rate": 10.0,
    },
    "ointment": {
        "lag_min": 5.0,
        "dose_fraction": 0.6,
        "drainage_rate": 5.0,
    },
    "insert": {
        "lag_min": 10.0,
        "dose_fraction": 0.9,
        "drainage_rate": 2.0,
    },
}

# Compartment volumes (mL)
_V_TEAR = 0.007
_V_CORNEA = 0.2
_V_AQUEOUS = 0.25

# Corneal surface area (cm^2)
_A_CORNEA = 1.3

# Aqueous elimination rate constant (per hour -> per min)
_K_ELIM_PER_MIN = 0.04 / 60.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CornealPermeationResult:
    drug_name: str
    dose_mg: float
    dosage_form: str
    times_min: list[float]
    c_tear_mg_mL: list[float]
    c_cornea_mg_mL: list[float]
    c_aqueous_mg_mL: list[float]
    cmax_aqueous_mg_mL: float
    tmax_aqueous_min: float
    auc_aqueous_mg_min_per_mL: float
    corneal_permeability_cm_s: float
    bioavailability_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trapz(times: list[float], values: list[float]) -> float:
    """Trapezoidal numerical integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def _potts_guy_permeability(logp: float, mw: float) -> float:
    """Corneal permeability via simplified Potts-Guy model (cm/s)."""
    exponent = 0.53 * logp - 0.0023 * mw - 6.7
    return 10.0**exponent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_corneal_permeation(
    drug_name: str,
    dose_mg: float,
    logp: float,
    mw: float,
    dosage_form: str = "eye_drop",
    t_end_min: float = 120.0,
    dt_min: float = 0.5,
) -> CornealPermeationResult:
    """Simulate corneal drug permeation using a 3-compartment ODE model.

    Compartments: tear film -> cornea -> aqueous humor.
    Uses Forward Euler integration.

    Parameters
    ----------
    drug_name : str
    dose_mg : float   Applied dose (mg, > 0).
    logp : float      Octanol-water partition coefficient.
    mw : float        Molecular weight (g/mol).
    dosage_form : str One of "eye_drop", "gel", "ointment", "insert".
    t_end_min : float Simulation end time (minutes, default 120).
    dt_min : float    Time step (minutes, default 0.5).

    Returns
    -------
    CornealPermeationResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if dosage_form not in _DOSAGE_FORMS:
        raise ValueError(
            f"dosage_form must be one of {list(_DOSAGE_FORMS.keys())}, got '{dosage_form}'"
        )

    form = _DOSAGE_FORMS[dosage_form]
    lag_min = form["lag_min"]
    dose_fraction = form["dose_fraction"]
    # drainage_rate is percent/min -> fractional/min
    k_drain = form["drainage_rate"] / 100.0  # per min

    # Corneal permeability (cm/s) -> convert to cm/min
    perm_cm_s = _potts_guy_permeability(logp, mw)
    perm_cm_min = perm_cm_s * 60.0

    # Transfer rate constants (per min)
    k_corn = perm_cm_min * _A_CORNEA / _V_TEAR  # tear -> cornea
    k_aq = perm_cm_min * _A_CORNEA / _V_CORNEA  # cornea -> aqueous

    # Initial conditions — all zero; tear is dosed after lag
    c_tear = 0.0
    c_cornea = 0.0
    c_aqueous = 0.0

    dose_applied = False

    times: list[float] = []
    ct: list[float] = []
    cc: list[float] = []
    ca: list[float] = []

    n_steps = int(t_end_min / dt_min) + 1

    for step in range(n_steps):
        t = step * dt_min

        # Apply dose at lag time (first step >= lag_min)
        if not dose_applied and t >= lag_min:
            c_tear += dose_mg * dose_fraction / _V_TEAR
            dose_applied = True

        times.append(t)
        ct.append(c_tear)
        cc.append(c_cornea)
        ca.append(c_aqueous)

        # Forward Euler derivatives
        dct = -(k_drain + k_corn) * c_tear
        dcc = k_corn * c_tear * _V_TEAR / _V_CORNEA - k_aq * c_cornea
        dca = k_aq * c_cornea * _V_CORNEA / _V_AQUEOUS - _K_ELIM_PER_MIN * c_aqueous

        c_tear += dct * dt_min
        c_cornea += dcc * dt_min
        c_aqueous += dca * dt_min

        # Clamp negatives (numerical artefacts)
        if c_tear < 0.0:
            c_tear = 0.0
        if c_cornea < 0.0:
            c_cornea = 0.0
        if c_aqueous < 0.0:
            c_aqueous = 0.0

    # Derived metrics
    cmax_aq = max(ca)
    tmax_aq = times[ca.index(cmax_aq)]
    auc_aq = _trapz(times, ca)

    # Bioavailability: fraction of dose reaching aqueous (mass basis)
    # mass_aqueous = AUC * k_elim * V_aqueous  (amount eliminated ~ absorbed at SS)
    mass_in_aqueous = auc_aq * _K_ELIM_PER_MIN * _V_AQUEOUS
    dose_applied_mg = dose_mg * dose_fraction
    bioavailability_pct = 100.0 * mass_in_aqueous / dose_applied_mg if dose_applied_mg > 0 else 0.0
    bioavailability_pct = max(0.0, min(100.0, bioavailability_pct))

    notes = (
        f"Corneal permeability: {perm_cm_s:.3e} cm/s; "
        f"dosage form: {dosage_form} (lag={lag_min} min)."
    )

    return CornealPermeationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosage_form=dosage_form,
        times_min=times,
        c_tear_mg_mL=ct,
        c_cornea_mg_mL=cc,
        c_aqueous_mg_mL=ca,
        cmax_aqueous_mg_mL=cmax_aq,
        tmax_aqueous_min=tmax_aq,
        auc_aqueous_mg_min_per_mL=auc_aq,
        corneal_permeability_cm_s=perm_cm_s,
        bioavailability_pct=bioavailability_pct,
        notes=notes,
    )


def compare_dosage_forms(
    drug_name: str,
    dose_mg: float,
    logp: float,
    mw: float,
) -> list[CornealPermeationResult]:
    """Compare all four dosage forms sorted by AUC in aqueous descending.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    logp : float
    mw : float

    Returns
    -------
    List[CornealPermeationResult]  4 entries, sorted by auc_aqueous desc.
    """
    results = []
    for form_name in _DOSAGE_FORMS:
        result = simulate_corneal_permeation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logp=logp,
            mw=mw,
            dosage_form=form_name,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_aqueous_mg_min_per_mL, reverse=True)
    return results
