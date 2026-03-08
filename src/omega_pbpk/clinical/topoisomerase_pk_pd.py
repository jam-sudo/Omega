"""Phase 476 — Topoisomerase Inhibitor PK/PD.

Models PK/PD of topoisomerase inhibitors (camptothecins, anthracyclines) using a
2-compartment IV model and Emax-based cell-kill pharmacodynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Drug database
# ---------------------------------------------------------------------------
_DRUG_DB: dict[str, dict] = {
    "irinotecan": {
        "type": "topo_I",
        "ic50_mg_L": 0.5,
        "emax": 0.9,
        "gamma": 1.5,
        "t_half_central_h": 6.0,
        "t_half_periph_h": 24.0,
    },
    "topotecan": {
        "type": "topo_I",
        "ic50_mg_L": 0.1,
        "emax": 0.95,
        "gamma": 1.2,
        "t_half_central_h": 3.0,
        "t_half_periph_h": 12.0,
    },
    "doxorubicin": {
        "type": "topo_II",
        "ic50_mg_L": 0.05,
        "emax": 0.85,
        "gamma": 2.0,
        "t_half_central_h": 1.0,
        "t_half_periph_h": 30.0,
    },
    "etoposide": {
        "type": "topo_II",
        "ic50_mg_L": 1.0,
        "emax": 0.80,
        "gamma": 1.0,
        "t_half_central_h": 4.0,
        "t_half_periph_h": 20.0,
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class TopoisomerasePKPDResult:
    """Results from a topoisomerase inhibitor PK/PD simulation."""

    drug_name: str
    drug_class: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_central_mg_L: list = field(default_factory=list)
    kill_rate_per_h: list = field(default_factory=list)
    auc_central_mg_h_per_L: float = 0.0
    cmax_central_mg_L: float = 0.0
    tumor_cell_kill_fraction: float = 0.0
    ic50_mg_L: float = 0.0
    emax: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def simulate_topoisomerase_pk_pd(
    drug_name: str,
    dose_mg: float,
    vd_central_L: float = 10.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> TopoisomerasePKPDResult:
    """Simulate PK/PD of a topoisomerase inhibitor given as an IV bolus.

    PK model: 2-compartment biexponential disposition.
      C(t) = (dose / Vd) * [alpha/(alpha+beta)*exp(-alpha*t)
                             + beta/(alpha+beta)*exp(-beta*t)]

    PD model: Emax Hill equation linking concentration to kill rate, then
    integrated to give the log-fractional cell kill.

    Parameters
    ----------
    drug_name:     One of {'irinotecan', 'topotecan', 'doxorubicin', 'etoposide'}.
    dose_mg:       IV bolus dose (mg).
    vd_central_L:  Central volume of distribution (L); default 10 L.
    t_end_h:       Simulation duration (h); default 72 h.
    dt_h:          Time step for numerical integration (h); default 0.1 h.

    Returns
    -------
    TopoisomerasePKPDResult with full time-series and summary metrics.
    """
    # --- Validation ---
    if drug_name not in _DRUG_DB:
        valid = ", ".join(sorted(_DRUG_DB.keys()))
        raise ValueError(f"Unknown drug '{drug_name}'. Valid options: {valid}")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vd_central_L <= 0:
        raise ValueError("vd_central_L must be > 0")

    params = _DRUG_DB[drug_name]
    ic50 = params["ic50_mg_L"]
    emax = params["emax"]
    gamma = params["gamma"]
    t_half_c = params["t_half_central_h"]
    t_half_p = params["t_half_periph_h"]
    drug_class = params["type"]

    alpha = math.log(2.0) / t_half_c  # fast disposition rate (h⁻¹)
    beta = math.log(2.0) / t_half_p  # slow disposition rate (h⁻¹)

    c0 = dose_mg / vd_central_L  # initial central concentration (mg/L)

    # Biexponential weighting factors
    w_alpha = alpha / (alpha + beta)
    w_beta = beta / (alpha + beta)

    # --- Time-course ---
    n_steps = max(1, int(round(t_end_h / dt_h)))
    times: list[float] = []
    c_central: list[float] = []
    kill_rates: list[float] = []

    for i in range(n_steps + 1):
        t = i * dt_h
        c = c0 * (w_alpha * math.exp(-alpha * t) + w_beta * math.exp(-beta * t))
        c_gamma = c**gamma
        ic50_gamma = ic50**gamma
        kr = emax * c_gamma / (ic50_gamma + c_gamma)

        times.append(t)
        c_central.append(c)
        kill_rates.append(kr)

    # --- AUC (trapezoidal) ---
    auc = sum(
        0.5 * (c_central[i] + c_central[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    cmax = c_central[0]  # IV bolus, maximum at t=0

    # --- Tumor cell kill fraction ---
    # Integrate kill rate to get cumulative log-kill
    cum_kill_integral = sum(
        0.5 * (kill_rates[i] + kill_rates[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    # cell_kill_fraction = 1 - exp(-integral of kill_rate * dt)
    kill_fraction = 1.0 - math.exp(-cum_kill_integral)
    kill_fraction = max(0.0, min(1.0, kill_fraction))

    notes = (
        f"2-cpt biexponential IV bolus: alpha={alpha:.4f} h-1, "
        f"beta={beta:.4f} h-1; Emax Hill PD (gamma={gamma}); "
        f"kill_fraction={kill_fraction:.4f}."
    )

    return TopoisomerasePKPDResult(
        drug_name=drug_name,
        drug_class=drug_class,
        dose_mg=dose_mg,
        times_h=times,
        c_central_mg_L=c_central,
        kill_rate_per_h=kill_rates,
        auc_central_mg_h_per_L=auc,
        cmax_central_mg_L=cmax,
        tumor_cell_kill_fraction=kill_fraction,
        ic50_mg_L=ic50,
        emax=emax,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Multi-drug comparison
# ---------------------------------------------------------------------------
def compare_topoisomerase_drugs(dose_mg: float = 100.0) -> list:
    """Compare all 4 topoisomerase inhibitors at the same dose.

    Parameters
    ----------
    dose_mg:  IV bolus dose applied to every drug (mg); default 100 mg.

    Returns
    -------
    List of TopoisomerasePKPDResult sorted by tumor_cell_kill_fraction descending.
    """
    results = [simulate_topoisomerase_pk_pd(drug_name=name, dose_mg=dose_mg) for name in _DRUG_DB]
    results.sort(key=lambda r: r.tumor_cell_kill_fraction, reverse=True)
    return results
