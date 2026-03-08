"""Phase 1049 — Drug Cardiac Toxicity Biomarker prediction."""

from dataclasses import dataclass


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        total += 0.5 * (values[i] + values[i + 1]) * dt
    return total


@dataclass
class CardiacToxicityResult:
    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    troponin_ng_mL: list
    bnp_ng_mL: list
    qt_interval_ms: list
    auc_plasma: float
    peak_troponin: float
    peak_bnp: float
    max_qt_ms: float
    baseline_qt_ms: float
    delta_qt_ms: float
    cardiac_risk_score: float
    risk_category: str
    notes: str


def simulate_cardiac_toxicity(
    drug_name: str,
    dose_mg: float,
    herg_ic50_mg_L: float = 10.0,
    cardiac_ic50_mg_L: float = 50.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 30.0,
    baseline_qt_ms: float = 400.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> CardiacToxicityResult:
    """Simulate cardiac toxicity biomarker dynamics using 1-cpt PK + biomarker ODEs."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if herg_ic50_mg_L <= 0:
        raise ValueError("herg_ic50_mg_L must be > 0")
    if cardiac_ic50_mg_L <= 0:
        raise ValueError("cardiac_ic50_mg_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (300.0 <= baseline_qt_ms <= 500.0):
        raise ValueError("baseline_qt_ms must be in [300, 500]")

    ke = cl_L_per_h / vd_L
    c0 = dose_mg / vd_L

    # Initial conditions
    c = c0
    trop = 0.01
    bnp = 0.01

    # Biomarker rate constants
    k_trop = 0.1
    k_elim_trop = 0.05
    k_bnp = 0.05
    k_elim_bnp = 0.02

    times = []
    c_list = []
    trop_list = []
    bnp_list = []
    qt_list = []

    t = 0.0
    while t <= t_end_h + 1e-9:
        # QT interval (instantaneous)
        e_herg = c / (herg_ic50_mg_L + c) if (herg_ic50_mg_L + c) > 0 else 0.0
        qt = baseline_qt_ms + 100.0 * e_herg

        times.append(t)
        c_list.append(c)
        trop_list.append(trop)
        bnp_list.append(bnp)
        qt_list.append(qt)

        # Forward Euler
        # PK
        dc_dt = -ke * c

        # Troponin (Hill=2)
        c_sq = c * c
        ic50_sq = cardiac_ic50_mg_L * cardiac_ic50_mg_L
        e_trop = c_sq / (ic50_sq + c_sq) if (ic50_sq + c_sq) > 0 else 0.0
        dtrop_dt = k_trop * e_trop * 10.0 - k_elim_trop * trop

        # BNP (Hill=1)
        e_bnp = c / (cardiac_ic50_mg_L + c) if (cardiac_ic50_mg_L + c) > 0 else 0.0
        dbnp_dt = k_bnp * e_bnp * 5.0 - k_elim_bnp * bnp

        c = max(0.0, c + dc_dt * dt_h)
        trop = max(0.0, trop + dtrop_dt * dt_h)
        bnp = max(0.0, bnp + dbnp_dt * dt_h)

        t += dt_h

    auc_plasma = _trapz(times, c_list)
    peak_troponin = max(trop_list)
    peak_bnp = max(bnp_list)
    max_qt_ms = max(qt_list)
    delta_qt_ms = max_qt_ms - baseline_qt_ms

    # Cardiac risk score
    trop_score = min(100.0, peak_troponin * 20.0)
    bnp_score = min(100.0, peak_bnp * 20.0)
    qt_score = min(100.0, delta_qt_ms * 2.0)
    cardiac_risk_score = 0.4 * qt_score + 0.35 * trop_score + 0.25 * bnp_score
    cardiac_risk_score = max(0.0, min(100.0, cardiac_risk_score))

    if cardiac_risk_score < 20.0:
        risk_category = "safe"
    elif cardiac_risk_score < 40.0:
        risk_category = "monitor"
    elif cardiac_risk_score < 70.0:
        risk_category = "warning"
    else:
        risk_category = "contraindicated"

    notes = (
        f"{drug_name}: peak troponin={peak_troponin:.3f} ng/mL, "
        f"peak BNP={peak_bnp:.3f} ng/mL, "
        f"delta QT={delta_qt_ms:.1f} ms, "
        f"risk={risk_category}"
    )

    return CardiacToxicityResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_list,
        troponin_ng_mL=trop_list,
        bnp_ng_mL=bnp_list,
        qt_interval_ms=qt_list,
        auc_plasma=auc_plasma,
        peak_troponin=peak_troponin,
        peak_bnp=peak_bnp,
        max_qt_ms=max_qt_ms,
        baseline_qt_ms=baseline_qt_ms,
        delta_qt_ms=delta_qt_ms,
        cardiac_risk_score=cardiac_risk_score,
        risk_category=risk_category,
        notes=notes,
    )
