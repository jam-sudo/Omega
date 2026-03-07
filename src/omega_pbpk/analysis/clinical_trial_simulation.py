"""
Phase 631 — Clinical Trial Simulation

Simulate clinical trials with virtual patient populations, dropout, and outcome analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PatientOutcome:
    patient_id: int
    arm: str  # "treatment","placebo","dose_low","dose_high"
    dose_mg: float
    cmax_mg_L: float
    auc_mg_h_L: float
    response_pct: float  # percentage response (0-100)
    responder: bool  # response_pct >= response_threshold
    adverse_event: bool  # based on Cmax threshold
    discontinued: bool  # dropout
    notes: str


@dataclass(frozen=True)
class TrialSimulationResult:
    trial_name: str
    n_patients_per_arm: int
    arms: tuple  # arm names
    outcomes: tuple  # tuple of PatientOutcome
    response_rate: dict  # {arm: fraction_responders}
    ae_rate: dict  # {arm: fraction_ae}
    dropout_rate: dict  # {arm: fraction_discontinued}
    treatment_effect: float  # response_rate_treatment - response_rate_placebo
    p_value_approx: float  # approximate p-value
    power_estimate: float  # estimated power (0-1)
    trial_success: bool  # p_value < 0.05 and treatment_effect > 0
    notes: str


# ---------------------------------------------------------------------------
# Simple LCG random number generator (standalone, no external deps)
# ---------------------------------------------------------------------------


class _LCG:
    """Linear Congruential Generator for reproducible pseudo-random numbers."""

    # Parameters from Numerical Recipes
    _A = 1664525
    _C = 1013904223
    _M = 2**32

    def __init__(self, seed: int = 42) -> None:
        self._state = seed % self._M

    def next_float(self) -> float:
        """Return uniform float in [0, 1)."""
        self._state = (self._A * self._state + self._C) % self._M
        return self._state / self._M

    def next_lognormal(self, mean: float, cv: float) -> float:
        """
        Sample from lognormal with given mean and CV (coefficient of variation).
        Uses Box-Muller transform.
        """
        mu = math.log(mean)
        sigma = math.sqrt(math.log(1 + cv**2))
        u1 = self.next_float()
        u2 = self.next_float()
        # Avoid log(0)
        if u1 <= 0:
            u1 = 1e-12
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return math.exp(mu + sigma * z)


# ---------------------------------------------------------------------------
# Helper: normal CDF via math.erf
# ---------------------------------------------------------------------------


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# PK helper functions
# ---------------------------------------------------------------------------


def _compute_cmax_oral(
    dose_mg: float,
    ka: float,
    ke: float,
    vd: float,
    f: float,
) -> float:
    """Compute Cmax via Bateman equation (1-compartment oral)."""
    if abs(ka - ke) < 1e-9:
        # Degenerate case: ka == ke, use limit
        tmax = 1.0 / ka
    else:
        tmax = math.log(ka / ke) / (ka - ke)
    cmax = f * dose_mg * ka / (vd * (ka - ke)) * (math.exp(-ke * tmax) - math.exp(-ka * tmax))
    return max(0.0, cmax)


def _compute_cmax_iv(dose_mg: float, vd: float) -> float:
    """Compute Cmax for IV bolus (instantaneous)."""
    return dose_mg / vd


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------


def approximate_p_value(n1: int, r1: int, n2: int, r2: int) -> float:
    """
    Approximate p-value for 2-proportion z-test.
    n1, r1: treatment n and responders; n2, r2: placebo.
    Uses pooled proportion.
    """
    if n1 <= 0 or n2 <= 0:
        return 1.0
    p1 = r1 / n1
    p2 = r2 / n2
    p_pool = (r1 + r2) / (n1 + n2)
    denom = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if denom <= 0:
        return 1.0
    z = (p1 - p2) / math.sqrt(denom)
    # Two-sided p-value
    p_val = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return max(0.0, min(1.0, p_val))


def estimate_power(
    n_per_arm: int,
    expected_treatment_response: float,
    expected_placebo_response: float,
    alpha: float = 0.05,
) -> float:
    """
    Estimate power for 2-proportion test.
    Use normal approximation.
    """
    p0 = expected_placebo_response / 100.0
    p1 = expected_treatment_response / 100.0
    # Standard error of the difference under Ha
    var = p0 * (1.0 - p0) / n_per_arm + p1 * (1.0 - p1) / n_per_arm
    if var <= 0:
        return 0.0
    se = math.sqrt(var)
    # One-sided z_alpha at given alpha
    z_alpha = 1.6449  # ~1.645 for one-sided 0.05
    z_effect = (p1 - p0) / se - z_alpha
    power = _normal_cdf(z_effect)
    return max(0.0, min(1.0, power))


def simulate_trial(
    trial_name: str,
    n_per_arm: int,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    emax: float = 80.0,
    ec50_mg_L: float = 2.0,
    omega_cl: float = 0.3,
    omega_vd: float = 0.2,
    placebo_response_pct: float = 20.0,
    ae_threshold_mg_L: float = 10.0,
    dropout_pct: float = 10.0,
    seed: int = 42,
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    response_threshold: float = 50.0,
) -> TrialSimulationResult:
    """
    Simulate a two-arm (treatment vs placebo) clinical trial with virtual patient populations.
    """
    # Validation
    if n_per_arm < 5:
        raise ValueError("n_per_arm must be >= 5")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if emax <= 0:
        raise ValueError("emax must be > 0")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")

    rng = _LCG(seed)
    outcomes_list: list[PatientOutcome] = []
    arms_tuple = ("treatment", "placebo")

    # --- Treatment arm ---
    for i in range(n_per_arm):
        pid = i
        cl_i = rng.next_lognormal(cl_L_per_h, omega_cl)
        vd_i = rng.next_lognormal(vd_L, omega_vd)
        ke_i = cl_i / vd_i

        if route == "oral":
            cmax = _compute_cmax_oral(dose_mg, ka_per_h, ke_i, vd_i, f_oral)
        else:
            cmax = _compute_cmax_iv(dose_mg, vd_i)

        # AUC = f*dose/CL (analytical)
        f = f_oral if route == "oral" else 1.0
        auc = f * dose_mg / cl_i

        # Response: Emax model + uniform noise
        response_base = emax * cmax / (ec50_mg_L + cmax)
        noise = (rng.next_float() - 0.5) * 10.0
        response = min(100.0, max(0.0, response_base + noise))

        # Dropout
        discontinued = rng.next_float() < (dropout_pct / 100.0)
        # AE
        adverse_event = cmax > ae_threshold_mg_L
        # Responder
        responder = response >= response_threshold

        outcomes_list.append(
            PatientOutcome(
                patient_id=pid,
                arm="treatment",
                dose_mg=dose_mg,
                cmax_mg_L=cmax,
                auc_mg_h_L=auc,
                response_pct=response,
                responder=responder,
                adverse_event=adverse_event,
                discontinued=discontinued,
                notes="treatment arm",
            )
        )

    # --- Placebo arm ---
    for i in range(n_per_arm):
        pid = n_per_arm + i
        # Placebo: tiny dose (conceptually 0), no drug PK
        cmax = 0.0
        auc = 0.0

        # Response: placebo + noise
        noise = (rng.next_float() - 0.5) * 10.0
        response = min(100.0, max(0.0, placebo_response_pct + noise))

        discontinued = rng.next_float() < (dropout_pct / 100.0)
        adverse_event = False  # no drug → no AE based on Cmax
        responder = response >= response_threshold

        outcomes_list.append(
            PatientOutcome(
                patient_id=pid,
                arm="placebo",
                dose_mg=0.0,
                cmax_mg_L=cmax,
                auc_mg_h_L=auc,
                response_pct=response,
                responder=responder,
                adverse_event=adverse_event,
                discontinued=discontinued,
                notes="placebo arm",
            )
        )

    # Compute summary statistics per arm
    arm_patients: dict[str, list[PatientOutcome]] = {"treatment": [], "placebo": []}
    for o in outcomes_list:
        arm_patients[o.arm].append(o)

    response_rate: dict[str, float] = {}
    ae_rate: dict[str, float] = {}
    dropout_rate: dict[str, float] = {}

    for arm, patients in arm_patients.items():
        n = len(patients)
        if n == 0:
            response_rate[arm] = 0.0
            ae_rate[arm] = 0.0
            dropout_rate[arm] = 0.0
        else:
            response_rate[arm] = sum(1 for p in patients if p.responder) / n
            ae_rate[arm] = sum(1 for p in patients if p.adverse_event) / n
            dropout_rate[arm] = sum(1 for p in patients if p.discontinued) / n

    treatment_effect = response_rate["treatment"] - response_rate["placebo"]

    # Approximate p-value
    trt_responders = sum(1 for p in arm_patients["treatment"] if p.responder)
    plb_responders = sum(1 for p in arm_patients["placebo"] if p.responder)
    p_val = approximate_p_value(n_per_arm, trt_responders, n_per_arm, plb_responders)

    # Power estimate
    expected_trt = response_rate["treatment"] * 100.0
    power = estimate_power(n_per_arm, expected_trt, placebo_response_pct)

    trial_success = p_val < 0.05 and treatment_effect > 0.0

    notes_parts = [
        f"Trial: {trial_name}",
        f"route={route}",
        f"n_per_arm={n_per_arm}",
        f"treatment_effect={treatment_effect:.3f}",
        f"p_value={p_val:.4f}",
        f"success={trial_success}",
    ]
    notes = "; ".join(notes_parts)

    return TrialSimulationResult(
        trial_name=trial_name,
        n_patients_per_arm=n_per_arm,
        arms=arms_tuple,
        outcomes=tuple(outcomes_list),
        response_rate=response_rate,
        ae_rate=ae_rate,
        dropout_rate=dropout_rate,
        treatment_effect=treatment_effect,
        p_value_approx=p_val,
        power_estimate=power,
        trial_success=trial_success,
        notes=notes,
    )
