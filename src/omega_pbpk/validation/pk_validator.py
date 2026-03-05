"""PK Validation Engine — compute industry-standard accuracy metrics.

Compares simulated plasma concentration-time profiles against observed clinical
data and computes AAFE, AFE, RMSE, within-2-fold %, fold errors, and NCA-derived
AUC/Cmax relative errors.

AAFE (Average Absolute Fold Error):
    AAFE = 10^( (1/n) * Σ |log10(Csim_i / Cobs_i)| )
    AAFE = 1.0 → perfect; < 2.0 → acceptable for PBPK (Eissing et al. 2011).

AFE (Average Fold Error, signed):
    AFE = 10^( (1/n) * Σ log10(Csim_i / Cobs_i) )
    AFE > 1 → over-prediction; < 1 → under-prediction.

Within-2-fold criterion:
    Fraction of timepoints where 0.5 ≤ Csim/Cobs ≤ 2.0.
    FDA/EMA PBPK guidance: ≥ 2/3 of timepoints within 2-fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from omega_pbpk._compat import np_trapz
from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug


@dataclass
class ValidationResult:
    """Full validation metrics comparing simulation to observed clinical data."""

    # Core regulatory metrics
    aafe: float
    """Average Absolute Fold Error. Target: < 2.0 for PBPK acceptance."""

    afe: float
    """Average Fold Error (signed). AFE > 1 = over-prediction."""

    within_2fold_pct: float
    """Percentage of timepoints with Csim/Cobs within [0.5, 2.0]. Target: ≥ 66.7%."""

    rmse_mg_per_L: float
    """Root mean squared error (mg/L) on interpolated vs observed concentrations."""

    # NCA-level metrics
    auc_obs_mg_h_L: float
    """Trapezoidal AUC of observed data (mg·h/L)."""

    auc_sim_mg_h_L: float
    """Trapezoidal AUC of simulated data at observed timepoints (mg·h/L)."""

    auc_relative_error: float
    """|AUC_sim - AUC_obs| / AUC_obs."""

    cmax_obs_mg_L: float
    """Observed Cmax (mg/L)."""

    cmax_sim_mg_L: float
    """Simulated Cmax (mg/L) from full profile."""

    cmax_relative_error: float
    """|Cmax_sim - Cmax_obs| / Cmax_obs."""

    # Detailed fold error distribution
    n_obs: int
    """Number of observed timepoints used."""

    fold_errors: list[float] = field(default_factory=list)
    """Per-timepoint fold errors (Csim / Cobs), length == n_obs."""

    # Regulatory verdict
    @property
    def aafe_pass(self) -> bool:
        """AAFE < 2.0 (PBPK acceptance criterion, Eissing et al. 2011)."""
        return self.aafe < 2.0

    @property
    def within_2fold_pass(self) -> bool:
        """≥ 2/3 of timepoints within 2-fold (FDA/EMA PBPK guidance)."""
        return self.within_2fold_pct >= 66.7

    @property
    def auc_pass(self) -> bool:
        """AUC relative error ≤ 30%."""
        return self.auc_relative_error <= 0.30

    @property
    def cmax_pass(self) -> bool:
        """Cmax relative error ≤ 30%."""
        return self.cmax_relative_error <= 0.30

    @property
    def overall_pass(self) -> bool:
        """True if AAFE, within-2-fold, AUC and Cmax criteria all pass."""
        return self.aafe_pass and self.within_2fold_pass and self.auc_pass and self.cmax_pass

    def summary(self) -> dict[str, object]:
        """Return a flat dict suitable for JSON serialization."""
        return {
            "aafe": self.aafe,
            "afe": self.afe,
            "within_2fold_pct": self.within_2fold_pct,
            "rmse_mg_per_L": self.rmse_mg_per_L,
            "auc_obs_mg_h_L": self.auc_obs_mg_h_L,
            "auc_sim_mg_h_L": self.auc_sim_mg_h_L,
            "auc_relative_error": self.auc_relative_error,
            "cmax_obs_mg_L": self.cmax_obs_mg_L,
            "cmax_sim_mg_L": self.cmax_sim_mg_L,
            "cmax_relative_error": self.cmax_relative_error,
            "n_obs": self.n_obs,
            "fold_errors": self.fold_errors,
            "aafe_pass": self.aafe_pass,
            "within_2fold_pass": self.within_2fold_pass,
            "auc_pass": self.auc_pass,
            "cmax_pass": self.cmax_pass,
            "overall_pass": self.overall_pass,
        }


def compute_validation_metrics(
    obs_time: np.ndarray,
    obs_conc: np.ndarray,
    sim_time: np.ndarray,
    sim_conc: np.ndarray,
) -> ValidationResult:
    """Compute PK validation metrics from observed and simulated arrays.

    Args:
        obs_time: Observed timepoints (h), shape (n,).
        obs_conc: Observed concentrations (mg/L), shape (n,). All values must be > 0.
        sim_time: Simulated timepoints (h), shape (m,).
        sim_conc: Simulated concentrations (mg/L), shape (m,).

    Returns:
        ValidationResult with all regulatory metrics.
    """
    obs_time = np.asarray(obs_time, dtype=np.float64)
    obs_conc = np.asarray(obs_conc, dtype=np.float64)
    sim_time = np.asarray(sim_time, dtype=np.float64)
    sim_conc = np.asarray(sim_conc, dtype=np.float64)

    if len(obs_time) == 0:
        raise ValueError("obs_time must not be empty")
    if len(obs_time) != len(obs_conc):
        raise ValueError("obs_time and obs_conc must have the same length")
    if np.any(obs_conc <= 0):
        raise ValueError("All observed concentrations must be > 0 for fold-error computation")

    # Interpolate simulation onto observed timepoints
    sim_at_obs = np.interp(obs_time, sim_time, sim_conc)
    sim_at_obs = np.clip(sim_at_obs, 1e-12, None)

    n = len(obs_time)
    fold_errors = sim_at_obs / obs_conc

    # AAFE and AFE
    log10_fe = np.log10(fold_errors)
    aafe = float(10.0 ** np.mean(np.abs(log10_fe)))
    afe = float(10.0 ** np.mean(log10_fe))

    # Within-2-fold
    within_2fold = np.sum((fold_errors >= 0.5) & (fold_errors <= 2.0))
    within_2fold_pct = float(100.0 * within_2fold / n)

    # RMSE
    rmse = float(np.sqrt(np.mean((sim_at_obs - obs_conc) ** 2)))

    # AUC (trapezoidal on observed timepoints)
    auc_obs = float(np_trapz(obs_conc, obs_time))
    auc_sim = float(np_trapz(sim_at_obs, obs_time))
    auc_re = abs(auc_sim - auc_obs) / max(auc_obs, 1e-12)

    # Cmax from full simulated profile vs observed
    cmax_obs = float(np.max(obs_conc))
    cmax_sim = float(np.max(sim_conc))
    cmax_re = abs(cmax_sim - cmax_obs) / max(cmax_obs, 1e-12)

    return ValidationResult(
        aafe=aafe,
        afe=afe,
        within_2fold_pct=within_2fold_pct,
        rmse_mg_per_L=rmse,
        auc_obs_mg_h_L=auc_obs,
        auc_sim_mg_h_L=auc_sim,
        auc_relative_error=float(auc_re),
        cmax_obs_mg_L=cmax_obs,
        cmax_sim_mg_L=cmax_sim,
        cmax_relative_error=float(cmax_re),
        n_obs=n,
        fold_errors=fold_errors.tolist(),
    )


def validate_drug(
    drug: Drug,
    observed_times: list[float] | np.ndarray,
    observed_conc: list[float] | np.ndarray,
    dose_mg: float,
    route: Literal["oral", "iv"] = "oral",
    body_weight: float = 70.0,
    t_end_h: float | None = None,
    dt_h: float = 0.1,
) -> ValidationResult:
    """Run PBPK simulation and validate against observed clinical data.

    Args:
        drug: Drug object with PK parameters.
        observed_times: Clinical observation timepoints (h).
        observed_conc: Clinical plasma concentrations (mg/L). All must be > 0.
        dose_mg: Administered dose (mg).
        route: "oral" or "iv".
        body_weight: Subject body weight (kg).
        t_end_h: Simulation end time (h). Defaults to 1.2× max observed time.
        dt_h: ODE solver step size (h).

    Returns:
        ValidationResult with AAFE, AFE, within-2-fold %, RMSE, AUC/Cmax metrics.
    """
    obs_t = np.asarray(observed_times, dtype=np.float64)
    obs_c = np.asarray(observed_conc, dtype=np.float64)

    if t_end_h is None:
        t_end_h = float(np.max(obs_t)) * 1.2
        t_end_h = max(t_end_h, 8.0)

    model = WholeBodyPBPK(drug, body_weight=body_weight)
    if route == "iv":
        model.setup_iv(dose_mg=dose_mg)
    else:
        model.setup_oral(dose_mg=dose_mg)

    result = model.simulate(t_end_h=t_end_h, dt_h=dt_h)
    sim_t = result.time_h
    sim_c = result.plasma_concentration()

    return compute_validation_metrics(obs_t, obs_c, sim_t, sim_c)


__all__ = ["ValidationResult", "compute_validation_metrics", "validate_drug"]
