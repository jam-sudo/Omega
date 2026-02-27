"""Receptor pharmacology: target engagement and occupancy over time.

Links plasma concentration-time profiles (from PBPK) to:
1. Receptor occupancy (equilibrium or kinetic binding)
2. Functional effect via the operational model (Black & Leff 1983)
3. EC50 / EC80 extraction from the occupancy-time curve

Models
------
EquilibriumOccupancy
    Langmuir isotherm at each time point (assumes binding equilibration
    is rapid relative to PK time scale, i.e. koff >> elimination rate).
    Appropriate when residence time RT = 1/koff ≪ t½.

KineticOccupancy
    Explicit on/off rate ODE for slow-binding drugs.
    dOcc/dt = kon × C(t) × (Rt - Occ) - koff × Occ
    where Rt = total receptor concentration (normalised to 1).
    Appropriate for drugs with RT (= 1/koff) ≥ 1 h (e.g. biologics,
    covalent inhibitors, tight binders like tiotropium).

OperationalModel
    Translates receptor occupancy to functional effect using the Black-Leff
    operational model (Black & Leff 1983, Proc R Soc Lond B 220:141-62):
        E = E0 + (Emax - E0) × τ × Occ / (1 + τ × Occ)
    where τ (tau) is the transduction ratio (efficacy × [Rt/KE]).
    τ = 1: full agonist (functional EC50 ≈ Kd)
    τ > 1: constitutive activity or amplified signalling
    τ < 1: partial agonist

EC50Extractor
    Fits the concentration-response relationship E = f(Cmax) across a
    range of doses to recover a functional EC50 value that can be used
    directly in the Emax PD models (qsp/pd_models.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OccupancyResult:
    """Receptor occupancy and functional effect over time.

    Attributes:
        time_h: Time array (h).
        occupancy: Receptor occupancy fraction (0–1) over time.
        effect: Functional effect over time (same units as E0/Emax).
        peak_occupancy: Maximum occupancy reached.
        tpeak_h: Time of peak occupancy (h).
        auc_occupancy_h: AUC of occupancy curve (h).
        model_type: "equilibrium" or "kinetic".
    """

    time_h: NDArray[np.float64]
    occupancy: NDArray[np.float64]
    effect: NDArray[np.float64]
    peak_occupancy: float
    tpeak_h: float
    auc_occupancy_h: float
    model_type: str


@dataclass(frozen=True)
class EC50Result:
    """Functional EC50 extracted from a simulated dose-response.

    Attributes:
        ec50_mg_L: Estimated functional EC50 (mg/L plasma concentration).
        emax_effect: Estimated maximum functional effect.
        e0_effect: Baseline effect (at zero drug).
        hill: Fitted Hill coefficient.
        r_squared: Goodness-of-fit of the Hill equation.
        doses_mg: Dose array used for fitting.
        effects_at_cmax: Effect at Cmax for each dose.
    """

    ec50_mg_L: float
    emax_effect: float
    e0_effect: float
    hill: float
    r_squared: float
    doses_mg: NDArray[np.float64]
    effects_at_cmax: NDArray[np.float64]


# ---------------------------------------------------------------------------
# Equilibrium occupancy model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EquilibriumOccupancy:
    """Langmuir equilibrium receptor occupancy model.

    Assumes binding equilibration is rapid relative to PK time scale.
    Suitable for drugs where koff >> elimination rate constant.

    Attributes:
        kd_mg_L: Equilibrium dissociation constant (mg/L).
            kd = koff / kon.  Lower Kd = tighter binding.
    """

    kd_mg_L: float = 1.0

    def compute(
        self,
        time_h: NDArray[np.floating[Any]],
        cp_mg_L: NDArray[np.floating[Any]],
        operational_model: OperationalModel | None = None,
    ) -> OccupancyResult:
        """Compute receptor occupancy and functional effect.

        Args:
            time_h: Time array (h).
            cp_mg_L: Plasma concentration (mg/L) from PBPK.
            operational_model: If provided, translates occupancy → effect.

        Returns:
            OccupancyResult with occupancy and effect time courses.
        """
        cp = np.maximum(cp_mg_L, 0.0)
        occ = cp / (self.kd_mg_L + cp)

        if operational_model is not None:
            effect = operational_model.effect(occ)
        else:
            effect = occ.copy()

        i_peak = int(np.argmax(occ))
        auc_occ = float(np.trapz(occ, time_h))

        return OccupancyResult(
            time_h=time_h.astype(np.float64),
            occupancy=occ.astype(np.float64),
            effect=effect.astype(np.float64),
            peak_occupancy=float(occ[i_peak]),
            tpeak_h=float(time_h[i_peak]),
            auc_occupancy_h=auc_occ,
            model_type="equilibrium",
        )


# ---------------------------------------------------------------------------
# Kinetic occupancy model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KineticOccupancy:
    """Kinetic receptor binding ODE model.

    Solves:
        dOcc/dt = kon × C(t) × (1 − Occ) − koff × Occ

    where Occ is fractional occupancy (0–1), C(t) is plasma concentration
    interpolated from the PBPK output.

    Attributes:
        kon_per_h_per_mg_L: Association rate constant (h⁻¹ per mg/L).
            Typical range: 0.1–10 h⁻¹(mg/L)⁻¹.
        koff_per_h: Dissociation rate constant (h⁻¹).
            Residence time RT = 1/koff (h).
        kd_mg_L: Equilibrium Kd = koff / kon (computed automatically).
    """

    kon_per_h_per_mg_L: float = 1.0
    koff_per_h: float = 1.0

    @property
    def kd_mg_L(self) -> float:
        """Equilibrium Kd = koff / kon (mg/L)."""
        return self.koff_per_h / max(self.kon_per_h_per_mg_L, 1e-12)

    @property
    def residence_time_h(self) -> float:
        """Mean receptor residence time = 1/koff (h)."""
        return 1.0 / max(self.koff_per_h, 1e-12)

    def compute(
        self,
        time_h: NDArray[np.floating[Any]],
        cp_mg_L: NDArray[np.floating[Any]],
        occ0: float = 0.0,
        operational_model: OperationalModel | None = None,
    ) -> OccupancyResult:
        """Solve the kinetic occupancy ODE.

        Args:
            time_h: Time array (h), must be monotonically increasing.
            cp_mg_L: Plasma concentration (mg/L).
            occ0: Initial fractional occupancy (default 0).
            operational_model: Optional functional translation.

        Returns:
            OccupancyResult with kinetic occupancy time course.
        """
        cp_interp = interp1d(
            time_h, np.maximum(cp_mg_L, 0.0),
            kind="linear", fill_value="extrapolate",
        )

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            occ = float(np.clip(y[0], 0.0, 1.0))
            c = max(float(cp_interp(t)), 0.0)
            docc = self.kon_per_h_per_mg_L * c * (1.0 - occ) - self.koff_per_h * occ
            return np.array([docc])

        sol = solve_ivp(
            rhs,
            (float(time_h[0]), float(time_h[-1])),
            [occ0],
            t_eval=time_h,
            method="LSODA",
            rtol=1e-8,
            atol=1e-10,
        )

        occ = np.clip(sol.y[0], 0.0, 1.0)

        if operational_model is not None:
            effect = operational_model.effect(occ)
        else:
            effect = occ.copy()

        i_peak = int(np.argmax(occ))
        auc_occ = float(np.trapz(occ, time_h))

        return OccupancyResult(
            time_h=time_h.astype(np.float64),
            occupancy=occ.astype(np.float64),
            effect=effect.astype(np.float64),
            peak_occupancy=float(occ[i_peak]),
            tpeak_h=float(time_h[i_peak]),
            auc_occupancy_h=auc_occ,
            model_type="kinetic",
        )


# ---------------------------------------------------------------------------
# Operational model (Black & Leff 1983)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperationalModel:
    """Black-Leff operational model: occupancy → functional effect.

    E(Occ) = E0 + (Emax − E0) × τ × Occ / (1 + τ × Occ)

    The transduction ratio τ encodes both intrinsic efficacy and the
    amplification of the signalling system downstream of the receptor:
        τ = [R_total] / KE × (intrinsic efficacy)

    For a full agonist in an unamplified system, τ ≈ 1 and the
    functional EC50 ≈ Kd.  For a partial agonist, τ < 1 and the
    observable Emax < 100%.

    Reference:
        Black JW, Leff P. Proc R Soc Lond B Biol Sci. 1983;220(1219):141-62.

    Attributes:
        e0: Baseline effect (zero drug).
        emax: Maximum possible system effect (full receptor activation).
        tau: Transduction ratio (> 0).
            τ >> 1: highly amplified system (agonist Emax ≈ system Emax)
            τ = 1: 1:1 coupling (functional EC50 ≈ Kd)
            τ < 1: partial agonist or weakly coupled system
    """

    e0: float = 0.0
    emax: float = 100.0
    tau: float = 1.0

    def effect(self, occupancy: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Compute functional effect from receptor occupancy array.

        Args:
            occupancy: Fractional receptor occupancy (0–1).

        Returns:
            Effect array (same shape as occupancy).
        """
        occ = np.clip(occupancy, 0.0, 1.0)
        numerator = self.tau * occ
        return self.e0 + (self.emax - self.e0) * numerator / (1.0 + numerator)

    @property
    def functional_ec50_fraction(self) -> float:
        """Occupancy fraction at 50% of maximum functional response.

        EC50_functional = 1 / (1 + τ)  (fractional occupancy units)
        """
        return 1.0 / (1.0 + max(self.tau, 1e-12))


# ---------------------------------------------------------------------------
# EC50 extractor (dose-response fitting)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EC50Extractor:
    """Extract functional EC50 from simulated dose-response data.

    Runs PBPK simulations over a log-spaced dose range, records the
    peak effect at Cmax for each dose, then fits a Hill equation to
    recover EC50 and Emax.

    Usage::

        extractor = EC50Extractor(
            occupancy_model=EquilibriumOccupancy(kd_mg_L=0.1),
            operational_model=OperationalModel(tau=2.0, emax=100.0),
        )
        ec50_result = extractor.fit_from_pbpk(drug, doses=np.logspace(-1, 2, 12))
    """

    occupancy_model: EquilibriumOccupancy | KineticOccupancy
    operational_model: OperationalModel | None = None

    def fit_from_cmax_array(
        self,
        cmax_array_mg_L: NDArray[np.floating[Any]],
        doses_mg: NDArray[np.floating[Any]],
    ) -> EC50Result:
        """Fit Hill equation to effect-at-Cmax vs dose data.

        Args:
            cmax_array_mg_L: Cmax values for each dose (mg/L).
            doses_mg: Corresponding dose values (mg).

        Returns:
            EC50Result with fitted pharmacodynamic parameters.
        """
        cmax = np.asarray(cmax_array_mg_L, dtype=np.float64)
        doses = np.asarray(doses_mg, dtype=np.float64)

        # Compute effect at each Cmax using occupancy model
        dummy_time = np.array([0.0, 1.0])
        effects_at_cmax = np.zeros(len(cmax))
        for i, c in enumerate(cmax):
            dummy_cp = np.array([c, c])
            if isinstance(self.occupancy_model, EquilibriumOccupancy):
                res = self.occupancy_model.compute(
                    dummy_time, dummy_cp, self.operational_model
                )
            else:
                res = self.occupancy_model.compute(
                    dummy_time, dummy_cp, operational_model=self.operational_model
                )
            effects_at_cmax[i] = float(res.effect[0])

        # Fit Hill equation: E = E0 + Emax × C^n / (EC50^n + C^n)
        # Using log-linear approximation for robust fitting
        ec50, emax, e0, hill, r2 = _fit_hill(cmax, effects_at_cmax)

        return EC50Result(
            ec50_mg_L=ec50,
            emax_effect=emax,
            e0_effect=e0,
            hill=hill,
            r_squared=r2,
            doses_mg=doses.astype(np.float64),
            effects_at_cmax=effects_at_cmax.astype(np.float64),
        )


def _fit_hill(
    conc: NDArray[np.float64],
    effect: NDArray[np.float64],
) -> tuple[float, float, float, float, float]:
    """Fit Hill equation to concentration-response data.

    Returns (ec50, emax, e0, hill, r_squared).
    Uses grid search over EC50 and hill followed by least-squares refinement.
    """
    from scipy.optimize import curve_fit

    e0 = float(effect[0]) if len(effect) > 0 else 0.0
    emax_est = float(effect[-1])

    def hill_fn(
        c: NDArray[np.float64], ec50: float, emax: float, gamma: float
    ) -> NDArray[np.float64]:
        c_g = np.power(np.maximum(c, 0.0), gamma)
        ec50_g = ec50 ** gamma
        return e0 + (emax - e0) * c_g / (ec50_g + c_g)

    try:
        popt, _ = curve_fit(
            hill_fn,
            conc,
            effect,
            p0=[float(np.median(conc)), emax_est, 1.0],
            bounds=([1e-6, -1e3, 0.1], [1e6, 1e3, 5.0]),
            maxfev=5000,
        )
        ec50, emax, hill = float(popt[0]), float(popt[1]), float(popt[2])
        pred = hill_fn(conc, ec50, emax, hill)
        ss_res = float(np.sum((effect - pred) ** 2))
        ss_tot = float(np.sum((effect - effect.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    except Exception:
        ec50 = float(np.median(conc))
        emax = emax_est
        hill = 1.0
        r2 = float("nan")

    return ec50, emax, e0, hill, r2


__all__ = [
    "OccupancyResult",
    "EC50Result",
    "EquilibriumOccupancy",
    "KineticOccupancy",
    "OperationalModel",
    "EC50Extractor",
]
