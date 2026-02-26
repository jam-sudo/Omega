"""Pharmacodynamic (PD) effect models.

Models:
  1. Emax — direct concentration-effect relationship
  2. Effect compartment — delayed response via ke0
  3. Indirect response — 4 types (stimulate/inhibit kin/kout)
  4. Tumor growth — Simeoni transit compartment model
  5. Biomarker turnover — production/degradation with drug effect
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class EmaxModel:
    """Direct Emax PD model: E = E0 + Emax × C^gamma / (EC50^gamma + C^gamma).

    Attributes:
        e0: Baseline effect.
        emax: Maximum drug effect.
        ec50: Concentration at 50% of Emax (mg/L).
        gamma: Hill coefficient (sigmoidicity).
    """

    e0: float = 0.0
    emax: float = 100.0
    ec50: float = 1.0
    gamma: float = 1.0

    def effect(self, concentration: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Calculate effect from concentration array."""
        c = np.maximum(concentration, 0.0)
        c_gamma = c**self.gamma
        ec50_gamma = self.ec50**self.gamma
        return self.e0 + self.emax * c_gamma / (ec50_gamma + c_gamma)


@dataclass(frozen=True)
class EffectCompartment:
    """Effect compartment PD model for delayed response.

    Uses Crank-Nicolson implicit integration for numerical stability.

    Attributes:
        ke0: Effect-site equilibration rate constant (h⁻¹).
        emax_model: Underlying Emax model.
    """

    ke0: float = 0.6
    emax_model: EmaxModel = field(default_factory=EmaxModel)

    def compute(
        self,
        time_h: NDArray[np.floating[Any]],
        cp: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Compute effect-site concentration and effect over time.

        Uses Crank-Nicolson (implicit trapezoidal) integration:
            Ce[i] = (Ce[i-1] + 0.5*dt*ke0*(Cp[i-1]+Cp[i])) / (1 + 0.5*dt*ke0)

        Returns:
            Tuple of (Ce array, Effect array).
        """
        n = len(time_h)
        ce = np.zeros(n)

        for i in range(1, n):
            dt = time_h[i] - time_h[i - 1]
            half_ke0_dt = 0.5 * self.ke0 * dt
            ce[i] = (ce[i - 1] + half_ke0_dt * (cp[i - 1] + cp[i])) / (1.0 + half_ke0_dt)

        effect = self.emax_model.effect(ce)
        return ce, effect


@dataclass(frozen=True)
class IndirectResponseModel:
    """Indirect response PD model (4 types).

    Type 1: Inhibition of kin   — dR/dt = kin × (1 - Imax×C/(IC50+C)) - kout × R
    Type 2: Inhibition of kout  — dR/dt = kin - kout × (1 - Imax×C/(IC50+C)) × R
    Type 3: Stimulation of kin  — dR/dt = kin × (1 + Smax×C/(SC50+C)) - kout × R
    Type 4: Stimulation of kout — dR/dt = kin - kout × (1 + Smax×C/(SC50+C)) × R

    Attributes:
        idr_type: Model type (1-4).
        kin: Zero-order production rate.
        kout: First-order elimination rate.
        imax_or_smax: Max inhibition (0-1 for types 1,2) or stimulation factor (types 3,4).
        ic50_or_sc50: Concentration for 50% of max effect (mg/L).
    """

    idr_type: int = 1
    kin: float = 1.0
    kout: float = 0.1
    imax_or_smax: float = 0.9
    ic50_or_sc50: float = 1.0

    def simulate(
        self,
        time_h: NDArray[np.floating[Any]],
        cp: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Simulate indirect response given plasma concentration time-course."""
        from scipy.interpolate import interp1d

        cp_interp = interp1d(time_h, cp, kind="linear", fill_value="extrapolate")
        r0 = self.kin / max(self.kout, 1e-12)

        def rhs(t: float, y: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
            c = max(float(cp_interp(t)), 0.0)
            r = y[0]
            drug_fn = self.imax_or_smax * c / (self.ic50_or_sc50 + c)

            if self.idr_type == 1:
                dr = self.kin * (1.0 - drug_fn) - self.kout * r
            elif self.idr_type == 2:
                dr = self.kin - self.kout * (1.0 - drug_fn) * r
            elif self.idr_type == 3:
                dr = self.kin * (1.0 + drug_fn) - self.kout * r
            elif self.idr_type == 4:
                dr = self.kin - self.kout * (1.0 + drug_fn) * r
            else:
                raise ValueError(f"Invalid IDR type: {self.idr_type}")
            return np.array([dr])

        sol = solve_ivp(
            rhs, (time_h[0], time_h[-1]), [r0], t_eval=time_h, method="LSODA", rtol=1e-8, atol=1e-10
        )
        return sol.y[0]


@dataclass(frozen=True)
class TumorGrowthModel:
    """Simeoni transit compartment tumor growth model.

    dV1/dt = (2*lambda0*lambda1*V1) / (lambda1 + 2*lambda0*V1) - k2*C*V1
    dV2/dt = k2*C*V1 - k1*V2
    dV3/dt = k1*V2 - k1*V3
    dV4/dt = k1*V3 - k1*V4
    Total tumor = V1 + V2 + V3 + V4

    Attributes:
        lambda0: Linear growth rate (1/h).
        lambda1: Exponential growth rate (volume/h).
        k1: Transit rate between damaged compartments (1/h).
        k2: Drug kill rate constant (L/mg/h).
        v0: Initial tumor volume (mm³).
    """

    lambda0: float = 0.003
    lambda1: float = 0.5
    k1: float = 0.01
    k2: float = 0.1
    v0: float = 100.0

    def simulate(
        self,
        time_h: NDArray[np.floating[Any]],
        cp: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Simulate tumor growth with drug effect."""
        from scipy.interpolate import interp1d

        cp_interp = interp1d(time_h, cp, kind="linear", fill_value="extrapolate")

        def rhs(t: float, y: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
            v1, v2, v3, v4 = y
            c = max(float(cp_interp(t)), 0.0)

            psi = (2.0 * self.lambda0 * self.lambda1 * v1) / max(
                self.lambda1 + 2.0 * self.lambda0 * v1, 1e-12
            )
            dv1 = psi - self.k2 * c * v1
            dv2 = self.k2 * c * v1 - self.k1 * v2
            dv3 = self.k1 * v2 - self.k1 * v3
            dv4 = self.k1 * v3 - self.k1 * v4
            return np.array([dv1, dv2, dv3, dv4])

        sol = solve_ivp(
            rhs,
            (time_h[0], time_h[-1]),
            [self.v0, 0, 0, 0],
            t_eval=time_h,
            method="LSODA",
            rtol=1e-8,
            atol=1e-10,
        )
        return np.sum(sol.y, axis=0)  # Total tumor volume


@dataclass(frozen=True)
class BiomarkerTurnover:
    """Biomarker turnover model with drug effect.

    dB/dt = ksyn × (1 + drug_effect) - kdeg × B

    Attributes:
        ksyn: Biomarker synthesis rate.
        kdeg: Biomarker degradation rate (1/h).
        emax: Maximum drug effect on synthesis.
        ec50: Concentration for 50% of max effect (mg/L).
    """

    ksyn: float = 10.0
    kdeg: float = 0.1
    emax: float = 2.0
    ec50: float = 1.0

    def simulate(
        self,
        time_h: NDArray[np.floating[Any]],
        cp: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Simulate biomarker response."""
        from scipy.interpolate import interp1d

        cp_interp = interp1d(time_h, cp, kind="linear", fill_value="extrapolate")
        b0 = self.ksyn / max(self.kdeg, 1e-12)

        def rhs(t: float, y: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
            c = max(float(cp_interp(t)), 0.0)
            drug_effect = self.emax * c / (self.ec50 + c)
            db = self.ksyn * (1.0 + drug_effect) - self.kdeg * y[0]
            return np.array([db])

        sol = solve_ivp(
            rhs, (time_h[0], time_h[-1]), [b0], t_eval=time_h, method="LSODA", rtol=1e-8, atol=1e-10
        )
        return sol.y[0]
