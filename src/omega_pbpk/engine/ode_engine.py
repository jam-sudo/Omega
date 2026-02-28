from __future__ import annotations

import numpy as np

from omega_pbpk._compat import np_trapz
from omega_pbpk.contracts import DrugSpec, PatientSpec, Regimen
from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug
from omega_pbpk.engine.interface import SimulationEngine, SimulationResult


class WholeBodyPBPKEngine(SimulationEngine):
    """DrugSpec/PatientSpec → 기존 WholeBodyPBPK ODE 실행 어댑터."""

    def _drugspec_to_drug(self, spec: DrugSpec) -> Drug:
        """DrugSpec → 기존 Drug 객체 변환."""
        return Drug(
            name=spec.name,
            mw=spec.mw,
            logP=spec.logP,
            pka=list(spec.pka),
            compound_type=spec.compound_type,
            fup=spec.fup,
            rbp=spec.rbp,
            smiles=spec.smiles,
            clint_hepatic_L_per_h=spec.clint_hepatic_L_per_h,
            clint_gut_L_per_h=spec.clint_gut_L_per_h,
            clr_L_per_h=spec.clr_L_per_h,
            peff=spec.peff,
            solubility_mg_mL=spec.solubility_mg_mL,
            kp=dict(spec.kp),
            permeability_limited={k: dict(v) for k, v in spec.permeability_limited.items()},
        )

    def run(
        self,
        drug: DrugSpec,
        patient: PatientSpec,
        regimen: Regimen,
        t_end_h: float = 24.0,
        n_points: int = 241,
        solver_config: dict | None = None,
    ) -> SimulationResult:
        d = self._drugspec_to_drug(drug)
        model = WholeBodyPBPK(d, body_weight=patient.body_weight_kg)

        dt_h = t_end_h / max(n_points - 1, 1)

        if regimen.route == "oral":
            model.setup_oral(dose_mg=regimen.dose_mg)
        elif regimen.route == "iv":
            model.setup_iv(dose_mg=regimen.dose_mg)
        elif regimen.route == "sc":
            model.setup_sc(dose_mg=regimen.dose_mg)
        else:
            raise ValueError(f"Unsupported route: {regimen.route}")

        raw = model.simulate(t_end_h=t_end_h, dt_h=dt_h)

        plasma_conc = raw.plasma_concentration()

        return SimulationResult(
            t=raw.time_h.astype(np.float64),
            amounts=raw.amounts.T.astype(np.float64),  # (n_states, n_timepoints)
            plasma_concentration=plasma_conc.astype(np.float64),
            drug_name=drug.name,
            route=regimen.route,
            dose_mg=regimen.dose_mg,
            success=True,
        )

    def pk_summary(self, result: SimulationResult) -> dict[str, float]:
        cp = result.plasma_concentration
        t = result.t
        auc = float(np_trapz(cp, t))

        cmax = result.Cmax
        n = len(t)
        start = max(n // 2, 1)
        cp_term = cp[start:]
        t_term = t[start:]
        mask = cp_term > cmax * 0.01
        if np.sum(mask) >= 3:
            log_cp = np.log(np.maximum(cp_term[mask], 1e-30))
            t_fit = t_term[mask]
            coeffs = np.polyfit(t_fit, log_cp, 1)
            ke = -coeffs[0]
            half_life = float(0.693 / max(ke, 1e-12)) if ke > 0 else float("inf")
        else:
            half_life = float("inf")

        return {
            "Cmax": result.Cmax,
            "Tmax": result.Tmax,
            "AUC": auc,
            "t_half": half_life,
        }
