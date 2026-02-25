"""Clinical dose optimization tools.

Includes:
  - First-in-human (FIH) dose calculation
  - Multi-dose steady-state simulation
  - Formulation comparison (F, ka)
  - Therapeutic window optimization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug


@dataclass(frozen=True)
class FIHDoseResult:
    """First-in-human dose calculation result."""

    method: str
    noael_mg_kg: float
    hek: float  # Human equivalent dose factor
    safety_factor: float
    fih_dose_mg: float
    body_weight_kg: float = 70.0


@dataclass(frozen=True)
class SteadyStateResult:
    """Multi-dose steady-state simulation result."""

    time_h: NDArray[np.floating[Any]]
    cp_mg_L: NDArray[np.floating[Any]]
    dose_mg: float
    interval_h: float
    n_doses: int
    css_max: float
    css_min: float
    css_avg: float
    accumulation_ratio: float
    fluctuation_percent: float


@dataclass(frozen=True)
class FormulationResult:
    """Formulation comparison result."""

    name: str
    bioavailability: float
    ka_h: float
    cmax: float
    tmax: float
    auc: float


class DoseOptimizer:
    """Clinical dose optimization engine."""

    def fih_dose(
        self,
        noael_mg_kg: float,
        species: str = "rat",
        body_weight_kg: float = 70.0,
        safety_factor: float = 10.0,
    ) -> FIHDoseResult:
        """Calculate first-in-human dose using allometric scaling.

        Uses FDA guidance body surface area (BSA) conversion factors.

        Args:
            noael_mg_kg: No observed adverse effect level (mg/kg) in animal.
            species: Animal species (rat, mouse, dog, monkey, rabbit).
            body_weight_kg: Human body weight (kg).
            safety_factor: Safety factor (default 10×).
        """
        # BSA conversion factors (FDA Guidance, 2005)
        # Human Equivalent Dose = Animal dose × (Animal Km / Human Km)
        km_factors = {
            "mouse": 3.0,
            "rat": 6.2,
            "rabbit": 12.0,
            "dog": 20.0,
            "monkey": 12.0,
        }

        human_km = 37.0  # kg/m² for 60kg human
        animal_km = km_factors.get(species.lower(), 6.2)

        hed_mg_kg = noael_mg_kg * (animal_km / human_km)
        fih_dose = hed_mg_kg * body_weight_kg / safety_factor

        return FIHDoseResult(
            method=f"BSA scaling from {species}",
            noael_mg_kg=noael_mg_kg,
            hek=round(animal_km / human_km, 4),
            safety_factor=safety_factor,
            fih_dose_mg=round(fih_dose, 4),
            body_weight_kg=body_weight_kg,
        )

    def optimize_dose(
        self,
        drug: Drug,
        mec_mg_L: float,
        mtc_mg_L: float,
        dose_range: tuple[float, float] = (0.1, 1000.0),
        n_steps: int = 50,
    ) -> dict[str, Any]:
        """Find optimal dose that keeps Cmax within therapeutic window.

        Args:
            drug: Drug compound parameters.
            mec_mg_L: Minimum effective concentration (mg/L).
            mtc_mg_L: Maximum tolerated concentration (mg/L).
            dose_range: (min_dose, max_dose) in mg.
            n_steps: Number of dose levels to evaluate.

        Returns:
            Dict with optimal dose, Cmax, AUC, and therapeutic index.
        """
        doses = np.linspace(dose_range[0], dose_range[1], n_steps)
        best: dict[str, Any] | None = None
        best_score = -1.0

        for dose in doses:
            model = WholeBodyPBPK(drug)
            if drug.route == "iv":
                model.setup_iv(float(dose))
            else:
                model.setup_oral(float(dose))

            result = model.simulate(t_end_h=24.0, dt_h=0.1)
            pk = result.pk_summary()
            cmax = pk["Cmax_mg_L"]

            if cmax < mec_mg_L or cmax > mtc_mg_L:
                continue

            # Score: maximize time within therapeutic window
            cp = result.plasma_concentration()
            in_window = np.sum((cp >= mec_mg_L) & (cp <= mtc_mg_L))
            score = in_window / len(cp)

            if score > best_score:
                best_score = score
                best = {
                    "optimal_dose_mg": round(float(dose), 2),
                    "Cmax_mg_L": cmax,
                    "AUC_mg_h_L": pk["AUC_mg_h_L"],
                    "time_in_window_pct": round(float(score * 100), 1),
                    "therapeutic_index": round(mtc_mg_L / max(cmax, 1e-12), 2),
                }

        if best is None:
            return {
                "error": "No dose achieves therapeutic window",
                "mec": mec_mg_L,
                "mtc": mtc_mg_L,
            }
        return best


class MultiDoseSimulator:
    """Multi-dose steady-state simulation.

    Simulates repeated dosing by superposition of single-dose profiles.
    """

    def simulate(
        self,
        drug: Drug,
        dose_mg: float,
        interval_h: float,
        n_days: int = 7,
        body_weight: float = 70.0,
    ) -> SteadyStateResult:
        """Simulate multi-dose PK profile.

        Args:
            drug: Drug compound.
            dose_mg: Dose per administration (mg).
            interval_h: Dosing interval (h).
            n_days: Number of days to simulate.
            body_weight: Body weight (kg).
        """
        total_time = n_days * 24.0
        n_doses = int(total_time / interval_h)
        dt = 0.1

        # Single-dose profile
        model = WholeBodyPBPK(drug, body_weight=body_weight)
        if drug.route == "iv":
            model.setup_iv(dose_mg)
        else:
            model.setup_oral(dose_mg)

        single = model.simulate(t_end_h=total_time, dt_h=dt)
        single_cp = single.plasma_concentration()
        single_t = single.time_h

        # Superposition
        t_out = single_t.copy()
        cp_total = np.zeros_like(single_cp)

        for i in range(n_doses):
            t_shift = i * interval_h
            # Shift index
            idx_shift = int(t_shift / dt)
            if idx_shift >= len(single_cp):
                break
            remaining = len(single_cp) - idx_shift
            cp_total[idx_shift : idx_shift + remaining] += single_cp[:remaining]

        # Steady-state metrics from last dosing interval
        last_dose_t = (n_doses - 1) * interval_h
        mask = (t_out >= last_dose_t) & (t_out < last_dose_t + interval_h)
        if np.any(mask):
            css_max = float(np.max(cp_total[mask]))
            css_min = float(np.min(cp_total[mask]))
            css_avg = float(np.mean(cp_total[mask]))
        else:
            css_max = float(np.max(cp_total))
            css_min = float(np.min(cp_total))
            css_avg = float(np.mean(cp_total))

        # First dose Cmax
        first_mask = t_out < interval_h
        cmax_first = float(np.max(cp_total[first_mask])) if np.any(first_mask) else css_max
        accumulation = css_max / max(cmax_first, 1e-12)
        fluctuation = 100 * (css_max - css_min) / max(css_avg, 1e-12)

        return SteadyStateResult(
            time_h=t_out,
            cp_mg_L=cp_total,
            dose_mg=dose_mg,
            interval_h=interval_h,
            n_doses=n_doses,
            css_max=round(css_max, 6),
            css_min=round(css_min, 6),
            css_avg=round(css_avg, 6),
            accumulation_ratio=round(accumulation, 3),
            fluctuation_percent=round(fluctuation, 1),
        )


class FormulationComparator:
    """Compare different formulations by varying F and ka."""

    def compare(
        self,
        drug: Drug,
        formulations: list[dict[str, Any]],
        dose_mg: float = 100.0,
    ) -> list[FormulationResult]:
        """Compare formulations with different bioavailability and absorption rates.

        Args:
            drug: Base drug compound.
            formulations: List of dicts with keys: name, bioavailability (0-1), ka_multiplier.
            dose_mg: Dose (mg).
        """
        results: list[FormulationResult] = []

        for form in formulations:
            name = form.get("name", "Unknown")
            f_oral = form.get("bioavailability", 1.0)
            ka_mult = form.get("ka_multiplier", 1.0)

            # Adjust effective dose by bioavailability
            effective_dose = dose_mg * f_oral

            # Modify absorption parameters
            modified_drug = Drug(
                **{
                    **drug.__dict__,
                    "peff": drug.peff * ka_mult,
                    "dose_mg": dose_mg,
                    "route": "oral",
                }
            )

            model = WholeBodyPBPK(modified_drug)
            model.setup_oral(effective_dose)
            result = model.simulate(t_end_h=48.0)
            pk = result.pk_summary()

            results.append(
                FormulationResult(
                    name=name,
                    bioavailability=f_oral,
                    ka_h=round(
                        drug.peff
                        * ka_mult
                        * 2
                        * 3600
                        * 1e-4
                        / (drug.particle_radius_um * 1e-4)
                        * 0.01,
                        3,
                    ),
                    cmax=pk["Cmax_mg_L"],
                    tmax=pk["Tmax_h"],
                    auc=pk["AUC_mg_h_L"],
                )
            )

        return results
