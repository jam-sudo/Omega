"""Population PK simulator — runs PBPK for N virtual subjects with covariate scaling."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class PopPKResult:
    """Population PK simulation results."""
    time_h: NDArray[np.float64]
    cp_matrix: NDArray[np.float64]       # shape (n_subjects, n_timepoints)
    cmax_samples: NDArray[np.float64]    # shape (n_subjects,)
    auc_samples: NDArray[np.float64]     # shape (n_subjects,)
    t_half_samples: NDArray[np.float64]  # shape (n_subjects,)
    n_subjects: int
    n_failed: int

    # Derived statistics (computed from cp_matrix)
    def median_cp(self) -> NDArray[np.float64]:
        return np.median(self.cp_matrix, axis=0)

    def percentile_cp(self, p: float) -> NDArray[np.float64]:
        return np.percentile(self.cp_matrix, p, axis=0)

    def cmax_stats(self) -> dict[str, float]:
        c = self.cmax_samples
        return {"mean": float(np.mean(c)), "median": float(np.median(c)),
                "p5": float(np.percentile(c, 5)), "p95": float(np.percentile(c, 95)),
                "cv_pct": float(np.std(c) / np.mean(c) * 100)}

    def auc_stats(self) -> dict[str, float]:
        a = self.auc_samples
        return {"mean": float(np.mean(a)), "median": float(np.median(a)),
                "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
                "cv_pct": float(np.std(a) / np.mean(a) * 100)}


def _run_subject_sim(args: tuple) -> tuple[int, NDArray | None, float, float, float]:
    """Worker: simulate one subject. Returns (idx, cp, cmax, auc, t_half)."""
    idx, drug_params, phys_params, dose_mg, route, n_points, t_end_h = args
    try:
        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.drugs.drug import Drug

        drug = Drug(**drug_params)

        # Apply covariate scaling to clearances
        gfr_scale = phys_params.get("gfr_scale", 1.0)
        hepatic_scale = phys_params.get("hepatic_scale", 1.0)
        # Modify CLr and CLint via drug parameters
        scaled_params = dict(drug_params)
        scaled_params["clint_hepatic_L_per_h"] = drug.clint_hepatic_L_per_h * hepatic_scale
        scaled_params["clr_L_per_h"] = drug.clr_L_per_h * gfr_scale
        scaled_drug = Drug(**scaled_params)

        model = WholeBodyPBPK(drug=scaled_drug, body_weight=phys_params.get("body_weight", 70.0))

        if route == "iv":
            model.setup_iv(dose_mg=dose_mg)
        else:
            model.setup_oral(dose_mg=dose_mg)

        # Use dt_h to produce exactly n_points time steps over [0, t_end_h]
        dt_h = t_end_h / (n_points - 1) if n_points > 1 else t_end_h
        result = model.simulate(t_end_h=t_end_h, dt_h=dt_h)
        cp_raw = result.plasma_concentration()

        # Interpolate to exactly n_points if solver returned a different length
        t_raw = result.time_h
        if len(cp_raw) != n_points:
            t_target = np.linspace(0, t_end_h, n_points)
            cp = np.interp(t_target, t_raw, cp_raw)
        else:
            cp = cp_raw

        t = np.linspace(0, t_end_h, n_points)

        cmax = float(np.max(cp))
        auc = float(np.trapezoid(cp, t))
        n = len(t)
        t0 = int(0.7 * n)
        if np.all(cp[t0:] > 1e-9):
            slope, _ = np.polyfit(t[t0:], np.log(cp[t0:] + 1e-12), 1)
            t_half = float(-np.log(2) / slope) if slope < -1e-9 else 99.0
        else:
            t_half = 99.0

        return idx, cp, cmax, auc, min(t_half, 99.0)
    except Exception as e:
        logger.debug(f"Subject {idx} failed: {e}")
        return idx, None, 0.0, 0.0, 0.0


class PopulationSimulator:
    """Run PBPK simulation across a virtual population.

    Samples N subjects from VirtualPopulation, scales physiological
    parameters per individual covariates, runs parallel PBPK simulations.

    Example:
        sim = PopulationSimulator(drug=midazolam_drug)
        result = sim.run(n_subjects=100, dose_mg=5.0, route="oral")
        print(result.cmax_stats())
    """

    def __init__(self, drug: Any) -> None:
        self.drug = drug

    def run(
        self,
        n_subjects: int = 100,
        dose_mg: float = 100.0,
        route: str = "oral",
        t_end_h: float = 24.0,
        n_timepoints: int = 121,
        seed: int | None = None,
        n_workers: int = 1,
    ) -> PopPKResult:
        """Simulate PK for N virtual subjects.

        Args:
            n_subjects: Number of virtual subjects to simulate
            dose_mg: Administered dose (mg)
            route: "oral" or "iv"
            t_end_h: Simulation end time (h)
            n_timepoints: Number of time points
            seed: Random seed for subject sampling
            n_workers: Parallel workers (1 = sequential, safer)
        """
        from omega_pbpk.population.physiology import VirtualPopulation

        # Sample virtual subjects
        vp = VirtualPopulation()
        subjects = vp.sample(n_subjects, seed=seed)

        # Build Drug params dict (picklable) — use actual Drug field names
        drug = self.drug
        drug_params: dict[str, Any] = {
            "name": getattr(drug, "name", "compound"),
            "mw": drug.mw,
            "logP": drug.logP,
            "fup": drug.fup,
            "rbp": drug.rbp,
            "pka": list(drug.pka) if drug.pka else [7.0],
            "clint_hepatic_L_per_h": drug.clint_hepatic_L_per_h,
            "clr_L_per_h": drug.clr_L_per_h,
        }
        # Carry over additional optional fields if they are non-default
        for attr in (
            "drug_type", "smiles", "clint", "fm", "clint_gut_L_per_h",
            "peff", "solubility_mg_mL", "particle_radius_um", "particle_density",
            "kp", "permeability_limited", "partition_method", "compound_type",
            "gut_clint_multiplier", "ka_sc", "f_sc",
        ):
            if hasattr(drug, attr):
                val = getattr(drug, attr)
                # Convert frozenset/mappingproxy-like fields to plain dicts/lists
                if hasattr(val, "items"):
                    val = dict(val)
                drug_params[attr] = val

        # Child-Pugh → hepatic scaling
        cp_scale = {"normal": 1.0, "A": 0.75, "B": 0.50, "C": 0.25}

        # Build args for each subject
        args_list = []
        for i, subj in enumerate(subjects):
            phys = {
                "body_weight": subj.body_weight_kg,
                "gfr_scale": subj.gfr_mL_min / 100.0,
                "hepatic_scale": cp_scale.get(subj.child_pugh, 1.0) * subj.cyp3a4_activity,
            }
            args_list.append((i, drug_params, phys, dose_mg, route, n_timepoints, t_end_h))

        # Run simulations (sequential for stability;
        # ProcessPoolExecutor has pickling issues with our ODE)
        n_t = n_timepoints
        cp_matrix = np.zeros((n_subjects, n_t))
        cmax_arr = np.zeros(n_subjects)
        auc_arr = np.zeros(n_subjects)
        t_half_arr = np.zeros(n_subjects)
        n_failed = 0

        for args in args_list:
            idx, cp, cmax, auc, t_half = _run_subject_sim(args)
            if cp is not None and len(cp) == n_t:
                cp_matrix[idx] = cp
                cmax_arr[idx] = cmax
                auc_arr[idx] = auc
                t_half_arr[idx] = t_half
            else:
                n_failed += 1

        # Remove failed subjects
        valid = cmax_arr > 0
        cp_matrix = cp_matrix[valid]
        cmax_arr = cmax_arr[valid]
        auc_arr = auc_arr[valid]
        t_half_arr = t_half_arr[valid]

        time_h = np.linspace(0, t_end_h, n_t)
        logger.info(f"PopPK: {valid.sum()}/{n_subjects} succeeded, {n_failed} failed")

        return PopPKResult(
            time_h=time_h,
            cp_matrix=cp_matrix,
            cmax_samples=cmax_arr,
            auc_samples=auc_arr,
            t_half_samples=t_half_arr,
            n_subjects=int(valid.sum()),
            n_failed=n_failed,
        )
