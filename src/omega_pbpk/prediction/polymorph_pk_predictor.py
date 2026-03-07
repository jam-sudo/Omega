"""Drug crystal polymorph PK predictor.

Models how different crystal polymorphs affect dissolution rate and oral PK.
Polymorphs differ in lattice energy -> different intrinsic solubility ->
different dissolution rate (Noyes-Whitney).

Dissolution model:
  f_diss(t) = min(1, 1 - exp(-0.1 * rsf * t))

PK model:
  1-compartment oral with absorbed_dose = dose_mg * f_oral * f_diss(4h)
  C(t) = (F * Dose_abs * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))

Relative solubility factors (rsf) vs Form I:
  Form I (stable):      rsf = 1.0
  Form II (metastable): rsf = 1.5-3.0
  Amorphous:            rsf = 10-1000

References
----------
- Pudipeddi M, Serajuddin ATM. J Pharm Sci. 2005;94(5):929-939.
- Bernstein J. Polymorphism in Molecular Crystals. 2002.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PolymorphPKResult:
    """PK result for a single crystal polymorph.

    Attributes
    ----------
    drug_name : Name of the drug compound.
    polymorph_name : Identifier for the polymorph (e.g., 'Form I', 'Amorphous').
    rsf : Relative solubility factor vs Form I (unitless).
    dose_mg : Administered dose (mg).
    f_dissolved_2h : Fraction dissolved at 2 hours (0-1).
    f_dissolved_4h : Fraction dissolved at 4 hours (0-1).
    cmax_mg_L : Peak plasma concentration (mg/L).
    tmax_h : Time of peak plasma concentration (h).
    auc_mg_h_per_L : Area under the concentration-time curve (mg*h/L).
    bioavailability_relative : AUC relative to Form I (1.0 for Form I baseline).
    stability_class : 'stable', 'metastable', or 'amorphous'.
    notes : Summary text describing key findings.
    """

    drug_name: str
    polymorph_name: str
    rsf: float
    dose_mg: float
    f_dissolved_2h: float
    f_dissolved_4h: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    bioavailability_relative: float
    stability_class: str
    notes: str


def _f_dissolved(rsf: float, t_h: float) -> float:
    """Dissolved fraction at time t using Noyes-Whitney simplified model.

    f_diss(t) = min(1.0, 1 - exp(-0.1 * rsf * t))
    """
    return min(1.0, 1.0 - math.exp(-0.1 * rsf * t_h))


def _stability_class(rsf: float) -> str:
    """Classify polymorph stability based on relative solubility factor."""
    if rsf >= 10.0:
        return "amorphous"
    if rsf > 1.0:
        return "metastable"
    return "stable"


def _simulate_1cpt_pk(
    absorbed_dose_mg: float,
    ka_per_h: float,
    ke_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Simulate 1-compartment oral PK using analytical solution with Forward Euler AUC.

    Uses analytical formula:
      C(t) = (Dose_abs * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))

    Falls back to monoexponential if ka == ke (flip approximation).
    """
    times: list[float] = []
    concs: list[float] = []

    n_steps = max(int(t_end_h / dt_h), 1)
    eps = 1e-9

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        if t == 0.0:
            concs.append(0.0)
        else:
            if abs(ka_per_h - ke_per_h) < eps:
                # Avoid division by zero: limit case ka -> ke
                # C(t) = (Dose_abs * ka / Vd) * t * exp(-ke*t)
                c = (absorbed_dose_mg * ka_per_h / vd_L) * t * math.exp(-ke_per_h * t)
            else:
                c = (
                    (absorbed_dose_mg * ka_per_h)
                    / (vd_L * (ka_per_h - ke_per_h))
                    * (math.exp(-ke_per_h * t) - math.exp(-ka_per_h * t))
                )
            concs.append(max(0.0, c))

    return times, concs


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC using manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def _find_cmax_tmax(times: list[float], concs: list[float]) -> tuple[float, float]:
    """Return (cmax, tmax_h) from concentration profile."""
    cmax = 0.0
    tmax_h = 0.0
    for t, c in zip(times, concs):
        if c > cmax:
            cmax = c
            tmax_h = t
    return cmax, tmax_h


def predict_polymorph_pk(
    drug_name: str,
    dose_mg: float,
    polymorph_name: str,
    rsf: float,
    ka_per_h: float = 1.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_oral: float = 0.8,
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> PolymorphPKResult:
    """Predict PK parameters for a drug crystal polymorph.

    Parameters
    ----------
    drug_name : Name of the drug compound.
    dose_mg : Oral dose administered (mg).
    polymorph_name : Label for this polymorph form (e.g., 'Form I', 'Form II').
    rsf : Relative solubility factor vs Form I (1.0 = Form I baseline).
    ka_per_h : First-order absorption rate constant (1/h).
    cl_L_per_h : Total clearance (L/h).
    vd_L : Volume of distribution (L).
    f_oral : Maximum oral bioavailability when fully dissolved (fraction 0-1).
    t_end_h : Simulation end time (h).
    dt_h : Time step for simulation (h).

    Returns
    -------
    PolymorphPKResult with PK parameters, dissolution fractions, and
    relative bioavailability vs Form I.

    Raises
    ------
    ValueError : If dose_mg, rsf, cl_L_per_h, or vd_L <= 0.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if rsf <= 0:
        raise ValueError("rsf must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    # Dissolution fractions at 2h and 4h
    f_diss_2h = _f_dissolved(rsf, 2.0)
    f_diss_4h = _f_dissolved(rsf, 4.0)

    # Effective absorbed dose: use f_dissolved at 4h as effective absorbed fraction
    absorbed_dose_mg = dose_mg * f_oral * f_diss_4h

    # Elimination rate constant
    ke_per_h = cl_L_per_h / vd_L

    # Simulate 1-compartment PK
    times, concs = _simulate_1cpt_pk(
        absorbed_dose_mg=absorbed_dose_mg,
        ka_per_h=ka_per_h,
        ke_per_h=ke_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    auc = _trapezoidal_auc(times, concs)
    cmax, tmax_h = _find_cmax_tmax(times, concs)

    # Compute Form I baseline for relative bioavailability
    f_diss_4h_form1 = _f_dissolved(1.0, 4.0)
    absorbed_dose_form1 = dose_mg * f_oral * f_diss_4h_form1
    times_form1, concs_form1 = _simulate_1cpt_pk(
        absorbed_dose_mg=absorbed_dose_form1,
        ka_per_h=ka_per_h,
        ke_per_h=ke_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    auc_form1 = _trapezoidal_auc(times_form1, concs_form1)

    if auc_form1 > 0:
        bioavailability_relative = auc / auc_form1
    else:
        bioavailability_relative = 1.0

    # Override to exactly 1.0 for Form I baseline (rsf == 1.0)
    if abs(rsf - 1.0) < 1e-9:
        bioavailability_relative = 1.0

    stab_class = _stability_class(rsf)

    notes = (
        f"{polymorph_name} (rsf={rsf:.2f}): stability={stab_class}, "
        f"f_diss_2h={f_diss_2h:.3f}, f_diss_4h={f_diss_4h:.3f}, "
        f"Cmax={cmax:.4f} mg/L, AUC={auc:.4f} mg*h/L, "
        f"relative_BA={bioavailability_relative:.3f}"
    )

    return PolymorphPKResult(
        drug_name=drug_name,
        polymorph_name=polymorph_name,
        rsf=rsf,
        dose_mg=dose_mg,
        f_dissolved_2h=f_diss_2h,
        f_dissolved_4h=f_diss_4h,
        cmax_mg_L=cmax,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        bioavailability_relative=bioavailability_relative,
        stability_class=stab_class,
        notes=notes,
    )


def compare_polymorphs(
    drug_name: str,
    dose_mg: float,
    polymorphs: list[dict],
    ka_per_h: float = 1.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_oral: float = 0.8,
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> list[PolymorphPKResult]:
    """Compare PK across multiple crystal polymorphs.

    Parameters
    ----------
    drug_name : Name of the drug compound.
    dose_mg : Oral dose (mg).
    polymorphs : List of dicts, each with keys 'name' (str) and 'rsf' (float).
    ka_per_h : Absorption rate constant (1/h).
    cl_L_per_h : Total clearance (L/h).
    vd_L : Volume of distribution (L).
    f_oral : Maximum oral bioavailability (fraction).
    t_end_h : Simulation end time (h).
    dt_h : Time step (h).

    Returns
    -------
    List of PolymorphPKResult sorted by auc_mg_h_per_L descending.
    """
    results: list[PolymorphPKResult] = []
    for poly in polymorphs:
        result = predict_polymorph_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            polymorph_name=poly["name"],
            rsf=poly["rsf"],
            ka_per_h=ka_per_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            f_oral=f_oral,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results


__all__ = [
    "PolymorphPKResult",
    "predict_polymorph_pk",
    "compare_polymorphs",
]
