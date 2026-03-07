"""Phase 381 — Placental Drug Transfer Model.

Models drug transfer across the placenta from maternal to fetal compartment
using a 2-compartment (maternal plasma + fetal) forward Euler ODE approach.

Model structure
---------------
Maternal plasma (Am, mg):
  dAm/dt = Rate_in(t) - (CL_m/Vd_m)*Am - k_trans*fu_m*Am/Vd_m * Vd_m
         = Rate_in(t) - ke_m*Am - k_trans*fu_m*Cm

Fetal compartment (Af, mg):
  dAf/dt = k_trans*fu_m*Cm*Vd_f/1 ... (see below)

Working in concentrations:
  Cm = Am / Vd_m
  Cf = Af / Vd_f

  dCm/dt = input_rate/Vd_m - ke_m*Cm - k_trans*fu_m*Cm
  dCf/dt = k_trans*fu_m*Cm*(Vd_m/Vd_f) - ke_f*Cf   [mass balance transfer]

  BUT spec says:
    Placental transfer rate (maternal→fetal):
        k_trans = placental_permeability * fu_maternal * c_maternal
    This is a rate in mg/h/L ... need to be careful with units.

  We interpret k_trans as a first-order-like term:
    Transfer flux (mg/h) = placental_permeability * fu_maternal * Cm * Vd_m
    (permeability here has units L/h effectively)

  dCm/dt = input_rate/Vd_m - ke_m*Cm - placental_permeability*fu_m*Cm
  dCf/dt = placental_permeability*fu_m*Cm*(Vd_m/Vd_f) - ke_f*Cf

References
----------
- Ke AB et al., Clin Pharmacol Ther. 2012;91(6):1017-31
- De Sousa Mendes M et al., Br J Clin Pharmacol. 2015;80(6):1340-50
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlacentalTransferResult:
    """Results from placental drug transfer simulation.

    Attributes
    ----------
    drug_name              : Name of the drug.
    dose_mg                : Administered maternal dose (mg).
    fetal_weight_kg        : Fetal body weight (kg).
    placental_permeability : Placental permeability coefficient (L/h).
    times_h                : Simulation time points (h).
    c_maternal_mg_L        : Maternal plasma concentration profile (mg/L).
    c_fetal_mg_L           : Fetal plasma concentration profile (mg/L).
    auc_maternal           : Maternal AUC 0->t_end (mg·h/L).
    auc_fetal              : Fetal AUC 0->t_end (mg·h/L).
    fetal_maternal_auc_ratio : AUC_fetal / AUC_maternal.
    cmax_maternal          : Maternal peak concentration (mg/L).
    cmax_fetal             : Fetal peak concentration (mg/L).
    fetal_maternal_cmax_ratio : Cmax_fetal / Cmax_maternal.
    exposure_risk          : Risk classification string.
    notes                  : Human-readable summary.
    """

    drug_name: str
    dose_mg: float
    fetal_weight_kg: float
    placental_permeability: float
    times_h: list[float] = field(default_factory=list)
    c_maternal_mg_L: list[float] = field(default_factory=list)
    c_fetal_mg_L: list[float] = field(default_factory=list)
    auc_maternal: float = 0.0
    auc_fetal: float = 0.0
    fetal_maternal_auc_ratio: float = 0.0
    cmax_maternal: float = 0.0
    cmax_fetal: float = 0.0
    fetal_maternal_cmax_ratio: float = 0.0
    exposure_risk: str = ""
    notes: str = ""


def simulate_placental_transfer(
    drug_name: str,
    dose_mg: float,
    cl_maternal_L_per_h: float,
    vd_maternal_L: float,
    fetal_weight_kg: float,
    placental_permeability: float,
    fu_maternal: float,
    fu_fetal: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    route: str = "iv",
    ka_per_h: float = 1.0,
) -> PlacentalTransferResult:
    """Simulate maternal and fetal drug concentrations with placental transfer.

    Uses forward Euler ODE integration with a 2-compartment model:
    maternal plasma and fetal compartment.

    Placental transfer flux (mg/h):
        J = placental_permeability * fu_maternal * Cm * Vd_m

    Parameters
    ----------
    drug_name              : Name of the drug.
    dose_mg                : Maternal dose (mg). Must be > 0.
    cl_maternal_L_per_h    : Maternal systemic clearance (L/h). Must be > 0.
    vd_maternal_L          : Maternal volume of distribution (L). Must be > 0.
    fetal_weight_kg        : Fetal body weight (kg). Must be > 0.
    placental_permeability : Placental permeability coefficient (L/h). Must be >= 0.
    fu_maternal            : Unbound fraction in maternal plasma (0, 1]. Must be in (0, 1].
    fu_fetal               : Unbound fraction in fetal plasma (0, 1]. Must be in (0, 1].
    t_end_h                : Simulation duration (h). Must be > 0.
    dt_h                   : Forward Euler time step (h). Must be > 0.
    route                  : 'iv' (bolus) or 'oral' (first-order absorption).
    ka_per_h               : First-order oral absorption rate constant (h^-1). Must be > 0.

    Returns
    -------
    PlacentalTransferResult with full time-course and summary metrics.

    Raises
    ------
    ValueError : If any required parameter is invalid.
    """
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_maternal_L_per_h <= 0:
        raise ValueError("cl_maternal_L_per_h must be > 0")
    if vd_maternal_L <= 0:
        raise ValueError("vd_maternal_L must be > 0")
    if fetal_weight_kg <= 0:
        raise ValueError("fetal_weight_kg must be > 0")
    if placental_permeability < 0:
        raise ValueError("placental_permeability must be >= 0")
    if not (0 < fu_maternal <= 1):
        raise ValueError("fu_maternal must be in (0, 1]")
    if not (0 < fu_fetal <= 1):
        raise ValueError("fu_fetal must be in (0, 1]")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")

    # Fetal PK parameters
    # Fetal Vd: 0.8 L/kg (as specified)
    vd_fetal_L = 0.8 * fetal_weight_kg
    # Fetal clearance: minimal — 0.1 * cl_maternal scaled by fetal/reference weight (3 kg)
    cl_fetal_L_per_h = 0.1 * cl_maternal_L_per_h * (fetal_weight_kg / 3.0)

    # Rate constants
    ke_m = cl_maternal_L_per_h / vd_maternal_L  # h^-1
    ke_f = cl_fetal_L_per_h / vd_fetal_L  # h^-1

    # Placental transfer: rate constant for maternal conc removal
    # Transfer flux (mg/h) = placental_permeability * fu_maternal * Cm * Vd_m
    # => dCm/dt contribution = -placental_permeability * fu_maternal * Cm
    # => dCf/dt contribution = +placental_permeability * fu_maternal * Cm * (Vd_m / Vd_f)
    k_perm_m = placental_permeability * fu_maternal  # h^-1 effective rate from maternal side
    vd_ratio = vd_maternal_L / vd_fetal_L

    n_steps = max(int(round(t_end_h / dt_h)), 1)
    times = [i * dt_h for i in range(n_steps + 1)]

    cm_arr = [0.0] * (n_steps + 1)
    cf_arr = [0.0] * (n_steps + 1)

    # Initial conditions
    if route == "iv":
        cm_arr[0] = dose_mg / vd_maternal_L
        depot = 0.0
    else:
        cm_arr[0] = 0.0
        depot = dose_mg

    # Forward Euler integration
    for i in range(n_steps):
        cm = cm_arr[i]
        cf = cf_arr[i]

        # Oral absorption
        if route == "oral":
            abs_flux_mg_per_h = ka_per_h * depot
            depot -= abs_flux_mg_per_h * dt_h
            depot = max(0.0, depot)
            input_rate_conc = abs_flux_mg_per_h / vd_maternal_L  # mg/L/h
        else:
            input_rate_conc = 0.0

        # Transfer flux in concentration units
        transfer_from_m = k_perm_m * cm  # mg/L/h removed from maternal

        # ODEs
        dcm = (input_rate_conc - ke_m * cm - transfer_from_m) * dt_h
        dcf = (transfer_from_m * vd_ratio - ke_f * cf) * dt_h

        cm_arr[i + 1] = max(0.0, cm + dcm)
        cf_arr[i + 1] = max(0.0, cf + dcf)

    # Trapezoidal AUC
    auc_maternal = _trapz(times, cm_arr)
    auc_fetal = _trapz(times, cf_arr)

    cmax_maternal = max(cm_arr)
    cmax_fetal = max(cf_arr)

    fetal_maternal_auc_ratio = auc_fetal / auc_maternal if auc_maternal > 1e-12 else 0.0
    fetal_maternal_cmax_ratio = cmax_fetal / cmax_maternal if cmax_maternal > 1e-12 else 0.0

    exposure_risk = fetal_drug_exposure_risk(fetal_maternal_auc_ratio, drug_name)

    notes = (
        f"Placental transfer simulation: {drug_name}, dose={dose_mg} mg {route}. "
        f"Fetal weight={fetal_weight_kg} kg, placental_permeability={placental_permeability} L/h. "
        f"Fetal/maternal AUC ratio={fetal_maternal_auc_ratio:.3f}. "
        f"Fetal Vd={vd_fetal_L:.2f} L, Fetal CL={cl_fetal_L_per_h:.4f} L/h. "
        f"Exposure risk: {exposure_risk}."
    )

    return PlacentalTransferResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fetal_weight_kg=fetal_weight_kg,
        placental_permeability=placental_permeability,
        times_h=times,
        c_maternal_mg_L=cm_arr,
        c_fetal_mg_L=cf_arr,
        auc_maternal=auc_maternal,
        auc_fetal=auc_fetal,
        fetal_maternal_auc_ratio=fetal_maternal_auc_ratio,
        cmax_maternal=cmax_maternal,
        cmax_fetal=cmax_fetal,
        fetal_maternal_cmax_ratio=fetal_maternal_cmax_ratio,
        exposure_risk=exposure_risk,
        notes=notes,
    )


def fetal_drug_exposure_risk(fetal_maternal_auc_ratio: float, drug_name: str) -> str:
    """Classify fetal drug exposure risk based on fetal/maternal AUC ratio.

    Parameters
    ----------
    fetal_maternal_auc_ratio : AUC_fetal / AUC_maternal. Must be >= 0.
    drug_name                : Name of the drug (used for context in output).

    Returns
    -------
    Risk classification: 'low', 'moderate', or 'high'.

    Raises
    ------
    ValueError : If fetal_maternal_auc_ratio is negative.
    """
    if fetal_maternal_auc_ratio < 0:
        raise ValueError("fetal_maternal_auc_ratio must be >= 0")
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")

    if fetal_maternal_auc_ratio < 0.1:
        return "low"
    elif fetal_maternal_auc_ratio <= 0.5:
        return "moderate"
    else:
        return "high"


def _trapz(x: list[float], y: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    total = 0.0
    for i in range(len(x) - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


__all__ = [
    "PlacentalTransferResult",
    "simulate_placental_transfer",
    "fetal_drug_exposure_risk",
]
