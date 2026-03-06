"""Mechanistic renal tubular secretion and reabsorption model.

Models glomerular filtration, active tubular secretion (Michaelis-Menten),
and passive reabsorption to compute net renal clearance and fe_urine.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TubularResult", "simulate_tubular_handling", "renal_clearance_components"]

_VALID_TRANSPORTERS = {"OAT1", "OAT3", "OCT2", "MATE1", "P-gp", "passive"}


@dataclass(frozen=True)
class TubularResult:
    """Result of a renal tubular handling simulation."""

    drug_name: str
    gfr_mL_per_min: float
    cl_filtration_mL_per_min: float
    cl_secretion_mL_per_min: float  # active secretion CL
    cl_reabsorption_mL_per_min: float  # magnitude of reabsorption reduction
    cl_renal_total_mL_per_min: float  # net renal CL
    fe_urine: float  # fraction excreted unchanged in urine
    secretion_saturation_pct: float  # % saturation of secretion transporter
    transporter: str  # "OAT1", "OAT3", "OCT2", "MATE1", "P-gp" or "passive"
    notes: str


def simulate_tubular_handling(
    drug_name: str,
    plasma_conc_mg_L: float,
    gfr_mL_per_min: float = 120.0,
    fu_plasma: float = 0.1,
    vmax_secretion_mg_per_min: float = 0.5,
    km_secretion_mg_L: float = 1.0,
    f_reabsorption: float = 0.3,
    transporter: str = "OAT1",
    cl_hepatic_L_per_h: float = 5.0,
    dose_mg: float = 100.0,
) -> TubularResult:
    """Simulate renal tubular handling at a single plasma concentration.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    plasma_conc_mg_L : float
        Unbound plasma concentration (mg/L). Must be >= 0.
    gfr_mL_per_min : float
        Glomerular filtration rate (mL/min). Must be >= 0.
    fu_plasma : float
        Fraction unbound in plasma, in (0, 1].
    vmax_secretion_mg_per_min : float
        Vmax for active tubular secretion transporter (mg/min). Must be > 0.
    km_secretion_mg_L : float
        Km for secretion transporter (mg/L). Must be > 0.
    f_reabsorption : float
        Passive reabsorption fraction (0–1).
    transporter : str
        Secretion transporter name.
    cl_hepatic_L_per_h : float
        Hepatic clearance (L/h), used to estimate fe_urine.
    dose_mg : float
        Dose (mg), used for context in notes only.

    Returns
    -------
    TubularResult
    """
    # --- Validation ---
    if plasma_conc_mg_L < 0:
        raise ValueError("plasma_conc_mg_L must be >= 0")
    if gfr_mL_per_min < 0:
        raise ValueError("gfr_mL_per_min must be >= 0")
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError("fu_plasma must be in (0, 1]")
    if vmax_secretion_mg_per_min <= 0:
        raise ValueError("vmax_secretion_mg_per_min must be > 0")
    if km_secretion_mg_L <= 0:
        raise ValueError("km_secretion_mg_L must be > 0")
    if not 0.0 <= f_reabsorption <= 1.0:
        raise ValueError("f_reabsorption must be in [0, 1]")

    # --- Filtration clearance (mL/min) ---
    cl_filt = gfr_mL_per_min * fu_plasma

    # --- Active secretion (Michaelis-Menten) ---
    # At zero concentration, secretion rate = 0 so cl_sec = Vmax/Km (linear limit)
    if plasma_conc_mg_L > 0:
        secretion_rate = (
            vmax_secretion_mg_per_min
            * plasma_conc_mg_L
            / (km_secretion_mg_L + plasma_conc_mg_L)
        )  # mg/min
        cl_sec = secretion_rate / plasma_conc_mg_L  # mL/min
    else:
        # Linear (unsaturated) limit: Vmax / Km
        cl_sec = vmax_secretion_mg_per_min / km_secretion_mg_L  # mL/min

    # Saturation percentage
    secretion_saturation_pct = (
        plasma_conc_mg_L / (km_secretion_mg_L + plasma_conc_mg_L) * 100.0
        if plasma_conc_mg_L > 0
        else 0.0
    )

    # --- Reabsorption (passive) ---
    cl_reabs = (cl_filt + cl_sec) * f_reabsorption  # mL/min reduction

    # --- Net renal CL ---
    cl_renal = max(0.0, cl_filt + cl_sec - cl_reabs)

    # --- fe_urine approximation ---
    # Convert hepatic CL from L/h to mL/min: L/h * 1000/60
    cl_hepatic_mL_per_min = cl_hepatic_L_per_h * 1000.0 / 60.0
    cl_total = cl_renal + cl_hepatic_mL_per_min
    fe_urine = cl_renal / cl_total if cl_total > 0 else 0.0
    fe_urine = min(1.0, max(0.0, fe_urine))

    sat_label = "low" if secretion_saturation_pct < 20 else (
        "moderate" if secretion_saturation_pct < 60 else "high"
    )
    notes = (
        f"Transporter: {transporter}; "
        f"secretion saturation {secretion_saturation_pct:.1f}% ({sat_label}); "
        f"reabsorption fraction {f_reabsorption:.2f}; "
        f"net CLrenal {cl_renal:.2f} mL/min"
    )

    return TubularResult(
        drug_name=drug_name,
        gfr_mL_per_min=gfr_mL_per_min,
        cl_filtration_mL_per_min=cl_filt,
        cl_secretion_mL_per_min=cl_sec,
        cl_reabsorption_mL_per_min=cl_reabs,
        cl_renal_total_mL_per_min=cl_renal,
        fe_urine=fe_urine,
        secretion_saturation_pct=secretion_saturation_pct,
        transporter=transporter,
        notes=notes,
    )


def renal_clearance_components(
    drug_name: str,
    plasma_conc_range_mg_L: list[float],
    gfr_mL_per_min: float = 120.0,
    **kwargs,
) -> list[TubularResult]:
    """Simulate tubular handling at each concentration to show secretion saturation.

    Parameters
    ----------
    drug_name : str
        Drug name.
    plasma_conc_range_mg_L : list[float]
        List of plasma concentrations (mg/L) to evaluate.
    gfr_mL_per_min : float
        Glomerular filtration rate (mL/min).
    **kwargs
        Additional keyword arguments forwarded to :func:`simulate_tubular_handling`.

    Returns
    -------
    list[TubularResult]
        One result per concentration in the input list.
    """
    return [
        simulate_tubular_handling(
            drug_name=drug_name,
            plasma_conc_mg_L=c,
            gfr_mL_per_min=gfr_mL_per_min,
            **kwargs,
        )
        for c in plasma_conc_range_mg_L
    ]
