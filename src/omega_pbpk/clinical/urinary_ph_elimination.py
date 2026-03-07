"""Urinary pH effect on renal elimination of ionizable drugs.

Models ion trapping: un-ionized drug is passively reabsorbed from urine;
ionized drug is trapped and excreted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UrinaryPhResult:
    drug_name: str
    ph_urine: float
    cl_renal_L_per_h: float
    cl_total_L_per_h: float
    t_half_h: float
    auc: float
    ionized_fraction_urine: float
    reabsorption_fraction: float
    f_excreted_urine: float
    notes: str


def ionized_fraction(ph: float, pka: float, drug_type: str) -> float:
    """Return the fraction of drug ionized at given urinary pH.

    Parameters
    ----------
    ph:
        Urinary pH.
    pka:
        Drug pKa.
    drug_type:
        'weak_acid' or 'weak_base'.

    Returns
    -------
    float
        Ionized fraction in [0, 1].
    """
    if drug_type not in ("weak_acid", "weak_base"):
        raise ValueError("drug_type must be 'weak_acid' or 'weak_base'")

    if drug_type == "weak_acid":
        # Henderson-Hasselbalch: fi = 1 / (1 + 10^(pKa - pH))
        return 1.0 / (1.0 + 10.0 ** (pka - ph))
    else:
        # weak_base: fi = 1 / (1 + 10^(pH - pKa))
        return 1.0 / (1.0 + 10.0 ** (ph - pka))


def reabsorption_fraction(
    ph_urine: float,
    pka: float,
    drug_type: str,
    f_reabsorption_max: float = 0.95,
) -> float:
    """Passive tubular reabsorption fraction.

    More un-ionized drug → more reabsorption.

    Parameters
    ----------
    ph_urine:
        Urinary pH.
    pka:
        Drug pKa.
    drug_type:
        'weak_acid' or 'weak_base'.
    f_reabsorption_max:
        Maximum reabsorption fraction when fully un-ionized.

    Returns
    -------
    float
        Reabsorption fraction in [0, 1].
    """
    fi = ionized_fraction(ph_urine, pka, drug_type)
    return f_reabsorption_max * (1.0 - fi)


def renal_cl_adjusted(
    cl_filtration_L_per_h: float,
    cl_secretion_L_per_h: float,
    ph_urine: float,
    pka: float,
    drug_type: str,
    f_reabsorption_max: float = 0.95,
) -> float:
    """Effective renal clearance accounting for pH-dependent reabsorption.

    CL_renal = (CL_filtration + CL_secretion) * (1 - reabsorption_fraction)

    Parameters
    ----------
    cl_filtration_L_per_h:
        GFR-based filtration clearance (L/h).
    cl_secretion_L_per_h:
        Active tubular secretion clearance (L/h).
    ph_urine:
        Urinary pH.
    pka:
        Drug pKa.
    drug_type:
        'weak_acid' or 'weak_base'.
    f_reabsorption_max:
        Maximum reabsorption fraction.

    Returns
    -------
    float
        Adjusted renal CL (L/h).
    """
    f_reabs = reabsorption_fraction(ph_urine, pka, drug_type, f_reabsorption_max)
    return (cl_filtration_L_per_h + cl_secretion_L_per_h) * (1.0 - f_reabs)


def simulate_urinary_ph_effect(
    drug_name: str,
    dose_mg: float,
    pka: float,
    drug_type: str,
    cl_nonrenal_L_per_h: float,
    vd_L: float,
    gfr_L_per_h: float = 7.2,
    cl_secretion_L_per_h: float = 0.0,
    ph_urine_values: list[float] | None = None,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list[UrinaryPhResult]:
    """Simulate 1-compartment PK at multiple urinary pH values.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    pka:
        Drug pKa.
    drug_type:
        'weak_acid' or 'weak_base'.
    cl_nonrenal_L_per_h:
        Non-renal (hepatic) clearance (L/h).
    vd_L:
        Volume of distribution (L).
    gfr_L_per_h:
        Glomerular filtration rate as clearance (L/h). Default 7.2 L/h (~120 mL/min).
    cl_secretion_L_per_h:
        Active secretion clearance (L/h). Default 0.
    ph_urine_values:
        List of urinary pH values to simulate. Defaults to [5.0, 6.0, 7.0, 8.0].
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step (h).

    Returns
    -------
    list[UrinaryPhResult]
        One result per pH value.
    """
    if drug_type not in ("weak_acid", "weak_base"):
        raise ValueError("drug_type must be 'weak_acid' or 'weak_base'")
    if ph_urine_values is None:
        ph_urine_values = [5.0, 6.0, 7.0, 8.0]

    results = []
    for ph in ph_urine_values:
        cl_renal = renal_cl_adjusted(gfr_L_per_h, cl_secretion_L_per_h, ph, pka, drug_type)
        cl_total = cl_nonrenal_L_per_h + cl_renal
        ke = cl_total / vd_L  # elimination rate constant (1/h)
        t_half = 0.693147 / ke if ke > 0 else float("inf")

        # Forward Euler 1-cpt IV bolus
        c = dose_mg / vd_L  # initial concentration (mg/L)
        auc = 0.0
        t = 0.0
        steps = int(t_end_h / dt_h)
        for _ in range(steps):
            c_prev = c
            dC_dt = -ke * c
            c = c + dC_dt * dt_h
            if c < 0.0:
                c = 0.0
            # Trapezoidal AUC
            auc += 0.5 * (c_prev + c) * dt_h
            t += dt_h

        # Fraction excreted in urine
        f_excreted = cl_renal / cl_total if cl_total > 0 else 0.0

        fi = ionized_fraction(ph, pka, drug_type)
        f_reabs = reabsorption_fraction(ph, pka, drug_type)

        # Clinical notes
        if drug_type == "weak_acid":
            if ph >= 7.0:
                note = "Alkaline urine: high ionization traps weak acid → increased renal CL"
            else:
                note = "Acidic urine: low ionization → more reabsorption → reduced renal CL"
        else:
            if ph <= 6.0:
                note = "Acidic urine: high ionization traps weak base → increased renal CL"
            else:
                note = "Alkaline urine: low ionization → more reabsorption → reduced renal CL"

        results.append(
            UrinaryPhResult(
                drug_name=drug_name,
                ph_urine=ph,
                cl_renal_L_per_h=cl_renal,
                cl_total_L_per_h=cl_total,
                t_half_h=t_half,
                auc=auc,
                ionized_fraction_urine=fi,
                reabsorption_fraction=f_reabs,
                f_excreted_urine=f_excreted,
                notes=note,
            )
        )
    return results


def compare_acidic_alkaline(
    drug_name: str,
    dose_mg: float,
    pka: float,
    drug_type: str,
    cl_nonrenal_L_per_h: float,
    vd_L: float,
    **kwargs,
) -> dict:
    """Compare renal elimination at acidic (pH 5.0) vs alkaline (pH 8.0) urine.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    pka:
        Drug pKa.
    drug_type:
        'weak_acid' or 'weak_base'.
    cl_nonrenal_L_per_h:
        Non-renal clearance (L/h).
    vd_L:
        Volume of distribution (L).
    **kwargs:
        Additional keyword arguments passed to simulate_urinary_ph_effect.

    Returns
    -------
    dict with keys:
        acidic_result, alkaline_result, auc_ratio (alkaline/acidic),
        t_half_ratio (alkaline/acidic), clinical_relevance.
    """
    results = simulate_urinary_ph_effect(
        drug_name=drug_name,
        dose_mg=dose_mg,
        pka=pka,
        drug_type=drug_type,
        cl_nonrenal_L_per_h=cl_nonrenal_L_per_h,
        vd_L=vd_L,
        ph_urine_values=[5.0, 8.0],
        **kwargs,
    )
    acidic = results[0]  # pH 5.0
    alkaline = results[1]  # pH 8.0

    auc_ratio = alkaline.auc / acidic.auc if acidic.auc > 0 else float("inf")
    t_half_ratio = alkaline.t_half_h / acidic.t_half_h if acidic.t_half_h > 0 else float("inf")

    # Determine clinical relevance based on magnitude of difference
    if abs(auc_ratio - 1.0) >= 0.25:
        relevance = "Clinically relevant: urinary pH manipulation significantly alters exposure"
    elif abs(auc_ratio - 1.0) >= 0.10:
        relevance = "Potentially relevant: moderate effect of urinary pH on exposure"
    else:
        relevance = "Minimal clinical relevance: urinary pH has little effect on overall exposure"

    return {
        "acidic_result": acidic,
        "alkaline_result": alkaline,
        "auc_ratio": auc_ratio,
        "t_half_ratio": t_half_ratio,
        "clinical_relevance": relevance,
    }
