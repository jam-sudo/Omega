"""
Population PK covariate modeling.

Implements standard covariate effect models (power, exponential, categorical)
and a deterministic individual PK parameter calculator.
"""

from __future__ import annotations

import math


def power_covariate(
    param_typical: float,
    covariate_value: float,
    covariate_reference: float,
    exponent: float = 0.75,
) -> float:
    """Apply a power-law covariate effect.

    param_individual = param_typical * (covariate / reference) ^ exponent

    Commonly used for weight on CL (exponent=0.75) and Vd (exponent=1.0).

    Parameters
    ----------
    param_typical : float
        Population typical parameter value.
    covariate_value : float
        Individual covariate value.
    covariate_reference : float
        Reference (median) covariate value.
    exponent : float
        Allometric exponent.

    Returns
    -------
    float
        Individual parameter value.
    """
    if covariate_reference <= 0:
        raise ValueError("covariate_reference must be positive")
    if covariate_value <= 0:
        raise ValueError("covariate_value must be positive")
    return param_typical * (covariate_value / covariate_reference) ** exponent


def exponential_covariate(
    param_typical: float,
    covariate_value: float,
    covariate_reference: float,
    theta: float,
) -> float:
    """Apply an exponential covariate effect.

    param_individual = param_typical * exp(theta * (covariate - reference))

    Commonly used for continuous covariates such as age or eGFR.

    Parameters
    ----------
    param_typical : float
        Population typical parameter value.
    covariate_value : float
        Individual covariate value.
    covariate_reference : float
        Reference covariate value (at which factor = 1).
    theta : float
        Covariate coefficient.

    Returns
    -------
    float
        Individual parameter value.
    """
    return param_typical * math.exp(theta * (covariate_value - covariate_reference))


def categorical_covariate(
    param_typical: float,
    category: str,
    category_effects: dict,
) -> float:
    """Apply a categorical covariate effect.

    param_individual = param_typical * category_effects.get(category, 1.0)

    Parameters
    ----------
    param_typical : float
        Population typical parameter value.
    category : str
        Individual's category (e.g., 'male', 'female').
    category_effects : dict
        Mapping of category name -> multiplicative factor.
        Unknown categories default to a factor of 1.0.

    Returns
    -------
    float
        Individual parameter value.
    """
    factor = category_effects.get(category, 1.0)
    return param_typical * factor


def compute_individual_pk(
    cl_typical: float,
    vd_typical: float,
    weight_kg: float,
    age_years: float,
    sex: str,
    egfr: float,
    clcr_ref: float = 100.0,
    weight_ref: float = 70.0,
    age_ref: float = 35.0,
    cl_weight_exp: float = 0.75,
    cl_egfr_exp: float = 1.0,
    cl_sex_female_factor: float = 0.9,
    vd_weight_exp: float = 1.0,
) -> dict:
    """Compute individual PK parameters using standard covariate effects.

    Covariate cascade applied to CL:
      1. Weight: CL_weight = cl_typical * (weight / 70)^0.75
      2. eGFR:   CL_egfr   = CL_weight  * (egfr / 100)^1.0
      3. Sex:    CL_sex    = CL_egfr    * (0.9 if female else 1.0)
      4. Age:    CL_age    = CL_sex     * exp(-0.005 * (age - 35))

    Vd covariate:
      Vd_weight = vd_typical * (weight / 70)^1.0

    Parameters
    ----------
    cl_typical : float
        Typical population clearance (L/h).
    vd_typical : float
        Typical population volume of distribution (L).
    weight_kg : float
        Individual body weight (kg).
    age_years : float
        Individual age (years).
    sex : str
        'male' or 'female'.
    egfr : float
        Estimated glomerular filtration rate (mL/min/1.73m²).
    clcr_ref : float
        Reference eGFR (default 100).
    weight_ref : float
        Reference weight (default 70 kg).
    age_ref : float
        Reference age (default 35 years).
    cl_weight_exp : float
        Weight exponent for CL (default 0.75).
    cl_egfr_exp : float
        eGFR exponent for CL (default 1.0).
    cl_sex_female_factor : float
        Multiplicative sex factor for females (default 0.9).
    vd_weight_exp : float
        Weight exponent for Vd (default 1.0).

    Returns
    -------
    dict with keys:
        cl_individual, vd_individual, ke, t_half_h, auc_dose, eta_cl
    """
    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive")
    if egfr < 0:
        raise ValueError("egfr must be non-negative")
    if cl_typical <= 0:
        raise ValueError("cl_typical must be positive")
    if vd_typical <= 0:
        raise ValueError("vd_typical must be positive")
    if weight_ref <= 0:
        raise ValueError("weight_ref must be positive")
    if clcr_ref <= 0:
        raise ValueError("clcr_ref must be positive")

    # Step 1: weight on CL
    cl_w = cl_typical * (weight_kg / weight_ref) ** cl_weight_exp

    # Step 2: eGFR on CL
    cl_egfr = cl_w * (egfr / clcr_ref) ** cl_egfr_exp

    # Step 3: sex on CL
    sex_factor = cl_sex_female_factor if sex.lower() == "female" else 1.0
    cl_sex = cl_egfr * sex_factor

    # Step 4: age on CL (mild effect, exp(-0.005*(age-ref)))
    cl_ind = cl_sex * math.exp(-0.005 * (age_years - age_ref))

    # Vd: weight only
    vd_ind = vd_typical * (weight_kg / weight_ref) ** vd_weight_exp

    # Derived PK metrics
    ke = cl_ind / vd_ind if vd_ind > 0 else 0.0
    t_half = math.log(2) / ke if ke > 0 else float("inf")
    auc_dose = 1.0 / cl_ind if cl_ind > 0 else float("inf")  # AUC per unit dose

    return {
        "cl_individual": cl_ind,
        "vd_individual": vd_ind,
        "ke": ke,
        "t_half_h": t_half,
        "auc_dose": auc_dose,
        "eta_cl": 0.0,  # deterministic; no IIV
    }


def covariate_sensitivity_table(
    cl_typical: float,
    vd_typical: float,
    covariates_to_vary: list[dict],
) -> list[dict]:
    """Build a covariate sensitivity table.

    For each covariate descriptor in `covariates_to_vary`, compute CL, Vd, and
    t½ fold-changes relative to the reference individual.

    Parameters
    ----------
    cl_typical : float
        Typical clearance (L/h).
    vd_typical : float
        Typical volume of distribution (L).
    covariates_to_vary : list[dict]
        Each dict must contain:
            - 'covariate': str  — one of 'weight', 'egfr', 'age', 'sex'
            - 'values': list    — covariate values to evaluate
            - 'reference': scalar — reference value for fold-change calculation

    Returns
    -------
    list[dict]
        One dict per (covariate, value) combination with keys:
        covariate, value, cl_fold_change, vd_fold_change, t_half_fold_change
    """
    # Reference individual: 70 kg, 35 years, male, eGFR=100
    ref_defaults = {"weight": 70.0, "age": 35.0, "sex": "male", "egfr": 100.0}

    ref_pk = compute_individual_pk(
        cl_typical=cl_typical,
        vd_typical=vd_typical,
        weight_kg=ref_defaults["weight"],
        age_years=ref_defaults["age"],
        sex=ref_defaults["sex"],
        egfr=ref_defaults["egfr"],
    )
    cl_ref = ref_pk["cl_individual"]
    vd_ref = ref_pk["vd_individual"]
    t_half_ref = ref_pk["t_half_h"]

    rows = []
    for cov_spec in covariates_to_vary:
        cov_name = cov_spec["covariate"]
        values = cov_spec["values"]

        for val in values:
            # Build kwargs for compute_individual_pk
            kwargs = {
                "cl_typical": cl_typical,
                "vd_typical": vd_typical,
                "weight_kg": ref_defaults["weight"],
                "age_years": ref_defaults["age"],
                "sex": ref_defaults["sex"],
                "egfr": ref_defaults["egfr"],
            }
            # Override the varying covariate
            if cov_name == "weight":
                kwargs["weight_kg"] = float(val)
            elif cov_name == "egfr":
                kwargs["egfr"] = float(val)
            elif cov_name == "age":
                kwargs["age_years"] = float(val)
            elif cov_name == "sex":
                kwargs["sex"] = str(val)
            else:
                # Unknown covariate — skip or pass through as-is
                pass

            pk = compute_individual_pk(**kwargs)
            cl_i = pk["cl_individual"]
            vd_i = pk["vd_individual"]
            t_half_i = pk["t_half_h"]

            rows.append(
                {
                    "covariate": cov_name,
                    "value": val,
                    "cl_fold_change": cl_i / cl_ref if cl_ref > 0 else float("nan"),
                    "vd_fold_change": vd_i / vd_ref if vd_ref > 0 else float("nan"),
                    "t_half_fold_change": (
                        t_half_i / t_half_ref
                        if t_half_ref > 0 and t_half_ref != float("inf")
                        else float("nan")
                    ),
                }
            )

    return rows
