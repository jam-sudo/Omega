"""
Phase 580 — Lean body mass and body composition calculations for PK dose adjustment.
"""


def lean_body_mass_james(height_cm: float, weight_kg: float, sex: str) -> float:
    """James formula for lean body mass (LBM) in kg.

    Male:   LBM = 1.1 * weight - 128 * (weight/height)^2
    Female: LBM = 1.07 * weight - 148 * (weight/height)^2

    Returns LBM clamped to a minimum of 10 kg.
    """
    _validate_sex(sex)
    _validate_positive(height_cm, "height_cm")
    _validate_positive(weight_kg, "weight_kg")

    ratio = weight_kg / height_cm
    if sex == "male":
        lbm = 1.1 * weight_kg - 128.0 * ratio**2
    else:
        lbm = 1.07 * weight_kg - 148.0 * ratio**2

    return max(lbm, 10.0)


def lean_body_mass_boer(height_cm: float, weight_kg: float, sex: str) -> float:
    """Boer formula for lean body mass (LBM) in kg.

    Male:   LBM = 0.407 * weight + 0.267 * height - 19.2
    Female: LBM = 0.252 * weight + 0.473 * height - 48.3

    Returns LBM clamped to a minimum of 10 kg.
    """
    _validate_sex(sex)
    _validate_positive(height_cm, "height_cm")
    _validate_positive(weight_kg, "weight_kg")

    if sex == "male":
        lbm = 0.407 * weight_kg + 0.267 * height_cm - 19.2
    else:
        lbm = 0.252 * weight_kg + 0.473 * height_cm - 48.3

    return max(lbm, 10.0)


def bmi(weight_kg: float, height_cm: float) -> float:
    """Body mass index: weight_kg / (height_m)^2."""
    _validate_positive(weight_kg, "weight_kg")
    _validate_positive(height_cm, "height_cm")
    height_m = height_cm / 100.0
    return weight_kg / (height_m**2)


def ideal_body_weight_devine(height_cm: float, sex: str) -> float:
    """Devine formula for ideal body weight (IBW) in kg.

    Male:   IBW = 50   + 2.3 * (height_in - 60)
    Female: IBW = 45.5 + 2.3 * (height_in - 60)

    height_in = height_cm / 2.54
    Returns IBW clamped to a minimum of 10 kg.
    """
    _validate_sex(sex)
    _validate_positive(height_cm, "height_cm")

    height_in = height_cm / 2.54
    if sex == "male":
        ibw = 50.0 + 2.3 * (height_in - 60.0)
    else:
        ibw = 45.5 + 2.3 * (height_in - 60.0)

    return max(ibw, 10.0)


def adjusted_body_weight(
    ibw_kg: float, actual_weight_kg: float, correction_factor: float = 0.4
) -> float:
    """Adjusted body weight for obese patients.

    ABW = IBW + correction_factor * (actual - IBW)

    If actual <= IBW, returns actual_weight (not obese).
    """
    if actual_weight_kg <= ibw_kg:
        return actual_weight_kg
    return ibw_kg + correction_factor * (actual_weight_kg - ibw_kg)


def dose_body_weight_method(
    dose_mg_per_kg: float,
    weight_kg: float,
    ibw_kg: float,
    lbm_kg: float,
    method: str = "actual",
) -> dict:
    """Compute dose based on chosen body weight method.

    method: 'actual', 'ideal', 'lean', 'adjusted'
    Returns: method, weight_used_kg, dose_mg
    """
    valid_methods = {"actual", "ideal", "lean", "adjusted"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    if method == "actual":
        w = weight_kg
    elif method == "ideal":
        w = ibw_kg
    elif method == "lean":
        w = lbm_kg
    elif method == "adjusted":
        w = adjusted_body_weight(ibw_kg, weight_kg)

    return {
        "method": method,
        "weight_used_kg": w,
        "dose_mg": dose_mg_per_kg * w,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_sex(sex: str) -> None:
    if sex not in {"male", "female"}:
        raise ValueError(f"sex must be 'male' or 'female', got '{sex}'")


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
