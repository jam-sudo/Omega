"""
Phase 247: Micro-dose PK predictor for Phase 0 studies (<100 ug doses).

Predicts PK parameters at microdose levels and extrapolates to conventional doses
assuming linear PK.
"""

from dataclasses import dataclass

VALID_DRUG_CLASSES = {"standard", "TMDD", "NTI", "saturable"}

_NONLINEAR_CLASSES = {"TMDD", "NTI", "saturable"}


@dataclass(frozen=True)
class MicrodosePKResult:
    """PK prediction result for a microdose study."""

    drug_name: str
    microdose_ug: float
    conventional_dose_mg: float
    cl_L_per_h: float
    vd_L: float
    t_half_h: float
    cmax_microdose_ng_L: float
    auc_microdose_ng_h_per_L: float
    cmax_scaling_factor: float
    auc_conventional_predicted_mg_h_per_L: float
    linearity_expected: bool
    dose_ratio: float
    notes: str


def predict_microdose_pk(
    drug_name: str,
    microdose_ug: float,
    conventional_dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    drug_class: str = "standard",
) -> MicrodosePKResult:
    """
    Predict PK parameters at microdose level and extrapolate to conventional dose.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    microdose_ug : float
        Microdose in micrograms (must be <= 100 ug per FDA Phase 0 definition).
    conventional_dose_mg : float
        Conventional therapeutic dose in milligrams.
    cl_L_per_h : float
        Clearance in L/h (must be > 0).
    vd_L : float
        Volume of distribution in L (must be > 0).
    drug_class : str
        One of "standard", "TMDD", "NTI", "saturable".

    Returns
    -------
    MicrodosePKResult
    """
    if microdose_ug <= 0:
        raise ValueError(f"microdose_ug must be > 0, got {microdose_ug}")
    if microdose_ug > 100.0:
        raise ValueError(
            f"microdose_ug must be <= 100.0 ug (FDA Phase 0 definition), got {microdose_ug}"
        )
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if drug_class not in VALID_DRUG_CLASSES:
        raise ValueError(f"drug_class must be one of {VALID_DRUG_CLASSES}, got '{drug_class}'")

    microdose_mg = microdose_ug / 1000.0
    if conventional_dose_mg <= microdose_mg:
        raise ValueError(
            f"conventional_dose_mg ({conventional_dose_mg}) must be > microdose_mg ({microdose_mg})"
        )

    # Elimination rate constant and half-life
    ke = cl_L_per_h / vd_L
    t_half_h = 0.693147 / ke

    # IV bolus microdose PK (linear assumption)
    # Cmax (ng/L): dose(mg) / Vd(L) * 1e6 to convert mg/L -> ng/L
    cmax_microdose_ng_L = microdose_mg / vd_L * 1.0e6

    # AUC (ng*h/L): dose(mg) / CL(L/h) * 1e6
    auc_microdose_ng_h_per_L = microdose_mg / cl_L_per_h * 1.0e6

    # Dose ratio for linear extrapolation
    dose_ratio = conventional_dose_mg / microdose_mg

    # Scaling factor (linear assumption)
    cmax_scaling_factor = dose_ratio

    # Conventional dose AUC prediction (mg*h/L)
    auc_conventional_predicted_mg_h_per_L = conventional_dose_mg / cl_L_per_h

    # Linearity expected only for standard class
    linearity_expected = drug_class not in _NONLINEAR_CLASSES

    if linearity_expected:
        notes = (
            f"{drug_class} drug: linear PK assumed; "
            f"dose_ratio={dose_ratio:.1f}x extrapolation expected to hold."
        )
    else:
        notes = (
            f"{drug_class} drug: nonlinear PK likely; "
            f"microdose may not predict conventional dose PK. "
            f"dose_ratio={dose_ratio:.1f}x scaling may be inaccurate."
        )

    return MicrodosePKResult(
        drug_name=drug_name,
        microdose_ug=microdose_ug,
        conventional_dose_mg=conventional_dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_half_h=t_half_h,
        cmax_microdose_ng_L=cmax_microdose_ng_L,
        auc_microdose_ng_h_per_L=auc_microdose_ng_h_per_L,
        cmax_scaling_factor=cmax_scaling_factor,
        auc_conventional_predicted_mg_h_per_L=auc_conventional_predicted_mg_h_per_L,
        linearity_expected=linearity_expected,
        dose_ratio=dose_ratio,
        notes=notes,
    )


def compare_microdose_levels(
    drug_name: str,
    conventional_dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    microdose_levels_ug: list[float] | None = None,
) -> list[MicrodosePKResult]:
    """
    Compare PK predictions across multiple microdose levels.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    conventional_dose_mg : float
        Conventional therapeutic dose in milligrams.
    cl_L_per_h : float
        Clearance in L/h.
    vd_L : float
        Volume of distribution in L.
    microdose_levels_ug : list of float, optional
        Microdose levels in ug to compare. Defaults to [1.0, 10.0, 50.0, 100.0].

    Returns
    -------
    list of MicrodosePKResult, sorted by microdose_ug ascending.
    """
    if microdose_levels_ug is None:
        microdose_levels_ug = [1.0, 10.0, 50.0, 100.0]

    results = []
    for level_ug in microdose_levels_ug:
        result = predict_microdose_pk(
            drug_name=drug_name,
            microdose_ug=level_ug,
            conventional_dose_mg=conventional_dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            drug_class="standard",
        )
        results.append(result)

    results.sort(key=lambda r: r.microdose_ug)
    return results
