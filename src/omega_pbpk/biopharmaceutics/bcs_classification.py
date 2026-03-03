"""BCS (Biopharmaceutics Classification System) classification and bioavailability prediction.

Implements FDA BCS guidance for classifying drugs based on solubility and
permeability, predicting in vitro dissolution profiles via Noyes-Whitney
kinetics, and estimating oral fraction absorbed.

BCS Class Summary:
  Class I   — High solubility, High permeability  (biowaiver eligible)
  Class II  — Low solubility,  High permeability  (dissolution-limited)
  Class III — High solubility, Low permeability   (permeability-limited)
  Class IV  — Low solubility,  Low permeability   (poor absorption)

Dose number: Do = (Dose_mg / GI_volume_mL) / Cs_mg_per_mL
  Do < 1 → high solubility; Do >= 1 → low solubility

Permeability threshold: Peff >= 2.0e-4 cm/s → high permeability
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HIGH_PERM_THRESHOLD = 2.0e-4  # cm/s  (FDA BCS guidance)
_DO_THRESHOLD = 1.0  # dimensionless dose number cutoff
_R_INTESTINE_CM = 1.75  # cm — effective intestinal radius (cylindrical)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BCSInput:
    """Input parameters for BCS classification and dissolution/absorption."""

    drug_name: str
    dose_mg: float
    solubility_mg_per_mL: float
    permeability_cm_per_s: float
    particle_radius_um: float = 25.0  # microns
    density_g_per_cm3: float = 1.2  # g/cm³
    diffusion_coeff_cm2_per_s: float = 5e-6  # cm²/s
    gi_volume_mL: float = 250.0  # mL (stomach + proximal small intestine)
    gi_transit_h: float = 3.5  # h (small intestinal transit)


@dataclass(frozen=True)
class DissolutionProfile:
    """Time-course dissolution profile from Noyes-Whitney model."""

    time_h: list[float]
    fraction_dissolved: list[float]
    t50_h: float  # time to 50% dissolved (h); inf if not reached
    t85_h: float  # time to 85% dissolved (h); inf if not reached


@dataclass(frozen=True)
class AbsorptionEstimate:
    """Oral fraction absorbed estimate from dissolution + permeability."""

    fraction_absorbed: float  # overall Fa = Fa_perm * Fa_diss
    fa_dissolution_limited: float  # fraction dissolved at transit time
    fa_permeability_limited: float  # Fa based on permeability alone
    rate_limiting_step: str  # "dissolution" or "permeability"


@dataclass(frozen=True)
class BCSReport:
    """Complete BCS classification and bioavailability report."""

    input: BCSInput
    bcs_class: str  # "I", "II", "III", "IV"
    dose_number: float
    dissolution: DissolutionProfile
    absorption: AbsorptionEstimate
    regulatory_note: str
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_dose_number(
    dose_mg: float,
    solubility_mg_per_mL: float,
    volume_mL: float = 250.0,
) -> float:
    """Compute BCS dose number Do = (dose_mg / volume_mL) / solubility_mg_per_mL.

    Args:
        dose_mg: Administered dose in mg.
        solubility_mg_per_mL: Aqueous solubility in mg/mL.
        volume_mL: GI volume for dissolution (default 250 mL).

    Returns:
        Dimensionless dose number Do.
    """
    cs = max(solubility_mg_per_mL, 1e-12)
    return (dose_mg / volume_mL) / cs


def classify_bcs(
    solubility_mg_per_mL: float,
    dose_mg: float,
    permeability_cm_per_s: float,
    volume_mL: float = 250.0,
) -> str:
    """Classify a drug into BCS class I–IV.

    Args:
        solubility_mg_per_mL: Aqueous solubility in mg/mL.
        dose_mg: Highest marketed dose in mg.
        permeability_cm_per_s: Effective intestinal permeability (cm/s).
        volume_mL: GI fluid volume used for dose number (default 250 mL).

    Returns:
        BCS class string: "I", "II", "III", or "IV".
    """
    do = compute_dose_number(dose_mg, solubility_mg_per_mL, volume_mL)
    high_sol = do < _DO_THRESHOLD
    high_perm = permeability_cm_per_s >= _HIGH_PERM_THRESHOLD

    if high_sol and high_perm:
        return "I"
    if not high_sol and high_perm:
        return "II"
    if high_sol and not high_perm:
        return "III"
    return "IV"


def estimate_dissolution(
    solubility_mg_per_mL: float,
    particle_radius_um: float,
    density_g_per_cm3: float,
    diffusion_coeff_cm2_per_s: float,
    dose_mg: float,
    dt_h: float = 0.01,
    t_end_h: float = 4.0,
) -> DissolutionProfile:
    """Estimate dissolution profile using simplified Noyes-Whitney model.

    Model: dm/dt = k_diss * Cs * (1 - m/dose)
    where k_diss = 3*D / (density * r²)  (units: h⁻¹ when consistent)
    m = dissolved mass (mg), Cs = solubility (mg/mL scaled to mg/cm³).

    Args:
        solubility_mg_per_mL: Aqueous solubility in mg/mL (= mg/cm³).
        particle_radius_um: Particle radius in microns.
        density_g_per_cm3: Particle density in g/cm³.
        diffusion_coeff_cm2_per_s: Drug diffusion coefficient in cm²/s.
        dose_mg: Total dose in mg.
        dt_h: Integration time step in hours.
        t_end_h: End time for simulation in hours.

    Returns:
        DissolutionProfile with time array, fraction dissolved, t50, t85.
    """
    # Convert units to CGS with time in seconds for k_diss, then convert to h⁻¹
    r_cm = particle_radius_um * 1e-4  # µm → cm
    D_cm2_s = diffusion_coeff_cm2_per_s
    rho = density_g_per_cm3  # g/cm³
    Cs = solubility_mg_per_mL  # mg/cm³ (since 1 mL = 1 cm³)

    # k_diss in s⁻¹ → convert to h⁻¹
    k_diss_per_s = 3.0 * D_cm2_s / (rho * r_cm**2)
    k_diss_per_h = k_diss_per_s * 3600.0

    n_steps = max(int(t_end_h / dt_h), 1)
    times: list[float] = []
    fractions: list[float] = []

    m = 0.0  # dissolved mass (mg)
    t50_h = math.inf
    t85_h = math.inf

    for i in range(n_steps + 1):
        t = i * dt_h
        frac = min(m / max(dose_mg, 1e-12), 1.0)
        times.append(round(t, 6))
        fractions.append(round(frac, 8))

        if frac >= 0.5 and math.isinf(t50_h):
            t50_h = t
        if frac >= 0.85 and math.isinf(t85_h):
            t85_h = t

        # Euler step: dm/dt = k_diss * Cs * (dose - m) / dose * dose
        #           = k_diss * Cs * (dose - m)  [mg/h if k in h⁻¹, Cs in mg/cm³]
        # But we need volume of solvent — use sink condition approximation:
        # dm/dt = k_diss * (dose - m) * Cs  (dissolution driving force)
        dm = k_diss_per_h * Cs * (dose_mg - m) * dt_h
        m = min(m + dm, dose_mg)

    return DissolutionProfile(
        time_h=times,
        fraction_dissolved=fractions,
        t50_h=round(t50_h, 4) if not math.isinf(t50_h) else t50_h,
        t85_h=round(t85_h, 4) if not math.isinf(t85_h) else t85_h,
    )


def predict_fraction_absorbed(
    permeability_cm_per_s: float,
    dissolution_profile: DissolutionProfile,
    gi_transit_h: float = 3.5,
) -> AbsorptionEstimate:
    """Estimate oral fraction absorbed from permeability and dissolution.

    Permeability-limited Fa (cylindrical intestine model):
      Fa_perm = 1 - exp(-2 * Peff * transit_s / R_intestine)

    Dissolution-limited Fa:
      Fa_diss = fraction dissolved at GI transit time

    Combined (rate-limiting concept):
      Fa = Fa_perm * Fa_diss

    Args:
        permeability_cm_per_s: Effective intestinal permeability (cm/s).
        dissolution_profile: DissolutionProfile from estimate_dissolution.
        gi_transit_h: Small intestinal transit time in hours.

    Returns:
        AbsorptionEstimate.
    """
    transit_s = gi_transit_h * 3600.0
    fa_perm = 1.0 - math.exp(-2.0 * permeability_cm_per_s * transit_s / _R_INTESTINE_CM)
    fa_perm = min(max(fa_perm, 0.0), 1.0)

    # Interpolate dissolution fraction at transit time
    times = dissolution_profile.time_h
    fracs = dissolution_profile.fraction_dissolved
    fa_diss = 0.0
    if times and gi_transit_h >= times[-1]:
        fa_diss = fracs[-1]
    elif times:
        for i in range(len(times) - 1):
            if times[i] <= gi_transit_h <= times[i + 1]:
                span = times[i + 1] - times[i]
                if span > 0:
                    alpha = (gi_transit_h - times[i]) / span
                    fa_diss = fracs[i] + alpha * (fracs[i + 1] - fracs[i])
                else:
                    fa_diss = fracs[i]
                break

    fa_diss = min(max(fa_diss, 0.0), 1.0)
    fa = fa_perm * fa_diss
    fa = min(max(fa, 0.0), 1.0)

    rate_limiting = "dissolution" if fa_diss < fa_perm else "permeability"

    return AbsorptionEstimate(
        fraction_absorbed=round(fa, 6),
        fa_dissolution_limited=round(fa_diss, 6),
        fa_permeability_limited=round(fa_perm, 6),
        rate_limiting_step=rate_limiting,
    )


_REGULATORY_NOTES: dict[str, str] = {
    "I": (
        "BCS Class I: high solubility, high permeability — eligible for "
        "biowaivers for immediate-release solid oral dosage forms (FDA/EMA guidance)"
    ),
    "II": (
        "BCS Class II: low solubility, high permeability — absorption is "
        "dissolution-limited; formulation strategies (e.g., nanoparticles, "
        "amorphous solid dispersions) recommended to improve bioavailability"
    ),
    "III": (
        "BCS Class III: high solubility, low permeability — absorption is "
        "permeability-limited; biowaiver possible under strict conditions "
        "(EMA guideline); permeation enhancers may be beneficial"
    ),
    "IV": (
        "BCS Class IV: low solubility, low permeability — poor oral "
        "bioavailability expected; significant formulation and/or prodrug "
        "strategies required"
    ),
}


def run_bcs_assessment(inp: BCSInput) -> BCSReport:
    """Orchestrate BCS classification and bioavailability prediction.

    Args:
        inp: BCSInput with drug physicochemical and dose parameters.

    Returns:
        BCSReport with classification, dissolution profile, and absorption estimate.

    Raises:
        ValueError: If key inputs are non-positive.
    """
    warnings: list[str] = []

    if inp.solubility_mg_per_mL <= 0:
        raise ValueError("solubility_mg_per_mL must be > 0")
    if inp.dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if inp.permeability_cm_per_s < 0:
        raise ValueError("permeability_cm_per_s must be >= 0")

    if inp.particle_radius_um <= 0:
        warnings.append("particle_radius_um <= 0; dissolution model may be unreliable")

    do = compute_dose_number(inp.dose_mg, inp.solubility_mg_per_mL, inp.gi_volume_mL)
    bcs_class = classify_bcs(
        inp.solubility_mg_per_mL, inp.dose_mg, inp.permeability_cm_per_s, inp.gi_volume_mL
    )

    dissolution = estimate_dissolution(
        solubility_mg_per_mL=inp.solubility_mg_per_mL,
        particle_radius_um=inp.particle_radius_um,
        density_g_per_cm3=inp.density_g_per_cm3,
        diffusion_coeff_cm2_per_s=inp.diffusion_coeff_cm2_per_s,
        dose_mg=inp.dose_mg,
    )

    absorption = predict_fraction_absorbed(
        permeability_cm_per_s=inp.permeability_cm_per_s,
        dissolution_profile=dissolution,
        gi_transit_h=inp.gi_transit_h,
    )

    regulatory_note = _REGULATORY_NOTES[bcs_class]

    return BCSReport(
        input=inp,
        bcs_class=bcs_class,
        dose_number=round(do, 6),
        dissolution=dissolution,
        absorption=absorption,
        regulatory_note=regulatory_note,
        warnings=warnings,
    )


__all__ = [
    "BCSInput",
    "DissolutionProfile",
    "AbsorptionEstimate",
    "BCSReport",
    "compute_dose_number",
    "classify_bcs",
    "estimate_dissolution",
    "predict_fraction_absorbed",
    "run_bcs_assessment",
]
