"""Phase 1053 — Drug Tumor Penetration Index.

Predict drug penetration through solid tumor tissue based on
physicochemical and microenvironment properties.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TumorPenetrationResult:
    drug_name: str
    mw_Da: float
    logp: float
    tumor_type: str
    penetration_index: float
    diffusion_coefficient_cm2_h: float
    effective_diffusion_depth_um: float
    interstitial_fluid_pressure_factor: float
    binding_factor: float
    vascular_access_score: float
    penetration_class: str
    dose_required_fold_increase: float
    notes: str


_TUMOR_PARAMS: dict[str, tuple[float, float, float]] = {
    "solid": (0.6, 0.7, 0.6),
    "hematologic": (0.9, 0.95, 0.9),
    "pancreatic": (0.3, 0.4, 0.3),
    "brain": (0.5, 0.6, 0.5),
}

_VASCULARITY_MULT: dict[str, float] = {
    "high": 1.3,
    "moderate": 1.0,
    "low": 0.5,
}

_VALID_IONIZATION = {"neutral", "acid", "base"}


def predict_tumor_penetration(
    drug_name: str,
    mw_Da: float,
    logp: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    tumor_type: str = "solid",
    tumor_vascularity: str = "moderate",
    has_efflux_pump: bool = False,
    protein_binding_pct: float = 90.0,
) -> TumorPenetrationResult:
    """Predict drug penetration through solid tumor tissue."""
    # --- Validation ---
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if not (0.0 <= protein_binding_pct <= 100.0):
        raise ValueError(f"protein_binding_pct must be in [0, 100], got {protein_binding_pct}")
    if tumor_type not in _TUMOR_PARAMS:
        raise ValueError(f"tumor_type must be one of {set(_TUMOR_PARAMS)}, got {tumor_type!r}")
    if tumor_vascularity not in _VASCULARITY_MULT:
        raise ValueError(
            f"tumor_vascularity must be one of {set(_VASCULARITY_MULT)}, got {tumor_vascularity!r}"
        )
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got {ionization_type!r}"
        )

    # --- Diffusion coefficient ---
    d_water = 10 ** (-5 - 0.001 * mw_Da)  # cm²/s
    d_water_cm2_h = d_water * 3600
    d_water_cm2_h = max(1e-6, min(1e-2, d_water_cm2_h))

    # --- Tumor type factors ---
    ifp_factor, matrix_factor, v_score = _TUMOR_PARAMS[tumor_type]

    # --- Vascularity ---
    v_mult = _VASCULARITY_MULT[tumor_vascularity]
    vascular_access_score = max(0.1, min(1.0, v_score * v_mult))

    # --- Effective diffusion ---
    d_eff = d_water_cm2_h * ifp_factor * matrix_factor

    # --- Ionization penalty at tumor pH 6.8 ---
    tumor_ph = 6.8
    if ionization_type == "acid":
        ionized_fraction = 1.0 / (1.0 + 10 ** (pka - tumor_ph))
    elif ionization_type == "base":
        ionized_fraction = 1.0 / (1.0 + 10 ** (tumor_ph - pka))
    else:
        ionized_fraction = 0.0
    ioniz_factor = max(0.1, 1.0 - 0.5 * ionized_fraction)

    # --- Protein binding ---
    fu_tumor = 1.0 - protein_binding_pct / 100.0
    binding_factor = fu_tumor  # [0, 1]

    # --- Efflux pump ---
    efflux_factor = 0.3 if has_efflux_pump else 1.0

    # --- Penetration index ---
    pi_raw = vascular_access_score * ioniz_factor * binding_factor * efflux_factor
    penetration_index = min(0.95, pi_raw * 2)
    penetration_index = max(0.01, penetration_index)

    # --- Effective diffusion depth ---
    depth_cm = (d_eff * 1.0) ** 0.5
    depth_um = depth_cm * 1e4

    # --- Dose required fold increase ---
    dose_required_fold_increase = max(1.0, 1.0 / penetration_index)

    # --- Penetration class ---
    if penetration_index > 0.7:
        penetration_class = "excellent"
    elif penetration_index >= 0.5:
        penetration_class = "good"
    elif penetration_index >= 0.3:
        penetration_class = "moderate"
    else:
        penetration_class = "poor"

    # --- Notes ---
    notes_parts = [
        f"Tumor type: {tumor_type} (IFP factor={ifp_factor:.2f}, matrix={matrix_factor:.2f}).",
        f"Vascularity: {tumor_vascularity} (score={vascular_access_score:.3f}).",
        f"Ionization at pH {tumor_ph}: ionized fraction={ionized_fraction:.3f}, ioniz_factor={ioniz_factor:.3f}.",
        f"fu_tumor={fu_tumor:.3f}, efflux={'yes' if has_efflux_pump else 'no'}.",
        f"PI={penetration_index:.3f} ({penetration_class}); dose fold increase={dose_required_fold_increase:.1f}x.",
    ]
    notes = " ".join(notes_parts)

    return TumorPenetrationResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logp=logp,
        tumor_type=tumor_type,
        penetration_index=penetration_index,
        diffusion_coefficient_cm2_h=d_water_cm2_h,
        effective_diffusion_depth_um=depth_um,
        interstitial_fluid_pressure_factor=ifp_factor,
        binding_factor=binding_factor,
        vascular_access_score=vascular_access_score,
        penetration_class=penetration_class,
        dose_required_fold_increase=dose_required_fold_increase,
        notes=notes,
    )
