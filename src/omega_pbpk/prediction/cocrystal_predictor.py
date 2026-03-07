"""Phase 810 — Cocrystal and amorphous dispersion predictor for solubility enhancement."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CocrystalResult:
    drug_name: str
    coformer_name: str
    strategy: str  # "cocrystal", "salt", "asd", "cyclodextrin"
    predicted_solubility_fold_increase: float
    predicted_dissolution_rate_fold: float
    spring_factor: float  # supersaturation potential
    parachute_duration_h: float  # time before precipitation (higher = better)
    coformer_compatibility: str  # "compatible", "incompatible", "uncertain"
    predicted_stability: str  # "stable", "unstable", "conditional"
    bioavailability_prediction: str  # "significantly_improved", "moderately_improved", "marginal"
    notes: str


_VALID_STRATEGIES = {"cocrystal", "salt", "asd", "cyclodextrin"}


def _bioavailability_label(fold: float) -> str:
    """Map fold increase to bioavailability prediction label."""
    if fold >= 10.0:
        return "significantly_improved"
    elif fold >= 3.0:
        return "moderately_improved"
    else:
        return "marginal"


def predict_cocrystal_enhancement(
    drug_name: str,
    logp: float,
    mw_da: float,
    pka: float | None = None,
    ionization_type: str = "neutral",
    intrinsic_solubility_mg_L: float = 10.0,
    strategy: str = "cocrystal",
    coformer_name: str = "generic",
    coformer_pka: float | None = None,
    coformer_logp: float = 0.0,
) -> CocrystalResult:
    """Predict solubility enhancement from cocrystal/salt/ASD/cyclodextrin strategy."""
    if intrinsic_solubility_mg_L <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_L must be positive, got {intrinsic_solubility_mg_L}"
        )
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {_VALID_STRATEGIES}, got {strategy!r}")

    notes_parts = []

    if strategy == "cocrystal":
        # Fold depends on pKa difference with coformer
        if pka is not None and coformer_pka is not None:
            pka_diff = abs(pka - coformer_pka)
            if pka_diff > 2.0:
                fold = 8.0  # salt-like interaction
                notes_parts.append(f"Salt-like cocrystal (DeltapKa={pka_diff:.1f})")
            else:
                fold = 3.0  # neutral cocrystal
                notes_parts.append(f"Neutral cocrystal (DeltapKa={pka_diff:.1f})")
        else:
            fold = 3.0  # default neutral cocrystal
            notes_parts.append("No pKa data; neutral cocrystal assumed")

        dissolution_fold = fold * 0.8  # slightly lower than solubility fold
        spring_factor = fold * 0.9
        parachute_duration_h = 6.0
        stability = "stable"
        compatibility = "compatible"

    elif strategy == "salt":
        # Requires pKa; best for ionizable drugs
        if pka is not None and coformer_pka is not None:
            # Basic drug: acid coformer pKa < drug pKa - 2
            if coformer_pka < pka - 2.0:
                fold = 50.0
                notes_parts.append("Salt formation confirmed by pKa rule")
            else:
                fold = 5.0
                notes_parts.append("Weak salt formation; moderate enhancement")
        elif ionization_type in ("acid", "base") and pka is not None:
            fold = 20.0
            notes_parts.append("Ionizable drug; assumed favourable salt")
        else:
            fold = 5.0
            notes_parts.append("Salt strategy for neutral/unknown drug; limited benefit")

        dissolution_fold = fold * 1.1
        spring_factor = fold * 0.7
        parachute_duration_h = 8.0
        stability = "stable"
        compatibility = "compatible" if fold >= 20.0 else "uncertain"

    elif strategy == "asd":
        # Amorphous solid dispersion: highest fold, shorter parachute
        fold = 30.0
        dissolution_fold = 25.0
        spring_factor = 30.0
        parachute_duration_h = 2.5  # moisture-sensitive → shorter
        stability = "conditional"  # moisture-sensitive
        compatibility = "compatible"
        notes_parts.append("ASD: high supersaturation, moisture-sensitive")

    else:  # cyclodextrin
        # logP 2-4 → 10x; else 5x
        if 2.0 <= logp <= 4.0:
            fold = 10.0
            notes_parts.append(f"Optimal logP range ({logp:.1f}) for cyclodextrin complexation")
        else:
            fold = 5.0
            notes_parts.append(f"Sub-optimal logP ({logp:.1f}) for cyclodextrin; moderate gain")

        dissolution_fold = fold * 0.9
        spring_factor = fold * 0.6
        parachute_duration_h = 5.0
        stability = "stable"
        compatibility = "compatible"

    # Spring factor cannot exceed fold (sanity)
    spring_factor = min(spring_factor, fold * 1.5)

    bioav_label = _bioavailability_label(fold)

    notes = "; ".join(notes_parts) if notes_parts else "Standard enhancement predicted"

    return CocrystalResult(
        drug_name=drug_name,
        coformer_name=coformer_name,
        strategy=strategy,
        predicted_solubility_fold_increase=fold,
        predicted_dissolution_rate_fold=dissolution_fold,
        spring_factor=spring_factor,
        parachute_duration_h=parachute_duration_h,
        coformer_compatibility=compatibility,
        predicted_stability=stability,
        bioavailability_prediction=bioav_label,
        notes=notes,
    )
