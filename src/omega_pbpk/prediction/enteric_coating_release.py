"""Phase 1075: Enteric Coating pH-Triggered Release Model."""

from __future__ import annotations

from dataclasses import dataclass

# Known EC coatings: name -> pH threshold
KNOWN_COATINGS: dict[str, float] = {
    "Eudragit_L100": 6.0,
    "Eudragit_S100": 7.0,
    "CAP": 5.8,
    "HPMCP_HP55": 5.5,
    "shellac": 7.3,
}

# GI compartment: (pH, transit_time_h)
GI_COMPARTMENTS: list[tuple[str, float, float]] = [
    ("Stomach", 1.5, 1.5),
    ("Duodenum", 5.5, 0.5),
    ("Jejunum", 6.5, 2.0),
    ("Ileum", 7.2, 3.0),
    ("Colon", 6.8, 24.0),
]

# First-order release rate when coating is dissolving (1/h)
K_RELEASE = 1.0


@dataclass
class EnterCoatReleaseResult:
    """Result of enteric coating release simulation."""

    drug_name: str
    coating: str
    dose_mg: float
    times_h: list
    cumulative_release_mg: list
    release_fraction_list: list
    compartment_release: dict
    primary_release_site: str
    t_release_onset_h: float
    t_50pct_release_h: float
    f_total_released: float
    notes: str


def simulate_enteric_coating_release(
    drug_name: str,
    dose_mg: float,
    coating: str,
    dt: float = 0.1,
) -> EnterCoatReleaseResult:
    """Simulate drug release from an enteric-coated dosage form.

    Args:
        drug_name: Name of the drug.
        dose_mg: Dose in mg. Must be > 0.
        coating: EC polymer name. Must be in KNOWN_COATINGS.
        dt: Time step in hours (default 0.1 h).

    Returns:
        EnterCoatReleaseResult with full release profile.

    Raises:
        ValueError: If dose_mg <= 0 or coating is unknown.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if coating not in KNOWN_COATINGS:
        raise ValueError(
            f"Unknown coating '{coating}'. Known coatings: {list(KNOWN_COATINGS.keys())}"
        )

    ph_threshold = KNOWN_COATINGS[coating]

    times_h: list[float] = []
    cumulative_release_mg: list[float] = []
    release_fraction_list: list[float] = []
    compartment_release: dict[str, float] = {c[0]: 0.0 for c in GI_COMPARTMENTS}

    remaining_mg = dose_mg
    cumulative_mg = 0.0
    t = 0.0

    t_release_onset_h = float("inf")
    t_50pct_release_h = float("inf")

    onset_threshold = 0.10 * dose_mg
    half_threshold = 0.50 * dose_mg

    for comp_name, comp_ph, comp_transit in GI_COMPARTMENTS:
        t_end = t + comp_transit
        n_steps = max(1, round(comp_transit / dt))
        actual_dt = comp_transit / n_steps

        for _ in range(n_steps):
            times_h.append(round(t, 6))
            cumulative_release_mg.append(cumulative_mg)
            release_fraction_list.append(cumulative_mg / dose_mg)

            if comp_ph >= ph_threshold and remaining_mg > 0.0:
                released = K_RELEASE * remaining_mg * actual_dt
                released = min(released, remaining_mg)
                remaining_mg -= released
                cumulative_mg += released
                compartment_release[comp_name] = compartment_release[comp_name] + released

                # Check onset (10%)
                if cumulative_mg >= onset_threshold and t_release_onset_h == float("inf"):
                    t_release_onset_h = t + actual_dt

                # Check 50%
                if cumulative_mg >= half_threshold and t_50pct_release_h == float("inf"):
                    t_50pct_release_h = t + actual_dt

            t += actual_dt

        t = t_end

    # Append final time point
    times_h.append(round(t, 6))
    cumulative_release_mg.append(cumulative_mg)
    release_fraction_list.append(cumulative_mg / dose_mg)

    # Determine primary release site
    primary_release_site = max(compartment_release, key=lambda k: compartment_release[k])
    if compartment_release[primary_release_site] == 0.0:
        primary_release_site = "None"

    f_total_released = cumulative_mg / dose_mg

    notes = (
        f"EC coating {coating} (pH threshold {ph_threshold}); "
        f"total released {f_total_released * 100:.1f}% of {dose_mg} mg dose."
    )

    return EnterCoatReleaseResult(
        drug_name=drug_name,
        coating=coating,
        dose_mg=dose_mg,
        times_h=times_h,
        cumulative_release_mg=cumulative_release_mg,
        release_fraction_list=release_fraction_list,
        compartment_release=compartment_release,
        primary_release_site=primary_release_site,
        t_release_onset_h=t_release_onset_h,
        t_50pct_release_h=t_50pct_release_h,
        f_total_released=f_total_released,
        notes=notes,
    )


def compare_coatings(
    drug_name: str,
    dose_mg: float,
    coatings: list[str] | None = None,
) -> list[EnterCoatReleaseResult]:
    """Compare drug release across multiple EC coatings.

    Args:
        drug_name: Drug name.
        dose_mg: Dose in mg.
        coatings: List of coating names. If None, tests all 5 known coatings.

    Returns:
        List of EnterCoatReleaseResult, sorted by f_total_released descending.
    """
    if coatings is None:
        coatings = list(KNOWN_COATINGS.keys())

    results = [simulate_enteric_coating_release(drug_name, dose_mg, c) for c in coatings]
    results.sort(key=lambda r: r.f_total_released, reverse=True)
    return results
