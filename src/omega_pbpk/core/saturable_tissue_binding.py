"""Saturable (nonlinear) tissue binding: Langmuir isotherm, concentration-dependent distribution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TissueBindingResult:
    drug_name: str
    tissue: str
    bmax_nmol_per_g: float  # maximum binding capacity
    kd_nM: float  # equilibrium dissociation constant
    concentrations_nM: tuple  # free concentrations tested
    bound_fractions: tuple  # fraction bound at each concentration
    fu_tissue_at_therapeutic: float  # fu in tissue at therapeutic conc
    occupancy_at_therapeutic: float  # receptor occupancy at therapeutic
    linear_kp: float  # tissue:plasma Kp at low concentration (linear regime)
    saturation_conc_nM: float  # concentration at 50% saturation = Bmax/2
    nonlinearity_flag: bool  # True if fu varies >50% across concentration range
    notes: str


def langmuir_bound(
    free_conc_nM: float,
    bmax_nmol_per_g: float,
    kd_nM: float,
) -> float:
    """Bound = Bmax * C / (Kd + C)  [nmol/g]."""
    if free_conc_nM <= 0.0:
        return 0.0
    return bmax_nmol_per_g * free_conc_nM / (kd_nM + free_conc_nM)


def fu_tissue(
    free_conc_nM: float,
    bmax_nmol_per_g: float,
    kd_nM: float,
    tissue_volume_mL_per_g: float = 1.0,
) -> float:
    """
    fu_tissue = free / total = free / (free + bound*tissue_volume_factor)
    free [nM], bound [nmol/g] needs unit conversion to nM equivalent:
    bound_nM_equiv = bound [nmol/g] * 1000 / tissue_volume_mL_per_g  [nM in tissue water]
    fu = free / (free + bound_nM_equiv)
    """
    if free_conc_nM <= 0.0:
        # At zero concentration, fu approaches 1 (no binding)
        return 1.0

    bound = langmuir_bound(free_conc_nM, bmax_nmol_per_g, kd_nM)
    bound_nM_equiv = bound * 1000.0 / tissue_volume_mL_per_g
    total_nM = free_conc_nM + bound_nM_equiv

    if total_nM <= 0.0:
        return 1.0

    return free_conc_nM / total_nM


def simulate_tissue_binding(
    drug_name: str,
    tissue: str,
    bmax_nmol_per_g: float,  # nmol/g tissue
    kd_nM: float,  # nM
    therapeutic_free_conc_nM: float,
    conc_range_nM: tuple = (0.1, 1.0, 10.0, 100.0, 1000.0),
    tissue_volume_mL_per_g: float = 1.0,
) -> TissueBindingResult:
    """Simulate saturable tissue binding using Langmuir isotherm."""
    if bmax_nmol_per_g <= 0:
        raise ValueError(f"bmax_nmol_per_g must be > 0, got {bmax_nmol_per_g}")
    if kd_nM <= 0:
        raise ValueError(f"kd_nM must be > 0, got {kd_nM}")
    if therapeutic_free_conc_nM <= 0:
        raise ValueError(f"therapeutic_free_conc_nM must be > 0, got {therapeutic_free_conc_nM}")
    if tissue_volume_mL_per_g <= 0:
        raise ValueError(f"tissue_volume_mL_per_g must be > 0, got {tissue_volume_mL_per_g}")
    for c in conc_range_nM:
        if c <= 0:
            raise ValueError(f"All concentrations in conc_range_nM must be > 0, got {c}")

    # Compute bound fractions across concentration range
    concentrations = tuple(conc_range_nM)
    bound_fractions_list = []
    fu_values = []

    for c in concentrations:
        bound = langmuir_bound(c, bmax_nmol_per_g, kd_nM)
        bound_nM_equiv = bound * 1000.0 / tissue_volume_mL_per_g
        total = c + bound_nM_equiv
        bf = bound_nM_equiv / total if total > 0 else 0.0
        bound_fractions_list.append(bf)
        fu_val = fu_tissue(c, bmax_nmol_per_g, kd_nM, tissue_volume_mL_per_g)
        fu_values.append(fu_val)

    # fu at therapeutic concentration
    fu_at_therapeutic = fu_tissue(
        therapeutic_free_conc_nM, bmax_nmol_per_g, kd_nM, tissue_volume_mL_per_g
    )

    # Occupancy at therapeutic = bound / bmax
    bound_at_therapeutic = langmuir_bound(therapeutic_free_conc_nM, bmax_nmol_per_g, kd_nM)
    occupancy_at_therapeutic = bound_at_therapeutic / bmax_nmol_per_g

    # Linear Kp: at C -> 0, Kp = 1 + Bmax/Kd * 1000/tissue_volume
    # (ratio of total to free at low concentration)
    linear_kp = 1.0 + (bmax_nmol_per_g / kd_nM) * (1000.0 / tissue_volume_mL_per_g)

    # Saturation concentration = Kd (50% occupancy occurs at C=Kd)
    saturation_conc_nM = kd_nM

    # Nonlinearity flag: fu varies >50% across concentration range
    if len(fu_values) >= 2:
        fu_min = min(fu_values)
        fu_max = max(fu_values)
        if fu_min > 0:
            relative_variation = (fu_max - fu_min) / fu_min
            nonlinearity_flag = relative_variation > 0.50
        else:
            nonlinearity_flag = True
    else:
        nonlinearity_flag = False

    # Build notes
    notes_parts = [
        f"Bmax={bmax_nmol_per_g:.2f} nmol/g, Kd={kd_nM:.2f} nM",
        f"fu_tissue at {therapeutic_free_conc_nM:.1f} nM = {fu_at_therapeutic:.3f}",
        f"Occupancy={occupancy_at_therapeutic:.3f}",
        f"Linear Kp={linear_kp:.2f}",
    ]
    if nonlinearity_flag:
        notes_parts.append("Nonlinear binding detected")

    notes = "; ".join(notes_parts)

    return TissueBindingResult(
        drug_name=drug_name,
        tissue=tissue,
        bmax_nmol_per_g=bmax_nmol_per_g,
        kd_nM=kd_nM,
        concentrations_nM=concentrations,
        bound_fractions=tuple(bound_fractions_list),
        fu_tissue_at_therapeutic=fu_at_therapeutic,
        occupancy_at_therapeutic=occupancy_at_therapeutic,
        linear_kp=linear_kp,
        saturation_conc_nM=saturation_conc_nM,
        nonlinearity_flag=nonlinearity_flag,
        notes=notes,
    )


def compare_tissues(
    drug_name: str,
    therapeutic_conc_nM: float,
    tissues: list,  # list of dicts: {"name":str, "bmax":float, "kd":float}
) -> list:
    """Return list of TissueBindingResult for each tissue."""
    results = []
    for tissue_dict in tissues:
        result = simulate_tissue_binding(
            drug_name=drug_name,
            tissue=tissue_dict["name"],
            bmax_nmol_per_g=tissue_dict["bmax"],
            kd_nM=tissue_dict["kd"],
            therapeutic_free_conc_nM=therapeutic_conc_nM,
        )
        results.append(result)
    return results
