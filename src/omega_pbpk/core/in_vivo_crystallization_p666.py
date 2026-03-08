"""Phase 666 — Drug Crystallization Kinetics in Vivo.

Models kinetics of drug crystallization in supersaturated biological fluids
(urine, bile, synovial fluid, aqueous humor) after administration.
"""

from dataclasses import dataclass, field

VALID_BIOFLUIDS = {"urine", "bile", "synovial_fluid", "aqueous_humor"}

BIOFLUID_PARAMS = {
    "urine": {"volume_mL": 1440.0, "ph": 6.0, "protein_mg_mL": 0.01, "flow_mL_h": 60.0},
    "bile": {"volume_mL": 50.0, "ph": 7.0, "protein_mg_mL": 2.0, "flow_mL_h": 0.0},
    "synovial_fluid": {
        "volume_mL": 2.0,
        "ph": 7.4,
        "protein_mg_mL": 20.0,
        "flow_mL_h": 0.0,
    },
    "aqueous_humor": {
        "volume_mL": 0.25,
        "ph": 7.2,
        "protein_mg_mL": 0.1,
        "flow_mL_h": 2.0,
    },
}


@dataclass
class InVivoCrystallizationResult:
    drug_name: str
    dose_mg: float
    biofluid: str
    times_h: list = field(default_factory=list)
    c_dissolved_mg_mL: list = field(default_factory=list)
    c_crystal_mg_mL: list = field(default_factory=list)
    supersaturation_ratio: list = field(default_factory=list)
    nucleation_rate_per_h: list = field(default_factory=list)
    crystallization_half_time_h: float = 0.0
    induction_time_h: float = 0.0
    max_crystal_fraction_pct: float = 0.0
    risk_category: str = "low"
    physiological_consequence: str = "minimal"
    notes: str = ""


def simulate_in_vivo_crystallization(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    logP: float,
    biofluid: str = "urine",
    concentration_mg_mL: float | None = None,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> InVivoCrystallizationResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if biofluid not in VALID_BIOFLUIDS:
        raise ValueError(f"biofluid must be one of {VALID_BIOFLUIDS}")

    params = BIOFLUID_PARAMS[biofluid]
    volume_mL = params["volume_mL"]
    protein_mg_mL = params["protein_mg_mL"]
    flow_mL_h = params["flow_mL_h"]

    if concentration_mg_mL is None:
        concentration_mg_mL = dose_mg / volume_mL * 0.1

    # Effective solubility
    sol_eff = solubility_mg_mL * (1.0 + protein_mg_mL * 0.01)

    # Initial conditions
    C_dissolved = concentration_mg_mL
    C_crystal = 0.0
    initial_conc = concentration_mg_mL

    n_steps = int(t_end_h / dt_h) + 1

    times_h = []
    c_dissolved_list = []
    c_crystal_list = []
    sr_list = []
    nuc_rate_list = []

    for i in range(n_steps):
        t = i * dt_h
        SR = C_dissolved / sol_eff if sol_eff > 0 else 0.0

        if SR > 1.0:
            nucleation_rate = 0.1 * (SR - 1.0) ** 2
            growth_rate = 0.5 * (SR - 1.0)
        else:
            nucleation_rate = 0.0
            growth_rate = 0.0

        times_h.append(t)
        c_dissolved_list.append(C_dissolved)
        c_crystal_list.append(C_crystal)
        sr_list.append(SR)
        nuc_rate_list.append(nucleation_rate)

        # Forward Euler
        dC_dissolved = -growth_rate * C_crystal - nucleation_rate * sol_eff
        if flow_mL_h > 0:
            dC_dissolved -= (flow_mL_h / volume_mL) * (C_dissolved - sol_eff)

        dC_crystal = growth_rate * C_crystal + nucleation_rate * sol_eff

        C_dissolved += dC_dissolved * dt_h
        C_crystal += dC_crystal * dt_h

        if C_dissolved < 0:
            C_dissolved = 0.0
        if C_crystal < 0:
            C_crystal = 0.0

    # Induction time
    threshold = 0.001 * initial_conc
    induction_time_h = t_end_h
    for i in range(len(c_crystal_list)):
        if c_crystal_list[i] > threshold:
            induction_time_h = times_h[i]
            break

    # Max crystal and half time
    max_crystal = max(c_crystal_list)
    crystallization_half_time_h = t_end_h
    if max_crystal > 0:
        target = max_crystal * 0.5
        for i in range(len(c_crystal_list)):
            if c_crystal_list[i] >= target:
                crystallization_half_time_h = times_h[i]
                break

    # Max crystal fraction
    total_final = c_dissolved_list[-1] + c_crystal_list[-1]
    if total_final > 0:
        max_crystal_fraction_pct = c_crystal_list[-1] / total_final * 100.0
    else:
        max_crystal_fraction_pct = 0.0

    # Risk category based on initial SR
    sr_initial = sr_list[0] if sr_list else 0.0
    if sr_initial > 5.0:
        risk_category = "critical"
    elif sr_initial > 2.0:
        risk_category = "high"
    elif sr_initial > 1.0:
        risk_category = "moderate"
    else:
        risk_category = "low"

    # Physiological consequence
    if biofluid == "urine" and risk_category in ("critical", "high"):
        physiological_consequence = "nephrolithiasis_risk"
    elif biofluid == "bile" and risk_category in ("critical", "high", "moderate"):
        physiological_consequence = "gallstone_risk"
    elif biofluid == "synovial_fluid" and risk_category in ("critical", "high", "moderate"):
        physiological_consequence = "crystal_arthropathy_risk"
    elif biofluid == "aqueous_humor" and risk_category in ("critical", "high", "moderate"):
        physiological_consequence = "ocular_precipitation_risk"
    else:
        physiological_consequence = "minimal"

    notes = (
        f"Biofluid={biofluid}, sol_eff={sol_eff:.4f} mg/mL, "
        f"initial SR={sr_initial:.2f}, risk={risk_category}"
    )

    return InVivoCrystallizationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        biofluid=biofluid,
        times_h=times_h,
        c_dissolved_mg_mL=c_dissolved_list,
        c_crystal_mg_mL=c_crystal_list,
        supersaturation_ratio=sr_list,
        nucleation_rate_per_h=nuc_rate_list,
        crystallization_half_time_h=crystallization_half_time_h,
        induction_time_h=induction_time_h,
        max_crystal_fraction_pct=max_crystal_fraction_pct,
        risk_category=risk_category,
        physiological_consequence=physiological_consequence,
        notes=notes,
    )
