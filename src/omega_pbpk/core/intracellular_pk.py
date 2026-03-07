"""Phase 639 — Intracellular PK with lysosomal trapping.

Models drug accumulation in lysosomes for lipophilic basic amines (e.g., chloroquine,
amiodarone). Uses Henderson-Hasselbalch ion-trapping theory and 3-compartment Forward
Euler ODE: Plasma → Cytosol → Lysosome.

Lysosomal accumulation factor (Daniel et al. 2004):
    Klyso = 1 + 10^(pKa_base - pH_lysosome)
This gives the fold-concentration in lysosomal vs cytosolic free drug at equilibrium.
"""

from dataclasses import dataclass


@dataclass
class IntracellularPKResult:
    drug_name: str
    pka_base: float
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_cytosol_mg_L: list
    c_lysosome_mg_L: list
    cmax_plasma: float
    cmax_cytosol: float
    cmax_lysosome: float
    auc_plasma: float
    auc_lysosome: float
    lysosomal_trap_ratio: float  # Clyso/Ccytosol at near-steady-state
    accumulation_index: float  # (cmax_lysosome / cmax_plasma)
    t_half_plasma_h: float
    notes: str


def _validate_params(
    dose_mg: float,
    vd_plasma_L: float,
    cl_L_per_h: float,
    f_lysosome_vol: float,
    ph_lysosome: float,
    ph_cytosol: float,
    pka_base: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if not (0 < f_lysosome_vol < 1):
        raise ValueError(f"f_lysosome_vol must be in (0, 1), got {f_lysosome_vol}")
    if not (4 <= ph_lysosome <= 7):
        raise ValueError(f"ph_lysosome must be in [4, 7], got {ph_lysosome}")
    if not (6 <= ph_cytosol <= 8):
        raise ValueError(f"ph_cytosol must be in [6, 8], got {ph_cytosol}")
    if not (0 < pka_base <= 14):
        raise ValueError(f"pka_base must be in (0, 14], got {pka_base}")


def predict_lysosomal_trapping(
    pka_base: float,
    ph_lysosome: float = 5.0,
    ph_cytosol: float = 7.2,
) -> dict:
    """Return dict with trap_ratio, fold_accumulation, risk_category ('low'/'moderate'/'high').

    Uses the Daniel et al. 2004 lysosomal accumulation model:
        Klyso = 1 + 10^(pKa_base - pH_lysosome)

    This represents the equilibrium fold-concentration of total drug in lysosomes
    relative to the cytosolic free drug concentration.
    """
    # Daniel 2004 lysosomal accumulation factor:
    #   Klyso = 1 + 10^(pKa_base - pH_lysosome)
    # This represents the total drug concentration ratio in the lysosomal lumen
    # relative to the neutral (freely diffusible) form at the same free concentration.
    k_lyso = 1.0 + 10.0 ** (pka_base - ph_lysosome)
    trap_ratio = k_lyso
    fold_accumulation = k_lyso

    if trap_ratio < 10.0:
        risk_category = "low"
    elif trap_ratio < 100.0:
        risk_category = "moderate"
    else:
        risk_category = "high"

    return {
        "trap_ratio": trap_ratio,
        "fold_accumulation": fold_accumulation,
        "risk_category": risk_category,
    }


def simulate_intracellular_pk(
    drug_name: str,
    dose_mg: float,
    pka_base: float = 8.0,
    logP: float = 2.0,
    vd_plasma_L: float = 50.0,
    cl_L_per_h: float = 5.0,
    f_lysosome_vol: float = 0.02,
    ph_lysosome: float = 5.0,
    ph_cytosol: float = 7.2,
    k_mem_per_h: float = 2.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> IntracellularPKResult:
    """Simulate intracellular PK with lysosomal trapping using 3-compartment Forward Euler ODE.

    Compartments:
    - Plasma: 1-cpt IV, first-order elimination (ke = CL/Vd)
    - Cytosol: passive exchange with plasma + trapping into lysosomes
    - Lysosome: ion-trapped drug, slow release back to cytosol

    The lysosomal trap ratio (Klyso = 1 + 10^(pKa - pH_lyso)) drives accumulation.
    At equilibrium: C_lyso / C_cyto = Klyso (the total concentration ratio).
    """
    _validate_params(
        dose_mg, vd_plasma_L, cl_L_per_h, f_lysosome_vol, ph_lysosome, ph_cytosol, pka_base
    )

    # Lysosomal accumulation factor (Daniel 2004)
    k_lyso = 1.0 + 10.0 ** (pka_base - ph_lysosome)
    k_cyto = 1.0 + 10.0 ** (pka_base - ph_cytosol)
    trap_ratio = k_lyso / k_cyto  # equilibrium Clyso/Ccyto

    # Plasma elimination rate constant
    ke = cl_L_per_h / vd_plasma_L

    # Membrane permeability modulated by logP (more lipophilic → faster membrane crossing)
    # Cap logP factor between 0.5 and 3.0 for stability
    logP_factor = max(0.5, min(3.0, 1.0 + logP / 5.0))
    k_mem_eff = k_mem_per_h * logP_factor

    # ODE rate constants for lysosomal trapping.
    # At equilibrium: k_trap * C_cyto = k_release * C_lyso
    # C_lyso / C_cyto = k_trap / k_release = trap_ratio
    # Choose k_release = k_lyso_base (slow release), k_trap = trap_ratio * k_release
    k_release = 0.1  # /h — slow lysosomal release
    k_trap = k_release * trap_ratio

    # Initial conditions (IV bolus into plasma)
    c_plasma = dose_mg / vd_plasma_L
    c_cytosol = 0.0
    c_lysosome = 0.0

    times_h = []
    c_plasma_list = []
    c_cytosol_list = []
    c_lysosome_list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_cytosol_list.append(c_cytosol)
        c_lysosome_list.append(c_lysosome)

        if t >= t_end_h:
            break

        # Forward Euler ODEs
        # Plasma: first-order elimination
        dC_plasma = -ke * c_plasma

        # Cytosol: passive influx from plasma, trapping loss to lysosome
        # Passive exchange with plasma (bidirectional)
        influx_cyto = k_mem_eff * c_plasma
        efflux_cyto = k_mem_eff * c_cytosol
        trap_loss = k_trap * c_cytosol
        release_gain = k_release * c_lysosome
        dC_cytosol = influx_cyto - efflux_cyto - trap_loss + release_gain

        # Lysosome: net accumulation from trapping, slow release back
        dC_lysosome = k_trap * c_cytosol - k_release * c_lysosome

        c_plasma = max(0.0, c_plasma + dC_plasma * dt_h)
        c_cytosol = max(0.0, c_cytosol + dC_cytosol * dt_h)
        c_lysosome = max(0.0, c_lysosome + dC_lysosome * dt_h)

        t = round(t + dt_h, 10)

    # Compute summary stats
    cmax_plasma = max(c_plasma_list)
    cmax_cytosol = max(c_cytosol_list)
    cmax_lysosome = max(c_lysosome_list)

    # Trapezoidal AUC
    def trap_auc(c_list: list, t_list: list) -> float:
        auc = 0.0
        for i in range(len(t_list) - 1):
            auc += (c_list[i] + c_list[i + 1]) / 2.0 * (t_list[i + 1] - t_list[i])
        return auc

    auc_plasma = trap_auc(c_plasma_list, times_h)
    auc_lysosome = trap_auc(c_lysosome_list, times_h)

    # Observed lysosomal trap ratio: C_lyso/C_cyto at peak lysosomal concentration
    # Use time of max lysosomal concentration
    max_lyso_idx = c_lysosome_list.index(cmax_lysosome)
    cyto_at_max_lyso = c_cytosol_list[max_lyso_idx]

    if cyto_at_max_lyso > 1e-12:
        lysosomal_trap_ratio_obs = cmax_lysosome / cyto_at_max_lyso
    else:
        # Fall back to theoretical trap_ratio
        lysosomal_trap_ratio_obs = trap_ratio

    accumulation_index = cmax_lysosome / cmax_plasma if cmax_plasma > 1e-12 else 0.0

    t_half_plasma_h = 0.693 / ke if ke > 0 else 0.0

    notes_parts = [
        f"Lysosomal accumulation factor Klyso={k_lyso:.1f} (Daniel 2004)",
        f"Trap ratio (Klyso/Kcyto)={trap_ratio:.1f}x",
        f"logP={logP:.1f} (k_mem_eff={k_mem_eff:.3f}/h)",
        f"k_trap={k_trap:.3f}/h, k_release={k_release:.3f}/h",
    ]
    pred = predict_lysosomal_trapping(pka_base, ph_lysosome, ph_cytosol)
    risk = pred["risk_category"]
    if risk == "high":
        notes_parts.append("HIGH lysosomal accumulation risk — phospholipidosis possible")
    elif risk == "moderate":
        notes_parts.append("MODERATE lysosomal accumulation")
    else:
        notes_parts.append("LOW lysosomal trapping")

    return IntracellularPKResult(
        drug_name=drug_name,
        pka_base=pka_base,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_cytosol_mg_L=c_cytosol_list,
        c_lysosome_mg_L=c_lysosome_list,
        cmax_plasma=cmax_plasma,
        cmax_cytosol=cmax_cytosol,
        cmax_lysosome=cmax_lysosome,
        auc_plasma=auc_plasma,
        auc_lysosome=auc_lysosome,
        lysosomal_trap_ratio=lysosomal_trap_ratio_obs,
        accumulation_index=accumulation_index,
        t_half_plasma_h=t_half_plasma_h,
        notes="; ".join(notes_parts),
    )
