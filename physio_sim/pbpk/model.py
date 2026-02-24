from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physio_sim.config import CompoundConfig, SubjectConfig
from physio_sim.pbpk.heuristics import get_partition_method

COMPARTMENTS: tuple[str, ...] = (
    "GI_lumen",
    "Gut_wall",
    "Portal_vein",
    "Liver",
    "Plasma",
    "Kidney",
    "Lung",
    "Muscle",
    "Fat",
    "Brain",
    "Rest",
    "Urine",
)
IDX = {name: i for i, name in enumerate(COMPARTMENTS)}


@dataclass(frozen=True)
class ModelParams:
    volumes_L: dict[str, float]
    flows_L_per_h: dict[str, float]
    kp: dict[str, float]
    ka_per_h: float
    clh_L_per_h: float
    clr_L_per_h: float
    fu_plasma: float
    first_pass_extraction: float | None


def _effective_clint(compound: CompoundConfig) -> float:
    if compound.ddi is None or not compound.ddi.enabled:
        return compound.clint_L_per_h
    if compound.ddi.ki_mg_per_L is None or compound.ddi.iu_mg_per_L is None:
        return compound.clint_L_per_h
    return compound.clint_L_per_h / (1.0 + compound.ddi.iu_mg_per_L / compound.ddi.ki_mg_per_L)


def build_params(subject: SubjectConfig, compound: CompoundConfig) -> ModelParams:
    q_gut = subject.tissues["gut_wall"].flow_L_per_h
    q_liver = subject.liver.flow_L_per_h
    q_ha = max(0.0, q_liver - q_gut)
    flows = {
        "Gut_wall": q_gut,
        "Liver": q_liver,
        "Liver_artery": q_ha,
        "Kidney": subject.kidney.flow_L_per_h,
        "Lung": subject.tissues["lung"].flow_L_per_h,
        "Muscle": subject.tissues["muscle"].flow_L_per_h,
        "Fat": subject.tissues["fat"].flow_L_per_h,
        "Brain": subject.tissues["brain"].flow_L_per_h,
        "Rest": subject.tissues["rest"].flow_L_per_h,
        "Portal_vein": q_gut,
    }
    volumes = {
        "Gut_wall": subject.tissues["gut_wall"].volume_L,
        "Portal_vein": subject.tissues["portal_vein"].volume_L,
        "Liver": subject.liver.volume_L,
        "Plasma": subject.plasma_volume_L,
        "Kidney": subject.kidney.volume_L,
        "Lung": subject.tissues["lung"].volume_L,
        "Muscle": subject.tissues["muscle"].volume_L,
        "Fat": subject.tissues["fat"].volume_L,
        "Brain": subject.tissues["brain"].volume_L,
        "Rest": subject.tissues["rest"].volume_L,
    }

    kp: dict[str, float] = {}
    tissues_for_kp = [
        "Gut_wall",
        "Portal_vein",
        "Liver",
        "Kidney",
        "Lung",
        "Muscle",
        "Fat",
        "Brain",
        "Rest",
    ]
    partition_method = get_partition_method(compound.partition_method)
    for tissue in tissues_for_kp:
        key = tissue.lower()
        if compound.kp is not None and key in compound.kp:
            kp[tissue] = compound.kp[key]
        else:
            kp[tissue] = partition_method(compound, key)

    clint_eff = _effective_clint(compound)
    qh = subject.liver.flow_L_per_h
    clh = (qh * compound.fu_plasma * clint_eff) / max(1e-9, qh + compound.fu_plasma * clint_eff)

    return ModelParams(
        volumes_L=volumes,
        flows_L_per_h=flows,
        kp=kp,
        ka_per_h=compound.ka_per_h,
        clh_L_per_h=clh,
        clr_L_per_h=compound.clr_L_per_h,
        fu_plasma=compound.fu_plasma,
        first_pass_extraction=compound.first_pass_extraction,
    )


def rhs(_t: float, y: np.ndarray, params: ModelParams) -> np.ndarray:
    y_safe = np.maximum(y, 0.0)
    dy = np.zeros_like(y_safe)

    c_plasma_total = y_safe[IDX["Plasma"]] / params.volumes_L["Plasma"]
    c_plasma_unbound = params.fu_plasma * c_plasma_total

    # Oral absorption
    ka = params.ka_per_h
    dy[IDX["GI_lumen"]] -= ka * y_safe[IDX["GI_lumen"]]
    dy[IDX["Gut_wall"]] += ka * y_safe[IDX["GI_lumen"]]

    def tissue_exchange(name: str) -> float:
        c_t = y_safe[IDX[name]] / params.volumes_L[name]
        q_t = params.flows_L_per_h[name]
        kp_t = params.kp[name]
        return float(q_t * (c_plasma_total - c_t / kp_t))

    for tissue in ["Kidney", "Lung", "Muscle", "Fat", "Brain", "Rest"]:
        exchange = tissue_exchange(tissue)
        dy[IDX[tissue]] += exchange
        dy[IDX["Plasma"]] -= exchange

    # Gut tissue perfusion exchange (with plasma)
    gut_exchange = tissue_exchange("Gut_wall")
    dy[IDX["Gut_wall"]] += gut_exchange
    dy[IDX["Plasma"]] -= gut_exchange

    # Portal vein transfer from gut wall to liver
    c_gut = y_safe[IDX["Gut_wall"]] / params.volumes_L["Gut_wall"]
    q_portal = params.flows_L_per_h["Portal_vein"]
    portal_in = q_portal * (c_gut / params.kp["Gut_wall"])
    if params.first_pass_extraction is not None:
        portal_in *= 1.0 - params.first_pass_extraction
    c_portal = y_safe[IDX["Portal_vein"]] / params.volumes_L["Portal_vein"]
    portal_out = q_portal * c_portal
    dy[IDX["Portal_vein"]] += portal_in - portal_out

    # Hepatic dual inflow: arterial + portal, venous outflow back to plasma
    q_ha = params.flows_L_per_h["Liver_artery"]
    liver_arterial_in = q_ha * c_plasma_total
    c_liver = y_safe[IDX["Liver"]] / params.volumes_L["Liver"]
    c_liver_venous = c_liver / params.kp["Liver"]
    liver_out = params.flows_L_per_h["Liver"] * c_liver_venous
    dy[IDX["Liver"]] += liver_arterial_in + portal_out - liver_out
    dy[IDX["Plasma"]] += liver_out - liver_arterial_in

    # Hepatic elimination strictly in liver using unbound concentration basis
    hepatic_elim = params.clh_L_per_h * (params.fu_plasma * c_liver_venous)
    dy[IDX["Liver"]] -= hepatic_elim

    # Renal elimination from plasma to urine sink using unbound plasma concentration
    renal_elim = params.clr_L_per_h * c_plasma_unbound
    dy[IDX["Plasma"]] -= renal_elim
    dy[IDX["Urine"]] += renal_elim

    return dy
