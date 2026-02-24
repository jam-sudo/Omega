from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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
    "Gut_metabolism",
)
IDX = {name: i for i, name in enumerate(COMPARTMENTS)}
EXCHANGE_TISSUES: tuple[str, ...] = ("Kidney", "Lung", "Muscle", "Fat", "Brain", "Rest")


@dataclass(frozen=True)
class ModelParams:
    volumes_L: dict[str, float]
    flows_L_per_h: dict[str, float]
    kp: dict[str, float]
    ka_per_h: float
    clh_L_per_h: float
    clr_L_per_h: float
    fu_plasma: float
    f_gut: float


@dataclass(frozen=True)
class ModelCache:
    exchange_indices: NDArray[np.int64]
    exchange_volumes: NDArray[np.float64]
    exchange_flows: NDArray[np.float64]
    exchange_kp: NDArray[np.float64]
    idx_gi_lumen: int
    idx_gut_wall: int
    idx_portal_vein: int
    idx_liver: int
    idx_plasma: int
    idx_urine: int
    idx_gut_metabolism: int
    volume_plasma: float
    volume_gut_wall: float
    volume_portal: float
    volume_liver: float
    flow_portal: float
    flow_liver_artery: float
    flow_liver: float
    kp_gut_wall: float
    kp_liver: float


def build_cache(params: ModelParams) -> ModelCache:
    exchange_indices = np.array([IDX[name] for name in EXCHANGE_TISSUES], dtype=np.int64)
    return ModelCache(
        exchange_indices=exchange_indices,
        exchange_volumes=np.array(
            [params.volumes_L[name] for name in EXCHANGE_TISSUES],
            dtype=float,
        ),
        exchange_flows=np.array(
            [params.flows_L_per_h[name] for name in EXCHANGE_TISSUES],
            dtype=float,
        ),
        exchange_kp=np.array([params.kp[name] for name in EXCHANGE_TISSUES], dtype=float),
        idx_gi_lumen=IDX["GI_lumen"],
        idx_gut_wall=IDX["Gut_wall"],
        idx_portal_vein=IDX["Portal_vein"],
        idx_liver=IDX["Liver"],
        idx_plasma=IDX["Plasma"],
        idx_urine=IDX["Urine"],
        idx_gut_metabolism=IDX["Gut_metabolism"],
        volume_plasma=params.volumes_L["Plasma"],
        volume_gut_wall=params.volumes_L["Gut_wall"],
        volume_portal=params.volumes_L["Portal_vein"],
        volume_liver=params.volumes_L["Liver"],
        flow_portal=params.flows_L_per_h["Portal_vein"],
        flow_liver_artery=params.flows_L_per_h["Liver_artery"],
        flow_liver=params.flows_L_per_h["Liver"],
        kp_gut_wall=params.kp["Gut_wall"],
        kp_liver=params.kp["Liver"],
    )


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
        f_gut=compound.f_gut,
    )


def rhs(
    _t: float,
    y: NDArray[np.float64],
    params: ModelParams,
    cache: ModelCache | None = None,
) -> NDArray[np.float64]:
    runtime_cache = cache if cache is not None else build_cache(params)
    y_safe = np.maximum(y, 0.0)
    dy = np.zeros_like(y_safe)

    c_plasma_total = y_safe[runtime_cache.idx_plasma] / runtime_cache.volume_plasma
    c_plasma_unbound = params.fu_plasma * c_plasma_total

    # Oral absorption
    ka = params.ka_per_h
    gi_loss = ka * y_safe[runtime_cache.idx_gi_lumen]
    dy[runtime_cache.idx_gi_lumen] -= gi_loss
    dy[runtime_cache.idx_gut_wall] += gi_loss

    tissue_amounts = y_safe[runtime_cache.exchange_indices]
    tissue_exchange = runtime_cache.exchange_flows * (
        c_plasma_total
        - (tissue_amounts / runtime_cache.exchange_volumes) / runtime_cache.exchange_kp
    )
    dy[runtime_cache.exchange_indices] += tissue_exchange
    dy[runtime_cache.idx_plasma] -= float(np.sum(tissue_exchange))

    # Gut tissue perfusion exchange (with plasma)
    c_gut = y_safe[runtime_cache.idx_gut_wall] / runtime_cache.volume_gut_wall
    gut_exchange = params.flows_L_per_h["Gut_wall"] * (
        c_plasma_total - c_gut / runtime_cache.kp_gut_wall
    )
    dy[runtime_cache.idx_gut_wall] += gut_exchange
    dy[runtime_cache.idx_plasma] -= gut_exchange

    # Portal vein transfer from gut wall to liver
    portal_in_raw = runtime_cache.flow_portal * (c_gut / runtime_cache.kp_gut_wall)
    portal_in = params.f_gut * portal_in_raw
    gut_metabolism_loss = (1.0 - params.f_gut) * portal_in_raw
    c_portal = y_safe[runtime_cache.idx_portal_vein] / runtime_cache.volume_portal
    portal_out = runtime_cache.flow_portal * c_portal
    dy[runtime_cache.idx_gut_wall] -= portal_in_raw
    dy[runtime_cache.idx_gut_metabolism] += gut_metabolism_loss
    dy[runtime_cache.idx_portal_vein] += portal_in - portal_out

    # Hepatic dual inflow: arterial + portal, venous outflow back to plasma
    liver_arterial_in = runtime_cache.flow_liver_artery * c_plasma_total
    c_liver = y_safe[runtime_cache.idx_liver] / runtime_cache.volume_liver
    c_liver_venous = c_liver / runtime_cache.kp_liver
    liver_out = runtime_cache.flow_liver * c_liver_venous
    dy[runtime_cache.idx_liver] += liver_arterial_in + portal_out - liver_out
    dy[runtime_cache.idx_plasma] += liver_out - liver_arterial_in

    # Hepatic elimination strictly in liver using unbound concentration basis
    hepatic_elim = params.clh_L_per_h * (params.fu_plasma * c_liver_venous)
    dy[runtime_cache.idx_liver] -= hepatic_elim

    # Renal elimination from plasma to urine sink using unbound plasma concentration
    renal_elim = params.clr_L_per_h * c_plasma_unbound
    dy[runtime_cache.idx_plasma] -= renal_elim
    dy[runtime_cache.idx_urine] += renal_elim

    return dy
