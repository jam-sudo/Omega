from __future__ import annotations

from collections.abc import Callable

from physio_sim.config import CompoundConfig

PartitionMethod = Callable[[CompoundConfig, str], float]


def heuristic_kp(logp: float, pka: float | None, tissue_name: str) -> float:
    """Heuristic-only Kp estimator (not validated for clinical use)."""
    base = 1.0 + 0.3 * logp
    if pka is not None:
        base += 0.05 * (pka - 7.0)
    tissue_adjust = {
        "fat": 1.6,
        "brain": 1.2,
        "muscle": 1.0,
        "rest": 0.9,
        "kidney": 1.0,
        "liver": 1.1,
        "gut_wall": 1.0,
        "portal_vein": 1.0,
    }.get(tissue_name, 1.0)
    return max(0.2, base * tissue_adjust)


def heuristic_partition_method(compound: CompoundConfig, tissue_name: str) -> float:
    """Default modular Kp calculation method."""
    return heuristic_kp(compound.logp, compound.pka, tissue_name)


def get_partition_method(method_name: str) -> PartitionMethod:
    """Return a partition method callable by name.

    This provides an extension point for future methods such as
    Rodgers & Rowland and Poulin & Theil.
    """
    methods: dict[str, PartitionMethod] = {
        "heuristic": heuristic_partition_method,
    }
    return methods.get(method_name, heuristic_partition_method)
