"""Organ dataclass for PBPK tissue compartments.

Supports both perfusion-limited and permeability-limited distribution.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Organ:
    """Single-organ compartment parameters.

    Attributes:
        name: Organ identifier.
        volume_L: Tissue volume (L).
        blood_flow_L_per_h: Blood flow to this organ (L/h).
        kp: Tissue:plasma partition coefficient.
        is_permeability_limited: Whether organ uses PS barrier model.
        ps_L_per_h: Permeability-surface area product (L/h), only for perm-limited.
        vascular_fraction: Fraction of organ volume that is vascular space (for perm-limited).
    """

    name: str
    volume_L: float
    blood_flow_L_per_h: float
    kp: float = 1.0
    is_permeability_limited: bool = False
    ps_L_per_h: float | None = None
    vascular_fraction: float = 0.04

    @property
    def vascular_volume_L(self) -> float:
        """Vascular space volume (L)."""
        if self.is_permeability_limited:
            return self.volume_L * self.vascular_fraction
        return 0.0

    @property
    def extravascular_volume_L(self) -> float:
        """Extravascular space volume (L)."""
        if self.is_permeability_limited:
            return self.volume_L * (1.0 - self.vascular_fraction)
        return self.volume_L
