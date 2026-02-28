from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class DrugSpec:
    name: str
    smiles: str | None = None
    mw: float = 300.0
    logP: float = 2.0
    pka: list[float] = field(default_factory=lambda: [7.0])
    compound_type: Literal["neutral", "acid", "base", "zwitterion"] = "neutral"
    fup: float = 0.5  # fraction unbound in plasma
    rbp: float = 1.0  # blood:plasma ratio
    clint_hepatic_L_per_h: float = 0.0
    clint_gut_L_per_h: float = 0.0
    clr_L_per_h: float = 0.0
    peff: float = 1.0  # effective permeability (cm/s × 10^-4)
    solubility_mg_mL: float = 1.0
    kp: dict[str, float] = field(default_factory=dict)
    permeability_limited: dict[str, dict[str, float]] = field(default_factory=dict)
    param_source: Literal["yaml", "python", "ml_predicted", "measured"] = "yaml"
    prediction_confidence: str = "unknown"

    def validate(self) -> None:
        """물리적으로 타당하지 않은 값이면 ValueError 발생"""
        if not (0 < self.fup <= 1):
            raise ValueError(f"fup must be in (0, 1], got {self.fup}")
        if self.rbp <= 0:
            raise ValueError(f"rbp must be > 0, got {self.rbp}")
        if self.mw <= 0:
            raise ValueError(f"mw must be > 0, got {self.mw}")
        if self.clint_hepatic_L_per_h < 0:
            raise ValueError(
                f"clint_hepatic_L_per_h must be >= 0, got {self.clint_hepatic_L_per_h}"
            )
        if self.clr_L_per_h < 0:
            raise ValueError(f"clr_L_per_h must be >= 0, got {self.clr_L_per_h}")
        if self.peff < 0:
            raise ValueError(f"peff must be >= 0, got {self.peff}")
        if self.solubility_mg_mL <= 0:
            raise ValueError(f"solubility_mg_mL must be > 0, got {self.solubility_mg_mL}")
        for organ, kp_val in self.kp.items():
            if kp_val < 0:
                raise ValueError(f"kp[{organ}] must be >= 0, got {kp_val}")
