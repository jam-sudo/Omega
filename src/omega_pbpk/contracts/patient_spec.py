from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PatientSpec:
    body_weight_kg: float = 70.0
    age_years: int = 30
    sex: Literal["male", "female"] = "male"
    height_cm: float = 170.0
    gfr_mL_min: float = 125.0
    cardiac_output_L_h: float = 390.0
    child_pugh: Literal["normal", "A", "B", "C"] = "normal"
    cyp3a4_activity: float = 1.0   # relative to EM
    cyp2d6_activity: float = 1.0
    cyp2c9_activity: float = 1.0
    hepatic_cl_factor: float = 1.0
    renal_cl_factor: float = 1.0

    def validate(self) -> None:
        if self.body_weight_kg <= 0:
            raise ValueError(f"body_weight_kg must be > 0, got {self.body_weight_kg}")
        if self.age_years < 0:
            raise ValueError(f"age_years must be >= 0, got {self.age_years}")
        if self.gfr_mL_min < 0:
            raise ValueError(f"gfr_mL_min must be >= 0, got {self.gfr_mL_min}")
        if self.cardiac_output_L_h <= 0:
            raise ValueError(f"cardiac_output_L_h must be > 0, got {self.cardiac_output_L_h}")
