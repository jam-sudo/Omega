from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class TissueConfig(BaseModel):
    volume_L: float = Field(gt=0)
    flow_L_per_h: float = Field(gt=0)


class SubjectConfig(BaseModel):
    body_weight_kg: float = Field(gt=0)
    cardiac_output_L_per_h: float = Field(gt=0)
    plasma_volume_L: float = Field(default=3.0, gt=0)
    hematocrit: float | None = Field(default=0.45)
    liver: TissueConfig
    kidney: TissueConfig
    tissues: dict[str, TissueConfig]

    @field_validator("hematocrit")
    @classmethod
    def validate_hematocrit(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not (0.0 <= value <= 1.0):
            msg = "hematocrit must be within [0, 1]"
            raise ValueError(msg)
        return value


class PDConfig(BaseModel):
    model: Literal["emax"] = "emax"
    e0: float
    emax: float
    ec50_mg_per_L: float = Field(gt=0)
    hill: float = Field(default=1.0, gt=0)
    ke0_per_h: float | None = Field(default=None, gt=0)


class DDIConfig(BaseModel):
    enabled: bool = False
    ki_mg_per_L: float | None = Field(default=None, gt=0)
    iu_mg_per_L: float | None = Field(default=None, ge=0)


class CompoundConfig(BaseModel):
    name: str
    mw_g_per_mol: float | None = Field(default=None, gt=0)
    logp: float
    pka: float | None = None
    fu_plasma: float
    ka_per_h: float = Field(gt=0)
    f_gut: float = Field(default=1.0)
    clint_L_per_h: float = Field(ge=0)
    clr_L_per_h: float = Field(ge=0)
    kp: dict[str, float] | None = None
    pd: PDConfig
    ddi: DDIConfig | None = None

    @field_validator("fu_plasma", "f_gut")
    @classmethod
    def validate_fraction(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            msg = "fraction parameters must be within [0, 1]"
            raise ValueError(msg)
        return value


class SimulationRequest(BaseModel):
    compound_file: str
    subject_file: str
    dose_mg: float = Field(gt=0)
    route: Literal["oral", "iv"]
    t_end_h: float = Field(gt=0)
    dt_out_h: float = Field(default=0.1, gt=0)


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"YAML root must be a mapping: {path}"
        raise ValueError(msg)
    return data


def load_subject(path: str | Path) -> SubjectConfig:
    return SubjectConfig.model_validate(_read_yaml(Path(path)))


def load_compound(path: str | Path) -> CompoundConfig:
    return CompoundConfig.model_validate(_read_yaml(Path(path)))


def friendly_validation_error(exc: ValidationError) -> str:
    return "\n".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
