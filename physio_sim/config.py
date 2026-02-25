from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from physio_sim.qsp.registry import validate_qsp_parameters


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
    partition_method: Literal["heuristic"] = "heuristic"
    clint_L_per_h: float = Field(ge=0)
    clr_L_per_h: float = Field(ge=0)
    kp: dict[str, float] | None = None
    pd: PDConfig
    ddi: DDIConfig | None = None

    @field_validator("fu_plasma", "f_gut")
    @classmethod
    def validate_fraction(cls, value: float | None) -> float | None:
        if value is None:
            return value
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
    seed: int | None = Field(default=None)


class QSPConfig(BaseModel):
    model: str
    signal: Literal["plasma_total", "plasma_unbound"] = "plasma_total"
    params: dict[str, float]

    @field_validator("params")
    @classmethod
    def _validate_params(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, raw in value.items():
            if not isinstance(raw, int | float):
                raise ValueError(f"qsp param {key} must be numeric")
            normalized[key] = float(raw)
        return normalized


def load_qsp(path: str | Path) -> QSPConfig:
    cfg = QSPConfig.model_validate(_read_yaml(Path(path)))
    validate_qsp_parameters(cfg.model, cfg.params)
    return cfg


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"YAML root must be a mapping: {path}"
        raise ValueError(msg)
    return cast(dict[str, object], data)


def load_subject(path: str | Path) -> SubjectConfig:
    return SubjectConfig.model_validate(_read_yaml(Path(path)))


def load_compound(path: str | Path) -> CompoundConfig:
    data = _read_yaml(Path(path))
    first_pass_extraction = data.get("first_pass_extraction")
    f_gut = data.get("f_gut")

    if first_pass_extraction is not None:
        if f_gut is None:
            if not isinstance(first_pass_extraction, float | int):
                msg = "first_pass_extraction must be a numeric fraction."
                raise ValueError(msg)
            data["f_gut"] = 1.0 - float(first_pass_extraction)
        warnings.warn(
            "'first_pass_extraction' is deprecated; use 'f_gut' "
            "(fraction escaping gut-wall metabolism).",
            DeprecationWarning,
            stacklevel=2,
        )
    return CompoundConfig.model_validate(data)


def friendly_validation_error(exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        error_map = cast(dict[str, Any], error)
        loc = cast(tuple[object, ...], error_map["loc"])
        message = cast(str, error_map["msg"])
        lines.append(f"{'.'.join(map(str, loc))}: {message}")
    return "\n".join(lines)


def configure_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
