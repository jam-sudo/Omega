from pathlib import Path

import pytest
from pydantic import ValidationError

from physio_sim.config import load_compound, load_qsp


def test_missing_required_field_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nfu_plasma: 0.5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_compound(bad)


def test_fraction_domain_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad2.yaml"
    bad.write_text(
        """
name: x
logp: 1.0
fu_plasma: 1.5
ka_per_h: 1.0
f_gut: 1.0
clint_L_per_h: 1.0
clr_L_per_h: 1.0
pd:
  model: emax
  e0: 0
  emax: 1
  ec50_mg_per_L: 1
  hill: 1
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_compound(bad)


def test_legacy_first_pass_extraction_maps_to_f_gut(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        """
name: x
logp: 1.0
fu_plasma: 0.5
ka_per_h: 1.0
first_pass_extraction: 0.2
clint_L_per_h: 1.0
clr_L_per_h: 1.0
pd:
  model: emax
  e0: 0
  emax: 1
  ec50_mg_per_L: 1
  hill: 1
""",
        encoding="utf-8",
    )
    with pytest.deprecated_call():
        cfg = load_compound(legacy)
    assert cfg.f_gut == pytest.approx(0.8)


def test_load_qsp_turnover_validation_missing_required_param(tmp_path: Path) -> None:
    cfg = tmp_path / "bad_qsp.yaml"
    cfg.write_text(
        """
model: turnover
signal: plasma_total
params:
  kin: 1.0
  kout: 0.2
  emax: 1.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_qsp(cfg)


def test_load_qsp_turnover_validation_negative_kout(tmp_path: Path) -> None:
    cfg = tmp_path / "bad_qsp.yaml"
    cfg.write_text(
        """
model: turnover
signal: plasma_total
params:
  kin: 1.0
  kout: -0.2
  emax: 1.0
  ec50_mg_per_L: 1.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_qsp(cfg)


def test_load_qsp_unknown_model_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "bad_qsp.yaml"
    cfg.write_text(
        """
model: unknown
signal: plasma_total
params:
  a: 1.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_qsp(cfg)
