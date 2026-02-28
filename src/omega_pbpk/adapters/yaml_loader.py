"""YAML → DrugSpec 단일 입력 경로 어댑터.

YAML 파일을 DrugSpec으로 변환하는 단일 진입점.
기존 Drug 하드코딩 파일(drugs/*.py)은 deprecated — spec_to_drug()로 역변환 가능.

Usage:
    from omega_pbpk.adapters import load_drug_spec
    spec = load_drug_spec("compounds/caffeine.yaml")
    spec.validate()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

from omega_pbpk.contracts.drug_spec import DrugSpec
from omega_pbpk.drugs.drug import Drug

# Drug.drug_type → DrugSpec.compound_type 매핑
_DRUG_TYPE_MAP: dict[str, Literal["neutral", "acid", "base", "zwitterion"]] = {
    "neutral": "neutral",
    "monoprotic_acid": "acid",
    "weak_acid": "acid",
    "acid": "acid",
    "monoprotic_base": "base",
    "weak_base": "base",
    "base": "base",
    "diprotic": "zwitterion",
    "zwitterion": "zwitterion",
    "amphoteric": "zwitterion",
}


def drug_to_spec(drug: Drug, param_source: str = "yaml") -> DrugSpec:
    """기존 Drug 객체 → DrugSpec 변환."""
    compound_type = _DRUG_TYPE_MAP.get(getattr(drug, "drug_type", "neutral"), "neutral")
    return DrugSpec(
        name=drug.name,
        smiles=drug.smiles,
        mw=drug.mw,
        logP=drug.logP,
        pka=list(drug.pka),
        compound_type=compound_type,
        fup=drug.fup,
        rbp=drug.rbp,
        clint_hepatic_L_per_h=drug.clint_hepatic_L_per_h,
        clint_gut_L_per_h=drug.clint_gut_L_per_h,
        clr_L_per_h=drug.clr_L_per_h,
        peff=drug.peff,
        solubility_mg_mL=drug.solubility_mg_mL,
        kp=dict(drug.kp),
        permeability_limited={k: dict(v) for k, v in drug.permeability_limited.items()},
        param_source=param_source,  # type: ignore[arg-type]
        prediction_confidence="measured" if param_source == "yaml" else "unknown",
    )


def spec_to_drug(spec: DrugSpec) -> Drug:
    """DrugSpec → 기존 Drug 객체 역변환 (ODE 엔진 호환용)."""
    # compound_type → drug_type 역매핑
    _reverse_map = {
        "neutral": "neutral",
        "acid": "monoprotic_acid",
        "base": "monoprotic_base",
        "zwitterion": "diprotic",
    }
    return Drug(
        name=spec.name,
        smiles=spec.smiles,
        mw=spec.mw,
        logP=spec.logP,
        pka=list(spec.pka),
        drug_type=_reverse_map.get(spec.compound_type, "neutral"),
        compound_type=spec.compound_type,
        fup=spec.fup,
        rbp=spec.rbp,
        clint_hepatic_L_per_h=spec.clint_hepatic_L_per_h,
        clint_gut_L_per_h=spec.clint_gut_L_per_h,
        clr_L_per_h=spec.clr_L_per_h,
        peff=spec.peff,
        solubility_mg_mL=spec.solubility_mg_mL,
        kp=dict(spec.kp),
        permeability_limited={k: dict(v) for k, v in spec.permeability_limited.items()},
    )


def load_drug_spec(path: str | Path) -> DrugSpec:
    """YAML 파일 → DrugSpec (단일 소스 오브 트루스 진입점).

    Args:
        path: compounds/*.yaml 경로. 절대경로 또는 리포 루트 기준 상대경로.

    Returns:
        DrugSpec (validate() 미호출 상태 — 필요시 caller가 호출)

    Raises:
        FileNotFoundError: YAML 파일 없음
    """
    from omega_pbpk.config import load_compound

    drug = load_compound(path)
    return drug_to_spec(drug, param_source="yaml")


def load_drug_spec_by_name(name: str, compounds_dir: str | Path | None = None) -> DrugSpec:
    """약물 이름으로 DrugSpec 로딩.

    Args:
        name: 약물 이름 (예: "caffeine", "warfarin")
        compounds_dir: compounds/ 디렉토리 경로. None이면 리포 루트 자동 탐색.

    Returns:
        DrugSpec
    """
    if compounds_dir is None:
        # 패키지 위치 기준으로 리포 루트 탐색
        pkg_dir = Path(__file__).parent.parent.parent.parent  # src/omega_pbpk/adapters → repo root
        compounds_dir = pkg_dir / "compounds"

    yaml_path = Path(compounds_dir) / f"{name.lower()}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Compound YAML not found: {yaml_path}. "
            f"Available: {[p.stem for p in Path(compounds_dir).glob('*.yaml')]}"
        )
    return load_drug_spec(yaml_path)
