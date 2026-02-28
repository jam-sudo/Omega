"""VirtualPopulation → PatientSpec 변환 어댑터."""

from __future__ import annotations

from omega_pbpk.contracts.patient_spec import PatientSpec

# Child-Pugh → hepatic_cl_factor 매핑 (physiology.py와 동일)
_CHILD_PUGH_FACTORS: dict[str, float] = {
    "normal": 1.0,
    "A": 0.75,
    "B": 0.50,
    "C": 0.25,
}

_REFERENCE_BW_KG = 70.0
_REFERENCE_CO_L_H = 390.0
_REFERENCE_GFR_ML_MIN = 125.0


def subject_covariates_to_patient_spec(covariates) -> PatientSpec:
    """SubjectCovariates → PatientSpec 변환.

    SubjectCovariates 필드:
      body_weight_kg, age_years (float), sex ("M"/"F"),
      height_cm, bmi, gfr_mL_min, child_pugh, cyp3a4_activity, cyp2d6_activity

    PatientSpec 추가 필드:
      cardiac_output_L_h  — allometric: 390 * (BW/70)^0.75
      hepatic_cl_factor   — Child-Pugh 인수에서 유도
      renal_cl_factor     — gfr_mL_min / 125.0 (정상 GFR 기준)
      cyp2c9_activity     — SubjectCovariates에 없으므로 기본값 1.0
    """
    bw = covariates.body_weight_kg
    co = _REFERENCE_CO_L_H * (bw / _REFERENCE_BW_KG) ** 0.75
    hepatic_factor = _CHILD_PUGH_FACTORS.get(covariates.child_pugh, 1.0)
    renal_factor = covariates.gfr_mL_min / _REFERENCE_GFR_ML_MIN
    sex_str = "male" if covariates.sex == "M" else "female"

    return PatientSpec(
        body_weight_kg=bw,
        age_years=int(covariates.age_years),
        sex=sex_str,  # type: ignore[arg-type]
        height_cm=covariates.height_cm,
        gfr_mL_min=covariates.gfr_mL_min,
        cardiac_output_L_h=round(co, 4),
        child_pugh=covariates.child_pugh,  # type: ignore[arg-type]
        cyp3a4_activity=covariates.cyp3a4_activity,
        cyp2d6_activity=covariates.cyp2d6_activity,
        cyp2c9_activity=1.0,
        hepatic_cl_factor=hepatic_factor,
        renal_cl_factor=renal_factor,
    )


def sample_patient_spec(n: int = 1, seed: int | None = None) -> list[PatientSpec]:
    """VirtualPopulation.sample()을 호출해 PatientSpec 리스트 반환."""
    from omega_pbpk.population.physiology import VirtualPopulation

    pop = VirtualPopulation(seed=seed if seed is not None else 42)
    subjects = pop.sample(n, seed=seed)
    return [subject_covariates_to_patient_spec(s) for s in subjects]
