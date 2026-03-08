"""Tests for Phase 285 — Neonatal PK Scaling.

Tests for src/omega_pbpk/clinical/neonatal_pk_scaling_p285.py
"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.neonatal_pk_scaling_p285 import (
    NeonatalPKResult,
    compare_gestational_ages,
    scale_neonatal_pk,
)

# --------------------------------------------------------------------------- #
# Shared fixture helpers
# --------------------------------------------------------------------------- #
ADULT_CL = 10.0  # L/h
ADULT_VD = 70.0  # L (roughly 1 L/kg for 70 kg adult)
ADULT_WT = 70.0  # kg
NEO_WT = 1.5  # kg (very preterm weight)
FU = 0.1
LOGP = 2.0


def _default_result(ga: float = 30.0, pna: float = 7.0) -> NeonatalPKResult:
    return scale_neonatal_pk(
        drug_name="TestDrug",
        adult_cl_L_per_h=ADULT_CL,
        adult_vd_L=ADULT_VD,
        adult_weight_kg=ADULT_WT,
        neonatal_weight_kg=NEO_WT,
        ga_weeks=ga,
        pna_days=pna,
        fu_adult=FU,
        logp=LOGP,
    )


# --------------------------------------------------------------------------- #
# 1. Return type
# --------------------------------------------------------------------------- #
def test_return_type():
    r = _default_result()
    assert isinstance(r, NeonatalPKResult)


# --------------------------------------------------------------------------- #
# 2. Field preservation
# --------------------------------------------------------------------------- #
def test_field_preservation():
    r = scale_neonatal_pk(
        drug_name="Amikacin",
        adult_cl_L_per_h=ADULT_CL,
        adult_vd_L=ADULT_VD,
        adult_weight_kg=ADULT_WT,
        neonatal_weight_kg=NEO_WT,
        ga_weeks=32.0,
        pna_days=3.0,
        fu_adult=FU,
        logp=LOGP,
    )
    assert r.drug_name == "Amikacin"
    assert r.ga_weeks == 32.0
    assert r.pna_days == 3.0
    assert r.neonatal_weight_kg == NEO_WT


# --------------------------------------------------------------------------- #
# 3. PMA calculation
# --------------------------------------------------------------------------- #
def test_pma_weeks_calculation():
    r = scale_neonatal_pk(
        drug_name="D",
        adult_cl_L_per_h=ADULT_CL,
        adult_vd_L=ADULT_VD,
        adult_weight_kg=ADULT_WT,
        neonatal_weight_kg=NEO_WT,
        ga_weeks=28.0,
        pna_days=14.0,
        fu_adult=FU,
        logp=LOGP,
    )
    assert abs(r.pma_weeks - (28.0 + 14.0 / 7.0)) < 1e-9


# --------------------------------------------------------------------------- #
# 4. CL > 0
# --------------------------------------------------------------------------- #
def test_cl_neonatal_positive():
    r = _default_result()
    assert r.cl_neonatal_L_per_h > 0.0


# --------------------------------------------------------------------------- #
# 5. Vd > 0
# --------------------------------------------------------------------------- #
def test_vd_neonatal_positive():
    r = _default_result()
    assert r.vd_neonatal_L > 0.0


# --------------------------------------------------------------------------- #
# 6. Maturation factor in (0, 1)
# --------------------------------------------------------------------------- #
def test_maturation_factor_range():
    for ga in [24.0, 28.0, 32.0, 37.0, 40.0]:
        r = _default_result(ga=ga)
        assert 0.0 < r.maturation_factor < 1.0, f"MF out of range for GA={ga}"


# --------------------------------------------------------------------------- #
# 7. Higher GA → higher MF
# --------------------------------------------------------------------------- #
def test_higher_ga_higher_mf():
    r_preterm = _default_result(ga=25.0, pna=0.0)
    r_term = _default_result(ga=40.0, pna=0.0)
    assert r_term.maturation_factor > r_preterm.maturation_factor


# --------------------------------------------------------------------------- #
# 8. Higher GA → higher cl_neonatal
# --------------------------------------------------------------------------- #
def test_higher_ga_higher_cl():
    r_low = _default_result(ga=24.0, pna=0.0)
    r_high = _default_result(ga=40.0, pna=0.0)
    assert r_high.cl_neonatal_L_per_h > r_low.cl_neonatal_L_per_h


# --------------------------------------------------------------------------- #
# 9. Higher PNA → higher PMA → higher MF → higher CL
# --------------------------------------------------------------------------- #
def test_higher_pna_higher_cl():
    r_day0 = _default_result(ga=30.0, pna=0.0)
    r_day28 = _default_result(ga=30.0, pna=28.0)
    assert r_day28.cl_neonatal_L_per_h > r_day0.cl_neonatal_L_per_h


# --------------------------------------------------------------------------- #
# 10. fu_neonatal >= fu_adult
# --------------------------------------------------------------------------- #
def test_fu_neonatal_geq_fu_adult():
    r = _default_result()
    assert r.fu_neonatal >= FU


# --------------------------------------------------------------------------- #
# 11. fu_neonatal <= 1
# --------------------------------------------------------------------------- #
def test_fu_neonatal_leq_one():
    # fu_adult=0.8 → fu_neonatal would be 1.2 → clamped to 1.0
    r = scale_neonatal_pk(
        drug_name="D",
        adult_cl_L_per_h=ADULT_CL,
        adult_vd_L=ADULT_VD,
        adult_weight_kg=ADULT_WT,
        neonatal_weight_kg=NEO_WT,
        ga_weeks=37.0,
        pna_days=0.0,
        fu_adult=0.8,
        logp=LOGP,
    )
    assert r.fu_neonatal <= 1.0


# --------------------------------------------------------------------------- #
# 12. t_half > 0
# --------------------------------------------------------------------------- #
def test_t_half_positive():
    r = _default_result()
    assert r.t_half_neonatal_h > 0.0
    assert math.isfinite(r.t_half_neonatal_h)


# --------------------------------------------------------------------------- #
# 13. maturation_stage valid values
# --------------------------------------------------------------------------- #
VALID_STAGES = {"extremely_preterm", "very_preterm", "preterm", "term"}


def test_maturation_stage_valid():
    for ga in [24.0, 28.0, 30.0, 35.0, 39.0]:
        r = _default_result(ga=ga)
        assert r.maturation_stage in VALID_STAGES


def test_maturation_stage_extremely_preterm():
    r = _default_result(ga=24.0)
    assert r.maturation_stage == "extremely_preterm"


def test_maturation_stage_very_preterm():
    r = _default_result(ga=29.0)
    assert r.maturation_stage == "very_preterm"


def test_maturation_stage_preterm():
    r = _default_result(ga=34.0)
    assert r.maturation_stage == "preterm"


def test_maturation_stage_term():
    r = _default_result(ga=39.0)
    assert r.maturation_stage == "term"


# --------------------------------------------------------------------------- #
# 14. dose_adjustment_factor > 0 and <= 1 for immature neonates
# --------------------------------------------------------------------------- #
def test_dose_adjustment_factor_range():
    r = _default_result(ga=28.0)
    assert r.dose_adjustment_factor > 0.0
    # For immature neonates MF < 1 so cl_neonatal < cl_allometric
    assert r.dose_adjustment_factor <= 1.0


# --------------------------------------------------------------------------- #
# 15. notes is non-empty string
# --------------------------------------------------------------------------- #
def test_notes_string():
    r = _default_result()
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0


# --------------------------------------------------------------------------- #
# 16. Negative logp → fat component zero
# --------------------------------------------------------------------------- #
def test_negative_logp_no_fat():
    r_neg = scale_neonatal_pk(
        drug_name="D",
        adult_cl_L_per_h=ADULT_CL,
        adult_vd_L=ADULT_VD,
        adult_weight_kg=ADULT_WT,
        neonatal_weight_kg=NEO_WT,
        ga_weeks=37.0,
        pna_days=0.0,
        fu_adult=FU,
        logp=-2.0,
    )
    r_pos = scale_neonatal_pk(
        drug_name="D",
        adult_cl_L_per_h=ADULT_CL,
        adult_vd_L=ADULT_VD,
        adult_weight_kg=ADULT_WT,
        neonatal_weight_kg=NEO_WT,
        ga_weeks=37.0,
        pna_days=0.0,
        fu_adult=FU,
        logp=5.0,
    )
    # Negative logp → no fat component → smaller Vd
    assert r_neg.vd_neonatal_L <= r_pos.vd_neonatal_L


# --------------------------------------------------------------------------- #
# 17. compare_gestational_ages — return type and length
# --------------------------------------------------------------------------- #
def test_compare_ga_return_length():
    gas = [24.0, 28.0, 32.0, 37.0, 40.0]
    results = compare_gestational_ages(
        drug_name="TestDrug",
        adult_cl_L_per_h=ADULT_CL,
        adult_vd_L=ADULT_VD,
        adult_weight_kg=ADULT_WT,
        neonatal_weight_kg=NEO_WT,
        pna_days=7.0,
        fu_adult=FU,
        logp=LOGP,
        ga_weeks_list=gas,
    )
    assert len(results) == len(gas)
    assert all(isinstance(r, NeonatalPKResult) for r in results)


# --------------------------------------------------------------------------- #
# 18. compare_gestational_ages — sorted ascending by cl_neonatal
# --------------------------------------------------------------------------- #
def test_compare_ga_sorted_ascending():
    gas = [40.0, 24.0, 30.0, 35.0]
    results = compare_gestational_ages(
        drug_name="TestDrug",
        adult_cl_L_per_h=ADULT_CL,
        adult_vd_L=ADULT_VD,
        adult_weight_kg=ADULT_WT,
        neonatal_weight_kg=NEO_WT,
        pna_days=0.0,
        fu_adult=FU,
        logp=LOGP,
        ga_weeks_list=gas,
    )
    cls = [r.cl_neonatal_L_per_h for r in results]
    assert cls == sorted(cls), f"Not sorted ascending: {cls}"


# --------------------------------------------------------------------------- #
# 19. Validation — negative CL
# --------------------------------------------------------------------------- #
def test_validation_negative_cl():
    with pytest.raises(ValueError, match="adult_cl_L_per_h"):
        scale_neonatal_pk("D", -1.0, ADULT_VD, ADULT_WT, NEO_WT, 30.0, 0.0, FU, LOGP)


# --------------------------------------------------------------------------- #
# 20. Validation — invalid GA
# --------------------------------------------------------------------------- #
def test_validation_ga_too_low():
    with pytest.raises(ValueError, match="ga_weeks"):
        scale_neonatal_pk("D", ADULT_CL, ADULT_VD, ADULT_WT, NEO_WT, 20.0, 0.0, FU, LOGP)


def test_validation_ga_too_high():
    with pytest.raises(ValueError, match="ga_weeks"):
        scale_neonatal_pk("D", ADULT_CL, ADULT_VD, ADULT_WT, NEO_WT, 45.0, 0.0, FU, LOGP)


# --------------------------------------------------------------------------- #
# 21. Validation — invalid PNA
# --------------------------------------------------------------------------- #
def test_validation_pna_negative():
    with pytest.raises(ValueError, match="pna_days"):
        scale_neonatal_pk("D", ADULT_CL, ADULT_VD, ADULT_WT, NEO_WT, 30.0, -1.0, FU, LOGP)


def test_validation_pna_too_high():
    with pytest.raises(ValueError, match="pna_days"):
        scale_neonatal_pk("D", ADULT_CL, ADULT_VD, ADULT_WT, NEO_WT, 30.0, 29.0, FU, LOGP)


# --------------------------------------------------------------------------- #
# 22. Validation — fu_adult out of range
# --------------------------------------------------------------------------- #
def test_validation_fu_adult_zero():
    with pytest.raises(ValueError, match="fu_adult"):
        scale_neonatal_pk("D", ADULT_CL, ADULT_VD, ADULT_WT, NEO_WT, 30.0, 0.0, 0.0, LOGP)


def test_validation_fu_adult_gt_one():
    with pytest.raises(ValueError, match="fu_adult"):
        scale_neonatal_pk("D", ADULT_CL, ADULT_VD, ADULT_WT, NEO_WT, 30.0, 0.0, 1.1, LOGP)


# --------------------------------------------------------------------------- #
# 23. Validation — logp out of range
# --------------------------------------------------------------------------- #
def test_validation_logp_too_low():
    with pytest.raises(ValueError, match="logp"):
        scale_neonatal_pk("D", ADULT_CL, ADULT_VD, ADULT_WT, NEO_WT, 30.0, 0.0, FU, -6.0)


def test_validation_logp_too_high():
    with pytest.raises(ValueError, match="logp"):
        scale_neonatal_pk("D", ADULT_CL, ADULT_VD, ADULT_WT, NEO_WT, 30.0, 0.0, FU, 11.0)
