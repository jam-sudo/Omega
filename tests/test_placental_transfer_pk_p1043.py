"""Tests for Drug Placental Transfer PK model (Phase 1043)."""

import pytest

from omega_pbpk.core.placental_transfer_pk_p1043 import (
    PlacentalTransferResult,
    simulate_placental_transfer,
)

_VALID_EXPOSURE_CATEGORIES = {"minimal", "low", "moderate", "high"}
_VALID_TRANSPORTER_EFFECTS = {"P-gp efflux", "passive", "bidirectional"}


def _default_result(**kwargs) -> PlacentalTransferResult:
    params = dict(
        drug_name="Warfarin",
        dose_mg=10.0,
        gestational_age_weeks=36,
        mw_Da=308.0,
        logp=2.7,
        fu_maternal=0.01,
        cl_maternal_L_per_h=5.0,
        vd_maternal_L=30.0,
        pgp_substrate=False,
        route="iv",
        t_end_h=24.0,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_placental_transfer(**params)


class TestReturnType:
    def test_returns_placental_transfer_result(self):
        result = _default_result()
        assert isinstance(result, PlacentalTransferResult)

    def test_drug_name_preserved(self):
        result = _default_result(drug_name="Aspirin")
        assert result.drug_name == "Aspirin"

    def test_dose_mg_preserved(self):
        result = _default_result(dose_mg=50.0)
        assert result.dose_mg == pytest.approx(50.0)

    def test_gestational_age_preserved(self):
        result = _default_result(gestational_age_weeks=28)
        assert result.gestational_age_weeks == 28


class TestInitialConditions:
    def test_c_maternal_positive_at_t0_for_iv(self):
        result = _default_result(route="iv")
        assert result.c_maternal_mg_L[0] > 0.0

    def test_c_umbilical_zero_at_t0(self):
        result = _default_result(route="iv")
        assert result.c_umbilical_mg_L[0] == pytest.approx(0.0)

    def test_c_amniotic_zero_at_t0(self):
        result = _default_result(route="iv")
        assert result.c_amniotic_mg_L[0] == pytest.approx(0.0)

    def test_c_maternal_zero_at_t0_for_oral(self):
        result = _default_result(route="oral", f_oral=0.8)
        assert result.c_maternal_mg_L[0] == pytest.approx(0.0)


class TestAUC:
    def test_auc_maternal_positive(self):
        result = _default_result()
        assert result.auc_maternal > 0.0

    def test_auc_fetal_positive(self):
        result = _default_result()
        assert result.auc_fetal > 0.0

    def test_fetal_to_maternal_ratio_positive(self):
        result = _default_result()
        assert result.fetal_to_maternal_ratio > 0.0

    def test_fetal_to_maternal_ratio_less_than_one(self):
        result = _default_result()
        assert result.fetal_to_maternal_ratio < 1.0


class TestPgpEffect:
    def test_pgp_substrate_lower_fetal_ratio(self):
        """P-gp efflux should reduce fetal exposure relative to non-substrate."""
        result_no_pgp = _default_result(
            pgp_substrate=False,
            fu_maternal=0.5,
            mw_Da=300.0,
            logp=3.0,
        )
        result_pgp = _default_result(
            pgp_substrate=True,
            fu_maternal=0.5,
            mw_Da=300.0,
            logp=3.0,
        )
        assert result_pgp.fetal_to_maternal_ratio < result_no_pgp.fetal_to_maternal_ratio

    def test_pgp_substrate_transporter_effect(self):
        result = _default_result(pgp_substrate=True)
        assert result.transporter_effect == "P-gp efflux"


class TestGestationalAge:
    def test_higher_ga_higher_auc_fetal(self):
        """More blood flow at higher GA → more fetal drug exposure."""
        result_early = _default_result(
            gestational_age_weeks=12,
            fu_maternal=0.5,
            mw_Da=200.0,
            logp=3.0,
        )
        result_late = _default_result(
            gestational_age_weeks=40,
            fu_maternal=0.5,
            mw_Da=200.0,
            logp=3.0,
        )
        assert result_late.auc_fetal > result_early.auc_fetal

    def test_all_valid_ga_run(self):
        for ga in [12, 20, 28, 36, 40]:
            result = _default_result(gestational_age_weeks=ga)
            assert isinstance(result, PlacentalTransferResult)


class TestExposureCategories:
    def test_fetal_exposure_category_valid(self):
        result = _default_result()
        assert result.fetal_exposure_category in _VALID_EXPOSURE_CATEGORIES

    def test_transporter_effect_valid(self):
        result = _default_result()
        assert result.transporter_effect in _VALID_TRANSPORTER_EFFECTS

    def test_low_permeability_gives_passive(self):
        # Very high MW and low logP → perm < 0.01 → "passive"
        result = _default_result(
            mw_Da=900.0,
            logp=-2.0,
            pgp_substrate=False,
        )
        assert result.transporter_effect == "passive"

    def test_high_logp_gives_bidirectional(self):
        result = _default_result(
            mw_Da=200.0,
            logp=5.0,
            pgp_substrate=False,
        )
        assert result.transporter_effect == "bidirectional"


class TestAmniotic:
    def test_amniotic_peak_positive(self):
        result = _default_result()
        assert max(result.c_amniotic_mg_L) > 0.0

    def test_amniotic_zero_at_t0(self):
        result = _default_result()
        assert result.c_amniotic_mg_L[0] == pytest.approx(0.0)


class TestPlacentalClearance:
    def test_placental_clearance_positive(self):
        result = _default_result()
        assert result.placental_clearance_L_h > 0.0

    def test_notes_nonempty(self):
        result = _default_result()
        assert isinstance(result.notes, str) and len(result.notes) > 0


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_placental_transfer(drug_name="X", dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_placental_transfer(drug_name="X", dose_mg=-5.0)

    def test_invalid_ga_raises(self):
        with pytest.raises(ValueError, match="gestational_age_weeks"):
            simulate_placental_transfer(drug_name="X", dose_mg=10.0, gestational_age_weeks=15)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_placental_transfer(drug_name="X", dose_mg=10.0, route="inhaled")

    def test_fu_maternal_zero_raises(self):
        with pytest.raises(ValueError, match="fu_maternal"):
            simulate_placental_transfer(drug_name="X", dose_mg=10.0, fu_maternal=0.0)

    def test_cl_maternal_zero_raises(self):
        with pytest.raises(ValueError, match="cl_maternal_L_per_h"):
            simulate_placental_transfer(drug_name="X", dose_mg=10.0, cl_maternal_L_per_h=0.0)

    def test_vd_maternal_zero_raises(self):
        with pytest.raises(ValueError, match="vd_maternal_L"):
            simulate_placental_transfer(drug_name="X", dose_mg=10.0, vd_maternal_L=0.0)
