"""Tests for pk_parameter_estimator — Phase 500."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.prediction.pk_parameter_estimator import (
    PKEstimateResult,
    estimate_cl_human,
    estimate_full_pk,
    estimate_oral_bioavailability,
    estimate_vd_human,
)


# ---------------------------------------------------------------------------
# estimate_cl_human
# ---------------------------------------------------------------------------

class TestEstimateClHuman:
    def test_returns_float(self):
        cl = estimate_cl_human(2.0, 300.0, 0.1, 60.0, 2)
        assert isinstance(cl, float)

    def test_within_bounds(self):
        cl = estimate_cl_human(2.0, 300.0, 0.1, 60.0, 2)
        assert 0.1 <= cl <= 200.0

    def test_high_logP_lower_cl(self):
        cl_low = estimate_cl_human(1.0, 300.0, 0.1, 60.0, 2)
        cl_high = estimate_cl_human(5.0, 300.0, 0.1, 60.0, 2)
        assert cl_high < cl_low

    def test_high_psa_higher_cl(self):
        cl_low_psa = estimate_cl_human(2.0, 300.0, 0.1, 40.0, 2)
        cl_high_psa = estimate_cl_human(2.0, 300.0, 0.1, 120.0, 2)
        assert cl_high_psa > cl_low_psa

    def test_invalid_mw_zero(self):
        with pytest.raises(ValueError, match="mw"):
            estimate_cl_human(2.0, 0.0, 0.1, 60.0, 2)

    def test_invalid_mw_negative(self):
        with pytest.raises(ValueError, match="mw"):
            estimate_cl_human(2.0, -100.0, 0.1, 60.0, 2)

    def test_invalid_psa_negative(self):
        with pytest.raises(ValueError, match="psa"):
            estimate_cl_human(2.0, 300.0, 0.1, -10.0, 2)

    def test_invalid_fu_over_1(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            estimate_cl_human(2.0, 300.0, 1.5, 60.0, 2)

    def test_clamps_to_min(self):
        # Very high logP and low fu should push toward minimum
        cl = estimate_cl_human(10.0, 300.0, 0.001, 10.0, 1)
        assert cl >= 0.1

    def test_clamps_to_max(self):
        cl = estimate_cl_human(-10.0, 300.0, 0.99, 200.0, 1)
        assert cl <= 200.0

    def test_logD_74_accepted(self):
        # Should not raise even when logD_74 is provided
        cl = estimate_cl_human(2.0, 300.0, 0.1, 60.0, 2, logD_74=1.5)
        assert cl > 0.0


# ---------------------------------------------------------------------------
# estimate_vd_human
# ---------------------------------------------------------------------------

class TestEstimateVdHuman:
    def test_returns_float(self):
        vd = estimate_vd_human(2.0, 300.0, 0.1, 60.0, 2)
        assert isinstance(vd, float)

    def test_within_bounds(self):
        vd = estimate_vd_human(2.0, 300.0, 0.1, 60.0, 2)
        assert 1.0 <= vd <= 1000.0

    def test_high_logP_higher_vd(self):
        vd_low = estimate_vd_human(1.0, 300.0, 0.1, 60.0, 2)
        vd_high = estimate_vd_human(5.0, 300.0, 0.1, 60.0, 2)
        assert vd_high > vd_low

    def test_high_psa_lower_vd(self):
        vd_low_psa = estimate_vd_human(2.0, 300.0, 0.1, 20.0, 2)
        vd_high_psa = estimate_vd_human(2.0, 300.0, 0.1, 150.0, 2)
        assert vd_high_psa < vd_low_psa

    def test_pka_accepted(self):
        vd = estimate_vd_human(2.0, 300.0, 0.1, 60.0, 2, pKa=8.5)
        assert vd > 0.0

    def test_invalid_mw(self):
        with pytest.raises(ValueError):
            estimate_vd_human(2.0, 0.0, 0.1, 60.0, 2)

    def test_invalid_fu_negative(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            estimate_vd_human(2.0, 300.0, -0.1, 60.0, 2)


# ---------------------------------------------------------------------------
# estimate_oral_bioavailability
# ---------------------------------------------------------------------------

class TestEstimateOralBioavailability:
    def test_returns_float(self):
        f = estimate_oral_bioavailability(2.0, 300.0, 60.0, 2)
        assert isinstance(f, float)

    def test_within_bounds(self):
        f = estimate_oral_bioavailability(2.0, 300.0, 60.0, 2)
        assert 0.01 <= f <= 0.95

    def test_high_psa_lower_f(self):
        f_low = estimate_oral_bioavailability(2.0, 300.0, 40.0, 2)
        f_high = estimate_oral_bioavailability(2.0, 300.0, 150.0, 2)
        assert f_high < f_low

    def test_high_mw_penalty(self):
        f_small = estimate_oral_bioavailability(2.0, 300.0, 60.0, 2)
        f_large = estimate_oral_bioavailability(2.0, 600.0, 60.0, 2)
        assert f_large < f_small

    def test_high_logP_lower_f(self):
        f_low_logP = estimate_oral_bioavailability(1.0, 300.0, 60.0, 2)
        f_high_logP = estimate_oral_bioavailability(5.0, 300.0, 60.0, 2)
        assert f_high_logP < f_low_logP

    def test_invalid_mw(self):
        with pytest.raises(ValueError):
            estimate_oral_bioavailability(2.0, 0.0, 60.0, 2)


# ---------------------------------------------------------------------------
# estimate_full_pk
# ---------------------------------------------------------------------------

class TestEstimateFullPk:
    def setup_method(self):
        self.base_kwargs = dict(
            compound_name="TestDrug",
            logP=2.0,
            mw=350.0,
            psa_A2=70.0,
            n_hbd=2,
            fu_plasma=0.1,
        )

    def test_returns_pkeestimateresult(self):
        result = estimate_full_pk(**self.base_kwargs)
        assert isinstance(result, PKEstimateResult)

    def test_t_half_formula(self):
        result = estimate_full_pk(**self.base_kwargs)
        expected_t_half = math.log(2.0) * result.vd_L / result.cl_L_per_h
        assert result.t_half_h == pytest.approx(expected_t_half, rel=1e-6)

    def test_mrt_formula(self):
        result = estimate_full_pk(**self.base_kwargs)
        expected_mrt = result.vd_L / result.cl_L_per_h
        assert result.mrt_h == pytest.approx(expected_mrt, rel=1e-6)

    def test_cl_within_bounds(self):
        result = estimate_full_pk(**self.base_kwargs)
        assert 0.1 <= result.cl_L_per_h <= 200.0

    def test_vd_within_bounds(self):
        result = estimate_full_pk(**self.base_kwargs)
        assert 1.0 <= result.vd_L <= 1000.0

    def test_f_oral_within_bounds(self):
        result = estimate_full_pk(**self.base_kwargs)
        assert 0.01 <= result.f_oral <= 0.95

    def test_lipinski_compliant_true(self):
        result = estimate_full_pk(**self.base_kwargs)
        assert result.lipinski_compliant is True

    def test_lipinski_noncompliant_high_mw(self):
        result = estimate_full_pk(
            compound_name="BigDrug",
            logP=2.0,
            mw=600.0,
            psa_A2=70.0,
            n_hbd=2,
        )
        assert result.lipinski_compliant is False

    def test_lipinski_noncompliant_high_logP(self):
        result = estimate_full_pk(
            compound_name="LipophilicDrug",
            logP=6.0,
            mw=350.0,
            psa_A2=70.0,
            n_hbd=2,
        )
        assert result.lipinski_compliant is False

    def test_bcs_class_i(self):
        # High F + high solubility
        result = estimate_full_pk(
            compound_name="Drug_BCS_I",
            logP=1.0,
            mw=200.0,
            psa_A2=40.0,
            n_hbd=1,
            fu_plasma=0.5,
            solubility_mg_mL=5.0,
        )
        assert result.bcs_class == "BCS I"

    def test_bcs_class_iv_low_everything(self):
        # Low F + low solubility
        result = estimate_full_pk(
            compound_name="Drug_BCS_IV",
            logP=5.0,
            mw=600.0,
            psa_A2=200.0,
            n_hbd=5,
            fu_plasma=0.01,
            solubility_mg_mL=0.001,
        )
        assert result.bcs_class == "BCS IV"

    def test_confidence_high(self):
        result = estimate_full_pk(
            compound_name="HighConf",
            logP=2.0,
            mw=300.0,
            psa_A2=60.0,
            n_hbd=2,
        )
        assert result.confidence == "high"

    def test_confidence_medium_high_mw(self):
        result = estimate_full_pk(
            compound_name="MedConf",
            logP=2.0,
            mw=550.0,  # one violation: MW >= 500
            psa_A2=60.0,
            n_hbd=2,
        )
        assert result.confidence == "medium"

    def test_confidence_low_two_violations(self):
        result = estimate_full_pk(
            compound_name="LowConf",
            logP=5.0,    # violation: logP > 4
            mw=600.0,    # violation: MW >= 500
            psa_A2=60.0,
            n_hbd=2,
        )
        assert result.confidence == "low"

    def test_invalid_mw(self):
        with pytest.raises(ValueError, match="mw"):
            estimate_full_pk(
                compound_name="Bad",
                logP=2.0,
                mw=-1.0,
                psa_A2=60.0,
                n_hbd=2,
            )

    def test_invalid_psa(self):
        with pytest.raises(ValueError, match="psa"):
            estimate_full_pk(
                compound_name="Bad",
                logP=2.0,
                mw=300.0,
                psa_A2=-5.0,
                n_hbd=2,
            )

    def test_invalid_fu_plasma(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            estimate_full_pk(
                compound_name="Bad",
                logP=2.0,
                mw=300.0,
                psa_A2=60.0,
                n_hbd=2,
                fu_plasma=1.5,
            )

    def test_compound_name_stored(self):
        result = estimate_full_pk(**self.base_kwargs)
        assert result.compound_name == "TestDrug"

    def test_descriptor_fields_stored(self):
        result = estimate_full_pk(**self.base_kwargs)
        assert result.logP == pytest.approx(2.0)
        assert result.mw == pytest.approx(350.0)
        assert result.psa_A2 == pytest.approx(70.0)

    def test_notes_nonempty(self):
        result = estimate_full_pk(**self.base_kwargs)
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_t_half_positive(self):
        result = estimate_full_pk(**self.base_kwargs)
        assert result.t_half_h > 0.0

    def test_mrt_positive(self):
        result = estimate_full_pk(**self.base_kwargs)
        assert result.mrt_h > 0.0
