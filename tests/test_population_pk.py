"""Tests for population PK fitting (Phase 54)."""

import math
import numpy as np
import pytest

from omega_pbpk.clinical.population_pk import (
    SubjectData,
    IndividualEstimate,
    PopulationPKResult,
    fit_population_pk,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

KE_TRUE = 0.1
KA_TRUE = 0.8
CL_TRUE = [4.0, 5.0, 6.0, 4.5, 5.5]
VD_TRUE = 40.0
DOSE = 100.0
TIMES = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]


def _oral_cp(t, cl, vd, ka, dose):
    ke = cl / vd
    if abs(ka - ke) < 1e-3:
        return dose / vd * ke * t * np.exp(-ke * t)
    return dose * ka / (vd * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


rng = np.random.default_rng(42)

ORAL_SUBJECTS = []
for i, cl in enumerate(CL_TRUE):
    cp = np.array([_oral_cp(t, cl, VD_TRUE, KA_TRUE, DOSE) for t in TIMES])
    noise = rng.normal(0, 0.1, size=len(TIMES))
    cp_obs = np.maximum(cp * (1 + noise), 0.0)
    ORAL_SUBJECTS.append(
        SubjectData(
            subject_id=f"S{i+1}",
            times=list(TIMES),
            conc=list(cp_obs),
            dose_mg=DOSE,
            route="oral",
            body_weight=60.0 + i * 5,
        )
    )

IV_SUBJECTS = []
for i, cl in enumerate(CL_TRUE):
    ke_i = cl / VD_TRUE
    cp = np.array([DOSE / VD_TRUE * np.exp(-ke_i * t) for t in TIMES])
    noise = rng.normal(0, 0.1, size=len(TIMES))
    cp_obs = np.maximum(cp * (1 + noise), 0.0)
    IV_SUBJECTS.append(
        SubjectData(
            subject_id=f"IV{i+1}",
            times=list(TIMES),
            conc=list(cp_obs),
            dose_mg=DOSE,
            route="iv",
            body_weight=60.0 + i * 5,
        )
    )


# ===================================================================
# Dataclass tests
# ===================================================================

class TestSubjectData:
    def test_dataclass_fields(self):
        s = ORAL_SUBJECTS[0]
        assert hasattr(s, "subject_id")
        assert hasattr(s, "times")
        assert hasattr(s, "conc")
        assert hasattr(s, "dose_mg")
        assert hasattr(s, "route")
        assert hasattr(s, "body_weight")

    def test_default_body_weight(self):
        s = SubjectData("x", [1.0], [1.0], 100.0, "oral")
        assert s.body_weight == 70.0


class TestIndividualEstimate:
    def test_dataclass_fields(self):
        e = IndividualEstimate(
            subject_id="test",
            cl_L_per_h=5.0,
            vd_L=40.0,
            ka_per_h=0.8,
            auc_pred=20.0,
            cmax_pred=2.5,
            rmse=0.1,
            converged=True,
        )
        assert e.subject_id == "test"
        assert e.cl_L_per_h == 5.0
        assert e.vd_L == 40.0
        assert e.ka_per_h == 0.8
        assert e.auc_pred == 20.0
        assert e.cmax_pred == 2.5
        assert e.rmse == 0.1
        assert e.converged is True


# ===================================================================
# Integration tests — oral fitting
# ===================================================================

@pytest.fixture(scope="module")
def oral_result():
    return fit_population_pk(ORAL_SUBJECTS)


@pytest.mark.integration
class TestFitOral:
    def test_returns_result_type(self, oral_result):
        assert isinstance(oral_result, PopulationPKResult)

    def test_n_subjects_correct(self, oral_result):
        assert oral_result.n_subjects == 5

    def test_cl_pop_in_range(self, oral_result):
        assert 3.0 <= oral_result.cl_pop_L_per_h <= 8.0

    def test_vd_pop_in_range(self, oral_result):
        assert 20.0 <= oral_result.vd_pop_L <= 100.0

    def test_ka_pop_positive(self, oral_result):
        assert oral_result.ka_pop_per_h > 0

    def test_omega_cl_nonneg(self, oral_result):
        assert oral_result.omega_cl_pct >= 0

    def test_omega_vd_nonneg(self, oral_result):
        assert oral_result.omega_vd_pct >= 0

    def test_individual_count(self, oral_result):
        assert len(oral_result.individual_estimates) == 5

    def test_all_converged(self, oral_result):
        for est in oral_result.individual_estimates:
            assert est.converged, f"{est.subject_id} did not converge"

    def test_individual_cl_positive(self, oral_result):
        for est in oral_result.individual_estimates:
            assert est.cl_L_per_h > 0, f"{est.subject_id} CL <= 0"

    def test_aic_finite(self, oral_result):
        assert math.isfinite(oral_result.aic)

    def test_bic_finite(self, oral_result):
        assert math.isfinite(oral_result.bic)

    def test_method_tag(self, oral_result):
        assert oral_result.method == "standard_two_stage"

    def test_no_covariate_default(self, oral_result):
        assert oral_result.bw_cl_exponent is None


# ===================================================================
# Integration tests — IV fitting
# ===================================================================

@pytest.fixture(scope="module")
def iv_result():
    return fit_population_pk(IV_SUBJECTS, model="1cpt_iv")


@pytest.mark.integration
class TestFitIV:
    def test_iv_ka_nan(self, iv_result):
        assert math.isnan(iv_result.ka_pop_per_h)

    def test_iv_omega_ka_nan(self, iv_result):
        assert math.isnan(iv_result.omega_ka_pct)

    def test_iv_cl_in_range(self, iv_result):
        assert 3.0 <= iv_result.cl_pop_L_per_h <= 8.0


# ===================================================================
# Covariate test
# ===================================================================

@pytest.fixture(scope="module")
def cov_result():
    return fit_population_pk(ORAL_SUBJECTS, fit_covariates=True)


@pytest.mark.integration
class TestCovariate:
    def test_covariate_exponent_not_none(self, cov_result):
        assert cov_result.bw_cl_exponent is not None

    def test_covariate_finite(self, cov_result):
        assert math.isfinite(cov_result.bw_cl_exponent)


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_sparse_subject_not_converged(self):
        sparse = SubjectData("sparse", [1.0, 2.0], [2.0, 1.5], 100.0, "oral")
        result = fit_population_pk(ORAL_SUBJECTS[:3] + [sparse])
        est = next(
            e for e in result.individual_estimates if e.subject_id == "sparse"
        )
        assert not est.converged
