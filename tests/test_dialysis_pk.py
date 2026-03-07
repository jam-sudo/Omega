"""Tests for Phase 220 — Dialysis / Haemodialysis PK."""

import pytest

from omega_pbpk.clinical.dialysis_pk import (
    DialysisPKResult,
    dialysis_extraction,
    simulate_dialysis_pk,
    supplemental_dose,
)

# ---------------------------------------------------------------------------
# dialysis_extraction
# ---------------------------------------------------------------------------


class TestDialysisExtraction:
    def test_high_fup_high_extraction(self):
        E = dialysis_extraction(fup=1.0, sieving_coeff=1.0)
        assert E == pytest.approx(0.95)  # capped at 0.95

    def test_low_fup_low_extraction(self):
        # Highly protein-bound drug: fup=0.01 → E very small
        E = dialysis_extraction(fup=0.01, sieving_coeff=0.8)
        assert E < 0.02

    def test_extraction_cap_at_0_95(self):
        E = dialysis_extraction(fup=1.0, sieving_coeff=1.0, qb_L_per_h=30.0)
        assert E <= 0.95

    def test_extraction_in_0_1(self):
        E = dialysis_extraction(fup=0.5, sieving_coeff=0.8)
        assert 0.0 <= E <= 1.0

    def test_invalid_fup_zero_raises(self):
        with pytest.raises(ValueError):
            dialysis_extraction(fup=0.0, sieving_coeff=0.8)

    def test_invalid_fup_above_1_raises(self):
        with pytest.raises(ValueError):
            dialysis_extraction(fup=1.1, sieving_coeff=0.8)

    def test_invalid_sieving_below_0_raises(self):
        with pytest.raises(ValueError):
            dialysis_extraction(fup=0.5, sieving_coeff=-0.1)

    def test_invalid_qb_raises(self):
        with pytest.raises(ValueError):
            dialysis_extraction(fup=0.5, sieving_coeff=0.8, qb_L_per_h=0.0)


# ---------------------------------------------------------------------------
# simulate_dialysis_pk
# ---------------------------------------------------------------------------


class TestSimulateDialysisPk:
    def _run(self, **kwargs):
        defaults = dict(
            drug_name="testdrug",
            dose_mg=500.0,
            cl_residual_L_per_h=0.5,
            vd_L=40.0,
            fup=0.5,
            sieving_coeff=0.8,
            qb_L_per_h=18.0,
            session_duration_h=4.0,
            interdialytic_interval_h=48.0,
            n_sessions=3,
        )
        defaults.update(kwargs)
        return simulate_dialysis_pk(**defaults)

    def test_returns_result_type(self):
        r = self._run()
        assert isinstance(r, DialysisPKResult)

    def test_concentration_drops_during_dialysis(self):
        r = self._run(n_sessions=1)
        # Identify a dialysis period
        dial_times = [i for i, d in enumerate(r.is_dialysis_time) if d]
        assert len(dial_times) > 0
        # Concentration at start of dialysis should be higher than at end
        c_start = r.c_plasma_mg_L[dial_times[0]]
        c_end = r.c_plasma_mg_L[dial_times[-1]]
        assert c_end < c_start

    def test_is_dialysis_time_has_true(self):
        r = self._run()
        assert any(r.is_dialysis_time)

    def test_fraction_removed_in_0_1(self):
        r = self._run()
        assert 0.0 <= r.fraction_removed_per_session <= 1.0

    def test_higher_fup_more_removed(self):
        r_high = self._run(fup=0.9, n_sessions=1)
        r_low = self._run(fup=0.05, n_sessions=1)
        assert r_high.fraction_removed_per_session > r_low.fraction_removed_per_session

    def test_cmax_equals_initial_concentration(self):
        r = self._run(dose_mg=500.0, vd_L=50.0)
        expected_c0 = 500.0 / 50.0
        assert abs(r.cmax_mg_L - expected_c0) / expected_c0 < 0.05

    def test_supplemental_dose_positive(self):
        r = self._run(fup=0.8)
        assert r.supplemental_dose_mg > 0.0

    def test_supplemental_dose_low_for_protein_bound(self):
        r_high_bound = self._run(fup=0.01)
        r_free = self._run(fup=0.9)
        assert r_high_bound.supplemental_dose_mg < r_free.supplemental_dose_mg

    def test_field_types(self):
        r = self._run(n_sessions=1)
        assert isinstance(r.times_h, list)
        assert isinstance(r.c_plasma_mg_L, list)
        assert isinstance(r.is_dialysis_time, list)
        assert isinstance(r.cmax_mg_L, float)
        assert isinstance(r.fraction_removed_per_session, float)
        assert isinstance(r.cl_dialysis_L_per_h, float)

    def test_list_lengths_match(self):
        r = self._run()
        assert len(r.times_h) == len(r.c_plasma_mg_L) == len(r.is_dialysis_time)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            self._run(dose_mg=0.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError):
            self._run(vd_L=0.0)

    def test_invalid_cl_residual_raises(self):
        with pytest.raises(ValueError):
            self._run(cl_residual_L_per_h=-1.0)


# ---------------------------------------------------------------------------
# supplemental_dose
# ---------------------------------------------------------------------------


class TestSupplementalDose:
    def _run(self, **kwargs):
        defaults = dict(
            drug_name="drug",
            dose_mg=500.0,
            fup=0.5,
            sieving_coeff=0.8,
            qb_L_per_h=18.0,
            session_duration_h=4.0,
            cl_L_per_h=1.0,
            vd_L=40.0,
        )
        defaults.update(kwargs)
        return supplemental_dose(**defaults)

    def test_positive_for_dialyzable_drug(self):
        s = self._run(fup=0.8)
        assert s > 0.0

    def test_low_for_protein_bound(self):
        s_free = self._run(fup=0.9)
        s_bound = self._run(fup=0.01)
        assert s_bound < s_free

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            self._run(dose_mg=-1.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError):
            self._run(vd_L=0.0)

    def test_returns_float(self):
        s = self._run()
        assert isinstance(s, float)
