"""Tests for Phase 669 — CYP Induction Lag."""

import pytest

from omega_pbpk.clinical.cyp_induction_lag_p669 import (
    CYPInductionLagResult,
    simulate_cyp_induction_lag,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    victim_name="Midazolam",
    victim_dose_mg=5.0,
    victim_cl_L_per_h=0.5,
    victim_vd_L=100.0,
    inducer_name="Rifampicin",
    inducer_dose_mg=600.0,
    enzyme_name="CYP3A4",
    emax_induction=5.0,
    ec50_induction_mg_L=10.0,
    t_treatment_h=168.0,
    t_washout_h=240.0,
)


def _run(**overrides):
    kw = {**DEFAULT_KWARGS, **overrides}
    return simulate_cyp_induction_lag(**kw)


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_return_type(self):
        r = _run()
        assert isinstance(r, CYPInductionLagResult)

    def test_list_lengths_equal(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.enzyme_activity) == n
        assert len(r.c_victim_mg_L) == n
        assert len(r.c_victim_no_induction_mg_L) == n
        assert len(r.induction_ratio) == n

    def test_times_monotonic(self):
        r = _run()
        for i in range(1, len(r.times_h)):
            assert r.times_h[i] > r.times_h[i - 1]

    def test_non_negative_concentrations(self):
        r = _run()
        assert all(c >= 0 for c in r.c_victim_mg_L)
        assert all(c >= 0 for c in r.c_victim_no_induction_mg_L)

    def test_non_negative_enzyme_activity(self):
        r = _run()
        assert all(e >= 0 for e in r.enzyme_activity)


# ---------------------------------------------------------------------------
# Induction dynamics
# ---------------------------------------------------------------------------


class TestInductionDynamics:
    def test_higher_emax_lower_aucr(self):
        r_low = _run(emax_induction=2.0)
        r_high = _run(emax_induction=8.0)
        assert r_high.aucr < r_low.aucr

    def test_aucr_less_than_1(self):
        r = _run()
        assert r.aucr < 1.0

    def test_t_onset_50pct_positive(self):
        r = _run()
        assert r.t_onset_50pct_h > 0

    def test_t_offset_50pct_positive(self):
        r = _run()
        assert r.t_offset_50pct_h > 0

    def test_max_fold_induction_ge_1(self):
        r = _run()
        assert r.max_fold_induction >= 1.0

    def test_enzyme_returns_toward_baseline_washout(self):
        r = _run()
        # Last enzyme value should be closer to 1.0 than the max
        max_e = max(r.enzyme_activity)
        last_e = r.enzyme_activity[-1]
        assert abs(last_e - 1.0) < abs(max_e - 1.0)

    def test_enzyme_at_end_pretreatment_near_baseline(self):
        r = _run(t_pretreatment_h=48.0)
        # Find enzyme at t = 48h (end of pretreatment)
        idx = None
        for i, t in enumerate(r.times_h):
            if abs(t - 48.0) < 0.5:
                idx = i
                break
        assert idx is not None
        assert abs(r.enzyme_activity[idx] - 1.0) < 0.05

    def test_induction_ratio_matches_enzyme(self):
        r = _run()
        for i in range(len(r.times_h)):
            assert abs(r.induction_ratio[i] - r.enzyme_activity[i]) < 1e-10

    def test_max_fold_induction_consistent(self):
        r = _run()
        assert abs(r.max_fold_induction - max(r.enzyme_activity)) < 1e-10


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_higher_inducer_dose_more_induction(self):
        r_low = _run(inducer_dose_mg=100.0)
        r_high = _run(inducer_dose_mg=1000.0)
        assert r_high.max_fold_induction > r_low.max_fold_induction

    def test_longer_treatment_higher_max(self):
        r_short = _run(t_treatment_h=48.0)
        r_long = _run(t_treatment_h=336.0)
        assert r_long.max_fold_induction >= r_short.max_fold_induction

    def test_no_induction_aucr_near_1(self):
        r = _run(emax_induction=0.0)
        assert abs(r.aucr - 1.0) < 0.05

    def test_zero_emax_enzyme_stays_at_1(self):
        r = _run(emax_induction=0.0)
        for e in r.enzyme_activity:
            assert abs(e - 1.0) < 0.05

    def test_t_max_induction_within_treatment(self):
        r = _run(t_pretreatment_h=0.0, t_treatment_h=168.0)
        # Max induction should occur during or at end of treatment
        assert r.t_max_induction_h <= 168.0 + 10.0  # small tolerance


# ---------------------------------------------------------------------------
# Victim drug PK
# ---------------------------------------------------------------------------


class TestVictimPK:
    def test_victim_concentration_decays(self):
        r = _run()
        # After absorption phase, concentrations should eventually decay
        max_c = max(r.c_victim_mg_L)
        assert max_c > 0
        assert r.c_victim_mg_L[-1] < max_c

    def test_noinduction_concentration_decays(self):
        r = _run()
        max_c = max(r.c_victim_no_induction_mg_L)
        assert max_c > 0
        assert r.c_victim_no_induction_mg_L[-1] < max_c

    def test_induction_lowers_peak(self):
        r = _run()
        # With induction, CL is higher so peak may be lower or similar
        # but AUC should be lower
        auc_induced = sum(
            0.5 * (r.c_victim_mg_L[i] + r.c_victim_mg_L[i - 1]) * (r.times_h[i] - r.times_h[i - 1])
            for i in range(1, len(r.times_h))
        )
        auc_ref = sum(
            0.5
            * (r.c_victim_no_induction_mg_L[i] + r.c_victim_no_induction_mg_L[i - 1])
            * (r.times_h[i] - r.times_h[i - 1])
            for i in range(1, len(r.times_h))
        )
        assert auc_induced < auc_ref


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_names_stored(self):
        r = _run()
        assert r.drug_name_victim == "Midazolam"
        assert r.inducer_name == "Rifampicin"
        assert r.enzyme_name == "CYP3A4"

    def test_doses_stored(self):
        r = _run()
        assert r.victim_dose_mg == 5.0
        assert r.inducer_dose_mg == 600.0

    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_cl_raises(self):
        with pytest.raises(ValueError):
            _run(victim_cl_L_per_h=-1.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            _run(victim_vd_L=0.0)

    def test_negative_emax_raises(self):
        with pytest.raises(ValueError):
            _run(emax_induction=-1.0)

    def test_zero_ec50_raises(self):
        with pytest.raises(ValueError):
            _run(ec50_induction_mg_L=0.0)

    def test_zero_kdeg_raises(self):
        with pytest.raises(ValueError):
            _run(kdeg_per_h=0.0)
