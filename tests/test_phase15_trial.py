"""Phase 15: Clinical trial simulation tests.

Tests cover:
  - DoseRegimen: properties, validation
  - Multi-dose superposition: steady-state > single-dose for accumulating drugs
  - SubjectPKResult: derived properties (AI, PTR)
  - ArmPKSummary: statistics from a synthetic cohort
  - BioequivalenceResult: BE window logic
  - compute_bioequivalence: Welch 90% CI, GMR
  - simulate_arm: returns valid results for small populations
  - run_clinical_trial: 1-arm and 2-arm parallel designs

For speed: small n_subjects (3-5) and a fast-clearing drug with short t½.
"""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.clinical.trial import (
    ArmPKSummary,
    BioequivalenceResult,
    DoseRegimen,
    SubjectPKResult,
    TrialArm,
    TrialResult,
    _extract_ss_pk,
    _superpose_doses,
    _terminal_half_life,
    _welch_90ci_ratio,
    compute_bioequivalence,
    run_clinical_trial,
    simulate_arm,
)
from omega_pbpk.drugs.drug import Drug

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fast_drug():
    """Fast-clearing oral drug suitable for short simulations."""
    return Drug(
        name="FastDrug",
        mw=300.0,
        logP=2.0,
        fup=0.30,
        rbp=1.0,
        pka=[8.0],
        clint_hepatic_L_per_h=25.0,  # high CLint → short t½
        clr_L_per_h=0.5,
    )


@pytest.fixture(scope="module")
def slow_drug():
    """Slowly cleared drug that accumulates over multiple doses."""
    return Drug(
        name="SlowDrug",
        mw=350.0,
        logP=2.5,
        fup=0.15,
        rbp=1.0,
        pka=[7.5],
        clint_hepatic_L_per_h=1.5,  # low CLint → longer t½, accumulation
        clr_L_per_h=0.1,
    )


@pytest.fixture(scope="module")
def regimen_qd():
    return DoseRegimen(dose_mg=100.0, n_doses=10, interval_h=24.0, route="oral")


@pytest.fixture(scope="module")
def regimen_bid():
    return DoseRegimen(dose_mg=50.0, n_doses=14, interval_h=12.0, route="oral")


# ============================================================================
# DoseRegimen
# ============================================================================


class TestDoseRegimen:
    def test_total_duration(self):
        r = DoseRegimen(dose_mg=100, n_doses=10, interval_h=24)
        assert r.total_duration_h == pytest.approx(240.0)

    def test_defaults(self):
        r = DoseRegimen(dose_mg=50)
        assert r.n_doses == 10
        assert r.interval_h == 24.0
        assert r.route == "oral"

    def test_is_frozen(self):
        r = DoseRegimen(dose_mg=100)
        with pytest.raises((AttributeError, TypeError)):
            r.dose_mg = 200  # type: ignore[misc]


# ============================================================================
# Internal helpers
# ============================================================================


class TestSuperpose:
    def test_output_shape(self):
        t = np.linspace(0, 24, 100)
        cp = np.exp(-0.1 * t)
        t_md, cp_md = _superpose_doses(cp, t, interval_h=24, n_doses=5)
        assert len(t_md) == len(cp_md)
        assert t_md[-1] >= 5 * 24

    def test_more_doses_more_area(self):
        """5 doses → more total AUC than 1 dose."""
        t = np.linspace(0, 24, 200)
        cp = np.exp(-0.15 * t)
        _, cp5 = _superpose_doses(cp, t, interval_h=24, n_doses=5)
        _, cp1 = _superpose_doses(cp, t, interval_h=24, n_doses=1)
        auc5 = np.trapezoid(cp5, _superpose_doses(cp, t, 24, 5)[0])
        auc1 = np.trapezoid(cp1, _superpose_doses(cp, t, 24, 1)[0])
        assert auc5 > auc1

    def test_monotone_convergence_to_ss(self):
        """Accumulating drug: Cmax at SS (n=10) > Cmax at first dose (n=1).

        t½=24h with 24h interval → AI≈2 → significant accumulation.
        Profile must span several intervals so early-dose tails are captured.
        """
        interval = 24.0
        # Single-dose profile must span at least 10 intervals so contributions
        # from early doses are not truncated when computing n=10 steady state.
        t = np.linspace(0, 10 * interval, 1000)
        cp = 1.0 * np.exp(-np.log(2) / interval * t)

        t_md1, cp_md1 = _superpose_doses(cp, t, interval_h=interval, n_doses=1)
        t_md10, cp_md10 = _superpose_doses(cp, t, interval_h=interval, n_doses=10)

        cmax_1 = float(np.max(cp_md1))
        # Cmax at SS: last interval of the 10-dose run
        mask_ss = t_md10 >= 9 * interval
        cmax_ss = float(np.max(cp_md10[mask_ss]))

        # For t½ = τ, accumulation index ≈ 2 → Cmax_ss ≈ 2 × Cmax_1
        assert cmax_ss > cmax_1 * 1.5

    def test_non_negative(self):
        t = np.linspace(0, 12, 100)
        cp = np.exp(-0.5 * t)
        _, cp_md = _superpose_doses(cp, t, interval_h=12, n_doses=7)
        assert np.all(cp_md >= 0)


class TestExtractSSPK:
    def test_returns_three_floats(self):
        t = np.linspace(0, 24, 200)
        cp = np.exp(-0.1 * t)
        t_md, cp_md = _superpose_doses(cp, t, 24, 5)
        cmax, cmin, auc = _extract_ss_pk(t_md, cp_md, 24, 5)
        assert isinstance(cmax, float)
        assert isinstance(cmin, float)
        assert isinstance(auc, float)

    def test_cmax_ge_cmin(self):
        t = np.linspace(0, 24, 200)
        cp = np.exp(-0.1 * t)
        t_md, cp_md = _superpose_doses(cp, t, 24, 5)
        cmax, cmin, auc = _extract_ss_pk(t_md, cp_md, 24, 5)
        assert cmax >= cmin

    def test_auc_positive(self):
        t = np.linspace(0, 24, 200)
        cp = 2.0 * np.exp(-0.2 * t)
        t_md, cp_md = _superpose_doses(cp, t, 24, 5)
        _, _, auc = _extract_ss_pk(t_md, cp_md, 24, 5)
        assert auc > 0


class TestTerminalHalfLife:
    def test_monoexponential(self):
        """Known t½ = 12 h should be recovered (within 20%)."""
        t = np.linspace(0, 72, 500)
        cp = np.exp(-np.log(2) / 12 * t)
        t_half = _terminal_half_life(t, cp)
        assert 9 <= t_half <= 16

    def test_returns_float(self):
        t = np.linspace(0, 24, 100)
        cp = np.exp(-0.1 * t)
        assert isinstance(_terminal_half_life(t, cp), float)

    def test_all_zero_returns_sentinel(self):
        t = np.linspace(0, 24, 100)
        cp = np.zeros(100)
        assert _terminal_half_life(t, cp) >= 99.0

    def test_positive_result(self):
        t = np.linspace(0, 48, 200)
        cp = np.exp(-0.05 * t)
        assert _terminal_half_life(t, cp) > 0


# ============================================================================
# SubjectPKResult
# ============================================================================


class TestSubjectPKResult:
    @pytest.fixture
    def pk(self):
        return SubjectPKResult(
            subject_id=0,
            body_weight_kg=70.0,
            cyp3a4_activity=1.0,
            cmax_ss=2.5,
            cmin_ss=0.5,
            auc_tau=30.0,
            cmax_1=1.0,
            auc_inf=20.0,
            t_half_h=8.0,
            tmax_h=1.5,
        )

    def test_accumulation_index(self, pk):
        assert pk.accumulation_index == pytest.approx(2.5, rel=1e-4)

    def test_peak_trough_ratio(self, pk):
        assert pk.peak_trough_ratio == pytest.approx(5.0, rel=1e-4)

    def test_unit_accumulation_when_equal(self):
        pk = SubjectPKResult(0, 70, 1.0, 1.0, 0.1, 5.0, 1.0, 5.0, 2.0, 0.5)
        assert pk.accumulation_index == pytest.approx(1.0, rel=1e-4)


# ============================================================================
# ArmPKSummary
# ============================================================================


class TestArmPKSummary:
    @pytest.fixture
    def subjects(self):
        return [
            SubjectPKResult(
                i,
                70.0,
                1.0,
                cmax_ss=1.0 + i * 0.1,
                cmin_ss=0.1 + i * 0.01,
                auc_tau=10.0 + i,
                cmax_1=0.8,
                auc_inf=8.0,
                t_half_h=6.0 + i * 0.5,
                tmax_h=1.0,
            )
            for i in range(10)
        ]

    def test_summary_n(self, subjects):
        s = ArmPKSummary.from_subjects("TestArm", subjects)
        assert s.n == 10

    def test_cmax_mean_in_range(self, subjects):
        s = ArmPKSummary.from_subjects("TestArm", subjects)
        assert 1.0 <= s.cmax_ss["mean"] <= 2.0

    def test_cv_pct_positive(self, subjects):
        s = ArmPKSummary.from_subjects("TestArm", subjects)
        assert s.cmax_ss["cv_pct"] >= 0.0

    def test_p5_le_median_le_p95(self, subjects):
        s = ArmPKSummary.from_subjects("TestArm", subjects)
        assert s.auc_tau["p5"] <= s.auc_tau["median"] <= s.auc_tau["p95"]

    def test_empty_subjects(self):
        s = ArmPKSummary.from_subjects("Empty", [])
        assert s.n == 0
        assert s.cmax_ss["mean"] == 0.0


# ============================================================================
# Bioequivalence
# ============================================================================


class TestWelchCI:
    def test_identical_arms_gmr_one(self):
        vals = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        gmr, lo, hi = _welch_90ci_ratio(vals, vals)
        assert gmr == pytest.approx(1.0, abs=0.01)

    def test_ci_contains_one_for_same_data(self):
        vals = np.array([2.0, 2.1, 1.9, 2.05, 1.95])
        gmr, lo, hi = _welch_90ci_ratio(vals, vals)
        assert lo <= 1.0 <= hi

    def test_test_twice_ref_gmr_two(self):
        ref = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        test = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        gmr, _, _ = _welch_90ci_ratio(test, ref)
        assert gmr == pytest.approx(2.0, abs=0.01)

    def test_ci_lower_le_gmr_le_ci_upper(self):
        rng = np.random.default_rng(0)
        a = rng.lognormal(0.1, 0.2, 20)
        b = rng.lognormal(0.0, 0.2, 20)
        gmr, lo, hi = _welch_90ci_ratio(a, b)
        assert lo <= gmr <= hi

    def test_returns_three_values(self):
        a = np.array([1.0, 1.1, 0.9])
        b = np.array([1.0, 1.0, 1.0])
        result = _welch_90ci_ratio(a, b)
        assert len(result) == 3


class TestBioequivalenceResult:
    def test_within_be_window(self):
        be = BioequivalenceResult(
            test_arm="T",
            reference_arm="R",
            auc_gmr=1.0,
            auc_ci_90_lower=0.85,
            auc_ci_90_upper=1.18,
            cmax_gmr=1.0,
            cmax_ci_90_lower=0.82,
            cmax_ci_90_upper=1.20,
        )
        assert be.is_bioequivalent is True
        assert be.auc_be is True
        assert be.cmax_be is True

    def test_outside_be_window_upper(self):
        be = BioequivalenceResult(
            test_arm="T",
            reference_arm="R",
            auc_gmr=1.3,
            auc_ci_90_lower=1.10,
            auc_ci_90_upper=1.50,
            cmax_gmr=1.3,
            cmax_ci_90_lower=1.10,
            cmax_ci_90_upper=1.50,
        )
        assert be.is_bioequivalent is False

    def test_outside_be_window_lower(self):
        be = BioequivalenceResult(
            test_arm="T",
            reference_arm="R",
            auc_gmr=0.70,
            auc_ci_90_lower=0.55,
            auc_ci_90_upper=0.82,
            cmax_gmr=0.70,
            cmax_ci_90_lower=0.55,
            cmax_ci_90_upper=0.82,
        )
        assert be.is_bioequivalent is False


class TestComputeBioequivalence:
    def _make_subjects(self, cmax_vals, auc_vals):
        return [
            SubjectPKResult(
                i,
                70.0,
                1.0,
                cmax_ss=c,
                cmin_ss=c * 0.2,
                auc_tau=a,
                cmax_1=c * 0.8,
                auc_inf=a * 1.5,
                t_half_h=8.0,
                tmax_h=1.0,
            )
            for i, (c, a) in enumerate(zip(cmax_vals, auc_vals))
        ]

    def test_returns_be_result(self):
        test = self._make_subjects([1.1] * 6, [11.0] * 6)
        ref = self._make_subjects([1.0] * 6, [10.0] * 6)
        be = compute_bioequivalence(test, ref)
        assert isinstance(be, BioequivalenceResult)

    def test_auc_gmr_positive(self):
        test = self._make_subjects([1.0] * 5, [10.0] * 5)
        ref = self._make_subjects([1.0] * 5, [10.0] * 5)
        be = compute_bioequivalence(test, ref)
        assert be.auc_gmr > 0.0

    def test_be_for_identical_arms(self):
        data = self._make_subjects(
            [1.0, 1.05, 0.95, 1.02, 0.98, 1.01], [10.0, 10.5, 9.5, 10.2, 9.8, 10.1]
        )
        be = compute_bioequivalence(data, data)
        assert be.is_bioequivalent is True

    def test_arm_names_set(self):
        test = self._make_subjects([1.0] * 4, [10.0] * 4)
        ref = self._make_subjects([1.0] * 4, [10.0] * 4)
        be = compute_bioequivalence(test, ref, "TestDrug", "PlaceboRef")
        assert be.test_arm == "TestDrug"
        assert be.reference_arm == "PlaceboRef"


# ============================================================================
# simulate_arm (integration — small n for speed)
# ============================================================================


class TestSimulateArm:
    def test_returns_list(self, fast_drug, regimen_qd):
        arm = TrialArm("Fast", fast_drug, regimen_qd)
        results = simulate_arm(arm, n_subjects=3, seed=0)
        assert isinstance(results, list)

    def test_positive_cmax(self, fast_drug, regimen_qd):
        arm = TrialArm("Fast", fast_drug, regimen_qd)
        results = simulate_arm(arm, n_subjects=3, seed=1)
        assert len(results) >= 1
        for r in results:
            assert r.cmax_ss > 0

    def test_positive_auc_tau(self, fast_drug, regimen_qd):
        arm = TrialArm("Fast", fast_drug, regimen_qd)
        results = simulate_arm(arm, n_subjects=3, seed=2)
        for r in results:
            assert r.auc_tau > 0

    def test_cmin_le_cmax(self, fast_drug, regimen_qd):
        arm = TrialArm("Fast", fast_drug, regimen_qd)
        results = simulate_arm(arm, n_subjects=3, seed=3)
        for r in results:
            assert r.cmin_ss <= r.cmax_ss

    def test_t_half_positive(self, fast_drug, regimen_qd):
        arm = TrialArm("Fast", fast_drug, regimen_qd)
        results = simulate_arm(arm, n_subjects=3, seed=4)
        for r in results:
            assert r.t_half_h > 0

    def test_subject_ids_assigned(self, fast_drug, regimen_qd):
        arm = TrialArm("Fast", fast_drug, regimen_qd)
        results = simulate_arm(arm, n_subjects=3, seed=5)
        ids = [r.subject_id for r in results]
        assert len(ids) == len(set(ids))  # unique IDs

    def test_iv_route(self, fast_drug):
        reg = DoseRegimen(dose_mg=10.0, n_doses=3, interval_h=8.0, route="iv")
        arm = TrialArm("IV", fast_drug, reg)
        results = simulate_arm(arm, n_subjects=2, seed=6)
        assert len(results) >= 1
        for r in results:
            assert r.cmax_ss > 0


# ============================================================================
# run_clinical_trial
# ============================================================================


class TestRunClinicalTrial:
    def test_single_arm_returns_trial_result(self, fast_drug, regimen_qd):
        arm = TrialArm("Mono", fast_drug, regimen_qd)
        result = run_clinical_trial([arm], n_subjects=3, seed=10)
        assert isinstance(result, TrialResult)

    def test_single_arm_no_be(self, fast_drug, regimen_qd):
        arm = TrialArm("Mono", fast_drug, regimen_qd)
        result = run_clinical_trial([arm], n_subjects=3, seed=11)
        assert result.bioequivalence is None

    def test_two_arms_has_be(self, fast_drug, regimen_qd, regimen_bid):
        arm_a = TrialArm("Test", fast_drug, regimen_qd)
        arm_b = TrialArm("Reference", fast_drug, regimen_bid)
        result = run_clinical_trial([arm_a, arm_b], n_subjects=3, seed=12)
        assert result.bioequivalence is not None
        assert isinstance(result.bioequivalence, BioequivalenceResult)

    def test_arm_summaries_count(self, fast_drug, regimen_qd, regimen_bid):
        arms = [
            TrialArm("A", fast_drug, regimen_qd),
            TrialArm("B", fast_drug, regimen_bid),
        ]
        result = run_clinical_trial(arms, n_subjects=3, seed=13)
        assert len(result.arm_summaries) == 2

    def test_design_keys(self, fast_drug, regimen_qd):
        arm = TrialArm("X", fast_drug, regimen_qd)
        result = run_clinical_trial([arm], n_subjects=3, seed=14)
        assert "arms" in result.design
        assert "n_subjects" in result.design
        assert "regimens" in result.design

    def test_summary_table_rows(self, fast_drug, regimen_qd, regimen_bid):
        arms = [TrialArm("A", fast_drug, regimen_qd), TrialArm("B", fast_drug, regimen_bid)]
        result = run_clinical_trial(arms, n_subjects=3, seed=15)
        table = result.summary_table()
        assert len(table) == 2
        for row in table:
            assert "Cmax_ss_mean_mg_L" in row
            assert "AUCtau_mean_mgh_L" in row

    def test_seed_reproducibility(self, fast_drug, regimen_qd):
        arm = TrialArm("Rep", fast_drug, regimen_qd)
        r1 = run_clinical_trial([arm], n_subjects=3, seed=99)
        r2 = run_clinical_trial([arm], n_subjects=3, seed=99)
        # Same seed → same Cmax means
        assert r1.arm_summaries[0].cmax_ss["mean"] == r2.arm_summaries[0].cmax_ss["mean"]

    def test_higher_dose_higher_exposure(self, fast_drug):
        low_dose = DoseRegimen(dose_mg=50, n_doses=5, interval_h=24)
        high_dose = DoseRegimen(dose_mg=200, n_doses=5, interval_h=24)
        arm_low = TrialArm("Low", fast_drug, low_dose)
        arm_high = TrialArm("High", fast_drug, high_dose)
        result = run_clinical_trial([arm_low, arm_high], n_subjects=3, seed=20)
        s_low = result.arm_summaries[0].auc_tau["mean"]
        s_high = result.arm_summaries[1].auc_tau["mean"]
        assert s_high > s_low

    def test_be_arms_with_same_drug_and_dose(self, fast_drug):
        """Identical arms should give GMR ≈ 1 and pass BE."""
        reg = DoseRegimen(dose_mg=100, n_doses=5, interval_h=24)
        arm_t = TrialArm("Test", fast_drug, reg)
        arm_r = TrialArm("Ref", fast_drug, reg)
        result = run_clinical_trial([arm_t, arm_r], n_subjects=5, seed=30)
        be = result.bioequivalence
        assert be is not None
        # GMR should be close to 1
        assert 0.7 <= be.auc_gmr <= 1.4
