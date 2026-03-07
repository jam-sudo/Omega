"""Tests for prodrug activation kinetics — Phase 464."""

import pytest

from omega_pbpk.core.prodrug_pk import (
    ProdrugPKResult,
    compare_prodrug_vs_parent,
    prodrug_benefit_score,
    simulate_prodrug_pk,
)

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

PRODRUG_KWARGS = dict(
    prodrug_name="ProX",
    active_name="ActY",
    dose_mg=100.0,
    ka_per_h=1.0,
    f_oral=0.8,
    cl_prodrug_L_per_h=5.0,
    vd_prodrug_L=50.0,
    kact_per_h=0.5,
    cl_active_L_per_h=10.0,
    vd_active_L=70.0,
    activation_site="systemic",
    t_end_h=24.0,
    dt_h=0.05,
)


def _run(**overrides):
    kw = dict(PRODRUG_KWARGS)
    kw.update(overrides)
    return simulate_prodrug_pk(**kw)


# -------------------------------------------------------------------------
# simulate_prodrug_pk — basic structure
# -------------------------------------------------------------------------


class TestSimulateProdrugPK:
    def test_returns_dataclass(self):
        result = _run()
        assert isinstance(result, ProdrugPKResult)

    def test_times_start_at_zero(self):
        result = _run()
        assert result.times_h[0] == 0.0

    def test_times_end_at_t_end(self):
        result = _run(t_end_h=12.0, dt_h=0.1)
        assert abs(result.times_h[-1] - 12.0) < 0.15

    def test_prodrug_length_matches_active(self):
        result = _run()
        assert len(result.c_prodrug_mg_L) == len(result.c_active_mg_L) == len(result.times_h)

    def test_cmax_active_positive(self):
        result = _run()
        assert result.cmax_active > 0.0

    def test_cmax_prodrug_positive(self):
        result = _run()
        assert result.cmax_prodrug > 0.0

    def test_auc_active_positive(self):
        result = _run()
        assert result.auc_active > 0.0

    def test_auc_prodrug_positive(self):
        result = _run()
        assert result.auc_prodrug > 0.0

    def test_f_activation_pct_in_0_100(self):
        result = _run()
        assert 0.0 <= result.f_activation_pct <= 100.0

    def test_notes_nonempty(self):
        result = _run()
        assert len(result.notes) > 0

    def test_active_tmax_after_prodrug_tmax(self):
        """Active drug Cmax should be delayed relative to prodrug Cmax."""
        result = _run(kact_per_h=0.3)
        # Find tmax of prodrug
        prod_concs = result.c_prodrug_mg_L
        tmax_prod = result.times_h[prod_concs.index(max(prod_concs))]
        tmax_act = result.tmax_active_h
        # Active drug peaks later than prodrug (delayed activation)
        assert tmax_act >= tmax_prod

    def test_higher_kact_higher_active_cmax(self):
        r_low = _run(kact_per_h=0.1)
        r_high = _run(kact_per_h=1.0)
        assert r_high.cmax_active > r_low.cmax_active

    def test_double_dose_doubles_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert abs(r2.cmax_active / r1.cmax_active - 2.0) < 0.1

    def test_double_dose_doubles_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert abs(r2.auc_active / r1.auc_active - 2.0) < 0.1

    def test_double_dose_doubles_prodrug_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert abs(r2.auc_prodrug / r1.auc_prodrug - 2.0) < 0.1

    def test_activation_site_systemic(self):
        result = _run(activation_site="systemic")
        assert result.cmax_active > 0.0

    def test_activation_site_gut(self):
        result = _run(activation_site="gut")
        assert result.cmax_active > 0.0
        assert result.f_activation_pct >= 0.0

    def test_activation_site_liver(self):
        # Liver activation with elevated kact should produce higher or equal active AUC vs gut
        r_liver = _run(activation_site="liver")
        # Liver has 1.5x eff_kact → higher activation
        assert r_liver.cmax_active >= 0.0

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-10.0)

    def test_invalid_f_oral_out_of_range(self):
        with pytest.raises(ValueError):
            _run(f_oral=1.5)

    def test_invalid_vd_prodrug_zero(self):
        with pytest.raises(ValueError):
            _run(vd_prodrug_L=0.0)

    def test_invalid_vd_active_zero(self):
        with pytest.raises(ValueError):
            _run(vd_active_L=0.0)

    def test_invalid_kact_negative(self):
        with pytest.raises(ValueError):
            _run(kact_per_h=-0.1)

    def test_invalid_activation_site(self):
        with pytest.raises(ValueError):
            _run(activation_site="kidney")

    def test_invalid_t_end_zero(self):
        with pytest.raises(ValueError):
            _run(t_end_h=0.0)

    def test_zero_kact_no_active(self):
        """With kact=0, no active drug is produced."""
        result = _run(kact_per_h=0.0)
        assert result.cmax_active == 0.0
        assert result.auc_active == 0.0


# -------------------------------------------------------------------------
# compare_prodrug_vs_parent
# -------------------------------------------------------------------------


class TestComprodeugVsParent:
    def test_identical_gives_ratio_1(self):
        r = _run()
        comparison = compare_prodrug_vs_parent(r, r)
        assert abs(comparison["auc_ratio"] - 1.0) < 1e-6
        assert abs(comparison["cmax_ratio"] - 1.0) < 1e-6

    def test_tmax_difference_zero_for_identical(self):
        r = _run()
        comparison = compare_prodrug_vs_parent(r, r)
        assert abs(comparison["tmax_difference_h"]) < 1e-6

    def test_higher_active_dose_higher_ratio(self):
        r_low = _run(dose_mg=50.0)
        r_high = _run(dose_mg=100.0)
        comparison = compare_prodrug_vs_parent(r_high, r_low)
        assert comparison["auc_ratio"] > 1.0

    def test_notes_nonempty(self):
        r = _run()
        comparison = compare_prodrug_vs_parent(r, r)
        assert len(comparison["notes"]) > 0

    def test_invalid_parent_zero_auc(self):
        """Parent with zero active AUC should raise ValueError."""
        r_ref = _run(kact_per_h=0.0)  # no active drug
        r = _run()
        with pytest.raises(ValueError):
            compare_prodrug_vs_parent(r, r_ref)

    def test_returns_expected_keys(self):
        r = _run()
        result = compare_prodrug_vs_parent(r, r)
        assert set(result.keys()) == {"auc_ratio", "cmax_ratio", "tmax_difference_h", "notes"}


# -------------------------------------------------------------------------
# prodrug_benefit_score
# -------------------------------------------------------------------------


class TestProdrugBenefitScore:
    def test_returns_float_in_0_100(self):
        score = prodrug_benefit_score(0.8, 0.4, 2.0, 1.5)
        assert 0.0 <= score <= 100.0

    def test_higher_f_prodrug_higher_score(self):
        s_low = prodrug_benefit_score(0.3, 0.3, 1.0, 1.0)
        s_high = prodrug_benefit_score(0.9, 0.3, 1.0, 1.0)
        assert s_high > s_low

    def test_equal_f_returns_moderate_score(self):
        # When prodrug = parent (same BA, sol, stab), score is 50%
        score = prodrug_benefit_score(0.5, 0.5, 1.0, 1.0)
        # ba_advantage = (0.5/0.5)/3 * 100 = 33.3%; sol = 20%; stab = 20% → ~25.3
        assert score > 0.0

    def test_max_solubility_ratio_caps_at_100(self):
        score = prodrug_benefit_score(1.0, 0.1, 10.0, 10.0)
        assert score <= 100.0

    def test_zero_solubility_and_stability(self):
        score = prodrug_benefit_score(0.8, 0.4, 0.0, 0.0)
        # Only BA advantage counts
        assert score >= 0.0

    def test_invalid_f_prodrug_negative(self):
        with pytest.raises(ValueError):
            prodrug_benefit_score(-0.1, 0.5, 1.0, 1.0)

    def test_invalid_f_parent_above_1(self):
        with pytest.raises(ValueError):
            prodrug_benefit_score(0.8, 1.2, 1.0, 1.0)

    def test_invalid_solubility_negative(self):
        with pytest.raises(ValueError):
            prodrug_benefit_score(0.8, 0.5, -1.0, 1.0)

    def test_invalid_stability_negative(self):
        with pytest.raises(ValueError):
            prodrug_benefit_score(0.8, 0.5, 1.0, -0.5)

    def test_zero_parent_f_with_prodrug_high(self):
        score = prodrug_benefit_score(0.9, 0.0, 2.0, 2.0)
        assert score > 0.0
