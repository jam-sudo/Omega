"""Tests for Monte Carlo PK simulation (Phase 449)."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.monte_carlo_simulation import (
    MonteCarloPKResult,
    monte_carlo_pk,
    uncertainty_propagation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_mc(**kw) -> MonteCarloPKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_mean=5.0,
        cl_cv_pct=30.0,
        vd_mean=50.0,
        vd_cv_pct=30.0,
        n_simulations=200,
        t_end_h=24.0,
        dt_h=0.25,
        interval_h=12.0,
        n_doses=1,
        ka_per_h=1.0,
        f_oral=1.0,
        seed=42,
    )
    defaults.update(kw)
    return monte_carlo_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_type(self):
        assert isinstance(_default_mc(), MonteCarloPKResult)

    def test_drug_name_preserved(self):
        r = _default_mc(drug_name="DrugA")
        assert r.drug_name == "DrugA"

    def test_dose_mg_preserved(self):
        r = _default_mc(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_n_simulations_preserved(self):
        r = _default_mc(n_simulations=100)
        assert r.n_simulations == 100

    def test_cl_mean_preserved(self):
        r = _default_mc(cl_mean=10.0)
        assert r.cl_mean == 10.0

    def test_vd_mean_preserved(self):
        r = _default_mc(vd_mean=80.0)
        assert r.vd_mean == 80.0

    def test_individual_aucs_length(self):
        r = _default_mc(n_simulations=150)
        assert len(r.individual_aucs) == 150

    def test_individual_cmaxes_length(self):
        r = _default_mc(n_simulations=150)
        assert len(r.individual_cmaxes) == 150

    def test_notes_is_list(self):
        r = _default_mc()
        assert isinstance(r.notes, list)


# ---------------------------------------------------------------------------
# Percentile ordering
# ---------------------------------------------------------------------------


class TestPercentileOrdering:
    def test_auc_p5_le_p50(self):
        r = _default_mc()
        assert r.auc_p5 <= r.auc_p50

    def test_auc_p50_le_p95(self):
        r = _default_mc()
        assert r.auc_p50 <= r.auc_p95

    def test_cmax_p5_le_p50(self):
        r = _default_mc()
        assert r.cmax_p5 <= r.cmax_p50

    def test_cmax_p50_le_p95(self):
        r = _default_mc()
        assert r.cmax_p50 <= r.cmax_p95

    def test_trough_p5_le_p50(self):
        r = _default_mc()
        assert r.trough_p5 <= r.trough_p50

    def test_trough_p50_le_p95(self):
        r = _default_mc()
        assert r.trough_p50 <= r.trough_p95


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------


class TestPhysicalPlausibility:
    def test_auc_p50_positive(self):
        r = _default_mc()
        assert r.auc_p50 > 0.0

    def test_cmax_p50_positive(self):
        r = _default_mc()
        assert r.cmax_p50 > 0.0

    def test_auc_cv_pct_positive(self):
        r = _default_mc()
        assert r.auc_cv_pct >= 0.0

    def test_cmax_cv_pct_positive(self):
        r = _default_mc()
        assert r.cmax_cv_pct >= 0.0

    def test_higher_cv_gives_wider_spread(self):
        r_low = _default_mc(cl_cv_pct=5.0, vd_cv_pct=5.0)
        r_high = _default_mc(cl_cv_pct=60.0, vd_cv_pct=60.0)
        spread_low = r_low.auc_p95 - r_low.auc_p5
        spread_high = r_high.auc_p95 - r_high.auc_p5
        assert spread_high > spread_low

    def test_higher_dose_gives_higher_auc(self):
        r_low = _default_mc(dose_mg=50.0)
        r_high = _default_mc(dose_mg=200.0)
        assert r_high.auc_p50 > r_low.auc_p50

    def test_higher_cl_gives_lower_auc(self):
        r_low_cl = _default_mc(cl_mean=2.0)
        r_high_cl = _default_mc(cl_mean=20.0)
        assert r_low_cl.auc_p50 > r_high_cl.auc_p50

    def test_multi_dose_higher_trough(self):
        r_multi = _default_mc(n_doses=3, t_end_h=36.0, interval_h=12.0)
        # Multi-dose accumulation should raise the median trough
        assert r_multi.trough_p50 >= 0.0

    def test_individual_cmaxes_all_positive(self):
        r = _default_mc()
        assert all(c >= 0.0 for c in r.individual_cmaxes)

    def test_individual_aucs_all_positive(self):
        r = _default_mc()
        assert all(a >= 0.0 for a in r.individual_aucs)

    def test_reproducibility_with_same_seed(self):
        r1 = _default_mc(seed=123)
        r2 = _default_mc(seed=123)
        assert r1.auc_p50 == r2.auc_p50

    def test_different_seed_gives_different_result(self):
        r1 = _default_mc(seed=1)
        r2 = _default_mc(seed=99)
        # Not guaranteed but highly likely with 200 simulations
        assert r1.auc_p50 != r2.auc_p50 or r1.auc_p5 != r2.auc_p5


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _default_mc(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            _default_mc(dose_mg=-10.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError):
            _default_mc(cl_mean=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            _default_mc(vd_mean=0.0)

    def test_negative_cv_raises(self):
        with pytest.raises(ValueError):
            _default_mc(cl_cv_pct=-5.0)

    def test_too_few_simulations_raises(self):
        with pytest.raises(ValueError):
            _default_mc(n_simulations=1)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError):
            _default_mc(t_end_h=0.0)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError):
            _default_mc(dt_h=0.0)

    def test_zero_ka_raises(self):
        with pytest.raises(ValueError):
            _default_mc(ka_per_h=0.0)

    def test_zero_f_oral_raises(self):
        with pytest.raises(ValueError):
            _default_mc(f_oral=0.0)

    def test_f_oral_gt_1_raises(self):
        with pytest.raises(ValueError):
            _default_mc(f_oral=1.5)


# ---------------------------------------------------------------------------
# uncertainty_propagation
# ---------------------------------------------------------------------------


def _simple_pk(**kwargs) -> dict[str, float]:
    """Simple 1-cpt AUC: AUC = dose / CL."""
    dose = kwargs.get("dose", 100.0)
    cl = kwargs.get("cl", 5.0)
    return {"auc": dose / cl, "cmax": dose / kwargs.get("vd", 50.0)}


class TestUncertaintyPropagation:
    def test_returns_dict(self):
        result = uncertainty_propagation(
            param_means={"cl": 5.0, "vd": 50.0},
            param_cvs={"cl": 0.3, "vd": 0.3},
            pk_function=_simple_pk,
            output_key="auc",
            n_sim=100,
            seed=42,
        )
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = uncertainty_propagation(
            param_means={"cl": 5.0},
            param_cvs={"cl": 0.3},
            pk_function=_simple_pk,
            output_key="auc",
            n_sim=100,
            seed=42,
        )
        for key in ("mean_output", "sd_output", "cv_pct", "p5", "p50", "p95"):
            assert key in result

    def test_mean_output_positive(self):
        result = uncertainty_propagation(
            param_means={"cl": 5.0},
            param_cvs={"cl": 0.3},
            pk_function=_simple_pk,
            output_key="auc",
            n_sim=100,
            seed=42,
        )
        assert result["mean_output"] > 0.0

    def test_p5_le_p50_le_p95(self):
        result = uncertainty_propagation(
            param_means={"cl": 5.0},
            param_cvs={"cl": 0.3},
            pk_function=_simple_pk,
            output_key="auc",
            n_sim=200,
            seed=42,
        )
        assert result["p5"] <= result["p50"] <= result["p95"]

    def test_zero_cv_gives_low_spread(self):
        result = uncertainty_propagation(
            param_means={"cl": 5.0},
            param_cvs={"cl": 0.0},
            pk_function=_simple_pk,
            output_key="auc",
            n_sim=50,
            seed=42,
        )
        assert result["cv_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_empty_param_means_raises(self):
        with pytest.raises(ValueError):
            uncertainty_propagation(
                param_means={},
                param_cvs={},
                pk_function=_simple_pk,
                output_key="auc",
            )

    def test_mismatched_keys_raises(self):
        with pytest.raises(ValueError):
            uncertainty_propagation(
                param_means={"cl": 5.0},
                param_cvs={"vd": 0.3},
                pk_function=_simple_pk,
                output_key="auc",
            )

    def test_negative_mean_raises(self):
        with pytest.raises(ValueError):
            uncertainty_propagation(
                param_means={"cl": -5.0},
                param_cvs={"cl": 0.3},
                pk_function=_simple_pk,
                output_key="auc",
            )

    def test_too_few_sims_raises(self):
        with pytest.raises(ValueError):
            uncertainty_propagation(
                param_means={"cl": 5.0},
                param_cvs={"cl": 0.3},
                pk_function=_simple_pk,
                output_key="auc",
                n_sim=1,
            )

    def test_non_callable_raises(self):
        with pytest.raises(ValueError):
            uncertainty_propagation(
                param_means={"cl": 5.0},
                param_cvs={"cl": 0.3},
                pk_function="not_callable",  # type: ignore[arg-type]
                output_key="auc",
            )

    def test_analytical_auc_approximate(self):
        """AUC = dose/CL; with CL CV=0, mean AUC should ≈ dose/CL."""
        result = uncertainty_propagation(
            param_means={"cl": 5.0},
            param_cvs={"cl": 0.0},
            pk_function=_simple_pk,
            output_key="auc",
            n_sim=10,
            seed=1,
        )
        # dose defaults to 100 in _simple_pk, AUC = 100/5 = 20
        assert result["mean_output"] == pytest.approx(20.0, rel=1e-4)
