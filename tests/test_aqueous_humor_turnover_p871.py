"""Tests for Phase 871 — Aqueous Humor Drug Turnover."""

import pytest

from omega_pbpk.core.aqueous_humor_turnover_p871 import (
    AqueousHumorTurnoverResult,
    simulate_aqueous_humor_turnover,
)

# Default test parameters
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=1.0,
    logP=2.0,
    mw_Da=300.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


class TestAqueousHumorTurnoverResult:
    def test_dataclass_fields(self):
        r = AqueousHumorTurnoverResult(drug_name="X", dose_mg=1.0, route="topical")
        assert r.drug_name == "X"
        assert r.dose_mg == 1.0
        assert r.route == "topical"
        assert isinstance(r.times_h, list)
        assert isinstance(r.c_aqueous_mg_L, list)
        assert isinstance(r.c_plasma_mg_L, list)

    def test_default_values(self):
        r = AqueousHumorTurnoverResult(drug_name="X", dose_mg=1.0, route="topical")
        assert r.cmax_aqueous_mg_L == 0.0
        assert r.auc_aqueous_mg_h_per_L == 0.0
        assert r.trabecular_outflow_pct == 0.0


class TestValidation:
    def test_dose_zero(self):
        with pytest.raises(ValueError):
            simulate_aqueous_humor_turnover(**{**DEFAULTS, "dose_mg": 0})

    def test_dose_negative(self):
        with pytest.raises(ValueError):
            simulate_aqueous_humor_turnover(**{**DEFAULTS, "dose_mg": -1})

    def test_cl_sys_zero(self):
        with pytest.raises(ValueError):
            simulate_aqueous_humor_turnover(**{**DEFAULTS, "cl_sys_L_per_h": 0})

    def test_vd_sys_zero(self):
        with pytest.raises(ValueError):
            simulate_aqueous_humor_turnover(**{**DEFAULTS, "vd_sys_L": 0})

    def test_aq_production_zero(self):
        with pytest.raises(ValueError):
            simulate_aqueous_humor_turnover(**{**DEFAULTS, "aq_production_mL_per_min": 0})

    def test_invalid_route(self):
        with pytest.raises(ValueError):
            simulate_aqueous_humor_turnover(**{**DEFAULTS, "route": "iv"})


class TestTopicalRoute:
    def test_returns_result(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS, route="topical")
        assert isinstance(r, AqueousHumorTurnoverResult)

    def test_route_stored(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS, route="topical")
        assert r.route == "topical"

    def test_times_start_at_zero(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS, route="topical")
        assert r.times_h[0] == 0.0

    def test_times_end_near_t_end(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS, route="topical", t_end_h=8.0)
        assert abs(r.times_h[-1] - 8.0) < 0.01

    def test_cmax_positive(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS, route="topical")
        assert r.cmax_aqueous_mg_L > 0

    def test_auc_positive(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS, route="topical")
        assert r.auc_aqueous_mg_h_per_L > 0

    def test_plasma_negligible(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS, route="topical")
        assert all(c == 0.0 for c in r.c_plasma_mg_L)

    def test_aqueous_concentration_rises_then_falls(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS, route="topical")
        cmax_idx = r.c_aqueous_mg_L.index(max(r.c_aqueous_mg_L))
        assert cmax_idx > 0
        assert cmax_idx < len(r.c_aqueous_mg_L) - 1


class TestSystemicRoute:
    def test_returns_result(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS, route="systemic")
        assert isinstance(r, AqueousHumorTurnoverResult)

    def test_plasma_positive(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS, route="systemic")
        assert max(r.c_plasma_mg_L) > 0

    def test_aqueous_positive(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS, route="systemic")
        assert r.cmax_aqueous_mg_L > 0

    def test_auc_positive_systemic(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS, route="systemic")
        assert r.auc_aqueous_mg_h_per_L > 0


class TestDrainageOutflow:
    def test_trabecular_80(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS)
        assert r.trabecular_outflow_pct == 80.0

    def test_uveoscleral_20(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS)
        assert r.uveoscleral_outflow_pct == 20.0

    def test_outflow_sums_to_100(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS)
        assert r.trabecular_outflow_pct + r.uveoscleral_outflow_pct == 100.0


class TestResidenceTime:
    def test_residence_time_positive(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS)
        assert r.residence_time_h > 0

    def test_residence_time_formula(self):
        prod_rate = 0.003
        r = simulate_aqueous_humor_turnover(**DEFAULTS, aq_production_mL_per_min=prod_rate)
        expected = 1.0 / (prod_rate * 60.0 / 0.25)
        assert abs(r.residence_time_h - expected) < 1e-6


class TestIOPEffect:
    def test_iop_negative_for_positive_logP(self):
        r = simulate_aqueous_humor_turnover(**{**DEFAULTS, "logP": 3.0})
        assert r.iop_effect_mmHg < 0

    def test_iop_zero_for_negative_logP(self):
        r = simulate_aqueous_humor_turnover(**{**DEFAULTS, "logP": -1.0})
        assert r.iop_effect_mmHg == 0.0

    def test_iop_clamped(self):
        r = simulate_aqueous_humor_turnover(**{**DEFAULTS, "logP": 25.0})
        assert r.iop_effect_mmHg >= -10.0


class TestDoseLinearity:
    def test_double_dose_doubles_cmax(self):
        r1 = simulate_aqueous_humor_turnover(**{**DEFAULTS, "dose_mg": 1.0}, route="topical")
        r2 = simulate_aqueous_humor_turnover(**{**DEFAULTS, "dose_mg": 2.0}, route="topical")
        ratio = r2.cmax_aqueous_mg_L / r1.cmax_aqueous_mg_L
        assert abs(ratio - 2.0) < 0.2


class TestNotes:
    def test_notes_contain_drug_name(self):
        r = simulate_aqueous_humor_turnover(**DEFAULTS)
        assert "TestDrug" in r.notes
