"""Tests for drug toxicokinetics model (Phase 569)."""

import pytest

from omega_pbpk.risk.toxicokinetics import (
    ToxicokineticsResult,
    screen_doses,
    simulate_toxicokinetics,
)

# Common parameters
PARAMS = dict(
    substance_name="TestDrug",
    dose_mg_per_kg=5.0,
    weight_kg=70.0,
    cl_L_per_h=10.0,
    vd_L=50.0,
)

LOW_DOSE_PARAMS = dict(
    substance_name="TestDrug",
    dose_mg_per_kg=0.001,
    weight_kg=70.0,
    cl_L_per_h=10.0,
    vd_L=50.0,
    noael_mg_L=10.0,
    loael_mg_L=20.0,
)

HIGH_DOSE_PARAMS = dict(
    substance_name="TestDrug",
    dose_mg_per_kg=100.0,
    weight_kg=70.0,
    cl_L_per_h=10.0,
    vd_L=50.0,
    noael_mg_L=0.1,
    loael_mg_L=0.5,
)


class TestSimulateToxicokinetics:
    def test_return_type(self):
        result = simulate_toxicokinetics(**PARAMS)
        assert isinstance(result, ToxicokineticsResult)

    def test_list_lengths(self):
        result = simulate_toxicokinetics(**PARAMS, t_end_h=10.0, dt_h=0.1)
        n_expected = int(10.0 / 0.1) + 1
        assert len(result.times_h) == n_expected
        assert len(result.c_systemic_mg_L) == n_expected

    def test_substance_name_preserved(self):
        result = simulate_toxicokinetics(**PARAMS)
        assert result.substance_name == "TestDrug"

    def test_low_dose_none_toxicity(self):
        result = simulate_toxicokinetics(**LOW_DOSE_PARAMS)
        assert result.toxicity_concern == "none"

    def test_high_dose_likely_or_severe(self):
        result = simulate_toxicokinetics(**HIGH_DOSE_PARAMS)
        assert result.toxicity_concern in ("likely", "severe")

    def test_tnoael_gte_tloael(self):
        result = simulate_toxicokinetics(**PARAMS)
        assert result.tnoael_h >= result.tloael_h

    def test_margin_of_safety_inversely_related_to_dose(self):
        r_low = simulate_toxicokinetics(**{**PARAMS, "dose_mg_per_kg": 1.0})
        r_high = simulate_toxicokinetics(**{**PARAMS, "dose_mg_per_kg": 50.0})
        assert r_low.margin_of_safety > r_high.margin_of_safety

    def test_cmax_positive(self):
        result = simulate_toxicokinetics(**PARAMS)
        assert result.cmax_mg_L > 0

    def test_auc_positive(self):
        result = simulate_toxicokinetics(**PARAMS)
        assert result.auc_mg_h_per_L > 0

    def test_dose_linearity_auc(self):
        r1 = simulate_toxicokinetics(**{**PARAMS, "dose_mg_per_kg": 5.0}, t_end_h=48.0)
        r2 = simulate_toxicokinetics(**{**PARAMS, "dose_mg_per_kg": 10.0}, t_end_h=48.0)
        ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
        assert 1.8 < ratio < 2.2

    def test_dose_linearity_cmax(self):
        r1 = simulate_toxicokinetics(**{**PARAMS, "dose_mg_per_kg": 5.0})
        r2 = simulate_toxicokinetics(**{**PARAMS, "dose_mg_per_kg": 10.0})
        ratio = r2.cmax_mg_L / r1.cmax_mg_L
        assert 1.8 < ratio < 2.2

    def test_possible_toxicity(self):
        # Tune dose so cmax is between noael and loael
        result = simulate_toxicokinetics(
            substance_name="TestDrug",
            dose_mg_per_kg=0.5,
            weight_kg=70.0,
            cl_L_per_h=10.0,
            vd_L=50.0,
            noael_mg_L=0.1,
            loael_mg_L=5.0,
        )
        assert result.toxicity_concern == "possible"

    def test_times_start_at_zero(self):
        result = simulate_toxicokinetics(**PARAMS)
        assert result.times_h[0] == 0.0

    def test_concentration_starts_at_zero(self):
        result = simulate_toxicokinetics(**PARAMS)
        assert result.c_systemic_mg_L[0] == 0.0

    def test_notes_contains_info(self):
        result = simulate_toxicokinetics(**PARAMS)
        assert "1-cpt" in result.notes


class TestValidation:
    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg_per_kg"):
            simulate_toxicokinetics(**{**PARAMS, "dose_mg_per_kg": -1})

    def test_zero_weight(self):
        with pytest.raises(ValueError, match="weight_kg"):
            simulate_toxicokinetics(**{**PARAMS, "weight_kg": 0})

    def test_negative_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_toxicokinetics(**{**PARAMS, "cl_L_per_h": -1})

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_toxicokinetics(**{**PARAMS, "vd_L": 0})

    def test_negative_noael(self):
        with pytest.raises(ValueError, match="noael_mg_L"):
            simulate_toxicokinetics(**PARAMS, noael_mg_L=-1)

    def test_loael_must_exceed_noael(self):
        with pytest.raises(ValueError, match="loael_mg_L"):
            simulate_toxicokinetics(**PARAMS, noael_mg_L=1.0, loael_mg_L=0.5)


class TestScreenDoses:
    def test_sorted_ascending(self):
        results = screen_doses("TestDrug", 70.0, 10.0, 50.0, 0.1, 0.5, [10.0, 1.0, 5.0])
        doses = [r.dose_mg_per_kg for r in results]
        assert doses == sorted(doses)

    def test_returns_list(self):
        results = screen_doses("TestDrug", 70.0, 10.0, 50.0, 0.1, 0.5, [1.0, 5.0])
        assert isinstance(results, list)
        assert len(results) == 2
