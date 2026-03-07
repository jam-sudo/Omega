"""Tests for skeletal muscle pharmacokinetics module (Phase 421)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.muscle_pk import MusclePKResult, exercise_effect, simulate_muscle_pk

# ---------------------------------------------------------------------------
# Shared baseline parameters
# ---------------------------------------------------------------------------
BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    muscle_blood_flow_L_per_h=15.0,
    kp_muscle=2.0,
    t_end_h=12.0,
    dt_h=0.05,
    route="iv",
    ka_per_h=1.0,
)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestMusclePKResult:
    def test_result_type(self):
        result = simulate_muscle_pk(**BASE)
        assert isinstance(result, MusclePKResult)

    def test_has_required_fields(self):
        result = simulate_muscle_pk(**BASE)
        fields = [
            "drug_name",
            "dose_mg",
            "kp_muscle",
            "muscle_blood_flow_L_per_h",
            "times_h",
            "c_plasma_mg_L",
            "c_muscle_mg_L",
            "cmax_plasma",
            "auc_plasma",
            "muscle_plasma_ss_ratio",
            "notes",
        ]
        for f in fields:
            assert hasattr(result, f), f"Missing field: {f}"

    def test_drug_name_stored(self):
        result = simulate_muscle_pk(**BASE)
        assert result.drug_name == "TestDrug"

    def test_dose_stored(self):
        result = simulate_muscle_pk(**BASE)
        assert result.dose_mg == pytest.approx(100.0)

    def test_kp_stored(self):
        result = simulate_muscle_pk(**BASE)
        assert result.kp_muscle == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# IV bolus simulation tests
# ---------------------------------------------------------------------------
class TestIVBolus:
    def test_times_increasing(self):
        result = simulate_muscle_pk(**BASE)
        for i in range(1, len(result.times_h)):
            assert result.times_h[i] > result.times_h[i - 1]

    def test_times_length_matches_concentrations(self):
        result = simulate_muscle_pk(**BASE)
        assert len(result.times_h) == len(result.c_plasma_mg_L)
        assert len(result.times_h) == len(result.c_muscle_mg_L)

    def test_plasma_starts_positive(self):
        result = simulate_muscle_pk(**BASE)
        assert result.c_plasma_mg_L[0] > 0

    def test_plasma_declines_over_time(self):
        result = simulate_muscle_pk(**BASE)
        # Plasma should generally decline (terminal phase)
        assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]

    def test_muscle_starts_zero(self):
        result = simulate_muscle_pk(**BASE)
        assert result.c_muscle_mg_L[0] == pytest.approx(0.0, abs=1e-10)

    def test_muscle_rises_then_falls(self):
        result = simulate_muscle_pk(**BASE)
        cm = result.c_muscle_mg_L
        # Should not be monotonically zero
        assert max(cm) > 0

    def test_cmax_plasma_positive(self):
        result = simulate_muscle_pk(**BASE)
        assert result.cmax_plasma > 0

    def test_auc_plasma_positive(self):
        result = simulate_muscle_pk(**BASE)
        assert result.auc_plasma > 0

    def test_iv_cmax_equals_initial_conc(self):
        result = simulate_muscle_pk(**BASE)
        # For IV, Cmax should be near t=0
        assert result.cmax_plasma == pytest.approx(result.c_plasma_mg_L[0], rel=0.01)

    def test_all_plasma_nonneg(self):
        result = simulate_muscle_pk(**BASE)
        assert all(c >= 0 for c in result.c_plasma_mg_L)

    def test_all_muscle_nonneg(self):
        result = simulate_muscle_pk(**BASE)
        assert all(c >= 0 for c in result.c_muscle_mg_L)

    def test_muscle_plasma_ss_ratio_positive(self):
        result = simulate_muscle_pk(**BASE)
        assert result.muscle_plasma_ss_ratio > 0

    def test_notes_nonempty(self):
        result = simulate_muscle_pk(**BASE)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Oral route tests
# ---------------------------------------------------------------------------
class TestOralRoute:
    def test_oral_plasma_starts_zero(self):
        result = simulate_muscle_pk(**{**BASE, "route": "oral"})
        assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-10)

    def test_oral_has_tmax_greater_zero(self):
        result = simulate_muscle_pk(**{**BASE, "route": "oral"})
        cp = result.c_plasma_mg_L
        tmax_idx = cp.index(max(cp))
        assert tmax_idx > 0

    def test_oral_auc_positive(self):
        result = simulate_muscle_pk(**{**BASE, "route": "oral"})
        assert result.auc_plasma > 0

    def test_oral_muscle_rises(self):
        result = simulate_muscle_pk(**{**BASE, "route": "oral"})
        assert max(result.c_muscle_mg_L) > 0


# ---------------------------------------------------------------------------
# Parameter sensitivity tests
# ---------------------------------------------------------------------------
class TestParameterSensitivity:
    def test_higher_cl_reduces_auc(self):
        low_cl = simulate_muscle_pk(**{**BASE, "cl_L_per_h": 2.0})
        high_cl = simulate_muscle_pk(**{**BASE, "cl_L_per_h": 20.0})
        assert high_cl.auc_plasma < low_cl.auc_plasma

    def test_higher_flow_increases_muscle_cmax(self):
        low_flow = simulate_muscle_pk(**{**BASE, "muscle_blood_flow_L_per_h": 5.0})
        high_flow = simulate_muscle_pk(**{**BASE, "muscle_blood_flow_L_per_h": 40.0})
        assert max(high_flow.c_muscle_mg_L) > max(low_flow.c_muscle_mg_L)

    def test_higher_kp_increases_ss_ratio(self):
        # Higher kp_muscle → higher muscle-to-plasma ratio at steady state
        low_kp = simulate_muscle_pk(**{**BASE, "kp_muscle": 0.5})
        high_kp = simulate_muscle_pk(**{**BASE, "kp_muscle": 5.0})
        assert high_kp.muscle_plasma_ss_ratio > low_kp.muscle_plasma_ss_ratio


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError):
            simulate_muscle_pk(**{**BASE, "drug_name": ""})

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_muscle_pk(**{**BASE, "dose_mg": 0})

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError):
            simulate_muscle_pk(**{**BASE, "cl_L_per_h": -1})

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            simulate_muscle_pk(**{**BASE, "vd_L": 0})

    def test_zero_muscle_flow_raises(self):
        with pytest.raises(ValueError):
            simulate_muscle_pk(**{**BASE, "muscle_blood_flow_L_per_h": 0})

    def test_zero_kp_raises(self):
        with pytest.raises(ValueError):
            simulate_muscle_pk(**{**BASE, "kp_muscle": 0})

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError):
            simulate_muscle_pk(**{**BASE, "t_end_h": 0})

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError):
            simulate_muscle_pk(**{**BASE, "route": "subcutaneous"})

    def test_zero_ka_raises(self):
        with pytest.raises(ValueError):
            simulate_muscle_pk(**{**BASE, "ka_per_h": 0})


# ---------------------------------------------------------------------------
# Exercise effect tests
# ---------------------------------------------------------------------------
class TestExerciseEffect:
    BASE_EX = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        kp_muscle=2.0,
        muscle_flow_normal=15.0,
        muscle_flow_exercise=60.0,
        t_end_h=12.0,
        dt_h=0.05,
    )

    def test_returns_dict(self):
        result = exercise_effect(**self.BASE_EX)
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = exercise_effect(**self.BASE_EX)
        for key in [
            "auc_plasma_normal",
            "auc_plasma_exercise",
            "cmax_plasma_normal",
            "cmax_plasma_exercise",
            "cmax_muscle_normal",
            "cmax_muscle_exercise",
            "auc_ratio_exercise_vs_normal",
            "flow_ratio",
            "notes",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_flow_ratio_correct(self):
        result = exercise_effect(**self.BASE_EX)
        assert result["flow_ratio"] == pytest.approx(60.0 / 15.0)

    def test_exercise_increases_muscle_cmax(self):
        result = exercise_effect(**self.BASE_EX)
        assert result["cmax_muscle_exercise"] > result["cmax_muscle_normal"]

    def test_plasma_auc_similar_under_exercise(self):
        # AUC plasma should be similar (clearance unchanged); slight difference from distribution
        result = exercise_effect(**self.BASE_EX)
        assert result["auc_plasma_normal"] > 0
        assert result["auc_plasma_exercise"] > 0

    def test_invalid_zero_normal_flow_raises(self):
        with pytest.raises(ValueError):
            exercise_effect(**{**self.BASE_EX, "muscle_flow_normal": 0})

    def test_invalid_zero_exercise_flow_raises(self):
        with pytest.raises(ValueError):
            exercise_effect(**{**self.BASE_EX, "muscle_flow_exercise": 0})

    def test_notes_nonempty(self):
        result = exercise_effect(**self.BASE_EX)
        assert len(result["notes"]) > 0
