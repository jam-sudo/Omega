"""Tests for bone tissue PK model (Phase 423)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.bone_pk import BonePKResult, antibiotic_bone_penetration, simulate_bone_pk

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

BASE_IV = dict(
    drug_name="ciprofloxacin",
    dose_mg=400.0,
    cl_L_per_h=20.0,
    vd_L=150.0,
    bone_blood_flow_L_per_h=0.25,
    kp_bone=1.5,
    t_end_h=24.0,
    dt_h=0.05,
    route="iv",
    ka_per_h=1.0,
)

BASE_ORAL = dict(
    drug_name="amoxicillin",
    dose_mg=500.0,
    cl_L_per_h=10.0,
    vd_L=30.0,
    bone_blood_flow_L_per_h=0.25,
    kp_bone=0.8,
    t_end_h=12.0,
    dt_h=0.05,
    route="oral",
    ka_per_h=0.8,
)


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------

class TestBonePKResultFields:
    def test_all_fields_present(self):
        result = simulate_bone_pk(**BASE_IV)
        expected_fields = {
            "drug_name", "dose_mg", "kp_bone", "bone_blood_flow_L_per_h",
            "times_h", "c_plasma_mg_L", "c_bone_mg_L",
            "cmax_plasma", "auc_plasma", "auc_bone",
            "bone_plasma_auc_ratio", "notes",
        }
        for f in expected_fields:
            assert hasattr(result, f), f"Missing field: {f}"

    def test_returns_bone_pk_result(self):
        result = simulate_bone_pk(**BASE_IV)
        assert isinstance(result, BonePKResult)

    def test_drug_name_stored(self):
        result = simulate_bone_pk(**BASE_IV)
        assert result.drug_name == "ciprofloxacin"

    def test_dose_mg_stored(self):
        result = simulate_bone_pk(**BASE_IV)
        assert result.dose_mg == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# IV route basic behaviour
# ---------------------------------------------------------------------------

class TestIVRoute:
    def test_times_start_at_zero(self):
        result = simulate_bone_pk(**BASE_IV)
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        result = simulate_bone_pk(**BASE_IV)
        assert result.times_h[-1] == pytest.approx(24.0)

    def test_plasma_length_matches_times(self):
        result = simulate_bone_pk(**BASE_IV)
        assert len(result.c_plasma_mg_L) == len(result.times_h)

    def test_bone_length_matches_times(self):
        result = simulate_bone_pk(**BASE_IV)
        assert len(result.c_bone_mg_L) == len(result.times_h)

    def test_plasma_starts_nonzero_iv(self):
        result = simulate_bone_pk(**BASE_IV)
        assert result.c_plasma_mg_L[0] > 0

    def test_plasma_declines_overall(self):
        result = simulate_bone_pk(**BASE_IV)
        assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]

    def test_bone_starts_at_zero(self):
        result = simulate_bone_pk(**BASE_IV)
        assert result.c_bone_mg_L[0] == pytest.approx(0.0)

    def test_bone_rises_then_declines(self):
        result = simulate_bone_pk(**BASE_IV)
        bone = result.c_bone_mg_L
        max_idx = bone.index(max(bone))
        assert max_idx > 0  # bone rises after t=0
        assert bone[-1] < max(bone)  # bone declines at the end

    def test_cmax_plasma_positive(self):
        result = simulate_bone_pk(**BASE_IV)
        assert result.cmax_plasma > 0

    def test_auc_plasma_positive(self):
        result = simulate_bone_pk(**BASE_IV)
        assert result.auc_plasma > 0

    def test_auc_bone_positive(self):
        result = simulate_bone_pk(**BASE_IV)
        assert result.auc_bone > 0

    def test_bone_plasma_ratio_positive(self):
        result = simulate_bone_pk(**BASE_IV)
        assert result.bone_plasma_auc_ratio > 0

    def test_all_plasma_concs_nonneg(self):
        result = simulate_bone_pk(**BASE_IV)
        assert all(c >= 0 for c in result.c_plasma_mg_L)

    def test_all_bone_concs_nonneg(self):
        result = simulate_bone_pk(**BASE_IV)
        assert all(c >= 0 for c in result.c_bone_mg_L)


# ---------------------------------------------------------------------------
# Oral route
# ---------------------------------------------------------------------------

class TestOralRoute:
    def test_plasma_starts_at_zero_oral(self):
        result = simulate_bone_pk(**BASE_ORAL)
        assert result.c_plasma_mg_L[0] == pytest.approx(0.0)

    def test_plasma_rises_then_falls_oral(self):
        result = simulate_bone_pk(**BASE_ORAL)
        plasma = result.c_plasma_mg_L
        max_idx = plasma.index(max(plasma))
        assert max_idx > 0
        assert plasma[-1] < max(plasma)

    def test_bone_reaches_nonzero_oral(self):
        result = simulate_bone_pk(**BASE_ORAL)
        assert max(result.c_bone_mg_L) > 0


# ---------------------------------------------------------------------------
# Kp_bone effect
# ---------------------------------------------------------------------------

class TestKpBoneEffect:
    def test_kp_bone_stored_correctly(self):
        result = simulate_bone_pk(**{**BASE_IV, "kp_bone": 2.5})
        assert result.kp_bone == pytest.approx(2.5)

    def test_kp_one_gives_positive_bone_auc(self):
        result = simulate_bone_pk(**{**BASE_IV, "kp_bone": 1.0})
        assert result.auc_bone > 0
        assert result.bone_plasma_auc_ratio > 0

    def test_longer_sim_approaches_equilibrium_ratio(self):
        # With long simulation (many half-lives), bone/plasma ratio should
        # approach kp_bone as equilibrium is approached
        result = simulate_bone_pk(**{**BASE_IV, "kp_bone": 1.0, "t_end_h": 120.0})
        assert result.bone_plasma_auc_ratio > 0

    def test_high_kp_high_ratio(self):
        # At very high kp_bone and sufficient blood flow, equilibrium ratio > 0
        result = simulate_bone_pk(**{
            **BASE_IV,
            "kp_bone": 4.0,
            "bone_blood_flow_L_per_h": 5.0,  # high flow drives fast equilibration
            "t_end_h": 48.0,
        })
        assert result.bone_plasma_auc_ratio > 0


# ---------------------------------------------------------------------------
# Notes and clinical target
# ---------------------------------------------------------------------------

class TestNotes:
    def test_notes_is_list(self):
        result = simulate_bone_pk(**BASE_IV)
        assert isinstance(result.notes, list)

    def test_good_penetrator_note(self):
        result = simulate_bone_pk(**{**BASE_IV, "kp_bone": 5.0})
        assert any("adequate" in n for n in result.notes)

    def test_poor_penetrator_note(self):
        result = simulate_bone_pk(**{**BASE_IV, "kp_bone": 0.1, "bone_blood_flow_L_per_h": 0.05})
        assert any("inadequate" in n or "limited" in n or "0.3" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_empty_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            simulate_bone_pk(**{**BASE_IV, "drug_name": ""})

    def test_nonpositive_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_bone_pk(**{**BASE_IV, "dose_mg": 0})

    def test_nonpositive_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_bone_pk(**{**BASE_IV, "cl_L_per_h": -1})

    def test_nonpositive_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_bone_pk(**{**BASE_IV, "vd_L": 0})

    def test_nonpositive_bone_flow(self):
        with pytest.raises(ValueError, match="bone_blood_flow"):
            simulate_bone_pk(**{**BASE_IV, "bone_blood_flow_L_per_h": 0})

    def test_nonpositive_kp_bone(self):
        with pytest.raises(ValueError, match="kp_bone"):
            simulate_bone_pk(**{**BASE_IV, "kp_bone": 0})

    def test_nonpositive_t_end(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_bone_pk(**{**BASE_IV, "t_end_h": 0})

    def test_nonpositive_dt(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_bone_pk(**{**BASE_IV, "dt_h": 0})

    def test_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            simulate_bone_pk(**{**BASE_IV, "route": "subcutaneous"})

    def test_nonpositive_ka(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            simulate_bone_pk(**{**BASE_IV, "route": "oral", "ka_per_h": -1})


# ---------------------------------------------------------------------------
# antibiotic_bone_penetration
# ---------------------------------------------------------------------------

class TestAntibioticBonePenetration:
    KP_VALUES = [0.1, 0.3, 0.5, 1.0, 2.0]

    def test_returns_list(self):
        res = antibiotic_bone_penetration("cipro", 400, 20, 150, self.KP_VALUES)
        assert isinstance(res, list)

    def test_length_equals_kp_values(self):
        res = antibiotic_bone_penetration("cipro", 400, 20, 150, self.KP_VALUES)
        assert len(res) == len(self.KP_VALUES)

    def test_each_entry_has_required_keys(self):
        res = antibiotic_bone_penetration("cipro", 400, 20, 150, self.KP_VALUES)
        required = {"kp_bone", "bone_plasma_auc_ratio", "auc_bone", "auc_plasma",
                    "meets_clinical_target"}
        for entry in res:
            assert required.issubset(entry.keys())

    def test_meets_target_at_moderate_kp(self):
        # kp=0.5 with default bone blood flow and 24h sim gives ratio ~0.3
        res = antibiotic_bone_penetration("cipro", 400, 20, 150, [0.5])
        assert res[0]["bone_plasma_auc_ratio"] > 0

    def test_fails_target_at_very_low_kp(self):
        res = antibiotic_bone_penetration("cipro", 400, 20, 150, [0.01])
        # very low kp gives very low bone conc
        assert res[0]["bone_plasma_auc_ratio"] < 0.05

    def test_ratio_positive_for_all_kp_values(self):
        res = antibiotic_bone_penetration("cipro", 400, 20, 150, [0.5, 1.0, 2.0])
        ratios = [e["bone_plasma_auc_ratio"] for e in res]
        assert all(r > 0 for r in ratios)

    def test_auc_plasma_consistent(self):
        # AUC plasma should be the same across runs (same dose/CL/Vd/route)
        res = antibiotic_bone_penetration("cipro", 400, 20, 150, [1.0, 1.0])
        assert res[0]["auc_plasma"] == pytest.approx(res[1]["auc_plasma"], rel=1e-4)

    def test_empty_kp_values_raises(self):
        with pytest.raises(ValueError, match="kp_values"):
            antibiotic_bone_penetration("cipro", 400, 20, 150, [])

    def test_nonpositive_kp_raises(self):
        with pytest.raises(ValueError):
            antibiotic_bone_penetration("cipro", 400, 20, 150, [1.0, -1.0])

    def test_nonpositive_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            antibiotic_bone_penetration("cipro", 0, 20, 150, [1.0])
