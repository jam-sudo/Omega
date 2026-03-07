"""Tests for target organ concentration prediction (Phase 424)."""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.prediction.target_organ_concentration import (
    OrganConcentrationResult,
    organ_exposure_profile,
    predict_organ_concentration,
)

# ---------------------------------------------------------------------------
# Shared synthetic plasma PK (1-cpt IV bolus)
# ---------------------------------------------------------------------------

KE = 0.15
VD = 40.0
DOSE = 200.0
T = np.linspace(0, 24, 289).tolist()
CP = [max((DOSE / VD) * np.exp(-KE * t), 0.0) for t in T]

BASE = dict(
    plasma_concs=CP,
    times_h=T,
    organ_name="liver",
    kp_organ=3.5,
    fu_plasma=0.3,
    fu_tissue=0.1,
)


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------


class TestOrganConcentrationResultFields:
    def test_all_fields_present(self):
        result = predict_organ_concentration(**BASE)
        expected = {
            "organ_name",
            "kp_organ",
            "fu_plasma",
            "fu_tissue",
            "times_h",
            "c_tissue_mg_L",
            "auc_tissue",
            "cmax_tissue",
            "tissue_plasma_auc_ratio",
            "notes",
        }
        for f in expected:
            assert hasattr(result, f), f"Missing field: {f}"

    def test_returns_correct_type(self):
        result = predict_organ_concentration(**BASE)
        assert isinstance(result, OrganConcentrationResult)

    def test_organ_name_stored(self):
        result = predict_organ_concentration(**BASE)
        assert result.organ_name == "liver"

    def test_kp_organ_stored(self):
        result = predict_organ_concentration(**BASE)
        assert result.kp_organ == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Core concentration calculations
# ---------------------------------------------------------------------------


class TestConcentrationCalculations:
    def test_tissue_length_matches_times(self):
        result = predict_organ_concentration(**BASE)
        assert len(result.c_tissue_mg_L) == len(T)

    def test_tissue_conc_is_kp_times_plasma(self):
        result = predict_organ_concentration(**BASE)
        for ct, cp in zip(result.c_tissue_mg_L[:10], CP[:10], strict=True):
            assert ct == pytest.approx(BASE["kp_organ"] * cp, rel=1e-6)

    def test_cmax_tissue_positive(self):
        result = predict_organ_concentration(**BASE)
        assert result.cmax_tissue > 0

    def test_auc_tissue_positive(self):
        result = predict_organ_concentration(**BASE)
        assert result.auc_tissue > 0

    def test_auc_tissue_equals_kp_times_auc_plasma(self):
        result = predict_organ_concentration(**BASE)
        auc_plasma = float(np.trapz(CP, T))
        assert result.auc_tissue == pytest.approx(BASE["kp_organ"] * auc_plasma, rel=1e-4)

    def test_tissue_plasma_ratio_equals_kp(self):
        result = predict_organ_concentration(**BASE)
        assert result.tissue_plasma_auc_ratio == pytest.approx(BASE["kp_organ"], rel=1e-4)

    def test_all_tissue_concs_nonneg(self):
        result = predict_organ_concentration(**BASE)
        assert all(c >= 0 for c in result.c_tissue_mg_L)

    def test_high_kp_gives_higher_auc(self):
        low = predict_organ_concentration(**{**BASE, "kp_organ": 1.0})
        high = predict_organ_concentration(**{**BASE, "kp_organ": 5.0})
        assert high.auc_tissue > low.auc_tissue


# ---------------------------------------------------------------------------
# Different organs
# ---------------------------------------------------------------------------


class TestOrganVariants:
    def test_kidney(self):
        result = predict_organ_concentration(**{**BASE, "organ_name": "kidney"})
        assert result.organ_name == "kidney"
        assert result.auc_tissue > 0

    def test_brain(self):
        result = predict_organ_concentration(**{**BASE, "organ_name": "brain", "kp_organ": 0.05})
        assert result.organ_name == "brain"
        assert any("brain" in n.lower() for n in result.notes)

    def test_lung(self):
        result = predict_organ_concentration(**{**BASE, "organ_name": "lung"})
        assert result.organ_name == "lung"

    def test_heart(self):
        result = predict_organ_concentration(**{**BASE, "organ_name": "heart"})
        assert result.organ_name == "heart"

    def test_muscle(self):
        result = predict_organ_concentration(**{**BASE, "organ_name": "muscle"})
        assert result.organ_name == "muscle"

    def test_fat(self):
        result = predict_organ_concentration(**{**BASE, "organ_name": "fat"})
        assert result.organ_name == "fat"


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_is_list(self):
        result = predict_organ_concentration(**BASE)
        assert isinstance(result.notes, list)

    def test_notes_not_empty(self):
        result = predict_organ_concentration(**BASE)
        assert len(result.notes) >= 1

    def test_consistent_kp_note(self):
        # fu_plasma/fu_tissue = 0.3/0.3 = 1.0, kp=1.0 → consistent
        result = predict_organ_concentration(
            **{**BASE, "kp_organ": 1.0, "fu_plasma": 0.3, "fu_tissue": 0.3}
        )
        assert any("consistent" in n.lower() for n in result.notes)

    def test_inconsistent_kp_note(self):
        # kp=10 but fu_plasma/fu_tissue=1 → very different
        result = predict_organ_concentration(
            **{**BASE, "kp_organ": 10.0, "fu_plasma": 0.5, "fu_tissue": 0.5}
        )
        assert any("deviates" in n.lower() or "active" in n.lower() for n in result.notes)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_empty_plasma_concs(self):
        with pytest.raises(ValueError, match="plasma_concs"):
            predict_organ_concentration(**{**BASE, "plasma_concs": []})

    def test_empty_times_h(self):
        with pytest.raises(ValueError, match="times_h"):
            predict_organ_concentration(**{**BASE, "times_h": []})

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            predict_organ_concentration(**{**BASE, "times_h": T[:5]})

    def test_invalid_organ_name(self):
        with pytest.raises(ValueError, match="organ_name"):
            predict_organ_concentration(**{**BASE, "organ_name": "spleen"})

    def test_nonpositive_kp_organ(self):
        with pytest.raises(ValueError, match="kp_organ"):
            predict_organ_concentration(**{**BASE, "kp_organ": 0})

    def test_zero_fu_plasma(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_organ_concentration(**{**BASE, "fu_plasma": 0})

    def test_zero_fu_tissue(self):
        with pytest.raises(ValueError, match="fu_tissue"):
            predict_organ_concentration(**{**BASE, "fu_tissue": 0})

    def test_fu_plasma_gt_one(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_organ_concentration(**{**BASE, "fu_plasma": 1.5})


# ---------------------------------------------------------------------------
# organ_exposure_profile
# ---------------------------------------------------------------------------


class TestOrganExposureProfile:
    KP_DICT = {"liver": 3.5, "kidney": 5.0, "brain": 0.1, "lung": 2.0}

    def test_returns_dict(self):
        profile = organ_exposure_profile(CP, T, self.KP_DICT, 0.3)
        assert isinstance(profile, dict)

    def test_all_organs_present(self):
        profile = organ_exposure_profile(CP, T, self.KP_DICT, 0.3)
        for organ in self.KP_DICT:
            assert organ in profile

    def test_each_entry_has_required_keys(self):
        profile = organ_exposure_profile(CP, T, self.KP_DICT, 0.3)
        required = {
            "c_tissue_mg_L",
            "auc_tissue",
            "cmax_tissue",
            "tissue_plasma_auc_ratio",
            "result",
        }
        for organ, entry in profile.items():
            assert required.issubset(entry.keys()), f"Missing keys for {organ}"

    def test_result_is_organ_concentration_result(self):
        profile = organ_exposure_profile(CP, T, self.KP_DICT, 0.3)
        for _organ, entry in profile.items():
            assert isinstance(entry["result"], OrganConcentrationResult)

    def test_auc_ratios_match_kp(self):
        profile = organ_exposure_profile(CP, T, self.KP_DICT, 0.3)
        for organ, kp in self.KP_DICT.items():
            ratio = profile[organ]["tissue_plasma_auc_ratio"]
            assert ratio == pytest.approx(kp, rel=1e-4)

    def test_kidney_higher_than_brain(self):
        profile = organ_exposure_profile(CP, T, self.KP_DICT, 0.3)
        assert profile["kidney"]["auc_tissue"] > profile["brain"]["auc_tissue"]

    def test_empty_plasma_raises(self):
        with pytest.raises(ValueError, match="plasma_concs"):
            organ_exposure_profile([], T, self.KP_DICT, 0.3)

    def test_empty_organ_dict_raises(self):
        with pytest.raises(ValueError, match="organ_kp_dict"):
            organ_exposure_profile(CP, T, {}, 0.3)

    def test_invalid_fu_plasma_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            organ_exposure_profile(CP, T, self.KP_DICT, 0.0)

    def test_invalid_fu_tissue_raises(self):
        with pytest.raises(ValueError, match="fu_tissue"):
            organ_exposure_profile(CP, T, self.KP_DICT, 0.3, fu_tissue=0.0)

    def test_single_organ(self):
        profile = organ_exposure_profile(CP, T, {"muscle": 1.2}, 0.5)
        assert "muscle" in profile
        assert profile["muscle"]["auc_tissue"] > 0

    def test_fat_organ(self):
        profile = organ_exposure_profile(CP, T, {"fat": 8.0}, 0.2)
        assert profile["fat"]["auc_tissue"] > 0
