"""Tests for inflammation-altered pharmacokinetics — Phase 498."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.inflammation_pk import (
    InflammationPKResult,
    acute_phase_protein_binding,
    cyp_activity_in_inflammation,
    inflammation_pk_scaling,
    simulate_inflamed_pk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_scaling(**kwargs) -> InflammationPKResult:
    return inflammation_pk_scaling("Midazolam", 5.0, 50.0, **kwargs)


# ---------------------------------------------------------------------------
# inflammation_pk_scaling
# ---------------------------------------------------------------------------


class TestInflammationPKScaling:
    def test_returns_result_instance(self):
        result = _default_scaling()
        assert isinstance(result, InflammationPKResult)

    def test_frozen_dataclass(self):
        result = _default_scaling()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = _default_scaling()
        assert result.drug_name == "Midazolam"

    def test_severity_preserved(self):
        result = _default_scaling(severity="severe")
        assert result.severity == "severe"

    def test_severe_cl_lower_than_moderate(self):
        r_mod = _default_scaling(severity="moderate")
        r_sev = _default_scaling(severity="severe")
        assert r_sev.cl_inflamed_L_per_h < r_mod.cl_inflamed_L_per_h

    def test_critical_cl_lower_than_severe(self):
        r_sev = _default_scaling(severity="severe")
        r_crit = _default_scaling(severity="critical")
        assert r_crit.cl_inflamed_L_per_h < r_sev.cl_inflamed_L_per_h

    def test_cl_reduction_pct_positive_for_non_mild(self):
        for sev in ("moderate", "severe", "critical"):
            result = _default_scaling(severity=sev)
            assert result.cl_reduction_pct > 0.0, f"Expected reduction for {sev}"

    def test_mild_cl_reduction_smallest(self):
        r_mild = _default_scaling(severity="mild")
        r_mod = _default_scaling(severity="moderate")
        assert r_mild.cl_reduction_pct < r_mod.cl_reduction_pct

    def test_dose_adjustment_factor_less_than_one(self):
        for sev in ("moderate", "severe", "critical"):
            result = _default_scaling(severity=sev)
            assert result.dose_adjustment_factor < 1.0, f"Expected <1 for {sev}"

    def test_t_half_longer_in_inflammation(self):
        # Normal t_half = ln2 * Vd / CL
        cl_norm = 5.0
        vd_norm = 50.0
        t_half_normal = math.log(2) * vd_norm / cl_norm
        result = inflammation_pk_scaling("X", cl_norm, vd_norm, severity="severe")
        assert result.t_half_inflamed_h > t_half_normal

    def test_vd_inflamed_larger_than_normal(self):
        result = _default_scaling(severity="severe")
        assert result.vd_inflamed_L > 50.0

    def test_notes_nonempty(self):
        result = _default_scaling()
        assert len(result.notes) > 0

    def test_cyp3a4_drug_largest_cl_reduction_severe(self):
        r_cyp3a4 = inflammation_pk_scaling(
            "Midazolam", 5.0, 50.0, severity="severe", cyp_isoforms=["CYP3A4"]
        )
        r_no_cyp = inflammation_pk_scaling("Midazolam", 5.0, 50.0, severity="severe")
        # CYP3A4 drug should have greater CL reduction
        assert r_cyp3a4.cl_reduction_pct >= r_no_cyp.cl_reduction_pct

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity"):
            inflammation_pk_scaling("X", 5.0, 50.0, severity="extreme")

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_normal_L_per_h"):
            inflammation_pk_scaling("X", 0.0, 50.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            inflammation_pk_scaling("X", 5.0, 0.0)


# ---------------------------------------------------------------------------
# acute_phase_protein_binding
# ---------------------------------------------------------------------------


class TestAcutePhaseProteinBinding:
    def test_returns_float(self):
        result = acute_phase_protein_binding(0.1, severity="moderate")
        assert isinstance(result, float)

    def test_result_in_range(self):
        result = acute_phase_protein_binding(0.1, severity="severe")
        assert 0.0 < result <= 1.0

    def test_albumin_fu_increases_with_severity(self):
        # Albumin decreases → bound fraction decreases → fu increases
        r_mild = acute_phase_protein_binding(0.1, severity="mild", protein_type="albumin")
        r_sev = acute_phase_protein_binding(0.1, severity="severe", protein_type="albumin")
        assert r_sev >= r_mild

    def test_aag_fu_decreases_with_severity(self):
        # AAG increases → bound fraction increases → fu decreases
        r_mild = acute_phase_protein_binding(0.1, severity="mild", protein_type="aag")
        r_sev = acute_phase_protein_binding(0.1, severity="severe", protein_type="aag")
        assert r_sev <= r_mild

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity"):
            acute_phase_protein_binding(0.1, severity="unknown")

    def test_invalid_fu_baseline_raises(self):
        with pytest.raises(ValueError, match="fu_baseline"):
            acute_phase_protein_binding(0.0)

    def test_invalid_protein_type_raises(self):
        with pytest.raises(ValueError, match="protein_type"):
            acute_phase_protein_binding(0.1, protein_type="globulin")


# ---------------------------------------------------------------------------
# cyp_activity_in_inflammation
# ---------------------------------------------------------------------------


class TestCYPActivityInInflammation:
    def test_returns_float(self):
        result = cyp_activity_in_inflammation("CYP3A4")
        assert isinstance(result, float)

    def test_result_between_zero_and_one(self):
        for cyp in ("CYP3A4", "CYP1A2", "CYP2C9", "CYP2D6"):
            result = cyp_activity_in_inflammation(cyp, "severe")
            assert 0.0 <= result <= 1.0, f"Out of range for {cyp}"

    def test_cyp3a4_most_suppressed_in_severe(self):
        cyp3a4 = cyp_activity_in_inflammation("CYP3A4", "severe")
        cyp2d6 = cyp_activity_in_inflammation("CYP2D6", "severe")
        assert cyp3a4 < cyp2d6

    def test_cyp1a2_highly_suppressed_in_severe(self):
        cyp1a2 = cyp_activity_in_inflammation("CYP1A2", "severe")
        assert cyp1a2 < 0.5

    def test_activity_decreases_with_severity(self):
        for cyp in ("CYP3A4", "CYP1A2", "CYP2C9"):
            mild = cyp_activity_in_inflammation(cyp, "mild")
            moderate = cyp_activity_in_inflammation(cyp, "moderate")
            severe = cyp_activity_in_inflammation(cyp, "severe")
            assert mild >= moderate >= severe, f"Expected decreasing for {cyp}"

    def test_mild_activity_close_to_normal(self):
        result = cyp_activity_in_inflammation("CYP3A4", "mild")
        assert result > 0.8  # mild = minimal suppression

    def test_unknown_cyp_returns_default(self):
        result = cyp_activity_in_inflammation("CYP99X", "severe")
        assert 0.0 < result <= 1.0

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity"):
            cyp_activity_in_inflammation("CYP3A4", "extreme")


# ---------------------------------------------------------------------------
# simulate_inflamed_pk
# ---------------------------------------------------------------------------


class TestSimulateInflamedPK:
    def test_returns_dict(self):
        result = simulate_inflamed_pk("X", 100.0, 5.0, 50.0)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = simulate_inflamed_pk("X", 100.0, 5.0, 50.0)
        for key in ("times_h", "c_inflamed_mg_L", "c_normal_mg_L", "auc_inflamed", "auc_normal"):
            assert key in result, f"Missing key: {key}"

    def test_auc_inflamed_greater_than_normal(self):
        result = simulate_inflamed_pk("X", 100.0, 5.0, 50.0, severity="severe")
        assert result["auc_inflamed"] > result["auc_normal"]

    def test_auc_ratio_greater_than_one(self):
        result = simulate_inflamed_pk("X", 100.0, 5.0, 50.0, severity="severe")
        assert result["auc_ratio"] > 1.0

    def test_concentrations_same_length_as_times(self):
        result = simulate_inflamed_pk("X", 100.0, 5.0, 50.0)
        n = len(result["times_h"])
        assert len(result["c_inflamed_mg_L"]) == n
        assert len(result["c_normal_mg_L"]) == n

    def test_pk_result_is_inflammation_result(self):
        result = simulate_inflamed_pk("X", 100.0, 5.0, 50.0)
        assert isinstance(result["pk_result"], InflammationPKResult)

    def test_drug_name_preserved(self):
        result = simulate_inflamed_pk("Warfarin", 10.0, 0.05, 10.0)
        assert result["drug_name"] == "Warfarin"

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_inflamed_pk("X", 0.0, 5.0, 50.0)

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity"):
            simulate_inflamed_pk("X", 100.0, 5.0, 50.0, severity="extreme")

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_normal_L_per_h"):
            simulate_inflamed_pk("X", 100.0, 0.0, 50.0)
