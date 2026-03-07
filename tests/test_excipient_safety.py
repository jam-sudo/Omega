"""Tests for risk/excipient_safety.py — Phase 536."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.excipient_safety import (
    ExcipientSafetyResult,
    assess_excipient,
    screen_formulation,
    suggest_alternatives,
)

# ---------------------------------------------------------------------------
# assess_excipient tests
# ---------------------------------------------------------------------------


class TestAssessExcipient:
    def test_sucrose_oral_acceptable(self):
        res = assess_excipient("sucrose", 5.0, "oral")
        assert res.safety_rating == "acceptable"
        assert res.within_limits is True

    def test_mannitol_oral_acceptable(self):
        res = assess_excipient("mannitol", 10.0, "oral")
        assert res.safety_rating == "acceptable"

    def test_cremophor_el_iv_caution_or_avoid(self):
        """cremophor_el IV has hypersensitivity concern → at least caution."""
        res = assess_excipient("cremophor_el", 0.3, "iv", daily_dose_g_per_day=0.3)
        assert res.safety_rating in {"caution", "avoid"}
        assert "hypersensitivity" in res.concerns

    def test_benzyl_alcohol_iv_above_limit_avoid(self):
        res = assess_excipient("benzyl_alcohol", 0.5, "iv", daily_dose_g_per_day=1.0)
        assert res.safety_rating == "avoid"
        assert res.within_limits is False

    def test_benzyl_alcohol_iv_within_limit_caution_or_avoid(self):
        """Even within IV limit, benzyl alcohol has neurotoxicity concern for neonates."""
        res = assess_excipient("benzyl_alcohol", 0.1, "iv", daily_dose_g_per_day=0.05)
        # Should be caution or avoid due to neurotoxicity_neonates
        assert res.safety_rating in {"caution", "avoid"}
        assert "neurotoxicity_neonates" in res.concerns

    def test_hpmc_iv_avoid(self):
        """HPMC is not for IV use."""
        res = assess_excipient("hpmc", 1.0, "iv")
        assert res.safety_rating == "avoid"
        assert "not_for_iv" in res.concerns
        assert res.within_limits is False

    def test_peg400_oral_within_limit(self):
        res = assess_excipient("peg400", 10.0, "oral")
        assert res.within_limits is True
        assert res.safety_rating in {"acceptable", "caution"}

    def test_peg400_oral_over_limit(self):
        res = assess_excipient("peg400", 50.0, "oral")
        assert res.within_limits is False
        assert res.safety_rating == "avoid"

    def test_propylene_glycol_oral_within_limit(self):
        res = assess_excipient("propylene_glycol", 3.0, "oral")
        assert res.within_limits is True

    def test_propylene_glycol_oral_over_limit(self):
        res = assess_excipient("propylene_glycol", 10.0, "oral")
        assert res.within_limits is False
        assert res.safety_rating == "avoid"

    def test_hpbcd_iv_within_limit(self):
        res = assess_excipient("hpbcd", 5.0, "iv", daily_dose_g_per_day=2.0)
        assert res.within_limits is True

    def test_unknown_excipient_graceful(self):
        """Unknown excipient should not raise; returns caution with unknown flag."""
        res = assess_excipient("unobtainium", 1.0, "oral")
        assert isinstance(res, ExcipientSafetyResult)
        assert res.safety_rating in {"caution", "avoid"}
        assert "unknown_excipient" in res.concerns

    def test_result_fields_present(self):
        res = assess_excipient("sucrose", 10.0, "oral")
        assert hasattr(res, "name")
        assert hasattr(res, "route")
        assert hasattr(res, "concentration_pct")
        assert hasattr(res, "within_limits")
        assert hasattr(res, "concerns")
        assert hasattr(res, "safety_rating")
        assert hasattr(res, "recommendation")
        assert hasattr(res, "notes")

    def test_validation_negative_concentration(self):
        with pytest.raises(ValueError, match="concentration_pct"):
            assess_excipient("sucrose", -1.0, "oral")

    def test_validation_unsupported_route(self):
        with pytest.raises(ValueError, match="route"):
            assess_excipient("sucrose", 1.0, "rectal")

    def test_case_insensitive_name(self):
        res1 = assess_excipient("Sucrose", 5.0, "oral")
        res2 = assess_excipient("SUCROSE", 5.0, "oral")
        assert res1.safety_rating == res2.safety_rating

    def test_polysorbate_80_iv_concern(self):
        res = assess_excipient("polysorbate_80", 0.05, "iv", daily_dose_g_per_day=0.05)
        assert "anaphylaxis" in res.concerns


# ---------------------------------------------------------------------------
# screen_formulation tests
# ---------------------------------------------------------------------------


class TestScreenFormulation:
    def test_basic_oral_formulation(self):
        excipients = [
            {"name": "sucrose", "concentration_pct": 5.0},
            {"name": "mannitol", "concentration_pct": 10.0},
        ]
        results = screen_formulation(excipients, "oral")
        assert len(results) == 2
        for r in results:
            assert isinstance(r, ExcipientSafetyResult)
            assert r.safety_rating == "acceptable"

    def test_mixed_formulation_flags_concerns(self):
        excipients = [
            {"name": "sucrose", "concentration_pct": 5.0},
            {"name": "cremophor_el", "concentration_pct": 0.5},
        ]
        results = screen_formulation(excipients, "oral")
        names = [r.name for r in results]
        assert "sucrose" in names
        assert "cremophor_el" in names

    def test_iv_combined_surfactant_flag(self):
        """If polysorbate_80 + cremophor_el combined > 1% IV, get extra flag."""
        excipients = [
            {"name": "polysorbate_80", "concentration_pct": 0.7},
            {"name": "cremophor_el", "concentration_pct": 0.7, "daily_dose_g_per_day": 0.3},
        ]
        results = screen_formulation(excipients, "iv")
        result_names = [r.name for r in results]
        assert "combined_surfactant_load" in result_names

    def test_iv_surfactant_below_threshold_no_extra_flag(self):
        excipients = [
            {"name": "polysorbate_80", "concentration_pct": 0.4, "daily_dose_g_per_day": 0.05},
        ]
        results = screen_formulation(excipients, "iv")
        result_names = [r.name for r in results]
        assert "combined_surfactant_load" not in result_names

    def test_empty_formulation(self):
        results = screen_formulation([], "oral")
        assert results == []


# ---------------------------------------------------------------------------
# suggest_alternatives tests
# ---------------------------------------------------------------------------


class TestSuggestAlternatives:
    def test_propylene_glycol_iv_suggests_peg400(self):
        alts = suggest_alternatives("propylene_glycol", "iv")
        assert "peg400" in alts

    def test_propylene_glycol_iv_suggests_hpbcd(self):
        alts = suggest_alternatives("propylene_glycol", "iv")
        assert "hpbcd" in alts

    def test_cremophor_el_iv_suggestions(self):
        alts = suggest_alternatives("cremophor_el", "iv")
        assert len(alts) > 0
        assert "polysorbate_80" in alts or "hpbcd" in alts

    def test_benzyl_alcohol_iv_suggests_phenol(self):
        alts = suggest_alternatives("benzyl_alcohol", "iv")
        assert "phenol" in alts

    def test_ethanol_iv_suggestions(self):
        alts = suggest_alternatives("ethanol", "iv")
        assert len(alts) > 0

    def test_unknown_excipient_returns_empty(self):
        alts = suggest_alternatives("unobtainium", "iv")
        assert isinstance(alts, list)
        # Unknown excipient has no known alternatives
        assert len(alts) == 0

    def test_returns_list(self):
        alts = suggest_alternatives("propylene_glycol", "oral")
        assert isinstance(alts, list)
