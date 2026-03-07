"""Tests for prediction/protein_binding_kinetics_p1024.py — Phase 1024."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.protein_binding_kinetics_p1024 import (
    ProteinBindingResult,
    predict_protein_binding,
    screen_protein_binding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> ProteinBindingResult:
    defaults = dict(
        drug_name="TestDrug",
        total_drug_mg_L=1.0,
        mw_Da=400.0,
        logp=2.0,
        pka=7.0,
        ionization_type="neutral",
        protein="albumin",
    )
    defaults.update(kwargs)
    return predict_protein_binding(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        r = _default()
        assert isinstance(r, ProteinBindingResult)

    def test_frozen_dataclass(self):
        r = _default()
        with pytest.raises((AttributeError, TypeError)):
            r.fu = 0.5  # type: ignore[misc]

    def test_drug_name_preserved(self):
        r = _default(drug_name="Warfarin")
        assert r.drug_name == "Warfarin"

    def test_protein_preserved(self):
        r = _default(protein="AGP")
        assert r.protein == "AGP"

    def test_notes_nonempty(self):
        r = _default()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# fu and percent_bound ranges
# ---------------------------------------------------------------------------


class TestRanges:
    def test_fu_in_range(self):
        r = _default()
        assert 0.0 < r.fu <= 1.0

    def test_percent_bound_in_range(self):
        r = _default()
        assert 0.0 <= r.percent_bound <= 100.0

    def test_saturation_fraction_in_range(self):
        r = _default()
        assert 0.0 <= r.saturation_fraction <= 1.0

    def test_fu_less_than_one(self):
        # Some binding always occurs with non-zero logP
        r = _default(logp=2.0)
        assert r.fu < 1.0

    def test_kd_positive(self):
        r = _default()
        assert r.kd_uM > 0.0


# ---------------------------------------------------------------------------
# Mass balance
# ---------------------------------------------------------------------------


class TestMassBalance:
    def test_bound_plus_free_equals_total(self):
        r = _default(total_drug_mg_L=5.0)
        assert r.bound_drug_mg_L + r.free_drug_mg_L == pytest.approx(r.total_drug_mg_L, abs=1e-6)

    def test_fu_consistent_with_free(self):
        r = _default()
        expected_fu = r.free_drug_mg_L / r.total_drug_mg_L
        assert r.fu == pytest.approx(expected_fu, abs=1e-6)

    def test_percent_bound_consistent_with_fu(self):
        r = _default()
        assert r.percent_bound == pytest.approx((1.0 - r.fu) * 100.0, abs=1e-9)


# ---------------------------------------------------------------------------
# logP effect on binding
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_higher_logp_lower_fu_albumin(self):
        # More lipophilic → tighter binding → lower fu
        r_low = _default(logp=0.0, protein="albumin")
        r_high = _default(logp=4.0, protein="albumin")
        assert r_high.fu < r_low.fu

    def test_higher_logp_lower_fu_agp(self):
        r_low = _default(logp=0.0, protein="AGP")
        r_high = _default(logp=4.0, protein="AGP")
        assert r_high.fu < r_low.fu


# ---------------------------------------------------------------------------
# Ionization type effect
# ---------------------------------------------------------------------------


class TestIonizationEffect:
    def test_acid_lower_fu_albumin_than_base(self):
        # Acids bind albumin more tightly (lower kd → lower fu)
        r_acid = _default(ionization_type="acid", protein="albumin")
        r_base = _default(ionization_type="base", protein="albumin")
        assert r_acid.fu < r_base.fu

    def test_base_lower_fu_agp_than_acid(self):
        # Bases bind AGP more tightly
        r_base = _default(ionization_type="base", protein="AGP")
        r_acid = _default(ionization_type="acid", protein="AGP")
        assert r_base.fu < r_acid.fu


# ---------------------------------------------------------------------------
# Saturation at high concentration
# ---------------------------------------------------------------------------


class TestSaturation:
    def test_high_conc_saturation_approaches_one(self):
        # Very high drug concentration should saturate binding sites
        r = _default(total_drug_mg_L=10000.0)
        assert r.saturation_fraction > 0.5


# ---------------------------------------------------------------------------
# screen_protein_binding
# ---------------------------------------------------------------------------


class TestScreenProteinBinding:
    def test_returns_list_of_length_two(self):
        results = screen_protein_binding(
            drug_name="Drug", total_drug_mg_L=1.0, mw_Da=400.0, logp=2.0
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_contains_proteinbindingresult_instances(self):
        results = screen_protein_binding(
            drug_name="Drug", total_drug_mg_L=1.0, mw_Da=400.0, logp=2.0
        )
        assert all(isinstance(r, ProteinBindingResult) for r in results)

    def test_sorted_by_fu_ascending(self):
        results = screen_protein_binding(
            drug_name="Drug", total_drug_mg_L=1.0, mw_Da=400.0, logp=2.0
        )
        fus = [r.fu for r in results]
        assert fus == sorted(fus)

    def test_both_proteins_covered(self):
        results = screen_protein_binding(
            drug_name="Drug", total_drug_mg_L=1.0, mw_Da=400.0, logp=2.0
        )
        proteins = {r.protein for r in results}
        assert proteins == {"albumin", "AGP"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_total_drug_raises(self):
        with pytest.raises(ValueError):
            _default(total_drug_mg_L=0.0)

    def test_negative_total_drug_raises(self):
        with pytest.raises(ValueError):
            _default(total_drug_mg_L=-1.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError):
            _default(mw_Da=0.0)

    def test_invalid_protein_raises(self):
        with pytest.raises(ValueError):
            _default(protein="fibrinogen")

    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError):
            _default(ionization_type="zwitterion")
