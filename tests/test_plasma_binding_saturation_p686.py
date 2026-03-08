"""Tests for Phase 686 — Plasma Binding Saturation."""

import pytest

from omega_pbpk.prediction.plasma_binding_saturation_p686 import (
    PlasmaBindingSaturationResult,
    simulate_plasma_binding_saturation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = dict(
    drug_name="DrugY",
    mw_Da=400.0,
    logP=3.0,
)


def _run(**overrides):
    kw = {**_BASE, **overrides}
    return simulate_plasma_binding_saturation(**kw)


# ---------------------------------------------------------------------------
# Return type / structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, PlasmaBindingSaturationResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "DrugY"

    def test_total_conc_is_list(self):
        r = _run()
        assert isinstance(r.total_conc_mg_L, list)

    def test_free_conc_is_list(self):
        r = _run()
        assert isinstance(r.free_conc_mg_L, list)

    def test_bound_conc_is_list(self):
        r = _run()
        assert isinstance(r.bound_conc_mg_L, list)

    def test_fu_list_is_list(self):
        r = _run()
        assert isinstance(r.fu_list, list)

    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)

    def test_protein_field(self):
        r = _run()
        assert r.protein == "albumin"


# ---------------------------------------------------------------------------
# Valid proteins
# ---------------------------------------------------------------------------


class TestProteins:
    def test_albumin_accepted(self):
        r = _run(protein="albumin")
        assert r.protein == "albumin"

    def test_aag_accepted(self):
        r = _run(protein="aag")
        assert r.protein == "aag"

    def test_invalid_protein_raises(self):
        with pytest.raises(ValueError, match="protein"):
            _run(protein="globulin")


# ---------------------------------------------------------------------------
# Binding properties
# ---------------------------------------------------------------------------


class TestBindingProperties:
    def test_fu_all_in_valid_range(self):
        r = _run()
        assert all(0 < fu <= 1.0 for fu in r.fu_list)

    def test_free_conc_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.free_conc_mg_L)

    def test_bound_conc_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.bound_conc_mg_L)

    def test_fu_increases_with_concentration(self):
        """fu should increase at higher concentrations (saturation effect)."""
        r = _run(n_points=50)
        # Compare first quarter vs last quarter
        n = len(r.fu_list)
        avg_low = sum(r.fu_list[: n // 4]) / (n // 4)
        avg_high = sum(r.fu_list[3 * n // 4 :]) / (n - 3 * n // 4)
        assert avg_high >= avg_low

    def test_higher_logp_lower_fu_at_low_conc(self):
        """Higher logP → lower Kd → tighter binding → lower fu at moderate concentrations."""
        r_low = _run(logP=1.0, therapeutic_conc_mg_L=1.0)
        r_high = _run(logP=4.0, therapeutic_conc_mg_L=1.0)
        # Compare fu at therapeutic (moderate) concentration where binding matters
        assert r_high.fu_at_therapeutic < r_low.fu_at_therapeutic

    def test_fu_at_therapeutic_valid(self):
        r = _run()
        assert 0 < r.fu_at_therapeutic <= 1.0

    def test_fu_at_toxic_valid(self):
        r = _run()
        assert 0 < r.fu_at_toxic <= 1.0

    def test_binding_saturated_is_bool(self):
        r = _run()
        assert isinstance(r.binding_saturated, bool)


# ---------------------------------------------------------------------------
# Binding parameters
# ---------------------------------------------------------------------------


class TestBindingParameters:
    def test_bmax_positive(self):
        r = _run()
        assert r.bmax_mg_L > 0

    def test_kd_positive(self):
        r = _run()
        assert r.kd_mg_L > 0

    def test_albumin_higher_bmax_than_aag(self):
        r_alb = _run(protein="albumin")
        r_aag = _run(protein="aag")
        assert r_alb.bmax_mg_L > r_aag.bmax_mg_L


# ---------------------------------------------------------------------------
# List lengths
# ---------------------------------------------------------------------------


class TestListLengths:
    def test_default_n_points(self):
        r = _run()
        assert len(r.total_conc_mg_L) == 50

    def test_custom_n_points(self):
        r = _run(n_points=20)
        assert len(r.total_conc_mg_L) == 20

    def test_all_lists_same_length(self):
        r = _run()
        n = len(r.total_conc_mg_L)
        assert len(r.free_conc_mg_L) == n
        assert len(r.bound_conc_mg_L) == n
        assert len(r.fu_list) == n


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _run(mw_Da=0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _run(mw_Da=-100)

    def test_zero_therapeutic_raises(self):
        with pytest.raises(ValueError, match="therapeutic"):
            _run(therapeutic_conc_mg_L=0)

    def test_zero_toxic_raises(self):
        with pytest.raises(ValueError, match="toxic"):
            _run(toxic_conc_mg_L=0)

    def test_n_points_too_small_raises(self):
        with pytest.raises(ValueError, match="n_points"):
            _run(n_points=3)
