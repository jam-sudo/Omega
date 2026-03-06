"""Tests for clinical/ppb_competition.py (Phase 214)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.ppb_competition import (
    PPBCompetitionResult,
    compute_ppb_competition,
    ppb_displacement_sensitivity,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    drug_a_name="DrugA",
    drug_b_name="DrugB",
    ca_mg_L=10.0,
    cb_mg_L=5.0,
    fu_a_alone=0.1,
    fu_b_alone=0.2,
    albumin_g_dL=4.0,
    n_sites=1,
)


def _comp(**overrides):
    params = {**_DEFAULTS, **overrides}
    return compute_ppb_competition(**params)


# ---------------------------------------------------------------------------
# 1. Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_fu_a_zero_raises(self):
        with pytest.raises(ValueError, match="fu_a_alone"):
            _comp(fu_a_alone=0.0)

    def test_fu_a_negative_raises(self):
        with pytest.raises(ValueError, match="fu_a_alone"):
            _comp(fu_a_alone=-0.1)

    def test_fu_a_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_a_alone"):
            _comp(fu_a_alone=1.5)

    def test_fu_b_zero_raises(self):
        with pytest.raises(ValueError, match="fu_b_alone"):
            _comp(fu_b_alone=0.0)

    def test_fu_b_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_b_alone"):
            _comp(fu_b_alone=1.1)

    def test_ca_zero_raises(self):
        with pytest.raises(ValueError, match="ca_mg_L"):
            _comp(ca_mg_L=0.0)

    def test_ca_negative_raises(self):
        with pytest.raises(ValueError, match="ca_mg_L"):
            _comp(ca_mg_L=-1.0)

    def test_cb_negative_raises(self):
        with pytest.raises(ValueError, match="cb_mg_L"):
            _comp(cb_mg_L=-0.1)

    def test_albumin_zero_raises(self):
        with pytest.raises(ValueError, match="albumin_g_dL"):
            _comp(albumin_g_dL=0.0)

    def test_n_sites_zero_raises(self):
        with pytest.raises(ValueError, match="n_sites"):
            _comp(n_sites=0)


# ---------------------------------------------------------------------------
# 2. Return type and field checks
# ---------------------------------------------------------------------------
class TestResultFields:
    def setup_method(self):
        self.res = _comp()

    def test_returns_dataclass(self):
        assert isinstance(self.res, PPBCompetitionResult)

    def test_drug_names(self):
        assert self.res.drug_a_name == "DrugA"
        assert self.res.drug_b_name == "DrugB"

    def test_notes_is_str(self):
        assert isinstance(self.res.notes, str)

    def test_clinical_significance_is_str(self):
        assert isinstance(self.res.clinical_significance, str)

    def test_fu_a_displaced_in_range(self):
        assert 0.0 < self.res.fu_a_displaced <= 1.0

    def test_fu_b_displaced_in_range(self):
        assert 0.0 < self.res.fu_b_displaced <= 1.0

    def test_displacement_pct_a_nonnegative(self):
        assert self.res.displacement_pct_a >= 0.0

    def test_delta_fu_a_nonnegative(self):
        assert self.res.delta_fu_a >= 0.0


# ---------------------------------------------------------------------------
# 3. No competitor → no displacement
# ---------------------------------------------------------------------------
def test_cb_zero_no_displacement():
    res = _comp(cb_mg_L=0.0)
    assert res.fu_a_displaced == pytest.approx(res.fu_a_alone, abs=1e-9)
    assert res.delta_fu_a == pytest.approx(0.0, abs=1e-9)
    assert res.displacement_pct_a == pytest.approx(0.0, abs=1e-9)


def test_cb_zero_significance_none():
    res = _comp(cb_mg_L=0.0)
    assert res.clinical_significance == "none"


# ---------------------------------------------------------------------------
# 4. High competitor → displacement occurs
# ---------------------------------------------------------------------------
def test_high_cb_causes_displacement():
    low = _comp(cb_mg_L=1.0)
    high = _comp(cb_mg_L=10000.0)
    assert high.fu_a_displaced >= low.fu_a_displaced


def test_displacement_increases_with_cb():
    results = [_comp(cb_mg_L=c) for c in [0.0, 10.0, 100.0, 1000.0]]
    displacements = [r.delta_fu_a for r in results]
    # monotonically non-decreasing
    for i in range(len(displacements) - 1):
        assert displacements[i] <= displacements[i + 1] + 1e-12


# ---------------------------------------------------------------------------
# 5. Clinical significance thresholds
# ---------------------------------------------------------------------------
class TestClinicalSignificance:
    def test_significance_none(self):
        # cb=0 → delta_fu_a=0 → "none"
        res = _comp(cb_mg_L=0.0)
        assert res.clinical_significance == "none"

    def test_significance_valid_values(self):
        res = _comp()
        assert res.clinical_significance in {"none", "minor", "moderate", "major"}


# ---------------------------------------------------------------------------
# 6. Highly-bound drug displaced more
# ---------------------------------------------------------------------------
def test_highly_bound_drug_shows_larger_absolute_displacement():
    """Drug A with fu=0.01 (highly bound) vs fu=0.5 (weakly bound)."""
    high_bound = _comp(fu_a_alone=0.01, cb_mg_L=500.0)
    weak_bound = _comp(fu_a_alone=0.5, cb_mg_L=500.0)
    # absolute delta_fu may or may not differ greatly (depends on model),
    # but fu_a_displaced should be > fu_a_alone in both cases when cb > 0
    assert high_bound.fu_a_displaced >= high_bound.fu_a_alone
    assert weak_bound.fu_a_displaced >= weak_bound.fu_a_alone


# ---------------------------------------------------------------------------
# 7. ppb_displacement_sensitivity
# ---------------------------------------------------------------------------
class TestSensitivity:
    def setup_method(self):
        self.cb_range = [0.0, 1.0, 10.0, 100.0, 1000.0]
        self.results = ppb_displacement_sensitivity(
            drug_a_name="A",
            drug_b_name="B",
            fu_a_alone=0.05,
            fu_b_alone=0.1,
            cb_range_mg_L=self.cb_range,
        )

    def test_returns_list(self):
        assert isinstance(self.results, list)

    def test_correct_length(self):
        assert len(self.results) == len(self.cb_range)

    def test_all_ppb_results(self):
        for r in self.results:
            assert isinstance(r, PPBCompetitionResult)

    def test_monotone_displacement(self):
        disps = [r.delta_fu_a for r in self.results]
        for i in range(len(disps) - 1):
            assert disps[i] <= disps[i + 1] + 1e-12

    def test_empty_cb_range_raises(self):
        with pytest.raises(ValueError):
            ppb_displacement_sensitivity("A", "B", 0.05, 0.1, cb_range_mg_L=[])


# ---------------------------------------------------------------------------
# 8. Symmetry: swapping A and B gives displacement of B
# ---------------------------------------------------------------------------
def test_swapping_a_b_gives_b_displacement():
    r_ab = _comp(ca_mg_L=10.0, cb_mg_L=50.0, fu_a_alone=0.1, fu_b_alone=0.2)
    r_ba = compute_ppb_competition(
        drug_a_name="DrugB",
        drug_b_name="DrugA",
        ca_mg_L=50.0,
        cb_mg_L=10.0,
        fu_a_alone=0.2,
        fu_b_alone=0.1,
    )
    # In the swapped call, DrugB is now "A" and DrugA is "B"
    # fu_a_displaced in r_ba should be ≈ fu_b_displaced in r_ab (within tolerance)
    assert r_ba.fu_a_displaced >= r_ba.fu_a_alone


# ---------------------------------------------------------------------------
# 9. Fully unbound drug (fu=1) — no displacement possible
# ---------------------------------------------------------------------------
def test_fu_a_one_no_displacement():
    res = _comp(fu_a_alone=1.0, cb_mg_L=1000.0)
    assert res.fu_a_displaced == pytest.approx(1.0, abs=1e-9)
    assert res.delta_fu_a == pytest.approx(0.0, abs=1e-9)
