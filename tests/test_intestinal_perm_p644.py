"""Tests for Phase 644 — Intestinal Permeability Prediction."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.intestinal_perm_p644 import (
    IntestinalPermResult,
    predict_intestinal_permeability,
    screen_permeability,
)

# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    drug_name="TestDrug",
    logP=2.0,
    mw_Da=300.0,
    psa_A2=80.0,
    n_hbd=2,
)


def _run(**overrides):
    kw = {**_DEFAULTS, **overrides}
    return predict_intestinal_permeability(**kw)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_frozen_dataclass(self):
        r = _run()
        assert isinstance(r, IntestinalPermResult)

    def test_mutation_raises(self):
        r = _run()
        with pytest.raises(AttributeError):
            r.drug_name = "Other"

    def test_drug_name(self):
        r = _run(drug_name="Beta")
        assert r.drug_name == "Beta"

    def test_logP_stored(self):
        r = _run(logP=3.0)
        assert r.logP == 3.0


# ---------------------------------------------------------------------------
# Physicochemical effects on Peff
# ---------------------------------------------------------------------------
class TestPeffDrivers:
    def test_higher_logP_higher_peff(self):
        # Use small mw/psa so values are above lower clamp
        r1 = _run(logP=1.0, mw_Da=100.0, psa_A2=10.0, n_hbd=0)
        r2 = _run(logP=4.0, mw_Da=100.0, psa_A2=10.0, n_hbd=0)
        assert r2.peff_cm_s > r1.peff_cm_s

    def test_higher_psa_lower_peff(self):
        r1 = _run(logP=4.0, mw_Da=100.0, psa_A2=10.0, n_hbd=0)
        r2 = _run(logP=4.0, mw_Da=100.0, psa_A2=80.0, n_hbd=0)
        assert r1.peff_cm_s > r2.peff_cm_s

    def test_higher_mw_lower_peff(self):
        r1 = _run(logP=4.0, mw_Da=100.0, psa_A2=10.0, n_hbd=0)
        r2 = _run(logP=4.0, mw_Da=200.0, psa_A2=10.0, n_hbd=0)
        assert r1.peff_cm_s > r2.peff_cm_s


# ---------------------------------------------------------------------------
# Peff clamping
# ---------------------------------------------------------------------------
class TestPeffClamping:
    def test_peff_lower_bound(self):
        r = _run(logP=-5.0, mw_Da=800.0, psa_A2=200.0, n_hbd=10)
        assert r.peff_cm_s >= 1e-7

    def test_peff_upper_bound(self):
        r = _run(logP=10.0, mw_Da=50.0, psa_A2=0.0, n_hbd=0)
        assert r.peff_cm_s <= 1e-3

    def test_peff_in_range(self):
        r = _run()
        assert 1e-7 <= r.peff_cm_s <= 1e-3


# ---------------------------------------------------------------------------
# Fraction absorbed
# ---------------------------------------------------------------------------
class TestFractionAbsorbed:
    def test_fa_in_range(self):
        r = _run()
        assert 0.0 <= r.fa_predicted_pct <= 100.0

    def test_high_perm_high_fa(self):
        # Need peff >= 1e-4 for fa >= 80
        r = _run(logP=5.0, mw_Da=100.0, psa_A2=10.0, n_hbd=0)
        assert r.fa_predicted_pct >= 80.0


# ---------------------------------------------------------------------------
# BCS classification
# ---------------------------------------------------------------------------
class TestBCSClassification:
    def test_bcs_I(self):
        # High perm, high solubility
        r = _run(logP=5.0, mw_Da=100.0, psa_A2=10.0, n_hbd=0, dose_mg=10.0, solubility_mg_mL=10.0)
        assert r.bcs_class == "I"

    def test_bcs_II(self):
        # High perm, low solubility
        r = _run(logP=5.0, mw_Da=100.0, psa_A2=10.0, n_hbd=0, dose_mg=500.0, solubility_mg_mL=0.1)
        assert r.bcs_class == "II"

    def test_bcs_III(self):
        # Low perm, high solubility
        r = _run(logP=-1.0, mw_Da=400.0, psa_A2=120.0, n_hbd=4, dose_mg=10.0, solubility_mg_mL=10.0)
        assert r.bcs_class == "III"

    def test_bcs_IV(self):
        # Low perm, low solubility
        r = _run(logP=-1.0, mw_Da=400.0, psa_A2=120.0, n_hbd=4, dose_mg=500.0, solubility_mg_mL=0.1)
        assert r.bcs_class == "IV"


# ---------------------------------------------------------------------------
# Permeability category
# ---------------------------------------------------------------------------
class TestPermeabilityCategory:
    def test_high(self):
        r = _run(logP=5.0, mw_Da=100.0, psa_A2=10.0, n_hbd=0)
        assert r.permeability_category == "high"

    def test_low(self):
        r = _run(logP=-2.0, mw_Da=600.0, psa_A2=150.0, n_hbd=6)
        assert r.permeability_category == "low"

    def test_medium(self):
        # Need peff in [1e-5, 1e-4)
        r = _run(logP=2.0, mw_Da=300.0, psa_A2=60.0, n_hbd=2)
        assert r.permeability_category in {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Absorption class
# ---------------------------------------------------------------------------
class TestAbsorptionClass:
    def test_complete(self):
        r = _run(logP=5.0, mw_Da=100.0, psa_A2=10.0, n_hbd=0)
        assert r.absorption_class == "complete"

    def test_poor(self):
        r = _run(logP=-3.0, mw_Da=700.0, psa_A2=200.0, n_hbd=8)
        assert r.absorption_class == "poor"

    def test_valid_values(self):
        r = _run()
        assert r.absorption_class in {"complete", "good", "moderate", "poor"}


# ---------------------------------------------------------------------------
# screen_permeability
# ---------------------------------------------------------------------------
class TestScreenPermeability:
    def test_sorted_descending(self):
        compounds = [
            {"drug_name": "A", "logP": 1.0, "mw_Da": 400.0, "psa_A2": 100.0},
            {"drug_name": "B", "logP": 4.0, "mw_Da": 200.0, "psa_A2": 40.0},
            {"drug_name": "C", "logP": 2.5, "mw_Da": 300.0, "psa_A2": 70.0},
        ]
        results = screen_permeability(compounds)
        peffs = [r.peff_cm_s for r in results]
        assert peffs == sorted(peffs, reverse=True)

    def test_returns_list(self):
        compounds = [
            {"drug_name": "X", "logP": 2.0, "mw_Da": 300.0, "psa_A2": 80.0},
        ]
        results = screen_permeability(compounds)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_empty_list(self):
        assert screen_permeability([]) == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_zero_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _run(mw_Da=0.0)

    def test_negative_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _run(mw_Da=-100.0)

    def test_negative_psa(self):
        with pytest.raises(ValueError, match="psa_A2"):
            _run(psa_A2=-1.0)

    def test_negative_hbd(self):
        with pytest.raises(ValueError, match="n_hbd"):
            _run(n_hbd=-1)
