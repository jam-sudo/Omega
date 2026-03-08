"""Tests for Phase 544: Volume of Distribution Determinants."""

import math

import pytest

from omega_pbpk.prediction.vd_determinants import (
    VdDeterminantsResult,
    predict_vd_determinants,
    screen_vd_prediction,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

_VP_L = 3.0  # plasma volume minimum


def _call(**kw):
    defaults = dict(
        drug_name="TestDrug",
        logP=1.0,
        mw_Da=300.0,
        pKa=7.4,
        drug_type="neutral",
        fup=0.1,
    )
    defaults.update(kw)
    return predict_vd_determinants(**defaults)


# ── Return type & fields ──────────────────────────────────────────────────────


class TestReturnType:
    def test_returns_dataclass(self):
        r = _call()
        assert isinstance(r, VdDeterminantsResult)

    def test_is_frozen(self):
        r = _call()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "other"  # type: ignore

    def test_has_all_fields(self):
        r = _call()
        for field in [
            "drug_name",
            "logP",
            "mw_Da",
            "pKa",
            "drug_type",
            "fup",
            "fut",
            "kpu_tissue",
            "vd_predicted_L",
            "vd_class",
            "distribution_pattern",
            "lipophilicity_contribution_pct",
            "notes",
        ]:
            assert hasattr(r, field)

    def test_drug_name_preserved(self):
        r = predict_vd_determinants("Warfarin", 2.6, 308.0, 5.0, "acid", 0.01)
        assert r.drug_name == "Warfarin"


# ── Physicochemical correctness ───────────────────────────────────────────────


class TestPhysicochemicalCorrectness:
    def test_high_logP_raises_fut(self):
        """High logP → lower fut (more tissue binding)."""
        r_low = _call(logP=0.0)
        r_high = _call(logP=5.0)
        assert r_high.fut < r_low.fut

    def test_high_logP_raises_kpu(self):
        """High logP → lower fut → higher Kpu."""
        r_low = _call(logP=0.0)
        r_high = _call(logP=5.0)
        assert r_high.kpu_tissue > r_low.kpu_tissue

    def test_high_logP_raises_vd(self):
        """High logP → higher Vd."""
        r_low = _call(logP=0.0)
        r_high = _call(logP=5.0)
        assert r_high.vd_predicted_L > r_low.vd_predicted_L

    def test_high_fup_raises_kpu(self):
        """High fup → higher Kpu (Kpu = fup / fut).

        Because fut depends only on logP (not fup), Kpu = fup/fut scales
        linearly with fup. Higher fup means more drug is unbound in plasma
        and thus a higher tissue-to-unbound-plasma partition ratio.
        """
        r_low_fup = _call(fup=0.01)
        r_high_fup = _call(fup=0.5)
        assert r_high_fup.kpu_tissue > r_low_fup.kpu_tissue

    def test_high_fup_raises_vd(self):
        """High fup → higher Vd (Kpu = fup/fut, Vd = Vp + Vt*Kpu)."""
        r_low_fup = _call(fup=0.01)
        r_high_fup = _call(fup=0.5)
        assert r_high_fup.vd_predicted_L > r_low_fup.vd_predicted_L

    def test_vd_minimum_is_plasma_volume(self):
        """Vd must be at least Vp = 3 L."""
        r = _call(logP=-5.0, fup=1.0)  # very hydrophilic, free drug
        assert r.vd_predicted_L >= _VP_L

    def test_vd_maximum_clamped(self):
        """Vd is clamped at 1000 L."""
        r = _call(logP=10.0, fup=0.001)
        assert r.vd_predicted_L <= 1000.0

    def test_fut_clamped_lower(self):
        """fut is clamped to at least 0.01."""
        r = _call(logP=100.0)  # extreme lipophilicity
        assert r.fut >= 0.01

    def test_fut_clamped_upper(self):
        """fut is clamped to at most 1.0."""
        r = _call(logP=-100.0)  # extreme hydrophilicity
        assert r.fut <= 1.0

    def test_kpu_equals_fup_over_fut(self):
        r = _call(logP=2.0, fup=0.1)
        expected_kpu = r.fup / r.fut
        assert math.isclose(r.kpu_tissue, expected_kpu, rel_tol=1e-9)


# ── vd_class classification ───────────────────────────────────────────────────


class TestVdClass:
    def test_very_low_class(self):
        # fup=0.1, logP=-2 → fut=1.0, Kpu=0.1, Vd=3+38*0.1=6.8 → very_low
        r = _call(logP=-2.0, fup=0.1)
        assert r.vd_class == "very_low"

    def test_low_class(self):
        r = _call(logP=0.0, fup=0.5)
        # Vd = 3 + 38 * (0.5/1.0) = 3 + 19 = 22 → "low"
        assert r.vd_class == "low"

    def test_moderate_class(self):
        # fup=0.5, logP=3: fut=1/(1+1.5)=0.4, Kpu=0.5/0.4=1.25, Vd=3+47.5=50.5 → moderate
        r = _call(logP=3.0, fup=0.5)
        assert r.vd_class == "moderate"

    def test_high_class(self):
        # fup=1.0, logP=9: fut=1/(1+4.5)=0.182, Kpu=5.5, Vd=3+209=212 → high
        r2 = _call(logP=9.0, fup=1.0)
        assert r2.vd_class == "high"

    def test_boundary_very_low_to_low(self):
        """Vd just below 10 → very_low; above 10 → low."""
        # Vd=3+38*Kpu. Kpu=(10-3)/38=0.184=fup/fut
        # fup=0.5,fut=0.5/0.184=2.72 → impossible (fut>1)
        # use fup=0.184 (and logP=0 → fut=1.0): Kpu=0.184, Vd=3+6.99=9.99→very_low
        r_below = predict_vd_determinants(
            "X", logP=0.0, mw_Da=200.0, pKa=7.0, drug_type="neutral", fup=0.183
        )
        r_above = predict_vd_determinants(
            "X", logP=0.0, mw_Da=200.0, pKa=7.0, drug_type="neutral", fup=0.2
        )
        assert r_below.vd_class == "very_low"
        assert r_above.vd_class == "low"


# ── distribution_pattern ──────────────────────────────────────────────────────


class TestDistributionPattern:
    def test_plasma_confined(self):
        # logP=-2 → fut=1.0, fup=0.1 → Kpu=0.1, Vd=3+3.8=6.8 L → plasma_confined
        r = predict_vd_determinants(
            "X", logP=-2.0, mw_Da=200.0, pKa=7.0, drug_type="neutral", fup=0.1
        )
        assert r.distribution_pattern == "plasma_confined"

    def test_tissue_distributed(self):
        r = predict_vd_determinants(
            "X", logP=0.0, mw_Da=200.0, pKa=7.0, drug_type="neutral", fup=0.3
        )
        # Kpu=0.3, Vd=3+11.4=14.4→low→tissue_distributed
        assert r.distribution_pattern == "tissue_distributed"

    def test_highly_tissue_distributed(self):
        r = predict_vd_determinants(
            "X", logP=9.0, mw_Da=200.0, pKa=7.0, drug_type="neutral", fup=1.0
        )
        assert r.distribution_pattern == "highly_tissue_distributed"


# ── lipophilicity_contribution_pct ────────────────────────────────────────────


class TestLipophilicityContribution:
    def test_positive_contribution(self):
        r = _call(logP=3.0, fup=0.5)
        assert r.lipophilicity_contribution_pct > 0.0

    def test_contribution_le_100(self):
        r = _call(logP=5.0)
        assert r.lipophilicity_contribution_pct <= 100.0

    def test_contribution_ge_0(self):
        r = _call(logP=-5.0, fup=1.0)
        assert r.lipophilicity_contribution_pct >= 0.0


# ── screen_vd_prediction ──────────────────────────────────────────────────────


class TestScreenVdPrediction:
    # Design: Kpu=fup/fut, Vd=3+38*Kpu
    # LipophilicDrug: logP=5 → fut=1/(1+2.5)=0.286, fup=0.9 → Kpu=3.15, Vd=123 L
    # ModerateDrug: logP=2 → fut=1/(1+1)=0.5, fup=0.3 → Kpu=0.6, Vd=25.8 L
    # HydrophilicDrug: logP=-1 → fut=1.0, fup=0.05 → Kpu=0.05, Vd=4.9 L
    _COMPOUNDS = [
        {
            "drug_name": "LipophilicDrug",
            "logP": 5.0,
            "mw_Da": 400.0,
            "pKa": 8.0,
            "drug_type": "base",
            "fup": 0.9,
        },
        {
            "drug_name": "HydrophilicDrug",
            "logP": -1.0,
            "mw_Da": 200.0,
            "pKa": 3.0,
            "drug_type": "acid",
            "fup": 0.05,
        },
        {
            "drug_name": "ModerateDrug",
            "logP": 2.0,
            "mw_Da": 300.0,
            "pKa": 7.0,
            "drug_type": "neutral",
            "fup": 0.3,
        },
    ]

    def test_returns_list(self):
        results = screen_vd_prediction(self._COMPOUNDS)
        assert isinstance(results, list)

    def test_length(self):
        results = screen_vd_prediction(self._COMPOUNDS)
        assert len(results) == 3

    def test_sorted_by_vd_descending(self):
        results = screen_vd_prediction(self._COMPOUNDS)
        vds = [r.vd_predicted_L for r in results]
        assert vds == sorted(vds, reverse=True)

    def test_all_result_objects(self):
        results = screen_vd_prediction(self._COMPOUNDS)
        for r in results:
            assert isinstance(r, VdDeterminantsResult)

    def test_lipophilic_first(self):
        results = screen_vd_prediction(self._COMPOUNDS)
        assert results[0].drug_name == "LipophilicDrug"

    def test_hydrophilic_last(self):
        results = screen_vd_prediction(self._COMPOUNDS)
        assert results[-1].drug_name == "HydrophilicDrug"

    def test_optional_fup_default(self):
        """screen should work without explicit fup (uses default 0.1)."""
        cmpds = [
            {"drug_name": "A", "logP": 1.0, "mw_Da": 300.0, "pKa": 7.0, "drug_type": "neutral"},
        ]
        results = screen_vd_prediction(cmpds)
        assert len(results) == 1
        assert math.isclose(results[0].fup, 0.1, rel_tol=1e-9)


# ── Validation errors ─────────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_drug_type(self):
        with pytest.raises(ValueError, match="drug_type"):
            _call(drug_type="zwitterion")

    def test_zero_fup(self):
        with pytest.raises(ValueError, match="fup"):
            _call(fup=0.0)

    def test_negative_fup(self):
        with pytest.raises(ValueError, match="fup"):
            _call(fup=-0.1)

    def test_fup_above_one(self):
        with pytest.raises(ValueError, match="fup"):
            _call(fup=1.1)

    def test_zero_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _call(mw_Da=0.0)

    def test_negative_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _call(mw_Da=-100.0)
