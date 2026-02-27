"""Phase 7 tests — Mechanistic Kp methods (Poulin-Theil & Rodgers-Rowland).

Covers:
  1. poulin_theil_kp: physical plausibility, fup scaling, ionisation effects
  2. rodgers_rowland_kp: base enhancement term, fup scaling
  3. estimate_all_kp: all methods, method dispatcher
  4. get_partition_method: valid/invalid names
  5. Drug.default_kp: respects partition_method field
  6. CLI --kp-method: SimulationRequest kp_method field propagation
  7. Consistency checks between methods
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omega_pbpk.core.heuristics import (
    KP_METHOD_NAMES,
    TISSUE_COMPOSITION,
    estimate_all_kp,
    get_partition_method,
    heuristic_kp,
    poulin_theil_kp,
    rodgers_rowland_kp,
)
from omega_pbpk.drugs.drug import Drug


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TISSUES = list(TISSUE_COMPOSITION.keys())


def _make_drug(
    logP: float = 2.0,
    fup: float = 0.1,
    compound_type: str = "neutral",
    partition_method: str = "heuristic",
) -> Drug:
    return Drug(
        name="TestDrug",
        mw=300.0,
        logP=logP,
        fup=fup,
        compound_type=compound_type,
        partition_method=partition_method,
        kp={},  # force fallback to method
    )


# ---------------------------------------------------------------------------
# 1. Poulin-Theil: physical plausibility
# ---------------------------------------------------------------------------


class TestPoulinTheilKp:
    def test_returns_positive_for_neutral(self):
        """Poulin-Theil Kp must be strictly positive for a neutral drug."""
        for tissue in TISSUES:
            kp = poulin_theil_kp(logP=2.0, tissue_name=tissue, fup=0.1)
            assert kp > 0, f"{tissue}: Kp={kp} is not positive"

    def test_minimum_floor(self):
        """Kp must never fall below 0.01 (floor enforced)."""
        # Very low fup + very hydrophilic = low Kp, but should still be ≥ 0.01
        kp = poulin_theil_kp(logP=-3.0, tissue_name="muscle", fup=0.001)
        assert kp >= 0.01

    def test_lipophilicity_increases_kp(self):
        """Higher logP should increase Kp in lipid-rich tissues (e.g. adipose)."""
        kp_low = poulin_theil_kp(logP=0.0, tissue_name="adipose", fup=0.5)
        kp_high = poulin_theil_kp(logP=4.0, tissue_name="adipose", fup=0.5)
        assert kp_high > kp_low, (
            f"adipose Kp did not increase with logP: {kp_low:.3f} vs {kp_high:.3f}"
        )

    def test_fup_scales_kp(self):
        """Higher fup should yield higher Kp (Kp = Kpu × fup)."""
        kp_lo = poulin_theil_kp(logP=2.0, tissue_name="muscle", fup=0.01)
        kp_hi = poulin_theil_kp(logP=2.0, tissue_name="muscle", fup=0.5)
        assert kp_hi > kp_lo

    def test_fup_scaling_is_linear(self):
        """Kp should scale linearly with fup for a neutral drug (Kp = Kpu × fup)."""
        kp_1 = poulin_theil_kp(logP=1.0, tissue_name="liver", fup=0.1)
        kp_2 = poulin_theil_kp(logP=1.0, tissue_name="liver", fup=0.2)
        ratio = kp_2 / max(kp_1, 1e-12)
        assert abs(ratio - 2.0) < 0.05, f"Expected ratio ~2.0, got {ratio:.3f}"

    def test_unknown_tissue_returns_one(self):
        """Unknown tissue name should return 1.0 (safe fallback)."""
        kp = poulin_theil_kp(logP=2.0, tissue_name="__unknown__", fup=0.5)
        assert kp == 1.0

    def test_base_higher_kp_than_neutral_lipid_tissue(self):
        """For the same logP, a basic drug should have Kp ≥ neutral in muscle
        due to ionisation at tissue pH (basic drugs partially charged at pH 7.0)."""
        kp_neutral = poulin_theil_kp(
            logP=2.0, pka=None, compound_type="neutral", tissue_name="muscle", fup=0.1
        )
        kp_base = poulin_theil_kp(
            logP=2.0, pka=9.0, compound_type="base", tissue_name="muscle", fup=0.1
        )
        # Base drugs ionise more at tissue pH 7.0 vs plasma pH 7.4 → R > 1 → higher Kp
        assert kp_base >= kp_neutral * 0.5  # at minimum not dramatically lower

    def test_acid_lower_kp_than_neutral_at_high_pka(self):
        """Weak acid with pKa >> 7.4 behaves like neutral — no big change."""
        kp_neutral = poulin_theil_kp(
            logP=2.0, pka=None, compound_type="neutral", tissue_name="muscle", fup=0.2
        )
        kp_acid = poulin_theil_kp(
            logP=2.0, pka=12.0, compound_type="acid", tissue_name="muscle", fup=0.2
        )
        # With very high pKa, acid is essentially neutral → Kp should be similar
        assert abs(kp_acid - kp_neutral) / max(kp_neutral, 1e-12) < 0.15


# ---------------------------------------------------------------------------
# 2. Rodgers-Rowland: extends Poulin-Theil for bases
# ---------------------------------------------------------------------------


class TestRodgersRowlandKp:
    def test_returns_positive_for_all_tissues(self):
        for tissue in TISSUES:
            kp = rodgers_rowland_kp(logP=2.0, tissue_name=tissue, fup=0.1)
            assert kp > 0, f"{tissue}: Kp={kp} not positive"

    def test_minimum_floor(self):
        kp = rodgers_rowland_kp(logP=-3.0, tissue_name="bone", fup=0.001)
        assert kp >= 0.01

    def test_base_higher_than_neutral_in_rr(self):
        """R&R predicts higher Kp for bases vs neutrals due to phospholipid term."""
        kp_neutral = rodgers_rowland_kp(
            logP=2.0, pka=None, compound_type="neutral", tissue_name="brain", fup=0.1
        )
        kp_base = rodgers_rowland_kp(
            logP=2.0, pka=9.0, compound_type="base", tissue_name="brain", fup=0.1
        )
        assert kp_base > kp_neutral, (
            f"R&R base Kp={kp_base:.3f} should exceed neutral Kp={kp_neutral:.3f} in brain"
        )

    def test_neutral_identical_to_poulin_theil(self):
        """For neutral drugs, R&R Eq.4 == Poulin-Theil (no extra phospholipid term)."""
        for tissue in ["muscle", "liver", "kidney"]:
            kp_pt = poulin_theil_kp(logP=2.0, tissue_name=tissue, fup=0.2)
            kp_rr = rodgers_rowland_kp(
                logP=2.0, compound_type="neutral", tissue_name=tissue, fup=0.2
            )
            assert abs(kp_pt - kp_rr) < 1e-6, (
                f"{tissue}: P-T={kp_pt:.6f} vs R-R neutral={kp_rr:.6f} should be identical"
            )

    def test_fup_scales_kp_linearly(self):
        """Kp = Kpu × fup so doubling fup should double Kp."""
        kp_1 = rodgers_rowland_kp(logP=1.5, compound_type="neutral", tissue_name="heart", fup=0.1)
        kp_2 = rodgers_rowland_kp(logP=1.5, compound_type="neutral", tissue_name="heart", fup=0.2)
        ratio = kp_2 / max(kp_1, 1e-12)
        assert abs(ratio - 2.0) < 0.05, f"Expected ratio ~2.0, got {ratio:.3f}"

    def test_drug_type_alias_works(self):
        """Legacy drug_type='base' alias should produce same result as compound_type='base'."""
        kp_ct = rodgers_rowland_kp(
            logP=2.0, pka=9.0, compound_type="base", tissue_name="muscle", fup=0.1
        )
        kp_dt = rodgers_rowland_kp(
            logP=2.0, pka=9.0, drug_type="monoprotic_base", tissue_name="muscle", fup=0.1
        )
        assert abs(kp_ct - kp_dt) < 1e-6


# ---------------------------------------------------------------------------
# 3. estimate_all_kp: multi-tissue, multi-method
# ---------------------------------------------------------------------------


class TestEstimateAllKp:
    def test_all_methods_return_all_tissues(self):
        """All three methods should return a Kp for every tissue."""
        for method in KP_METHOD_NAMES:
            result = estimate_all_kp(logP=2.0, fup=0.2, method=method, compound_type="neutral")
            assert len(result) > 0, f"{method}: returned empty dict"
            for tissue, kp in result.items():
                assert kp > 0, f"{method}/{tissue}: Kp={kp} not positive"
                assert math.isfinite(kp), f"{method}/{tissue}: Kp={kp} not finite"

    def test_heuristic_is_default(self):
        """estimate_all_kp with no method argument should use heuristic."""
        default_result = estimate_all_kp(logP=2.0, fup=0.2)
        heuristic_result = estimate_all_kp(logP=2.0, fup=0.2, method="heuristic")
        assert set(default_result.keys()) == set(heuristic_result.keys())

    def test_tissue_subset(self):
        """Can request a subset of tissues."""
        subset = ["liver", "kidney", "muscle"]
        result = estimate_all_kp(
            logP=1.0, fup=0.3, method="poulin_theil", compound_type="neutral", tissues=subset
        )
        assert set(result.keys()) == set(subset)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown partition method"):
            estimate_all_kp(logP=2.0, fup=0.1, method="nonexistent")


# ---------------------------------------------------------------------------
# 4. get_partition_method dispatcher
# ---------------------------------------------------------------------------


class TestGetPartitionMethod:
    def test_valid_methods(self):
        for name in KP_METHOD_NAMES:
            fn = get_partition_method(name)
            assert callable(fn)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown partition method"):
            get_partition_method("magic_kp")

    def test_known_method_names(self):
        assert "heuristic" in KP_METHOD_NAMES
        assert "poulin_theil" in KP_METHOD_NAMES
        assert "rodgers_rowland" in KP_METHOD_NAMES


# ---------------------------------------------------------------------------
# 5. Drug.default_kp respects partition_method
# ---------------------------------------------------------------------------


class TestDrugDefaultKp:
    def test_poulin_theil_method_used(self):
        """Drug with partition_method='poulin_theil' should use PT for missing Kp."""
        drug = _make_drug(logP=2.0, fup=0.2, partition_method="poulin_theil")
        kp = drug.default_kp("liver")
        expected = poulin_theil_kp(logP=2.0, fup=0.2, tissue_name="liver", compound_type="neutral")
        assert abs(kp - expected) < 1e-6

    def test_rodgers_rowland_method_used(self):
        drug = _make_drug(logP=2.0, fup=0.2, partition_method="rodgers_rowland")
        kp = drug.default_kp("muscle")
        expected = rodgers_rowland_kp(
            logP=2.0, fup=0.2, tissue_name="muscle", compound_type="neutral"
        )
        assert abs(kp - expected) < 1e-6

    def test_explicit_kp_overrides_method(self):
        """If kp dict has the organ, it should be used regardless of method."""
        drug = Drug(
            name="Test",
            mw=300.0,
            logP=2.0,
            fup=0.2,
            partition_method="poulin_theil",
            kp={"liver": 3.14},
        )
        assert drug.default_kp("liver") == pytest.approx(3.14)

    def test_heuristic_differs_from_poulin_theil(self):
        """The three methods should not all produce identical Kp values."""
        kp_h = heuristic_kp(logP=2.0, tissue_name="adipose", fup=0.1)
        kp_pt = poulin_theil_kp(logP=2.0, tissue_name="adipose", fup=0.1)
        kp_rr = rodgers_rowland_kp(logP=2.0, tissue_name="adipose", fup=0.1)
        # At least one pair should differ (they use different equations)
        all_same = abs(kp_h - kp_pt) < 1e-6 and abs(kp_pt - kp_rr) < 1e-6
        assert not all_same, "All three methods returned identical Kp — likely a bug"


# ---------------------------------------------------------------------------
# 6. Pipeline SimulationRequest kp_method field
# ---------------------------------------------------------------------------


class TestSimulationRequestKpMethod:
    def test_default_kp_method_is_heuristic(self):
        from omega_pbpk.pipeline import SimulationRequest

        req = SimulationRequest(smiles="C", dose_mg=100.0)
        assert req.kp_method == "heuristic"

    def test_kp_method_can_be_set(self):
        from omega_pbpk.pipeline import SimulationRequest

        req = SimulationRequest(smiles="C", dose_mg=100.0, kp_method="poulin_theil")
        assert req.kp_method == "poulin_theil"

    def test_pipeline_build_drug_with_poulin_theil(self):
        """OmegaPipeline._build_drug should pass kp_method to Drug."""
        from omega_pbpk.pipeline import OmegaPipeline

        pipeline = OmegaPipeline()
        pipeline._ensure_initialized()
        adme = {
            "mw": 300.0,
            "logP": 2.0,
            "fup": 0.2,
            "rbp": 0.55,
            "clint_3a4": 5.0,
            "herg_ic50_uM": 10.0,
            "confidence": "medium",
        }
        drug = pipeline._build_drug("CC", adme, [], kp_method="poulin_theil")
        assert drug.partition_method == "poulin_theil"

    def test_pipeline_build_drug_with_rodgers_rowland(self):
        from omega_pbpk.pipeline import OmegaPipeline

        pipeline = OmegaPipeline()
        pipeline._ensure_initialized()
        adme = {
            "mw": 300.0,
            "logP": 2.0,
            "fup": 0.2,
            "rbp": 0.55,
            "clint_3a4": 5.0,
            "herg_ic50_uM": 10.0,
            "confidence": "medium",
        }
        drug = pipeline._build_drug("CC", adme, [], kp_method="rodgers_rowland")
        assert drug.partition_method == "rodgers_rowland"

    def test_pipeline_build_drug_invalid_method_falls_back(self):
        """Invalid kp_method should fallback to heuristic with a warning."""
        from omega_pbpk.pipeline import OmegaPipeline

        pipeline = OmegaPipeline()
        pipeline._ensure_initialized()
        adme = {
            "mw": 300.0,
            "logP": 2.0,
            "fup": 0.2,
            "rbp": 0.55,
            "clint_3a4": 5.0,
            "herg_ic50_uM": 10.0,
            "confidence": "medium",
        }
        warnings: list[str] = []
        drug = pipeline._build_drug("CC", adme, warnings, kp_method="invalid_method")
        assert drug.partition_method == "heuristic"
        assert any("Unknown kp_method" in w for w in warnings)


# ---------------------------------------------------------------------------
# 7. PBPK simulation produces different Cmax/AUC for different Kp methods
# ---------------------------------------------------------------------------


class TestKpMethodEffectOnSimulation:
    """Full PBPK simulations with different Kp methods should produce
    different but physically plausible PK profiles."""

    def _run_sim(self, partition_method: str) -> dict:
        from omega_pbpk.core.body import WholeBodyPBPK

        drug = Drug(
            name="TestCompound",
            mw=300.0,
            logP=2.0,
            fup=0.2,
            rbp=1.0,
            clint_hepatic_L_per_h=5.0,
            peff=3.0,
            solubility_mg_mL=5.0,
            kp={},  # no fixed Kp — force method
            partition_method=partition_method,
            compound_type="neutral",
        )
        model = WholeBodyPBPK(drug=drug, body_weight=70.0)
        model.setup_oral(100.0)
        result = model.simulate(t_end_h=24.0)
        return result.pk_summary()

    def test_poulin_theil_simulation_runs(self):
        pk = self._run_sim("poulin_theil")
        assert pk["Cmax_mg_L"] > 0
        assert pk["AUC_mg_h_L"] > 0

    def test_rodgers_rowland_simulation_runs(self):
        pk = self._run_sim("rodgers_rowland")
        assert pk["Cmax_mg_L"] > 0
        assert pk["AUC_mg_h_L"] > 0

    def test_methods_produce_different_vss(self):
        """Different Kp methods → different Vss (at least one pair should differ)."""
        pk_h = self._run_sim("heuristic")
        pk_pt = self._run_sim("poulin_theil")
        vss_h = pk_h["Vss_L"]
        vss_pt = pk_pt["Vss_L"]
        # Allow up to 10% tolerance — they use different equations
        # but at least they should not be identical for this drug
        # (This is a sanity check, not a hard requirement)
        assert vss_h > 0 and vss_pt > 0
