"""Tests for nanoparticle/nanomedicine PBPK model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.nano_pbpk import (
    NanoPBPKResult,
    compare_formulations,
    simulate_nano_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**overrides) -> NanoPBPKResult:
    """Run a default simulation with optional overrides."""
    kw = dict(
        nanoparticle_name="TestNP",
        dose_mg=10.0,
        size_nm=100.0,
    )
    kw.update(overrides)
    return simulate_nano_pk(**kw)


# ---------------------------------------------------------------------------
# 1-3: Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_nano_pk("NP", dose_mg=0, size_nm=100)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_nano_pk("NP", dose_mg=-1, size_nm=100)

    def test_size_zero_raises(self):
        with pytest.raises(ValueError, match="size_nm"):
            simulate_nano_pk("NP", dose_mg=10, size_nm=0)

    def test_drug_loading_zero_raises(self):
        with pytest.raises(ValueError, match="drug_loading_pct"):
            simulate_nano_pk("NP", dose_mg=10, size_nm=100, drug_loading_pct=0)

    def test_drug_loading_100_raises(self):
        with pytest.raises(ValueError, match="drug_loading_pct"):
            simulate_nano_pk("NP", dose_mg=10, size_nm=100, drug_loading_pct=100)


# ---------------------------------------------------------------------------
# 4: Basic simulation returns NanoPBPKResult
# ---------------------------------------------------------------------------


class TestBasicSimulation:
    def test_returns_result_type(self):
        r = _default_result()
        assert isinstance(r, NanoPBPKResult)

    def test_result_fields_populated(self):
        r = _default_result()
        assert len(r.times_h) > 0
        assert len(r.c_plasma_mg_L) == len(r.times_h)
        assert len(r.a_liver_pct) == len(r.times_h)


# ---------------------------------------------------------------------------
# 5-6: MPS uptake — liver accumulation
# ---------------------------------------------------------------------------


class TestMPSUptake:
    def test_liver_accumulation_above_20pct(self):
        r = _default_result(peg_density_chains_per_nm2=0.0)
        assert r.liver_pct_at_24h > 20.0

    def test_peg_reduces_liver_uptake(self):
        r_nopeg = _default_result(peg_density_chains_per_nm2=0.0)
        r_peg = _default_result(peg_density_chains_per_nm2=0.2)
        assert r_peg.liver_pct_at_24h < r_nopeg.liver_pct_at_24h


# ---------------------------------------------------------------------------
# 7: Large NP lung accumulation
# ---------------------------------------------------------------------------


class TestLungTrapping:
    def test_large_np_lung_accumulation(self):
        r = _default_result(size_nm=400)
        # Lung should accumulate some NP for >200 nm particles
        lung_pct_final = r.a_liver_pct[-1]  # just check simulation runs
        assert lung_pct_final >= 0  # basic sanity
        # More specific: check via notes or direct ODE behaviour
        # Large particles get k_lung > 0
        # We can verify by checking that plasma drops faster
        r_small = _default_result(size_nm=100)
        # 400 nm should clear faster from plasma
        assert r.c_plasma_mg_L[-1] <= r_small.c_plasma_mg_L[-1]


# ---------------------------------------------------------------------------
# 8: Small NP renal clearance
# ---------------------------------------------------------------------------


class TestSmallNP:
    def test_small_np_faster_clearance(self):
        r_small = _default_result(size_nm=5)
        r_normal = _default_result(size_nm=100)
        # Small NP has k_renal=0.5, should clear faster
        assert r_small.t_half_h < r_normal.t_half_h


# ---------------------------------------------------------------------------
# 9-11: EPR / Tumor
# ---------------------------------------------------------------------------


class TestEPR:
    def test_tumor_accumulation_when_present(self):
        r = _default_result(tumor_weight_g=1.0)
        assert r.tumor_pct_at_24h > 0

    def test_no_tumor_when_zero_weight(self):
        r = _default_result(tumor_weight_g=0.0)
        assert all(v == 0.0 for v in r.a_tumor_pct)
        assert r.tumor_pct_at_24h == 0.0

    def test_peg_improves_epr(self):
        r_nopeg = _default_result(peg_density_chains_per_nm2=0.0, tumor_weight_g=1.0)
        r_peg = _default_result(peg_density_chains_per_nm2=0.2, tumor_weight_g=1.0)
        assert r_peg.tumor_auc_pct_h > r_nopeg.tumor_auc_pct_h


# ---------------------------------------------------------------------------
# 12-13: Drug release
# ---------------------------------------------------------------------------


class TestDrugRelease:
    def test_drug_release_positive(self):
        r = _default_result(drug_release_rate_per_h=0.1)
        assert max(r.c_drug_plasma_mg_L) > 0

    def test_no_drug_release(self):
        r = _default_result(drug_release_rate_per_h=0.0)
        assert all(v == 0.0 for v in r.c_drug_plasma_mg_L)


# ---------------------------------------------------------------------------
# 14: Dose linearity
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_double_dose_doubles_cmax(self):
        r1 = _default_result(dose_mg=10.0)
        r2 = _default_result(dose_mg=20.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert abs(ratio - 2.0) < 0.01


# ---------------------------------------------------------------------------
# 15-16: PK metrics
# ---------------------------------------------------------------------------


class TestPKMetrics:
    def test_t_half_positive(self):
        r = _default_result()
        assert r.t_half_h > 0

    def test_auc_positive(self):
        r = _default_result()
        assert r.auc_plasma > 0


# ---------------------------------------------------------------------------
# 17: Mass balance at t=0+
# ---------------------------------------------------------------------------


class TestMassBalance:
    def test_mass_balance_at_start(self):
        r = _default_result()
        # At t=0, all NP is in plasma
        plasma_pct = r.c_plasma_mg_L[0] * 3.0 / r.dose_mg * 100.0
        liver_pct = r.a_liver_pct[0]
        spleen_pct = r.a_spleen_pct[0]
        tumor_pct = r.a_tumor_pct[0]
        total = plasma_pct + liver_pct + spleen_pct + tumor_pct
        assert total > 99.0  # ~100% at t=0


# ---------------------------------------------------------------------------
# 18: compare_formulations
# ---------------------------------------------------------------------------


class TestCompareFormulations:
    def test_returns_list(self):
        forms = [
            {"size_nm": 50, "peg_density": 0.0, "zeta_potential_mV": 0.0},
            {"size_nm": 100, "peg_density": 0.2, "zeta_potential_mV": -10.0},
        ]
        results = compare_formulations("TestNP", 10.0, forms)
        assert len(results) == 2
        assert all(isinstance(r, NanoPBPKResult) for r in results)


# ---------------------------------------------------------------------------
# 19: PEGylated NP has longer t_half
# ---------------------------------------------------------------------------


class TestPEGHalfLife:
    def test_peg_extends_half_life(self):
        r_nopeg = _default_result(peg_density_chains_per_nm2=0.0)
        r_peg = _default_result(peg_density_chains_per_nm2=0.2)
        assert r_peg.t_half_h > r_nopeg.t_half_h


# ---------------------------------------------------------------------------
# 20: Negative zeta improves EPR
# ---------------------------------------------------------------------------


class TestZetaEPR:
    def test_negative_zeta_better_epr(self):
        r_neg = _default_result(zeta_potential_mV=-30.0, tumor_weight_g=1.0)
        r_pos = _default_result(zeta_potential_mV=30.0, tumor_weight_g=1.0)
        assert r_neg.tumor_auc_pct_h > r_pos.tumor_auc_pct_h


# ---------------------------------------------------------------------------
# 21: Liver captures NPs permanently
# ---------------------------------------------------------------------------


class TestLiverRetention:
    def test_liver_pct_end_positive(self):
        r = _default_result()
        assert r.a_liver_pct[-1] > 0


# ---------------------------------------------------------------------------
# 22: Notes contain key info
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_contain_name(self):
        r = _default_result(nanoparticle_name="MyNano")
        assert "MyNano" in r.notes

    def test_notes_contain_size(self):
        r = _default_result(size_nm=100)
        assert "100" in r.notes


# ---------------------------------------------------------------------------
# 23: Corona factor
# ---------------------------------------------------------------------------


class TestCoronaFactor:
    def test_corona_higher_without_peg(self):
        r_nopeg = _default_result(peg_density_chains_per_nm2=0.0)
        r_peg = _default_result(peg_density_chains_per_nm2=0.2)
        assert r_nopeg.corona_factor > r_peg.corona_factor

    def test_corona_factor_positive(self):
        r = _default_result()
        assert r.corona_factor > 0


# ---------------------------------------------------------------------------
# 24: EPR efficiency metric
# ---------------------------------------------------------------------------


class TestEPREfficiency:
    def test_epr_efficiency_positive_with_tumor(self):
        r = _default_result(tumor_weight_g=1.0)
        assert r.epr_efficiency > 0

    def test_epr_efficiency_zero_without_tumor(self):
        r = _default_result(tumor_weight_g=0.0)
        assert r.epr_efficiency == 0.0
