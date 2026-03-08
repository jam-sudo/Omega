"""Tests for Phase 608 — Drug Volatility and Inhalation PK."""

import math

import pytest

from omega_pbpk.core.inhalation_pk_p608 import (
    InhalationPKResult,
    _ka_lung_from_logp,
    _lung_deposition_fraction,
    compare_particle_sizes,
    simulate_inhalation_pk,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_result(**kwargs):
    """Build an InhalationPKResult with sensible defaults."""
    defaults = dict(
        drug_name="Salbutamol",
        dose_mg=2.5,
        particle_size_um=3.0,
        logP=0.5,
        cl_sys_L_per_h=15.0,
        vd_sys_L=180.0,
        lung_weight_g=1000.0,
        t_end_h=24.0,
        dt_h=0.05,
    )
    defaults.update(kwargs)
    return simulate_inhalation_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_inhalation_pk_result(self):
        r = make_result()
        assert isinstance(r, InhalationPKResult)

    def test_result_is_mutable(self):
        r = make_result()
        # Plain @dataclass — should be mutable
        r.drug_name = "Modified"
        assert r.drug_name == "Modified"

    def test_all_fields_present(self):
        r = make_result()
        for field in (
            "drug_name",
            "dose_mg",
            "particle_size_um",
            "times_h",
            "c_lung_mg_g",
            "c_systemic_mg_L",
            "cmax_lung_mg_g",
            "tmax_lung_h",
            "cmax_systemic_mg_L",
            "tmax_systemic_h",
            "auc_lung_mg_h_per_g",
            "auc_systemic_mg_h_per_L",
            "lung_deposition_pct",
            "t_half_lung_h",
            "bioavailability_pct",
            "notes",
        ):
            assert hasattr(r, field), f"Missing field: {field}"

    def test_times_is_list(self):
        r = make_result()
        assert isinstance(r.times_h, list)

    def test_c_lung_is_list(self):
        r = make_result()
        assert isinstance(r.c_lung_mg_g, list)

    def test_c_systemic_is_list(self):
        r = make_result()
        assert isinstance(r.c_systemic_mg_L, list)

    def test_equal_length_profiles(self):
        r = make_result()
        assert len(r.times_h) == len(r.c_lung_mg_g) == len(r.c_systemic_mg_L)

    def test_times_start_at_zero(self):
        r = make_result()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_drug_name_stored(self):
        r = make_result(drug_name="Fluticasone")
        assert r.drug_name == "Fluticasone"


# ---------------------------------------------------------------------------
# Deposition by particle size
# ---------------------------------------------------------------------------


class TestDepositionByParticleSize:
    def test_ultrafine_deposition(self):
        assert _lung_deposition_fraction(0.5) == pytest.approx(0.10)

    def test_optimal_alveolar_deposition(self):
        assert _lung_deposition_fraction(3.0) == pytest.approx(0.60)

    def test_tracheobronchial_deposition(self):
        assert _lung_deposition_fraction(7.0) == pytest.approx(0.40)

    def test_oropharyngeal_deposition(self):
        assert _lung_deposition_fraction(15.0) == pytest.approx(0.15)

    def test_boundary_1um_alveolar(self):
        # Exactly 1 µm is in [1, 5] range → 60%
        assert _lung_deposition_fraction(1.0) == pytest.approx(0.60)

    def test_boundary_5um_alveolar(self):
        # Exactly 5 µm is in [1, 5] range → 60%
        assert _lung_deposition_fraction(5.0) == pytest.approx(0.60)

    def test_lung_deposition_pct_reflected_in_result(self):
        r3 = make_result(particle_size_um=3.0)
        r15 = make_result(particle_size_um=15.0)
        assert r3.lung_deposition_pct > r15.lung_deposition_pct

    def test_lung_deposition_pct_range(self):
        for sz in [0.5, 3.0, 7.0, 15.0]:
            r = make_result(particle_size_um=sz)
            assert 0.0 < r.lung_deposition_pct <= 100.0


# ---------------------------------------------------------------------------
# logP effect on absorption rate
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_ka_lung_increases_with_logp(self):
        ka_low = _ka_lung_from_logp(-2.0)
        ka_high = _ka_lung_from_logp(5.0)
        assert ka_high > ka_low

    def test_ka_lung_clamped_minimum(self):
        ka = _ka_lung_from_logp(-100.0)
        assert ka == pytest.approx(0.1)

    def test_ka_lung_clamped_maximum(self):
        ka = _ka_lung_from_logp(100.0)
        assert ka == pytest.approx(5.0)

    def test_high_logp_higher_systemic_absorption(self):
        r_low = make_result(logP=-2.0)
        r_high = make_result(logP=4.0)
        # Higher logP → faster absorption → higher peak systemic
        assert r_high.cmax_systemic_mg_L > r_low.cmax_systemic_mg_L


# ---------------------------------------------------------------------------
# Systemic AUC and PK metrics
# ---------------------------------------------------------------------------


class TestSystemicPK:
    def test_systemic_auc_positive(self):
        r = make_result()
        assert r.auc_systemic_mg_h_per_L > 0.0

    def test_lung_auc_positive(self):
        r = make_result()
        assert r.auc_lung_mg_h_per_g > 0.0

    def test_cmax_systemic_positive(self):
        r = make_result()
        assert r.cmax_systemic_mg_L > 0.0

    def test_cmax_lung_at_t0(self):
        r = make_result()
        # Lung concentration starts at max (purely eliminates after t=0)
        assert r.c_lung_mg_g[0] == pytest.approx(r.cmax_lung_mg_g, rel=1e-3)

    def test_tmax_systemic_after_zero(self):
        r = make_result()
        assert r.tmax_systemic_h > 0.0

    def test_higher_dose_proportional_auc(self):
        r1 = make_result(dose_mg=1.0)
        r5 = make_result(dose_mg=5.0)
        ratio = r5.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
        assert ratio == pytest.approx(5.0, rel=0.05)


# ---------------------------------------------------------------------------
# t_half_lung
# ---------------------------------------------------------------------------


class TestTHalfLung:
    def test_t_half_lung_positive(self):
        r = make_result()
        assert r.t_half_lung_h > 0.0

    def test_t_half_lung_formula(self):
        logP = 0.5
        ka = _ka_lung_from_logp(logP)
        k_mcc = 0.3
        expected_t_half = math.log(2) / (ka + k_mcc)
        r = make_result(logP=logP)
        assert r.t_half_lung_h == pytest.approx(expected_t_half, rel=1e-6)


# ---------------------------------------------------------------------------
# Bioavailability
# ---------------------------------------------------------------------------


class TestBioavailability:
    def test_bioavailability_in_range(self):
        r = make_result()
        assert 0.0 <= r.bioavailability_pct <= 100.0

    def test_higher_deposition_higher_bioavailability(self):
        r_alveolar = make_result(particle_size_um=3.0)  # 60% dep
        r_oropharyngeal = make_result(particle_size_um=15.0)  # 15% dep
        assert r_alveolar.bioavailability_pct >= r_oropharyngeal.bioavailability_pct


# ---------------------------------------------------------------------------
# Particle size comparison
# ---------------------------------------------------------------------------


class TestParticleSizeComparison:
    def test_compare_returns_list(self):
        results = compare_particle_sizes("Salbutamol", 2.5, 0.5, 15.0, 180.0)
        assert isinstance(results, list)

    def test_compare_default_4_sizes(self):
        results = compare_particle_sizes("Salbutamol", 2.5, 0.5, 15.0, 180.0)
        assert len(results) == 4

    def test_compare_custom_sizes(self):
        results = compare_particle_sizes("Drug", 1.0, 1.0, 10.0, 100.0, sizes=[2.0, 5.0])
        assert len(results) == 2

    def test_compare_sorted_by_auc_desc(self):
        results = compare_particle_sizes("Salbutamol", 2.5, 0.5, 15.0, 180.0)
        aucs = [r.auc_systemic_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_compare_best_is_alveolar_range(self):
        results = compare_particle_sizes("Drug", 2.5, 0.5, 15.0, 180.0)
        # Best (highest systemic AUC) should be alveolar range (1-5 µm)
        assert results[0].lung_deposition_pct == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError):
            make_result(dose_mg=0.0)

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError):
            make_result(dose_mg=-1.0)

    def test_invalid_particle_size_zero(self):
        with pytest.raises(ValueError):
            make_result(particle_size_um=0.0)

    def test_invalid_cl_sys_zero(self):
        with pytest.raises(ValueError):
            make_result(cl_sys_L_per_h=0.0)

    def test_invalid_vd_sys_zero(self):
        with pytest.raises(ValueError):
            make_result(vd_sys_L=0.0)

    def test_invalid_lung_weight_zero(self):
        with pytest.raises(ValueError):
            make_result(lung_weight_g=0.0)

    def test_notes_not_empty(self):
        r = make_result()
        assert len(r.notes) > 0
