"""Tests for microbiome drug metabolism module (Phase 289)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.microbiome_metabolism import (
    MicrobiomePKResult,
    ProdrugActivationResult,
    antibiotic_disruption_effect,
    microbiome_metabolism_rate,
    prodrug_activation,
    simulate_microbiome_drug_fate,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _sim(**kw) -> MicrobiomePKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        ka_abs_per_h=1.0,
        vmax_gut_mg_per_h=10.0,
        km_gut_mg_L=5.0,
        cl_L_per_h=5.0,
        vd_L=30.0,
        microbiome_density=1.0,
        dt_h=0.1,
    )
    defaults.update(kw)
    return simulate_microbiome_drug_fate(**defaults)


def _prodrug(**kw) -> ProdrugActivationResult:
    defaults = dict(
        drug_name="ProDrug",
        prodrug_dose_mg=100.0,
        ka_abs_per_h=0.8,
        activation_vmax=8.0,
        activation_km=10.0,
        cl_L_per_h=4.0,
        vd_L=20.0,
        microbiome_density=1.0,
        dt_h=0.1,
    )
    defaults.update(kw)
    return prodrug_activation(**defaults)


# ---------------------------------------------------------------------------
# microbiome_metabolism_rate
# ---------------------------------------------------------------------------

class TestMicrobiomeMetabolismRate:
    def test_zero_concentration_returns_zero(self):
        assert microbiome_metabolism_rate(0.0, 10.0, 5.0) == 0.0

    def test_negative_concentration_returns_zero(self):
        assert microbiome_metabolism_rate(-1.0, 10.0, 5.0) == 0.0

    def test_at_km_returns_half_vmax(self):
        rate = microbiome_metabolism_rate(5.0, 10.0, 5.0, microbiome_density=1.0)
        assert abs(rate - 5.0) < 1e-6  # Vmax/2 = 5

    def test_density_zero_returns_zero(self):
        assert microbiome_metabolism_rate(100.0, 10.0, 5.0, microbiome_density=0.0) == 0.0

    def test_density_scales_linearly(self):
        r1 = microbiome_metabolism_rate(5.0, 10.0, 5.0, microbiome_density=1.0)
        r2 = microbiome_metabolism_rate(5.0, 10.0, 5.0, microbiome_density=0.5)
        assert abs(r2 - r1 / 2) < 1e-6

    def test_high_concentration_approaches_vmax(self):
        rate = microbiome_metabolism_rate(1000.0, 10.0, 1.0)
        assert rate > 9.9  # near Vmax=10


# ---------------------------------------------------------------------------
# simulate_microbiome_drug_fate
# ---------------------------------------------------------------------------

class TestSimulateMicrobiomeDrugFate:
    def test_returns_result_type(self):
        assert isinstance(_sim(), MicrobiomePKResult)

    def test_frozen_dataclass(self):
        r = _sim()
        with pytest.raises((AttributeError, TypeError)):
            r.cmax = 999.0  # type: ignore[misc]

    def test_times_and_plasma_same_length(self):
        r = _sim()
        assert len(r.times_h) == len(r.c_plasma)

    def test_cmax_positive(self):
        assert _sim().cmax > 0

    def test_auc_positive(self):
        assert _sim().auc > 0

    def test_f_metabolized_in_01(self):
        r = _sim()
        assert 0.0 <= r.f_metabolized_by_microbiome <= 1.0

    def test_no_microbiome_no_metabolism(self):
        r = _sim(microbiome_density=0.0)
        assert r.f_metabolized_by_microbiome == pytest.approx(0.0, abs=1e-6)

    def test_higher_density_more_metabolism(self):
        r_low = _sim(microbiome_density=0.1)
        r_high = _sim(microbiome_density=1.0)
        assert r_high.f_metabolized_by_microbiome > r_low.f_metabolized_by_microbiome

    def test_higher_density_lower_auc(self):
        """More microbiome metabolism reduces oral bioavailability."""
        r_low = _sim(microbiome_density=0.1)
        r_high = _sim(microbiome_density=1.0)
        assert r_high.auc < r_low.auc

    def test_drug_name_stored(self):
        r = _sim(drug_name="Aspirin")
        assert r.drug_name == "Aspirin"

    def test_microbiome_density_stored(self):
        r = _sim(microbiome_density=0.5)
        assert r.microbiome_density == pytest.approx(0.5)

    def test_validation_dose_negative(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=-1.0)

    def test_validation_ka_zero(self):
        with pytest.raises(ValueError, match="ka_abs_per_h"):
            _sim(ka_abs_per_h=0.0)

    def test_validation_density_out_of_range(self):
        with pytest.raises(ValueError, match="microbiome_density"):
            _sim(microbiome_density=1.5)

    def test_plasma_non_negative(self):
        r = _sim()
        assert all(c >= 0 for c in r.c_plasma)

    def test_notes_contains_drug_name(self):
        r = _sim(drug_name="MyDrug")
        assert "MyDrug" in r.notes


# ---------------------------------------------------------------------------
# antibiotic_disruption_effect
# ---------------------------------------------------------------------------

class TestAntibioticDisruptionEffect:
    def _run(self, **kw):
        defaults = dict(
            drug_name="TestDrug",
            dose_mg=100.0,
            ka_abs_per_h=1.0,
            vmax_gut_mg_per_h=10.0,
            km_gut_mg_L=5.0,
            cl_L_per_h=5.0,
            vd_L=30.0,
            dt_h=0.1,
        )
        defaults.update(kw)
        return antibiotic_disruption_effect(**defaults)

    def test_returns_dict_with_keys(self):
        result = self._run()
        assert "normal" in result
        assert "disrupted" in result
        assert "auc_ratio" in result

    def test_disrupted_higher_auc_than_normal(self):
        """After antibiotics (disrupted microbiome), more drug reaches systemic circulation."""
        result = self._run(disruption_factor=0.05)
        assert result["auc_ratio"] > 1.0

    def test_disruption_factor_1_gives_ratio_1(self):
        """If disruption_factor=1.0, same as normal microbiome."""
        result = self._run(disruption_factor=1.0)
        assert result["auc_ratio"] == pytest.approx(1.0, rel=1e-3)

    def test_normal_result_type(self):
        result = self._run()
        assert isinstance(result["normal"], MicrobiomePKResult)

    def test_disrupted_result_type(self):
        result = self._run()
        assert isinstance(result["disrupted"], MicrobiomePKResult)


# ---------------------------------------------------------------------------
# prodrug_activation
# ---------------------------------------------------------------------------

class TestProdrugActivation:
    def test_returns_result_type(self):
        assert isinstance(_prodrug(), ProdrugActivationResult)

    def test_frozen_dataclass(self):
        r = _prodrug()
        with pytest.raises((AttributeError, TypeError)):
            r.cmax = 999.0  # type: ignore[misc]

    def test_f_activated_in_01(self):
        r = _prodrug()
        assert 0.0 <= r.f_activated <= 1.0

    def test_zero_density_no_activation(self):
        r = _prodrug(microbiome_density=0.0)
        assert r.f_activated == pytest.approx(0.0, abs=1e-6)
        assert r.cmax == pytest.approx(0.0, abs=1e-6)

    def test_higher_density_more_activation(self):
        r_low = _prodrug(microbiome_density=0.1)
        r_high = _prodrug(microbiome_density=1.0)
        assert r_high.f_activated > r_low.f_activated

    def test_higher_density_higher_auc(self):
        r_low = _prodrug(microbiome_density=0.1)
        r_high = _prodrug(microbiome_density=1.0)
        assert r_high.auc > r_low.auc

    def test_times_plasma_same_length(self):
        r = _prodrug()
        assert len(r.times_h) == len(r.c_active_plasma)

    def test_plasma_non_negative(self):
        r = _prodrug()
        assert all(c >= 0 for c in r.c_active_plasma)

    def test_drug_name_stored(self):
        r = _prodrug(drug_name="Sulfasalazine")
        assert r.drug_name == "Sulfasalazine"

    def test_validation_prodrug_dose_negative(self):
        with pytest.raises(ValueError, match="prodrug_dose_mg"):
            _prodrug(prodrug_dose_mg=-50.0)

    def test_validation_ka_zero(self):
        with pytest.raises(ValueError, match="ka_abs_per_h"):
            _prodrug(ka_abs_per_h=0.0)

    def test_notes_contains_density(self):
        r = _prodrug(microbiome_density=0.5)
        assert "0.50" in r.notes
