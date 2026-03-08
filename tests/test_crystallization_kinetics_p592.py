"""Tests for Phase 592 — Drug Crystallization Kinetics."""

import pytest

from omega_pbpk.core.crystallization_kinetics_p592 import (
    CrystallizationKineticsResult,
    simulate_crystallization_kinetics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cs_mg_mL=0.5,  # solubility 0.5 mg/mL
        ka_crystallization_per_h=2.0,
    )
    defaults.update(kwargs)
    return simulate_crystallization_kinetics(**defaults)


def _supersaturated(**kwargs):
    """High dose → strong supersaturation."""
    defaults = dict(
        drug_name="Supersaturated",
        dose_mg=500.0,
        cs_mg_mL=0.1,
        ka_crystallization_per_h=5.0,
    )
    defaults.update(kwargs)
    return simulate_crystallization_kinetics(**defaults)


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_crystallization_kinetics_result(self):
        r = _default()
        assert isinstance(r, CrystallizationKineticsResult)

    def test_drug_name_stored(self):
        r = _default(drug_name="Itraconazole")
        assert r.drug_name == "Itraconazole"

    def test_dose_mg_stored(self):
        r = _default(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_times_is_list(self):
        r = _default()
        assert isinstance(r.times_h, list)

    def test_c_dissolved_is_list(self):
        r = _default()
        assert isinstance(r.c_dissolved_mg_mL, list)

    def test_c_crystalline_is_list(self):
        r = _default()
        assert isinstance(r.c_crystalline_mg_mL, list)

    def test_supersaturation_ratio_is_list(self):
        r = _default()
        assert isinstance(r.supersaturation_ratio, list)

    def test_mass_fraction_is_list(self):
        r = _default()
        assert isinstance(r.mass_fraction_crystalline, list)

    def test_all_lists_same_length(self):
        r = _default()
        n = len(r.times_h)
        assert len(r.c_dissolved_mg_mL) == n
        assert len(r.c_crystalline_mg_mL) == n
        assert len(r.supersaturation_ratio) == n
        assert len(r.mass_fraction_crystalline) == n

    def test_induction_time_is_float(self):
        r = _default()
        assert isinstance(r.induction_time_h, float)

    def test_t90_is_float(self):
        r = _default()
        assert isinstance(r.t_90pct_crystallized_h, float)

    def test_peak_supersaturation_is_float(self):
        r = _default()
        assert isinstance(r.peak_supersaturation, float)

    def test_auc_dissolved_is_positive(self):
        r = _default()
        assert r.auc_dissolved_mg_h_per_mL > 0

    def test_absorption_efficiency_is_float(self):
        r = _default()
        assert isinstance(r.absorption_efficiency_pct, float)

    def test_notes_is_string(self):
        r = _default()
        assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# List lengths
# ---------------------------------------------------------------------------


class TestListLengths:
    def test_length_consistent_with_steps(self):
        r = simulate_crystallization_kinetics(
            drug_name="Drug",
            dose_mg=50.0,
            cs_mg_mL=1.0,
            ka_crystallization_per_h=1.0,
            t_end_h=2.0,
            dt_h=0.1,
        )
        expected_n = int(round(2.0 / 0.1)) + 1
        assert len(r.times_h) == expected_n

    def test_times_start_at_zero(self):
        r = _default()
        assert r.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Induction time
# ---------------------------------------------------------------------------


class TestInductionTime:
    def test_induction_time_positive_when_supersaturated(self):
        r = _supersaturated()
        assert r.induction_time_h < r.times_h[-1]

    def test_induction_time_default_tend_when_no_crystallization(self):
        """No supersaturation → no crystallization → induction time = t_end."""
        r = simulate_crystallization_kinetics(
            drug_name="Drug",
            dose_mg=1.0,
            cs_mg_mL=10.0,  # well above dose/volume = 0.004 mg/mL
            ka_crystallization_per_h=5.0,
            volume_fluid_mL=250.0,
            t_end_h=4.0,
        )
        assert r.induction_time_h == r.times_h[-1]

    def test_induction_time_lte_tend(self):
        r = _supersaturated()
        assert r.induction_time_h <= r.times_h[-1]

    def test_zero_ka_means_no_crystallization(self):
        r = simulate_crystallization_kinetics(
            drug_name="Drug",
            dose_mg=1000.0,
            cs_mg_mL=0.1,
            ka_crystallization_per_h=0.0,  # no crystallization
            volume_fluid_mL=250.0,
            t_end_h=4.0,
        )
        assert r.induction_time_h == r.times_h[-1]
        assert all(c == pytest.approx(0.0) for c in r.c_crystalline_mg_mL)


# ---------------------------------------------------------------------------
# Supersaturation
# ---------------------------------------------------------------------------


class TestSupersaturation:
    def test_peak_supersaturation_at_least_one_when_dose_above_cs(self):
        r = _supersaturated()
        assert r.peak_supersaturation >= 1.0

    def test_peak_supersaturation_equals_max_ratio(self):
        r = _default()
        assert r.peak_supersaturation == pytest.approx(max(r.supersaturation_ratio))

    def test_c_dissolved_nonneg(self):
        r = _default()
        assert all(c >= 0.0 for c in r.c_dissolved_mg_mL)

    def test_c_crystalline_nonneg(self):
        r = _default()
        assert all(c >= 0.0 for c in r.c_crystalline_mg_mL)

    def test_mass_fraction_in_0_1(self):
        r = _default()
        assert all(0.0 <= f <= 1.0 for f in r.mass_fraction_crystalline)

    def test_supersaturation_ratio_at_t0_equals_initial_ratio(self):
        r = simulate_crystallization_kinetics(
            drug_name="Drug",
            dose_mg=50.0,  # 50/250=0.2; cs=0.1 → SR=2.0
            cs_mg_mL=0.1,
            ka_crystallization_per_h=0.0,  # freeze: no change
            volume_fluid_mL=250.0,
        )
        assert r.supersaturation_ratio[0] == pytest.approx(2.0, rel=0.01)

    def test_supersaturation_cap_applied(self):
        """Dose creating >20× supersaturation should be capped at 20×."""
        r = simulate_crystallization_kinetics(
            drug_name="Drug",
            dose_mg=1e6,
            cs_mg_mL=1.0,
            ka_crystallization_per_h=0.0,
            volume_fluid_mL=250.0,
        )
        assert r.supersaturation_ratio[0] <= 20.0 + 1e-6


# ---------------------------------------------------------------------------
# Polymer inhibition
# ---------------------------------------------------------------------------


class TestPolymerInhibition:
    def test_polymer_slows_crystallization(self):
        """Adding polymer increases induction time."""
        r_no_poly = _supersaturated(polymer_conc_mg_mL=0.0)
        r_with_poly = _supersaturated(polymer_conc_mg_mL=5.0)
        # With polymer: slower crystallization → longer induction or less crystallized
        assert r_with_poly.induction_time_h >= r_no_poly.induction_time_h

    def test_high_polymer_reduces_crystalline_fraction(self):
        r_no_poly = _supersaturated(polymer_conc_mg_mL=0.0, t_end_h=4.0)
        r_high_poly = _supersaturated(polymer_conc_mg_mL=20.0, t_end_h=4.0)
        final_cryst_no = r_no_poly.mass_fraction_crystalline[-1]
        final_cryst_hi = r_high_poly.mass_fraction_crystalline[-1]
        assert final_cryst_hi <= final_cryst_no


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=-1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=0.0)

    def test_zero_cs_raises(self):
        with pytest.raises(ValueError, match="cs_mg_mL"):
            _default(cs_mg_mL=0.0)

    def test_negative_cs_raises(self):
        with pytest.raises(ValueError, match="cs_mg_mL"):
            _default(cs_mg_mL=-1.0)

    def test_negative_ka_raises(self):
        with pytest.raises(ValueError, match="ka_crystallization"):
            _default(ka_crystallization_per_h=-0.1)

    def test_zero_volume_raises(self):
        with pytest.raises(ValueError, match="volume_fluid_mL"):
            _default(volume_fluid_mL=0.0)
