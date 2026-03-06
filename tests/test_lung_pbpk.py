"""Tests for inhaled lung PBPK model with regional deposition and absorption."""

import pytest

from omega_pbpk.core.lung_pbpk import (
    InhaledDeposition,
    LungPBPKResult,
    compare_devices,
    simulate_lung_pk,
)

_PARAMS = dict(
    drug_name="albuterol",
    dose_mg=0.1,
    cl_L_per_h=5.0,
    vd_L=50.0,
    device="dpi",
)


class TestInputValidation:
    def test_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_lung_pk("D", 0, 5.0, 50.0)

    def test_cl_zero(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_lung_pk("D", 0.1, 0, 50.0)

    def test_vd_zero(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_lung_pk("D", 0.1, 5.0, 0)

    def test_unknown_device(self):
        with pytest.raises(ValueError, match="Unknown device"):
            simulate_lung_pk("D", 0.1, 5.0, 50.0, device="jet_pack")


class TestBasicSimulation:
    def setup_method(self):
        self.r = simulate_lung_pk(**_PARAMS)

    def test_result_type(self):
        assert isinstance(self.r, LungPBPKResult)

    def test_cmax_positive(self):
        assert self.r.cmax_mg_L > 0

    def test_tmax_positive(self):
        assert self.r.tmax_h >= 0

    def test_auc_positive(self):
        assert self.r.auc_0_t > 0

    def test_profile_consistent_length(self):
        n = len(self.r.times_h)
        assert len(self.r.cp_mg_L) == n
        assert len(self.r.a_op_mg) == n
        assert len(self.r.a_tb_mg) == n
        assert len(self.r.a_pul_mg) == n

    def test_concentrations_non_negative(self):
        assert all(c >= 0 for c in self.r.cp_mg_L)
        assert all(a >= 0 for a in self.r.a_op_mg)
        assert all(a >= 0 for a in self.r.a_tb_mg)
        assert all(a >= 0 for a in self.r.a_pul_mg)

    def test_lung_bioavailability_in_range(self):
        assert 0.0 <= self.r.lung_bioavailability <= 1.0

    def test_total_bioavailability_in_range(self):
        assert 0.0 <= self.r.total_bioavailability <= 1.0

    def test_total_f_gte_lung_f(self):
        """Total F ≥ lung F (swallowed drug adds to total)."""
        assert self.r.total_bioavailability >= self.r.lung_bioavailability

    def test_drug_name_stored(self):
        assert self.r.drug_name == "albuterol"

    def test_device_stored(self):
        assert self.r.device == "dpi"

    def test_dose_stored(self):
        assert self.r.dose_mg == 0.1

    def test_deposition_type(self):
        assert isinstance(self.r.deposition, InhaledDeposition)

    def test_notes_non_empty(self):
        assert len(self.r.notes) > 0

    def test_gi_mass_non_negative(self):
        assert all(a >= 0 for a in self.r.a_gi_mg)

    def test_gi_mass_increases(self):
        """Cumulative GI mass should be non-decreasing."""
        gi = self.r.a_gi_mg
        for i in range(1, len(gi)):
            assert gi[i] >= gi[i - 1] - 1e-9


class TestDeviceComparison:
    def test_dpi_higher_pulmonary_deposition_than_mdi(self):
        """DPI has higher pulmonary fraction than MDI (more efficient)."""
        from omega_pbpk.core.lung_pbpk import _DEVICE_DEPOSITION

        dep_dpi = _DEVICE_DEPOSITION["dpi"]
        dep_mdi = _DEVICE_DEPOSITION["mdi"]
        assert dep_dpi.pulmonary > dep_mdi.pulmonary

    def test_mdi_higher_op_deposition_than_dpi(self):
        from omega_pbpk.core.lung_pbpk import _DEVICE_DEPOSITION

        dep_mdi = _DEVICE_DEPOSITION["mdi"]
        dep_dpi = _DEVICE_DEPOSITION["dpi"]
        assert dep_mdi.oropharyngeal > dep_dpi.oropharyngeal

    def test_nebulizer_high_pulmonary_fraction(self):
        from omega_pbpk.core.lung_pbpk import _DEVICE_DEPOSITION

        dep_neb = _DEVICE_DEPOSITION["nebulizer"]
        assert dep_neb.pulmonary >= 0.50

    def test_all_devices_work(self):
        for device in ["mdi", "dpi", "nebulizer", "soft_mist"]:
            r = simulate_lung_pk(**{**_PARAMS, "device": device})
            assert r.cmax_mg_L > 0

    def test_dpi_higher_lung_f_than_mdi(self):
        """DPI has higher pulmonary fraction → higher lung bioavailability."""
        r_dpi = simulate_lung_pk(**{**_PARAMS, "device": "dpi"})
        r_mdi = simulate_lung_pk(**{**_PARAMS, "device": "mdi"})
        assert r_dpi.lung_bioavailability > r_mdi.lung_bioavailability


class TestDepositionFractions:
    def test_fractions_sum_to_le_1(self):
        from omega_pbpk.core.lung_pbpk import _DEVICE_DEPOSITION

        for dev, dep in _DEVICE_DEPOSITION.items():
            total = dep.oropharyngeal + dep.tracheobronchial + dep.pulmonary + dep.exhaled
            assert total == pytest.approx(1.0, abs=1e-9), f"{dev} fractions don't sum to 1"

    def test_all_fractions_non_negative(self):
        from omega_pbpk.core.lung_pbpk import _DEVICE_DEPOSITION

        for dep in _DEVICE_DEPOSITION.values():
            assert dep.oropharyngeal >= 0
            assert dep.tracheobronchial >= 0
            assert dep.pulmonary >= 0
            assert dep.exhaled >= 0


class TestPKScaling:
    def test_double_dose_double_cmax(self):
        """Linear PK: 2× dose → 2× Cmax."""
        r1 = simulate_lung_pk(**{**_PARAMS, "dose_mg": 0.1})
        r2 = simulate_lung_pk(**{**_PARAMS, "dose_mg": 0.2})
        assert r2.cmax_mg_L == pytest.approx(r1.cmax_mg_L * 2, rel=0.05)

    def test_higher_ka_pulmonary_faster_absorption(self):
        """Higher ka_pulmonary → earlier tmax and higher Cmax."""
        r_slow = simulate_lung_pk(**{**_PARAMS, "ka_pulmonary_per_h": 1.0})
        r_fast = simulate_lung_pk(**{**_PARAMS, "ka_pulmonary_per_h": 10.0})
        assert r_fast.tmax_h <= r_slow.tmax_h

    def test_higher_cl_lower_auc(self):
        r_low = simulate_lung_pk(**{**_PARAMS, "cl_L_per_h": 2.0})
        r_high = simulate_lung_pk(**{**_PARAMS, "cl_L_per_h": 20.0})
        assert r_high.auc_0_t < r_low.auc_0_t


class TestCustomDeposition:
    def test_custom_deposition_applied(self):
        """Custom deposition with 90% pulmonary → high lung bioavailability."""
        custom = InhaledDeposition(
            oropharyngeal=0.02,
            tracheobronchial=0.05,
            pulmonary=0.90,
            exhaled=0.03,
        )
        r = simulate_lung_pk(**_PARAMS, custom_deposition=custom)
        assert r.lung_bioavailability > 0.70  # high pulmonary fraction → high F


class TestCompareDevices:
    def test_returns_list(self):
        results = compare_devices("drug", 0.1, 5.0, 50.0)
        assert len(results) == 4  # all 4 built-in devices

    def test_all_results_valid(self):
        results = compare_devices("drug", 0.1, 5.0, 50.0)
        for r in results:
            assert isinstance(r, LungPBPKResult)
            assert r.cmax_mg_L > 0

    def test_subset_devices(self):
        results = compare_devices("drug", 0.1, 5.0, 50.0, devices=["mdi", "dpi"])
        assert len(results) == 2
