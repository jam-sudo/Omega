"""Tests for Phase 193 — Inhalation PK Model."""

import pytest

from omega_pbpk.core.inhalation_pk_p193 import (
    InhalationPKResult,
    compare_devices,
    simulate_inhalation_pk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _default_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=1.0,
        device_type="mdi",
        cl_L_per_h=10.0,
        vd_L=50.0,
        t_end_h=12.0,
        dt_h=0.05,
    )
    defaults.update(kwargs)
    return simulate_inhalation_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_inhalation_pk_result(self):
        result = _default_result()
        assert isinstance(result, InhalationPKResult)

    def test_drug_name_preserved(self):
        result = _default_result(drug_name="Budesonide")
        assert result.drug_name == "Budesonide"

    def test_dose_mg_preserved(self):
        result = _default_result(dose_mg=2.5)
        assert result.dose_mg == pytest.approx(2.5)

    def test_device_type_preserved_upper(self):
        result = _default_result(device_type="dpi")
        assert result.device_type == "DPI"

    def test_times_h_is_list(self):
        result = _default_result()
        assert isinstance(result.times_h, list)

    def test_c_plasma_is_list(self):
        result = _default_result()
        assert isinstance(result.c_plasma_mg_L, list)

    def test_times_and_concs_same_length(self):
        result = _default_result()
        assert len(result.times_h) == len(result.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# Plasma concentration profile
# ---------------------------------------------------------------------------
class TestPlasmaProfile:
    def test_c_plasma_starts_at_zero(self):
        result = _default_result()
        assert result.c_plasma_mg_L[0] == pytest.approx(0.0)

    def test_c_plasma_rises_from_zero(self):
        result = _default_result()
        # at some point it should be positive
        assert max(result.c_plasma_mg_L) > 0.0

    def test_c_plasma_non_negative(self):
        result = _default_result()
        assert all(c >= 0.0 for c in result.c_plasma_mg_L)

    def test_cmax_positive(self):
        result = _default_result()
        assert result.cmax_mg_L > 0.0

    def test_auc_positive(self):
        result = _default_result()
        assert result.auc_mg_h_per_L > 0.0

    def test_tmax_positive(self):
        result = _default_result()
        assert result.tmax_h > 0.0

    def test_cmax_equals_max_in_list(self):
        result = _default_result()
        assert result.cmax_mg_L == pytest.approx(max(result.c_plasma_mg_L))


# ---------------------------------------------------------------------------
# Deposition fractions
# ---------------------------------------------------------------------------
class TestDepositionFractions:
    def test_mdi_central_fraction(self):
        result = _default_result(device_type="mdi")
        assert result.central_deposition_fraction == pytest.approx(0.60)

    def test_mdi_peripheral_fraction(self):
        result = _default_result(device_type="mdi")
        assert result.peripheral_deposition_fraction == pytest.approx(0.40)

    def test_fractions_sum_to_one(self):
        for device in ("mdi", "dpi", "nebulizer"):
            result = _default_result(device_type=device)
            total = result.central_deposition_fraction + result.peripheral_deposition_fraction
            assert total == pytest.approx(1.0), f"Failed for {device}"

    def test_peripheral_fraction_equals_one_minus_central(self):
        result = _default_result(device_type="dpi")
        assert result.peripheral_deposition_fraction == pytest.approx(
            1.0 - result.central_deposition_fraction
        )

    def test_nebulizer_has_higher_peripheral_than_mdi(self):
        neb = _default_result(device_type="nebulizer")
        mdi = _default_result(device_type="mdi")
        assert neb.peripheral_deposition_fraction > mdi.peripheral_deposition_fraction


# ---------------------------------------------------------------------------
# Absorbed fraction
# ---------------------------------------------------------------------------
class TestAbsorbedFraction:
    def test_f_absorbed_in_0_1(self):
        result = _default_result()
        assert 0.0 < result.f_absorbed_fraction <= 1.0

    def test_f_absorbed_not_zero(self):
        result = _default_result()
        assert result.f_absorbed_fraction > 0.0


# ---------------------------------------------------------------------------
# compare_devices
# ---------------------------------------------------------------------------
class TestCompareDevices:
    def test_returns_three_results(self):
        results = compare_devices("TestDrug", 1.0, 10.0, 50.0)
        assert len(results) == 3

    def test_all_are_inhalation_pk_result(self):
        results = compare_devices("TestDrug", 1.0, 10.0, 50.0)
        assert all(isinstance(r, InhalationPKResult) for r in results)

    def test_sorted_by_auc_ascending(self):
        results = compare_devices("TestDrug", 1.0, 10.0, 50.0)
        aucs = [r.auc_mg_h_per_L for r in results]
        assert aucs == sorted(aucs)

    def test_three_different_device_types(self):
        results = compare_devices("TestDrug", 1.0, 10.0, 50.0)
        device_types = {r.device_type for r in results}
        assert device_types == {"MDI", "DPI", "NEBULIZER"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_raises_on_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_inhalation_pk("Drug", 0.0, "mdi", 10.0, 50.0)

    def test_raises_on_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_inhalation_pk("Drug", -1.0, "mdi", 10.0, 50.0)

    def test_raises_on_invalid_device(self):
        with pytest.raises(ValueError, match="device_type"):
            simulate_inhalation_pk("Drug", 1.0, "spacer", 10.0, 50.0)

    def test_raises_on_zero_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_inhalation_pk("Drug", 1.0, "mdi", 0.0, 50.0)

    def test_raises_on_negative_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_inhalation_pk("Drug", 1.0, "mdi", -5.0, 50.0)

    def test_raises_on_zero_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_inhalation_pk("Drug", 1.0, "mdi", 10.0, 0.0)

    def test_case_insensitive_device(self):
        result = simulate_inhalation_pk("Drug", 1.0, "MDI", 10.0, 50.0)
        assert result.device_type == "MDI"

    def test_case_insensitive_nebulizer(self):
        result = simulate_inhalation_pk("Drug", 1.0, "Nebulizer", 10.0, 50.0)
        assert result.device_type == "NEBULIZER"
