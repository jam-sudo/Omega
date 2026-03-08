"""Tests for Phase 579 - Hepatic Sinusoid Drug Transport Model."""

import pytest

from omega_pbpk.core.hepatic_sinusoid_transport_p579 import (
    HepaticSinusoidTransportResult,
    compare_transporter_activities,
    predict_sinusoid_transport,
)


def _default_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        cl_passive_L_per_h=5.0,
        oatp_km_mg_L=2.0,
        oatp_vmax_L_per_h=10.0,
        mrp2_km_mg_L=0.5,
        mrp2_vmax_L_per_h=3.0,
    )
    defaults.update(kwargs)
    return predict_sinusoid_transport(**defaults)


class TestReturnType:
    def test_returns_correct_type(self):
        r = _default_result()
        assert isinstance(r, HepaticSinusoidTransportResult)

    def test_has_all_fields(self):
        r = _default_result()
        assert r.drug_name == "TestDrug"
        assert isinstance(r.oatp_activity, float)
        assert isinstance(r.mrp2_activity, float)
        assert isinstance(r.cl_uptake_L_per_h, float)
        assert isinstance(r.cl_efflux_L_per_h, float)
        assert isinstance(r.cl_net_L_per_h, float)
        assert isinstance(r.extraction_ratio_net, float)
        assert isinstance(r.f_oral_transporter, float)
        assert isinstance(r.ddi_vulnerability, str)
        assert isinstance(r.pgx_variability, str)
        assert isinstance(r.notes, str)

    def test_frozen_dataclass(self):
        r = _default_result()
        with pytest.raises(AttributeError):
            r.drug_name = "Other"


class TestExtractionRatio:
    def test_e_h_in_range(self):
        r = _default_result()
        assert 0.0 <= r.extraction_ratio_net <= 1.0

    def test_f_oral_in_range(self):
        r = _default_result()
        assert 0.0 <= r.f_oral_transporter <= 1.0

    def test_f_oral_plus_e_h_equals_one(self):
        r = _default_result()
        assert abs(r.f_oral_transporter + r.extraction_ratio_net - 1.0) < 1e-12


class TestOATPActivity:
    def test_reduced_oatp_lowers_uptake(self):
        r_normal = _default_result(oatp_activity=1.0)
        r_reduced = _default_result(oatp_activity=0.3)
        assert r_reduced.cl_uptake_L_per_h < r_normal.cl_uptake_L_per_h

    def test_reduced_oatp_lowers_e_h(self):
        r_normal = _default_result(oatp_activity=1.0)
        r_reduced = _default_result(oatp_activity=0.3)
        assert r_reduced.extraction_ratio_net <= r_normal.extraction_ratio_net

    def test_zero_oatp_activity(self):
        r = _default_result(oatp_activity=0.0)
        # Only passive uptake remains
        assert r.cl_uptake_L_per_h > 0


class TestMRP2Activity:
    def test_reduced_mrp2_lowers_efflux(self):
        r_normal = _default_result(mrp2_activity=1.0)
        r_reduced = _default_result(mrp2_activity=0.2)
        assert r_reduced.cl_efflux_L_per_h < r_normal.cl_efflux_L_per_h

    def test_mrp2_change_affects_net_cl(self):
        r_normal = _default_result(mrp2_activity=1.0)
        r_reduced = _default_result(mrp2_activity=0.2)
        # Lower efflux -> higher net CL (uptake - efflux)
        assert r_reduced.cl_net_L_per_h >= r_normal.cl_net_L_per_h


class TestClNetNonNegative:
    def test_cl_net_non_negative(self):
        # High efflux, low uptake
        r = _default_result(
            cl_passive_L_per_h=0.1,
            oatp_vmax_L_per_h=0.1,
            mrp2_vmax_L_per_h=100.0,
            mrp2_km_mg_L=0.01,
        )
        assert r.cl_net_L_per_h >= 0.0


class TestDDIVulnerability:
    def test_high_ddi(self):
        # Large OATP contribution
        r = _default_result(
            cl_passive_L_per_h=1.0,
            oatp_vmax_L_per_h=50.0,
            oatp_km_mg_L=1.0,
            fup=0.01,
        )
        assert r.ddi_vulnerability == "high"

    def test_low_ddi(self):
        # Very low OATP contribution
        r = _default_result(
            cl_passive_L_per_h=100.0,
            oatp_vmax_L_per_h=0.1,
            oatp_km_mg_L=10.0,
            fup=1.0,
        )
        assert r.ddi_vulnerability == "low"

    def test_moderate_ddi(self):
        # cl_oatp > 0.5 * cl_passive_uptake but <= cl_passive_uptake
        # cl_passive_uptake = cl_passive * fup
        # cl_oatp = oatp_vmax / (1 + oatp_km / 1.0)
        # Need: 0.5 * cl_passive * fup < cl_oatp <= cl_passive * fup
        # fup=1.0, cl_passive=10 -> cl_passive_uptake=10
        # cl_oatp = vmax / (1 + km) -> need ~7 -> vmax=14, km=1 -> 14/2=7
        r = _default_result(
            cl_passive_L_per_h=10.0,
            oatp_vmax_L_per_h=14.0,
            oatp_km_mg_L=1.0,
            fup=1.0,
        )
        assert r.ddi_vulnerability == "moderate"


class TestPGxVariability:
    def test_significant_pgx(self):
        # High OATP fraction
        r = _default_result(
            cl_passive_L_per_h=1.0,
            oatp_vmax_L_per_h=50.0,
            oatp_km_mg_L=1.0,
            fup=0.01,
        )
        assert r.pgx_variability == "significant"

    def test_minimal_pgx(self):
        # Low OATP fraction
        r = _default_result(
            cl_passive_L_per_h=100.0,
            oatp_vmax_L_per_h=0.1,
            oatp_km_mg_L=10.0,
            fup=1.0,
        )
        assert r.pgx_variability == "minimal"


class TestCompareTransporterActivities:
    def test_sorted_descending_by_f_oral(self):
        scenarios = [
            {"oatp_activity": 1.0, "mrp2_activity": 1.0, "fup": 0.1},
            {"oatp_activity": 0.1, "mrp2_activity": 1.0, "fup": 0.1},
            {"oatp_activity": 0.5, "mrp2_activity": 0.5, "fup": 0.1},
        ]
        results = compare_transporter_activities("DrugX", 5.0, 2.0, 10.0, 0.5, 3.0, scenarios)
        assert len(results) == 3
        for i in range(1, len(results)):
            assert results[i - 1].f_oral_transporter >= results[i].f_oral_transporter

    def test_empty_scenarios(self):
        results = compare_transporter_activities("DrugX", 5.0, 2.0, 10.0, 0.5, 3.0, [])
        assert results == []


class TestValidation:
    def test_fup_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(fup=0.0)

    def test_fup_negative_raises(self):
        with pytest.raises(ValueError):
            _default_result(fup=-0.1)

    def test_fup_above_one_raises(self):
        with pytest.raises(ValueError):
            _default_result(fup=1.1)

    def test_oatp_vmax_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(oatp_vmax_L_per_h=0.0)

    def test_oatp_km_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(oatp_km_mg_L=0.0)

    def test_mrp2_vmax_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(mrp2_vmax_L_per_h=0.0)

    def test_mrp2_km_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(mrp2_km_mg_L=0.0)
