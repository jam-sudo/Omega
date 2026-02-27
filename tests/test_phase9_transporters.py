"""Phase 9 — Transporter-mediated disposition tests.

Tests cover:
- TransporterKinetics: linear and MM modes
- TransporterSet: aggregate CL methods, inhibition, property flags
- build_transporter_set() factory
- TransporterInhibition: inhibition factor
- WholeBodyPBPK integration: OATP effect on liver CL and AUC
- WholeBodyPBPK integration: renal active secretion increases excretion
- WholeBodyPBPK integration: P-gp efflux reduces oral bioavailability
- Drug dataclass transporter field
"""

import math
import pytest
import numpy as np

from omega_pbpk.core.transporters import (
    TransporterInhibition,
    TransporterKinetics,
    TransporterSet,
    build_transporter_set,
)
from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_uptake():
    return TransporterKinetics(
        name="OATP1B1", organ="liver", direction="uptake", clint_L_per_h=5.0
    )


@pytest.fixture
def mm_uptake():
    """Michaelis-Menten OATP1B1: Vmax=10 mg/h, Km=1 mg/L."""
    return TransporterKinetics(
        name="OATP1B1", organ="liver", direction="uptake",
        vmax_mg_per_h=10.0, km_mg_per_L=1.0
    )


@pytest.fixture
def linear_pgp():
    return TransporterKinetics(
        name="P-gp", organ="gut_wall", direction="efflux", clint_L_per_h=3.0
    )


@pytest.fixture
def reference_drug():
    """Simple neutral drug with no transporters."""
    return Drug(
        name="RefDrug",
        mw=400.0,
        logP=2.0,
        fup=0.3,
        rbp=1.0,
        clint_hepatic_L_per_h=5.0,
        clr_L_per_h=0.5,
        peff=2.0,
        route="oral",
    )


# ---------------------------------------------------------------------------
# TransporterKinetics — linear mode
# ---------------------------------------------------------------------------

class TestTransporterKineticsLinear:
    def test_effective_cl_is_constant(self, linear_uptake):
        # In linear mode, CL doesn't depend on concentration
        for cu in (0.0, 0.1, 1.0, 10.0, 100.0):
            assert linear_uptake.effective_cl(cu) == pytest.approx(5.0, rel=1e-9)

    def test_transport_rate_linear(self, linear_uptake):
        # Rate = CL × Cu
        assert linear_uptake.transport_rate(2.0) == pytest.approx(10.0, rel=1e-9)

    def test_transport_rate_zero_at_zero_conc(self, linear_uptake):
        assert linear_uptake.transport_rate(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_negative_conc_clamped(self, linear_uptake):
        assert linear_uptake.transport_rate(-1.0) >= 0.0


# ---------------------------------------------------------------------------
# TransporterKinetics — Michaelis-Menten mode
# ---------------------------------------------------------------------------

class TestTransporterKineticsMM:
    def test_cl_at_km_is_half_max(self, mm_uptake):
        # CL = Vmax/(Km + Cu) → at Cu=Km: CL = Vmax/(2*Km) = 5.0
        assert mm_uptake.effective_cl(1.0) == pytest.approx(5.0, rel=1e-6)

    def test_cl_decreases_with_concentration(self, mm_uptake):
        cl_low = mm_uptake.effective_cl(0.1)
        cl_high = mm_uptake.effective_cl(10.0)
        assert cl_low > cl_high

    def test_cl_at_zero_approaches_vmax_over_km(self, mm_uptake):
        # Cu→0: CL ≈ Vmax/Km = 10/1 = 10
        cl_near_zero = mm_uptake.effective_cl(1e-8)
        assert cl_near_zero == pytest.approx(10.0, rel=0.001)

    def test_rate_at_km_is_half_vmax(self, mm_uptake):
        # Rate = CL × Cu = (Vmax/(Km+Cu)) × Cu = 10/(1+1)*1 = 5
        assert mm_uptake.transport_rate(1.0) == pytest.approx(5.0, rel=1e-6)

    def test_rate_saturates(self, mm_uptake):
        # At very high Cu: Rate → Vmax
        rate = mm_uptake.transport_rate(1000.0)
        assert rate == pytest.approx(10.0, rel=0.01)

    def test_mm_mode_overrides_linear(self):
        t = TransporterKinetics(
            name="T", organ="liver", direction="uptake",
            clint_L_per_h=99.0,  # should be ignored when vmax > 0
            vmax_mg_per_h=5.0,
            km_mg_per_L=1.0,
        )
        # Should use MM, not linear CL=99
        assert t.effective_cl(1.0) == pytest.approx(2.5, rel=1e-6)


# ---------------------------------------------------------------------------
# TransporterInhibition
# ---------------------------------------------------------------------------

class TestTransporterInhibition:
    def test_no_inhibition_factor_one_when_zero_inhibitor(self):
        inh = TransporterInhibition("OATP1B1", ic50_uM=1.0, inhibitor_concentration_uM=0.0)
        assert inh.inhibition_factor == pytest.approx(1.0, rel=1e-9)

    def test_inhibition_factor_at_ic50(self):
        # [I] = IC50 → factor = 0.5
        inh = TransporterInhibition("OATP1B1", ic50_uM=2.0, inhibitor_concentration_uM=2.0)
        assert inh.inhibition_factor == pytest.approx(0.5, rel=1e-6)

    def test_inhibition_factor_full_inhibition(self):
        # Very high [I] → factor → 0
        inh = TransporterInhibition("OATP1B1", ic50_uM=0.001, inhibitor_concentration_uM=1000.0)
        assert inh.inhibition_factor < 0.01

    def test_infinite_ic50_no_inhibition(self):
        inh = TransporterInhibition("OATP1B1", ic50_uM=float("inf"), inhibitor_concentration_uM=100.0)
        assert inh.inhibition_factor == pytest.approx(1.0, rel=1e-9)

    def test_zero_ic50_no_inhibition(self):
        inh = TransporterInhibition("OATP1B1", ic50_uM=0.0, inhibitor_concentration_uM=100.0)
        assert inh.inhibition_factor == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# TransporterSet
# ---------------------------------------------------------------------------

class TestTransporterSetProperties:
    def test_empty_set_has_no_hepatic_uptake(self):
        ts = TransporterSet()
        assert ts.has_hepatic_uptake is False

    def test_empty_set_has_no_gut_efflux(self):
        ts = TransporterSet()
        assert ts.has_gut_efflux is False

    def test_empty_set_has_no_renal_secretion(self):
        ts = TransporterSet()
        assert ts.has_renal_secretion is False

    def test_oatp_set_has_hepatic_uptake(self, linear_uptake):
        ts = TransporterSet(oatp1b1=linear_uptake)
        assert ts.has_hepatic_uptake is True

    def test_pgp_set_has_gut_efflux(self, linear_pgp):
        ts = TransporterSet(pgp=linear_pgp)
        assert ts.has_gut_efflux is True

    def test_oct2_set_has_renal_secretion(self):
        t = TransporterKinetics("OCT2", "kidney", "uptake", clint_L_per_h=2.0)
        ts = TransporterSet(oct2=t)
        assert ts.has_renal_secretion is True


class TestTransporterSetAggregateCL:
    def test_empty_hepatic_uptake_cl_zero(self):
        ts = TransporterSet()
        assert ts.hepatic_uptake_cl(1.0) == pytest.approx(0.0, abs=1e-12)

    def test_single_oatp_cl(self, linear_uptake):
        ts = TransporterSet(oatp1b1=linear_uptake)
        assert ts.hepatic_uptake_cl(1.0) == pytest.approx(5.0, rel=1e-9)

    def test_oatp1b1_plus_oatp1b3_cl_additive(self):
        t1 = TransporterKinetics("OATP1B1", "liver", "uptake", clint_L_per_h=5.0)
        t2 = TransporterKinetics("OATP1B3", "liver", "uptake", clint_L_per_h=3.0)
        ts = TransporterSet(oatp1b1=t1, oatp1b3=t2)
        assert ts.hepatic_uptake_cl(1.0) == pytest.approx(8.0, rel=1e-9)

    def test_gut_efflux_cl(self, linear_pgp):
        ts = TransporterSet(pgp=linear_pgp)
        assert ts.gut_efflux_cl(2.0) == pytest.approx(3.0, rel=1e-9)

    def test_renal_secretion_cl_additive(self):
        oct2 = TransporterKinetics("OCT2", "kidney", "uptake", clint_L_per_h=4.0)
        oat1 = TransporterKinetics("OAT1", "kidney", "uptake", clint_L_per_h=2.0)
        ts = TransporterSet(oct2=oct2, oat1=oat1)
        assert ts.renal_secretion_cl(0.5) == pytest.approx(6.0, rel=1e-9)

    def test_bw_scaling_applied(self, linear_uptake):
        ts = TransporterSet(oatp1b1=linear_uptake)
        cl_full = ts.hepatic_uptake_cl(1.0, scale=1.0)
        cl_half = ts.hepatic_uptake_cl(1.0, scale=0.5)
        assert cl_half == pytest.approx(0.5 * cl_full, rel=1e-9)


class TestTransporterSetInhibition:
    def test_inhibition_reduces_cl(self, linear_uptake):
        ts = TransporterSet(oatp1b1=linear_uptake)
        cl_uninhibited = ts.hepatic_uptake_cl(1.0)

        # Add 50% inhibition: [I] = IC50
        ts.add_inhibition(TransporterInhibition("OATP1B1", ic50_uM=1.0, inhibitor_concentration_uM=1.0))
        cl_inhibited = ts.hepatic_uptake_cl(1.0)
        assert cl_inhibited == pytest.approx(0.5 * cl_uninhibited, rel=1e-6)

    def test_inhibition_of_wrong_transporter_has_no_effect(self, linear_uptake):
        ts = TransporterSet(oatp1b1=linear_uptake)
        ts.add_inhibition(TransporterInhibition("P-gp", ic50_uM=1.0, inhibitor_concentration_uM=1.0))
        # OATP1B1 unaffected
        assert ts.hepatic_uptake_cl(1.0) == pytest.approx(5.0, rel=1e-9)

    def test_multiple_inhibitions_multiplicative(self, linear_uptake):
        ts = TransporterSet(oatp1b1=linear_uptake)
        # Two independent inhibitors each at IC50 → factor = 0.5 × 0.5 = 0.25
        ts.add_inhibition(TransporterInhibition("OATP1B1", ic50_uM=1.0, inhibitor_concentration_uM=1.0))
        ts.add_inhibition(TransporterInhibition("OATP1B1", ic50_uM=2.0, inhibitor_concentration_uM=2.0))
        cl = ts.hepatic_uptake_cl(1.0)
        assert cl == pytest.approx(0.25 * 5.0, rel=1e-6)


# ---------------------------------------------------------------------------
# build_transporter_set factory
# ---------------------------------------------------------------------------

class TestBuildTransporterSet:
    def test_empty_call_returns_all_none(self):
        ts = build_transporter_set()
        assert ts.oatp1b1 is None
        assert ts.pgp is None
        assert ts.oct2 is None

    def test_single_transporter_set(self):
        ts = build_transporter_set(oatp1b1_cl=5.0)
        assert ts.oatp1b1 is not None
        assert ts.oatp1b1.clint_L_per_h == pytest.approx(5.0)
        assert ts.oatp1b3 is None

    def test_multiple_transporters(self):
        ts = build_transporter_set(
            oatp1b1_cl=5.0,
            pgp_cl=3.0,
            oct2_cl=2.0,
        )
        assert ts.oatp1b1 is not None
        assert ts.pgp is not None
        assert ts.oct2 is not None
        assert ts.oatp1b3 is None

    def test_names_set_correctly(self):
        ts = build_transporter_set(oatp1b1_cl=1.0, pgp_cl=1.0, oct2_cl=1.0)
        assert ts.oatp1b1.name == "OATP1B1"
        assert ts.pgp.name == "P-gp"
        assert ts.oct2.name == "OCT2"

    def test_organs_set_correctly(self):
        ts = build_transporter_set(oatp1b1_cl=1.0, pgp_cl=1.0, oct2_cl=1.0)
        assert ts.oatp1b1.organ == "liver"
        assert ts.pgp.organ == "gut_wall"
        assert ts.oct2.organ == "kidney"

    def test_mate_transporters(self):
        ts = build_transporter_set(mate1_cl=3.0, mate2_cl=2.0)
        assert ts.mate1 is not None
        assert ts.mate2 is not None
        assert ts.has_renal_secretion is True


# ---------------------------------------------------------------------------
# Drug dataclass transporter field
# ---------------------------------------------------------------------------

class TestDrugTransporterField:
    def test_drug_default_transporters_none(self):
        drug = Drug(name="Test")
        assert drug.transporters is None

    def test_drug_with_transporter_set(self):
        ts = build_transporter_set(oatp1b1_cl=5.0)
        drug = Drug(name="Statin", transporters=ts)
        assert drug.transporters is not None
        assert drug.transporters.has_hepatic_uptake is True

    def test_drug_transporters_preserved_in_pbpk_model(self):
        ts = build_transporter_set(oatp1b1_cl=5.0)
        drug = Drug(name="Statin", transporters=ts, clint_hepatic_L_per_h=2.0, fup=0.05)
        model = WholeBodyPBPK(drug, body_weight=70.0)
        assert model._transporters.has_hepatic_uptake is True


# ---------------------------------------------------------------------------
# WholeBodyPBPK integration — OATP increases hepatic extraction / reduces AUC
# ---------------------------------------------------------------------------

class TestOatpIntegration:
    def _run_iv(self, oatp_cl=0.0, dose=50.0):
        ts = build_transporter_set(oatp1b1_cl=oatp_cl) if oatp_cl > 0 else None
        drug = Drug(
            name="OATPSubstrate",
            mw=500.0,
            logP=1.5,
            fup=0.05,
            rbp=0.8,
            clint_hepatic_L_per_h=2.0,
            clr_L_per_h=0.0,
            transporters=ts,
        )
        model = WholeBodyPBPK(drug, body_weight=70.0)
        model.setup_iv(dose_mg=dose)
        return model.simulate(t_end_h=24.0)

    def test_oatp_reduces_auc(self):
        result_no_oatp = self._run_iv(oatp_cl=0.0)
        result_with_oatp = self._run_iv(oatp_cl=20.0)  # strong OATP contribution
        auc_no = result_no_oatp.pk_summary()["AUC_mg_h_L"]
        auc_with = result_with_oatp.pk_summary()["AUC_mg_h_L"]
        assert auc_with < auc_no

    def test_oatp_increases_apparent_cl(self):
        result_no = self._run_iv(oatp_cl=0.0)
        result_with = self._run_iv(oatp_cl=20.0)
        cl_no = result_no.pk_summary()["CL_L_h"]
        cl_with = result_with.pk_summary()["CL_L_h"]
        assert cl_with > cl_no

    def test_oatp_mass_balance_maintained(self):
        result = self._run_iv(oatp_cl=10.0, dose=100.0)
        total_at_end = result.amounts[-1].sum()
        # Drug mass should be ≤ original dose (cleared)
        assert total_at_end <= 100.0 * 1.05  # allow 5% numerical tolerance

    def test_oatp_zero_same_as_no_transporters(self):
        """TransporterSet() with no active transporters should give identical AUC."""
        common = dict(
            name="Test", mw=500.0, logP=1.5, fup=0.05, rbp=0.8,
            clint_hepatic_L_per_h=2.0, clr_L_per_h=0.0,
        )
        drug_none = Drug(**common, transporters=None)
        drug_empty = Drug(**common, transporters=TransporterSet())

        m_none = WholeBodyPBPK(drug_none, 70.0)
        m_none.setup_iv(50.0)
        result_none = m_none.simulate(t_end_h=24.0)

        m_empty = WholeBodyPBPK(drug_empty, 70.0)
        m_empty.setup_iv(50.0)
        result_empty = m_empty.simulate(t_end_h=24.0)

        auc_none = result_none.pk_summary()["AUC_mg_h_L"]
        auc_empty = result_empty.pk_summary()["AUC_mg_h_L"]
        assert auc_none == pytest.approx(auc_empty, rel=1e-6)


# ---------------------------------------------------------------------------
# WholeBodyPBPK integration — renal active secretion increases CLr
# ---------------------------------------------------------------------------

class TestRenalSecretionIntegration:
    def _run_iv(self, sec_cl=0.0, dose=100.0):
        ts = build_transporter_set(oct2_cl=sec_cl) if sec_cl > 0 else None
        drug = Drug(
            name="OCT2Substrate",
            mw=300.0,
            logP=-1.5,
            fup=0.8,
            rbp=1.0,
            clint_hepatic_L_per_h=0.5,  # minimal hepatic
            clr_L_per_h=5.0,            # passive filtration
            transporters=ts,
        )
        model = WholeBodyPBPK(drug, body_weight=70.0)
        model.setup_iv(dose_mg=dose)
        return model.simulate(t_end_h=24.0)

    def test_active_secretion_reduces_auc(self):
        r_no = self._run_iv(sec_cl=0.0)
        r_sec = self._run_iv(sec_cl=30.0)
        assert r_sec.pk_summary()["AUC_mg_h_L"] < r_no.pk_summary()["AUC_mg_h_L"]

    def test_active_secretion_increases_total_cl(self):
        r_no = self._run_iv(sec_cl=0.0)
        r_sec = self._run_iv(sec_cl=30.0)
        assert r_sec.pk_summary()["CL_L_h"] > r_no.pk_summary()["CL_L_h"]

    def test_active_secretion_increases_excreted_renal_mass(self):
        from omega_pbpk.core.body import IDX_EXC_RENAL
        r_no = self._run_iv(sec_cl=0.0, dose=100.0)
        r_sec = self._run_iv(sec_cl=30.0, dose=100.0)
        exc_no = r_no.amounts[-1, IDX_EXC_RENAL]
        exc_sec = r_sec.amounts[-1, IDX_EXC_RENAL]
        assert exc_sec > exc_no

    def test_mass_balance_with_secretion(self):
        r = self._run_iv(sec_cl=20.0, dose=100.0)
        # All drug should be accounted for
        assert r.amounts[-1].sum() <= 100.0 * 1.05


# ---------------------------------------------------------------------------
# WholeBodyPBPK integration — P-gp efflux reduces oral bioavailability
# ---------------------------------------------------------------------------

class TestPgpEflluxIntegration:
    def _run_oral(self, pgp_cl=0.0, dose=100.0):
        ts = build_transporter_set(pgp_cl=pgp_cl) if pgp_cl > 0 else None
        drug = Drug(
            name="PgpSubstrate",
            mw=400.0,
            logP=3.0,
            fup=0.3,
            rbp=1.0,
            clint_hepatic_L_per_h=3.0,
            clr_L_per_h=0.5,
            peff=2.0,
            transporters=ts,
            route="oral",
        )
        model = WholeBodyPBPK(drug, body_weight=70.0)
        model.setup_oral(dose_mg=dose)
        return model.simulate(t_end_h=24.0)

    def test_pgp_reduces_cmax(self):
        r_no = self._run_oral(pgp_cl=0.0)
        r_pgp = self._run_oral(pgp_cl=10.0)
        cmax_no = r_no.pk_summary()["Cmax_mg_L"]
        cmax_pgp = r_pgp.pk_summary()["Cmax_mg_L"]
        assert cmax_pgp < cmax_no

    def test_pgp_reduces_auc(self):
        r_no = self._run_oral(pgp_cl=0.0)
        r_pgp = self._run_oral(pgp_cl=10.0)
        assert r_pgp.pk_summary()["AUC_mg_h_L"] < r_no.pk_summary()["AUC_mg_h_L"]

    def test_pgp_increases_fecal_excretion(self):
        from omega_pbpk.core.body import IDX_EXC_FECAL
        r_no = self._run_oral(pgp_cl=0.0, dose=100.0)
        r_pgp = self._run_oral(pgp_cl=10.0, dose=100.0)
        fecal_no = r_no.amounts[-1, IDX_EXC_FECAL]
        fecal_pgp = r_pgp.amounts[-1, IDX_EXC_FECAL]
        assert fecal_pgp > fecal_no

    def test_mass_balance_with_pgp(self):
        r = self._run_oral(pgp_cl=5.0, dose=100.0)
        assert r.amounts[-1].sum() <= 100.0 * 1.05
