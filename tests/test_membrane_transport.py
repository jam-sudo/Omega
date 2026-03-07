"""Tests for membrane transport simulation -- Phase 459."""

from __future__ import annotations

import pytest

from omega_pbpk.core.membrane_transport import (
    Caco2Result,
    active_transport_rate,
    classify_permeability,
    efflux_ratio,
    passive_permeability,
    simulate_caco2,
)

# ---------------------------------------------------------------------------
# passive_permeability
# ---------------------------------------------------------------------------


class TestPassivePermeability:
    """Tests for passive_permeability."""

    def test_high_logP_low_psa_gives_high_papp(self):
        """High logP + low PSA molecule should yield high permeability."""
        papp = passive_permeability(logP=4.0, mw=300.0, psa_A2=20.0, n_hbd=1)
        assert papp > 20.0e-6, f"Expected high Papp, got {papp}"

    def test_high_psa_high_hbd_gives_low_papp(self):
        """High PSA + many H-bond donors should yield low permeability."""
        papp = passive_permeability(logP=0.0, mw=500.0, psa_A2=150.0, n_hbd=8)
        assert papp < 2.0e-6, f"Expected low Papp, got {papp}"

    def test_neutral_vs_anionic_charge(self):
        """Anionic compounds should have lower Papp than neutral."""
        p_neutral = passive_permeability(
            logP=2.0, mw=300.0, psa_A2=60.0, n_hbd=2, charge_state="neutral"
        )
        p_anionic = passive_permeability(
            logP=2.0, mw=300.0, psa_A2=60.0, n_hbd=2, charge_state="anionic"
        )
        assert p_neutral > p_anionic

    def test_cationic_higher_than_neutral(self):
        """Cationic compounds typically have higher membrane affinity."""
        p_neutral = passive_permeability(
            logP=2.0, mw=300.0, psa_A2=60.0, n_hbd=2, charge_state="neutral"
        )
        p_cationic = passive_permeability(
            logP=2.0, mw=300.0, psa_A2=60.0, n_hbd=2, charge_state="cationic"
        )
        assert p_cationic > p_neutral

    def test_returns_float(self):
        papp = passive_permeability(logP=2.0, mw=350.0, psa_A2=50.0, n_hbd=3)
        assert isinstance(papp, float)

    def test_papp_within_physical_bounds(self):
        """Papp should be clamped to [1e-9, 1e-4] cm/s."""
        papp_extreme_high = passive_permeability(logP=10.0, mw=50.0, psa_A2=0.0, n_hbd=0)
        papp_extreme_low = passive_permeability(logP=-10.0, mw=1000.0, psa_A2=300.0, n_hbd=20)
        assert papp_extreme_high <= 1.0e-4
        assert papp_extreme_low >= 1.0e-9

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw must be > 0"):
            passive_permeability(logP=2.0, mw=0.0, psa_A2=50.0, n_hbd=2)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError):
            passive_permeability(logP=2.0, mw=-100.0, psa_A2=50.0, n_hbd=2)

    def test_negative_psa_raises(self):
        with pytest.raises(ValueError, match="psa_A2 must be >= 0"):
            passive_permeability(logP=2.0, mw=300.0, psa_A2=-5.0, n_hbd=2)

    def test_negative_hbd_raises(self):
        with pytest.raises(ValueError, match="n_hbd must be >= 0"):
            passive_permeability(logP=2.0, mw=300.0, psa_A2=50.0, n_hbd=-1)

    def test_invalid_charge_state_raises(self):
        with pytest.raises(ValueError, match="charge_state must be one of"):
            passive_permeability(logP=2.0, mw=300.0, psa_A2=50.0, n_hbd=2, charge_state="unknown")

    def test_zero_psa_zero_hbd(self):
        """Zero PSA and zero H-bond donors should give higher Papp than typical."""
        papp_none = passive_permeability(logP=2.0, mw=300.0, psa_A2=0.0, n_hbd=0)
        papp_typical = passive_permeability(logP=2.0, mw=300.0, psa_A2=80.0, n_hbd=4)
        assert papp_none > papp_typical


# ---------------------------------------------------------------------------
# active_transport_rate
# ---------------------------------------------------------------------------


class TestActiveTransportRate:
    """Tests for active_transport_rate."""

    def test_half_saturation(self):
        """At [S] = Km, rate should be Vmax/2."""
        vmax = 100.0
        km = 50.0
        rate = active_transport_rate(50.0, vmax, km)
        assert abs(rate - vmax / 2.0) < 1e-9

    def test_low_conc_linear(self):
        """At [S] << Km, rate is approximately linear in [S]."""
        km = 1000.0
        vmax = 200.0
        r1 = active_transport_rate(1.0, vmax, km)
        r2 = active_transport_rate(2.0, vmax, km)
        assert abs(r2 / r1 - 2.0) < 0.05

    def test_high_conc_approaches_vmax(self):
        """At very high [S], rate approaches Vmax."""
        rate = active_transport_rate(1e6, 100.0, 1.0)
        assert rate > 99.9

    def test_zero_conc_gives_zero_rate(self):
        rate = active_transport_rate(0.0, 100.0, 50.0)
        assert rate == 0.0

    def test_negative_conc_raises(self):
        with pytest.raises(ValueError, match="substrate_conc_uM must be >= 0"):
            active_transport_rate(-1.0, 100.0, 50.0)

    def test_negative_vmax_raises(self):
        with pytest.raises(ValueError, match="vmax_pmol_cm2_min must be >= 0"):
            active_transport_rate(10.0, -100.0, 50.0)

    def test_zero_km_raises(self):
        with pytest.raises(ValueError, match="km_uM must be > 0"):
            active_transport_rate(10.0, 100.0, 0.0)

    def test_returns_float(self):
        result = active_transport_rate(10.0, 100.0, 50.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# efflux_ratio
# ---------------------------------------------------------------------------


class TestEffluxRatio:
    """Tests for efflux_ratio."""

    def test_symmetric_transport_er_one(self):
        er = efflux_ratio(10e-6, 10e-6)
        assert abs(er - 1.0) < 1e-9

    def test_efflux_ratio_gt_2_pgp_substrate(self):
        """ER > 2 should be considered P-gp substrate."""
        er = efflux_ratio(5e-6, 15e-6)
        assert er > 2.0

    def test_zero_ab_raises(self):
        with pytest.raises(ValueError, match="papp_ab_cm_s must be > 0"):
            efflux_ratio(0.0, 10e-6)

    def test_negative_ba_raises(self):
        with pytest.raises(ValueError, match="papp_ba_cm_s must be >= 0"):
            efflux_ratio(10e-6, -5e-6)

    def test_high_er_value(self):
        """Strong efflux should give ER >> 2."""
        er = efflux_ratio(1e-6, 30e-6)
        assert er == pytest.approx(30.0, rel=1e-6)


# ---------------------------------------------------------------------------
# classify_permeability
# ---------------------------------------------------------------------------


class TestClassifyPermeability:
    """Tests for classify_permeability."""

    def test_high_class(self):
        assert classify_permeability(25e-6) == "high"

    def test_boundary_high(self):
        assert classify_permeability(20e-6) == "high"

    def test_medium_class(self):
        assert classify_permeability(10e-6) == "medium"

    def test_low_class(self):
        assert classify_permeability(0.5e-6) == "low"

    def test_boundary_low(self):
        assert classify_permeability(2e-6) == "medium"

    def test_negative_papp_raises(self):
        with pytest.raises(ValueError, match="papp_cm_s must be >= 0"):
            classify_permeability(-1.0e-6)

    def test_zero_papp_is_low(self):
        assert classify_permeability(0.0) == "low"


# ---------------------------------------------------------------------------
# simulate_caco2
# ---------------------------------------------------------------------------


class TestSimulateCaco2:
    """Tests for simulate_caco2."""

    def test_returns_caco2_result(self):
        result = simulate_caco2("TestDrug", logP=2.5, mw=350.0, psa_A2=60.0, n_hbd=3)
        assert isinstance(result, Caco2Result)

    def test_drug_name_preserved(self):
        result = simulate_caco2("Ibuprofen", logP=3.5, mw=206.0, psa_A2=37.0, n_hbd=1)
        assert result.drug_name == "Ibuprofen"

    def test_lipophilic_low_psa_high_permeability(self):
        """Lipophilic, low-PSA drug should be classified as 'high' or 'medium'."""
        result = simulate_caco2("LipoDrug", logP=4.5, mw=300.0, psa_A2=15.0, n_hbd=0)
        assert result.permeability_class in {"high", "medium"}

    def test_pgp_efflux_increases_ba(self):
        """pgp_efflux=True should give higher BA than AB."""
        base = simulate_caco2("Drug", logP=2.0, mw=400.0, psa_A2=80.0, n_hbd=3)
        pgp = simulate_caco2("Drug", logP=2.0, mw=400.0, psa_A2=80.0, n_hbd=3, pgp_efflux=True)
        assert pgp.papp_ba_cm_s > base.papp_ba_cm_s

    def test_pgp_efflux_raises_efflux_ratio(self):
        """With P-gp efflux, ER should be >= 2."""
        result = simulate_caco2("Drug", logP=2.0, mw=400.0, psa_A2=80.0, n_hbd=3, pgp_efflux=True)
        assert result.efflux_ratio >= 2.0

    def test_pgp_substrate_flag_with_efflux(self):
        result = simulate_caco2("Drug", logP=2.0, mw=400.0, psa_A2=80.0, n_hbd=3, pgp_efflux=True)
        assert result.pgp_substrate is True

    def test_no_pgp_efflux_er_near_one(self):
        """Without efflux, ER should be ~ 1."""
        result = simulate_caco2("Drug", logP=2.0, mw=350.0, psa_A2=60.0, n_hbd=2, pgp_efflux=False)
        assert abs(result.efflux_ratio - 1.0) < 0.01

    def test_predicted_fa_pct_range(self):
        result = simulate_caco2("Drug", logP=2.0, mw=350.0, psa_A2=60.0, n_hbd=2)
        assert 0.0 <= result.predicted_fa_pct <= 100.0

    def test_high_permeability_high_fa(self):
        result = simulate_caco2("HighPerm", logP=5.0, mw=250.0, psa_A2=10.0, n_hbd=0)
        assert result.predicted_fa_pct > 80.0

    def test_notes_is_string(self):
        result = simulate_caco2("Drug", logP=2.0, mw=350.0, psa_A2=60.0, n_hbd=2)
        assert isinstance(result.notes, str) and len(result.notes) > 0

    def test_active_transport_increases_ab(self):
        """Adding active uptake should increase AB permeability."""
        passive = simulate_caco2("Drug", logP=0.5, mw=400.0, psa_A2=100.0, n_hbd=5)
        active = simulate_caco2(
            "Drug", logP=0.5, mw=400.0, psa_A2=100.0, n_hbd=5, km_uM=50.0, vmax=200.0
        )
        assert active.papp_ab_cm_s >= passive.papp_ab_cm_s
