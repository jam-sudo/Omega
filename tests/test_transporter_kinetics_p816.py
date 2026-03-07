"""Tests for core/transporter_kinetics_p816.py — Phase 816."""

from __future__ import annotations

import pytest

from omega_pbpk.core.transporter_kinetics_p816 import (
    TransporterKineticsResult,
    calculate_transporter_kinetics,
    screen_inhibitors,
)


class TestReturnType:
    def test_returns_dataclass(self):
        result = calculate_transporter_kinetics(
            drug_name="Rosuvastatin",
            transporter_name="OATP1B1",
            substrate_conc_uM=10.0,
            km_uM=5.0,
            vmax_pmol_per_min_per_mg=100.0,
        )
        assert isinstance(result, TransporterKineticsResult)

    def test_drug_name_preserved(self):
        result = calculate_transporter_kinetics(
            drug_name="Metformin",
            transporter_name="OCT2",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=50.0,
        )
        assert result.drug_name == "Metformin"

    def test_transporter_name_preserved(self):
        result = calculate_transporter_kinetics(
            drug_name="DrugA",
            transporter_name="P-gp",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=200.0,
            direction="efflux",
        )
        assert result.transporter_name == "P-gp"


class TestNoInhibitor:
    def test_transport_fraction_is_one_without_inhibitor(self):
        result = calculate_transporter_kinetics(
            drug_name="D",
            transporter_name="OATP1B1",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
        )
        assert result.transport_fraction == pytest.approx(1.0)

    def test_ddi_risk_low_without_inhibitor(self):
        result = calculate_transporter_kinetics(
            drug_name="D",
            transporter_name="OATP1B1",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
        )
        assert result.ddi_risk == "low"

    def test_inhibitor_name_empty_without_inhibitor(self):
        result = calculate_transporter_kinetics(
            drug_name="D",
            transporter_name="OATP1B1",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
        )
        assert result.inhibitor_name == ""
        assert result.ki_uM == 0.0
        assert result.inhibitor_conc_uM == 0.0


class TestWithInhibitor:
    def test_transport_fraction_reduced_with_inhibitor(self):
        result = calculate_transporter_kinetics(
            drug_name="D",
            transporter_name="OATP1B1",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
            inhibitor_name="Rifampin",
            inhibitor_conc_uM=50.0,
            ki_uM=2.0,
        )
        assert result.transport_fraction < 1.0

    def test_high_inhibitor_gives_high_ddi_risk(self):
        """Very high [I]/Ki ratio should give transport_fraction < 0.5 → high risk."""
        result = calculate_transporter_kinetics(
            drug_name="D",
            transporter_name="OATP1B1",
            substrate_conc_uM=1.0,
            km_uM=1.0,
            vmax_pmol_per_min_per_mg=100.0,
            inhibitor_name="StrongInh",
            inhibitor_conc_uM=1000.0,
            ki_uM=1.0,
        )
        assert result.ddi_risk == "high"
        assert result.transport_fraction < 0.5

    def test_moderate_inhibitor_gives_moderate_ddi_risk(self):
        """[I]/Ki ~1 at low substrate → transport_fraction between 0.5 and 0.8."""
        # Km_app = km*(1 + I/Ki), flux_inh/flux_uninh = (km + C)/(km*(1+I/Ki) + C)
        # km=10, C=5, I=10, Ki=10 → km_app=20, flux_inh = 100*5/25=20, uninh=100*5/15=33.3
        # ratio = 20/33.3 = 0.6 → moderate
        result = calculate_transporter_kinetics(
            drug_name="D",
            transporter_name="OATP1B1",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
            inhibitor_name="MidInh",
            inhibitor_conc_uM=10.0,
            ki_uM=10.0,
        )
        assert result.ddi_risk == "moderate"
        assert 0.5 <= result.transport_fraction < 0.8

    def test_weak_inhibitor_gives_low_ddi_risk(self):
        """Weak inhibitor (high Ki, low conc) → near-baseline flux → low risk."""
        result = calculate_transporter_kinetics(
            drug_name="D",
            transporter_name="OATP1B1",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
            inhibitor_name="WeakInh",
            inhibitor_conc_uM=0.001,
            ki_uM=1000.0,
        )
        assert result.ddi_risk == "low"
        assert result.transport_fraction > 0.8


class TestSaturation:
    def test_flux_approaches_vmax_at_high_concentration(self):
        """At C >> Km, flux should approach Vmax."""
        vmax = 100.0
        km = 1.0
        result = calculate_transporter_kinetics(
            drug_name="D",
            transporter_name="T",
            substrate_conc_uM=10000.0,
            km_uM=km,
            vmax_pmol_per_min_per_mg=vmax,
        )
        assert result.net_flux_pmol_per_min_per_mg > 0.99 * vmax

    def test_flux_at_km_is_half_vmax(self):
        """At C = Km, flux = Vmax/2 (Michaelis-Menten)."""
        vmax = 80.0
        km = 5.0
        result = calculate_transporter_kinetics(
            drug_name="D",
            transporter_name="T",
            substrate_conc_uM=km,
            km_uM=km,
            vmax_pmol_per_min_per_mg=vmax,
        )
        assert result.net_flux_pmol_per_min_per_mg == pytest.approx(vmax / 2.0, rel=1e-6)


class TestDirection:
    def test_efflux_ratio_2_for_efflux(self):
        result = calculate_transporter_kinetics(
            drug_name="D",
            transporter_name="P-gp",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
            direction="efflux",
        )
        assert result.efflux_ratio == 2.0

    def test_efflux_ratio_1_for_uptake(self):
        result = calculate_transporter_kinetics(
            drug_name="D",
            transporter_name="OATP1B1",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
            direction="uptake",
        )
        assert result.efflux_ratio == 1.0


class TestValidation:
    def test_negative_km_raises(self):
        with pytest.raises(ValueError, match="km_uM must be > 0"):
            calculate_transporter_kinetics(
                drug_name="D",
                transporter_name="T",
                substrate_conc_uM=5.0,
                km_uM=-1.0,
                vmax_pmol_per_min_per_mg=100.0,
            )

    def test_zero_km_raises(self):
        with pytest.raises(ValueError, match="km_uM must be > 0"):
            calculate_transporter_kinetics(
                drug_name="D",
                transporter_name="T",
                substrate_conc_uM=5.0,
                km_uM=0.0,
                vmax_pmol_per_min_per_mg=100.0,
            )

    def test_zero_vmax_raises(self):
        with pytest.raises(ValueError, match="vmax_pmol_per_min_per_mg must be > 0"):
            calculate_transporter_kinetics(
                drug_name="D",
                transporter_name="T",
                substrate_conc_uM=5.0,
                km_uM=10.0,
                vmax_pmol_per_min_per_mg=0.0,
            )

    def test_negative_vmax_raises(self):
        with pytest.raises(ValueError, match="vmax_pmol_per_min_per_mg must be > 0"):
            calculate_transporter_kinetics(
                drug_name="D",
                transporter_name="T",
                substrate_conc_uM=5.0,
                km_uM=10.0,
                vmax_pmol_per_min_per_mg=-50.0,
            )

    def test_negative_substrate_conc_raises(self):
        with pytest.raises(ValueError, match="substrate_conc_uM must be >= 0"):
            calculate_transporter_kinetics(
                drug_name="D",
                transporter_name="T",
                substrate_conc_uM=-1.0,
                km_uM=10.0,
                vmax_pmol_per_min_per_mg=100.0,
            )

    def test_zero_substrate_conc_valid(self):
        result = calculate_transporter_kinetics(
            drug_name="D",
            transporter_name="T",
            substrate_conc_uM=0.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
        )
        assert result.net_flux_pmol_per_min_per_mg == pytest.approx(0.0)


class TestScreenInhibitors:
    def test_returns_list(self):
        inhibitors = [
            {"name": "InhA", "ki_uM": 1.0, "conc_uM": 10.0},
            {"name": "InhB", "ki_uM": 10.0, "conc_uM": 10.0},
            {"name": "InhC", "ki_uM": 100.0, "conc_uM": 10.0},
        ]
        results = screen_inhibitors(
            drug_name="D",
            transporter_name="OATP1B1",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
            inhibitors=inhibitors,
        )
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_by_transport_fraction_ascending(self):
        """Most inhibitory (lowest transport_fraction) comes first."""
        inhibitors = [
            {"name": "Weak", "ki_uM": 1000.0, "conc_uM": 1.0},
            {"name": "Strong", "ki_uM": 0.1, "conc_uM": 100.0},
            {"name": "Moderate", "ki_uM": 5.0, "conc_uM": 10.0},
        ]
        results = screen_inhibitors(
            drug_name="D",
            transporter_name="OATP1B1",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
            inhibitors=inhibitors,
        )
        fractions = [r.transport_fraction for r in results]
        for i in range(len(fractions) - 1):
            assert fractions[i] <= fractions[i + 1]

    def test_each_result_correct_type(self):
        inhibitors = [{"name": "X", "ki_uM": 5.0, "conc_uM": 20.0}]
        results = screen_inhibitors(
            drug_name="D",
            transporter_name="T",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
            inhibitors=inhibitors,
        )
        assert isinstance(results[0], TransporterKineticsResult)

    def test_strongest_inhibitor_first(self):
        """Inhibitor with lowest Ki at fixed conc should appear first."""
        inhibitors = [
            {"name": "Strong", "ki_uM": 0.5, "conc_uM": 10.0},
            {"name": "Weak", "ki_uM": 50.0, "conc_uM": 10.0},
        ]
        results = screen_inhibitors(
            drug_name="D",
            transporter_name="OATP1B1",
            substrate_conc_uM=5.0,
            km_uM=10.0,
            vmax_pmol_per_min_per_mg=100.0,
            inhibitors=inhibitors,
        )
        assert results[0].inhibitor_name == "Strong"
        assert results[1].inhibitor_name == "Weak"
