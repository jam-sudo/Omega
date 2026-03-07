"""Tests for enzyme kinetics simulator — Phase 455."""

from __future__ import annotations

import pytest

from omega_pbpk.core.enzyme_kinetics import (
    EnzymeKineticsResult,
    calculate_kinetic_params,
    competitive_inhibition,
    michaelis_menten_rate,
    mixed_inhibition,
    simulate_enzyme_kinetics,
    uncompetitive_inhibition,
)

# ---------------------------------------------------------------------------
# michaelis_menten_rate
# ---------------------------------------------------------------------------


class TestMichaelisMentenRate:
    def test_at_high_substrate_approaches_vmax(self):
        """At [S] >> Km, rate ≈ Vmax."""
        rate = michaelis_menten_rate(substrate_uM=10_000.0, vmax_uM_per_min=100.0, km_uM=1.0)
        assert rate == pytest.approx(100.0, rel=1e-3)

    def test_at_km_equals_half_vmax(self):
        """At [S] = Km, rate = Vmax / 2."""
        rate = michaelis_menten_rate(substrate_uM=50.0, vmax_uM_per_min=80.0, km_uM=50.0)
        assert rate == pytest.approx(40.0, rel=1e-10)

    def test_zero_substrate_gives_zero_rate(self):
        rate = michaelis_menten_rate(substrate_uM=0.0, vmax_uM_per_min=100.0, km_uM=10.0)
        assert rate == pytest.approx(0.0, abs=1e-12)

    def test_rate_less_than_vmax(self):
        rate = michaelis_menten_rate(substrate_uM=5.0, vmax_uM_per_min=100.0, km_uM=10.0)
        assert rate < 100.0

    def test_rate_positive_for_positive_substrate(self):
        rate = michaelis_menten_rate(substrate_uM=10.0, vmax_uM_per_min=50.0, km_uM=5.0)
        assert rate > 0.0

    def test_raises_on_nonpositive_vmax(self):
        with pytest.raises(ValueError, match="vmax"):
            michaelis_menten_rate(substrate_uM=10.0, vmax_uM_per_min=0.0, km_uM=5.0)

    def test_raises_on_nonpositive_km(self):
        with pytest.raises(ValueError, match="km"):
            michaelis_menten_rate(substrate_uM=10.0, vmax_uM_per_min=50.0, km_uM=-1.0)

    def test_raises_on_negative_substrate(self):
        with pytest.raises(ValueError, match="substrate"):
            michaelis_menten_rate(substrate_uM=-1.0, vmax_uM_per_min=50.0, km_uM=5.0)


# ---------------------------------------------------------------------------
# competitive_inhibition
# ---------------------------------------------------------------------------


class TestCompetitiveInhibition:
    def test_increases_apparent_km(self):
        """Competitive inhibitor increases effective Km; rate at [S]=Km_true decreases."""
        s = 10.0  # = Km_true
        vmax, km, ki, inh = 100.0, 10.0, 5.0, 10.0
        rate_no_inh = michaelis_menten_rate(s, vmax, km)
        rate_with_inh = competitive_inhibition(s, vmax, km, ki, inh)
        assert rate_with_inh < rate_no_inh

    def test_vmax_unchanged_at_high_substrate(self):
        """At [S] >> Km_app, rate approaches Vmax regardless of competitive inhibitor."""
        rate = competitive_inhibition(
            substrate_uM=1e6, vmax=100.0, km=10.0, ki_uM=5.0, inhibitor_uM=100.0
        )
        assert rate == pytest.approx(100.0, rel=1e-3)

    def test_no_inhibitor_equals_mm(self):
        """With inhibitor_uM=0, result equals standard MM."""
        s, vmax, km = 20.0, 80.0, 10.0
        assert competitive_inhibition(s, vmax, km, ki_uM=5.0, inhibitor_uM=0.0) == pytest.approx(
            michaelis_menten_rate(s, vmax, km)
        )

    def test_raises_on_nonpositive_ki(self):
        with pytest.raises(ValueError, match="ki_uM"):
            competitive_inhibition(10.0, 100.0, 10.0, ki_uM=0.0, inhibitor_uM=5.0)

    def test_raises_on_negative_inhibitor(self):
        with pytest.raises(ValueError, match="inhibitor_uM"):
            competitive_inhibition(10.0, 100.0, 10.0, ki_uM=5.0, inhibitor_uM=-1.0)


# ---------------------------------------------------------------------------
# uncompetitive_inhibition
# ---------------------------------------------------------------------------


class TestUncompetitiveInhibition:
    def test_decreases_vmax(self):
        """Uncompetitive inhibitor reduces Vmax_app."""
        rate_high_s = uncompetitive_inhibition(
            substrate_uM=1e6, vmax=100.0, km=10.0, ki_uM=5.0, inhibitor_uM=10.0
        )
        # Vmax_app = Vmax / (1 + I/Ki) = 100 / 3 ≈ 33.3
        assert rate_high_s == pytest.approx(100.0 / 3.0, rel=1e-3)

    def test_decreases_km_proportionally(self):
        """Km_app = Km / (1 + I/Ki)."""
        vmax, km, ki, inh = 100.0, 10.0, 5.0, 5.0
        factor = 1.0 + inh / ki  # = 2
        km_app = km / factor  # = 5
        # At [S] = km_app, rate should be vmax_app / 2
        rate = uncompetitive_inhibition(km_app, vmax, km, ki, inh)
        vmax_app = vmax / factor
        assert rate == pytest.approx(vmax_app / 2.0, rel=1e-8)

    def test_no_inhibitor_equals_mm(self):
        s, vmax, km = 20.0, 80.0, 10.0
        assert uncompetitive_inhibition(s, vmax, km, ki_uM=5.0, inhibitor_uM=0.0) == pytest.approx(
            michaelis_menten_rate(s, vmax, km)
        )

    def test_raises_on_nonpositive_ki(self):
        with pytest.raises(ValueError, match="ki_uM"):
            uncompetitive_inhibition(10.0, 100.0, 10.0, ki_uM=-1.0, inhibitor_uM=5.0)


# ---------------------------------------------------------------------------
# mixed_inhibition
# ---------------------------------------------------------------------------


class TestMixedInhibition:
    def test_changes_both_km_and_vmax(self):
        """Mixed inhibition: both Km_app and Vmax_app differ from uninhibited."""
        vmax, km, ki, ki_prime, inh = 100.0, 10.0, 5.0, 10.0, 10.0
        alpha = 1.0 + inh / ki  # 3
        alpha_prime = 1.0 + inh / ki_prime  # 2
        vmax_app = vmax / alpha_prime  # 50
        km_app = km * alpha / alpha_prime  # 15

        rate = mixed_inhibition(km_app, vmax, km, ki, ki_prime, inh)
        # At [S] = km_app: rate = vmax_app / 2
        assert rate == pytest.approx(vmax_app / 2.0, rel=1e-8)

    def test_no_inhibitor_equals_mm(self):
        s, vmax, km = 20.0, 80.0, 10.0
        assert mixed_inhibition(
            s, vmax, km, ki_uM=5.0, ki_prime_uM=5.0, inhibitor_uM=0.0
        ) == pytest.approx(michaelis_menten_rate(s, vmax, km))

    def test_raises_on_nonpositive_ki_prime(self):
        with pytest.raises(ValueError, match="ki_prime_uM"):
            mixed_inhibition(10.0, 100.0, 10.0, ki_uM=5.0, ki_prime_uM=0.0, inhibitor_uM=5.0)

    def test_raises_on_negative_inhibitor(self):
        with pytest.raises(ValueError, match="inhibitor_uM"):
            mixed_inhibition(10.0, 100.0, 10.0, ki_uM=5.0, ki_prime_uM=5.0, inhibitor_uM=-2.0)


# ---------------------------------------------------------------------------
# simulate_enzyme_kinetics
# ---------------------------------------------------------------------------


class TestSimulateEnzymeKinetics:
    def test_returns_correct_dataclass(self):
        result = simulate_enzyme_kinetics((0.1, 200.0), vmax=100.0, km=10.0)
        assert isinstance(result, EnzymeKineticsResult)

    def test_no_inhibition_baseline(self):
        result = simulate_enzyme_kinetics((0.1, 200.0), vmax=100.0, km=10.0, inhibition_type="none")
        assert result.inhibition_type == "none"
        assert result.km_apparent == pytest.approx(10.0)
        assert result.vmax_apparent == pytest.approx(100.0)

    def test_n_points_respected(self):
        result = simulate_enzyme_kinetics((1.0, 100.0), vmax=50.0, km=5.0, n_points=20)
        assert len(result.substrate_range) == 20
        assert len(result.rates) == 20

    def test_max_rate_is_max_of_rates(self):
        result = simulate_enzyme_kinetics((0.1, 1000.0), vmax=100.0, km=10.0)
        assert result.max_rate == pytest.approx(max(result.rates))

    def test_competitive_increases_km_apparent(self):
        km = 10.0
        result = simulate_enzyme_kinetics(
            (0.1, 500.0),
            vmax=100.0,
            km=km,
            inhibitor_uM=10.0,
            ki_uM=5.0,
            inhibition_type="competitive",
        )
        assert result.km_apparent > km

    def test_uncompetitive_decreases_vmax_apparent(self):
        vmax = 100.0
        result = simulate_enzyme_kinetics(
            (0.1, 500.0),
            vmax=vmax,
            km=10.0,
            inhibitor_uM=10.0,
            ki_uM=5.0,
            inhibition_type="uncompetitive",
        )
        assert result.vmax_apparent < vmax

    def test_mixed_changes_both(self):
        """Mixed inhibition changes Vmax (always) and Km when ki != ki_prime.

        The simulate_enzyme_kinetics helper uses ki_uM for both competitive and
        uncompetitive components (symmetric mixed inhibition), so alpha ==
        alpha_prime and Km_app == Km.  We verify Vmax_app changes; Km change is
        tested directly via mixed_inhibition() in TestMixedInhibition.
        """
        vmax, km = 100.0, 10.0
        result = simulate_enzyme_kinetics(
            (0.1, 500.0),
            vmax=vmax,
            km=km,
            inhibitor_uM=10.0,
            ki_uM=5.0,
            inhibition_type="mixed",
        )
        # Vmax_app = Vmax / (1 + I/Ki) = 100 / 3 < 100 → always changes
        assert result.vmax_apparent < vmax
        # At least one of the apparent parameters differs from the uninhibited case
        assert result.vmax_apparent != vmax or result.km_apparent != km

    def test_raises_on_invalid_inhibition_type(self):
        with pytest.raises(ValueError, match="inhibition_type"):
            simulate_enzyme_kinetics((1.0, 100.0), vmax=50.0, km=5.0, inhibition_type="banana")

    def test_raises_when_ki_missing_for_competitive(self):
        with pytest.raises(ValueError, match="ki_uM"):
            simulate_enzyme_kinetics(
                (1.0, 100.0), vmax=50.0, km=5.0, inhibitor_uM=5.0, inhibition_type="competitive"
            )

    def test_raises_on_bad_substrate_range(self):
        with pytest.raises(ValueError):
            simulate_enzyme_kinetics((100.0, 1.0), vmax=50.0, km=5.0)

    def test_rates_monotonically_increasing(self):
        """MM rates increase with substrate (no inhibition)."""
        result = simulate_enzyme_kinetics((0.1, 200.0), vmax=100.0, km=10.0)
        for i in range(len(result.rates) - 1):
            assert result.rates[i] <= result.rates[i + 1]


# ---------------------------------------------------------------------------
# calculate_kinetic_params
# ---------------------------------------------------------------------------


class TestCalculateKineticParams:
    def _generate_mm_data(self, vmax: float, km: float, n: int = 10):
        substrates = [km * (i + 1) / 2 for i in range(n)]
        rates = [vmax * s / (km + s) for s in substrates]
        return substrates, rates

    def test_recovers_vmax_approx(self):
        vmax, km = 100.0, 10.0
        subs, rates = self._generate_mm_data(vmax, km)
        params = calculate_kinetic_params(subs, rates)
        assert params["vmax_estimated"] == pytest.approx(vmax, rel=0.05)

    def test_recovers_km_approx(self):
        vmax, km = 100.0, 10.0
        subs, rates = self._generate_mm_data(vmax, km)
        params = calculate_kinetic_params(subs, rates)
        assert params["km_estimated"] == pytest.approx(km, rel=0.05)

    def test_returns_dict_with_correct_keys(self):
        subs, rates = self._generate_mm_data(50.0, 5.0)
        params = calculate_kinetic_params(subs, rates)
        assert "vmax_estimated" in params
        assert "km_estimated" in params

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_kinetic_params([1.0, 2.0], [1.0])

    def test_raises_on_too_few_points(self):
        with pytest.raises(ValueError, match="2 data points"):
            calculate_kinetic_params([5.0], [10.0])

    def test_raises_on_nonpositive_substrate(self):
        with pytest.raises(ValueError, match="substrate"):
            calculate_kinetic_params([0.0, 5.0], [10.0, 20.0])

    def test_raises_on_nonpositive_rate(self):
        with pytest.raises(ValueError, match="rate"):
            calculate_kinetic_params([1.0, 5.0], [0.0, 20.0])
