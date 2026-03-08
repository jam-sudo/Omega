"""
Tests for Phase 411 — Volatile Anesthetic PK/PD Model
"""

import pytest

from omega_pbpk.core.volatile_anesthetic_pk import (
    _AGENT_PROPS,
    VolatileAnestheticResult,
    compare_agents,
    simulate_volatile_anesthetic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _default_sim(agent="sevoflurane", inspired=2.0, t_ind=30.0, t_end=60.0):
    return simulate_volatile_anesthetic(
        drug_name="test",
        agent=agent,
        inspired_conc_pct=inspired,
        t_induction_min=t_ind,
        t_end_min=t_end,
    )


# ---------------------------------------------------------------------------
# Return type & structure
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_correct_type(self):
        r = _default_sim()
        assert isinstance(r, VolatileAnestheticResult)

    def test_list_fields_populated(self):
        r = _default_sim()
        assert len(r.times_min) > 0
        assert len(r.fa_fi_ratio) == len(r.times_min)
        assert len(r.c_blood_mL_per_100mL) == len(r.times_min)
        assert len(r.c_brain_mL_per_100mL) == len(r.times_min)
        assert len(r.mac_fraction) == len(r.times_min)

    def test_scalar_fields_positive(self):
        r = _default_sim()
        assert r.blood_gas_coefficient > 0
        assert r.mac_pct > 0

    def test_agent_stored_correctly(self):
        r = _default_sim(agent="isoflurane")
        assert r.agent == "isoflurane"

    def test_inspired_conc_stored(self):
        r = _default_sim(inspired=3.0)
        assert r.inspired_conc_pct == 3.0


# ---------------------------------------------------------------------------
# FA/FI dynamics
# ---------------------------------------------------------------------------
class TestFAFIDynamics:
    def test_fafi_rises_toward_one_during_wash_in(self):
        r = _default_sim(agent="sevoflurane", t_ind=30.0, t_end=30.0)
        # At end of wash-in, FA/FI should be substantially above 0
        assert r.fa_fi_ratio[-1] > 0.3

    def test_fafi_monotonically_increases_during_wash_in(self):
        r = _default_sim(agent="sevoflurane", t_ind=60.0, t_end=60.0)
        fafi = r.fa_fi_ratio
        for i in range(1, min(len(fafi), 100)):
            assert fafi[i] >= fafi[i - 1] - 1e-9

    def test_fafi_starts_near_zero(self):
        r = _default_sim()
        assert r.fa_fi_ratio[0] == pytest.approx(0.0, abs=1e-6)

    def test_fafi_bounded_zero_to_one(self):
        r = _default_sim()
        for v in r.fa_fi_ratio:
            assert 0.0 <= v <= 1.0 + 1e-9

    def test_fafi_decreases_after_discontinuation(self):
        r = _default_sim(agent="sevoflurane", t_ind=30.0, t_end=60.0)
        times = r.times_min
        fafi = r.fa_fi_ratio
        # Find index just after discontinuation
        idx_disc = next(i for i, t in enumerate(times) if t >= 30.0)
        # FA/FI at end should be less than at discontinuation
        assert fafi[-1] < fafi[idx_disc]


# ---------------------------------------------------------------------------
# Blood-gas coefficient effect on speed
# ---------------------------------------------------------------------------
class TestBloodGasEffect:
    def test_lower_blood_gas_faster_equilibration(self):
        """desflurane (0.45) equilibrates faster than halothane (2.54)."""
        r_des = _default_sim(agent="desflurane", t_ind=30.0, t_end=30.0)
        r_hal = _default_sim(agent="halothane", t_ind=30.0, t_end=30.0)
        # At same time point, desflurane FA/FI should be higher
        assert r_des.fa_fi_ratio[-1] > r_hal.fa_fi_ratio[-1]

    def test_sevoflurane_vs_isoflurane_equilibration(self):
        """sevoflurane (0.65) faster than isoflurane (1.4)."""
        r_sev = _default_sim(agent="sevoflurane", t_ind=30.0, t_end=30.0)
        r_iso = _default_sim(agent="isoflurane", t_ind=30.0, t_end=30.0)
        assert r_sev.fa_fi_ratio[-1] > r_iso.fa_fi_ratio[-1]


# ---------------------------------------------------------------------------
# Blood & brain concentrations
# ---------------------------------------------------------------------------
class TestConcentrations:
    def test_c_blood_positive_during_wash_in(self):
        r = _default_sim()
        # After time 0 (and before discontinuation), c_blood should be > 0
        assert any(v > 0 for v in r.c_blood_mL_per_100mL)

    def test_c_brain_follows_c_blood(self):
        """c_brain lags c_blood during wash-in."""
        r = _default_sim(agent="sevoflurane", t_ind=30.0, t_end=30.0)
        # At end of wash-in, c_brain <= c_blood (brain lags)
        assert r.c_brain_mL_per_100mL[-1] <= r.c_blood_mL_per_100mL[-1] + 1e-9

    def test_c_brain_nonnegative(self):
        r = _default_sim()
        for v in r.c_brain_mL_per_100mL:
            assert v >= -1e-9

    def test_blood_gas_coefficient_matches_agent(self):
        for agent, props in _AGENT_PROPS.items():
            r = simulate_volatile_anesthetic(
                drug_name="x", agent=agent, inspired_conc_pct=2.0, t_end_min=20.0
            )
            assert r.blood_gas_coefficient == props["blood_gas"]


# ---------------------------------------------------------------------------
# MAC and milestones
# ---------------------------------------------------------------------------
class TestMACAndMilestones:
    def test_mac_fraction_positive_after_wash_in(self):
        r = _default_sim(t_ind=60.0, t_end=60.0)
        assert r.mac_fraction[-1] > 0

    def test_induction_time_positive(self):
        r = _default_sim()
        assert r.time_to_induction_min >= 0

    def test_50pct_mac_before_induction(self):
        r = _default_sim(agent="desflurane", inspired=6.0, t_ind=60.0, t_end=60.0)
        assert r.time_to_50pct_mac_min <= r.time_to_induction_min + 1e-6

    def test_desflurane_fastest_induction(self):
        """Desflurane has lowest blood:gas → fastest FA/FI equilibration.

        Compare FA/FI at a fixed time when all agents are given at twice
        their own MAC (so all can reach 1 MAC within the window).
        """
        t_wash = 30.0
        fafi_vals = {}
        for a, props in _AGENT_PROPS.items():
            # Use 2x MAC so every agent can reach 1 MAC
            inspired = props["mac_pct"] * 2.0
            r = simulate_volatile_anesthetic(
                drug_name=a,
                agent=a,
                inspired_conc_pct=inspired,
                t_induction_min=t_wash,
                t_end_min=t_wash,
            )
            fafi_vals[a] = r.fa_fi_ratio[-1]
        # Desflurane (lowest blood:gas) should have the highest FA/FI
        assert fafi_vals["desflurane"] == max(fafi_vals.values())

    def test_halothane_slowest_induction(self):
        """Halothane has highest blood:gas → slowest FA/FI equilibration."""
        t_wash = 30.0
        fafi_vals = {}
        for a, props in _AGENT_PROPS.items():
            inspired = props["mac_pct"] * 2.0
            r = simulate_volatile_anesthetic(
                drug_name=a,
                agent=a,
                inspired_conc_pct=inspired,
                t_induction_min=t_wash,
                t_end_min=t_wash,
            )
            fafi_vals[a] = r.fa_fi_ratio[-1]
        # Halothane (highest blood:gas) should have the lowest FA/FI
        assert fafi_vals["halothane"] == min(fafi_vals.values())

    def test_awakening_time_nonnegative(self):
        r = _default_sim()
        assert r.time_to_awakening_min >= 0

    def test_washout_decreases_fafi(self):
        r = simulate_volatile_anesthetic(
            drug_name="sevo",
            agent="sevoflurane",
            inspired_conc_pct=2.0,
            t_induction_min=30.0,
            t_end_min=90.0,
        )
        times = r.times_min
        fafi = r.fa_fi_ratio
        idx_disc = next(i for i, t in enumerate(times) if t >= 30.1)
        # FA/FI must be lower 30 min after discontinuation
        idx_later = next((i for i, t in enumerate(times) if t >= 60.0), -1)
        if idx_later > 0:
            assert fafi[idx_later] < fafi[idx_disc]


# ---------------------------------------------------------------------------
# compare_agents
# ---------------------------------------------------------------------------
class TestCompareAgents:
    def test_returns_four_results(self):
        results = compare_agents()
        assert len(results) == 4

    def test_sorted_by_induction_time_ascending(self):
        results = compare_agents(inspired_conc_pct=2.0, t_induction_min=30.0)
        times = [r.time_to_induction_min for r in results]
        assert times == sorted(times)

    def test_sorted_ascending_induction_times(self):
        """Verify sort is strictly ascending (earlier = shorter induction)."""
        results = compare_agents(inspired_conc_pct=2.0, t_induction_min=30.0)
        times = [r.time_to_induction_min for r in results]
        # All times should be non-decreasing
        for i in range(len(times) - 1):
            assert times[i] <= times[i + 1] + 1e-9

    def test_all_agents_present(self):
        results = compare_agents()
        agents_present = {r.agent for r in results}
        assert agents_present == set(_AGENT_PROPS.keys())


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_invalid_inspired_conc(self):
        with pytest.raises(ValueError, match="inspired_conc_pct"):
            simulate_volatile_anesthetic("x", "sevoflurane", 0.0)

    def test_invalid_agent(self):
        with pytest.raises(ValueError, match="agent"):
            simulate_volatile_anesthetic("x", "ether", 2.0)

    def test_invalid_alveolar_vent(self):
        with pytest.raises(ValueError, match="alveolar_ventilation"):
            simulate_volatile_anesthetic("x", "sevoflurane", 2.0, alveolar_ventilation_L_per_min=0)

    def test_invalid_cardiac_output(self):
        with pytest.raises(ValueError, match="cardiac_output"):
            simulate_volatile_anesthetic("x", "sevoflurane", 2.0, cardiac_output_L_per_min=-1.0)

    def test_invalid_t_end(self):
        with pytest.raises(ValueError, match="t_end_min"):
            simulate_volatile_anesthetic("x", "sevoflurane", 2.0, t_end_min=0)
