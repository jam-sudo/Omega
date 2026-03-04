"""Tests for allometric scaling and multi-species physiology.

Tests cover:
  1. allometric_scale: human → rat clearance scaling
  2. Exponent 0.75 (metabolic rate / flow scaling)
  3. Always returns positive values
  4. get_species_physiology for human, rat, mouse, dog
  5. Mouse cardiac output < rat cardiac output (smaller body)
  6. All species return consistent parameter keys
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _import_allometric_scale():
    """Import allometric_scale; skip if not yet available."""
    try:
        from omega_pbpk.population.physiology import allometric_scale
    except ImportError:
        pytest.skip("allometric_scale not yet implemented in omega_pbpk.population.physiology")
    return allometric_scale


def _import_get_species_physiology():
    """Import get_species_physiology; skip if not yet available."""
    try:
        from omega_pbpk.population.physiology import get_species_physiology
    except ImportError:
        pytest.skip(
            "get_species_physiology not yet implemented in omega_pbpk.population.physiology"
        )
    return get_species_physiology


# ---------------------------------------------------------------------------
# Allometric scaling tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAllometricScale:
    def test_allometric_scale_human_to_rat(self):
        """Rat clearance (allometrically scaled from human) is less than human value.

        Formula: CL_rat = CL_human * (BW_rat / BW_human) ^ exponent
        With exponent=0.75: (0.25/70)^0.75 * 100 ≈ 1.6 L/h
        Rat absolute clearance < human (100 L/h) since rat is smaller.
        """
        allometric_scale = _import_allometric_scale()
        rat_cl = allometric_scale(
            human_value=100.0,
            human_bw=70.0,
            target_bw=0.25,
            exponent=0.75,
        )
        assert rat_cl < 100.0, (
            f"Rat clearance ({rat_cl}) should be less than human clearance (100 L/h)"
        )
        # Rough magnitude check: should be in the range 0.5–5 L/h
        assert 0.5 <= rat_cl <= 5.0, (
            f"Allometrically scaled rat clearance ({rat_cl:.3f} L/h) "
            "outside expected range 0.5-5 L/h"
        )

    def test_allometric_scale_exponent_0_75_for_flow(self):
        """BW^0.75 scaling: (0.25/70)^0.75 * 100 ≈ 1.6 L/h."""
        allometric_scale = _import_allometric_scale()
        rat_cl = allometric_scale(
            human_value=100.0,
            human_bw=70.0,
            target_bw=0.25,
            exponent=0.75,
        )
        # Expected: 100 * (0.25/70)^0.75
        import math

        expected = 100.0 * (0.25 / 70.0) ** 0.75
        assert math.isclose(rat_cl, expected, rel_tol=0.01), (
            f"Got {rat_cl:.4f}, expected ~{expected:.4f}"
        )

    def test_allometric_scale_returns_positive(self):
        """allometric_scale always returns a strictly positive value."""
        allometric_scale = _import_allometric_scale()
        for bw in (0.02, 0.25, 10.0, 70.0, 600.0):
            result = allometric_scale(
                human_value=100.0,
                human_bw=70.0,
                target_bw=bw,
                exponent=0.75,
            )
            assert result > 0, f"Non-positive result for target_bw={bw}: {result}"


# ---------------------------------------------------------------------------
# Multi-species physiology tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpeciesAllFourAvailable:
    def test_species_all_four_available(self):
        """get_species_physiology works for human, rat, mouse, and dog."""
        get_species_physiology = _import_get_species_physiology()
        for species in ("human", "rat", "mouse", "dog"):
            phys = get_species_physiology(species)
            assert isinstance(phys, dict), f"Expected dict for {species}, got {type(phys)}"
            assert len(phys) > 0, f"Empty physiology dict for {species}"


@pytest.mark.unit
class TestMouseCardiacOutputLessThanRat:
    def test_mouse_cardiac_output_less_than_rat(self):
        """Mouse (0.02 kg) absolute cardiac output < rat (0.25 kg)."""
        get_species_physiology = _import_get_species_physiology()
        mouse_phys = get_species_physiology("mouse")
        rat_phys = get_species_physiology("rat")

        # The key may be 'cardiac_output_L_h' or similar
        co_key = None
        for candidate in ("cardiac_output_L_h", "cardiac_output", "co_L_h"):
            if candidate in mouse_phys:
                co_key = candidate
                break

        if co_key is None:
            pytest.skip("cardiac_output key not found in species physiology dict")

        mouse_co = mouse_phys[co_key]
        rat_co = rat_phys[co_key]
        assert mouse_co < rat_co, f"Mouse CO ({mouse_co}) should be less than rat CO ({rat_co})"


@pytest.mark.unit
class TestSpeciesPhysiologyKeysConsistent:
    def test_species_physiology_keys_consistent(self):
        """All 4 species return the same set of physiological parameter keys."""
        get_species_physiology = _import_get_species_physiology()
        species_list = ("human", "rat", "mouse", "dog")
        key_sets = {}
        for species in species_list:
            phys = get_species_physiology(species)
            key_sets[species] = set(phys.keys())

        reference_keys = key_sets["human"]
        for species in species_list[1:]:
            assert key_sets[species] == reference_keys, (
                f"{species} keys differ from human: "
                f"extra={key_sets[species] - reference_keys}, "
                f"missing={reference_keys - key_sets[species]}"
            )
