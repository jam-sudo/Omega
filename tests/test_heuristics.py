"""Unit tests for Kp estimation methods (heuristic and Rodgers & Rowland).

Tests cover:
  - heuristic_kp: all 17 tissues, boundary conditions, ionisation classes
  - rodgers_rowland_kp: all 17 tissues, neutral/acid/base/zwitterion paths
  - estimate_all_kp: completeness and minimum floor
  - get_partition_method: dispatcher correctness
"""

from __future__ import annotations

import pytest

from omega_pbpk.core.heuristics import (
    _TISSUE_FACTORS,
    TISSUE_COMPOSITION,
    estimate_all_kp,
    get_partition_method,
    heuristic_kp,
    rodgers_rowland_kp,
)

ALL_TISSUES = list(_TISSUE_FACTORS.keys())
RR_TISSUES = list(TISSUE_COMPOSITION.keys())


# ---------------------------------------------------------------------------
# heuristic_kp — basic behaviour
# ---------------------------------------------------------------------------


class TestHeuristicKpBasic:
    def test_returns_float(self):
        assert isinstance(heuristic_kp(2.0), float)

    def test_minimum_floor(self):
        # Even for very negative logP, Kp must be >= 0.1
        assert heuristic_kp(-10.0, tissue_name="muscle") >= 0.1

    def test_increases_with_logP(self):
        kp_low = heuristic_kp(0.0, tissue_name="liver")
        kp_high = heuristic_kp(4.0, tissue_name="liver")
        assert kp_high > kp_low

    def test_adipose_higher_than_muscle(self):
        # Adipose has tissue_factor=2.5, muscle=0.95
        kp_fat = heuristic_kp(3.0, tissue_name="adipose")
        kp_mus = heuristic_kp(3.0, tissue_name="muscle")
        assert kp_fat > kp_mus

    def test_unknown_tissue_defaults_to_rest_factor(self):
        kp_unknown = heuristic_kp(2.0, tissue_name="unknown_organ")
        # Both use factor=0.9 / 1.0; unknown falls back to 1.0 default
        assert kp_unknown >= 0.1

    def test_fup_zero_clamps_correctly(self):
        # fup=0 edge case: should not produce NaN or negative
        kp = heuristic_kp(2.0, fup=1e-9, tissue_name="muscle")
        assert kp >= 0.1
        assert not (kp != kp)  # NaN check


# ---------------------------------------------------------------------------
# heuristic_kp — all 17 tissues
# ---------------------------------------------------------------------------


class TestHeuristicKpAllTissues:
    @pytest.mark.parametrize("tissue", ALL_TISSUES)
    def test_neutral_logP2(self, tissue):
        kp = heuristic_kp(2.0, pka=None, drug_type="neutral", tissue_name=tissue, fup=0.5)
        assert kp >= 0.1, f"Kp floor violated for tissue={tissue}"
        assert kp < 1000.0, f"Kp unrealistically high for tissue={tissue}"

    @pytest.mark.parametrize("tissue", ALL_TISSUES)
    def test_base_pka8(self, tissue):
        kp = heuristic_kp(2.0, pka=8.0, drug_type="base", tissue_name=tissue, fup=0.3)
        assert kp >= 0.1

    @pytest.mark.parametrize("tissue", ALL_TISSUES)
    def test_acid_pka4(self, tissue):
        kp = heuristic_kp(2.0, pka=4.0, drug_type="acid", tissue_name=tissue, fup=0.1)
        assert kp >= 0.1

    @pytest.mark.parametrize("tissue", ALL_TISSUES)
    def test_zwitterion(self, tissue):
        kp = heuristic_kp(1.5, pka=6.5, drug_type="zwitterion", tissue_name=tissue, fup=0.4)
        assert kp >= 0.1


# ---------------------------------------------------------------------------
# heuristic_kp — ionisation effects
# ---------------------------------------------------------------------------


class TestHeuristicKpIonisation:
    def test_base_increases_kp(self):
        kp_neutral = heuristic_kp(2.0, pka=None, drug_type="neutral", tissue_name="muscle")
        kp_base = heuristic_kp(2.0, pka=9.0, drug_type="base", tissue_name="muscle")
        # Highly basic drug: fraction_ionised near 1 → kp multiplied up
        assert kp_base >= kp_neutral

    def test_acid_does_not_greatly_exceed_neutral(self):
        kp_neutral = heuristic_kp(2.0, pka=None, drug_type="neutral", tissue_name="muscle")
        kp_acid = heuristic_kp(2.0, pka=3.0, drug_type="acid", tissue_name="muscle")
        # Acid scaling factor <= 1.0, so kp_acid <= kp_neutral
        assert kp_acid <= kp_neutral + 0.01

    def test_compound_type_overrides_drug_type(self):
        kp_via_drug_type = heuristic_kp(2.0, pka=8.0, drug_type="base", tissue_name="liver")
        kp_via_compound_type = heuristic_kp(
            2.0, pka=8.0, drug_type="neutral", compound_type="base", tissue_name="liver"
        )
        assert abs(kp_via_drug_type - kp_via_compound_type) < 1e-6


# ---------------------------------------------------------------------------
# rodgers_rowland_kp — basic behaviour
# ---------------------------------------------------------------------------


class TestRodgersRowlandBasic:
    def test_returns_float(self):
        assert isinstance(rodgers_rowland_kp(2.0), float)

    def test_minimum_floor(self):
        assert rodgers_rowland_kp(-10.0, tissue_name="muscle") >= 0.01

    def test_increases_with_logP(self):
        kp_low = rodgers_rowland_kp(0.0, tissue_name="liver")
        kp_high = rodgers_rowland_kp(4.0, tissue_name="liver")
        assert kp_high > kp_low

    def test_unknown_tissue_returns_1(self):
        kp = rodgers_rowland_kp(2.0, tissue_name="unknown_organ")
        assert kp == 1.0

    def test_base_higher_than_neutral_in_muscle(self):
        # Bases bind acidic phospholipids → higher Kp
        kp_neutral = rodgers_rowland_kp(
            2.0, pka=None, compound_type="neutral", tissue_name="muscle"
        )
        kp_base = rodgers_rowland_kp(2.0, pka=9.0, compound_type="base", tissue_name="muscle")
        assert kp_base > kp_neutral


# ---------------------------------------------------------------------------
# rodgers_rowland_kp — all tissue compositions
# ---------------------------------------------------------------------------


class TestRodgersRowlandAllTissues:
    @pytest.mark.parametrize("tissue", RR_TISSUES)
    def test_neutral_logP2(self, tissue):
        kp = rodgers_rowland_kp(2.0, pka=None, compound_type="neutral", tissue_name=tissue, fup=0.5)
        assert kp >= 0.01, f"Kp floor violated for tissue={tissue}"
        assert kp < 5000.0, f"Kp unrealistically high for tissue={tissue}, kp={kp}"

    @pytest.mark.parametrize("tissue", RR_TISSUES)
    def test_base_pka9(self, tissue):
        kp = rodgers_rowland_kp(2.0, pka=9.0, compound_type="base", tissue_name=tissue, fup=0.3)
        assert kp >= 0.01

    @pytest.mark.parametrize("tissue", RR_TISSUES)
    def test_acid_pka4(self, tissue):
        kp = rodgers_rowland_kp(2.0, pka=4.0, compound_type="acid", tissue_name=tissue, fup=0.1)
        assert kp >= 0.01

    @pytest.mark.parametrize("tissue", RR_TISSUES)
    def test_zwitterion(self, tissue):
        kp = rodgers_rowland_kp(
            1.5, pka=6.5, compound_type="zwitterion", tissue_name=tissue, fup=0.4
        )
        assert kp >= 0.01


# ---------------------------------------------------------------------------
# rodgers_rowland_kp — ionisation path coverage
# ---------------------------------------------------------------------------


class TestRodgersRowlandIonisation:
    def test_acid_path_eq4(self):
        # Acid uses Eq. 4 (neutral/acid path)
        kp = rodgers_rowland_kp(2.0, pka=4.0, compound_type="acid", tissue_name="liver")
        assert kp > 0.01

    def test_base_path_eq5_phospholipid_term(self):
        # Base uses Eq. 5 which adds fp_t * max(ion - 1, 0) term
        # Verify that base > neutral for a tissue with significant phospholipid (brain, fp=0.055)
        kp_n = rodgers_rowland_kp(2.0, pka=None, compound_type="neutral", tissue_name="brain")
        kp_b = rodgers_rowland_kp(2.0, pka=9.0, compound_type="base", tissue_name="brain")
        assert kp_b > kp_n

    def test_no_nan(self):
        for ct in ("neutral", "acid", "base", "zwitterion"):
            kp = rodgers_rowland_kp(2.0, pka=7.4, compound_type=ct, tissue_name="muscle")
            assert kp == kp, f"NaN detected for compound_type={ct}"


# ---------------------------------------------------------------------------
# estimate_all_kp — completeness
# ---------------------------------------------------------------------------


class TestEstimateAllKp:
    def test_returns_all_tissues(self):
        kp_dict = estimate_all_kp(2.0)
        assert set(kp_dict.keys()) == set(ALL_TISSUES)

    def test_all_values_above_floor(self):
        kp_dict = estimate_all_kp(2.0, pka=8.0, drug_type="base", fup=0.3)
        for tissue, kp in kp_dict.items():
            assert kp >= 0.1, f"Floor violated: {tissue}={kp}"

    def test_custom_tissue_list(self):
        subset = ["liver", "kidney", "lung"]
        kp_dict = estimate_all_kp(2.0, tissues=subset)
        assert set(kp_dict.keys()) == set(subset)


# ---------------------------------------------------------------------------
# get_partition_method — dispatcher
# ---------------------------------------------------------------------------


class TestGetPartitionMethod:
    def test_heuristic_returns_callable(self):
        fn = get_partition_method("heuristic")
        assert callable(fn)
        assert fn(2.0, tissue_name="liver") >= 0.1

    def test_rodgers_rowland_returns_callable(self):
        fn = get_partition_method("rodgers_rowland")
        assert callable(fn)
        assert fn(2.0, tissue_name="liver") >= 0.01

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown partition method"):
            get_partition_method("nonexistent_method")
