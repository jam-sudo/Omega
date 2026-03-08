"""Tests for Phase 418 — tpsa_calculator_p418.py"""

import pytest

from omega_pbpk.prediction.tpsa_calculator_p418 import (
    TPSAResult,
    calculate_tpsa,
    screen_compounds,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _drug_like(**kwargs):
    defaults = dict(
        drug_name="Aspirin",
        n_h_donors=1,
        n_h_acceptors=4,
        n_aromatic_rings=1,
        mw_Da=180.0,
    )
    defaults.update(kwargs)
    return calculate_tpsa(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_tpsa_result(self):
        result = _drug_like()
        assert isinstance(result, TPSAResult)


# ---------------------------------------------------------------------------
# Drug name preserved
# ---------------------------------------------------------------------------


class TestDrugName:
    def test_drug_name_preserved(self):
        result = _drug_like(drug_name="TestCompound")
        assert result.drug_name == "TestCompound"


# ---------------------------------------------------------------------------
# tpsa >= 0
# ---------------------------------------------------------------------------


class TestTPSANonNegative:
    def test_tpsa_non_negative(self):
        result = _drug_like(n_h_donors=0, n_h_acceptors=0)
        assert result.tpsa_angstrom2 >= 0.0

    def test_tpsa_zero_when_no_polar_groups(self):
        result = _drug_like(n_h_donors=0, n_h_acceptors=0, has_amide_bonds=0, has_phosphate=False)
        assert result.tpsa_angstrom2 == 0.0


# ---------------------------------------------------------------------------
# TPSA increases with more H-donors
# ---------------------------------------------------------------------------


class TestHDonorEffect:
    def test_more_donors_higher_tpsa(self):
        low = _drug_like(n_h_donors=1, n_h_acceptors=2)
        high = _drug_like(n_h_donors=5, n_h_acceptors=2)
        assert high.tpsa_angstrom2 > low.tpsa_angstrom2


# ---------------------------------------------------------------------------
# TPSA increases with more H-acceptors
# ---------------------------------------------------------------------------


class TestHAcceptorEffect:
    def test_more_acceptors_higher_tpsa(self):
        low = _drug_like(n_h_donors=1, n_h_acceptors=1)
        high = _drug_like(n_h_donors=1, n_h_acceptors=8)
        assert high.tpsa_angstrom2 > low.tpsa_angstrom2


# ---------------------------------------------------------------------------
# Phosphate group adds to tpsa
# ---------------------------------------------------------------------------


class TestPhosphateEffect:
    def test_phosphate_raises_tpsa(self):
        without = _drug_like(has_phosphate=False)
        with_phos = _drug_like(has_phosphate=True)
        assert with_phos.tpsa_angstrom2 > without.tpsa_angstrom2

    def test_phosphate_adds_40_5(self):
        without = _drug_like(has_phosphate=False)
        with_phos = _drug_like(has_phosphate=True)
        assert abs(with_phos.tpsa_angstrom2 - without.tpsa_angstrom2 - 40.5) < 1e-9


# ---------------------------------------------------------------------------
# Oral absorption class
# ---------------------------------------------------------------------------


class TestOralAbsorptionClass:
    def test_low_tpsa_excellent(self):
        # TPSA = 0 → excellent
        result = _drug_like(n_h_donors=0, n_h_acceptors=0)
        assert result.oral_absorption_class == "excellent"

    def test_high_tpsa_poor(self):
        # n_h_acceptors=7 → 26.02*7=182.14 → poor
        result = _drug_like(n_h_donors=0, n_h_acceptors=7)
        assert result.oral_absorption_class == "poor"

    def test_moderate_tpsa_moderate(self):
        # n_h_acceptors=4 → 26.02*4=104.08 → moderate
        result = _drug_like(n_h_donors=0, n_h_acceptors=4)
        assert result.oral_absorption_class == "moderate"


# ---------------------------------------------------------------------------
# CNS penetration class
# ---------------------------------------------------------------------------


class TestCNSClass:
    def test_low_tpsa_likely(self):
        result = _drug_like(n_h_donors=0, n_h_acceptors=0)
        assert result.cns_penetration_class == "likely"

    def test_high_tpsa_unlikely(self):
        # n_h_acceptors=4 → tpsa=104 → unlikely
        result = _drug_like(n_h_donors=0, n_h_acceptors=4)
        assert result.cns_penetration_class == "unlikely"

    def test_medium_tpsa_possible(self):
        # n_h_acceptors=3 → tpsa=78.06 → possible
        result = _drug_like(n_h_donors=0, n_h_acceptors=3)
        assert result.cns_penetration_class == "possible"


# ---------------------------------------------------------------------------
# BBB score: higher tpsa → lower bbb_score
# ---------------------------------------------------------------------------


class TestBBBScore:
    def test_higher_tpsa_lower_bbb(self):
        low_tpsa = _drug_like(n_h_donors=0, n_h_acceptors=1)
        high_tpsa = _drug_like(n_h_donors=2, n_h_acceptors=5)
        assert low_tpsa.bbb_score > high_tpsa.bbb_score

    def test_bbb_zero_when_tpsa_ge_140(self):
        # n_h_acceptors=6 → tpsa=156.12 >= 140
        result = _drug_like(n_h_donors=0, n_h_acceptors=6)
        assert result.bbb_score == 0.0


# ---------------------------------------------------------------------------
# GI absorption: higher tpsa → lower gi_absorption
# ---------------------------------------------------------------------------


class TestGIAbsorption:
    def test_higher_tpsa_lower_gi(self):
        low_tpsa = _drug_like(n_h_donors=0, n_h_acceptors=0)
        high_tpsa = _drug_like(n_h_donors=1, n_h_acceptors=3)
        assert low_tpsa.gi_absorption_pct > high_tpsa.gi_absorption_pct

    def test_gi_absorption_range(self):
        result = _drug_like()
        assert 0.0 <= result.gi_absorption_pct <= 100.0


# ---------------------------------------------------------------------------
# RO5 violations
# ---------------------------------------------------------------------------


class TestRO5Violations:
    def test_four_violations(self):
        # MW=600 (>500), need logP>5, donors=6 (>5), acceptors=11 (>10)
        # logP = 0.4*3 - 0.1*6 - 0.1*11 + 600*0.01 - 1.0
        #      = 1.2 - 0.6 - 1.1 + 6.0 - 1.0 = 4.5  → not > 5
        # Force with more rings: 0.4*8 - 0.1*6 - 0.1*11 + 600*0.01 - 1.0
        #                      = 3.2 - 0.6 - 1.1 + 6.0 - 1.0 = 6.5 → > 5
        result = calculate_tpsa(
            drug_name="BigDrug",
            n_h_donors=6,
            n_h_acceptors=11,
            n_aromatic_rings=8,
            mw_Da=600.0,
        )
        assert result.ro5_violations == 4

    def test_zero_violations_drug_like(self):
        result = calculate_tpsa(
            drug_name="NiceDrug",
            n_h_donors=1,
            n_h_acceptors=3,
            n_aromatic_rings=1,
            mw_Da=200.0,
        )
        assert result.ro5_violations == 0


# ---------------------------------------------------------------------------
# drug_likeness_score between 0 and 100
# ---------------------------------------------------------------------------


class TestDrugLikenessScore:
    def test_range(self):
        result = _drug_like()
        assert 0.0 <= result.drug_likeness_score <= 100.0

    def test_range_high_violations(self):
        result = calculate_tpsa(
            drug_name="PoorDrug",
            n_h_donors=10,
            n_h_acceptors=15,
            n_aromatic_rings=0,
            mw_Da=1200.0,
        )
        assert 0.0 <= result.drug_likeness_score <= 100.0


# ---------------------------------------------------------------------------
# screen_compounds
# ---------------------------------------------------------------------------


class TestScreenCompounds:
    def _compounds(self):
        return [
            dict(drug_name="A", n_h_donors=0, n_h_acceptors=2, n_aromatic_rings=2, mw_Da=200.0),
            dict(drug_name="B", n_h_donors=5, n_h_acceptors=10, n_aromatic_rings=0, mw_Da=480.0),
            dict(drug_name="C", n_h_donors=1, n_h_acceptors=4, n_aromatic_rings=1, mw_Da=250.0),
        ]

    def test_returns_list(self):
        result = screen_compounds(self._compounds())
        assert isinstance(result, list)

    def test_sorted_by_drug_likeness_descending(self):
        results = screen_compounds(self._compounds())
        scores = [r.drug_likeness_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_length_preserved(self):
        compounds = self._compounds()
        results = screen_compounds(compounds)
        assert len(results) == len(compounds)

    def test_all_tpsa_result_instances(self):
        results = screen_compounds(self._compounds())
        for r in results:
            assert isinstance(r, TPSAResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_mw_zero(self):
        with pytest.raises(ValueError, match="mw_Da"):
            calculate_tpsa(
                drug_name="X",
                n_h_donors=1,
                n_h_acceptors=2,
                n_aromatic_rings=1,
                mw_Da=0.0,
            )

    def test_mw_negative(self):
        with pytest.raises(ValueError, match="mw_Da"):
            calculate_tpsa(
                drug_name="X",
                n_h_donors=1,
                n_h_acceptors=2,
                n_aromatic_rings=1,
                mw_Da=-50.0,
            )

    def test_negative_donors(self):
        with pytest.raises(ValueError, match="n_h_donors"):
            calculate_tpsa(
                drug_name="X",
                n_h_donors=-1,
                n_h_acceptors=2,
                n_aromatic_rings=1,
                mw_Da=300.0,
            )

    def test_negative_acceptors(self):
        with pytest.raises(ValueError, match="n_h_acceptors"):
            calculate_tpsa(
                drug_name="X",
                n_h_donors=1,
                n_h_acceptors=-2,
                n_aromatic_rings=1,
                mw_Da=300.0,
            )
