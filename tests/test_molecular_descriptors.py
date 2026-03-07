"""Tests for Phase 575 — prediction/molecular_descriptors.py"""


from omega_pbpk.prediction.molecular_descriptors import (
    drug_likeness_score,
    ghose_filter,
    lipinski_ro5,
    molar_refractivity_estimate,
    veber_rules,
)

# ──────────────────────────────────────────────
# lipinski_ro5
# ──────────────────────────────────────────────


class TestLipinskiRo5:
    def test_aspirin_passes_all(self):
        """Aspirin: MW=180, logP=1.2, HBD=1, HBA=4 → passes Ro5"""
        result = lipinski_ro5(180.0, 1.2, 1, 4)
        assert result["passes_ro5"] is True
        assert result["violations"] == []
        assert result["drug_like_score"] == 4

    def test_high_mw_fails(self):
        """MW=600 violates MW > 500 rule"""
        result = lipinski_ro5(600.0, 2.0, 2, 5)
        assert "MW > 500" in result["violations"]
        assert result["passes_ro5"] is False
        assert result["drug_like_score"] == 3

    def test_high_logp_fails(self):
        """logP=6 violates logP > 5 rule"""
        result = lipinski_ro5(300.0, 6.0, 1, 4)
        assert "logP > 5" in result["violations"]
        assert result["passes_ro5"] is False

    def test_high_hbd_fails(self):
        """HBD=6 violates HBD > 5 rule"""
        result = lipinski_ro5(300.0, 2.0, 6, 4)
        assert "HBD > 5" in result["violations"]
        assert result["passes_ro5"] is False

    def test_high_hba_fails(self):
        """HBA=11 violates HBA > 10 rule"""
        result = lipinski_ro5(300.0, 2.0, 2, 11)
        assert "HBA > 10" in result["violations"]
        assert result["passes_ro5"] is False

    def test_two_violations_score(self):
        """Two violations → drug_like_score=2"""
        result = lipinski_ro5(600.0, 6.0, 2, 5)
        assert len(result["violations"]) == 2
        assert result["drug_like_score"] == 2

    def test_violations_is_list(self):
        """violations should always be a list"""
        result = lipinski_ro5(200.0, 1.0, 1, 3)
        assert isinstance(result["violations"], list)

    def test_boundary_mw_500_passes(self):
        """MW exactly 500 should pass"""
        result = lipinski_ro5(500.0, 2.0, 2, 5)
        assert "MW > 500" not in result["violations"]

    def test_boundary_hba_10_passes(self):
        """HBA exactly 10 should pass"""
        result = lipinski_ro5(300.0, 2.0, 2, 10)
        assert "HBA > 10" not in result["violations"]


# ──────────────────────────────────────────────
# veber_rules
# ──────────────────────────────────────────────


class TestVeberRules:
    def test_ideal_compound_passes(self):
        result = veber_rules(80.0, 5)
        assert result["passes_veber"] is True
        assert result["violations"] == []

    def test_high_psa_fails(self):
        """PSA=200 fails"""
        result = veber_rules(200.0, 5)
        assert "PSA > 140 A^2" in result["violations"]
        assert result["passes_veber"] is False

    def test_high_rotbonds_fails(self):
        """11 rotatable bonds fails"""
        result = veber_rules(80.0, 11)
        assert "Rotatable bonds > 10" in result["violations"]
        assert result["passes_veber"] is False

    def test_both_fail(self):
        result = veber_rules(200.0, 15)
        assert len(result["violations"]) == 2
        assert result["passes_veber"] is False


# ──────────────────────────────────────────────
# ghose_filter
# ──────────────────────────────────────────────


class TestGhoseFilter:
    def test_typical_drug_passes(self):
        result = ghose_filter(300.0, 2.0, 25, 80.0)
        assert result["passes_ghose"] is True
        assert result["violations"] == []

    def test_low_mw_fails(self):
        """MW=100 fails lower bound"""
        result = ghose_filter(100.0, 2.0, 25, 80.0)
        assert "MW not in [160, 480]" in result["violations"]
        assert result["passes_ghose"] is False

    def test_high_mw_fails(self):
        result = ghose_filter(500.0, 2.0, 25, 80.0)
        assert "MW not in [160, 480]" in result["violations"]

    def test_logp_out_of_range_fails(self):
        result = ghose_filter(300.0, 6.0, 25, 80.0)
        assert "logP not in [-0.4, 5.6]" in result["violations"]

    def test_n_atoms_out_of_range_fails(self):
        result = ghose_filter(300.0, 2.0, 15, 80.0)
        assert "n_atoms not in [20, 70]" in result["violations"]

    def test_mr_out_of_range_fails(self):
        result = ghose_filter(300.0, 2.0, 25, 30.0)
        assert "MR not in [40, 130]" in result["violations"]


# ──────────────────────────────────────────────
# drug_likeness_score
# ──────────────────────────────────────────────


class TestDrugLikenessScore:
    def test_ideal_drug_high_score(self):
        """Ideal drug: MW=300, logP=2, HBD=2, HBA=4, PSA=80, rotbonds=5"""
        result = drug_likeness_score(300.0, 2.0, 2, 4, 80.0, 5)
        # Max achievable ~55 (Lipinski 25 + Veber 25 + PSA bonus 5)
        assert result["score_0_100"] >= 50.0
        assert result["category"] in ("excellent", "good")

    def test_poor_drug_low_score(self):
        """Poor drug: high MW, high logP, many violations"""
        result = drug_likeness_score(800.0, 8.0, 8, 15, 200.0, 15)
        assert result["score_0_100"] < 40.0

    def test_score_never_negative(self):
        """Score always >= 0"""
        result = drug_likeness_score(1000.0, 10.0, 10, 15, 300.0, 20)
        assert result["score_0_100"] >= 0.0

    def test_score_never_above_100(self):
        """Score always <= 100"""
        result = drug_likeness_score(200.0, 1.0, 1, 2, 60.0, 3)
        assert result["score_0_100"] <= 100.0

    def test_returns_ro5_and_veber(self):
        result = drug_likeness_score(300.0, 2.0, 2, 4, 80.0, 5)
        assert "ro5_result" in result
        assert "veber_result" in result
        assert isinstance(result["ro5_result"], dict)
        assert isinstance(result["veber_result"], dict)

    def test_category_excellent(self):
        result = drug_likeness_score(250.0, 1.5, 1, 3, 70.0, 4)
        assert result["category"] == "excellent"

    def test_category_poor(self):
        result = drug_likeness_score(800.0, 8.0, 8, 15, 200.0, 15)
        assert result["category"] == "poor"


# ──────────────────────────────────────────────
# molar_refractivity_estimate
# ──────────────────────────────────────────────


class TestMolarRefractivityEstimate:
    def test_positive_for_valid_inputs(self):
        """MR should be positive for typical drug-like inputs"""
        mr = molar_refractivity_estimate(300.0, 2.0)
        assert mr > 0.0

    def test_aspirin_estimate(self):
        """Aspirin: MW=180, logP=1.2 → 0.2*180 + 5*1.2 + 2 = 36+6+2=44"""
        mr = molar_refractivity_estimate(180.0, 1.2)
        assert abs(mr - 44.0) < 1e-9

    def test_higher_mw_gives_higher_mr(self):
        mr1 = molar_refractivity_estimate(200.0, 2.0)
        mr2 = molar_refractivity_estimate(400.0, 2.0)
        assert mr2 > mr1

    def test_higher_logp_gives_higher_mr(self):
        mr1 = molar_refractivity_estimate(300.0, 1.0)
        mr2 = molar_refractivity_estimate(300.0, 3.0)
        assert mr2 > mr1
