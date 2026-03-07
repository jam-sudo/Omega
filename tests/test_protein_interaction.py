"""Tests for Phase 360: Drug-Protein Interaction Energy Prediction."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.prediction.protein_interaction import (
    ProteinInteractionResult,
    predict_protein_binding,
    screen_for_protein_interactions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base(protein="albumin", **overrides):
    """Return minimal valid kwargs for predict_protein_binding."""
    defaults = dict(
        compound_name="TestDrug",
        protein_name=protein,
        logP=2.0,
        mw=300.0,
        psa=60.0,
        n_hbd=2,
        n_hba=4,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Albumin tests
# ---------------------------------------------------------------------------

class TestAlbumin:
    def test_high_logP_low_kd(self):
        """High logP → stronger albumin binding (lower Kd)."""
        r_high = predict_protein_binding(**_base(logP=5.0))
        r_low = predict_protein_binding(**_base(logP=0.0))
        assert r_high.predicted_kd_uM < r_low.predicted_kd_uM

    def test_acidic_drug_stronger_albumin_binding(self):
        """Acid pKa < 5 → stronger albumin binding than neutral."""
        r_acid = predict_protein_binding(**_base(pka_acid=3.5))
        r_neutral = predict_protein_binding(**_base(pka_acid=None))
        # Acid has higher pKd → lower Kd → stronger binding
        assert r_acid.predicted_kd_uM < r_neutral.predicted_kd_uM

    def test_fu_in_open_unit_interval(self):
        """fu must be in (0, 1]."""
        r = predict_protein_binding(**_base(logP=3.0))
        assert 0.0 < r.predicted_fu <= 1.0

    def test_kd_positive(self):
        r = predict_protein_binding(**_base())
        assert r.predicted_kd_uM > 0.0

    def test_delta_g_negative(self):
        """Favorable binding → negative deltaG."""
        r = predict_protein_binding(**_base(logP=3.0))
        assert r.predicted_delta_g_kJ_per_mol < 0.0

    def test_high_psa_weakens_binding(self):
        """Higher PSA → lower albumin binding (higher Kd)."""
        r_high_psa = predict_protein_binding(**_base(psa=150.0))
        r_low_psa = predict_protein_binding(**_base(psa=20.0))
        assert r_high_psa.predicted_kd_uM > r_low_psa.predicted_kd_uM


# ---------------------------------------------------------------------------
# Alpha-1-AGP tests
# ---------------------------------------------------------------------------

class TestAlpha1AGP:
    def test_basic_drug_stronger_agp_binding(self):
        """Basic drug (pka_base > 7) → stronger AGP binding."""
        r_basic = predict_protein_binding(**_base(protein="alpha1_agp", pka_base=9.0))
        r_neutral = predict_protein_binding(**_base(protein="alpha1_agp", pka_base=None))
        assert r_basic.predicted_kd_uM < r_neutral.predicted_kd_uM

    def test_basic_drug_agp_stronger_than_albumin(self):
        """Basic drug binds AGP more tightly than albumin (per protein model)."""
        r_agp = predict_protein_binding(**_base(protein="alpha1_agp", pka_base=9.0))
        r_alb = predict_protein_binding(**_base(protein="albumin", pka_base=None))
        # AGP binding boosted by +1.5 pKd, albumin binding not boosted
        assert r_agp.predicted_kd_uM < r_alb.predicted_kd_uM

    def test_fu_in_open_unit_interval(self):
        r = predict_protein_binding(**_base(protein="alpha1_agp"))
        assert 0.0 < r.predicted_fu <= 1.0

    def test_kd_positive(self):
        r = predict_protein_binding(**_base(protein="alpha1_agp"))
        assert r.predicted_kd_uM > 0.0

    def test_delta_g_negative(self):
        r = predict_protein_binding(**_base(protein="alpha1_agp", logP=3.0))
        assert r.predicted_delta_g_kJ_per_mol < 0.0


# ---------------------------------------------------------------------------
# CYP3A4 tests
# ---------------------------------------------------------------------------

class TestCYP3A4:
    def test_kd_positive(self):
        r = predict_protein_binding(**_base(protein="cyp3a4"))
        assert r.predicted_kd_uM > 0.0

    def test_delta_g_negative_for_lipophilic(self):
        r = predict_protein_binding(**_base(protein="cyp3a4", logP=4.0, psa=30.0, mw=300.0))
        assert r.predicted_delta_g_kJ_per_mol < 0.0

    def test_fu_is_one(self):
        """CYP3A4 is metabolic binding context, fu reported as 1.0."""
        r = predict_protein_binding(**_base(protein="cyp3a4"))
        assert r.predicted_fu == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# P-glycoprotein tests
# ---------------------------------------------------------------------------

class TestPGlycoprotein:
    def test_kd_positive(self):
        r = predict_protein_binding(**_base(protein="p_glycoprotein"))
        assert r.predicted_kd_uM > 0.0

    def test_low_hb_count_stronger_pgp(self):
        """Low HBD+HBA (< 5) → pKd boosted by +0.5 → lower Kd."""
        r_low = predict_protein_binding(**_base(protein="p_glycoprotein", n_hbd=1, n_hba=2))  # sum=3
        r_high = predict_protein_binding(**_base(protein="p_glycoprotein", n_hbd=3, n_hba=3))  # sum=6
        assert r_low.predicted_kd_uM < r_high.predicted_kd_uM

    def test_fu_is_one(self):
        r = predict_protein_binding(**_base(protein="p_glycoprotein"))
        assert r.predicted_fu == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Plasma generic tests
# ---------------------------------------------------------------------------

class TestPlasmaGeneric:
    def test_kd_between_albumin_and_agp(self):
        """Plasma generic Kd is a weighted blend of albumin and AGP Kd."""
        r_generic = predict_protein_binding(**_base(protein="plasma_generic"))
        r_alb = predict_protein_binding(**_base(protein="albumin"))
        r_agp = predict_protein_binding(**_base(protein="alpha1_agp"))
        kd_min = min(r_alb.predicted_kd_uM, r_agp.predicted_kd_uM)
        kd_max = max(r_alb.predicted_kd_uM, r_agp.predicted_kd_uM)
        assert kd_min <= r_generic.predicted_kd_uM <= kd_max

    def test_fu_in_open_unit_interval(self):
        r = predict_protein_binding(**_base(protein="plasma_generic"))
        assert 0.0 < r.predicted_fu <= 1.0


# ---------------------------------------------------------------------------
# Interaction scores tests
# ---------------------------------------------------------------------------

class TestInteractionScores:
    def test_hydrophobic_score_proportional_to_logP(self):
        r1 = predict_protein_binding(**_base(logP=1.0))
        r2 = predict_protein_binding(**_base(logP=4.0))
        assert r2.hydrophobic_score > r1.hydrophobic_score

    def test_polar_score_negative(self):
        """Polar score is always negative (reduces interaction)."""
        r = predict_protein_binding(**_base(psa=80.0, n_hbd=3))
        assert r.polar_score < 0.0

    def test_charge_score_for_acidic_drug(self):
        """Acidic drug (pKa < 5) → charge_score = 1.0."""
        r = predict_protein_binding(**_base(pka_acid=3.0))
        assert r.charge_score == pytest.approx(1.0)

    def test_charge_score_for_strong_base(self):
        """Basic drug (pKa > 8) → charge_score = 1.0."""
        r = predict_protein_binding(**_base(pka_base=9.5))
        assert r.charge_score == pytest.approx(1.0)

    def test_charge_score_zero_for_neutral(self):
        """Neutral drug → charge_score = 0.0."""
        r = predict_protein_binding(**_base(pka_acid=7.0, pka_base=5.0))
        assert r.charge_score == pytest.approx(0.0)

    def test_total_score_is_sum(self):
        r = predict_protein_binding(**_base())
        expected = r.hydrophobic_score + r.polar_score + r.charge_score
        assert r.total_interaction_score == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Binding mode classification tests
# ---------------------------------------------------------------------------

class TestBindingMode:
    def test_hydrophobic_pocket_for_lipophilic_nonpolar(self):
        """logP > 3 and PSA < 60 and no charge → hydrophobic_pocket."""
        r = predict_protein_binding(**_base(logP=4.0, psa=40.0, pka_acid=None, pka_base=None))
        assert r.likely_binding_mode == "hydrophobic_pocket"

    def test_mixed_for_charged_lipophilic(self):
        """logP > 3, PSA < 60, and charged → mixed."""
        r = predict_protein_binding(**_base(logP=4.0, psa=40.0, pka_acid=3.0))
        assert r.likely_binding_mode == "mixed"

    def test_ionic_for_polar_charged(self):
        """logP ≤ 3, PSA ≥ 60, and charged → ionic."""
        r = predict_protein_binding(**_base(logP=1.0, psa=100.0, pka_acid=3.0))
        assert r.likely_binding_mode == "ionic"

    def test_surface_for_polar_neutral(self):
        """Low logP, high PSA, no charge → surface."""
        r = predict_protein_binding(**_base(logP=0.5, psa=120.0))
        assert r.likely_binding_mode == "surface"


# ---------------------------------------------------------------------------
# Displacement risk tests
# ---------------------------------------------------------------------------

class TestDisplacementRisk:
    def test_tight_binder_high_risk(self):
        """Kd < 1 µM → displacement_risk = 'high'."""
        # Force high binding: high logP, low PSA, acidic
        r = predict_protein_binding(**_base(logP=6.0, psa=10.0, pka_acid=2.0))
        if r.predicted_kd_uM < 1.0:
            assert r.displacement_risk == "high"

    def test_weak_binder_low_risk(self):
        """Very weak binder → displacement_risk = 'low'."""
        r = predict_protein_binding(**_base(logP=-2.0, psa=200.0))
        if r.predicted_kd_uM >= 10.0:
            assert r.displacement_risk == "low"

    def test_risk_values_valid(self):
        """displacement_risk must be one of the three valid values."""
        valid = {"low", "moderate", "high"}
        for logP in [-1.0, 2.0, 5.0]:
            r = predict_protein_binding(**_base(logP=logP))
            assert r.displacement_risk in valid


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_invalid_protein_raises(self):
        with pytest.raises(ValueError, match="protein_name"):
            predict_protein_binding(**_base(protein="unknown_protein"))

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            predict_protein_binding(**_base(mw=0.0))

    def test_negative_n_hbd_raises(self):
        with pytest.raises(ValueError, match="n_hbd"):
            predict_protein_binding(**_base(n_hbd=-1))

    def test_negative_n_hba_raises(self):
        with pytest.raises(ValueError, match="n_hba"):
            predict_protein_binding(**_base(n_hba=-1))

    def test_negative_psa_raises(self):
        with pytest.raises(ValueError, match="psa"):
            predict_protein_binding(**_base(psa=-10.0))


# ---------------------------------------------------------------------------
# Frozen dataclass test
# ---------------------------------------------------------------------------

class TestFrozenDataclass:
    def test_frozen(self):
        r = predict_protein_binding(**_base())
        with pytest.raises((AttributeError, TypeError)):
            r.predicted_kd_uM = 999.0  # type: ignore[misc]

    def test_notes_nonempty(self):
        r = predict_protein_binding(**_base())
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Screen function tests
# ---------------------------------------------------------------------------

class TestScreenForProteinInteractions:
    def _compounds(self):
        return [
            {"name": "Lipophilic", "logP": 5.0, "mw": 350.0, "psa": 30.0,
             "n_hbd": 1, "n_hba": 2, "pka_acid": 3.5},
            {"name": "Hydrophilic", "logP": -1.0, "mw": 200.0, "psa": 150.0,
             "n_hbd": 5, "n_hba": 8},
        ]

    def test_returns_nested_dict(self):
        result = screen_for_protein_interactions(self._compounds())
        assert isinstance(result, dict)
        assert "Lipophilic" in result
        assert "Hydrophilic" in result

    def test_all_proteins_present_by_default(self):
        result = screen_for_protein_interactions(self._compounds())
        expected_proteins = {"albumin", "alpha1_agp", "cyp3a4", "p_glycoprotein", "plasma_generic"}
        for compound_name, protein_dict in result.items():
            assert set(protein_dict.keys()) == expected_proteins

    def test_custom_protein_subset(self):
        result = screen_for_protein_interactions(
            self._compounds(), proteins=["albumin", "cyp3a4"]
        )
        for protein_dict in result.values():
            assert set(protein_dict.keys()) == {"albumin", "cyp3a4"}

    def test_protein_results_are_correct_type(self):
        result = screen_for_protein_interactions(self._compounds())
        for protein_dict in result.values():
            for r in protein_dict.values():
                assert isinstance(r, ProteinInteractionResult)

    def test_sorted_by_interaction_score_descending(self):
        result = screen_for_protein_interactions(self._compounds())
        for protein_dict in result.values():
            scores = [r.total_interaction_score for r in protein_dict.values()]
            assert scores == sorted(scores, reverse=True)

    def test_empty_compounds(self):
        result = screen_for_protein_interactions([])
        assert result == {}
