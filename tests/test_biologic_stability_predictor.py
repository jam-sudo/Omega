"""Tests for biologic stability predictor — Phase 456."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.biologic_stability_predictor import (
    ChemicalStabilityResult,
    ColloidalStabilityResult,
    ThermalStabilityResult,
    assess_overall_stability,
    predict_chemical_stability,
    predict_colloidal_stability,
    predict_thermal_stability,
)

# ---------------------------------------------------------------------------
# ThermalStability
# ---------------------------------------------------------------------------


class TestPredictThermalStability:
    def test_more_disulfides_higher_tm(self):
        """Each disulfide bond raises predicted Tm."""
        result_low = predict_thermal_stability(150.0, 7.0, n_disulfide_bonds=2)
        result_high = predict_thermal_stability(150.0, 7.0, n_disulfide_bonds=8)
        assert result_high.tm_C_predicted > result_low.tm_C_predicted

    def test_glycosylation_improves_tm(self):
        """Glycosylation adds to Tm."""
        no_glycan = predict_thermal_stability(150.0, 7.0, n_disulfide_bonds=4, glycosylated=False)
        glycan = predict_thermal_stability(150.0, 7.0, n_disulfide_bonds=4, glycosylated=True)
        assert glycan.tm_C_predicted > no_glycan.tm_C_predicted

    def test_extreme_pi_reduces_tm(self):
        """Extreme pI (< 5 or > 9) should reduce Tm."""
        neutral = predict_thermal_stability(150.0, 7.0, n_disulfide_bonds=4)
        acidic_pi = predict_thermal_stability(150.0, 4.0, n_disulfide_bonds=4)
        assert acidic_pi.tm_C_predicted < neutral.tm_C_predicted

    def test_high_stability_category_above_70(self):
        """Molecules with Tm ≥ 70 °C get 'high' stability."""
        result = predict_thermal_stability(150.0, 7.0, n_disulfide_bonds=10)
        # Base=60, +15, total 75 → high
        assert result.stability_category == "high"

    def test_low_stability_category(self):
        """Molecules with Tm < 60 °C get 'low' stability."""
        result = predict_thermal_stability(5.0, 4.0, n_disulfide_bonds=0)
        # Base=60, −4 (extreme pI), small MW bonus ≈ 0, total 56 → low
        assert result.stability_category == "low"

    def test_returns_thermal_stability_result(self):
        result = predict_thermal_stability(150.0, 8.0, n_disulfide_bonds=4)
        assert isinstance(result, ThermalStabilityResult)

    def test_raises_on_nonpositive_mw(self):
        with pytest.raises(ValueError, match="mw_kDa"):
            predict_thermal_stability(0.0, 7.0, n_disulfide_bonds=4)

    def test_raises_on_invalid_pi(self):
        with pytest.raises(ValueError, match="isoelectric_point"):
            predict_thermal_stability(150.0, 15.0, n_disulfide_bonds=4)

    def test_raises_on_negative_disulfide(self):
        with pytest.raises(ValueError, match="n_disulfide_bonds"):
            predict_thermal_stability(150.0, 7.0, n_disulfide_bonds=-1)

    def test_notes_string_nonempty(self):
        result = predict_thermal_stability(150.0, 7.0, n_disulfide_bonds=4)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# ChemicalStability
# ---------------------------------------------------------------------------


class TestPredictChemicalStability:
    def test_high_asn_gives_high_deamidation_risk(self):
        """Many Asn residues → high deamidation risk."""
        result = predict_chemical_stability(n_asp=2, n_asn=10, n_met=1, n_trp=1)
        assert result.deamidation_risk == "high"

    def test_low_asn_gives_low_deamidation_risk(self):
        result = predict_chemical_stability(n_asp=2, n_asn=1, n_met=1, n_trp=1)
        assert result.deamidation_risk == "low"

    def test_high_met_gives_high_oxidation_risk(self):
        """Many Met residues → high oxidation risk."""
        result = predict_chemical_stability(n_asp=2, n_asn=1, n_met=8, n_trp=0)
        assert result.oxidation_risk == "high"

    def test_high_trp_contributes_to_oxidation(self):
        result = predict_chemical_stability(n_asp=2, n_asn=1, n_met=1, n_trp=6)
        assert result.oxidation_risk in ("moderate", "high")

    def test_low_ph_increases_deamidation_risk(self):
        """pH < 5 shifts deamidation risk up."""
        normal = predict_chemical_stability(n_asp=2, n_asn=2, n_met=1, n_trp=0, pH=7.4)
        acidic = predict_chemical_stability(n_asp=2, n_asn=2, n_met=1, n_trp=0, pH=4.0)
        risks = ("low", "moderate", "high")
        assert risks.index(acidic.deamidation_risk) >= risks.index(normal.deamidation_risk)

    def test_high_temperature_increases_oxidation(self):
        """T > 40 °C shifts oxidation risk up."""
        normal = predict_chemical_stability(n_asp=1, n_asn=1, n_met=2, n_trp=0, temperature_C=25.0)
        hot = predict_chemical_stability(n_asp=1, n_asn=1, n_met=2, n_trp=0, temperature_C=45.0)
        risks = ("low", "moderate", "high")
        assert risks.index(hot.oxidation_risk) >= risks.index(normal.oxidation_risk)

    def test_overall_risk_worst_of_three(self):
        """overall_chemical_risk = worst among deamidation/oxidation/hydrolysis."""
        result = predict_chemical_stability(n_asp=2, n_asn=10, n_met=1, n_trp=1)
        risks = ("low", "moderate", "high")
        max_idx = max(
            risks.index(result.deamidation_risk),
            risks.index(result.oxidation_risk),
            risks.index(result.hydrolysis_risk),
        )
        assert result.overall_chemical_risk == risks[max_idx]

    def test_returns_chemical_stability_result(self):
        result = predict_chemical_stability(n_asp=3, n_asn=3, n_met=2, n_trp=1)
        assert isinstance(result, ChemicalStabilityResult)

    def test_raises_on_negative_residue_count(self):
        with pytest.raises(ValueError):
            predict_chemical_stability(n_asp=-1, n_asn=2, n_met=1, n_trp=0)

    def test_raises_on_invalid_ph(self):
        with pytest.raises(ValueError, match="pH"):
            predict_chemical_stability(n_asp=2, n_asn=2, n_met=1, n_trp=0, pH=15.0)


# ---------------------------------------------------------------------------
# ColloidalStability
# ---------------------------------------------------------------------------


class TestPredictColloidalStability:
    def test_ph_near_pi_gives_poor_stability(self):
        """pH ≈ pI → very small zeta potential → unstable."""
        result = predict_colloidal_stability(
            mw_kDa=150.0, isoelectric_point=7.0, pH=7.0, ionic_strength_mM=10.0
        )
        assert result.colloidal_stability == "unstable"

    def test_ph_far_from_pi_gives_stable(self):
        """Large |pH - pI| → large zeta → stable."""
        result = predict_colloidal_stability(
            mw_kDa=150.0, isoelectric_point=9.0, pH=4.0, ionic_strength_mM=10.0
        )
        assert result.colloidal_stability == "stable"

    def test_high_ionic_strength_reduces_stability(self):
        """Higher ionic strength screens charge → lower |zeta| → worse stability."""
        low_salt = predict_colloidal_stability(150.0, 9.0, pH=4.0, ionic_strength_mM=10.0)
        high_salt = predict_colloidal_stability(150.0, 9.0, pH=4.0, ionic_strength_mM=2000.0)
        assert abs(low_salt.zeta_potential_mV) > abs(high_salt.zeta_potential_mV)

    def test_zeta_sign_above_pi(self):
        """At pH > pI, molecule is negatively charged → zeta < 0 (pI - pH < 0)."""
        # pI=6, pH=8 → pI - pH = -2 → raw_zeta = 8 * (-2) = -16 < 0
        result = predict_colloidal_stability(150.0, 6.0, pH=8.0, ionic_strength_mM=10.0)
        assert result.zeta_potential_mV < 0.0

    def test_zeta_sign_below_pi(self):
        """At pH < pI, molecule is positively charged → zeta > 0 (pI - pH > 0)."""
        # pI=9, pH=5 → pI - pH = 4 → raw_zeta = 8 * 4 = 32 > 0
        result = predict_colloidal_stability(150.0, 9.0, pH=5.0, ionic_strength_mM=10.0)
        assert result.zeta_potential_mV > 0.0

    def test_returns_colloidal_stability_result(self):
        result = predict_colloidal_stability(150.0, 8.0, pH=6.0)
        assert isinstance(result, ColloidalStabilityResult)

    def test_raises_on_nonpositive_mw(self):
        with pytest.raises(ValueError, match="mw_kDa"):
            predict_colloidal_stability(0.0, 7.0)

    def test_raises_on_invalid_ionic_strength(self):
        with pytest.raises(ValueError, match="ionic_strength"):
            predict_colloidal_stability(150.0, 7.0, ionic_strength_mM=-10.0)

    def test_notes_nonempty(self):
        result = predict_colloidal_stability(150.0, 7.0, pH=6.0)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# assess_overall_stability
# ---------------------------------------------------------------------------


class TestAssessOverallStability:
    def test_returns_all_sub_results(self):
        result = assess_overall_stability(
            mw_kDa=150.0,
            isoelectric_point=7.0,
            n_disulfide_bonds=4,
            n_asp=5,
            n_asn=3,
            n_met=2,
            n_trp=1,
        )
        assert "thermal" in result
        assert "chemical" in result
        assert "colloidal" in result
        assert "overall_risk" in result

    def test_thermal_is_correct_type(self):
        result = assess_overall_stability(
            mw_kDa=150.0,
            isoelectric_point=7.0,
            n_disulfide_bonds=4,
            n_asp=2,
            n_asn=2,
            n_met=1,
            n_trp=0,
        )
        assert isinstance(result["thermal"], ThermalStabilityResult)

    def test_chemical_is_correct_type(self):
        result = assess_overall_stability(
            mw_kDa=150.0,
            isoelectric_point=7.0,
            n_disulfide_bonds=4,
            n_asp=2,
            n_asn=2,
            n_met=1,
            n_trp=0,
        )
        assert isinstance(result["chemical"], ChemicalStabilityResult)

    def test_colloidal_is_correct_type(self):
        result = assess_overall_stability(
            mw_kDa=150.0,
            isoelectric_point=7.0,
            n_disulfide_bonds=4,
            n_asp=2,
            n_asn=2,
            n_met=1,
            n_trp=0,
        )
        assert isinstance(result["colloidal"], ColloidalStabilityResult)

    def test_overall_risk_valid_string(self):
        result = assess_overall_stability(
            mw_kDa=150.0,
            isoelectric_point=7.0,
            n_disulfide_bonds=4,
            n_asp=2,
            n_asn=2,
            n_met=1,
            n_trp=0,
        )
        assert result["overall_risk"] in ("low", "moderate", "high")

    def test_high_instability_gives_high_risk(self):
        """Molecule with many unstable features → overall high risk."""
        result = assess_overall_stability(
            mw_kDa=5.0,
            isoelectric_point=4.0,  # extreme pI → thermal low, colloidal bad at pH 7.4
            n_disulfide_bonds=0,
            n_asp=10,
            n_asn=10,
            n_met=10,
            n_trp=5,
            pH=7.0,
        )
        assert result["overall_risk"] == "high"

    def test_stable_molecule_low_risk(self):
        """Well-behaved mAb-like molecule → overall low or moderate risk.

        Uses pI=9.0 and pH=5.5 so |pI - pH| = 3.5 → zeta ≈ 28 mV (stable).
        High disulfide count → Tm > 70 (thermal high → thermal risk low).
        Low residue counts → chemical risk low.
        """
        result = assess_overall_stability(
            mw_kDa=150.0,
            isoelectric_point=9.0,  # pI far from pH 5.5 → large zeta
            n_disulfide_bonds=12,  # high Tm
            n_asp=2,
            n_asn=1,
            n_met=1,
            n_trp=0,
            pH=5.5,
        )
        assert result["overall_risk"] in ("low", "moderate")
