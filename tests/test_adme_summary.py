"""Tests for comprehensive ADME summary prediction module (Phase 422)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.adme_summary import ADMESummary, predict_adme_summary

# ---------------------------------------------------------------------------
# Shared baseline: ibuprofen-like (BCS II, acid, CYP2C9)
# ---------------------------------------------------------------------------
IBUPROFEN_LIKE = dict(
    drug_name="Ibuprofen",
    mw=206.0,
    logP=3.97,
    psa=37.3,
    n_hbd=1,
    n_hba=2,
    pka=4.4,
    molecule_type="acid",
    fu_plasma=0.01,
    logD_pH74=0.97,
)

# BCS I example: caffeine-like
CAFFEINE_LIKE = dict(
    drug_name="Caffeine",
    mw=194.0,
    logP=-0.07,
    psa=58.4,
    n_hbd=0,
    n_hba=6,
    pka=14.0,
    molecule_type="neutral",
    fu_plasma=0.65,
    logD_pH74=-0.07,
)

# BCS III example: highly hydrophilic
HYDROPHILIC = dict(
    drug_name="Hydrophilic",
    mw=300.0,
    logP=-2.0,
    psa=80.0,
    n_hbd=2,
    n_hba=5,
    pka=8.0,
    molecule_type="neutral",
    fu_plasma=0.5,
)

# Lipinski violator
LARGE_MOLECULE = dict(
    drug_name="LargeMol",
    mw=700.0,
    logP=6.0,
    psa=180.0,
    n_hbd=7,
    n_hba=12,
    pka=7.0,
    molecule_type="neutral",
    fu_plasma=0.02,
)


# ---------------------------------------------------------------------------
# Dataclass / return type tests
# ---------------------------------------------------------------------------
class TestADMESummaryType:
    def test_returns_adme_summary(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert isinstance(result, ADMESummary)

    def test_has_all_fields(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        fields = [
            "drug_name", "mw", "logP", "psa", "n_hbd", "n_hba",
            "bcs_class", "lipinski_pass", "absorption_class", "absorption_notes",
            "vd_estimated_L_per_kg", "ppb_pct", "cns_penetration",
            "primary_cyp", "excretion_route", "t_half_estimated_h",
            "risk_flags", "overall_adme_score", "notes",
        ]
        for f in fields:
            assert hasattr(result, f), f"Missing field: {f}"

    def test_risk_flags_is_tuple(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert isinstance(result.risk_flags, tuple)

    def test_drug_name_stored(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert result.drug_name == "Caffeine"

    def test_mw_stored(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert result.mw == pytest.approx(194.0)

    def test_logP_stored(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert result.logP == pytest.approx(-0.07)


# ---------------------------------------------------------------------------
# BCS classification tests
# ---------------------------------------------------------------------------
class TestBCSClassification:
    def test_bcs_class_i_high_perm_high_sol(self):
        # Low MW, low logP, low PSA → BCS I
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert result.bcs_class == "I"

    def test_bcs_class_ii_high_perm_low_sol(self):
        # High logP → low solubility; PSA low → high perm → BCS II
        result = predict_adme_summary(**IBUPROFEN_LIKE)
        assert result.bcs_class == "II"

    def test_bcs_class_iii_low_perm_high_sol(self):
        # logP < -1 → low perm; logP < 2 and MW < 500 → high sol → BCS III
        result = predict_adme_summary(**HYDROPHILIC)
        assert result.bcs_class == "III"

    def test_bcs_class_iv_low_perm_low_sol(self):
        # High PSA and high MW and high logP
        result = predict_adme_summary(**LARGE_MOLECULE)
        assert result.bcs_class == "IV"

    def test_absorption_class_high_for_bcs_i(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert result.absorption_class == "high"

    def test_absorption_class_low_for_bcs_iii(self):
        result = predict_adme_summary(**HYDROPHILIC)
        assert result.absorption_class == "low"


# ---------------------------------------------------------------------------
# Lipinski Rule of Five tests
# ---------------------------------------------------------------------------
class TestLipinski:
    def test_caffeine_lipinski_pass(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert result.lipinski_pass is True

    def test_large_molecule_lipinski_fail(self):
        result = predict_adme_summary(**LARGE_MOLECULE)
        assert result.lipinski_pass is False

    def test_borderline_mw_500_pass(self):
        params = {**CAFFEINE_LIKE, "drug_name": "Border", "mw": 500.0, "logP": 4.0,
                  "n_hbd": 4, "n_hba": 9}
        result = predict_adme_summary(**params)
        assert result.lipinski_pass is True

    def test_hbd_violation(self):
        params = {**CAFFEINE_LIKE, "drug_name": "HBDViol", "n_hbd": 6}
        result = predict_adme_summary(**params)
        assert result.lipinski_pass is False


# ---------------------------------------------------------------------------
# Distribution tests
# ---------------------------------------------------------------------------
class TestDistribution:
    def test_vd_in_physiological_range(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert 0.04 <= result.vd_estimated_L_per_kg <= 20.0

    def test_ppb_from_fu(self):
        result = predict_adme_summary(**IBUPROFEN_LIKE)
        assert result.ppb_pct == pytest.approx(99.0, rel=0.01)  # fu=0.01 → 99%

    def test_cns_penetration_lipophilic_small(self):
        # logP=2, PSA=50, MW=300 → should penetrate CNS
        params = dict(
            drug_name="CNSDrug",
            mw=300.0, logP=2.0, psa=50.0,
            n_hbd=1, n_hba=3, pka=8.0,
            molecule_type="base", fu_plasma=0.2,
        )
        result = predict_adme_summary(**params)
        assert result.cns_penetration is True

    def test_no_cns_penetration_polar_large(self):
        result = predict_adme_summary(**LARGE_MOLECULE)
        assert result.cns_penetration is False

    def test_high_fu_increases_vd(self):
        lo_fu = predict_adme_summary(**{**CAFFEINE_LIKE, "fu_plasma": 0.05})
        hi_fu = predict_adme_summary(**{**CAFFEINE_LIKE, "fu_plasma": 0.9})
        assert hi_fu.vd_estimated_L_per_kg > lo_fu.vd_estimated_L_per_kg


# ---------------------------------------------------------------------------
# Metabolism & Excretion tests
# ---------------------------------------------------------------------------
class TestMetabolismExcretion:
    def test_high_logP_gives_cyp3a4(self):
        params = {**CAFFEINE_LIKE, "drug_name": "Lipo", "logP": 4.0}
        result = predict_adme_summary(**params)
        assert result.primary_cyp == "CYP3A4"

    def test_negative_logP_gives_renal(self):
        result = predict_adme_summary(**HYDROPHILIC)
        assert result.primary_cyp == "renal"

    def test_acid_cyp2c9(self):
        params = dict(
            drug_name="Acid",
            mw=250.0, logP=1.0, psa=60.0,
            n_hbd=1, n_hba=3, pka=4.0,
            molecule_type="acid", fu_plasma=0.2,
        )
        result = predict_adme_summary(**params)
        assert result.primary_cyp == "CYP2C9"

    def test_basic_amine_cyp2d6(self):
        params = dict(
            drug_name="Base",
            mw=250.0, logP=2.0, psa=40.0,
            n_hbd=1, n_hba=3, pka=9.0,
            molecule_type="base", fu_plasma=0.3,
        )
        result = predict_adme_summary(**params)
        assert result.primary_cyp == "CYP2D6"

    def test_excretion_renal_for_hydrophilic(self):
        result = predict_adme_summary(**HYDROPHILIC)
        assert result.excretion_route == "renal"

    def test_excretion_hepatic_for_lipophilic(self):
        result = predict_adme_summary(**IBUPROFEN_LIKE)
        assert result.excretion_route == "hepatic"

    def test_t_half_positive(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert result.t_half_estimated_h > 0

    def test_t_half_in_range(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert 0.5 <= result.t_half_estimated_h <= 200.0


# ---------------------------------------------------------------------------
# Risk flags tests
# ---------------------------------------------------------------------------
class TestRiskFlags:
    def test_large_molecule_has_poor_absorption_flag(self):
        result = predict_adme_summary(**LARGE_MOLECULE)
        assert "poor_absorption" in result.risk_flags

    def test_cns_drug_may_have_narrow_ti_flag(self):
        # Very lipophilic + CNS active
        params = dict(
            drug_name="NarrowTI",
            mw=350.0, logP=5.5, psa=50.0,
            n_hbd=1, n_hba=3, pka=8.0,
            molecule_type="base", fu_plasma=0.01,
        )
        result = predict_adme_summary(**params)
        assert "narrow_TI_risk" in result.risk_flags

    def test_non_cns_large_polar_has_poor_cns_flag(self):
        result = predict_adme_summary(**LARGE_MOLECULE)
        assert "poor_CNS_penetration" in result.risk_flags


# ---------------------------------------------------------------------------
# Overall ADME score tests
# ---------------------------------------------------------------------------
class TestADMEScore:
    def test_score_in_range(self):
        for params in [CAFFEINE_LIKE, IBUPROFEN_LIKE, HYDROPHILIC, LARGE_MOLECULE]:
            result = predict_adme_summary(**params)
            assert 0.0 <= result.overall_adme_score <= 100.0

    def test_bcs_i_scores_higher_than_bcs_iv(self):
        bcs1 = predict_adme_summary(**CAFFEINE_LIKE)
        bcs4 = predict_adme_summary(**LARGE_MOLECULE)
        assert bcs1.overall_adme_score > bcs4.overall_adme_score

    def test_notes_nonempty(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert len(result.notes) > 0

    def test_absorption_notes_nonempty(self):
        result = predict_adme_summary(**CAFFEINE_LIKE)
        assert len(result.absorption_notes) > 0


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError):
            predict_adme_summary(**{**CAFFEINE_LIKE, "drug_name": ""})

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError):
            predict_adme_summary(**{**CAFFEINE_LIKE, "mw": 0})

    def test_negative_psa_raises(self):
        with pytest.raises(ValueError):
            predict_adme_summary(**{**CAFFEINE_LIKE, "psa": -1})

    def test_negative_hbd_raises(self):
        with pytest.raises(ValueError):
            predict_adme_summary(**{**CAFFEINE_LIKE, "n_hbd": -1})

    def test_negative_hba_raises(self):
        with pytest.raises(ValueError):
            predict_adme_summary(**{**CAFFEINE_LIKE, "n_hba": -1})

    def test_zero_fu_raises(self):
        with pytest.raises(ValueError):
            predict_adme_summary(**{**CAFFEINE_LIKE, "fu_plasma": 0})

    def test_fu_above_one_raises(self):
        with pytest.raises(ValueError):
            predict_adme_summary(**{**CAFFEINE_LIKE, "fu_plasma": 1.1})

    def test_invalid_molecule_type_raises(self):
        with pytest.raises(ValueError):
            predict_adme_summary(**{**CAFFEINE_LIKE, "molecule_type": "amphoteric"})

    def test_explicit_logD_used(self):
        # Should not raise; logD_pH74 overrides estimation
        result = predict_adme_summary(**{**CAFFEINE_LIKE, "logD_pH74": -1.0})
        assert isinstance(result, ADMESummary)
