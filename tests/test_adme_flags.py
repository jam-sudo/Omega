"""Tests for Phase 611 — ADME Liability Flagging."""

from __future__ import annotations

from omega_pbpk.prediction.adme_flags import ADMEFlag, ADMEFlagResult, flag_adme_liabilities

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {"absorption", "distribution", "metabolism", "excretion", "toxicity"}
VALID_SEVERITIES = {"info", "warning", "concern", "critical"}
VALID_RISKS = {"low", "moderate", "high", "very_high"}


def _good_drug(**kwargs):
    """Return a 'clean' drug with good physicochemical properties."""
    defaults = dict(
        drug_name="GoodDrug",
        logP=2.0,
        mw=300.0,
        psa=60.0,
        n_hbd=2,
        n_hba=4,
        solubility_mg_mL=5.0,
        n_aromatic_rings=1,
        n_rotatable_bonds=4,
        pka_acid=None,
        pka_base=None,
        fu_plasma=0.3,
        vd_L=50.0,
    )
    defaults.update(kwargs)
    return flag_adme_liabilities(**defaults)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_returns_result():
    result = _good_drug()
    assert isinstance(result, ADMEFlagResult)


def test_n_total_flags_matches():
    result = _good_drug()
    assert result.n_total_flags == len(result.flags)


def test_flags_are_adme_flag_instances():
    result = _good_drug(n_aromatic_rings=4, logP=6, mw=550, psa=150, solubility_mg_mL=0.05)
    for flag in result.flags:
        assert isinstance(flag, ADMEFlag)


def test_category_valid_values():
    result = _good_drug(n_aromatic_rings=4, logP=6, mw=550, psa=150, solubility_mg_mL=0.05)
    for flag in result.flags:
        assert flag.category in VALID_CATEGORIES


def test_severity_valid_values():
    result = _good_drug(n_aromatic_rings=4, logP=6, mw=550, psa=150, solubility_mg_mL=0.05)
    for flag in result.flags:
        assert flag.severity in VALID_SEVERITIES


def test_overall_risk_valid():
    result = _good_drug()
    assert result.overall_risk in VALID_RISKS


def test_notes_nonempty():
    result = _good_drug()
    assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# Clean drug
# ---------------------------------------------------------------------------


def test_clean_drug_no_flags():
    result = _good_drug()
    # A clean drug should have few or no flags
    assert result.n_total_flags <= 2


def test_good_drug_low_risk():
    result = _good_drug()
    assert result.overall_risk in {"low", "moderate"}


# ---------------------------------------------------------------------------
# Absorption flags
# ---------------------------------------------------------------------------


def test_low_solubility_critical():
    result = _good_drug(solubility_mg_mL=0.01)
    flag_names = [f.flag_name for f in result.flags]
    assert "low_solubility" in flag_names
    low_sol = next(f for f in result.flags if f.flag_name == "low_solubility")
    assert low_sol.severity == "critical"


def test_high_mw_flag():
    result = _good_drug(mw=600.0)
    flag_names = [f.flag_name for f in result.flags]
    assert "high_mw" in flag_names


def test_high_psa_flag():
    result = _good_drug(psa=150.0)
    flag_names = [f.flag_name for f in result.flags]
    assert "high_psa" in flag_names
    high_psa = next(f for f in result.flags if f.flag_name == "high_psa")
    assert high_psa.severity == "critical"


def test_excess_hbd_flag():
    result = _good_drug(n_hbd=6)
    flag_names = [f.flag_name for f in result.flags]
    assert "excess_hbd" in flag_names


def test_pass_absorption_false_when_critical():
    result = _good_drug(solubility_mg_mL=0.01)  # critical absorption flag
    assert result.pass_absorption is False


# ---------------------------------------------------------------------------
# Distribution flags
# ---------------------------------------------------------------------------


def test_high_protein_binding_flag():
    result = _good_drug(fu_plasma=0.02)
    flag_names = [f.flag_name for f in result.flags]
    assert "high_protein_binding" in flag_names
    hpb = next(f for f in result.flags if f.flag_name == "high_protein_binding")
    assert hpb.severity == "concern"


def test_cns_concern_for_large_polar():
    result = _good_drug(mw=500.0, psa=100.0)
    flag_names = [f.flag_name for f in result.flags]
    assert "cns_penetration_concern" in flag_names


# ---------------------------------------------------------------------------
# Metabolism flags
# ---------------------------------------------------------------------------


def test_pan_assay_flag():
    result = _good_drug(n_aromatic_rings=4)
    flag_names = [f.flag_name for f in result.flags]
    assert "pan_assay_interference" in flag_names


# ---------------------------------------------------------------------------
# Excretion flags
# ---------------------------------------------------------------------------


def test_excretion_flags_triggered():
    result = _good_drug(logP=4.0, mw=600.0)
    flag_names = [f.flag_name for f in result.flags]
    assert "bile_excretion_risk" in flag_names


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_in_0_100():
    result = _good_drug(n_aromatic_rings=4, logP=6, mw=550, psa=150, solubility_mg_mL=0.05)
    assert 0.0 <= result.developability_score <= 100.0


def test_developability_score_decreases_with_violations():
    clean = _good_drug()
    bad = _good_drug(
        logP=6,
        mw=600,
        psa=150,
        n_hbd=6,
        n_hba=11,
        solubility_mg_mL=0.05,
        n_aromatic_rings=4,
        fu_plasma=0.02,
    )
    assert bad.developability_score < clean.developability_score


def test_n_critical_counts_correctly():
    result = _good_drug(solubility_mg_mL=0.01, psa=150.0)
    expected = sum(1 for f in result.flags if f.severity == "critical")
    assert result.n_critical == expected


def test_n_concerns_counts_correctly():
    result = _good_drug(n_hbd=6, fu_plasma=0.02, n_hba=11)
    expected = sum(1 for f in result.flags if f.severity == "concern")
    assert result.n_concerns == expected


def test_critical_flags_high_risk():
    # Many critical flags → very_high risk
    result = _good_drug(
        solubility_mg_mL=0.01,  # critical
        psa=150.0,  # critical
        mw=750.0,  # critical
    )
    assert result.overall_risk == "very_high"


# ---------------------------------------------------------------------------
# Combined scenarios
# ---------------------------------------------------------------------------


def test_multiple_concerns_high_risk():
    result = _good_drug(logP=6.0, psa=150.0, mw=600.0)
    assert result.overall_risk in {"high", "very_high"}
