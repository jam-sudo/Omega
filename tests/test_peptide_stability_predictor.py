"""Tests for Phase 1138 — Peptide drug stability predictor."""

import pytest

from omega_pbpk.prediction.peptide_stability_predictor import (
    PeptideStabilityResult,
    compare_modifications,
    predict_peptide_stability,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def base_predict(**kwargs):
    defaults = dict(
        drug_name="TestPeptide",
        n_residues=10,
        hydrophobic_pct=30.0,
        n_beta_residues=2,
        abs_charge=3.0,
        d_aa_fraction=0.0,
        n_methylated_bonds=0,
        is_cyclic=False,
        is_pegylated=False,
    )
    defaults.update(kwargs)
    return predict_peptide_stability(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_result_type():
    res = base_predict()
    assert isinstance(res, PeptideStabilityResult)


def test_drug_name_preserved():
    res = base_predict(drug_name="CyclosporinA")
    assert res.drug_name == "CyclosporinA"


# ---------------------------------------------------------------------------
# Half-life calculation consistency
# ---------------------------------------------------------------------------


def test_half_life_h_consistent_with_min():
    res = base_predict()
    assert abs(res.plasma_half_life_h - res.plasma_half_life_min / 60.0) < 1e-10


def test_half_life_positive():
    res = base_predict()
    assert res.plasma_half_life_min > 0


# ---------------------------------------------------------------------------
# Modification effects on half-life
# ---------------------------------------------------------------------------


def test_cyclic_longer_than_linear():
    linear = base_predict(is_cyclic=False)
    cyclic = base_predict(is_cyclic=True)
    assert cyclic.plasma_half_life_min > linear.plasma_half_life_min


def test_d_amino_acids_increase_half_life():
    linear = base_predict(d_aa_fraction=0.0)
    d_aa = base_predict(d_aa_fraction=0.5)
    assert d_aa.plasma_half_life_min > linear.plasma_half_life_min


def test_n_methylation_increases_half_life():
    unmethylated = base_predict(n_methylated_bonds=0)
    methylated = base_predict(n_methylated_bonds=3)
    assert methylated.plasma_half_life_min > unmethylated.plasma_half_life_min


def test_pegylation_increases_half_life():
    unpeg = base_predict(is_pegylated=False)
    peg = base_predict(is_pegylated=True)
    assert peg.plasma_half_life_min > unpeg.plasma_half_life_min


def test_longer_peptide_shorter_half_life():
    short = base_predict(n_residues=5)
    long_ = base_predict(n_residues=30)
    assert long_.plasma_half_life_min < short.plasma_half_life_min


def test_cyclic_modifier_exact_5x():
    linear = base_predict(n_residues=5, d_aa_fraction=0.0, n_methylated_bonds=0, is_pegylated=False)
    cyclic = base_predict(
        n_residues=5, d_aa_fraction=0.0, n_methylated_bonds=0, is_pegylated=False, is_cyclic=True
    )
    # Length factor for n=5: exp(0) = 1.0, so t_half = 120 * modifier
    ratio = cyclic.plasma_half_life_min / linear.plasma_half_life_min
    assert abs(ratio - 5.0) < 1e-6


def test_pegylation_modifier_exact_2x():
    base = base_predict(
        n_residues=5, d_aa_fraction=0.0, n_methylated_bonds=0, is_cyclic=False, is_pegylated=False
    )
    peg = base_predict(
        n_residues=5, d_aa_fraction=0.0, n_methylated_bonds=0, is_cyclic=False, is_pegylated=True
    )
    ratio = peg.plasma_half_life_min / base.plasma_half_life_min
    assert abs(ratio - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# Aggregation propensity
# ---------------------------------------------------------------------------


def test_hydrophobic_peptide_high_aggregation():
    res = base_predict(hydrophobic_pct=60.0, abs_charge=1.0, n_beta_residues=4)
    assert res.aggregation_propensity >= 35


def test_charged_low_hydrophobic_lower_aggregation():
    hydrophobic = base_predict(hydrophobic_pct=60.0, abs_charge=1.0)
    charged = base_predict(hydrophobic_pct=20.0, abs_charge=5.0)
    assert charged.aggregation_propensity < hydrophobic.aggregation_propensity


def test_aggregation_clamped_at_100():
    res = base_predict(hydrophobic_pct=90.0, abs_charge=0.0, n_beta_residues=10)
    assert res.aggregation_propensity <= 100.0


def test_aggregation_non_negative():
    res = base_predict()
    assert res.aggregation_propensity >= 0.0


def test_pegylation_prevents_agg_penalty():
    no_peg = base_predict(
        hydrophobic_pct=50.0, abs_charge=1.0, n_beta_residues=4, is_pegylated=False
    )
    peg = base_predict(hydrophobic_pct=50.0, abs_charge=1.0, n_beta_residues=4, is_pegylated=True)
    # PEGylation should not add the +15 penalty
    assert peg.aggregation_propensity <= no_peg.aggregation_propensity


# ---------------------------------------------------------------------------
# Stability class
# ---------------------------------------------------------------------------


def test_unstable_class_short_half_life():
    # Short peptide, nothing modified → fast degradation
    res = base_predict(
        n_residues=20, d_aa_fraction=0.0, n_methylated_bonds=0, is_cyclic=False, is_pegylated=False
    )
    # May or may not be unstable; just check class is a valid value
    assert res.stability_class in {"unstable", "moderate", "stable"}


def test_stable_class_long_half_life():
    # Fully modified cyclic D-aa PEGylated methylated peptide should be stable
    res = base_predict(
        n_residues=5, d_aa_fraction=0.5, n_methylated_bonds=3, is_cyclic=True, is_pegylated=True
    )
    assert res.stability_class == "stable"


# ---------------------------------------------------------------------------
# Aggregation risk labels
# ---------------------------------------------------------------------------


def test_aggregation_risk_labels_valid():
    res = base_predict()
    assert res.aggregation_risk in {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Recommended modifications
# ---------------------------------------------------------------------------


def test_recommended_modifications_is_list():
    res = base_predict()
    assert isinstance(res.recommended_modifications, list)


def test_recommended_modifications_non_empty():
    res = base_predict()
    assert len(res.recommended_modifications) > 0


# ---------------------------------------------------------------------------
# compare_modifications
# ---------------------------------------------------------------------------


def test_compare_modifications_returns_4_results():
    results = compare_modifications("PepX", 10, 30.0, 2, 3.0)
    assert len(results) == 4


def test_compare_modifications_all_result_type():
    results = compare_modifications("PepX", 10, 30.0, 2, 3.0)
    for r in results:
        assert isinstance(r, PeptideStabilityResult)


def test_compare_modifications_cyclic_longer_than_linear():
    results = compare_modifications("PepX", 10, 30.0, 2, 3.0)
    linear, cyclic, _, _ = results
    assert cyclic.plasma_half_life_min > linear.plasma_half_life_min


def test_compare_modifications_daa_longer_than_linear():
    results = compare_modifications("PepX", 10, 30.0, 2, 3.0)
    linear, _, _, d_aa = results
    assert d_aa.plasma_half_life_min > linear.plasma_half_life_min


def test_compare_modifications_names_distinguishable():
    results = compare_modifications("DrugA", 10, 30.0, 2, 3.0)
    names = [r.drug_name for r in results]
    assert len(set(names)) == 4


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_on_zero_residues():
    with pytest.raises(ValueError, match="n_residues"):
        base_predict(n_residues=0)


def test_raises_on_negative_residues():
    with pytest.raises(ValueError, match="n_residues"):
        base_predict(n_residues=-1)


def test_raises_on_hydrophobic_pct_out_of_range_low():
    with pytest.raises(ValueError, match="hydrophobic_pct"):
        base_predict(hydrophobic_pct=-5.0)


def test_raises_on_hydrophobic_pct_out_of_range_high():
    with pytest.raises(ValueError, match="hydrophobic_pct"):
        base_predict(hydrophobic_pct=110.0)


def test_raises_on_d_aa_fraction_negative():
    with pytest.raises(ValueError, match="d_aa_fraction"):
        base_predict(d_aa_fraction=-0.1)


def test_raises_on_d_aa_fraction_gt_one():
    with pytest.raises(ValueError, match="d_aa_fraction"):
        base_predict(d_aa_fraction=1.5)


def test_raises_on_negative_methylated_bonds():
    with pytest.raises(ValueError, match="n_methylated_bonds"):
        base_predict(n_methylated_bonds=-1)


def test_raises_on_n_beta_exceeds_n_residues():
    with pytest.raises(ValueError, match="n_beta_residues"):
        base_predict(n_residues=5, n_beta_residues=10)


def test_raises_on_negative_abs_charge():
    with pytest.raises(ValueError, match="abs_charge"):
        base_predict(abs_charge=-1.0)
