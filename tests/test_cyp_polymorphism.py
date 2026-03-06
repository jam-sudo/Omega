"""Tests for CYP polymorphism database and PGx analyzer."""

import pytest

from omega_pbpk.pharmacogenomics.cyp_polymorphism import (
    ALLELE_DB,
    ALLELE_FREQUENCIES,
    PGxAnalyzer,
)

# ── ALLELE_DB structure ──────────────────────────────────────────────


def test_allele_db_has_five_genes():
    assert len(ALLELE_DB) == 5
    expected = {"CYP2D6", "CYP2C19", "CYP2C9", "CYP3A5", "CYP1A2"}
    assert set(ALLELE_DB.keys()) == expected


@pytest.mark.parametrize("gene", ALLELE_DB.keys())
def test_each_gene_has_at_least_three_alleles(gene):
    assert len(ALLELE_DB[gene]) >= 3


def test_cyp2d6_star1_activity_score():
    alleles = {a.allele: a for a in ALLELE_DB["CYP2D6"]}
    assert alleles["*1"].activity_score == 1.0


def test_cyp2d6_star4_no_function():
    alleles = {a.allele: a for a in ALLELE_DB["CYP2D6"]}
    assert alleles["*4"].function == "no_function"
    assert alleles["*4"].activity_score == 0.0


def test_cyp2c19_star17_increased():
    alleles = {a.allele: a for a in ALLELE_DB["CYP2C19"]}
    assert alleles["*17"].function == "increased"
    assert alleles["*17"].activity_score > 1.0


# ── ALLELE_FREQUENCIES structure ─────────────────────────────────────


def test_allele_frequencies_has_correct_structure():
    assert set(ALLELE_FREQUENCIES.keys()) == set(ALLELE_DB.keys())
    for _gene, allele_freqs in ALLELE_FREQUENCIES.items():
        for _allele_name, pop_dict in allele_freqs.items():
            assert isinstance(pop_dict, dict)
            for freq in pop_dict.values():
                assert 0.0 <= freq <= 1.0


# ── PGxAnalyzer phenotype from diplotype ─────────────────────────────

analyzer = PGxAnalyzer()


def test_cyp2d6_star4_star4_is_pm():
    scaling = analyzer.clint_scaling("CYP2D6", "*4/*4")
    # *4+*4 = 0.0 → PM, scaling = 0.0/2.0 = 0.0
    assert scaling == 0.0


def test_cyp2d6_star1_star1_is_nm():
    scaling = analyzer.clint_scaling("CYP2D6", "*1/*1")
    # *1+*1 = 2.0 → NM, scaling = 2.0/2.0 = 1.0
    assert scaling == 1.0


def test_cyp2d6_star1x2_star1_is_um():
    # *1x2 (2.0) + *1 (1.0) = 3.0 → UM
    scaling = analyzer.clint_scaling("CYP2D6", "*1x2/*1")
    assert scaling > 1.0  # UM has increased clearance


def test_pm_clint_scaling_below_one():
    # *4/*4 → PM → scaling 0.0
    scaling = analyzer.clint_scaling("CYP2D6", "*4/*4")
    assert scaling < 1.0


def test_um_clint_scaling_above_one():
    # *1x2/*1 → UM → scaling 1.5
    scaling = analyzer.clint_scaling("CYP2D6", "*1x2/*1")
    assert scaling > 1.0


# ── PGxResult dataclass fields ───────────────────────────────────────


def test_pgx_result_fields():
    results = analyzer.analyze_gene("CYP2D6")
    r = results[0]
    assert hasattr(r, "gene")
    assert hasattr(r, "diplotype")
    assert hasattr(r, "phenotype")
    assert hasattr(r, "activity_score")
    assert hasattr(r, "clint_scaling_factor")
    assert hasattr(r, "population_frequency")


# ── Specific diplotype phenotypes ────────────────────────────────────


def test_cyp2c9_star3_star3_is_im():
    # *3 (0.25) + *3 (0.25) = 0.5 → IM (>= 0.25)
    scaling = analyzer.clint_scaling("CYP2C9", "*3/*3")
    assert scaling == pytest.approx(0.25)  # 0.5 / 2.0


def test_cyp3a5_star3_star3_is_pm():
    # *3 (0.0) + *3 (0.0) = 0.0 → PM
    scaling = analyzer.clint_scaling("CYP3A5", "*3/*3")
    assert scaling == 0.0


def test_cyp1a2_star1f_star1f_is_um():
    # *1F (1.5) + *1F (1.5) = 3.0 → UM
    scaling = analyzer.clint_scaling("CYP1A2", "*1F/*1F")
    assert scaling > 1.0


# ── Error handling ───────────────────────────────────────────────────


def test_invalid_gene_raises_error():
    with pytest.raises(ValueError, match="Unknown gene"):
        analyzer.analyze_gene("CYP_FAKE")


def test_unknown_allele_raises_error():
    with pytest.raises(ValueError, match="Unknown allele"):
        analyzer.clint_scaling("CYP2D6", "*99/*1")


# ── Activity score arithmetic ────────────────────────────────────────


def test_activity_score_sum_star4_star4_is_zero():
    alleles = {a.allele: a for a in ALLELE_DB["CYP2D6"]}
    assert alleles["*4"].activity_score + alleles["*4"].activity_score == 0.0


def test_activity_score_cyp2c19_star1_star17():
    alleles = {a.allele: a for a in ALLELE_DB["CYP2C19"]}
    total = alleles["*1"].activity_score + alleles["*17"].activity_score
    assert total == 2.5
    # 2.5 >= 2.0 → UM
    assert PGxAnalyzer._score_to_phenotype(total) == "UM"


# ── Population frequency field ───────────────────────────────────────


def test_population_frequency_in_range():
    results = analyzer.analyze_gene("CYP2D6", "Global")
    for r in results:
        assert 0.0 <= r.population_frequency <= 1.0
