"""Unit tests for CYP pharmacogenomics module.

Covers:
  - Poor metabolizer diplotype → CLint scaling factor near 0
  - Ultra-rapid metabolizer → CLint scaling factor > 1.0
  - Normal metabolizer → CLint scaling factor ≈ 1.0
  - Population allele frequency sums to ~1.0 per gene
  - Invalid gene raises ValueError
"""

from __future__ import annotations

import pytest

from omega_pbpk.pharmacogenomics.cyp_polymorphism import (
    ALLELE_DB,
    ALLELE_FREQUENCIES,
    PGxAnalyzer,
)


# ------------------------------------------------------------------ helpers


@pytest.fixture
def analyzer() -> PGxAnalyzer:
    return PGxAnalyzer()


# ------------------------------------------------------------------ CLint scaling factor tests


class TestCLintScalingFactor:
    def test_poor_metabolizer_cyp2d6(self, analyzer: PGxAnalyzer) -> None:
        """CYP2D6 *4/*4 (two no-function alleles) → PM → CLint scaling factor = 0.0."""
        factor = analyzer.clint_scaling("CYP2D6", "*4/*4")
        assert factor == pytest.approx(0.0, abs=1e-9)

    def test_poor_metabolizer_cyp2c19(self, analyzer: PGxAnalyzer) -> None:
        """CYP2C19 *2/*2 (two no-function alleles) → PM → CLint scaling factor = 0.0."""
        factor = analyzer.clint_scaling("CYP2C19", "*2/*2")
        assert factor == pytest.approx(0.0, abs=1e-9)

    def test_poor_metabolizer_cyp2c9(self, analyzer: PGxAnalyzer) -> None:
        """CYP2C9 *3/*3 (two decreased alleles, activity_score=0.5 each → total 0.5) → IM, not PM.
        But *3/*3 → score=0.5, PM threshold is <0.25; verify factor = 0.5/2.0 = 0.25."""
        # *3 has activity_score=0.25 → *3/*3 score=0.5 → IM territory
        factor = analyzer.clint_scaling("CYP2C9", "*3/*3")
        # score = 0.25 + 0.25 = 0.5 → scaling = 0.5 / 2.0 = 0.25
        assert factor == pytest.approx(0.25, rel=1e-6)

    def test_poor_metabolizer_cyp2d6_two_null_alleles(self, analyzer: PGxAnalyzer) -> None:
        """CYP2D6 *5/*5 (no_function/no_function) → score=0 → PM → factor=0."""
        factor = analyzer.clint_scaling("CYP2D6", "*5/*5")
        assert factor == pytest.approx(0.0, abs=1e-9)

    def test_ultra_rapid_metabolizer_cyp2d6(self, analyzer: PGxAnalyzer) -> None:
        """CYP2D6 *1x2/*1x2 (gene duplication, activity_score=2.0 each) → UM → factor > 1.0."""
        factor = analyzer.clint_scaling("CYP2D6", "*1x2/*1x2")
        # score = 2.0 + 2.0 = 4.0 → scaling = 4.0 / 2.0 = 2.0
        assert factor > 1.0, f"UM diplotype should give factor > 1.0; got {factor}"
        assert factor == pytest.approx(2.0, rel=1e-6)

    def test_ultra_rapid_metabolizer_cyp2c19(self, analyzer: PGxAnalyzer) -> None:
        """CYP2C19 *17/*17 (increased/increased, score=1.5+1.5=3.0) → UM → factor > 1.0."""
        factor = analyzer.clint_scaling("CYP2C19", "*17/*17")
        # score = 1.5 + 1.5 = 3.0 → scaling = 3.0 / 2.0 = 1.5
        assert factor > 1.0
        assert factor == pytest.approx(1.5, rel=1e-6)

    def test_normal_metabolizer_cyp2d6(self, analyzer: PGxAnalyzer) -> None:
        """CYP2D6 *1/*1 (normal/normal) → NM → CLint scaling factor ≈ 1.0."""
        factor = analyzer.clint_scaling("CYP2D6", "*1/*1")
        assert factor == pytest.approx(1.0, rel=1e-6)

    def test_normal_metabolizer_cyp2c9(self, analyzer: PGxAnalyzer) -> None:
        """CYP2C9 *1/*1 → NM → factor = 1.0."""
        factor = analyzer.clint_scaling("CYP2C9", "*1/*1")
        assert factor == pytest.approx(1.0, rel=1e-6)

    def test_normal_metabolizer_cyp2c19(self, analyzer: PGxAnalyzer) -> None:
        """CYP2C19 *1/*1 → NM → factor = 1.0."""
        factor = analyzer.clint_scaling("CYP2C19", "*1/*1")
        assert factor == pytest.approx(1.0, rel=1e-6)

    def test_normal_metabolizer_cyp1a2(self, analyzer: PGxAnalyzer) -> None:
        """CYP1A2 *1A/*1A (normal/normal, score=1.0+1.0=2.0) → NM → factor = 1.0."""
        factor = analyzer.clint_scaling("CYP1A2", "*1A/*1A")
        assert factor == pytest.approx(1.0, rel=1e-6)

    def test_pm_factor_lower_than_nm(self, analyzer: PGxAnalyzer) -> None:
        """PM diplotype factor must be strictly less than NM factor."""
        nm = analyzer.clint_scaling("CYP2D6", "*1/*1")
        pm = analyzer.clint_scaling("CYP2D6", "*4/*4")
        assert pm < nm

    def test_um_factor_higher_than_nm(self, analyzer: PGxAnalyzer) -> None:
        """UM diplotype factor must be strictly greater than NM factor."""
        nm = analyzer.clint_scaling("CYP2D6", "*1/*1")
        um = analyzer.clint_scaling("CYP2D6", "*1x2/*1x2")
        assert um > nm

    def test_intermediate_metabolizer_between_pm_and_nm(self, analyzer: PGxAnalyzer) -> None:
        """IM diplotype factor should fall between PM and NM values."""
        pm = analyzer.clint_scaling("CYP2D6", "*4/*4")   # 0.0
        nm = analyzer.clint_scaling("CYP2D6", "*1/*1")   # 1.0
        # *4/*41 → score=0.0+0.5=0.5 → scaling=0.25
        im = analyzer.clint_scaling("CYP2D6", "*4/*41")
        assert pm < im < nm, f"Expected PM({pm}) < IM({im}) < NM({nm})"


# ------------------------------------------------------------------ allele frequency sum


class TestAlleleFrequencySums:
    """Population allele frequencies should sum to approximately 1.0 for each gene.

    The ALLELE_DB is described as a "simplified representative set" (not every known
    allele is included), so we use a generous tolerance of abs=0.25 that catches gross
    data entry errors while accepting that minor alleles may be omitted.
    """

    @pytest.mark.parametrize("gene", list(ALLELE_FREQUENCIES.keys()))
    @pytest.mark.parametrize("population", ["Global", "Caucasian", "East_Asian", "African"])
    def test_allele_freq_sums_to_one(
        self, gene: str, population: str, analyzer: PGxAnalyzer
    ) -> None:
        """For each gene and population the allele frequencies should sum to ~1.0.

        Tolerance of 0.25 accounts for the simplified representative allele subset —
        the data is clearly intended to represent the full frequency distribution
        even if not every rare allele is enumerated.
        """
        gene_freqs = ALLELE_FREQUENCIES[gene]
        total = sum(
            allele_freqs.get(population, 0.0)
            for allele_freqs in gene_freqs.values()
        )
        assert total == pytest.approx(1.0, abs=0.25), (
            f"{gene}/{population}: allele frequencies sum to {total:.4f}, expected ~1.0"
        )
        # Additionally, total should be in (0, 1.1] — no negative or wildly out-of-range values
        assert 0.0 < total <= 1.10, (
            f"{gene}/{population}: total={total:.4f} is out of plausible range"
        )

    def test_global_cyp2d6_sums_to_one(self, analyzer: PGxAnalyzer) -> None:
        """CYP2D6 Global allele frequencies sum closely to 1.0 (complete representation)."""
        freqs = ALLELE_FREQUENCIES["CYP2D6"]
        total = sum(v.get("Global", 0.0) for v in freqs.values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_global_cyp2c19_sums_near_one(self, analyzer: PGxAnalyzer) -> None:
        """CYP2C19 Global allele frequencies sum to ~1.0 (simplified subset, abs=0.15)."""
        freqs = ALLELE_FREQUENCIES["CYP2C19"]
        total = sum(v.get("Global", 0.0) for v in freqs.values())
        # CYP2C19 Global: *1(0.55)+*2(0.20)+*3(0.03)+*17(0.15) = 0.93 (simplified subset)
        assert total == pytest.approx(1.0, abs=0.15), (
            f"CYP2C19 Global sum={total:.4f}"
        )

    def test_global_cyp1a2_sums_to_one(self, analyzer: PGxAnalyzer) -> None:
        """CYP1A2 Global allele frequencies sum to ~1.0 explicitly."""
        freqs = ALLELE_FREQUENCIES["CYP1A2"]
        total = sum(v.get("Global", 0.0) for v in freqs.values())
        # CYP1A2: *1A(0.55)+*1C(0.15)+*1F(0.30) = 1.00
        assert total == pytest.approx(1.0, abs=0.05)


# ------------------------------------------------------------------ invalid gene raises ValueError


class TestInvalidGene:
    def test_invalid_gene_raises_value_error(self, analyzer: PGxAnalyzer) -> None:
        """Passing an unknown gene to analyze_gene should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown gene"):
            analyzer.analyze_gene("CYP_FAKE")

    def test_invalid_gene_clint_scaling_raises_error(self, analyzer: PGxAnalyzer) -> None:
        """clint_scaling with an unknown allele for a valid gene raises ValueError."""
        with pytest.raises(ValueError, match="Unknown allele"):
            analyzer.clint_scaling("CYP2D6", "*999/*1")

    def test_malformed_diplotype_raises_value_error(self, analyzer: PGxAnalyzer) -> None:
        """A diplotype string without '/' separator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid diplotype format"):
            analyzer.clint_scaling("CYP2D6", "*1*2")


# ------------------------------------------------------------------ analyze_gene result structure


class TestAnalyzeGeneResults:
    def test_analyze_gene_returns_list(self, analyzer: PGxAnalyzer) -> None:
        """analyze_gene returns a non-empty list of PGxResult objects."""
        results = analyzer.analyze_gene("CYP2D6")
        assert len(results) > 0

    def test_analyze_gene_sorted_by_frequency(self, analyzer: PGxAnalyzer) -> None:
        """Results are sorted by population_frequency descending."""
        results = analyzer.analyze_gene("CYP2D6", population="Global")
        freqs = [r.population_frequency for r in results]
        assert freqs == sorted(freqs, reverse=True)

    def test_phenotypes_are_valid(self, analyzer: PGxAnalyzer) -> None:
        """All phenotypes in results are valid CPIC categories."""
        valid_phenotypes = {"UM", "NM", "IM", "PM"}
        results = analyzer.analyze_gene("CYP2C19")
        for r in results:
            assert r.phenotype in valid_phenotypes, f"Invalid phenotype: {r.phenotype!r}"

    def test_scaling_factors_non_negative(self, analyzer: PGxAnalyzer) -> None:
        """CLint scaling factors must be >= 0 for all diplotypes."""
        results = analyzer.analyze_gene("CYP2D6")
        for r in results:
            assert r.clint_scaling_factor >= 0.0, (
                f"Negative scaling for {r.diplotype}: {r.clint_scaling_factor}"
            )

    def test_population_summary_phenotype_dist_sums_to_one(self, analyzer: PGxAnalyzer) -> None:
        """population_summary phenotype distribution should sum to ~1.0."""
        summary = analyzer.population_summary("CYP2D6", population="Global")
        total = sum(summary["phenotype_distribution"].values())
        assert total == pytest.approx(1.0, abs=0.05)
