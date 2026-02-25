"""CYP polymorphism database and genotype→phenotype→CLint scaling.

Implements CPIC (Clinical Pharmacogenetics Implementation Consortium)
standard allele definitions for 5 major CYP enzymes:
  CYP2D6, CYP2C19, CYP2C9, CYP3A5, CYP1A2

Population allele frequencies from PharmGKB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CYPAllele:
    """CYP enzyme allele definition."""

    gene: str
    allele: str
    function: str  # "normal", "decreased", "no_function", "increased"
    activity_score: float  # CPIC activity score


@dataclass(frozen=True)
class PGxResult:
    """Pharmacogenomics analysis result for one enzyme."""

    gene: str
    diplotype: str
    phenotype: str  # "UM", "NM", "IM", "PM"
    activity_score: float
    clint_scaling_factor: float
    population_frequency: float


# CPIC allele database (simplified representative set)
ALLELE_DB: dict[str, list[CYPAllele]] = {
    "CYP2D6": [
        CYPAllele("CYP2D6", "*1", "normal", 1.0),
        CYPAllele("CYP2D6", "*2", "normal", 1.0),
        CYPAllele("CYP2D6", "*4", "no_function", 0.0),
        CYPAllele("CYP2D6", "*5", "no_function", 0.0),
        CYPAllele("CYP2D6", "*10", "decreased", 0.25),
        CYPAllele("CYP2D6", "*17", "decreased", 0.5),
        CYPAllele("CYP2D6", "*41", "decreased", 0.5),
        CYPAllele("CYP2D6", "*1x2", "increased", 2.0),
    ],
    "CYP2C19": [
        CYPAllele("CYP2C19", "*1", "normal", 1.0),
        CYPAllele("CYP2C19", "*2", "no_function", 0.0),
        CYPAllele("CYP2C19", "*3", "no_function", 0.0),
        CYPAllele("CYP2C19", "*17", "increased", 1.5),
    ],
    "CYP2C9": [
        CYPAllele("CYP2C9", "*1", "normal", 1.0),
        CYPAllele("CYP2C9", "*2", "decreased", 0.5),
        CYPAllele("CYP2C9", "*3", "decreased", 0.25),
    ],
    "CYP3A5": [
        CYPAllele("CYP3A5", "*1", "normal", 1.0),
        CYPAllele("CYP3A5", "*3", "no_function", 0.0),
        CYPAllele("CYP3A5", "*6", "no_function", 0.0),
    ],
    "CYP1A2": [
        CYPAllele("CYP1A2", "*1A", "normal", 1.0),
        CYPAllele("CYP1A2", "*1C", "decreased", 0.5),
        CYPAllele("CYP1A2", "*1F", "increased", 1.5),
    ],
}

# Population allele frequencies: gene → allele → population → frequency
ALLELE_FREQUENCIES: dict[str, dict[str, dict[str, float]]] = {
    "CYP2D6": {
        "*1": {"Caucasian": 0.36, "East_Asian": 0.25, "African": 0.35, "Global": 0.33},
        "*2": {"Caucasian": 0.27, "East_Asian": 0.15, "African": 0.20, "Global": 0.22},
        "*4": {"Caucasian": 0.21, "East_Asian": 0.01, "African": 0.07, "Global": 0.12},
        "*5": {"Caucasian": 0.03, "East_Asian": 0.06, "African": 0.06, "Global": 0.05},
        "*10": {"Caucasian": 0.02, "East_Asian": 0.43, "African": 0.06, "Global": 0.15},
        "*17": {"Caucasian": 0.01, "East_Asian": 0.01, "African": 0.20, "Global": 0.06},
        "*41": {"Caucasian": 0.09, "East_Asian": 0.04, "African": 0.04, "Global": 0.06},
        "*1x2": {"Caucasian": 0.01, "East_Asian": 0.005, "African": 0.02, "Global": 0.01},
    },
    "CYP2C19": {
        "*1": {"Caucasian": 0.63, "East_Asian": 0.40, "African": 0.65, "Global": 0.55},
        "*2": {"Caucasian": 0.15, "East_Asian": 0.30, "African": 0.17, "Global": 0.20},
        "*3": {"Caucasian": 0.004, "East_Asian": 0.08, "African": 0.01, "Global": 0.03},
        "*17": {"Caucasian": 0.21, "East_Asian": 0.02, "African": 0.17, "Global": 0.15},
    },
    "CYP2C9": {
        "*1": {"Caucasian": 0.79, "East_Asian": 0.95, "African": 0.88, "Global": 0.86},
        "*2": {"Caucasian": 0.13, "East_Asian": 0.01, "African": 0.04, "Global": 0.07},
        "*3": {"Caucasian": 0.08, "East_Asian": 0.04, "African": 0.02, "Global": 0.05},
    },
    "CYP3A5": {
        "*1": {"Caucasian": 0.07, "East_Asian": 0.30, "African": 0.70, "Global": 0.30},
        "*3": {"Caucasian": 0.90, "East_Asian": 0.65, "African": 0.25, "Global": 0.65},
        "*6": {"Caucasian": 0.03, "East_Asian": 0.05, "African": 0.05, "Global": 0.05},
    },
    "CYP1A2": {
        "*1A": {"Caucasian": 0.55, "East_Asian": 0.60, "African": 0.50, "Global": 0.55},
        "*1C": {"Caucasian": 0.10, "East_Asian": 0.25, "African": 0.15, "Global": 0.15},
        "*1F": {"Caucasian": 0.35, "East_Asian": 0.15, "African": 0.35, "Global": 0.30},
    },
}

# Phenotype definitions based on activity score
PHENOTYPE_THRESHOLDS = {
    "UM": 2.0,  # Ultra-rapid metabolizer
    "NM": 1.0,  # Normal metabolizer
    "IM": 0.25,  # Intermediate metabolizer
    "PM": 0.0,  # Poor metabolizer
}


class PGxAnalyzer:
    """Pharmacogenomics analyzer for CYP polymorphism impact."""

    def analyze_gene(
        self,
        gene: str,
        population: str = "Global",
    ) -> list[PGxResult]:
        """Analyze all diplotype combinations for a gene in a population.

        Returns list of PGxResult sorted by population frequency.
        """
        if gene not in ALLELE_DB:
            raise ValueError(f"Unknown gene: {gene}. Available: {list(ALLELE_DB.keys())}")

        alleles = ALLELE_DB[gene]
        freqs = ALLELE_FREQUENCIES.get(gene, {})
        results: list[PGxResult] = []

        # Generate diplotype combinations
        for i, a1 in enumerate(alleles):
            for a2 in alleles[i:]:
                score = a1.activity_score + a2.activity_score
                phenotype = self._score_to_phenotype(score)
                scaling = self._score_to_scaling(score)

                f1 = freqs.get(a1.allele, {}).get(population, 0.01)
                f2 = freqs.get(a2.allele, {}).get(population, 0.01)
                freq = f1 * f2 * (2 if a1.allele != a2.allele else 1)

                diplotype = f"{a1.allele}/{a2.allele}"
                results.append(
                    PGxResult(
                        gene=gene,
                        diplotype=diplotype,
                        phenotype=phenotype,
                        activity_score=score,
                        clint_scaling_factor=round(scaling, 3),
                        population_frequency=round(freq, 6),
                    )
                )

        results.sort(key=lambda r: r.population_frequency, reverse=True)
        return results

    def clint_scaling(self, gene: str, diplotype: str) -> float:
        """Get CLint scaling factor for a specific diplotype.

        Returns:
            Scaling factor (0.0 for PM, 1.0 for NM, >1.0 for UM).
        """
        alleles = diplotype.split("/")
        if len(alleles) != 2:
            raise ValueError(f"Invalid diplotype format: {diplotype}")

        allele_db = {a.allele: a for a in ALLELE_DB.get(gene, [])}
        scores = []
        for a_name in alleles:
            if a_name not in allele_db:
                raise ValueError(f"Unknown allele {a_name} for {gene}")
            scores.append(allele_db[a_name].activity_score)

        return self._score_to_scaling(sum(scores))

    def population_summary(self, gene: str, population: str = "Global") -> dict[str, Any]:
        """Summarize phenotype distribution in a population."""
        results = self.analyze_gene(gene, population)
        phenotype_freq: dict[str, float] = {"UM": 0, "NM": 0, "IM": 0, "PM": 0}
        for r in results:
            phenotype_freq[r.phenotype] += r.population_frequency

        total = sum(phenotype_freq.values())
        if total > 0:
            phenotype_freq = {k: round(v / total, 4) for k, v in phenotype_freq.items()}

        return {
            "gene": gene,
            "population": population,
            "phenotype_distribution": phenotype_freq,
            "n_diplotypes": len(results),
        }

    @staticmethod
    def _score_to_phenotype(score: float) -> str:
        """Convert activity score to CPIC phenotype."""
        if score >= 2.0:
            return "UM"
        if score >= 1.0:
            return "NM"
        if score >= 0.25:
            return "IM"
        return "PM"

    @staticmethod
    def _score_to_scaling(score: float) -> float:
        """Convert activity score to CLint scaling factor.

        NM (score=2.0) → 1.0; PM (score=0) → 0.0; UM (score>2.0) → proportional.
        """
        return score / 2.0  # Normalized so NM (*1/*1, score=2.0) = 1.0
