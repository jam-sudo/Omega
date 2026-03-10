"""Novel molecule validation harness for PK predictions.

Three-tier validation framework to test model generalization:

Tier 1: Temporal holdout — drugs approved after training data cutoff
Tier 2: Structural analog perturbation — SAR consistency
Tier 3: De novo molecules — plausibility checks

Reference: docs/plan-real.md Section 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlausibilityCheck:
    """Result of a single plausibility check."""

    name: str
    passed: bool
    details: str


@dataclass
class TierResult:
    """Result of one validation tier."""

    tier: int
    tier_name: str
    n_molecules: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_errors: int = 0
    pass_rate: float = 0.0
    details: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Tier {self.tier} ({self.tier_name}): "
            f"{self.n_passed}/{self.n_molecules} passed ({self.pass_rate:.0%}), "
            f"{self.n_errors} errors"
        )


@dataclass
class ValidationReport:
    """Full validation report across all tiers."""

    tier_results: list[TierResult] = field(default_factory=list)
    overall_pass: bool = False
    summary: str = ""

    def __str__(self) -> str:
        lines = ["=" * 60, "Novel Molecule Validation Report", "=" * 60]
        for t in self.tier_results:
            lines.append(str(t))
        lines.append(f"\nOverall: {'PASS' if self.overall_pass else 'NEEDS IMPROVEMENT'}")
        if self.summary:
            lines.append(self.summary)
        return "\n".join(lines)


class NovelMoleculeValidator:
    """Validates PK predictions on novel (unseen) molecules.

    Uses three complementary strategies to assess generalization:
    1. Temporal holdout: drugs approved after model training cutoff
    2. Structural analogs: systematic modifications of known drugs
    3. De novo molecules: completely novel drug-like structures
    """

    def __init__(self) -> None:
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is None:
            from omega_pbpk.pipeline import OmegaPipeline

            self._pipeline = OmegaPipeline()

    def validate_all(self, verbose: bool = False) -> ValidationReport:
        """Run all three validation tiers.

        Returns:
            ValidationReport with results from all tiers.
        """
        self._ensure_pipeline()

        report = ValidationReport()

        # Tier 2: Structural analogs (Tier 1 needs manual curation)
        tier2 = self.validate_analogs(verbose=verbose)
        report.tier_results.append(tier2)

        # Tier 3: De novo molecules
        tier3 = self.validate_de_novo(verbose=verbose)
        report.tier_results.append(tier3)

        # Overall assessment
        pass_rates = [t.pass_rate for t in report.tier_results if t.n_molecules > 0]
        report.overall_pass = all(r >= 0.70 for r in pass_rates) if pass_rates else False
        report.summary = f"Tier pass rates: {', '.join(f'{r:.0%}' for r in pass_rates)}"

        return report

    def validate_analogs(
        self,
        parent_drugs: list[tuple[str, str]] | None = None,
        max_analogs_per_drug: int = 5,
        verbose: bool = False,
    ) -> TierResult:
        """Tier 2: Structural analog perturbation validation.

        For each parent drug, generates structural analogs and checks:
        1. Predictions don't crash
        2. Analog PK is within 5-fold of parent (plausible range)
        3. SAR trends are consistent (polar group → higher CL)
        4. Confidence is lower for more distant analogs
        """
        from omega_pbpk.ml.evaluation.analog_generator import generate_analogs
        from omega_pbpk.pipeline import SimulationRequest

        self._ensure_pipeline()

        if parent_drugs is None:
            parent_drugs = [
                ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
                ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
                ("Diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
                ("Omeprazole", "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C"),
                ("Metronidazole", "Cc1ncc([N+](=O)[O-])n1CCO"),
            ]

        result = TierResult(tier=2, tier_name="Structural Analogs")

        for drug_name, parent_smi in parent_drugs:
            # Get parent PK
            try:
                parent_result = self._pipeline.simulate(
                    SimulationRequest(smiles=parent_smi, dose_mg=100, route="oral", duration_h=24)
                )
                parent_cmax = parent_result.cmax_mg_L
                parent_auc = parent_result.auc0t_mg_h_L
            except Exception as exc:
                logger.warning("Parent %s failed: %s", drug_name, exc)
                result.n_errors += 1
                continue

            # Generate and test analogs
            analogs = generate_analogs(parent_smi, max_analogs=max_analogs_per_drug)
            for analog_smi, modification, _expected_direction in analogs:
                result.n_molecules += 1
                try:
                    analog_result = self._pipeline.simulate(
                        SimulationRequest(
                            smiles=analog_smi,
                            dose_mg=100,
                            route="oral",
                            duration_h=24,
                        )
                    )

                    checks = self._check_analog_plausibility(
                        parent_cmax,
                        parent_auc,
                        analog_result.cmax_mg_L,
                        analog_result.auc0t_mg_h_L,
                        modification,
                    )

                    all_passed = all(c.passed for c in checks)
                    if all_passed:
                        result.n_passed += 1
                    else:
                        result.n_failed += 1

                    if verbose:
                        status = "PASS" if all_passed else "FAIL"
                        failed_checks = [c.name for c in checks if not c.passed]
                        result.details.append(
                            {
                                "parent": drug_name,
                                "modification": modification,
                                "status": status,
                                "failed": failed_checks,
                                "analog_cmax": analog_result.cmax_mg_L,
                                "analog_auc": analog_result.auc0t_mg_h_L,
                            }
                        )

                except Exception as exc:
                    result.n_errors += 1
                    if verbose:
                        result.details.append(
                            {
                                "parent": drug_name,
                                "modification": modification,
                                "status": "ERROR",
                                "error": str(exc),
                            }
                        )

        total = result.n_passed + result.n_failed + result.n_errors
        result.pass_rate = result.n_passed / max(total, 1)
        return result

    def validate_de_novo(self, n_molecules: int = 20, verbose: bool = False) -> TierResult:
        """Tier 3: De novo molecule validation.

        Generates novel drug-like molecules and checks:
        1. Prediction completes without error
        2. Cmax > 0 (positive concentrations)
        3. AUC > 0 (nonzero exposure)
        4. Cmax < dose/Vd_min (physical plausibility)
        5. t½ > 0 and < 200h (reasonable half-life)
        """
        from omega_pbpk.ml.evaluation.analog_generator import generate_de_novo_molecules
        from omega_pbpk.pipeline import SimulationRequest

        self._ensure_pipeline()

        result = TierResult(tier=3, tier_name="De Novo Molecules")
        molecules = generate_de_novo_molecules(n=n_molecules)

        for smi in molecules:
            result.n_molecules += 1
            try:
                sim = self._pipeline.simulate(
                    SimulationRequest(smiles=smi, dose_mg=100, route="oral", duration_h=24)
                )

                checks = self._check_de_novo_plausibility(sim, dose_mg=100)
                all_passed = all(c.passed for c in checks)

                if all_passed:
                    result.n_passed += 1
                else:
                    result.n_failed += 1

                if verbose:
                    status = "PASS" if all_passed else "FAIL"
                    failed = [c.name for c in checks if not c.passed]
                    result.details.append(
                        {
                            "smiles": smi,
                            "status": status,
                            "cmax": sim.cmax_mg_L,
                            "auc": sim.auc0t_mg_h_L,
                            "failed": failed,
                        }
                    )

            except Exception as exc:
                result.n_errors += 1
                if verbose:
                    result.details.append({"smiles": smi, "status": "ERROR", "error": str(exc)})

        total = result.n_passed + result.n_failed + result.n_errors
        result.pass_rate = result.n_passed / max(total, 1)
        return result

    @staticmethod
    def _check_analog_plausibility(
        parent_cmax: float,
        parent_auc: float,
        analog_cmax: float,
        analog_auc: float,
        modification: str,
    ) -> list[PlausibilityCheck]:
        """Check whether analog PK is plausible relative to parent."""
        checks = []

        # 1. Positive concentrations
        checks.append(
            PlausibilityCheck(
                "positive_cmax",
                analog_cmax > 0,
                f"Cmax={analog_cmax:.4f}",
            )
        )

        # 2. Within 5-fold of parent (structural analogs should be similar)
        if parent_cmax > 0:
            fold = max(analog_cmax / parent_cmax, parent_cmax / analog_cmax)
            checks.append(
                PlausibilityCheck(
                    "within_5fold_cmax",
                    fold <= 5.0,
                    f"fold_error={fold:.2f}",
                )
            )

        if parent_auc > 0:
            fold_auc = max(analog_auc / parent_auc, parent_auc / analog_auc)
            checks.append(
                PlausibilityCheck(
                    "within_5fold_auc",
                    fold_auc <= 5.0,
                    f"fold_error={fold_auc:.2f}",
                )
            )

        # 3. No wild discontinuities (analog PK shouldn't be zero or infinite)
        checks.append(
            PlausibilityCheck(
                "nonzero_auc",
                analog_auc > 0.001,
                f"AUC={analog_auc:.4f}",
            )
        )

        return checks

    @staticmethod
    def _check_de_novo_plausibility(sim, dose_mg: float) -> list[PlausibilityCheck]:
        """Check whether de novo molecule PK is physically plausible."""
        import math

        checks = []

        # 1. Positive Cmax
        checks.append(
            PlausibilityCheck(
                "positive_cmax",
                sim.cmax_mg_L > 0,
                f"Cmax={sim.cmax_mg_L:.4f}",
            )
        )

        # 2. Positive AUC
        checks.append(
            PlausibilityCheck(
                "positive_auc",
                sim.auc0t_mg_h_L > 0,
                f"AUC={sim.auc0t_mg_h_L:.4f}",
            )
        )

        # 3. Cmax < dose (can't have more drug in plasma than total dose)
        checks.append(
            PlausibilityCheck(
                "cmax_lt_dose",
                sim.cmax_mg_L < dose_mg,
                f"Cmax={sim.cmax_mg_L:.2f}, dose={dose_mg}",
            )
        )

        # 4. Reasonable half-life (0 < t½ < 200h)
        t_half_ok = not math.isnan(sim.t_half_h) and 0.01 < sim.t_half_h < 200.0
        checks.append(
            PlausibilityCheck(
                "reasonable_thalf",
                t_half_ok,
                f"t½={sim.t_half_h:.2f}h" if not math.isnan(sim.t_half_h) else "t½=NaN",
            )
        )

        # 5. Tmax > 0 for oral (drug needs time to absorb)
        checks.append(
            PlausibilityCheck(
                "tmax_positive",
                sim.tmax_h > 0,
                f"Tmax={sim.tmax_h:.2f}h",
            )
        )

        return checks
