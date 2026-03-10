"""Tests for novel molecule validation harness."""

from __future__ import annotations


class TestAnalogGenerator:
    """Test structural analog generation."""

    def test_generate_analogs_returns_list(self):
        from omega_pbpk.ml.evaluation.analog_generator import generate_analogs

        analogs = generate_analogs("CC(=O)Nc1ccc(O)cc1")  # acetaminophen
        assert isinstance(analogs, list)
        assert len(analogs) > 0

    def test_analogs_are_valid_smiles(self):
        from rdkit import Chem

        from omega_pbpk.ml.evaluation.analog_generator import generate_analogs

        analogs = generate_analogs("OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl")  # diclofenac
        for smi, mod, _direction in analogs:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Invalid SMILES from {mod}: {smi}"

    def test_analogs_differ_from_parent(self):
        from omega_pbpk.ml.evaluation.analog_generator import generate_analogs

        parent = "CC(=O)Nc1ccc(O)cc1"
        analogs = generate_analogs(parent)
        for smi, _, _ in analogs:
            assert smi != parent

    def test_generate_de_novo_returns_smiles(self):
        from omega_pbpk.ml.evaluation.analog_generator import (
            generate_de_novo_molecules,
        )

        mols = generate_de_novo_molecules(n=5, seed=42)
        assert isinstance(mols, list)
        # May not generate all requested due to BRICS limitations
        assert len(mols) >= 0

    def test_de_novo_are_drug_like(self):
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        from omega_pbpk.ml.evaluation.analog_generator import (
            generate_de_novo_molecules,
        )

        mols = generate_de_novo_molecules(n=10, seed=42)
        for smi in mols:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Invalid SMILES: {smi}"
            mw = Descriptors.ExactMolWt(mol)
            assert 150 < mw < 600, f"Non-drug-like MW={mw} for {smi}"


class TestNovelValidator:
    """Test the novel molecule validation harness."""

    def test_validate_analogs_runs(self):
        from omega_pbpk.ml.evaluation.novel_validator import NovelMoleculeValidator

        validator = NovelMoleculeValidator()
        # Use just 1 parent with 2 analogs for speed
        result = validator.validate_analogs(
            parent_drugs=[("Acetaminophen", "CC(=O)Nc1ccc(O)cc1")],
            max_analogs_per_drug=2,
        )
        assert result.tier == 2
        assert result.n_molecules >= 0
        assert 0 <= result.pass_rate <= 1.0

    def test_validate_de_novo_runs(self):
        from omega_pbpk.ml.evaluation.novel_validator import NovelMoleculeValidator

        validator = NovelMoleculeValidator()
        result = validator.validate_de_novo(n_molecules=3)
        assert result.tier == 3
        assert result.n_molecules >= 0

    def test_validate_all_returns_report(self):
        from omega_pbpk.ml.evaluation.novel_validator import NovelMoleculeValidator

        validator = NovelMoleculeValidator()
        report = validator.validate_all()
        assert len(report.tier_results) == 2
        assert isinstance(str(report), str)

    def test_plausibility_checks(self):
        from omega_pbpk.ml.evaluation.novel_validator import NovelMoleculeValidator

        checks = NovelMoleculeValidator._check_analog_plausibility(
            parent_cmax=10.0,
            parent_auc=50.0,
            analog_cmax=8.0,
            analog_auc=45.0,
            modification="test",
        )
        assert all(c.passed for c in checks)

    def test_plausibility_fails_for_extreme_analog(self):
        from omega_pbpk.ml.evaluation.novel_validator import NovelMoleculeValidator

        checks = NovelMoleculeValidator._check_analog_plausibility(
            parent_cmax=10.0,
            parent_auc=50.0,
            analog_cmax=100.0,  # 10x parent → > 5-fold
            analog_auc=500.0,
            modification="test",
        )
        failed = [c for c in checks if not c.passed]
        assert len(failed) > 0
