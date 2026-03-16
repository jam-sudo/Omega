"""Test that P-gp substrates get reduced bioavailability in the pipeline."""


class TestPgpCorrection:
    def test_verapamil_flagged_as_pgp(self):
        """Verapamil (known P-gp substrate) should be flagged in adme_properties."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        result = pipeline.simulate(
            SimulationRequest(
                smiles="CC(C)C(CCCN(C)CCC1=CC(=C(C=C1)OC)OC)(C#N)C2=CC(=C(C=C2)OC)OC",
                dose_mg=80.0,
                route="oral",
            )
        )
        assert result.adme_properties.get("pgp_substrate") is True

    def test_caffeine_not_pgp(self):
        """Caffeine (not a P-gp substrate) should not be flagged."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        result = pipeline.simulate(
            SimulationRequest(
                smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                dose_mg=200.0,
                route="oral",
            )
        )
        assert result.adme_properties.get("pgp_substrate") is False

    def test_pgp_warning_present(self):
        """P-gp correction should add a warning to the result."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        result = pipeline.simulate(
            SimulationRequest(
                smiles="CC(C)C(CCCN(C)CCC1=CC(=C(C=C1)OC)OC)(C#N)C2=CC(=C(C=C2)OC)OC",
                dose_mg=80.0,
                route="oral",
            )
        )
        pgp_warnings = [w for w in result.warnings if "P-gp" in w]
        assert len(pgp_warnings) > 0
