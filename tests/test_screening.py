"""Tests for batch screening engine."""


class TestBatchScreening:
    def test_batch_predict_multiple(self):
        """Batch prediction on 3 SMILES should return 3 results."""
        from omega_pbpk.screening.batch import batch_predict

        smiles_list = [
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
            "CC(=O)NC1=CC=C(O)C=C1",  # acetaminophen
            "CC(C)CC1=CC=C(CC(C)C(=O)O)C=C1",  # ibuprofen
        ]
        results = batch_predict(smiles_list, dose_mg=200.0)
        assert len(results) == 3
        assert all(r["cmax_mg_L"] > 0 for r in results)
        assert all("smiles" in r for r in results)

    def test_batch_handles_invalid_smiles(self):
        """Invalid SMILES should not crash batch — pipeline falls back to defaults."""
        from omega_pbpk.screening.batch import batch_predict

        results = batch_predict(["INVALID_SMILES_XYZ", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"])
        assert len(results) == 2
        # Pipeline uses default ADME for invalid SMILES (doesn't error)
        # Both should return valid predictions
        assert all("smiles" in r for r in results)
        assert results[1]["cmax_mg_L"] > 0

    def test_batch_ranking(self):
        """Results should be rankable by score."""
        from omega_pbpk.screening.batch import batch_predict, rank_results

        smiles_list = [
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
            "CC(=O)NC1=CC=C(O)C=C1",  # acetaminophen
        ]
        results = batch_predict(smiles_list, dose_mg=200.0)
        ranked = rank_results(results, objective="cmax")
        assert ranked[0]["cmax_mg_L"] >= ranked[1]["cmax_mg_L"]
        assert ranked[0]["rank"] == 1

    def test_ranking_errors_at_end(self):
        """Errors should appear at end of ranked list."""
        from omega_pbpk.screening.batch import rank_results

        results = [
            {"smiles": "A", "error": "bad"},
            {"smiles": "B", "cmax_mg_L": 1.0},
            {"smiles": "C", "cmax_mg_L": 2.0},
        ]
        ranked = rank_results(results, objective="cmax")
        assert ranked[0]["cmax_mg_L"] == 2.0
        assert ranked[-1]["smiles"] == "A"
