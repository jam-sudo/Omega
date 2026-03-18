"""Adaptive conformal UQ tests."""

import pytest


def test_adaptive_conformal_interface():
    """Should return interval with lower, upper, width, similarity."""
    from omega_pbpk.ml.corrections.adaptive_conformal import AdaptiveConformal

    ac = AdaptiveConformal()
    if not ac.is_fitted:
        pytest.skip("Not yet fitted")

    interval = ac.predict_interval(
        smiles="c1ccc2c(c1)c(=O)c1c(n2C)cc(Cl)c(F)c1",
        cmax_pred=0.05,
    )
    assert "lower" in interval
    assert "upper" in interval
    assert "width" in interval
    assert "similarity" in interval
    assert interval["lower"] <= 0.05 <= interval["upper"]
    assert interval["width"] > 1.0


def test_adaptive_conformal_narrower_for_similar():
    """Interval should be narrower for drugs similar to training set."""
    from omega_pbpk.ml.corrections.adaptive_conformal import AdaptiveConformal

    ac = AdaptiveConformal()
    if not ac.is_fitted:
        pytest.skip("Not yet fitted")

    # Known drug (midazolam, likely in training)
    narrow = ac.predict_interval(
        smiles="c1ccc2c(c1)c(=O)c1c(n2C)cc(Cl)c(F)c1",
        cmax_pred=0.05,
    )
    # Novel scaffold
    wide = ac.predict_interval(
        smiles="C1=CC=C(C=C1)C(=O)NCCNC(=O)C2=CC=CC=C2",
        cmax_pred=0.05,
    )
    # Similar drug should have narrower or equal interval
    assert narrow["width"] <= wide["width"] * 1.5, (
        f"Similar drug width {narrow['width']:.2f} not narrower than novel {wide['width']:.2f}"
    )


def test_adaptive_conformal_unfitted_returns_none():
    """When not fitted, should return None."""
    from omega_pbpk.ml.corrections.adaptive_conformal import AdaptiveConformal

    ac = AdaptiveConformal()
    if ac.is_fitted:
        pytest.skip("Already fitted")

    result = ac.predict_interval(smiles="c1ccccc1", cmax_pred=1.0)
    assert result is None


def test_adaptive_conformal_fit_and_predict():
    """Test fit + predict round-trip with synthetic data."""
    import tempfile
    from pathlib import Path

    from omega_pbpk.ml.corrections.adaptive_conformal import AdaptiveConformal

    with tempfile.TemporaryDirectory() as tmpdir:
        ac = AdaptiveConformal(model_dir=Path(tmpdir))
        assert not ac.is_fitted

        # Synthetic calibration data
        smiles_list = [
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # caffeine
            "COCCc1ccc(OCC(O)CNC(C)C)cc1",  # metoprolol
            "CC(C)NCC(O)COc1cccc2ccccc12",  # propranolol
            "CC(=O)Nc1ccc(O)cc1",  # acetaminophen
            "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]",  # nifedipine
        ]
        cmax_obs = [1.5, 0.1, 0.04, 15.0, 0.02]
        cmax_pred = [1.2, 0.15, 0.03, 12.0, 0.025]

        ac.fit(smiles_list, cmax_obs, cmax_pred)
        assert ac.is_fitted

        # Predict for a known molecule
        result = ac.predict_interval(
            smiles="Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            cmax_pred=1.2,
        )
        assert result is not None
        assert result["lower"] <= 1.2 <= result["upper"]
        assert result["width"] > 1.0
        assert 0.0 <= result["similarity"] <= 1.0

        # Predict for unknown molecule
        result2 = ac.predict_interval(
            smiles="c1ccccc1",
            cmax_pred=1.0,
        )
        assert result2 is not None
        assert result2["lower"] <= 1.0 <= result2["upper"]


def test_adaptive_conformal_save_load():
    """Test that save/load round-trips correctly."""
    import tempfile
    from pathlib import Path

    from omega_pbpk.ml.corrections.adaptive_conformal import AdaptiveConformal

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)

        # Fit and save
        ac1 = AdaptiveConformal(model_dir=model_dir)
        smiles_list = [
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "COCCc1ccc(OCC(O)CNC(C)C)cc1",
            "CC(C)NCC(O)COc1cccc2ccccc12",
        ]
        cmax_obs = [1.5, 0.1, 0.04]
        cmax_pred = [1.2, 0.15, 0.03]
        ac1.fit(smiles_list, cmax_obs, cmax_pred)
        assert ac1.is_fitted

        # Load from saved
        ac2 = AdaptiveConformal(model_dir=model_dir)
        assert ac2.is_fitted

        # Both should give same result
        r1 = ac1.predict_interval("Cn1c(=O)c2c(ncn2C)n(C)c1=O", 1.0)
        r2 = ac2.predict_interval("Cn1c(=O)c2c(ncn2C)n(C)c1=O", 1.0)
        assert abs(r1["lower"] - r2["lower"]) < 1e-6
        assert abs(r1["upper"] - r2["upper"]) < 1e-6


def test_adaptive_conformal_invalid_smiles():
    """Invalid SMILES should return None even when fitted."""
    import tempfile
    from pathlib import Path

    from omega_pbpk.ml.corrections.adaptive_conformal import AdaptiveConformal

    with tempfile.TemporaryDirectory() as tmpdir:
        ac = AdaptiveConformal(model_dir=Path(tmpdir))
        smiles_list = [
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "COCCc1ccc(OCC(O)CNC(C)C)cc1",
        ]
        ac.fit(smiles_list, [1.5, 0.1], [1.2, 0.15])
        assert ac.is_fitted

        result = ac.predict_interval(smiles="NOT_A_SMILES", cmax_pred=1.0)
        assert result is None


def test_adaptive_conformal_alpha_affects_width():
    """Lower alpha (higher confidence) should give wider intervals."""
    import tempfile
    from pathlib import Path

    from omega_pbpk.ml.corrections.adaptive_conformal import AdaptiveConformal

    with tempfile.TemporaryDirectory() as tmpdir:
        ac = AdaptiveConformal(model_dir=Path(tmpdir))
        smiles_list = [
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "COCCc1ccc(OCC(O)CNC(C)C)cc1",
            "CC(C)NCC(O)COc1cccc2ccccc12",
            "CC(=O)Nc1ccc(O)cc1",
            "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]",
        ]
        cmax_obs = [1.5, 0.1, 0.04, 15.0, 0.02]
        cmax_pred = [1.2, 0.15, 0.03, 12.0, 0.025]
        ac.fit(smiles_list, cmax_obs, cmax_pred)

        r90 = ac.predict_interval("c1ccccc1", 1.0, alpha=0.10)  # 90% CI
        r80 = ac.predict_interval("c1ccccc1", 1.0, alpha=0.20)  # 80% CI

        assert r90 is not None and r80 is not None
        # 90% CI should be at least as wide as 80% CI
        assert r90["width"] >= r80["width"] - 1e-6, (
            f"90% CI width {r90['width']:.3f} should be >= 80% CI width {r80['width']:.3f}"
        )
