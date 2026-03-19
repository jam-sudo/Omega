"""CmaxMetaLearner unit tests."""

import numpy as np


def test_meta_features_to_array():
    from omega_pbpk.ml.models.direct_pk.meta_learner import MetaFeatures

    mf = MetaFeatures(
        cmax_pbpk=1.0,
        cmax_ml=2.0,
        dose_mg=100.0,
        logP=2.5,
        TPSA_norm=0.3,
        MW_norm=0.5,
        fup=0.1,
        clint=5.0,
        is_acid=1.0,
        is_base=0.0,
        pgp_flag=0.0,
    )
    arr = mf.to_array()
    assert arr.shape == (12,)
    assert np.isfinite(arr).all()
    # log_cmax_ratio = log10(1.0) - log10(2.0) = 0 - 0.301 = -0.301
    assert abs(arr[3] - (np.log10(1.0) - np.log10(2.0))) < 0.01


def test_meta_learner_fallback():
    # Use a non-existent path so model is not loaded
    from pathlib import Path

    from omega_pbpk.ml.models.direct_pk.meta_learner import (
        CmaxMetaLearner,
        MetaFeatures,
    )

    learner = CmaxMetaLearner(model_path=Path("/tmp/nonexistent_meta_model.json"))
    assert not learner.is_loaded
    mf = MetaFeatures(
        cmax_pbpk=4.0,
        cmax_ml=1.0,
        dose_mg=100.0,
        logP=2.0,
        TPSA_norm=0.3,
        MW_norm=0.5,
        fup=0.1,
        clint=5.0,
        is_acid=0.0,
        is_base=0.0,
        pgp_flag=0.0,
    )
    result = learner.predict(mf)
    # Geometric mean of 4 and 1 = 2.0
    assert abs(result - 2.0) < 0.01


def test_feature_names_count():
    from omega_pbpk.ml.models.direct_pk.meta_learner import FEATURE_NAMES

    assert len(FEATURE_NAMES) == 12


def test_meta_features_edge_cases():
    """Test that edge cases (very small values) don't produce NaN/Inf."""
    from omega_pbpk.ml.models.direct_pk.meta_learner import MetaFeatures

    mf = MetaFeatures(
        cmax_pbpk=1e-15,
        cmax_ml=1e-15,
        dose_mg=0.001,
        logP=-3.0,
        TPSA_norm=0.0,
        MW_norm=0.1,
        fup=1e-6,
        clint=0.001,
        is_acid=0.0,
        is_base=1.0,
        pgp_flag=1.0,
    )
    arr = mf.to_array()
    assert arr.shape == (12,)
    assert np.isfinite(arr).all()


def test_meta_property():
    """Test that .meta returns a dict even when no model is loaded."""
    from pathlib import Path

    from omega_pbpk.ml.models.direct_pk.meta_learner import CmaxMetaLearner

    learner = CmaxMetaLearner(model_path=Path("/tmp/nonexistent_meta_model.json"))
    assert isinstance(learner.meta, dict)
    assert len(learner.meta) == 0
