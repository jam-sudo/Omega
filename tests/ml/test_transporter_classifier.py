"""Transporter substrate classifier tests."""

import pytest

TRANSPORTER_KEYS = ["pgp", "oatp1b1", "bcrp", "oct2", "oat", "pept1"]


def test_transporter_classifier_interface():
    """Classifier should return probabilities for all transporters."""
    from omega_pbpk.ml.corrections.transporter_classifier import TransporterClassifier

    clf = TransporterClassifier()
    probs = clf.predict("CC(=O)Oc1ccccc1C(=O)O")  # aspirin

    for key in TRANSPORTER_KEYS:
        assert key in probs, f"Missing key: {key}"
    for name, prob in probs.items():
        assert 0.0 <= prob <= 1.0, f"{name} probability {prob} out of range"


def test_transporter_untrained_returns_default():
    """When not trained, all probabilities should be 0.5."""
    from omega_pbpk.ml.corrections.transporter_classifier import TransporterClassifier

    clf = TransporterClassifier()
    if clf.is_trained:
        pytest.skip("Models already trained")
    probs = clf.predict("c1ccccc1")  # benzene
    for name, prob in probs.items():
        assert prob == 0.5, f"{name} should be 0.5 when untrained, got {prob}"


def test_transporter_invalid_smiles():
    """Invalid SMILES should return default probabilities without crashing."""
    from omega_pbpk.ml.corrections.transporter_classifier import TransporterClassifier

    clf = TransporterClassifier()
    probs = clf.predict("NOT_A_VALID_SMILES")
    assert len(probs) == len(TRANSPORTER_KEYS)
    for _name, prob in probs.items():
        assert 0.0 <= prob <= 1.0


def test_transporter_known_pgp_substrate():
    """If trained, verapamil should have high P-gp probability."""
    from omega_pbpk.ml.corrections.transporter_classifier import TransporterClassifier

    clf = TransporterClassifier()
    if not clf.is_trained:
        pytest.skip("Models not trained yet")
    # Verapamil SMILES (well-known P-gp substrate)
    probs = clf.predict("COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC")
    assert probs["pgp"] > 0.5, f"Verapamil P-gp prob should be >0.5, got {probs['pgp']}"


def test_transporter_known_oatp_substrate():
    """If trained, atorvastatin should have high OATP1B1 probability."""
    from omega_pbpk.ml.corrections.transporter_classifier import TransporterClassifier

    clf = TransporterClassifier()
    if not clf.is_trained:
        pytest.skip("Models not trained yet")
    # Atorvastatin SMILES
    probs = clf.predict(
        "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)[O-])c(-c2ccc(F)cc2)c(-c2ccccc2)c1C(=O)Nc1ccccc1"
    )
    assert probs["oatp1b1"] > 0.5, (
        f"Atorvastatin OATP1B1 prob should be >0.5, got {probs['oatp1b1']}"
    )


def test_transporter_feature_extraction():
    """Feature extraction should produce correct dimensionality."""
    from omega_pbpk.ml.corrections.transporter_classifier import (
        N_FEATURES,
        TransporterClassifier,
    )

    clf = TransporterClassifier()
    features = clf._extract_features("c1ccccc1")  # benzene
    assert features is not None
    assert features.shape == (N_FEATURES,), f"Expected {N_FEATURES} features, got {features.shape}"


def test_transporter_batch_consistency():
    """Calling predict twice on same SMILES should give same result."""
    from omega_pbpk.ml.corrections.transporter_classifier import TransporterClassifier

    clf = TransporterClassifier()
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    probs1 = clf.predict(smiles)
    probs2 = clf.predict(smiles)
    for key in TRANSPORTER_KEYS:
        assert probs1[key] == probs2[key], f"{key} differs between calls"
