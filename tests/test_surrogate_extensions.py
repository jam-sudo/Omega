import numpy as np
import pytest
from omega_pbpk.surrogate import PKSurrogate
from omega_pbpk.surrogate.data_generator import generate_training_data


def test_expected_features_constant():
    assert PKSurrogate.EXPECTED_FEATURES == ["logP", "fup", "clint_L_h", "mw", "rbp", "peff"]
    assert len(PKSurrogate.FULL_FEATURES) == 18


def test_full_features_composition():
    assert PKSurrogate.FULL_FEATURES[:6] == PKSurrogate.EXPECTED_FEATURES
    assert PKSurrogate.FULL_FEATURES[6:15] == PKSurrogate.PATIENT_FEATURES
    assert PKSurrogate.FULL_FEATURES[15:] == PKSurrogate.REGIMEN_FEATURES
    assert len(PKSurrogate.PATIENT_FEATURES) == 9
    assert len(PKSurrogate.REGIMEN_FEATURES) == 3


def test_6d_backward_compat():
    """기존 6D 모드 하위호환 확인."""
    data = generate_training_data(n_samples=50, n_workers=1)
    model = PKSurrogate(n_input=6)
    model.train(data.X, data.y, epochs=5, verbose=False)
    x = data.X[0]
    pred = model.predict(x)
    assert pred.shape == (4,)
    assert np.all(pred > 0)


def test_conformal_calibration_coverage():
    """Conformal prediction이 90% coverage를 근사하는지 확인."""
    data = generate_training_data(n_samples=200, n_workers=1)
    n_train = 150
    X_train, y_train = data.X[:n_train], data.y[:n_train]
    X_cal, y_cal = data.X[n_train:], data.y[n_train:]

    model = PKSurrogate(n_input=6)
    model.train(X_train, y_train, epochs=20, verbose=False)
    model.calibrate(X_cal, y_cal)

    # test coverage on calibration set (should be ~90%)
    covered = 0
    for x, y_true_norm in zip(X_cal, y_cal):
        y_true_orig = y_true_norm * model.y_std + model.y_mean
        y_pred, ci_lo, ci_hi = model.predict_conformal(x, alpha=0.10)
        if np.all(ci_lo <= y_true_orig) and np.all(y_true_orig <= ci_hi):
            covered += 1
    coverage = covered / len(X_cal)
    assert coverage >= 0.80, f"Coverage {coverage:.2%} < 80%"


def test_conformal_not_calibrated_raises():
    """calibrate() 없이 predict_conformal() 호출 시 RuntimeError."""
    model = PKSurrogate(n_input=6)
    x = np.ones(6)
    with pytest.raises(RuntimeError):
        model.predict_conformal(x)


def test_feature_mismatch_raises():
    """FULL_FEATURES 불일치 시 ValueError."""
    model = PKSurrogate(n_input=18)
    with pytest.raises((ValueError, RuntimeError)):
        model.predict_with_patient(
            drug_params={"logP": 1.0},  # 불완전한 dict
            patient_params={},
            regimen_params={},
        )


def test_predict_with_patient_wrong_n_input():
    """n_input=6 모델에서 predict_with_patient() 호출 시 ValueError."""
    model = PKSurrogate(n_input=6)
    with pytest.raises(ValueError, match="n_input=18"):
        model.predict_with_patient(
            drug_params={k: 1.0 for k in PKSurrogate.EXPECTED_FEATURES},
            patient_params={k: 1.0 for k in PKSurrogate.PATIENT_FEATURES},
            regimen_params={k: 1.0 for k in PKSurrogate.REGIMEN_FEATURES},
        )


def test_predict_with_patient_18d():
    """18D 모델에서 predict_with_patient()가 올바른 shape를 반환하는지 확인."""
    model = PKSurrogate(n_input=18)
    # Generate synthetic 18D training data
    rng = np.random.default_rng(0)
    n = 50
    X = rng.uniform(0.1, 2.0, (n, 18))
    y = rng.uniform(0.5, 10.0, (n, 4))
    model.train(X, y, epochs=5, verbose=False)

    drug_params = {k: 1.0 for k in PKSurrogate.EXPECTED_FEATURES}
    patient_params = {k: 1.0 for k in PKSurrogate.PATIENT_FEATURES}
    regimen_params = {k: 1.0 for k in PKSurrogate.REGIMEN_FEATURES}

    pred = model.predict_with_patient(drug_params, patient_params, regimen_params)
    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))
