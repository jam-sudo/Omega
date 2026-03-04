"""Unit tests for PKSurrogate feature contract, output validity, and reproducibility.

These tests validate the ADME-Plugin → Surrogate data contract so that any
ML-first reconstruction that changes the feature set triggers an early,
clear failure rather than a silent dimension mismatch.
"""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.surrogate import PKSurrogate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def untrained_surrogate() -> PKSurrogate:
    """Default PKSurrogate with He-initialized (random) weights."""
    return PKSurrogate()


@pytest.fixture
def trained_surrogate() -> PKSurrogate:
    """Minimal trained surrogate on synthetic data for output shape/sign tests."""
    rng = np.random.default_rng(42)
    n = 80
    X = rng.uniform([[-2, 0.01, 0.1, 100, 0.5, 0.5]], [[6, 1.0, 200, 800, 5.0, 10.0]], (n, 6))
    # Synthetic PK outputs: Cmax, AUC, Tmax, t_half (all positive)
    y = np.column_stack(
        [
            rng.uniform(0.01, 10, n),  # Cmax
            rng.uniform(0.1, 100, n),  # AUC
            rng.uniform(0.25, 6, n),  # Tmax
            rng.uniform(1, 20, n),  # t_half
        ]
    )
    s = PKSurrogate(n_input=6, n_output=4)
    s.param_names = PKSurrogate.EXPECTED_FEATURES
    s.output_names = PKSurrogate.EXPECTED_OUTPUTS
    s.train(X, y, epochs=5, verbose=False)
    return s


# ---------------------------------------------------------------------------
# T04a: EXPECTED_FEATURES class constant
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExpectedFeaturesContract:
    def test_expected_features_is_class_attribute(self):
        assert hasattr(PKSurrogate, "EXPECTED_FEATURES")

    def test_expected_features_length(self):
        assert len(PKSurrogate.EXPECTED_FEATURES) == 6

    def test_expected_features_names(self):
        assert PKSurrogate.EXPECTED_FEATURES == ["logP", "fup", "clint_L_h", "mw", "rbp", "peff"]

    def test_expected_outputs_is_class_attribute(self):
        assert hasattr(PKSurrogate, "EXPECTED_OUTPUTS")

    def test_expected_outputs_names(self):
        assert PKSurrogate.EXPECTED_OUTPUTS == ["cmax_mg_L", "auc_mg_h_L", "tmax_h", "t_half_h"]

    def test_expected_features_match_n_input_default(self):
        """Default n_input must equal len(EXPECTED_FEATURES)."""
        s = PKSurrogate()
        assert s.n_input == len(PKSurrogate.EXPECTED_FEATURES)

    def test_expected_outputs_match_n_output_default(self):
        s = PKSurrogate()
        assert s.n_output == len(PKSurrogate.EXPECTED_OUTPUTS)


# ---------------------------------------------------------------------------
# T04b: validate_feature_contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateFeatureContract:
    def test_valid_params_pass(self, untrained_surrogate):
        valid = {"logP": 2.0, "fup": 0.5, "clint_L_h": 10.0, "mw": 300.0, "rbp": 1.0, "peff": 2.0}
        untrained_surrogate.validate_feature_contract(valid)  # no exception

    def test_missing_key_raises_value_error(self, untrained_surrogate):
        missing_peff = {"logP": 2.0, "fup": 0.5, "clint_L_h": 10.0, "mw": 300.0, "rbp": 1.0}
        with pytest.raises(ValueError, match="feature contract violation"):
            untrained_surrogate.validate_feature_contract(missing_peff)

    def test_missing_multiple_keys_listed_in_error(self, untrained_surrogate):
        partial = {"logP": 2.0}
        with pytest.raises(ValueError) as exc_info:
            untrained_surrogate.validate_feature_contract(partial)
        msg = str(exc_info.value)
        assert "fup" in msg or "clint_L_h" in msg

    def test_extra_keys_allowed(self, untrained_surrogate):
        """Extra keys from ADME plugin must not cause failures."""
        extra = {
            "logP": 2.0,
            "fup": 0.5,
            "clint_L_h": 10.0,
            "mw": 300.0,
            "rbp": 1.0,
            "peff": 2.0,
            "logD": 1.5,
            "pka": 7.4,  # extra keys
        }
        untrained_surrogate.validate_feature_contract(extra)  # no exception

    def test_wrong_feature_name_raises(self, untrained_surrogate):
        """logD instead of logP should raise (missing logP)."""
        wrong_names = {
            "logD": 2.0,
            "fup": 0.5,
            "clint_L_h": 10.0,
            "mw": 300.0,
            "rbp": 1.0,
            "peff": 2.0,
        }
        with pytest.raises(ValueError, match="logP"):
            untrained_surrogate.validate_feature_contract(wrong_names)


# ---------------------------------------------------------------------------
# Non-negativity: outputs must always be ≥ 0
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNonNegativity:
    def test_predict_returns_nonnegative_single(self, trained_surrogate):
        x = np.array([2.0, 0.5, 10.0, 300.0, 1.0, 2.0])
        y = trained_surrogate.predict(x)
        assert np.all(y >= 0), f"Negative predictions: {y}"

    def test_predict_returns_nonnegative_batch(self, trained_surrogate):
        rng = np.random.default_rng(0)
        X = rng.uniform([[-2, 0.01, 0.1, 100, 0.5, 0.5]], [[6, 1.0, 200, 800, 5.0, 10.0]], (50, 6))
        y = trained_surrogate.predict(X)
        assert np.all(y >= 0), f"Batch has negative predictions: {y.min(axis=0)}"

    def test_predict_with_extreme_inputs_nonnegative(self, trained_surrogate):
        """Even very extreme inputs must not produce negative outputs."""
        extremes = np.array(
            [
                [10.0, 0.001, 0.001, 2000.0, 10.0, 0.001],  # extreme high logP
                [-5.0, 1.0, 500.0, 50.0, 0.1, 50.0],  # extreme negative logP
            ]
        )
        y = trained_surrogate.predict(extremes)
        assert np.all(y >= 0)

    def test_untrained_predict_nonnegative(self, untrained_surrogate):
        """Even untrained surrogate must clamp outputs to ≥ 0."""
        x = np.array([2.0, 0.5, 10.0, 300.0, 1.0, 2.0])
        y = untrained_surrogate.predict(x)
        assert np.all(y >= 0)


# ---------------------------------------------------------------------------
# Reproducibility (determinism)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestReproducibility:
    def test_same_input_same_output_trained(self, trained_surrogate):
        x = np.array([2.0, 0.5, 10.0, 300.0, 1.0, 2.0])
        y1 = trained_surrogate.predict(x)
        y2 = trained_surrogate.predict(x)
        np.testing.assert_array_equal(y1, y2)

    def test_batch_vs_single_consistency(self, trained_surrogate):
        """Batch prediction must equal row-by-row single predictions."""
        rng = np.random.default_rng(7)
        X = rng.uniform(0.1, 5.0, (10, 6))
        y_batch = trained_surrogate.predict(X)
        y_single = np.array([trained_surrogate.predict(X[i]) for i in range(10)])
        np.testing.assert_allclose(y_batch, y_single, rtol=1e-10)

    def test_save_load_round_trip(self, trained_surrogate, tmp_path):
        """Saved and reloaded model must produce identical outputs."""
        save_dir = tmp_path / "surrogate_test"
        trained_surrogate.save(save_dir)
        loaded = PKSurrogate.load(save_dir)
        x = np.array([1.5, 0.3, 20.0, 250.0, 0.8, 3.0])
        y_orig = trained_surrogate.predict(x)
        y_loaded = loaded.predict(x)
        np.testing.assert_allclose(y_orig, y_loaded, rtol=1e-10)

    def test_training_with_same_seed_reproducible(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 5.0, (40, 6))
        y = rng.uniform(0.1, 10.0, (40, 4))
        s1 = PKSurrogate(n_input=6, n_output=4)
        s2 = PKSurrogate(n_input=6, n_output=4)
        s1.train(X, y, epochs=3, seed=0, verbose=False)
        s2.train(X, y, epochs=3, seed=0, verbose=False)
        x = np.array([2.0, 0.5, 10.0, 300.0, 1.0, 2.0])
        np.testing.assert_allclose(s1.predict(x), s2.predict(x), rtol=1e-10)


# ---------------------------------------------------------------------------
# Output shape & structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOutputShape:
    def test_predict_single_returns_1d_array(self, trained_surrogate):
        x = np.array([2.0, 0.5, 10.0, 300.0, 1.0, 2.0])
        y = trained_surrogate.predict(x)
        assert y.ndim == 1
        assert len(y) == 4

    def test_predict_batch_returns_2d_array(self, trained_surrogate):
        X = np.ones((5, 6))
        y = trained_surrogate.predict(X)
        assert y.ndim == 2
        assert y.shape == (5, 4)

    def test_all_outputs_finite(self, trained_surrogate):
        rng = np.random.default_rng(1)
        X = rng.uniform(0.1, 5.0, (20, 6))
        y = trained_surrogate.predict(X)
        assert np.all(np.isfinite(y)), f"Non-finite predictions: {y}"
