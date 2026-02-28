"""Integration tests: Surrogate model output must approximate ODE output.

Validates the Surrogate ↔ ODE agreement contract:
  - AUC relative error  ≤ 20% (vs ODE on same drug)
  - Cmax relative error ≤ 25% (vs ODE on same drug)

These thresholds are intentionally generous to allow for the inherent
approximation of the MLP surrogate; they are designed to catch catastrophic
mismatches (e.g., feature dimension mismatch, un-normalised inputs, wrong
output order) rather than evaluate prediction accuracy.

The surrogate must be trained first using build_training_dataset.
Tests that require a trained surrogate are marked `@pytest.mark.slow`
because training takes several seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUC_RE_MAX = 0.20   # surrogate AUC must be within 20% of ODE AUC
CMAX_RE_MAX = 0.25  # surrogate Cmax must be within 25% of ODE Cmax

# Canonical parameter set for test drugs (input feature order matches EXPECTED_FEATURES)
# [logP, fup, clint_L_h, mw, rbp, peff]
TEST_DRUGS = {
    "caffeine":   [0.07, 0.65, 2.0,  194.19, 0.8,  4.0],
    "midazolam":  [3.89, 0.03, 60.0, 325.77, 0.53, 2.0],
    "propranolol":[3.48, 0.13, 70.0, 259.34, 0.81, 3.0],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_ode(params: list[float], dose_mg: float = 100.0) -> tuple[float, float]:
    """Run ODE simulation for given params, return (Cmax, AUC)."""
    from omega_pbpk._compat import np_trapz
    from omega_pbpk.core.body import WholeBodyPBPK
    from omega_pbpk.drugs.drug import Drug

    logP, fup, clint_L_h, mw, rbp, peff = params
    drug = Drug(
        name="test",
        mw=mw,
        logP=logP,
        fup=max(fup, 0.001),
        rbp=rbp,
        clint_hepatic_L_per_h=clint_L_h,
        peff=peff,
    )
    model = WholeBodyPBPK(drug=drug)
    model.setup_oral(dose_mg=dose_mg)
    result = model.simulate(t_end_h=24.0)
    cp = result.plasma_concentration()
    t = result.time_h
    cmax = float(np.max(cp))
    auc = float(np_trapz(cp, t))
    return cmax, auc


def _train_surrogate_on_drugs(drug_params: list[list[float]], n_aug: int = 15) -> "PKSurrogate":  # noqa: F821
    """Train a minimal surrogate on the test drugs with augmentation."""
    from omega_pbpk.surrogate import PKSurrogate
    from omega_pbpk.surrogate.train import build_training_dataset

    X, y = build_training_dataset(n_samples=n_aug, seed=42)
    s = PKSurrogate(n_input=6, n_output=4)
    s.param_names = PKSurrogate.EXPECTED_FEATURES
    s.output_names = PKSurrogate.EXPECTED_OUTPUTS
    s.train(X, y, epochs=50, verbose=False)
    return s


# ---------------------------------------------------------------------------
# Contract: surrogate EXPECTED_FEATURES matches ODE input convention
# ---------------------------------------------------------------------------

class TestFeatureOrderContract:
    def test_surrogate_feature_order_matches_train_order(self):
        """EXPECTED_FEATURES must match the column order in train._params_to_array."""
        from omega_pbpk.surrogate import PKSurrogate
        from omega_pbpk.surrogate.train import _params_to_array

        expected = PKSurrogate.EXPECTED_FEATURES
        ref_drug = {
            "logP": 2.0, "fup": 0.5, "clint": 10.0,
            "mw": 300.0, "rbp": 1.0, "peff": 2.0,
        }
        arr = _params_to_array(ref_drug)
        for i, feat_name in enumerate(expected):
            # Verify each feature name maps to the correct index
            # by checking that manually extracting the value matches
            key_map = {"clint_L_h": "clint"}
            ref_key = key_map.get(feat_name, feat_name)
            assert float(arr[i]) == pytest.approx(ref_drug[ref_key], rel=1e-9), (
                f"Feature '{feat_name}' at index {i} does not match "
                f"_params_to_array output ({arr[i]} != {ref_drug[ref_key]})"
            )

    def test_expected_features_count_matches_default_n_input(self):
        from omega_pbpk.surrogate import PKSurrogate
        assert len(PKSurrogate.EXPECTED_FEATURES) == PKSurrogate().n_input

    def test_expected_outputs_count_matches_default_n_output(self):
        from omega_pbpk.surrogate import PKSurrogate
        assert len(PKSurrogate.EXPECTED_OUTPUTS) == PKSurrogate().n_output


# ---------------------------------------------------------------------------
# ODE: baseline outputs for reference drugs
# ---------------------------------------------------------------------------

class TestODEBaseline:
    def test_ode_caffeine_positive_cmax_auc(self):
        cmax, auc = _run_ode(TEST_DRUGS["caffeine"])
        assert cmax > 0
        assert auc > 0

    def test_ode_midazolam_positive_cmax_auc(self):
        cmax, auc = _run_ode(TEST_DRUGS["midazolam"])
        assert cmax > 0
        assert auc > 0

    def test_ode_outputs_finite(self):
        for name, params in TEST_DRUGS.items():
            cmax, auc = _run_ode(params)
            assert np.isfinite(cmax), f"{name}: Cmax is not finite"
            assert np.isfinite(auc), f"{name}: AUC is not finite"


# ---------------------------------------------------------------------------
# Surrogate vs ODE agreement (slow — requires training)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSurrogateVsODE:
    @pytest.fixture(scope="class")
    def trained_surrogate(self):
        params_list = list(TEST_DRUGS.values())
        return _train_surrogate_on_drugs(params_list)

    def _check_agreement(self, surrogate, params: list[float], drug_name: str):
        x = np.array(params, dtype=float)
        y_surr = surrogate.predict(x)
        cmax_surr = float(y_surr[0])
        auc_surr = float(y_surr[1])

        cmax_ode, auc_ode = _run_ode(params)

        auc_re = abs(auc_surr - auc_ode) / max(auc_ode, 1e-12)
        cmax_re = abs(cmax_surr - cmax_ode) / max(cmax_ode, 1e-12)

        assert auc_re <= AUC_RE_MAX, (
            f"[{drug_name}] Surrogate AUC RE={auc_re:.3f} > {AUC_RE_MAX}. "
            f"surrogate={auc_surr:.4f}, ode={auc_ode:.4f}"
        )
        assert cmax_re <= CMAX_RE_MAX, (
            f"[{drug_name}] Surrogate Cmax RE={cmax_re:.3f} > {CMAX_RE_MAX}. "
            f"surrogate={cmax_surr:.4f}, ode={cmax_ode:.4f}"
        )

    def test_caffeine_agreement(self, trained_surrogate):
        self._check_agreement(trained_surrogate, TEST_DRUGS["caffeine"], "caffeine")

    def test_midazolam_agreement(self, trained_surrogate):
        self._check_agreement(trained_surrogate, TEST_DRUGS["midazolam"], "midazolam")

    def test_propranolol_agreement(self, trained_surrogate):
        self._check_agreement(trained_surrogate, TEST_DRUGS["propranolol"], "propranolol")

    def test_surrogate_outputs_nonnegative(self, trained_surrogate):
        for params in TEST_DRUGS.values():
            y = trained_surrogate.predict(np.array(params))
            assert np.all(y >= 0)

    def test_surrogate_outputs_finite(self, trained_surrogate):
        for params in TEST_DRUGS.values():
            y = trained_surrogate.predict(np.array(params))
            assert np.all(np.isfinite(y))
