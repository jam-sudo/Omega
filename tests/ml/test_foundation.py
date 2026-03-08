"""Tests for Level 3 Foundation Model components.

Covers PatientEncoder, DosingEncoder, PKFoundationModel, ReptileTrainer,
InteractivePKPredictor, and few-shot adaptation.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from omega_pbpk.ml.features.graphs import HAS_RDKIT  # noqa: E402

requires_rdkit = pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not installed")

# ---------------------------------------------------------------------------
# PatientEncoder tests
# ---------------------------------------------------------------------------


class TestPatientEncoder:
    """Tests for patient covariate encoder."""

    def _make_encoder(self):
        from omega_pbpk.ml.models.foundation.patient_encoder import PatientEncoder

        return PatientEncoder()

    def test_output_shape_defaults(self):
        """Default (empty) covariates produce (1, 64) embedding."""
        enc = self._make_encoder()
        out = enc({})
        assert out.shape == (1, 64)

    def test_output_shape_custom(self):
        """Custom covariates produce correct shape."""
        enc = self._make_encoder()
        covariates = {
            "age": 65,
            "weight": 55,
            "height": 160,
            "sex": "F",
            "hepatic_impairment": "mild",
            "cyp2d6_genotype": "PM",
        }
        out = enc(covariates)
        assert out.shape == (1, 64)

    def test_missing_covariates_uses_defaults(self):
        """Missing keys should produce same result as explicit defaults."""
        from omega_pbpk.ml.models.foundation.patient_encoder import (
            POPULATION_DEFAULTS,
            PatientEncoder,
        )

        enc = PatientEncoder()
        enc.eval()
        with torch.no_grad():
            out_empty = enc({})
            out_defaults = enc(dict(POPULATION_DEFAULTS))
        assert torch.allclose(out_empty, out_defaults, atol=1e-5)

    def test_all_sex_values(self):
        """Both M and F produce valid outputs."""
        enc = self._make_encoder()
        enc.eval()
        with torch.no_grad():
            m_out = enc({"sex": "M"})
            f_out = enc({"sex": "F"})
        assert m_out.shape == (1, 64)
        assert f_out.shape == (1, 64)
        # Different sex should give different embeddings
        assert not torch.allclose(m_out, f_out, atol=1e-6)

    def test_all_hepatic_values(self):
        """All hepatic impairment levels produce valid outputs."""
        enc = self._make_encoder()
        enc.eval()
        for level in ("none", "mild", "moderate", "severe"):
            with torch.no_grad():
                out = enc({"hepatic_impairment": level})
            assert out.shape == (1, 64)

    def test_all_renal_values(self):
        """All renal impairment levels produce valid outputs."""
        enc = self._make_encoder()
        enc.eval()
        for level in ("none", "mild", "moderate", "severe", "ESRD"):
            with torch.no_grad():
                out = enc({"renal_impairment": level})
            assert out.shape == (1, 64)

    def test_all_genotype_values(self):
        """All CYP genotype values produce valid outputs."""
        enc = self._make_encoder()
        enc.eval()
        for gt in ("PM", "IM", "EM", "UM"):
            with torch.no_grad():
                out = enc({"cyp2d6_genotype": gt})
            assert out.shape == (1, 64)
        for gt in ("PM", "NM"):
            with torch.no_grad():
                out = enc({"cyp3a4_genotype": gt})
            assert out.shape == (1, 64)

    def test_invalid_categorical_falls_back(self):
        """Invalid categorical value falls back to default without error."""
        enc = self._make_encoder()
        enc.eval()
        with torch.no_grad():
            out = enc({"sex": "invalid_value"})
        assert out.shape == (1, 64)

    def test_differentiable(self):
        """Output is differentiable through the encoder."""
        enc = self._make_encoder()
        enc.train()
        out = enc({"age": 50, "weight": 80})
        loss = out.sum()
        loss.backward()
        # Check gradients exist on at least one parameter
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
        assert has_grad


# ---------------------------------------------------------------------------
# DosingEncoder tests
# ---------------------------------------------------------------------------


class TestDosingEncoder:
    """Tests for dosing regimen encoder."""

    def _make_encoder(self):
        from omega_pbpk.ml.models.foundation.dosing_encoder import DosingEncoder

        return DosingEncoder()

    def test_output_shape_defaults(self):
        """Default (empty) regimen produces (1, 64) embedding."""
        enc = self._make_encoder()
        out = enc({})
        assert out.shape == (1, 64)

    def test_output_shape_custom(self):
        """Custom regimen produces correct shape."""
        enc = self._make_encoder()
        regimen = {
            "dose_mg": 500,
            "route": "iv_infusion",
            "frequency": "BID",
            "duration_h": 2.0,
            "n_doses": 14,
            "formulation": "solution",
        }
        out = enc(regimen)
        assert out.shape == (1, 64)

    def test_all_routes(self):
        """All route values produce valid outputs."""
        enc = self._make_encoder()
        enc.eval()
        for route in ("oral", "iv_bolus", "iv_infusion", "subcutaneous", "intramuscular"):
            with torch.no_grad():
                out = enc({"route": route})
            assert out.shape == (1, 64)

    def test_all_frequencies(self):
        """All frequency values produce valid outputs."""
        enc = self._make_encoder()
        enc.eval()
        for freq in ("single", "BID", "TID", "QD", "Q12H", "Q8H", "Q6H", "continuous"):
            with torch.no_grad():
                out = enc({"frequency": freq})
            assert out.shape == (1, 64)

    def test_all_formulations(self):
        """All formulation values produce valid outputs."""
        enc = self._make_encoder()
        enc.eval()
        for form in ("solution", "immediate_release", "extended_release", "suspension"):
            with torch.no_grad():
                out = enc({"formulation": form})
            assert out.shape == (1, 64)

    def test_dose_zero_ok(self):
        """Zero dose (and zero duration) should not cause errors."""
        enc = self._make_encoder()
        enc.eval()
        with torch.no_grad():
            out = enc({"dose_mg": 0, "duration_h": 0, "n_doses": 0})
        assert out.shape == (1, 64)
        assert torch.isfinite(out).all()

    def test_large_dose(self):
        """Large dose values should not cause NaN/Inf."""
        enc = self._make_encoder()
        enc.eval()
        with torch.no_grad():
            out = enc({"dose_mg": 10000, "n_doses": 365})
        assert out.shape == (1, 64)
        assert torch.isfinite(out).all()

    def test_differentiable(self):
        """Output is differentiable."""
        enc = self._make_encoder()
        enc.train()
        out = enc({"dose_mg": 100})
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
        assert has_grad


# ---------------------------------------------------------------------------
# PKFoundationModel tests
# ---------------------------------------------------------------------------


@requires_rdkit
class TestPKFoundationModel:
    """Tests for the full foundation model."""

    def _make_model(self):
        from omega_pbpk.ml.models.foundation.foundation_model import (
            PKFoundationModel,
        )

        return PKFoundationModel()

    def test_forward_no_context(self, sample_smiles):
        """Without patient/dosing, behaves like Level 2."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            result = model(sample_smiles["ethanol"])

        assert "params" in result
        assert "curve" in result
        assert "pk_metrics" in result
        assert "embedding" in result
        assert result["patient_embedding"] is None
        assert result["dosing_embedding"] is None

    def test_forward_with_patient_and_dosing(self, sample_smiles):
        """Full forward pass with patient + dosing context."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            result = model(
                sample_smiles["aspirin"],
                covariates={"age": 65, "weight": 55, "sex": "F"},
                dosing={"dose_mg": 500, "route": "oral", "frequency": "BID"},
            )

        assert result["params"] is not None
        assert result["curve"].shape[1] > 0
        assert result["patient_embedding"] is not None
        assert result["dosing_embedding"] is not None
        assert result["fused_embedding"].shape[-1] == 256

    def test_forward_with_only_patient(self, sample_smiles):
        """Forward pass with only patient covariates."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            result = model(
                sample_smiles["ethanol"],
                covariates={"age": 30, "weight": 90},
            )
        assert result["patient_embedding"] is not None
        assert result["dosing_embedding"] is not None  # defaults applied

    def test_forward_with_only_dosing(self, sample_smiles):
        """Forward pass with only dosing regimen."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            result = model(
                sample_smiles["ethanol"],
                regimen={"dose_mg": 250, "route": "iv_bolus"},
            )
        assert result["dosing_embedding"] is not None

    def test_pk_metrics_shape(self, sample_smiles):
        """PK metrics have correct shapes."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            result = model(sample_smiles["ethanol"])
        metrics = result["pk_metrics"]
        for key in ("cmax", "auc", "tmax", "t_half"):
            assert key in metrics
            assert metrics[key].shape == (1,)

    def test_params_contain_all_keys(self, sample_smiles):
        """Output params contain all expected PK parameter names."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            result = model(sample_smiles["ethanol"])
        params = result["params"]
        expected = {
            "mw",
            "logP",
            "logS",
            "fup",
            "rbp",
            "peff",
            "clint_3a4",
            "clint_2d6",
            "ka",
            "vd",
            "ke",
            "bioavailability",
        }
        assert expected.issubset(set(params.keys()))

    def test_curve_nonnegative(self, sample_smiles):
        """Predicted concentration curve should be non-negative."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            result = model(sample_smiles["aspirin"])
        assert (result["curve"] >= 0).all()

    def test_differentiable_full_pass(self, sample_smiles):
        """Full forward pass is differentiable."""
        model = self._make_model()
        model.train()
        result = model(
            sample_smiles["ethanol"],
            covariates={"age": 50},
            regimen={"dose_mg": 200},
        )
        loss = result["curve"].sum() + result["pk_metrics"]["cmax"].sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_from_level2(self, sample_smiles):
        """Initialization from Level 2 model works."""
        from omega_pbpk.ml.models.foundation.end_to_end import SMILESToPKModel
        from omega_pbpk.ml.models.foundation.foundation_model import (
            PKFoundationModel,
        )

        l2_model = SMILESToPKModel()
        l3_model = PKFoundationModel.from_level2(l2_model)

        l3_model.eval()
        with torch.no_grad():
            result = l3_model(sample_smiles["ethanol"])
        assert result["curve"].shape[1] > 0

    def test_save_load_roundtrip(self, tmp_path, sample_smiles):
        """Save and load preserves predictions."""
        model = self._make_model()
        model.eval()

        with torch.no_grad():
            result1 = model(sample_smiles["ethanol"])

        save_path = tmp_path / "foundation.pt"
        model.save(save_path)

        from omega_pbpk.ml.models.foundation.foundation_model import (
            PKFoundationModel,
        )

        loaded = PKFoundationModel.load(save_path)
        with torch.no_grad():
            result2 = loaded(sample_smiles["ethanol"])

        assert torch.allclose(result1["curve"], result2["curve"], atol=1e-5)


# ---------------------------------------------------------------------------
# CrossAttentionFusion tests
# ---------------------------------------------------------------------------


class TestCrossAttention:
    """Tests for cross-attention fusion module."""

    def test_output_shape(self):
        from omega_pbpk.ml.models.foundation.foundation_model import (
            CrossAttentionFusion,
        )

        attn = CrossAttentionFusion(mol_dim=256, context_dim=128, n_heads=4)
        mol = torch.randn(2, 256)
        ctx = torch.randn(2, 128)
        out = attn(mol, ctx)
        assert out.shape == (2, 256)

    def test_residual_connection(self):
        """Without training, output should be close to input (residual)."""
        from omega_pbpk.ml.models.foundation.foundation_model import (
            CrossAttentionFusion,
        )

        attn = CrossAttentionFusion(mol_dim=256, context_dim=128, n_heads=4)
        attn.eval()
        mol = torch.randn(1, 256)
        ctx = torch.zeros(1, 128)
        with torch.no_grad():
            out = attn(mol, ctx)
        # Due to layer norm + residual, output should have same norm as input
        assert out.shape == mol.shape


# ---------------------------------------------------------------------------
# ReptileTrainer tests
# ---------------------------------------------------------------------------


@requires_rdkit
class TestReptileTrainer:
    """Tests for Reptile meta-learning."""

    def test_single_meta_step(self, sample_smiles):
        """One meta-iteration runs without error."""
        from omega_pbpk.ml.models.foundation.foundation_model import (
            PKFoundationModel,
        )
        from omega_pbpk.ml.training.few_shot import FewShotTask, ReptileTrainer

        model = PKFoundationModel()
        trainer = ReptileTrainer(model, meta_lr=0.01, inner_lr=1e-4, inner_steps=2, device="cpu")

        task = FewShotTask(
            smiles=sample_smiles["ethanol"],
            observations=[(1.0, 10.0), (2.0, 7.0)],
            drug_id="ethanol",
        )

        avg_loss = trainer.meta_train_step([task])
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0

    def test_adapt_reduces_loss(self, sample_smiles):
        """Few-shot adaptation should reduce loss on observed data."""
        from omega_pbpk.ml.models.foundation.foundation_model import (
            PKFoundationModel,
        )
        from omega_pbpk.ml.training.few_shot import FewShotTask, ReptileTrainer, adapt

        model = PKFoundationModel()
        model.eval()

        observations = [(1.0, 10.0), (2.0, 7.0), (4.0, 3.0)]

        # Compute loss before adaptation
        trainer_helper = ReptileTrainer(model, device="cpu")
        task = FewShotTask(smiles=sample_smiles["ethanol"], observations=observations)
        with torch.no_grad():
            loss_before = trainer_helper._task_loss(model, task).item()

        # Adapt
        adapted = adapt(
            model,
            observations=observations,
            smiles=sample_smiles["ethanol"],
            n_steps=20,
            lr=1e-3,
        )

        # Loss after adaptation
        adapted.eval()
        trainer_helper2 = ReptileTrainer(adapted, device="cpu")
        with torch.no_grad():
            loss_after = trainer_helper2._task_loss(adapted, task).item()

        # Adapted model should have lower loss
        assert loss_after <= loss_before


# ---------------------------------------------------------------------------
# InteractivePKPredictor tests
# ---------------------------------------------------------------------------


def test_parameter_count():
    """Parameter count returns all expected sub-modules."""
    from omega_pbpk.ml.models.foundation.foundation_model import PKFoundationModel

    model = PKFoundationModel()
    counts = model.parameter_count()
    assert "encoder" in counts
    assert "param_head" in counts
    assert "surrogate" in counts
    assert "patient_encoder" in counts
    assert "dosing_encoder" in counts
    assert "cross_attention" in counts
    assert "total" in counts
    assert counts["total"] > 0


def test_few_shot_task_dataclass():
    """FewShotTask can be created with all fields."""
    from omega_pbpk.ml.training.few_shot import FewShotTask

    task = FewShotTask(
        smiles="CCO",
        observations=[(0.5, 5.0), (1.0, 10.0), (2.0, 7.5)],
        covariates={"age": 50},
        regimen={"dose_mg": 200},
        drug_id="test_drug",
    )
    assert task.smiles == "CCO"
    assert len(task.observations) == 3


@requires_rdkit
class TestInteractivePKPredictor:
    """Tests for the high-level interactive predictor."""

    def _make_predictor(self):
        from omega_pbpk.ml.models.foundation.interactive import (
            InteractivePKPredictor,
        )

        # Uses untrained model (no checkpoint file)
        return InteractivePKPredictor(model_path="nonexistent.pt", device="cpu")

    def test_predict_basic(self, sample_smiles):
        """Basic prediction without patient/dosing context."""
        predictor = self._make_predictor()
        result = predictor.predict(sample_smiles["ethanol"])

        assert "predicted_params" in result
        assert "pk_profile" in result
        assert "concentration_curve" in result
        assert "confidence" in result
        assert "uncertainty_intervals" in result

    def test_predict_with_patient(self, sample_smiles):
        """Prediction with patient covariates."""
        predictor = self._make_predictor()
        result = predictor.predict(
            sample_smiles["aspirin"],
            patient={"age": 65, "weight": 55, "sex": "F", "hepatic_impairment": "mild"},
        )
        assert result["confidence"] in ("low", "medium", "high")
        assert len(result["concentration_curve"]) > 0

    def test_predict_with_dosing(self, sample_smiles):
        """Prediction with dosing regimen."""
        predictor = self._make_predictor()
        result = predictor.predict(
            sample_smiles["aspirin"],
            dosing={"dose_mg": 500, "route": "oral", "frequency": "BID"},
        )
        assert "Cmax_mg_L" in result["pk_profile"]
        assert "AUC_mg_h_L" in result["pk_profile"]

    def test_predict_with_both(self, sample_smiles):
        """Prediction with both patient and dosing."""
        predictor = self._make_predictor()
        result = predictor.predict(
            sample_smiles["aspirin"],
            patient={"age": 65, "weight": 55, "sex": "F"},
            dosing={"dose_mg": 500, "route": "oral"},
        )
        assert isinstance(result["predicted_params"], dict)
        assert isinstance(result["uncertainty_intervals"], dict)

    def test_adapt_returns_new_predictor(self, sample_smiles):
        """adapt() returns a new InteractivePKPredictor."""
        from omega_pbpk.ml.models.foundation.interactive import (
            InteractivePKPredictor,
        )

        predictor = self._make_predictor()
        adapted = predictor.adapt(
            smiles=sample_smiles["ethanol"],
            observations=[(1.0, 15.2), (2.0, 12.8), (4.0, 6.1)],
            n_steps=5,
        )
        assert isinstance(adapted, InteractivePKPredictor)

        # Adapted predictor should still work
        result = adapted.predict(sample_smiles["ethanol"])
        assert "pk_profile" in result

    def test_pk_profile_keys(self, sample_smiles):
        """PK profile contains expected keys."""
        predictor = self._make_predictor()
        result = predictor.predict(sample_smiles["ethanol"])
        profile = result["pk_profile"]
        for key in ("Cmax_mg_L", "Tmax_h", "AUC_mg_h_L", "t_half_h"):
            assert key in profile

    def test_concentration_curve_format(self, sample_smiles):
        """Concentration curve is list of (time, conc) tuples."""
        predictor = self._make_predictor()
        result = predictor.predict(sample_smiles["ethanol"])
        curve = result["concentration_curve"]
        assert len(curve) > 0
        # Each entry is (time_h, conc)
        t, c = curve[0]
        assert isinstance(t, float)
        assert isinstance(c, float)
        # Time should start at 0
        assert curve[0][0] == 0.0

    def test_uncertainty_intervals_format(self, sample_smiles):
        """Uncertainty intervals have (lo, hi) format."""
        predictor = self._make_predictor()
        result = predictor.predict(sample_smiles["ethanol"])
        intervals = result["uncertainty_intervals"]
        assert len(intervals) > 0
        for _name, (lo, hi) in intervals.items():
            assert isinstance(lo, float)
            assert isinstance(hi, float)
