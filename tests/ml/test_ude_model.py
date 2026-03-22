"""Tests for UDE multi-task PK model."""

import pytest

torch = pytest.importorskip("torch")


def test_model_forward():
    from omega_pbpk.ml.models.ude.model import MultiTaskPKModel

    model = MultiTaskPKModel()
    x = torch.randn(4, 2057)
    dose = torch.tensor([100.0, 200.0, 50.0, 400.0])
    cmax, pk_params = model.predict_cmax(x, dose)
    assert cmax.shape == (4,)
    assert all(c > 0 for c in cmax)


def test_pk_params_physical_defaults():
    """Zero input → bias-initialized defaults should be physical."""
    from omega_pbpk.ml.models.ude.model import MultiTaskPKModel

    model = MultiTaskPKModel()
    x = torch.zeros(1, 2057)
    dose = torch.tensor([100.0])
    cmax, params = model.predict_cmax(x, dose)
    F, Vd, ka, ke = params["F"], params["Vd"], params["ka"], params["ke"]
    assert 0 < F.item() < 1, f"F={F.item()}"
    assert 0.5 < Vd.item() < 5000, f"Vd={Vd.item()}"
    assert 0.01 < ka.item() < 20, f"ka={ka.item()}"
    assert 0.001 < ke.item() < 10, f"ke={ke.item()}"


def test_cmax_formula_known_case():
    """Test 1-cpt formula against hand-calculated result."""
    from omega_pbpk.ml.models.ude.model import safe_cmax_1cpt

    # F=1, dose=100mg, Vd=100L, ka=2/h, ke=0.1/h
    # tmax = ln(20)/1.9 = 1.578h
    # Cmax ≈ 0.854 mg/L
    F = torch.tensor([1.0])
    dose = torch.tensor([100.0])
    Vd = torch.tensor([100.0])
    ka = torch.tensor([2.0])
    ke = torch.tensor([0.1])
    cmax = safe_cmax_1cpt(F, dose, Vd, ka, ke)
    expected = 0.854
    assert abs(cmax.item() - expected) < 0.05, f"Cmax={cmax.item():.4f}, expected ~{expected}"


def test_cmax_singularity_ka_equals_ke():
    """When ka ≈ ke, the degenerate formula should be used."""
    from omega_pbpk.ml.models.ude.model import safe_cmax_1cpt

    F = torch.tensor([0.5])
    dose = torch.tensor([200.0])
    Vd = torch.tensor([50.0])
    ka = torch.tensor([1.005])
    ke = torch.tensor([1.000])
    cmax = safe_cmax_1cpt(F, dose, Vd, ka, ke)
    # Degenerate: Cmax = F * dose / (Vd * e) = 0.5 * 200 / (50 * 2.718) ≈ 0.736
    assert cmax.item() > 0
    assert cmax.item() < 10  # sanity


def test_gradient_flows():
    """Verify gradients propagate through the 1-cpt formula to encoder."""
    from omega_pbpk.ml.models.ude.model import MultiTaskPKModel

    model = MultiTaskPKModel()
    x = torch.randn(2, 2057)
    dose = torch.tensor([100.0, 200.0])
    cmax, _ = model.predict_cmax(x, dose)
    loss = cmax.sum()
    loss.backward()
    has_grad = False
    for _name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradient flowed to any parameter"


def test_auxiliary_heads():
    from omega_pbpk.ml.models.ude.model import MultiTaskPKModel

    model = MultiTaskPKModel()
    x = torch.randn(3, 2057)
    clint = model.predict_clint(x)
    fup = model.predict_fup(x)
    assert clint.shape == (3,)
    assert all(c > 0 for c in clint), f"CLint has non-positive: {clint}"
    assert fup.shape == (3,)
    assert all(0 < f <= 1 for f in fup), f"fup out of range: {fup}"


def test_batch_consistency():
    """Same input should produce same output regardless of batch composition."""
    from omega_pbpk.ml.models.ude.model import MultiTaskPKModel

    model = MultiTaskPKModel()
    model.eval()
    x1 = torch.randn(1, 2057)
    dose1 = torch.tensor([100.0])

    # Single prediction
    with torch.no_grad():
        cmax_single, _ = model.predict_cmax(x1, dose1)

    # In a batch of 3
    x_batch = torch.cat([x1, torch.randn(2, 2057)], dim=0)
    dose_batch = torch.tensor([100.0, 200.0, 50.0])
    with torch.no_grad():
        cmax_batch, _ = model.predict_cmax(x_batch, dose_batch)

    assert abs(cmax_single.item() - cmax_batch[0].item()) < 1e-5
