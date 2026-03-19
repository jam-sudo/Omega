"""Drug registry module tests."""


def test_benchmark_drugs_importable():
    from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS

    assert isinstance(BENCHMARK_DRUGS, dict)
    assert len(BENCHMARK_DRUGS) >= 24


def test_benchmark_drugs_have_required_fields():
    from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS

    for name, info in BENCHMARK_DRUGS.items():
        assert "smiles" in info, f"{name} missing smiles"
        assert "dose_mg" in info, f"{name} missing dose_mg"
        assert isinstance(info["smiles"], str)
        assert info["dose_mg"] > 0


def test_core24_subset():
    from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS, CORE24_NAMES

    assert len(CORE24_NAMES) == 24
    for name in CORE24_NAMES:
        assert name in BENCHMARK_DRUGS, f"{name} not in BENCHMARK_DRUGS"


def test_get_core24():
    from omega_pbpk.data.drug_registry import get_core24

    core = get_core24()
    assert len(core) == 24
    assert "caffeine" in core
