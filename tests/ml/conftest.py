"""Shared fixtures for ML tests."""

import pytest


@pytest.fixture
def sample_smiles():
    """Common test SMILES strings."""
    return {
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "ethanol": "CCO",
        "midazolam": "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2",
        "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    }


@pytest.fixture
def sample_adme_properties():
    """Known ADME properties for testing."""
    # Will be populated with real values in later branches
    return {}
