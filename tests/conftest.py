import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow tests requiring training")
    config.addinivalue_line("markers", "benchmark: benchmark tests")
    config.addinivalue_line("markers", "regression: regression snapshot tests")
    config.addinivalue_line("markers", "integration: integration tests")
