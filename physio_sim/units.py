from __future__ import annotations


def assert_nonnegative(name: str, value: float) -> None:
    if value < 0:
        msg = f"{name} must be >= 0, got {value}"
        raise ValueError(msg)


def assert_fraction(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        msg = f"{name} must be in [0, 1], got {value}"
        raise ValueError(msg)
