"""[DEPRECATED] Use omega_pbpk.api.app directly.

This module is kept for backward compatibility.
"""
import warnings

warnings.warn(
    "omega_pbpk.api.server is deprecated. Use omega_pbpk.api.app instead.",
    DeprecationWarning,
    stacklevel=2,
)

from omega_pbpk.api.app import app  # noqa: E402  re-export


def create_app():
    warnings.warn(
        "create_app() is deprecated. Use omega_pbpk.api.app directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return app
