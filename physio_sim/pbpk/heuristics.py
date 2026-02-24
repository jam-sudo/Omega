from __future__ import annotations


def heuristic_kp(logp: float, pka: float | None, tissue_name: str) -> float:
    """Heuristic-only Kp estimator (not validated for clinical use)."""
    base = 1.0 + 0.3 * logp
    if pka is not None:
        base += 0.05 * (pka - 7.0)
    tissue_adjust = {
        "fat": 1.6,
        "brain": 1.2,
        "muscle": 1.0,
        "rest": 0.9,
        "kidney": 1.0,
        "liver": 1.1,
        "gut_wall": 1.0,
        "portal_vein": 1.0,
    }.get(tissue_name, 1.0)
    return max(0.2, base * tissue_adjust)
