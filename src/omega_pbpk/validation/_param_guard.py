"""Parameter range validation gateway.

Validates ODE input parameters before WholeBodyPBPK setup to prevent
solver divergence caused by out-of-range ML/plugin predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

# Physical plausibility bounds for PBPK parameters
_BOUNDS: dict[str, tuple[float, float]] = {
    "clint_hepatic_L_per_h": (0.0, 500.0),  # max ~hepatic blood flow ceiling
    "clint_gut_L_per_h": (0.0, 200.0),
    "fup": (1e-4, 1.0),
    "rbp": (0.1, 20.0),
    "mw": (50.0, 2000.0),
    "logP": (-6.0, 12.0),
    "peff": (1e-6, 50.0),  # cm/s × 1e-4 scale typical range
    "ka_per_h": (1e-4, 100.0),
}

# Kp (tissue-to-plasma partition coefficient) bounds
_KP_MIN = 1e-4
_KP_MAX = 1000.0


@dataclass(frozen=True)
class ParamViolation:
    """A single out-of-range parameter finding."""

    param: str
    value: float
    lower: float
    upper: float

    def __str__(self) -> str:
        return f"{self.param}={self.value:.4g} outside [{self.lower:.4g}, {self.upper:.4g}]"


def check_drug_params(
    *,
    clint_hepatic_L_per_h: float | None = None,
    clint_gut_L_per_h: float | None = None,
    fup: float | None = None,
    rbp: float | None = None,
    mw: float | None = None,
    logP: float | None = None,
    peff: float | None = None,
    ka_per_h: float | None = None,
    kp: dict[str, float] | None = None,
    raise_on_violation: bool = True,
) -> list[ParamViolation]:
    """Validate drug parameters against physical plausibility bounds.

    Args:
        clint_hepatic_L_per_h: Hepatic intrinsic clearance (L/h).
        clint_gut_L_per_h: Gut wall intrinsic clearance (L/h).
        fup: Fraction unbound in plasma.
        rbp: Red blood cell-to-plasma ratio.
        mw: Molecular weight (Da).
        logP: Octanol-water partition coefficient.
        peff: Effective permeability.
        ka_per_h: Absorption rate constant (1/h).
        kp: Dict of tissue partition coefficients.
        raise_on_violation: If True, raise ValueError on first violation batch.

    Returns:
        List of ParamViolation for any out-of-range values.

    Raises:
        ValueError: If raise_on_violation=True and any violations found.
    """
    locals_map = {
        "clint_hepatic_L_per_h": clint_hepatic_L_per_h,
        "clint_gut_L_per_h": clint_gut_L_per_h,
        "fup": fup,
        "rbp": rbp,
        "mw": mw,
        "logP": logP,
        "peff": peff,
        "ka_per_h": ka_per_h,
    }

    violations: list[ParamViolation] = []

    for name, value in locals_map.items():
        if value is None:
            continue
        lo, hi = _BOUNDS[name]
        if not (lo <= value <= hi):
            violations.append(ParamViolation(param=name, value=value, lower=lo, upper=hi))

    if kp is not None:
        for tissue, kp_val in kp.items():
            if not (_KP_MIN <= kp_val <= _KP_MAX):
                violations.append(
                    ParamViolation(
                        param=f"kp[{tissue}]",
                        value=kp_val,
                        lower=_KP_MIN,
                        upper=_KP_MAX,
                    )
                )

    if violations and raise_on_violation:
        details = "; ".join(str(v) for v in violations)
        raise ValueError(
            f"Drug parameter(s) outside physical plausibility bounds: {details}. "
            "Check ML/plugin predictions for out-of-range outputs."
        )

    return violations


# ---------------------------------------------------------------------------
# validate_drug_params — legacy-compatible gateway (positional-arg style)
# ---------------------------------------------------------------------------

CLR_MAX_L_H = 200.0  # Approximate maximum renal blood flow
CLINT_MAX_L_H = 5000.0  # Realistic maximum CLint
KP_MIN = 1e-4
KP_MAX = 1000.0
KA_MIN = 0.0
KA_MAX = 100.0  # 1/h
FUP_MIN = 1e-4
FUP_MAX = 1.0
RBP_MIN = 0.1
RBP_MAX = 20.0


def validate_drug_params(
    clint_hepatic: float,
    clint_gut: float,
    clr: float,
    fup: float,
    rbp: float,
    kp: dict[str, float] | None = None,
    ka: float | None = None,
    drug_name: str = "unknown",
) -> None:
    """Validate drug parameters against physical plausibility bounds.

    Raises ValueError on the first detected violation so that ODE setup is
    aborted before the solver can diverge.

    Args:
        clint_hepatic: Hepatic intrinsic clearance (L/h). Must be in [0, CLINT_MAX_L_H].
        clint_gut: Gut-wall intrinsic clearance (L/h). Must be >= 0.
        clr: Renal clearance (L/h). Must be in [0, CLR_MAX_L_H].
        fup: Fraction unbound in plasma. Must be in [FUP_MIN, FUP_MAX].
        rbp: Red-blood-cell-to-plasma concentration ratio. Must be in [RBP_MIN, RBP_MAX].
        kp: Optional tissue partition coefficients. Each value must be in [KP_MIN, KP_MAX].
        ka: Optional absorption rate constant (1/h). Must be in [KA_MIN, KA_MAX].
        drug_name: Name used in error messages.

    Raises:
        ValueError: If any parameter is outside its physical plausibility bounds.
    """
    prefix = f"Drug '{drug_name}'"

    if clint_hepatic < 0:
        raise ValueError(f"{prefix}: clint_hepatic must be >= 0, got {clint_hepatic}")
    if clint_hepatic > CLINT_MAX_L_H:
        raise ValueError(
            f"{prefix}: clint_hepatic {clint_hepatic} > physical max {CLINT_MAX_L_H} L/h"
        )
    if clint_gut < 0:
        raise ValueError(f"{prefix}: clint_gut must be >= 0, got {clint_gut}")
    if clr < 0:
        raise ValueError(f"{prefix}: clr must be >= 0, got {clr}")
    if clr > CLR_MAX_L_H:
        raise ValueError(f"{prefix}: clr {clr} > renal blood flow max {CLR_MAX_L_H} L/h")
    if not (FUP_MIN <= fup <= FUP_MAX):
        raise ValueError(f"{prefix}: fup must be in [{FUP_MIN}, {FUP_MAX}], got {fup}")
    if not (RBP_MIN <= rbp <= RBP_MAX):
        raise ValueError(f"{prefix}: rbp must be in [{RBP_MIN}, {RBP_MAX}], got {rbp}")
    if kp is not None:
        for organ, val in kp.items():
            if val < KP_MIN:
                raise ValueError(f"{prefix}: kp[{organ}] must be >= {KP_MIN}, got {val}")
            if val > KP_MAX:
                raise ValueError(f"{prefix}: kp[{organ}] must be <= {KP_MAX}, got {val}")
    if ka is not None:
        if ka < KA_MIN:
            raise ValueError(f"{prefix}: ka must be >= 0, got {ka}")
        if ka > KA_MAX:
            raise ValueError(f"{prefix}: ka must be <= {KA_MAX} 1/h, got {ka}")


__all__ = ["ParamViolation", "check_drug_params", "validate_drug_params"]
