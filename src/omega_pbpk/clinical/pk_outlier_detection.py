"""Phase 584 - Pharmacokinetic Outlier Detection."""

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PKOutlierDetectionResult:
    n_subjects: int
    n_outliers: int
    outlier_indices: tuple
    mean_auc: float
    sd_auc: float
    cv_auc_pct: float
    threshold_lower: float
    threshold_upper: float
    auc_values: tuple
    outlier_aucs: tuple
    method: str
    notes: str


def detect_pk_outliers(
    auc_values_list: list,
    method: str = "2SD",
) -> PKOutlierDetectionResult:
    """Detect outlier PK profiles using statistical methods."""
    if method not in ("2SD", "IQR"):
        raise ValueError(f"method must be '2SD' or 'IQR', got '{method}'")
    n = len(auc_values_list)
    if n < 3:
        raise ValueError("Need at least 3 data points")

    aucs = list(auc_values_list)
    mean_auc = sum(aucs) / n
    var = sum((x - mean_auc) ** 2 for x in aucs) / (n - 1)
    sd_auc = math.sqrt(var)
    cv_pct = (sd_auc / mean_auc * 100.0) if mean_auc != 0 else 0.0

    if method == "2SD":
        lower = mean_auc - 2.0 * sd_auc
        upper = mean_auc + 2.0 * sd_auc
    else:  # IQR
        sorted_aucs = sorted(aucs)
        q25 = sorted_aucs[n // 4]
        q75 = sorted_aucs[3 * n // 4]
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr

    outlier_indices = []
    outlier_aucs = []
    for i, val in enumerate(aucs):
        if val < lower or val > upper:
            outlier_indices.append(i)
            outlier_aucs.append(val)

    notes_parts = []
    if len(outlier_indices) > 0:
        notes_parts.append(f"{len(outlier_indices)} outlier(s) detected")
    else:
        notes_parts.append("No outliers detected")
    notes_parts.append(f"CV={cv_pct:.1f}%")

    return PKOutlierDetectionResult(
        n_subjects=n,
        n_outliers=len(outlier_indices),
        outlier_indices=tuple(outlier_indices),
        mean_auc=mean_auc,
        sd_auc=sd_auc,
        cv_auc_pct=cv_pct,
        threshold_lower=lower,
        threshold_upper=upper,
        auc_values=tuple(aucs),
        outlier_aucs=tuple(outlier_aucs),
        method=method,
        notes="; ".join(notes_parts),
    )


def simulate_population_and_detect(
    drug_name: str,
    typical_cl: float,
    typical_vd: float,
    omega_cl: float = 0.3,
    omega_vd: float = 0.2,
    n_subjects: int = 50,
    seed: int = 42,
    method: str = "2SD",
) -> PKOutlierDetectionResult:
    """Simulate log-normal CL/Vd population and detect PK outliers."""
    rng = random.Random(seed)
    dose = 100.0  # mg

    auc_values = []
    for _ in range(n_subjects):
        # Log-normal: CL_i = typical_cl * exp(eta), eta ~ N(0, omega^2)
        eta_cl = rng.gauss(0, omega_cl)
        cl_i = typical_cl * math.exp(eta_cl)
        auc_i = dose / cl_i
        auc_values.append(auc_i)

    return detect_pk_outliers(auc_values, method=method)
