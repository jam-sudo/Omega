"""Forest plot data for meta-analysis: effect sizes, confidence intervals, heterogeneity."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StudyEffect:
    study_id: str
    effect_size: float
    se: float
    weight: float
    ci_low: float
    ci_high: float
    n_subjects: int
    notes: str


@dataclass(frozen=True)
class ForestPlotResult:
    studies: tuple
    n_studies: int
    pooled_effect: float
    pooled_se: float
    pooled_ci_low: float
    pooled_ci_high: float
    q_statistic: float
    i_squared_pct: float
    tau_squared: float
    re_pooled_effect: float
    re_ci_low: float
    re_ci_high: float
    heterogeneity_p: float
    recommendation: str
    notes: str


def create_study_effect(
    study_id: str,
    effect_size: float,
    se: float,
    n_subjects: int,
) -> StudyEffect:
    """Create a StudyEffect with CI and weight computed."""
    if se <= 0:
        raise ValueError("se must be positive")
    weight = 1.0 / (se * se)
    ci_low = effect_size - 1.96 * se
    ci_high = effect_size + 1.96 * se
    notes = f"Study {study_id}: effect={effect_size:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]"
    return StudyEffect(
        study_id=study_id,
        effect_size=effect_size,
        se=se,
        weight=weight,
        ci_low=ci_low,
        ci_high=ci_high,
        n_subjects=n_subjects,
        notes=notes,
    )


def compute_forest_plot(studies: list) -> ForestPlotResult:
    """
    Compute fixed-effect and random-effects pooled estimates.
    Requires >= 2 studies.
    """
    if len(studies) < 2:
        raise ValueError("At least 2 studies are required for meta-analysis")

    n_studies = len(studies)

    # Fixed-effect pooled estimate
    weights = [s.weight for s in studies]
    effects = [s.effect_size for s in studies]
    ses = [s.se for s in studies]

    sum_w = sum(weights)
    pooled_effect_fe = sum(w * es for w, es in zip(weights, effects)) / sum_w
    pooled_se_fe = math.sqrt(1.0 / sum_w)

    # Cochran's Q
    q_statistic = sum(w * (es - pooled_effect_fe) ** 2 for w, es in zip(weights, effects))
    df = n_studies - 1

    # I-squared
    if q_statistic > 0:
        i_squared = max(0.0, (q_statistic - df) / q_statistic * 100.0)
    else:
        i_squared = 0.0

    # DerSimonian-Laird tau-squared
    sum_w2 = sum(w * w for w in weights)
    c_factor = sum_w - sum_w2 / sum_w
    if c_factor > 0 and q_statistic > df:
        tau_squared = (q_statistic - df) / c_factor
    else:
        tau_squared = 0.0

    # Random effects weights and pooled estimate
    re_weights = [1.0 / (se * se + tau_squared) for se in ses]
    sum_re_w = sum(re_weights)
    re_pooled = sum(rw * es for rw, es in zip(re_weights, effects)) / sum_re_w
    re_se = math.sqrt(1.0 / sum_re_w)
    re_ci_low = re_pooled - 1.96 * re_se
    re_ci_high = re_pooled + 1.96 * re_se

    # Approximate heterogeneity p-value from Q and df
    if q_statistic < df:
        heterogeneity_p = 0.5
    elif q_statistic > 2 * df:
        heterogeneity_p = 0.01
    else:
        heterogeneity_p = 0.1

    # Recommendation based on I-squared
    if i_squared < 25:
        recommendation = "low_heterogeneity_fixed_effect_preferred"
    elif i_squared < 75:
        recommendation = "moderate_heterogeneity_random_effects_preferred"
    else:
        recommendation = "high_heterogeneity_interpret_cautiously"

    pooled_ci_low = pooled_effect_fe - 1.96 * pooled_se_fe
    pooled_ci_high = pooled_effect_fe + 1.96 * pooled_se_fe

    notes = (
        f"Meta-analysis of {n_studies} studies. "
        f"Q={q_statistic:.3f} (df={df}), I²={i_squared:.1f}%, τ²={tau_squared:.4f}. "
        f"FE pooled={pooled_effect_fe:.4f}, RE pooled={re_pooled:.4f}."
    )

    return ForestPlotResult(
        studies=tuple(studies),
        n_studies=n_studies,
        pooled_effect=pooled_effect_fe,
        pooled_se=pooled_se_fe,
        pooled_ci_low=pooled_ci_low,
        pooled_ci_high=pooled_ci_high,
        q_statistic=q_statistic,
        i_squared_pct=i_squared,
        tau_squared=tau_squared,
        re_pooled_effect=re_pooled,
        re_ci_low=re_ci_low,
        re_ci_high=re_ci_high,
        heterogeneity_p=heterogeneity_p,
        recommendation=recommendation,
        notes=notes,
    )


def auc_ratio_to_log_effect(auc_ratio: float, n: int, cv_pct: float = 20.0) -> StudyEffect:
    """
    Convert AUC ratio from a study to a log-scale StudyEffect.

    SE estimated from n and CV using: se = cv/sqrt(n) (log-scale approximation).
    """
    if auc_ratio <= 0:
        raise ValueError("auc_ratio must be positive")
    if n <= 0:
        raise ValueError("n must be positive")

    log_effect = math.log(auc_ratio)
    se = (cv_pct / 100.0) / math.sqrt(n)
    weight = 1.0 / (se * se)
    ci_low = log_effect - 1.96 * se
    ci_high = log_effect + 1.96 * se
    notes = (
        f"AUC ratio={auc_ratio:.3f} → log-effect={log_effect:.4f}, "
        f"SE={se:.4f} (CV={cv_pct}%, n={n})"
    )

    return StudyEffect(
        study_id=f"AUC_ratio_{auc_ratio:.2f}",
        effect_size=log_effect,
        se=se,
        weight=weight,
        ci_low=ci_low,
        ci_high=ci_high,
        n_subjects=n,
        notes=notes,
    )
