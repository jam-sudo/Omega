"""Visual Predictive Check (VPC) plot for population PK."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def plot_vpc(
    time_h: NDArray,
    cp_matrix: NDArray,
    observed_time: NDArray | None = None,
    observed_cp: NDArray | None = None,
    title: str = "Visual Predictive Check",
    output_path: str | None = None,
    percentiles: tuple[float, float, float] = (5.0, 50.0, 95.0),
) -> None:
    """Generate a VPC plot (population prediction bands + observed data).

    Args:
        time_h: Time array (h), shape (n_timepoints,)
        cp_matrix: Simulated concentrations, shape (n_subjects, n_timepoints)
        observed_time: Observed time points (optional)
        observed_cp: Observed concentrations (optional)
        title: Plot title
        output_path: Save path (.png or .pdf). If None, show interactively.
        percentiles: Lower, median, upper percentile bands.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib required for VPC plots: pip install matplotlib") from err

    p_lo, p_med, p_hi = percentiles
    lo = np.percentile(cp_matrix, p_lo, axis=0)
    med = np.percentile(cp_matrix, p_med, axis=0)
    hi = np.percentile(cp_matrix, p_hi, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Shaded prediction interval
    ax.fill_between(
        time_h, lo, hi, alpha=0.25, color="steelblue", label=f"{p_lo:.0f}–{p_hi:.0f}th percentile"
    )
    # Median line
    ax.plot(time_h, med, color="steelblue", linewidth=2, label=f"Median ({p_med:.0f}th)")
    # Boundary lines
    ax.plot(time_h, lo, color="steelblue", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.plot(time_h, hi, color="steelblue", linewidth=0.8, linestyle="--", alpha=0.6)

    # Observed data
    if observed_time is not None and observed_cp is not None:
        ax.scatter(
            observed_time, observed_cp, color="black", zorder=5, s=30, label="Observed", alpha=0.8
        )

    ax.set_xlabel("Time (h)", fontsize=12)
    ax.set_ylabel("Plasma concentration (mg/L)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_forest(
    parameter: str,
    subgroup_labels: list[str],
    medians: list[float],
    p5_values: list[float],
    p95_values: list[float],
    reference_value: float | None = None,
    title: str = "Forest Plot",
    output_path: str | None = None,
) -> None:
    """Generate a forest plot for population PK subgroup comparison.

    Args:
        parameter: PK parameter name (e.g., "AUC", "Cmax")
        subgroup_labels: List of subgroup names
        medians: Median values per subgroup
        p5_values: 5th percentile per subgroup
        p95_values: 95th percentile per subgroup
        reference_value: Reference value (drawn as vertical line)
        title: Plot title
        output_path: Save path. If None, show interactively.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib required: pip install matplotlib") from err

    n = len(subgroup_labels)
    y_pos = list(range(n))

    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.6 + 1)))
    for i, (_label, med, lo, hi) in enumerate(
        zip(subgroup_labels, medians, p5_values, p95_values, strict=False)
    ):
        ax.plot([lo, hi], [i, i], color="steelblue", linewidth=2)
        ax.scatter([med], [i], color="steelblue", s=60, zorder=5)

    if reference_value is not None:
        ax.axvline(reference_value, color="gray", linestyle="--", alpha=0.6, label="Reference")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(subgroup_labels, fontsize=10)
    ax.set_xlabel(parameter, fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
