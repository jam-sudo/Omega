"""Publication-quality PK plots.

Generates linear and semilog concentration-time curves with
optional therapeutic window overlay.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.info("matplotlib not available; plotting disabled.")


class PKPlotter:
    """Publication-quality PK plot generator."""

    def plot_pk(
        self,
        time_h: NDArray[np.floating[Any]],
        cp_mg_L: NDArray[np.floating[Any]],
        title: str = "Plasma Concentration-Time Profile",
        mec: float | None = None,
        mtc: float | None = None,
        save_path: str | Path | None = None,
        observed_data: dict[str, NDArray[np.floating[Any]]] | None = None,
    ) -> Any | None:
        """Generate dual-panel (linear + semilog) PK plot.

        Args:
            time_h: Time array (h).
            cp_mg_L: Plasma concentration array (mg/L).
            title: Plot title.
            mec: Minimum effective concentration (mg/L).
            mtc: Maximum tolerated concentration (mg/L).
            save_path: File path to save (PNG/PDF).
            observed_data: Optional dict with "time_h" and "cp_mg_L" arrays.

        Returns:
            Figure object if matplotlib available, None otherwise.
        """
        if not HAS_MPL:
            logger.warning("Cannot plot: matplotlib not installed.")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Linear scale
        ax1.plot(time_h, cp_mg_L, "b-", linewidth=2, label="Predicted")
        if observed_data is not None:
            ax1.plot(
                observed_data["time_h"],
                observed_data["cp_mg_L"],
                "ro",
                markersize=6,
                label="Observed",
            )
        if mec is not None:
            ax1.axhline(mec, color="green", linestyle="--", alpha=0.7, label=f"MEC ({mec})")
        if mtc is not None:
            ax1.axhline(mtc, color="red", linestyle="--", alpha=0.7, label=f"MTC ({mtc})")
        ax1.set_xlabel("Time (h)", fontsize=12)
        ax1.set_ylabel("Plasma Concentration (mg/L)", fontsize=12)
        ax1.set_title(f"{title} — Linear", fontsize=13)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, time_h[-1])

        # Semilog scale
        cp_pos = np.maximum(cp_mg_L, 1e-12)
        ax2.semilogy(time_h, cp_pos, "b-", linewidth=2, label="Predicted")
        if observed_data is not None:
            obs_pos = np.maximum(observed_data["cp_mg_L"], 1e-12)
            ax2.semilogy(observed_data["time_h"], obs_pos, "ro", markersize=6, label="Observed")
        if mec is not None:
            ax2.axhline(mec, color="green", linestyle="--", alpha=0.7)
        if mtc is not None:
            ax2.axhline(mtc, color="red", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Time (h)", fontsize=12)
        ax2.set_ylabel("Plasma Concentration (mg/L)", fontsize=12)
        ax2.set_title(f"{title} — Semilog", fontsize=13)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, which="both")
        ax2.set_xlim(0, time_h[-1])

        plt.tight_layout()

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            logger.info("Plot saved to %s", path)
            plt.close(fig)

        return fig

    def plot_population(
        self,
        time_h: NDArray[np.floating[Any]],
        cp_matrix: NDArray[np.floating[Any]],
        title: str = "Population PK",
        save_path: str | Path | None = None,
    ) -> Any | None:
        """Plot population PK with median and percentile bands.

        Args:
            time_h: Time array (h).
            cp_matrix: (N_subjects × N_time) concentration matrix.
            title: Plot title.
            save_path: File path to save.
        """
        if not HAS_MPL:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))

        median = np.median(cp_matrix, axis=0)
        p5 = np.percentile(cp_matrix, 5, axis=0)
        p95 = np.percentile(cp_matrix, 95, axis=0)
        p25 = np.percentile(cp_matrix, 25, axis=0)
        p75 = np.percentile(cp_matrix, 75, axis=0)

        ax.fill_between(time_h, p5, p95, alpha=0.15, color="blue", label="5th-95th %ile")
        ax.fill_between(time_h, p25, p75, alpha=0.3, color="blue", label="25th-75th %ile")
        ax.plot(time_h, median, "b-", linewidth=2, label="Median")

        ax.set_xlabel("Time (h)", fontsize=12)
        ax.set_ylabel("Plasma Concentration (mg/L)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig

    def plot_multidose(
        self,
        time_h: NDArray[np.floating[Any]],
        cp_mg_L: NDArray[np.floating[Any]],
        dose_times: list[float],
        title: str = "Multi-Dose PK Profile",
        save_path: str | Path | None = None,
    ) -> Any | None:
        """Plot multi-dose PK profile with dose markers."""
        if not HAS_MPL:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_h, cp_mg_L, "b-", linewidth=1.5)

        for dt in dose_times:
            ax.axvline(dt, color="gray", linestyle=":", alpha=0.4)

        ax.set_xlabel("Time (h)", fontsize=12)
        ax.set_ylabel("Plasma Concentration (mg/L)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig
