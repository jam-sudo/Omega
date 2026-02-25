"""ICRP reference man physiology and Monte Carlo virtual population generator.

Reference: ICRP Publication 89 (2002) — Basic Anatomical and Physiological
Data for Use in Radiological Protection: Reference Values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OrganPhysiology:
    """Physiological parameters for a single organ."""

    volume_L: float
    blood_flow_fraction: float  # Fraction of cardiac output
    cv_volume: float = 0.15  # Inter-individual variability CV
    cv_flow: float = 0.20


# ICRP Reference Man — 70 kg adult male
ICRP_ORGANS: dict[str, OrganPhysiology] = {
    "lung": OrganPhysiology(0.50, 1.0, cv_volume=0.10),
    "brain": OrganPhysiology(1.45, 0.12, cv_volume=0.05, cv_flow=0.10),
    "heart": OrganPhysiology(0.33, 0.04),
    "kidney": OrganPhysiology(0.31, 0.19, cv_flow=0.15),
    "liver": OrganPhysiology(1.80, 0.065, cv_volume=0.20, cv_flow=0.20),
    "spleen": OrganPhysiology(0.15, 0.03),
    "gut_wall": OrganPhysiology(1.03, 0.15, cv_flow=0.20),
    "pancreas": OrganPhysiology(0.10, 0.01),
    "thymus": OrganPhysiology(0.02, 0.002),
    "reproductive": OrganPhysiology(0.04, 0.002),
    "rest": OrganPhysiology(2.50, 0.038),
    "adipose": OrganPhysiology(14.5, 0.052, cv_volume=0.40, cv_flow=0.30),
    "muscle": OrganPhysiology(28.0, 0.17, cv_volume=0.15, cv_flow=0.20),
    "bone": OrganPhysiology(4.86, 0.05),
    "skin": OrganPhysiology(3.30, 0.05),
}

REFERENCE_BW_KG = 70.0
REFERENCE_CO_L_H = 390.0  # Cardiac output (L/h)
REFERENCE_GFR_L_H = 7.5  # GFR (~125 mL/min)
REFERENCE_MPPGL = 40.0  # pmol CYP/mg microsomal protein


@dataclass(frozen=True)
class ReferenceMan:
    """ICRP Reference Man (70 kg adult male) physiological parameters."""

    body_weight_kg: float = 70.0
    cardiac_output_L_h: float = 390.0
    gfr_L_h: float = 7.5
    hematocrit: float = 0.45
    plasma_volume_L: float = 3.0
    blood_volume_L: float = 5.2
    liver_weight_g: float = 1800.0
    mppgl: float = 40.0  # pmol CYP/mg microsomal protein
    organs: dict[str, OrganPhysiology] = field(default_factory=lambda: dict(ICRP_ORGANS))


@dataclass
class VirtualSubject:
    """A single virtual subject with sampled physiology."""

    body_weight_kg: float
    cardiac_output_L_h: float
    gfr_L_h: float
    liver_weight_g: float
    mppgl: float
    organ_volumes: dict[str, float]
    organ_flows: dict[str, float]


class VirtualPopulation:
    """Monte Carlo virtual population generator.

    Generates N virtual subjects by sampling physiological parameters
    from log-normal distributions around ICRP reference values.
    """

    def __init__(self, n: int = 50, seed: int = 42) -> None:
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.ref = ReferenceMan()

    def _sample_lognormal(self, mean: float, cv: float) -> float:
        """Sample from log-normal distribution given mean and CV."""
        if cv <= 0:
            return mean
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(mean) - 0.5 * sigma**2
        return float(self.rng.lognormal(mu, sigma))

    def generate(self) -> list[VirtualSubject]:
        """Generate N virtual subjects with correlated physiology."""
        subjects: list[VirtualSubject] = []

        for _ in range(self.n):
            bw = self._sample_lognormal(70.0, 0.15)
            scale = bw / 70.0

            # Cardiac output scales with BW^0.75 (allometric)
            co = 390.0 * scale**0.75
            co = self._sample_lognormal(co, 0.10)

            gfr = 7.5 * scale**0.75
            gfr = self._sample_lognormal(gfr, 0.15)

            liver_wt = 1800.0 * scale
            liver_wt = self._sample_lognormal(liver_wt, 0.15)

            mppgl = self._sample_lognormal(40.0, 0.30)

            volumes: dict[str, float] = {}
            flows: dict[str, float] = {}
            total_flow_frac = 0.0

            for name, phys in ICRP_ORGANS.items():
                v = phys.volume_L * scale
                v = self._sample_lognormal(v, phys.cv_volume)
                volumes[name] = v

                f = phys.blood_flow_fraction
                f = self._sample_lognormal(f, phys.cv_flow)
                flows[name] = f
                if name != "lung":
                    total_flow_frac += f

            # Renormalize flows to sum to CO (exclude lung = 100%)
            for name in flows:
                if name != "lung":
                    flows[name] = co * flows[name] / max(total_flow_frac, 1e-12)
                else:
                    flows[name] = co

            subjects.append(
                VirtualSubject(
                    body_weight_kg=bw,
                    cardiac_output_L_h=co,
                    gfr_L_h=gfr,
                    liver_weight_g=liver_wt,
                    mppgl=mppgl,
                    organ_volumes=volumes,
                    organ_flows=flows,
                )
            )

        return subjects

    def summary_stats(self, subjects: list[VirtualSubject]) -> dict[str, Any]:
        """Compute population summary statistics."""
        bws = np.array([s.body_weight_kg for s in subjects])
        cos = np.array([s.cardiac_output_L_h for s in subjects])
        return {
            "n": len(subjects),
            "body_weight_kg": {
                "mean": float(np.mean(bws)),
                "std": float(np.std(bws)),
                "cv": float(np.std(bws) / np.mean(bws)),
            },
            "cardiac_output_L_h": {
                "mean": float(np.mean(cos)),
                "std": float(np.std(cos)),
                "cv": float(np.std(cos) / np.mean(cos)),
            },
        }
