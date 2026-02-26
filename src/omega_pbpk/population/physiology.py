"""ICRP reference man physiology and Monte Carlo virtual population generator.

Reference: ICRP Publication 89 (2002) — Basic Anatomical and Physiological
Data for Use in Radiological Protection: Reference Values.

Multi-species allometric scaling references:
  - Davies B & Morris T, Pharm Res, 1993 (allometric relationships)
  - Brown RP et al., Toxicol Sci, 1997 (physiological parameters for rats, mice, dogs)
  - ICRP Publication 89, 2002 (human reference values)
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


# ---------------------------------------------------------------------------
# Multi-species physiological parameters (allometrically scaled)
# ---------------------------------------------------------------------------
# Reference body weights: human=70 kg, rat=0.25 kg, mouse=0.02 kg, dog=10 kg
#
# Scaling rules (Davies & Morris 1993, Brown et al. 1997):
#   Organ volumes (except brain): V = V_human * (BW / 70)^1.0
#   Brain volume:                 V = V_human * (BW / 70)^0.7
#   Blood flows:                  Q = Q_human * (BW / 70)^0.75
#   Cardiac output:               CO = 15 * BW^0.75  (L/h)
#
# Blood flow fractions (fraction of cardiac output) are conserved across
# species; absolute flows scale with total cardiac output.
# ---------------------------------------------------------------------------


def allometric_scale(
    human_value: float,
    human_bw: float = 70.0,
    target_bw: float = 0.25,
    exponent: float = 0.75,
) -> float:
    """Scale a physiological parameter allometrically from human to target species.

    Args:
        human_value: Value in the human reference (70 kg by default).
        human_bw: Human body weight (kg), default 70.0.
        target_bw: Target species body weight (kg).
        exponent: Allometric exponent (0.75 for flows/CO, 1.0 for volumes,
                  0.7 for brain volume).

    Returns:
        Scaled value for target species.

    Example:
        Rat cardiac output from 390 L/h human reference:
        allometric_scale(390.0, 70.0, 0.25, 0.75) -> approximately 17.5 L/h
    """
    if human_bw <= 0 or target_bw <= 0:
        raise ValueError("Body weights must be positive.")
    return human_value * (target_bw / human_bw) ** exponent


def _build_species_physiology(bw_kg: float) -> dict[str, float]:
    """Build a physiology dict for a species with given body weight.

    Uses allometric scaling from human reference (70 kg ICRP values).
    Organ volumes scale as BW^1.0 (brain as BW^0.7).
    Blood flows scale as BW^0.75.
    Cardiac output = 15 * BW^0.75 (L/h).

    Args:
        bw_kg: Body weight in kg.

    Returns:
        Dict of physiological parameters matching human physiology keys.
    """
    co = allometric_scale(REFERENCE_CO_L_H, 70.0, bw_kg, 0.75)
    gfr = allometric_scale(REFERENCE_GFR_L_H, 70.0, bw_kg, 0.75)

    physiology: dict[str, float] = {
        "body_weight_kg": bw_kg,
        "cardiac_output_L_h": round(co, 6),
        "gfr_L_h": round(gfr, 6),
    }

    for organ_name, organ_phys in ICRP_ORGANS.items():
        # Brain uses BW^0.7 exponent; all other organs BW^1.0
        vol_exp = 0.7 if organ_name == "brain" else 1.0
        scaled_vol = allometric_scale(organ_phys.volume_L, 70.0, bw_kg, vol_exp)
        physiology[f"{organ_name}_volume_L"] = round(scaled_vol, 8)

        # Flow = fraction of cardiac output (fraction conserved across species)
        scaled_flow = organ_phys.blood_flow_fraction * co
        physiology[f"{organ_name}_flow_L_h"] = round(scaled_flow, 8)

    return physiology


# Reference body weights per species (kg)
_SPECIES_BW: dict[str, float] = {
    "human": 70.0,
    "rat": 0.25,
    "mouse": 0.02,
    "dog": 10.0,
}

# Pre-computed species physiology dicts
SPECIES_PHYSIOLOGY: dict[str, dict[str, float]] = {
    species: _build_species_physiology(bw)
    for species, bw in _SPECIES_BW.items()
}


def get_species_physiology(species: str = "human") -> dict[str, float]:
    """Return physiological parameter dict for a given species.

    Args:
        species: Species name, case-insensitive. Supported: "human", "rat",
                 "mouse", "dog".

    Returns:
        Dict of physiological parameters (volumes in L, flows in L/h).

    Raises:
        ValueError: If the species is not supported.

    Example:
        phys = get_species_physiology("rat")
        phys["body_weight_kg"]  # 0.25
    """
    key = species.lower().strip()
    if key not in SPECIES_PHYSIOLOGY:
        supported = ", ".join(sorted(SPECIES_PHYSIOLOGY.keys()))
        raise ValueError(
            f"Unknown species '{species}'. Supported species: {supported}."
        )
    return SPECIES_PHYSIOLOGY[key]


# ---------------------------------------------------------------------------
# Virtual population with detailed covariates (Task 4)
# ---------------------------------------------------------------------------

# Child-Pugh hepatic clearance scaling factors
_CHILD_PUGH_FACTORS: dict[str, float] = {
    "normal": 1.0,
    "A": 0.75,
    "B": 0.50,
    "C": 0.25,
}

# Child-Pugh prevalence probabilities [normal, A, B, C]
_CHILD_PUGH_PROBS = [0.85, 0.10, 0.04, 0.01]
_CHILD_PUGH_GRADES = ["normal", "A", "B", "C"]


@dataclass
class SubjectCovariates:
    """Covariates for a single virtual subject.

    Attributes:
        body_weight_kg: Body weight (kg).
        age_years: Age (years).
        sex: Biological sex ("M" or "F").
        height_cm: Height (cm).
        bmi: Body mass index (kg/m^2).
        gfr_mL_min: Glomerular filtration rate (mL/min).
        child_pugh: Hepatic function class ("normal", "A", "B", or "C").
        cyp3a4_activity: CYP3A4 activity relative to typical (1.0).
        cyp2d6_activity: CYP2D6 activity relative to typical (1.0).
    """

    body_weight_kg: float
    age_years: float
    sex: str  # "M" or "F"
    height_cm: float
    bmi: float
    gfr_mL_min: float
    child_pugh: str  # "normal", "A", "B", "C"
    cyp3a4_activity: float
    cyp2d6_activity: float


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

    Supports two modes:
      - generate(): legacy method returning VirtualSubject objects with
        organ volumes and flows (for backward compatibility).
      - sample(): new method returning SubjectCovariates objects with
        rich clinical covariates for PopPK/PBPK analyses.
    """

    def __init__(self, n: int = 50, seed: int = 42) -> None:
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.ref = ReferenceMan()

    def _sample_lognormal(
        self,
        mean: float,
        cv: float,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Sample from log-normal distribution given mean and CV."""
        _rng = rng if rng is not None else self.rng
        if cv <= 0:
            return mean
        sigma = float(np.sqrt(np.log(1 + cv**2)))
        mu = float(np.log(mean) - 0.5 * sigma**2)
        return float(_rng.lognormal(mu, sigma))

    def sample(self, n: int, seed: int | None = None) -> list[SubjectCovariates]:
        """Sample N virtual subjects using log-normal/normal distributions.

        Distributions (approximate NHANES-based):
          - weight: log-normal, mu=70 kg, CV=20%
          - age: uniform 18-75
          - sex: 50/50 M/F
          - GFR: log-normal, mu=100 mL/min, CV=25%
          - Child-Pugh: 85% normal, 10% A, 4% B, 1% C
          - CYP3A4 activity: log-normal, mu=1.0, CV=40%
          - CYP2D6 activity: log-normal, mu=1.0, CV=50%

        Args:
            n: Number of subjects to generate.
            seed: Optional random seed for reproducibility.

        Returns:
            List of SubjectCovariates, one per virtual subject.
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        subjects: list[SubjectCovariates] = []

        for _ in range(n):
            # Body weight: log-normal, mu=70 kg, CV=20%
            bw = self._sample_lognormal(70.0, 0.20, rng)

            # Age: uniform 18-75
            age = float(rng.uniform(18.0, 75.0))

            # Sex: 50/50
            sex = "M" if rng.random() < 0.5 else "F"

            # Height correlated with sex; CV approximately 4%
            mean_height = 175.0 if sex == "M" else 162.0
            height = self._sample_lognormal(mean_height, 0.04, rng)

            bmi = bw / (height / 100.0) ** 2

            # GFR: log-normal, mu=100 mL/min, CV=25%
            gfr = self._sample_lognormal(100.0, 0.25, rng)

            # Child-Pugh: categorical
            cp_idx = int(rng.choice(len(_CHILD_PUGH_GRADES), p=_CHILD_PUGH_PROBS))
            child_pugh = _CHILD_PUGH_GRADES[cp_idx]

            # CYP activities: log-normal
            cyp3a4 = self._sample_lognormal(1.0, 0.40, rng)
            cyp2d6 = self._sample_lognormal(1.0, 0.50, rng)

            subjects.append(
                SubjectCovariates(
                    body_weight_kg=round(bw, 2),
                    age_years=round(age, 1),
                    sex=sex,
                    height_cm=round(height, 1),
                    bmi=round(bmi, 2),
                    gfr_mL_min=round(gfr, 2),
                    child_pugh=child_pugh,
                    cyp3a4_activity=round(cyp3a4, 4),
                    cyp2d6_activity=round(cyp2d6, 4),
                )
            )

        return subjects

    def scale_physiology(
        self,
        covariates: SubjectCovariates,
        base_physiology: dict[str, float],
    ) -> dict[str, float]:
        """Scale physiological parameters for a specific subject.

        Scaling rules:
          - Organ volumes scaled by (BW / ref_BW)^1.0
          - Blood flows scaled by (BW / ref_BW)^0.75
          - GFR-dependent clearance scaled by GFR / 100 mL/min
          - Hepatic clearance scaled by Child-Pugh factor
            (normal=1.0, A=0.75, B=0.5, C=0.25)

        Args:
            covariates: Subject covariates from :meth:`sample`.
            base_physiology: Reference physiology dict (e.g. from
                :func:`get_species_physiology`).

        Returns:
            Scaled physiology dict for the individual subject, with additional
            keys: cyp3a4_activity, cyp2d6_activity, hepatic_cl_factor.
        """
        bw = covariates.body_weight_kg
        ref_bw = base_physiology.get("body_weight_kg", 70.0)
        vol_scale = bw / ref_bw
        flow_scale = (bw / ref_bw) ** 0.75
        gfr_scale = covariates.gfr_mL_min / 100.0
        cp_factor = _CHILD_PUGH_FACTORS.get(covariates.child_pugh, 1.0)

        scaled: dict[str, float] = {}
        for key, value in base_physiology.items():
            if key == "body_weight_kg":
                scaled[key] = bw
            elif key == "cardiac_output_L_h":
                scaled[key] = value * flow_scale
            elif key == "gfr_L_h":
                # Subject GFR / reference 100 mL/min (= 6 L/h)
                scaled[key] = value * gfr_scale
            elif key.endswith("_volume_L"):
                scaled[key] = value * vol_scale
            elif key.endswith("_flow_L_h"):
                scaled[key] = value * flow_scale
            else:
                scaled[key] = value

        # Attach subject-specific scaling metadata
        scaled["cyp3a4_activity"] = covariates.cyp3a4_activity
        scaled["cyp2d6_activity"] = covariates.cyp2d6_activity
        scaled["hepatic_cl_factor"] = cp_factor

        return scaled

    def generate(self) -> list[VirtualSubject]:
        """Generate N virtual subjects with correlated physiology (legacy method).

        Samples body weight, cardiac output, GFR, liver weight, and MPPGL
        from log-normal distributions. Organ volumes and flows are scaled
        from ICRP reference values.

        Returns:
            List of VirtualSubject objects.
        """
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
