"""Inhalational anesthetic PK model — Phase 272.

Models alveolar gas exchange, blood uptake, and tissue equilibration
for volatile anesthetics. Based on Mapleson's fat/lean compartment model.

Key parameters:
- Blood/gas partition coefficient (lambda_b): solubility in blood
- FA/FI: fraction of alveolar to inspired concentration

References
----------
- Eger EI. Anesthetic Uptake and Action. Williams & Wilkins, 1974.
- Mapleson WW, Br J Anaesth. 1963;35:129-35
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnestheticPKResult:
    """Result from inhalational anesthetic PK simulation."""

    drug_name: str
    fa_fi_ratio: list[float]  # Alveolar/inspired concentration ratio
    times_min: list[float]  # Time in minutes
    c_arterial: list[float]  # Arterial blood concentration (relative to inspired)
    c_brain: list[float]  # Brain/CNS concentration (relative)
    c_muscle: list[float]  # Muscle concentration (relative)
    c_fat: list[float]  # Fat concentration (relative)
    time_to_half_fa_fi_min: float  # Time for FA/FI to reach 0.5
    induction_time_min: float  # Time for brain conc to reach 0.5
    notes: str


def simulate_inhalational_anesthetic(
    drug_name: str,
    lambda_bg: float,  # Blood/gas partition coefficient
    lambda_brain_blood: float = 1.4,  # Brain/blood partition
    lambda_muscle_blood: float = 3.0,  # Muscle/blood partition
    lambda_fat_blood: float = 80.0,  # Fat/blood partition
    alveolar_ventilation_L_per_min: float = 4.0,
    cardiac_output_L_per_min: float = 5.0,
    fi: float = 1.0,  # Inspired fraction (normalized)
    t_end_min: float = 30.0,  # Simulation duration (min)
    dt_min: float = 0.1,
) -> AnestheticPKResult:
    """Simulate inhalational anesthetic uptake.

    Uses a simplified Mapleson compartment model:
    Alveolar → Arterial blood → 3 tissue compartments (brain, muscle, fat).

    Parameters
    ----------
    drug_name : Anesthetic name (e.g., 'sevoflurane', 'isoflurane').
    lambda_bg : Blood/gas partition coefficient. Must be > 0.
    lambda_brain_blood : Brain/blood partition coefficient. Must be > 0.
    lambda_muscle_blood : Muscle/blood partition coefficient. Must be > 0.
    lambda_fat_blood : Fat/blood partition coefficient. Must be > 0.
    alveolar_ventilation_L_per_min : VA (L/min). Must be > 0.
    cardiac_output_L_per_min : Q (L/min). Must be > 0.
    fi : Inspired fraction (arbitrary normalized units). Must be > 0.
    t_end_min : Simulation duration (min). Must be > 0.
    dt_min : Time step (min).

    Returns
    -------
    AnestheticPKResult
    """
    for name, val in [
        ("lambda_bg", lambda_bg),
        ("lambda_brain_blood", lambda_brain_blood),
        ("lambda_muscle_blood", lambda_muscle_blood),
        ("lambda_fat_blood", lambda_fat_blood),
        ("alveolar_ventilation_L_per_min", alveolar_ventilation_L_per_min),
        ("cardiac_output_L_per_min", cardiac_output_L_per_min),
        ("fi", fi),
        ("t_end_min", t_end_min),
    ]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0")

    # Tissue volumes (fraction of 70 kg body)
    v_brain = 1.5  # L
    v_muscle = 25.0  # L
    v_fat = 14.0  # L
    v_blood = 5.0  # L

    # Blood flow fractions
    q_brain = cardiac_output_L_per_min * 0.12  # 12% to brain
    q_muscle = cardiac_output_L_per_min * 0.17  # 17% to muscle
    q_fat = cardiac_output_L_per_min * 0.05  # 5% to fat

    n_steps = int(t_end_min / dt_min) + 1
    times = [i * dt_min for i in range(n_steps)]

    fa = [0.0] * n_steps  # Alveolar fraction
    c_art = [0.0] * n_steps  # Arterial blood (relative)
    c_br = [0.0] * n_steps  # Brain
    c_mu = [0.0] * n_steps  # Muscle
    c_ft = [0.0] * n_steps  # Fat

    for i in range(1, n_steps):
        fa_prev = fa[i - 1]
        c_art_prev = c_art[i - 1]
        c_br_prev = c_br[i - 1]
        c_mu_prev = c_mu[i - 1]
        c_ft_prev = c_ft[i - 1]

        # Alveolar concentration update
        # dFA/dt = (VA * (FI - FA) - Q * lambda_bg * (FA - Ca/lambda_bg)) / FRC
        # Simplified: FA approaches FI with time constant = FRC/(VA + Q*lambda_bg)
        # Using simple uptake: Uptake = Q * lambda_bg * (FA - Ca)
        # where Ca = arterial blood conc (in same units as FA via lambda_bg)

        # Arterial blood from alveolus
        uptake = cardiac_output_L_per_min * lambda_bg * (fa_prev - c_art_prev / lambda_bg)

        # FRC (functional residual capacity) ~ 2.5 L
        frc = 2.5
        d_fa = (alveolar_ventilation_L_per_min * (fi - fa_prev) - uptake) / frc * dt_min

        # Tissue uptake
        # Simplified: tissue conc equilibrates with arterial via blood flow and partition
        d_cbr = (
            q_brain
            * (c_art_prev - c_br_prev / lambda_brain_blood)
            / (v_brain * lambda_brain_blood)
            * dt_min
        )
        d_cmu = (
            q_muscle
            * (c_art_prev - c_mu_prev / lambda_muscle_blood)
            / (v_muscle * lambda_muscle_blood)
            * dt_min
        )
        d_cft = (
            q_fat
            * (c_art_prev - c_ft_prev / lambda_fat_blood)
            / (v_fat * lambda_fat_blood)
            * dt_min
        )

        # Arterial blood: mixed venous return minus alveolar equilibration
        d_cart = (
            (
                q_brain * (c_br_prev / lambda_brain_blood - c_art_prev)
                + q_muscle * (c_mu_prev / lambda_muscle_blood - c_art_prev)
                + q_fat * (c_ft_prev / lambda_fat_blood - c_art_prev)
            )
            / v_blood
            + uptake / v_blood
        ) * dt_min

        fa[i] = max(0.0, min(fi, fa_prev + d_fa))
        c_art[i] = max(0.0, c_art_prev + d_cart)
        c_br[i] = max(0.0, c_br_prev + d_cbr)
        c_mu[i] = max(0.0, c_mu_prev + d_cmu)
        c_ft[i] = max(0.0, c_ft_prev + d_cft)

    # FA/FI ratio
    fa_fi = [f / fi for f in fa]

    # Time to FA/FI = 0.5
    t_half_fafi = t_end_min
    for i, ratio in enumerate(fa_fi):
        if ratio >= 0.5:
            t_half_fafi = times[i]
            break

    # Induction time: brain reaches 0.5 * equilibrium (~fi * lambda_bg equivalently)
    target_brain = 0.5 * lambda_brain_blood * fi  # approximate equilibrium
    induction_t = t_end_min
    for i, cb in enumerate(c_br):
        if cb >= target_brain * 0.5:
            induction_t = times[i]
            break

    notes = (
        f"{drug_name}: lambda_b/g={lambda_bg:.2f}. "
        f"FA/FI=0.5 at {t_half_fafi:.1f} min. "
        f"Brain 50% equilibrium at {induction_t:.1f} min."
    )

    return AnestheticPKResult(
        drug_name=drug_name,
        fa_fi_ratio=fa_fi,
        times_min=times,
        c_arterial=c_art,
        c_brain=c_br,
        c_muscle=c_mu,
        c_fat=c_ft,
        time_to_half_fa_fi_min=t_half_fafi,
        induction_time_min=induction_t,
        notes=notes,
    )


__all__ = ["AnestheticPKResult", "simulate_inhalational_anesthetic"]
