"""Phase 1054 — Drug Synaptic Cleft PK.

Predict drug concentrations in the synaptic cleft for CNS drugs
affecting neurotransmission.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SynapticCleftResult:
    drug_name: str
    target_neurotransmitter: str
    c_synaptic_cleft_nM: float
    receptor_occupancy_pct: float
    neurotransmitter_effect: str
    synaptic_half_life_ms: float
    brain_to_synapse_ratio: float
    therapeutic_window_synapse: str
    notes: str


_VALID_NEUROTRANSMITTERS = {
    "dopamine",
    "serotonin",
    "GABA",
    "glutamate",
    "norepinephrine",
}

_VALID_MECHANISMS = {
    "reuptake_inhibition",
    "receptor_antagonism",
    "receptor_agonism",
}


def predict_synaptic_cleft_pk(
    drug_name: str,
    plasma_conc_mg_L: float,
    mw_Da: float,
    logp: float,
    fu_plasma: float = 0.3,
    fu_brain: float = 0.1,
    target_neurotransmitter: str = "dopamine",
    ki_nM: float = 100.0,
    plasma_to_brain_ratio: float = 0.0,
    mechanism: str = "reuptake_inhibition",
    therapeutic_conc_nM: float = 10.0,
    toxic_conc_nM: float = 1000.0,
) -> SynapticCleftResult:
    """Predict drug concentration and effect in the synaptic cleft."""
    # --- Validation ---
    if plasma_conc_mg_L <= 0:
        raise ValueError(f"plasma_conc_mg_L must be > 0, got {plasma_conc_mg_L}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if not (0.0 < fu_brain <= 1.0):
        raise ValueError(f"fu_brain must be in (0, 1], got {fu_brain}")
    if ki_nM <= 0:
        raise ValueError(f"ki_nM must be > 0, got {ki_nM}")
    if target_neurotransmitter not in _VALID_NEUROTRANSMITTERS:
        raise ValueError(
            f"target_neurotransmitter must be one of {_VALID_NEUROTRANSMITTERS}, "
            f"got {target_neurotransmitter!r}"
        )
    if mechanism not in _VALID_MECHANISMS:
        raise ValueError(f"mechanism must be one of {_VALID_MECHANISMS}, got {mechanism!r}")

    # --- Plasma free concentration ---
    c_free_plasma_mg_L = plasma_conc_mg_L * fu_plasma
    c_free_plasma_nM = c_free_plasma_mg_L * 1e6 / mw_Da

    # --- Brain Kp ---
    if plasma_to_brain_ratio == 0.0:
        kp_brain = 10 ** (0.5 * logp - 0.01 * mw_Da) * fu_plasma
        kp_brain = max(0.01, min(2.0, kp_brain))
    else:
        kp_brain = plasma_to_brain_ratio

    c_brain_nM = c_free_plasma_nM * kp_brain
    c_synaptic_nM = c_brain_nM * fu_brain

    # --- Receptor occupancy ---
    ro = c_synaptic_nM / (c_synaptic_nM + ki_nM)
    receptor_occupancy_pct = ro * 100.0

    # --- Synaptic half-life ---
    tau_ms = 5.0 + 0.01 * mw_Da
    synaptic_half_life_ms = tau_ms * 0.693

    # --- Neurotransmitter effect ---
    if mechanism == "reuptake_inhibition" and ro > 0.2:
        neurotransmitter_effect = "reuptake_block"
    elif mechanism == "receptor_antagonism" and ro > 0.2:
        neurotransmitter_effect = "inhibition"
    elif mechanism == "receptor_agonism" and ro > 0.2:
        neurotransmitter_effect = "enhancement"
    else:
        neurotransmitter_effect = "none"

    # --- Brain to synapse ratio ---
    brain_to_synapse_ratio = c_brain_nM / c_synaptic_nM  # = 1 / fu_brain

    # --- Therapeutic window ---
    if c_synaptic_nM < therapeutic_conc_nM:
        therapeutic_window_synapse = "sub-therapeutic"
    elif c_synaptic_nM < toxic_conc_nM:
        therapeutic_window_synapse = "therapeutic"
    else:
        therapeutic_window_synapse = "supratherapeutic"

    # --- Notes ---
    notes_parts = [
        f"Target: {target_neurotransmitter}, mechanism: {mechanism}.",
        f"C_plasma_free={c_free_plasma_nM:.2f} nM, kp_brain={kp_brain:.4f}.",
        f"C_brain={c_brain_nM:.2f} nM, C_synapse={c_synaptic_nM:.2f} nM.",
        f"RO={receptor_occupancy_pct:.1f}% (Ki={ki_nM} nM).",
        f"Effect: {neurotransmitter_effect}; t1/2_syn={synaptic_half_life_ms:.2f} ms.",
        f"Window: {therapeutic_window_synapse}.",
    ]
    notes = " ".join(notes_parts)

    return SynapticCleftResult(
        drug_name=drug_name,
        target_neurotransmitter=target_neurotransmitter,
        c_synaptic_cleft_nM=c_synaptic_nM,
        receptor_occupancy_pct=receptor_occupancy_pct,
        neurotransmitter_effect=neurotransmitter_effect,
        synaptic_half_life_ms=synaptic_half_life_ms,
        brain_to_synapse_ratio=brain_to_synapse_ratio,
        therapeutic_window_synapse=therapeutic_window_synapse,
        notes=notes,
    )
