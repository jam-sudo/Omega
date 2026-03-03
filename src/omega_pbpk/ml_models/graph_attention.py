"""Graph Attention Network (GAT) for multi-task ADME prediction.

Architecture:
  - Pure NumPy (no PyTorch/TensorFlow)
  - AlphaFold-inspired iterative message passing (3 rounds)
  - Multi-head attention (4 heads) with edge features
  - Multi-task: fup, CLint_3A4, Peff, logS

Training:
  - Mini-batch SGD with numerical gradients (finite differences)
  - Target transforms: logit(fup), log10(clint), log10(peff*1e4), logS
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MolecularGraph",
    "smiles_to_graph",
    "GraphAttentionLayer",
    "MultiHeadGraphAttention",
    "MolecularGAT",
    "load_training_data",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "P"]
N_NODE_FEATURES = 13  # 8 one-hot + aromatic + HBD + HBA + halogen + in_ring
N_EDGE_FEATURES = 4  # single, double, triple, aromatic

BOND_SINGLE = 0
BOND_DOUBLE = 1
BOND_TRIPLE = 2
BOND_AROMATIC = 3

_HBD_ATOMS = {"N", "O"}
_HBA_ATOMS = {"N", "O", "F", "S"}
_HALOGEN_ATOMS = {"F", "Cl", "Br"}

_TASK_NAMES = ["fup", "clint_3a4", "peff", "logS"]


# ---------------------------------------------------------------------------
# MolecularGraph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MolecularGraph:
    """Frozen container for a molecular graph.

    Attributes:
        node_features: (n_atoms, 13) float32 array.
        adjacency: (n_atoms, n_atoms) binary float32 matrix.
        edge_features: (n_atoms, n_atoms, 4) float32 bond features.
        num_atoms: number of heavy atoms.
    """

    node_features: NDArray[np.float32]
    adjacency: NDArray[np.float32]
    edge_features: NDArray[np.float32]
    num_atoms: int


def _minimal_graph() -> MolecularGraph:
    nf = np.zeros((1, N_NODE_FEATURES), dtype=np.float32)
    nf[0, 0] = 1.0  # carbon
    return MolecularGraph(
        node_features=nf,
        adjacency=np.zeros((1, 1), dtype=np.float32),
        edge_features=np.zeros((1, 1, N_EDGE_FEATURES), dtype=np.float32),
        num_atoms=1,
    )


# ---------------------------------------------------------------------------
# SMILES parser
# ---------------------------------------------------------------------------


def smiles_to_graph(smiles: str) -> MolecularGraph:
    """Parse a SMILES string into a MolecularGraph.

    Supports organic atoms (C,N,O,S,F,Cl,Br,P), aromatic atoms
    (lowercase), branches (), ring closures (digit or %nn), and
    bond types (- = # :).  Invalid SMILES returns a 1-atom graph.

    Args:
        smiles: SMILES string.

    Returns:
        MolecularGraph instance.
    """
    if not smiles or not smiles.strip():
        return _minimal_graph()
    try:
        return _parse_smiles(smiles.strip())
    except Exception:
        logger.debug(
            "smiles_to_graph: failed to parse %r; returning minimal graph",
            smiles,
        )
        return _minimal_graph()


def _parse_bracket_atom(content: str) -> tuple[str, bool]:
    """Extract (symbol, is_aromatic) from bracket atom content."""
    i = 0
    while i < len(content) and content[i].isdigit():
        i += 1
    if i >= len(content):
        return "C", False
    aromatic = content[i].islower()
    if i + 1 < len(content) and content[i : i + 2].capitalize() in ("Cl", "Br"):
        return content[i : i + 2].capitalize(), False
    return content[i].upper(), aromatic


def _parse_smiles(smiles: str) -> MolecularGraph:  # noqa: C901
    atoms: list[dict[str, Any]] = []
    bonds: list[tuple[int, int, int]] = []
    ring_open: dict[str, tuple[int, int]] = {}
    branch_stack: list[int] = []

    prev_atom = -1
    pending_bond: int | None = None
    i = 0

    while i < len(smiles):
        c = smiles[i]

        # Branch open
        if c == "(":
            branch_stack.append(prev_atom)
            i += 1
            continue

        # Branch close
        if c == ")":
            if branch_stack:
                prev_atom = branch_stack.pop()
            pending_bond = None
            i += 1
            continue

        # Explicit bond types
        if c == "=":
            pending_bond = BOND_DOUBLE
            i += 1
            continue
        if c == "#":
            pending_bond = BOND_TRIPLE
            i += 1
            continue
        if c == ":":
            pending_bond = BOND_AROMATIC
            i += 1
            continue
        if c in "-/\\":
            pending_bond = BOND_SINGLE
            i += 1
            continue

        # Ring closure: %nn
        if c == "%" and i + 2 < len(smiles):
            ring_id = smiles[i + 1 : i + 3]
            bt = pending_bond if pending_bond is not None else BOND_SINGLE
            pending_bond = None
            if prev_atom >= 0:
                _close_or_open_ring(ring_id, prev_atom, bt, atoms, bonds, ring_open)
            i += 3
            continue

        # Ring closure: single digit
        if c.isdigit() and prev_atom >= 0:
            ring_id = c
            bt = pending_bond if pending_bond is not None else BOND_SINGLE
            pending_bond = None
            _close_or_open_ring(ring_id, prev_atom, bt, atoms, bonds, ring_open)
            i += 1
            continue

        # Bracket atom [...]
        if c == "[":
            j = smiles.find("]", i)
            if j == -1:
                raise ValueError("Unclosed bracket")
            sym, aromatic = _parse_bracket_atom(smiles[i + 1 : j])
            atom_idx = len(atoms)
            atoms.append({"symbol": sym, "aromatic": aromatic, "in_ring": False})
            if prev_atom >= 0:
                bt = _default_bond(pending_bond, atoms, prev_atom, aromatic)
                pending_bond = None
                bonds.append((prev_atom, atom_idx, bt))
                bonds.append((atom_idx, prev_atom, bt))
            prev_atom = atom_idx
            i = j + 1
            continue

        # Two-char atoms: Cl, Br
        if c in ("C", "B") and i + 1 < len(smiles):
            two = smiles[i : i + 2]
            if two in ("Cl", "Br"):
                atom_idx = len(atoms)
                atoms.append({"symbol": two, "aromatic": False, "in_ring": False})
                if prev_atom >= 0:
                    bt = pending_bond if pending_bond is not None else BOND_SINGLE
                    pending_bond = None
                    bonds.append((prev_atom, atom_idx, bt))
                    bonds.append((atom_idx, prev_atom, bt))
                prev_atom = atom_idx
                i += 2
                continue

        # Uppercase organic atoms
        if c.isupper() and c in "CNOSFPBI":
            atom_idx = len(atoms)
            atoms.append({"symbol": c, "aromatic": False, "in_ring": False})
            if prev_atom >= 0:
                bt = pending_bond if pending_bond is not None else BOND_SINGLE
                pending_bond = None
                bonds.append((prev_atom, atom_idx, bt))
                bonds.append((atom_idx, prev_atom, bt))
            prev_atom = atom_idx
            i += 1
            continue

        # Lowercase aromatic atoms
        if c.islower() and c in "cnosp":
            sym = c.upper()
            atom_idx = len(atoms)
            atoms.append({"symbol": sym, "aromatic": True, "in_ring": True})
            if prev_atom >= 0:
                bt = _default_bond(pending_bond, atoms, prev_atom, True)
                pending_bond = None
                bonds.append((prev_atom, atom_idx, bt))
                bonds.append((atom_idx, prev_atom, bt))
            prev_atom = atom_idx
            i += 1
            continue

        i += 1  # skip unrecognised chars

    if not atoms:
        return _minimal_graph()

    n = len(atoms)
    nf = np.zeros((n, N_NODE_FEATURES), dtype=np.float32)
    adj = np.zeros((n, n), dtype=np.float32)
    ef = np.zeros((n, n, N_EDGE_FEATURES), dtype=np.float32)

    for idx, atom in enumerate(atoms):
        sym = atom["symbol"]
        if sym in ATOM_TYPES:
            nf[idx, ATOM_TYPES.index(sym)] = 1.0
        nf[idx, 8] = float(atom["aromatic"])
        nf[idx, 9] = float(sym in _HBD_ATOMS)
        nf[idx, 10] = float(sym in _HBA_ATOMS)
        nf[idx, 11] = float(sym in _HALOGEN_ATOMS)
        nf[idx, 12] = float(atom["in_ring"])

    for i_idx, j_idx, bt in bonds:
        adj[i_idx, j_idx] = 1.0
        ef[i_idx, j_idx, bt] = 1.0

    return MolecularGraph(
        node_features=nf,
        adjacency=adj,
        edge_features=ef,
        num_atoms=n,
    )


def _default_bond(
    pending: int | None,
    atoms: list[dict[str, Any]],
    prev_idx: int,
    curr_aromatic: bool,
) -> int:
    if pending is not None:
        return pending
    if atoms[prev_idx]["aromatic"] and curr_aromatic:
        return BOND_AROMATIC
    return BOND_SINGLE


def _close_or_open_ring(
    ring_id: str,
    atom_idx: int,
    bond_type: int,
    atoms: list[dict[str, Any]],
    bonds: list[tuple[int, int, int]],
    ring_open: dict[str, tuple[int, int]],
) -> None:
    if ring_id in ring_open:
        opener_idx, opener_bond = ring_open.pop(ring_id)
        # Determine bond: prefer explicit; infer aromatic if both atoms are
        if bond_type != BOND_SINGLE:
            bt = bond_type
        elif opener_bond != BOND_SINGLE:
            bt = opener_bond
        elif atoms[opener_idx]["aromatic"] and atoms[atom_idx]["aromatic"]:
            bt = BOND_AROMATIC
        else:
            bt = BOND_SINGLE
        bonds.append((opener_idx, atom_idx, bt))
        bonds.append((atom_idx, opener_idx, bt))
        atoms[opener_idx]["in_ring"] = True
        atoms[atom_idx]["in_ring"] = True
    else:
        ring_open[ring_id] = (atom_idx, bond_type)


# ---------------------------------------------------------------------------
# GraphAttentionLayer
# ---------------------------------------------------------------------------


class GraphAttentionLayer:
    """Single-head graph attention with edge features.

    Attention: e_ij = LeakyReLU(a^T [W*h_i || W*h_j || edge_ij])
    alpha_ij = masked softmax over neighbors
    h_i' = sum_j alpha_ij * W*h_j  +  residual  (LayerNorm applied)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_dim: int = N_EDGE_FEATURES,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        lim_w = np.sqrt(6.0 / (in_features + out_features))
        self.W = rng.uniform(-lim_w, lim_w, (in_features, out_features)).astype(np.float32)

        attn_dim = 2 * out_features + edge_dim
        lim_a = np.sqrt(6.0 / attn_dim)
        self.a = rng.uniform(-lim_a, lim_a, (attn_dim,)).astype(np.float32)

        self.ln_gamma = np.ones(out_features, dtype=np.float32)
        self.ln_beta = np.zeros(out_features, dtype=np.float32)

        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim

        if in_features != out_features:
            lim_r = np.sqrt(6.0 / (in_features + out_features))
            self.W_res: NDArray | None = rng.uniform(
                -lim_r, lim_r, (in_features, out_features)
            ).astype(np.float32)
        else:
            self.W_res = None

    @staticmethod
    def _leaky_relu(x: NDArray, alpha: float = 0.2) -> NDArray:
        return np.where(x >= 0, x, alpha * x)

    def _layer_norm(self, x: NDArray) -> NDArray:
        mu = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        return self.ln_gamma * (x - mu) / std + self.ln_beta

    def forward(
        self,
        node_features: NDArray,
        adjacency: NDArray,
        edge_features: NDArray,
    ) -> NDArray:
        """Compute updated node features. Returns (n, out_features)."""
        n = node_features.shape[0]
        Wh = node_features @ self.W  # (n, out)

        Wh_i = np.broadcast_to(Wh[:, np.newaxis, :], (n, n, self.out_features))
        Wh_j = np.broadcast_to(Wh[np.newaxis, :, :], (n, n, self.out_features))
        concat = np.concatenate(
            [
                np.array(Wh_i),
                np.array(Wh_j),
                edge_features,
            ],
            axis=-1,
        )  # (n, n, 2*out+edge)
        e = self._leaky_relu(concat @ self.a)  # (n, n)

        mask = adjacency > 0
        e_masked = np.where(mask, e, -1e9)
        e_max = e_masked.max(axis=1, keepdims=True)
        e_exp = np.exp(np.clip(e_masked - e_max, -50, 0))
        e_exp = np.where(mask, e_exp, 0.0)
        alpha = e_exp / (e_exp.sum(axis=1, keepdims=True) + 1e-8)  # (n, n)

        h_new = alpha @ Wh  # (n, out)
        residual = node_features @ self.W_res if self.W_res is not None else node_features
        return self._layer_norm(h_new + residual)

    def get_attention_weights(
        self,
        node_features: NDArray,
        adjacency: NDArray,
        edge_features: NDArray,
    ) -> NDArray:
        """Return (n, n) attention weight matrix."""
        n = node_features.shape[0]
        Wh = node_features @ self.W
        Wh_i = np.broadcast_to(Wh[:, np.newaxis, :], (n, n, self.out_features))
        Wh_j = np.broadcast_to(Wh[np.newaxis, :, :], (n, n, self.out_features))
        concat = np.concatenate([np.array(Wh_i), np.array(Wh_j), edge_features], axis=-1)
        e = self._leaky_relu(concat @ self.a)
        mask = adjacency > 0
        e_masked = np.where(mask, e, -1e9)
        e_max = e_masked.max(axis=1, keepdims=True)
        e_exp = np.exp(np.clip(e_masked - e_max, -50, 0))
        e_exp = np.where(mask, e_exp, 0.0)
        return e_exp / (e_exp.sum(axis=1, keepdims=True) + 1e-8)

    def get_params(self) -> dict[str, NDArray]:
        p: dict[str, NDArray] = {
            "W": self.W,
            "a": self.a,
            "ln_gamma": self.ln_gamma,
            "ln_beta": self.ln_beta,
        }
        if self.W_res is not None:
            p["W_res"] = self.W_res
        return p

    def set_params(self, p: dict[str, NDArray]) -> None:
        self.W = p["W"]
        self.a = p["a"]
        self.ln_gamma = p["ln_gamma"]
        self.ln_beta = p["ln_beta"]
        if "W_res" in p:
            self.W_res = p["W_res"]


# ---------------------------------------------------------------------------
# MultiHeadGraphAttention
# ---------------------------------------------------------------------------


class MultiHeadGraphAttention:
    """4-head graph attention with specialized head roles.

    Heads:
      0 — general structural
      1 — H-bonding patterns
      2 — lipophilic interactions
      3 — polarity patterns

    Output: concat(heads) → linear projection → out_features.
    """

    HEAD_NAMES = ["structural", "hbonding", "lipophilic", "polarity"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        seed: int = 42,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads

        self.heads = [
            GraphAttentionLayer(in_features, out_features, seed=seed + h) for h in range(n_heads)
        ]

        rng = np.random.default_rng(seed + 100)
        concat_dim = n_heads * out_features
        lim = np.sqrt(6.0 / (concat_dim + out_features))
        self.W_proj = rng.uniform(-lim, lim, (concat_dim, out_features)).astype(np.float32)
        self.b_proj = np.zeros(out_features, dtype=np.float32)

    def forward(
        self,
        node_features: NDArray,
        adjacency: NDArray,
        edge_features: NDArray,
    ) -> NDArray:
        """Compute multi-head attention output. Returns (n, out_features)."""
        head_out = [h.forward(node_features, adjacency, edge_features) for h in self.heads]
        concat = np.concatenate(head_out, axis=-1)  # (n, n_heads*out)
        return np.tanh(concat @ self.W_proj + self.b_proj)

    def get_attention_weights(
        self,
        node_features: NDArray,
        adjacency: NDArray,
        edge_features: NDArray,
    ) -> list[NDArray]:
        """Return list of (n, n) attention matrices, one per head."""
        return [
            h.get_attention_weights(node_features, adjacency, edge_features) for h in self.heads
        ]

    def get_params(self) -> dict[str, Any]:
        return {
            "W_proj": self.W_proj,
            "b_proj": self.b_proj,
            "heads": [h.get_params() for h in self.heads],
        }

    def set_params(self, p: dict[str, Any]) -> None:
        self.W_proj = p["W_proj"]
        self.b_proj = p["b_proj"]
        for head, hp in zip(self.heads, p["heads"], strict=False):
            head.set_params(hp)


# ---------------------------------------------------------------------------
# MolecularGAT
# ---------------------------------------------------------------------------


class MolecularGAT:
    """Graph Attention Network for multi-task ADME prediction.

    Pipeline:
        1. Node embedding: Linear(13 → hidden_dim)
        2. n_rounds × MultiHeadGraphAttention(hidden_dim → hidden_dim)
        3. Mean-pooling readout → graph embedding (hidden_dim,)
        4. 4 task heads (hidden_dim → 1)

    Target transforms (applied before training):
        fup    → logit  (log(fup/(1-fup)))
        clint  → log10(clint + 0.01)
        peff   → log10(peff*1e4 + 0.001)
        logS   → identity

    NOTE: fit() uses numerical gradients (finite differences) which is
    slow for large models. Use small n_epochs for practical training.
    """

    def __init__(
        self,
        n_node_features: int = N_NODE_FEATURES,
        hidden_dim: int = 32,
        n_heads: int = 4,
        n_rounds: int = 3,
        n_tasks: int = 4,
        seed: int = 42,
    ) -> None:
        self.n_node_features = n_node_features
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_rounds = n_rounds
        self.n_tasks = n_tasks
        self._seed = seed
        self._trained = False

        rng = np.random.default_rng(seed)

        # Embedding layer
        lim_e = np.sqrt(6.0 / (n_node_features + hidden_dim))
        self._W_embed = rng.uniform(-lim_e, lim_e, (n_node_features, hidden_dim)).astype(np.float32)
        self._b_embed = np.zeros(hidden_dim, dtype=np.float32)

        # Attention layers
        self._attn_layers = [
            MultiHeadGraphAttention(hidden_dim, hidden_dim, n_heads, seed=seed + r * 10)
            for r in range(n_rounds)
        ]

        # Task heads: (W: hidden_dim×1, b: scalar)
        self._task_heads: list[tuple[NDArray, NDArray]] = []
        for _t in range(n_tasks):
            lim_t = np.sqrt(6.0 / (hidden_dim + 1))
            W_t = rng.uniform(-lim_t, lim_t, (hidden_dim, 1)).astype(np.float32)
            b_t = np.zeros(1, dtype=np.float32)
            self._task_heads.append((W_t, b_t))

        # Standardisation parameters (set during fit)
        self._target_means = np.zeros(n_tasks, dtype=np.float32)
        self._target_stds = np.ones(n_tasks, dtype=np.float32)

    # ------------------------------------------------------------------
    # Target transforms
    # ------------------------------------------------------------------

    @staticmethod
    def _xform_fup(v: float) -> float:
        v = float(np.clip(v, 1e-4, 1 - 1e-4))
        return float(np.log(v / (1.0 - v)))

    @staticmethod
    def _inv_fup(logit: float) -> float:
        return float(1.0 / (1.0 + np.exp(-float(np.clip(logit, -20, 20)))))

    @staticmethod
    def _xform_clint(v: float) -> float:
        return float(np.log10(max(float(v), 1e-6) + 0.01))

    @staticmethod
    def _inv_clint(lv: float) -> float:
        return float(max(10.0 ** float(lv) - 0.01, 0.0))

    @staticmethod
    def _xform_peff(v: float) -> float:
        return float(np.log10(max(float(v) * 1e4, 1e-10) + 0.001))

    @staticmethod
    def _inv_peff(lv: float) -> float:
        return float(max(10.0 ** float(lv) - 0.001, 0.0))

    @staticmethod
    def _xform_logs(v: float) -> float:
        return float(v)

    def _build_target_matrix(self, td: dict[str, NDArray]) -> NDArray:
        n = len(next(iter(td.values())))
        Y = np.zeros((n, self.n_tasks), dtype=np.float32)
        if "fup" in td:
            Y[:, 0] = [self._xform_fup(v) for v in td["fup"]]
        clint = td.get("clint_3a4", td.get("clint", np.zeros(n)))
        Y[:, 1] = [self._xform_clint(v) for v in clint]
        if "peff" in td:
            Y[:, 2] = [self._xform_peff(v) for v in td["peff"]]
        if "logS" in td:
            Y[:, 3] = [self._xform_logs(v) for v in td["logS"]]
        return Y

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _embed(self, node_features: NDArray) -> NDArray:
        return np.tanh(node_features @ self._W_embed + self._b_embed)

    def forward(self, graph: MolecularGraph) -> NDArray:
        """Forward pass. Returns (n_tasks,) on standardised-transformed scale."""
        h = self._embed(graph.node_features)
        for attn in self._attn_layers:
            h = attn.forward(h, graph.adjacency, graph.edge_features)
        graph_emb = h.mean(axis=0)  # (hidden,)
        preds = np.zeros(self.n_tasks, dtype=np.float32)
        for t, (W_t, b_t) in enumerate(self._task_heads):
            preds[t] = float(graph_emb @ W_t + b_t)
        return preds

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_single(self, smiles: str) -> dict[str, Any]:
        """Predict ADME properties for one SMILES string.

        Returns:
            Dict with keys fup, clint_3a4, peff (×10⁻⁴ cm/s), logS,
            confidence (0–1).
        """
        graph = smiles_to_graph(smiles)
        raw = self.forward(graph)
        # Un-standardise then back-transform
        raw_u = raw * self._target_stds + self._target_means
        fup = float(np.clip(self._inv_fup(float(raw_u[0])), 0.001, 0.999))
        clint = float(np.clip(self._inv_clint(float(raw_u[1])), 0.001, 200.0))
        peff_raw = float(np.clip(self._inv_peff(float(raw_u[2])), 0.0, 100.0))
        logs = float(raw_u[3])
        confidence = self._compute_confidence(graph)
        return {
            "fup": round(fup, 4),
            "clint_3a4": round(clint, 3),
            "peff": round(peff_raw, 3),  # returned as ×10⁻⁴ cm/s
            "logS": round(logs, 2),
            "confidence": round(confidence, 3),
        }

    def predict_batch(self, smiles_list: list[str]) -> list[dict[str, Any]]:
        """Predict ADME properties for multiple SMILES strings."""
        return [self.predict_single(s) for s in smiles_list]

    def _compute_confidence(self, graph: MolecularGraph) -> float:
        """Confidence from attention-weight variance across heads (0–1)."""
        if graph.num_atoms < 2:
            return 0.5
        h = self._embed(graph.node_features)
        variances: list[float] = []
        for attn in self._attn_layers:
            wts = attn.get_attention_weights(h, graph.adjacency, graph.edge_features)
            stacked = np.stack(wts, axis=0)  # (n_heads, n, n)
            variances.append(float(stacked.var(axis=0).mean()))
            h = attn.forward(h, graph.adjacency, graph.edge_features)
        mean_var = float(np.mean(variances)) if variances else 0.1
        return float(np.clip(np.exp(-10.0 * mean_var), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Parameter access (for numerical gradient training)
    # ------------------------------------------------------------------

    def _collect_param_arrays(self) -> list[NDArray]:
        """Return list of references to all parameter arrays."""
        params: list[NDArray] = [self._W_embed, self._b_embed]
        for attn in self._attn_layers:
            p = attn.get_params()
            params.extend([p["W_proj"], p["b_proj"]])
            for hp in p["heads"]:
                params.extend([hp["W"], hp["a"], hp["ln_gamma"], hp["ln_beta"]])
                if "W_res" in hp:
                    params.append(hp["W_res"])
        for W_t, b_t in self._task_heads:
            params.extend([W_t, b_t])
        return params

    # ------------------------------------------------------------------
    # Training (numerical gradients)
    # ------------------------------------------------------------------

    def _batch_loss(self, graphs: list[MolecularGraph], Y_std: NDArray) -> float:
        total = 0.0
        for g, y in zip(graphs, Y_std, strict=False):
            diff = self.forward(g) - y
            total += float(np.dot(diff, diff)) / self.n_tasks
        return total / max(len(graphs), 1)

    def fit(
        self,
        smiles_list: list[str],
        targets_dict: dict[str, NDArray],
        n_epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 16,
    ) -> list[float]:
        """Train using mini-batch SGD with numerical (finite-difference) gradients.

        NOTE: Training is slow due to O(n_params × 2) forward passes per
        parameter update. For large models use small n_epochs (e.g. 2–5).

        Args:
            smiles_list: Training SMILES.
            targets_dict: Dict of {task: array} for fup, clint_3a4, peff, logS.
            n_epochs: Training epochs.
            lr: Learning rate.
            batch_size: Mini-batch size.

        Returns:
            List of per-epoch losses.
        """
        graphs = [smiles_to_graph(s) for s in smiles_list]
        Y = self._build_target_matrix(targets_dict)

        self._target_means = Y.mean(axis=0).astype(np.float32)
        std = Y.std(axis=0)
        self._target_stds = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        Y_std = ((Y - self._target_means) / self._target_stds).astype(np.float32)

        n = len(graphs)
        rng = np.random.default_rng(self._seed)
        eps = 1e-5
        loss_history: list[float] = []

        for epoch in range(n_epochs):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = idx[start : start + batch_size]
                bg = [graphs[k] for k in batch_idx]
                bY = Y_std[batch_idx]

                # Numerical gradients: 2 forward passes per parameter element
                for p_arr in self._collect_param_arrays():
                    flat = p_arr.ravel()  # view → in-place modification
                    grad = np.zeros(len(flat), dtype=np.float32)
                    for k in range(len(flat)):
                        orig = float(flat[k])
                        flat[k] = orig + eps
                        lp = self._batch_loss(bg, bY)
                        flat[k] = orig - eps
                        lm = self._batch_loss(bg, bY)
                        flat[k] = orig
                        grad[k] = float((lp - lm) / (2.0 * eps))
                    flat -= lr * grad

                epoch_loss += self._batch_loss(bg, bY)
                n_batches += 1

            epoch_loss /= max(n_batches, 1)
            loss_history.append(epoch_loss)
            if epoch % max(1, n_epochs // 5) == 0:
                logger.debug("GAT epoch %d/%d  loss=%.4f", epoch + 1, n_epochs, epoch_loss)

        self._trained = True
        return loss_history

    # ------------------------------------------------------------------
    # Attention maps
    # ------------------------------------------------------------------

    def get_attention_maps(self, smiles: str) -> dict[str, Any]:
        """Return per-layer, per-head attention weight matrices.

        Args:
            smiles: SMILES string.

        Returns:
            Dict with num_atoms and layers list (each with layer index
            and per-head (n, n) attention matrices as lists).
        """
        graph = smiles_to_graph(smiles)
        h = self._embed(graph.node_features)
        result: dict[str, Any] = {"num_atoms": graph.num_atoms, "layers": []}
        for layer_idx, attn in enumerate(self._attn_layers):
            wts = attn.get_attention_weights(h, graph.adjacency, graph.edge_features)
            result["layers"].append(
                {
                    "layer": layer_idx,
                    "heads": {
                        MultiHeadGraphAttention.HEAD_NAMES[hi]: w.tolist()
                        for hi, w in enumerate(wts)
                    },
                }
            )
            h = attn.forward(h, graph.adjacency, graph.edge_features)
        return result

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model parameters to a .npz file."""
        path = Path(path)
        sd: dict[str, NDArray] = {
            "W_embed": self._W_embed,
            "b_embed": self._b_embed,
            "target_means": self._target_means,
            "target_stds": self._target_stds,
        }
        for li, attn in enumerate(self._attn_layers):
            p = attn.get_params()
            sd[f"l{li}_W_proj"] = p["W_proj"]
            sd[f"l{li}_b_proj"] = p["b_proj"]
            for hi, hp in enumerate(p["heads"]):
                sd[f"l{li}_h{hi}_W"] = hp["W"]
                sd[f"l{li}_h{hi}_a"] = hp["a"]
                sd[f"l{li}_h{hi}_gamma"] = hp["ln_gamma"]
                sd[f"l{li}_h{hi}_beta"] = hp["ln_beta"]
                if "W_res" in hp:
                    sd[f"l{li}_h{hi}_W_res"] = hp["W_res"]
        for t, (W_t, b_t) in enumerate(self._task_heads):
            sd[f"task_{t}_W"] = W_t
            sd[f"task_{t}_b"] = b_t
        np.savez(path, **sd)
        logger.info("MolecularGAT saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model parameters from a .npz file."""
        path = Path(path)
        data = np.load(path)
        self._W_embed = data["W_embed"]
        self._b_embed = data["b_embed"]
        self._target_means = data["target_means"]
        self._target_stds = data["target_stds"]
        for li, attn in enumerate(self._attn_layers):
            p: dict[str, Any] = {
                "W_proj": data[f"l{li}_W_proj"],
                "b_proj": data[f"l{li}_b_proj"],
                "heads": [],
            }
            for hi in range(self.n_heads):
                hp: dict[str, NDArray] = {
                    "W": data[f"l{li}_h{hi}_W"],
                    "a": data[f"l{li}_h{hi}_a"],
                    "ln_gamma": data[f"l{li}_h{hi}_gamma"],
                    "ln_beta": data[f"l{li}_h{hi}_beta"],
                }
                key = f"l{li}_h{hi}_W_res"
                if key in data:
                    hp["W_res"] = data[key]
                p["heads"].append(hp)
            attn.set_params(p)
        for t in range(self.n_tasks):
            self._task_heads[t] = (data[f"task_{t}_W"], data[f"task_{t}_b"])
        self._trained = True
        logger.info("MolecularGAT loaded from %s", path)


# ---------------------------------------------------------------------------
# Training data loader
# ---------------------------------------------------------------------------

_DRUG_SMILES: dict[str, str] = {
    "midazolam": "CN1C(=O)CN=C(c2ccccc2Cl)c2cc(F)ccc21",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "propranolol": "CC(C)NCC(O)COc1cccc2ccccc12",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "atenolol": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "verapamil": ("COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC"),
    "nifedipine": ("COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]"),
    "diltiazem": ("COc1ccc2c(c1)CC(OC(C)=O)C(c1ccc(OC)cc1)N2CC(C)C"),
    "simvastatin": ("CCC(C)(C)C(=O)O[C@@H]1C[C@@H](O)CC2=CC(=O)OC[C@H]12"),
    "atorvastatin": "CC(C)c1n(CC(C)C(=O)O)c(-c2ccccc2F)cc1-c1ccc(F)cc1",
    "omeprazole": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "lansoprazole": "FC(F)(F)COc1ccnc2[nH]c(S(=O)Cc3cnc(C)c(OC)c3C)nc12",
    "clopidogrel": "COC(=O)[C@@H](c1ccccc1Cl)N1CCc2sccc2C1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "naproxen": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "diclofenac": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "celecoxib": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(N)cc2)cc1",
    "morphine": "OC1CC2N(C)CC[C@@]3([C@@H]2O1)c1ccc(O)cc1O3",
    "fentanyl": "CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1",
    "tramadol": "OC1(c2ccccc2OC)CC(CN(C)C)CC1",
    "fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "sertraline": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
    "paroxetine": "Fc1ccc([C@@H]2COCCN2Cc2ccc3c(c2)OCO3)cc1",
    "citalopram": "OCCCN(C)CCC1(OCc2cc(C#N)ccc21)c1ccc(F)cc1",
    "amitriptyline": "CN(C)CCCC1(c2ccccc2)c2ccccc2CC=C1",
    "haloperidol": "OC1(CCN(CCCl)c2ccc(Cl)cc2)CCCCC1=O",
    "olanzapine": "Cc1ccc2nc3cccc(N4CCN(C)CC4)c3sc2c1",
    "risperidone": "Cc1c(F)ccc2c1CCN(CCC1OCCO1)C2=O",
    "quetiapine": "OCCN1CCN(c2ccc3c(c2)Sc2ccccc2N3CCCCO)CC1",
    "clozapine": "CN1CCN(c2nc3ccccc3nc3ccc(Cl)cc23)CC1",
    "alprazolam": "Cc1nnc2n1-c1ccc(Cl)cc1C=NC2",
    "diazepam": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "lorazepam": "OC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2NC1=O",
    "zolpidem": "Cc1ccc(-c2nc3ccc(C)cn3c2CC(=O)N(C)C)cc1",
    "venlafaxine": "COc1ccc(C2(CN(C)C)CCCC2)cc1O",
    "duloxetine": "CNCCc1cccc2cccc(OC(c3cccs3)c12)c12",
    "bupropion": "CC(C)(N)CC(=O)c1cccc(Cl)c1",
    "carbamazepine": "NC(=O)N1c2ccccc2C=Cc2ccccc21",
    "valproic_acid": "CCCC(CCC)C(=O)O",
    "phenytoin": "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",
    "lamotrigine": "Nc1ncc(-c2ccc(Cl)cc2Cl)nc1N",
    "levetiracetam": "CC[C@@H](CC(N)=O)N1CCCC1=O",
    "gabapentin": "NCC1(CC(=O)O)CCCCC1",
    "pregabalin": "CC(CN)CC(C)C(=O)O",
    "amoxicillin": ("CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O"),
    "ciprofloxacin": "O=C(O)c1cn2c(n1)cc(N1CCNCC1)c(=O)c2F",
    "levofloxacin": "CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
    "metronidazole": "Cc1ncc([N+](=O)[O-])n1CCO",
    "fluconazole": "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
    "dexamethasone": (
        "C[C@@H]1C[C@H]2[C@@H]3CC(=O)C4=CC(=O)CC[C@]4(C)[C@@H]3[C@@H](O)C[C@@]2(C)[C@H]1C"
    ),
    "prednisone": "O=C1CC[C@H]2[C@@H]3CC(=O)C4=CC(=O)CC[C@]4(C)[C@H]3CC(=O)[C@@]2(C)C1",
    "prednisolone": "OC1CC(=O)C2=CC(=O)CC[C@]3(C)[C@@H](O)CC[C@@]23C1",
    "tacrolimus": (
        "CC1OC(=O)[C@H](CC(=O)/C=C/CC=CC(C)CC(CC)"
        "C(=O)C[C@H](O)[C@H](C)CC(O)[C@H](C)/C=C/C(=O)O1)"
        "N1CCCC1"
    ),
    "methotrexate": ("CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1"),
}


def load_training_data() -> tuple[list[str], dict[str, NDArray]]:
    """Load ADME training data from adme_reference.csv.

    Maps compound names to SMILES via the built-in lookup dictionary.
    Only compounds with valid SMILES and non-NaN targets are included.

    Returns:
        (smiles_list, targets_dict) where targets_dict has keys
        'fup', 'clint_3a4', 'peff', 'logS'.
    """
    ref_path = (
        Path(__file__).parent.parent.parent.parent  # ml_models/  # omega_pbpk/  # src/  # Omega/
        / "data"
        / "adme_reference.csv"
    )
    if not ref_path.exists():
        logger.warning("adme_reference.csv not found at %s", ref_path)
        return [], {}

    smiles_out: list[str] = []
    fup_out: list[float] = []
    clint_out: list[float] = []
    peff_out: list[float] = []
    logs_out: list[float] = []

    with open(ref_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_name = row.get("name", "").strip().lower()
            name = raw_name.replace(" ", "_").replace("-", "_")
            if name not in _DRUG_SMILES:
                continue
            smi = _DRUG_SMILES[name]
            try:
                fup = float(row.get("fup", "nan"))
                clint = float(row.get("clint_3a4_uL_min_pmol", "nan"))
                peff = float(row.get("peff_cm_s", "nan"))
                mw = float(row.get("mw", "nan"))
                logp = float(row.get("logP", "nan"))
            except (ValueError, TypeError):
                continue
            if any(np.isnan(v) for v in [fup, clint, peff, mw, logp]):
                continue
            logs = -0.01 * mw - 0.8 * logp + 1.0
            smiles_out.append(smi)
            fup_out.append(float(np.clip(fup, 1e-4, 0.9999)))
            clint_out.append(float(np.clip(clint, 1e-6, 200.0)))
            peff_out.append(float(np.clip(peff, 1e-10, 1.0)))
            logs_out.append(float(logs))

    if not smiles_out:
        logger.warning("No valid training compounds found in adme_reference.csv")
        return [], {}

    logger.info("Loaded %d training compounds for GAT", len(smiles_out))
    targets: dict[str, NDArray] = {
        "fup": np.array(fup_out, dtype=np.float32),
        "clint_3a4": np.array(clint_out, dtype=np.float32),
        "peff": np.array(peff_out, dtype=np.float32),
        "logS": np.array(logs_out, dtype=np.float32),
    }
    return smiles_out, targets
