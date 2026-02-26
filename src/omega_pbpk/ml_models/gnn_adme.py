"""GNN MPNN architecture for multi-task ADME property prediction.

Architecture:
  - 3-layer Message Passing Neural Network (MPNN)
  - Attention-weighted readout
  - Multi-task prediction heads (6 endpoints)
  - ~200K parameters

Endpoints:
  1. logP
  2. logS (aqueous solubility)
  3. logPeff (intestinal permeability)
  4. logfup (fraction unbound)
  5. logCLint_3A4 (CYP3A4 clearance)
  6. loghERG_IC50 (hERG inhibition)

Requires PyTorch + torch_geometric for training. This module defines the
architecture scaffold and can be used without PyTorch for type checking.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch not available; GNN model is scaffold-only.")


# Feature dimensions
ATOM_FEATURES = 32  # Node feature dimension
BOND_FEATURES = 8  # Edge feature dimension
HIDDEN_DIM = 128  # MPNN hidden dimension
N_LAYERS = 3  # Number of message passing layers
N_ENDPOINTS = 6  # Multi-task output heads

ENDPOINT_NAMES = ["logP", "logS", "logPeff", "logfup", "logCLint_3A4", "loghERG_IC50"]


if HAS_TORCH:

    class MessagePassingLayer(nn.Module):
        """Single MPNN message passing step."""

        def __init__(self, hidden_dim: int, edge_dim: int) -> None:
            super().__init__()
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
        ) -> torch.Tensor:
            row, col = edge_index
            # Message computation
            msg_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
            messages = self.edge_mlp(msg_input)

            # Aggregation (sum)
            agg = torch.zeros_like(x)
            agg.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)

            # Node update
            update_input = torch.cat([x, agg], dim=-1)
            x_new = self.node_mlp(update_input)
            return self.norm(x + x_new)  # Residual connection

    class AttentionReadout(nn.Module):
        """Attention-weighted graph readout."""

        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
            attn_weights = self.attention(x)
            # Softmax per graph in batch
            attn_weights = attn_weights - attn_weights.max()
            attn_weights = torch.exp(attn_weights)

            # Sum attention per graph for normalization
            attn_sum = torch.zeros(batch.max() + 1, 1, device=x.device)
            attn_sum.scatter_add_(0, batch.unsqueeze(-1), attn_weights)
            attn_weights = attn_weights / (attn_sum[batch] + 1e-8)

            # Weighted sum
            weighted = x * attn_weights
            out = torch.zeros(batch.max() + 1, x.size(-1), device=x.device)
            out.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted), weighted)
            return out

    class ADME_MPNN(nn.Module):
        """Multi-task MPNN for ADME property prediction.

        Architecture: Atom embedding → 3× MPNN layers → Attention readout → 6 prediction heads
        """

        def __init__(
            self,
            atom_dim: int = ATOM_FEATURES,
            edge_dim: int = BOND_FEATURES,
            hidden_dim: int = HIDDEN_DIM,
            n_layers: int = N_LAYERS,
            n_endpoints: int = N_ENDPOINTS,
        ) -> None:
            super().__init__()
            self.atom_encoder = nn.Linear(atom_dim, hidden_dim)
            self.mp_layers = nn.ModuleList(
                [MessagePassingLayer(hidden_dim, edge_dim) for _ in range(n_layers)]
            )
            self.readout = AttentionReadout(hidden_dim)
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim // 2, 1),
                    )
                    for _ in range(n_endpoints)
                ]
            )

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            batch: torch.Tensor,
        ) -> torch.Tensor:
            h = self.atom_encoder(x)
            for layer in self.mp_layers:
                h = layer(h, edge_index, edge_attr)
            graph_emb = self.readout(h, batch)
            predictions = torch.cat([head(graph_emb) for head in self.heads], dim=-1)
            return predictions

        def count_parameters(self) -> int:
            """Count total trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:
    # Scaffold classes when PyTorch is not available
    class ADME_MPNN:  # type: ignore[no-redef]
        """Scaffold ADME_MPNN (PyTorch not installed)."""

        def __init__(self, **kwargs: Any) -> None:
            logger.warning("PyTorch not installed. GNN model is scaffold-only.")

        def count_parameters(self) -> int:
            return 0


def model_summary() -> dict[str, Any]:
    """Return model architecture summary."""
    return {
        "name": "ADME_MPNN",
        "atom_features": ATOM_FEATURES,
        "bond_features": BOND_FEATURES,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS,
        "n_endpoints": N_ENDPOINTS,
        "endpoint_names": ENDPOINT_NAMES,
        "estimated_parameters": "~200K",
        "torch_available": HAS_TORCH,
    }
