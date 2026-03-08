"""Graph neural network encoder that maps molecular graphs to latent representations.

Implements a 3-layer Message Passing Neural Network (MPNN) with edge-conditioned
message functions. Provides both a PyTorch Geometric backend and a pure-PyTorch
fallback using adjacency matrix operations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from omega_pbpk.ml.features.graphs import ATOM_FEATURE_DIM, BOND_FEATURE_DIM

try:
    from torch_geometric.nn import NNConv, global_max_pool, global_mean_pool

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class _PureTorchMPLayer(nn.Module):
    """Pure-PyTorch message passing layer using adjacency matrix operations.

    Implements edge-conditioned message passing without PyG:
        messages_j = edge_nn(edge_attr) @ node_features[src]
        aggregated_i = scatter_add(messages, dst)
        out = linear(cat(node_features, aggregated_i))
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Edge network: maps edge features to a in_dim x out_dim weight matrix
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_dim, in_dim * out_dim),
        )
        self.node_update = nn.Linear(in_dim + out_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (N, in_dim)
        edge_index : Tensor (2, E)
        edge_attr : Tensor (E, edge_dim)

        Returns
        -------
        Tensor (N, out_dim)
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            # No edges: just transform node features
            agg = torch.zeros(num_nodes, self.out_dim, device=x.device, dtype=x.dtype)
            return self.node_update(torch.cat([x, agg], dim=-1))

        src, dst = edge_index[0], edge_index[1]

        # Compute edge-conditioned messages
        # edge_nn output: (E, in_dim * out_dim) -> reshape to (E, in_dim, out_dim)
        edge_weights = self.edge_nn(edge_attr).view(num_edges, self.in_dim, self.out_dim)

        # Source node features: (E, in_dim, 1)
        src_feats = x[src].unsqueeze(-1)

        # Messages: (E, out_dim)
        messages = torch.bmm(edge_weights.transpose(1, 2), src_feats).squeeze(-1)

        # Aggregate by destination (scatter_add)
        agg = torch.zeros(num_nodes, self.out_dim, device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, messages)

        # Update
        return self.node_update(torch.cat([x, agg], dim=-1))


class _PyGMPLayer(nn.Module):
    """PyG-based message passing layer using NNConv."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int) -> None:
        super().__init__()
        edge_nn = nn.Sequential(
            nn.Linear(edge_dim, in_dim * out_dim),
        )
        self.conv = NNConv(in_dim, out_dim, edge_nn, aggr="add")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv(x, edge_index, edge_attr)


class MolecularEncoder(nn.Module):
    """3-layer MPNN that produces a fixed-size molecular embedding.

    Architecture:
        Layer 1: atom_features (32-dim) -> 128-dim
        Layer 2: 128 -> 256-dim
        Layer 3: 256 -> 256-dim
        Each layer: message passing + batch norm + ReLU + dropout(0.1)
        Readout: concat(global_mean_pool, global_max_pool) -> 512 -> Linear -> 256

    Parameters
    ----------
    atom_dim : int
        Atom feature dimension (default: ATOM_FEATURE_DIM = 32).
    bond_dim : int
        Bond feature dimension (default: BOND_FEATURE_DIM = 10).
    hidden_dims : tuple[int, ...]
        Hidden dimensions for each MPNN layer. Default: (128, 256, 256).
    embedding_dim : int
        Output embedding dimension. Default: 256.
    dropout : float
        Dropout rate. Default: 0.1.
    """

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden_dims: tuple[int, ...] = (128, 256, 256),
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        layer_cls = _PyGMPLayer if HAS_PYG else _PureTorchMPLayer

        # Build MPNN layers
        dims = [atom_dim, *hidden_dims]
        self.mp_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(len(hidden_dims)):
            self.mp_layers.append(layer_cls(dims[i], dims[i + 1], bond_dim))
            self.bn_layers.append(nn.BatchNorm1d(dims[i + 1]))

        # Readout: concat mean+max pooling (2 * last_hidden) -> embedding_dim
        last_hidden = hidden_dims[-1]
        self.readout = nn.Linear(2 * last_hidden, embedding_dim)

    def forward(self, data: object) -> torch.Tensor:
        """Forward pass: molecular graph -> embedding vector.

        Parameters
        ----------
        data : Data or _FallbackData or _FallbackBatch
            Graph data with attributes: x, edge_index, edge_attr, batch.

        Returns
        -------
        Tensor of shape (batch_size, embedding_dim)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = getattr(data, "batch", None)

        # If no batch attribute, assume single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Message passing layers
        for mp, bn in zip(self.mp_layers, self.bn_layers):
            x = mp(x, edge_index, edge_attr)
            # BatchNorm1d requires >1 sample during training; skip for single-node
            if x.size(0) > 1 or not self.training:
                x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling
        if HAS_PYG:
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
        else:
            mean_pool = _scatter_mean(x, batch)
            max_pool = _scatter_max(x, batch)

        # Readout
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.readout(pooled)


def _scatter_mean(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch scatter mean pooling."""
    num_graphs = int(index.max().item()) + 1
    out = torch.zeros(num_graphs, src.size(1), device=src.device, dtype=src.dtype)
    count = torch.zeros(num_graphs, 1, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    ones = torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype)
    count.index_add_(0, index, ones)
    return out / count.clamp(min=1)


def _scatter_max(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch scatter max pooling."""
    num_graphs = int(index.max().item()) + 1
    out = torch.full(
        (num_graphs, src.size(1)),
        float("-inf"),
        device=src.device,
        dtype=src.dtype,
    )
    for i in range(num_graphs):
        mask = index == i
        if mask.any():
            out[i] = src[mask].max(dim=0).values
    return out
