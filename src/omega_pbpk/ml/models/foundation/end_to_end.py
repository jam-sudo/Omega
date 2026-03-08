"""End-to-end SMILES-to-PK model wiring GNN encoder, parameter head, and ODE surrogate.

Architecture:
    SMILES -> smiles_to_graph -> MolecularEncoder (256-dim)
           -> PKParameterHead (named PK params)
           -> DifferentiableODESurrogate (241-pt C(t) curve)
           -> PK metric extraction (Cmax, AUC, Tmax, t_half)

All operations are differentiable (PyTorch only, no numpy) to enable
gradient-based training through the entire pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn

from omega_pbpk.ml.data.synthetic import DT_H, N_TIMEPOINTS, T_END_H
from omega_pbpk.ml.features.graphs import smiles_to_graph
from omega_pbpk.ml.models.foundation.gnn_encoder import MolecularEncoder
from omega_pbpk.ml.models.foundation.param_head import PKParameterHead
from omega_pbpk.ml.models.surrogate.differentiable_ode import (
    DifferentiableODESurrogate,
)

logger = logging.getLogger(__name__)


class SMILESToPKModel(nn.Module):
    """End-to-end model: SMILES (or graph) -> PK params + C(t) curve + PK metrics.

    Wires together:
    1. MolecularEncoder: graph -> 256-dim embedding
    2. PKParameterHead: embedding -> named PK parameters
    3. DifferentiableODESurrogate: 6D param vector -> 241-point C(t) curve

    The surrogate is used ONLY during training for backprop.
    At inference, use MLPKPredictor which runs the real ODE.

    Parameters
    ----------
    embedding_dim : int
        GNN output embedding dimension (default: 256).
    surrogate_n_output : int
        Number of timepoints in surrogate output (default: 241).
    surrogate_hidden : int
        Hidden dim for surrogate network (default: 256).
    dt_h : float
        Time step in hours for PK metric extraction (default: 0.1).
    """

    # The 6 parameters the surrogate expects, in order
    SURROGATE_PARAM_ORDER = ("logP", "fup", "clint_L_h", "mw", "rbp", "peff")

    def __init__(
        self,
        embedding_dim: int = 256,
        surrogate_n_output: int = N_TIMEPOINTS,
        surrogate_hidden: int = 256,
        dt_h: float = DT_H,
    ) -> None:
        super().__init__()

        self.dt_h = dt_h
        self.embedding_dim = embedding_dim

        # Sub-modules
        self.encoder = MolecularEncoder(embedding_dim=embedding_dim)
        self.param_head = PKParameterHead(embedding_dim=embedding_dim)
        self.surrogate = DifferentiableODESurrogate(
            n_input=6,
            n_output=surrogate_n_output,
            hidden_dim=surrogate_hidden,
        )

    def forward(
        self, smiles_or_graph: Union[str, Any]
    ) -> dict[str, Any]:
        """Forward pass: SMILES string or graph -> full PK prediction.

        Parameters
        ----------
        smiles_or_graph : str or Data-like
            Either a SMILES string (converted to graph internally) or
            a pre-computed graph object with x, edge_index, edge_attr.

        Returns
        -------
        dict with keys:
            - params: dict[str, Tensor] of named PK parameters
            - curve: Tensor (batch, 241) concentration-time curve
            - pk_metrics: dict[str, Tensor] with cmax, auc, tmax, t_half
            - embedding: Tensor (batch, 256) molecular embedding
        """
        # 1. Convert SMILES to graph if needed
        if isinstance(smiles_or_graph, str):
            graph = smiles_to_graph(smiles_or_graph)
            if graph is None:
                raise ValueError(
                    f"Could not parse SMILES: {smiles_or_graph}"
                )
        else:
            graph = smiles_or_graph

        # Move graph tensors to same device as model
        device = next(self.parameters()).device
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        graph.edge_attr = graph.edge_attr.to(device)
        if hasattr(graph, "batch") and graph.batch is not None:
            graph.batch = graph.batch.to(device)

        # 2. GNN encoder -> embedding
        embedding = self.encoder(graph)  # (batch, 256)

        # 3. Parameter head -> named PK params
        pk_params = self.param_head(embedding)  # dict[str, (batch,)]

        # 4. Extract 6D vector for surrogate
        surrogate_input = self._extract_surrogate_params(pk_params)  # (batch, 6)

        # 5. Surrogate -> C(t) curve in log1p space, then expm1
        curve_log = self.surrogate(surrogate_input)  # (batch, 241)
        curve = torch.expm1(torch.clamp(curve_log, max=20.0))
        curve = torch.clamp(curve, min=0.0)

        # 6. Extract PK metrics from curve
        pk_metrics = self._extract_pk_metrics(curve)

        return {
            "params": pk_params,
            "curve": curve,
            "pk_metrics": pk_metrics,
            "embedding": embedding,
        }

    def _extract_surrogate_params(
        self, pk_params: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Map PKParameterHead output names to the 6D surrogate input vector.

        Surrogate expects: [logP, fup, clint_L_h, mw, rbp, peff]

        The param_head outputs clint_3a4 and clint_2d6 in uL/min/pmol units.
        The surrogate expects clint_L_h (total hepatic clearance in L/h).
        We apply IVIVE scaling: clint_total * 40 * 45 * 1800 / 1e6 / 60.

        Parameters
        ----------
        pk_params : dict from PKParameterHead.forward()

        Returns
        -------
        Tensor of shape (batch, 6)
        """
        batch_size = pk_params["logP"].shape[0]

        # IVIVE scaling factor: MPPGL(40) * mg_protein/g(45) * liver_wt(1800g) / 1e6 / 60
        ivive_factor = 40.0 * 45.0 * 1800.0 / 1e6 / 60.0  # ~0.054

        # Total CLint from both CYP enzymes, then scale
        clint_total = pk_params["clint_3a4"] + pk_params["clint_2d6"]
        clint_L_h = clint_total * ivive_factor

        params_list = [
            pk_params["logP"],
            pk_params["fup"],
            clint_L_h,
            pk_params["mw"],
            pk_params["rbp"],
            pk_params["peff"],
        ]

        return torch.stack(params_list, dim=-1)  # (batch, 6)

    def _extract_pk_metrics(
        self, curve: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Extract PK metrics from predicted C(t) curve using differentiable ops.

        Parameters
        ----------
        curve : Tensor of shape (batch, n_timepoints)
            Predicted plasma concentration-time curve.

        Returns
        -------
        dict with keys: cmax, auc, tmax, t_half
            Each value has shape (batch,).
        """
        batch_size, n_time = curve.shape

        # Cmax: max concentration
        cmax, cmax_idx = torch.max(curve, dim=-1)  # (batch,)

        # Tmax: time of max concentration
        tmax = cmax_idx.float() * self.dt_h  # (batch,)

        # AUC: trapezoidal rule (differentiable)
        # AUC = sum( (C[i] + C[i+1]) / 2 * dt ) for i in 0..n-2
        auc = torch.trapz(curve, dx=self.dt_h)  # (batch,)

        # t_half: time for curve to drop to Cmax/2 after Tmax
        # Use a differentiable soft approximation
        t_half = self._compute_half_life(curve, cmax, cmax_idx)

        return {
            "cmax": cmax,
            "auc": auc,
            "tmax": tmax,
            "t_half": t_half,
        }

    def _compute_half_life(
        self,
        curve: torch.Tensor,
        cmax: torch.Tensor,
        cmax_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute terminal half-life from C(t) curve.

        Uses log-linear regression on the terminal phase (post-Tmax)
        to estimate the elimination rate constant, then t_half = ln(2)/ke.

        For differentiability, uses a soft mask weighted regression.

        Parameters
        ----------
        curve : (batch, n_time)
        cmax : (batch,)
        cmax_idx : (batch,)

        Returns
        -------
        t_half : (batch,)
        """
        batch_size, n_time = curve.shape
        device = curve.device

        # Time vector
        t = torch.arange(n_time, device=device, dtype=curve.dtype) * self.dt_h

        # Log concentration (with floor for numerical stability)
        log_curve = torch.log(curve + 1e-12)  # (batch, n_time)

        # Create soft mask for post-Tmax region
        time_idx = torch.arange(n_time, device=device).unsqueeze(0)  # (1, n_time)
        # Sigmoid mask: ~1 for t > tmax, ~0 for t < tmax
        mask = torch.sigmoid(
            (time_idx.float() - cmax_idx.unsqueeze(1).float() - 0.5) * 5.0
        )  # (batch, n_time)

        # Also mask out very low concentrations (below 1% of Cmax)
        conc_mask = torch.sigmoid(
            (curve - cmax.unsqueeze(1) * 0.01) * 100.0
        )  # (batch, n_time)

        weights = mask * conc_mask  # (batch, n_time)
        weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights_norm = weights / weights_sum

        # Weighted linear regression: log(C) = a + b*t, b = -ke
        t_expand = t.unsqueeze(0).expand(batch_size, -1)  # (batch, n_time)

        # Weighted means
        t_mean = (weights_norm * t_expand).sum(dim=-1)  # (batch,)
        y_mean = (weights_norm * log_curve).sum(dim=-1)  # (batch,)

        # Weighted slope
        t_centered = t_expand - t_mean.unsqueeze(1)
        y_centered = log_curve - y_mean.unsqueeze(1)

        numerator = (weights_norm * t_centered * y_centered).sum(dim=-1)
        denominator = (weights_norm * t_centered * t_centered).sum(dim=-1).clamp(min=1e-8)

        slope = numerator / denominator  # (batch,) -- should be negative

        # ke = -slope, t_half = ln(2) / ke
        ke = torch.clamp(-slope, min=1e-6)  # ensure positive
        t_half = 0.693147 / ke

        # Clamp to reasonable range
        t_half = torch.clamp(t_half, min=0.1, max=1000.0)

        return t_half

    def save(self, path: str | Path) -> None:
        """Save model checkpoint.

        Parameters
        ----------
        path : str or Path
            Output file path (.pt).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "embedding_dim": self.embedding_dim,
                "surrogate_n_output": self.surrogate.n_output,
                "surrogate_hidden": self.surrogate.hidden_dim,
                "dt_h": self.dt_h,
            },
            str(path),
        )
        logger.info("Saved SMILESToPKModel to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> SMILESToPKModel:
        """Load model from checkpoint.

        Parameters
        ----------
        path : str or Path
            Checkpoint file path (.pt).
        device : str
            Device to load to.

        Returns
        -------
        Loaded SMILESToPKModel in eval mode.
        """
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        model = cls(
            embedding_dim=checkpoint["embedding_dim"],
            surrogate_n_output=checkpoint["surrogate_n_output"],
            surrogate_hidden=checkpoint["surrogate_hidden"],
            dt_h=checkpoint["dt_h"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(device)
        return model

    def parameter_count(self) -> dict[str, int]:
        """Return parameter counts per sub-module."""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "encoder": _count(self.encoder),
            "param_head": _count(self.param_head),
            "surrogate": _count(self.surrogate),
            "total": _count(self),
            "trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


__all__ = ["SMILESToPKModel"]
