"""Level 3 PK Foundation Model with cross-attention fusion.

Combines molecular (GNN), patient, and dosing encoders via cross-attention
to produce patient-specific PK predictions. When called without patient
or dosing context, degrades gracefully to Level 2 behavior.

Architecture:
    Molecule -> MolecularEncoder -> 256-dim (Query)
    Patient  -> PatientEncoder   ->  64-dim ─┐
    Dosing   -> DosingEncoder    ->  64-dim ─┤-> concat -> 128-dim (Key/Value)
    Cross-attention(Q=mol, KV=context) -> 256-dim fused
    -> PKParameterHead -> PK params
    -> DifferentiableODESurrogate -> C(t) curve + PK metrics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from omega_pbpk.ml.data.synthetic import DT_H, N_TIMEPOINTS
from omega_pbpk.ml.features.graphs import smiles_to_graph
from omega_pbpk.ml.models.foundation.dosing_encoder import DosingEncoder
from omega_pbpk.ml.models.foundation.end_to_end import SMILESToPKModel
from omega_pbpk.ml.models.foundation.gnn_encoder import MolecularEncoder
from omega_pbpk.ml.models.foundation.param_head import PKParameterHead
from omega_pbpk.ml.models.foundation.patient_encoder import PatientEncoder
from omega_pbpk.ml.models.surrogate.differentiable_ode import (
    DifferentiableODESurrogate,
)

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention: molecular embedding attends to patient+dosing context.

    Parameters
    ----------
    mol_dim : int
        Molecular embedding dimension (query dim). Default: 256.
    context_dim : int
        Patient+dosing context dimension (key/value dim). Default: 128.
    n_heads : int
        Number of attention heads. Default: 4.
    dropout : float
        Attention dropout. Default: 0.1.
    """

    def __init__(
        self,
        mol_dim: int = 256,
        context_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.mol_dim = mol_dim
        self.context_dim = context_dim
        self.n_heads = n_heads
        self.head_dim = mol_dim // n_heads

        assert mol_dim % n_heads == 0, (
            f"mol_dim ({mol_dim}) must be divisible by n_heads ({n_heads})"
        )

        # Project query from mol_dim, key/value from context_dim
        self.q_proj = nn.Linear(mol_dim, mol_dim)
        self.k_proj = nn.Linear(context_dim, mol_dim)
        self.v_proj = nn.Linear(context_dim, mol_dim)
        self.out_proj = nn.Linear(mol_dim, mol_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(mol_dim)

    def forward(
        self,
        mol_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention.

        Parameters
        ----------
        mol_emb : Tensor of shape (batch, mol_dim)
            Molecular embedding (query).
        context : Tensor of shape (batch, context_dim)
            Concatenated patient + dosing embedding (key/value).

        Returns
        -------
        Tensor of shape (batch, mol_dim)
            Fused representation with residual connection.
        """
        batch_size = mol_emb.shape[0]

        # Treat each as a single-token sequence for attention
        # Q: (batch, 1, mol_dim), K/V: (batch, 1, mol_dim)
        q = self.q_proj(mol_emb).unsqueeze(1)  # (B, 1, mol_dim)
        k = self.k_proj(context).unsqueeze(1)  # (B, 1, mol_dim)
        v = self.v_proj(context).unsqueeze(1)  # (B, 1, mol_dim)

        # Reshape for multi-head: (B, n_heads, 1, head_dim)
        q = q.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim**0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, 1, 1)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, H, 1, head_dim)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.mol_dim).squeeze(1)
        )  # (B, mol_dim)

        # Output projection + residual + layer norm
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        return self.layer_norm(mol_emb + output)


class PKFoundationModel(nn.Module):
    """Level 3 foundation model: SMILES + patient + dosing -> PK.

    Wraps the Level 2 SMILESToPKModel components and adds patient/dosing
    encoders with cross-attention fusion. When patient/dosing context is
    not provided, behaves identically to the Level 2 model.

    Parameters
    ----------
    embedding_dim : int
        Molecular embedding dimension (default: 256).
    patient_dim : int
        Patient encoder output dimension (default: 64).
    dosing_dim : int
        Dosing encoder output dimension (default: 64).
    n_attention_heads : int
        Number of cross-attention heads (default: 4).
    surrogate_n_output : int
        Number of timepoints in surrogate output.
    surrogate_hidden : int
        Surrogate network hidden dimension.
    dt_h : float
        Time step in hours.
    """

    # Inherit param order from SMILESToPKModel
    SURROGATE_PARAM_ORDER = SMILESToPKModel.SURROGATE_PARAM_ORDER

    def __init__(
        self,
        embedding_dim: int = 256,
        patient_dim: int = 64,
        dosing_dim: int = 64,
        n_attention_heads: int = 4,
        surrogate_n_output: int = N_TIMEPOINTS,
        surrogate_hidden: int = 256,
        dt_h: float = DT_H,
    ) -> None:
        super().__init__()

        self.dt_h = dt_h
        self.embedding_dim = embedding_dim
        self.patient_dim = patient_dim
        self.dosing_dim = dosing_dim

        # Level 2 components
        self.encoder = MolecularEncoder(embedding_dim=embedding_dim)
        self.param_head = PKParameterHead(embedding_dim=embedding_dim)
        self.surrogate = DifferentiableODESurrogate(
            n_input=6,
            n_output=surrogate_n_output,
            hidden_dim=surrogate_hidden,
        )

        # Level 3 components
        self.patient_encoder = PatientEncoder(output_dim=patient_dim)
        self.dosing_encoder = DosingEncoder(output_dim=dosing_dim)
        self.cross_attention = CrossAttentionFusion(
            mol_dim=embedding_dim,
            context_dim=patient_dim + dosing_dim,
            n_heads=n_attention_heads,
        )

    def forward(
        self,
        smiles_or_graph: str | Any,
        covariates: dict[str, Any] | None = None,
        regimen: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Forward pass: molecule + optional patient/dosing -> PK.

        Parameters
        ----------
        smiles_or_graph : str or graph Data
            SMILES string or pre-computed molecular graph.
        covariates : dict or None
            Patient covariates. If None, uses population defaults (Level 2 mode).
        regimen : dict or None
            Dosing regimen. If None, uses default (100 mg oral single dose).

        Returns
        -------
        dict with:
            - params: dict[str, Tensor] PK parameters
            - curve: Tensor (batch, n_timepoints) C(t) curve
            - pk_metrics: dict with cmax, auc, tmax, t_half
            - embedding: Tensor (batch, embedding_dim) molecular embedding
            - patient_embedding: Tensor (batch, patient_dim) or None
            - dosing_embedding: Tensor (batch, dosing_dim) or None
            - fused_embedding: Tensor (batch, embedding_dim) fused representation
        """
        # 1. Parse molecular input
        if isinstance(smiles_or_graph, str):
            graph = smiles_to_graph(smiles_or_graph)
            if graph is None:
                raise ValueError(f"Could not parse SMILES: {smiles_or_graph}")
        else:
            graph = smiles_or_graph

        # Move graph to model device
        device = next(self.parameters()).device
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        graph.edge_attr = graph.edge_attr.to(device)
        if hasattr(graph, "batch") and graph.batch is not None:
            graph.batch = graph.batch.to(device)

        # 2. Molecular encoding
        mol_emb = self.encoder(graph)  # (batch, 256)

        # 3. Patient + dosing encoding
        if covariates is not None or regimen is not None:
            pat_cov = covariates if covariates is not None else {}
            dos_reg = regimen if regimen is not None else {}

            pat_emb = self.patient_encoder(pat_cov)  # (batch, 64)
            dos_emb = self.dosing_encoder(dos_reg)  # (batch, 64)

            # Ensure batch dimensions match
            if pat_emb.shape[0] != mol_emb.shape[0]:
                pat_emb = pat_emb.expand(mol_emb.shape[0], -1)
            if dos_emb.shape[0] != mol_emb.shape[0]:
                dos_emb = dos_emb.expand(mol_emb.shape[0], -1)

            # 4. Cross-attention fusion
            context = torch.cat([pat_emb, dos_emb], dim=-1)  # (batch, 128)
            fused = self.cross_attention(mol_emb, context)  # (batch, 256)
        else:
            # Level 2 mode: no patient/dosing context
            pat_emb = None
            dos_emb = None
            fused = mol_emb

        # 5. PK parameter prediction from fused embedding
        pk_params = self.param_head(fused)

        # 6. Surrogate forward pass
        surrogate_input = self._extract_surrogate_params(pk_params)
        curve_log = self.surrogate(surrogate_input)
        curve = torch.expm1(torch.clamp(curve_log, max=20.0))
        curve = torch.clamp(curve, min=0.0)

        # 7. PK metrics
        pk_metrics = self._extract_pk_metrics(curve)

        return {
            "params": pk_params,
            "curve": curve,
            "pk_metrics": pk_metrics,
            "embedding": mol_emb,
            "patient_embedding": pat_emb,
            "dosing_embedding": dos_emb,
            "fused_embedding": fused,
        }

    def _extract_surrogate_params(self, pk_params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Map named PK params to 6D surrogate input vector.

        Same IVIVE scaling as SMILESToPKModel.
        """
        ivive_factor = 40.0 * 45.0 * 1800.0 / 1e6 / 60.0
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
        return torch.stack(params_list, dim=-1)

    def _extract_pk_metrics(self, curve: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract PK metrics from C(t) curve (differentiable)."""
        batch_size, n_time = curve.shape
        device = curve.device

        cmax, cmax_idx = torch.max(curve, dim=-1)
        tmax = cmax_idx.float() * self.dt_h
        auc = torch.trapz(curve, dx=self.dt_h)

        # Half-life via log-linear regression on terminal phase
        t = torch.arange(n_time, device=device, dtype=curve.dtype) * self.dt_h
        log_curve = torch.log(curve + 1e-12)

        time_idx = torch.arange(n_time, device=device).unsqueeze(0)
        mask = torch.sigmoid((time_idx.float() - cmax_idx.unsqueeze(1).float() - 0.5) * 5.0)
        conc_mask = torch.sigmoid((curve - cmax.unsqueeze(1) * 0.01) * 100.0)
        weights = mask * conc_mask
        weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights_norm = weights / weights_sum

        t_expand = t.unsqueeze(0).expand(batch_size, -1)
        t_mean = (weights_norm * t_expand).sum(dim=-1)
        y_mean = (weights_norm * log_curve).sum(dim=-1)
        t_centered = t_expand - t_mean.unsqueeze(1)
        y_centered = log_curve - y_mean.unsqueeze(1)

        numerator = (weights_norm * t_centered * y_centered).sum(dim=-1)
        denominator = (weights_norm * t_centered * t_centered).sum(dim=-1).clamp(min=1e-8)
        slope = numerator / denominator
        ke = torch.clamp(-slope, min=1e-6)
        t_half = torch.clamp(0.693147 / ke, min=0.1, max=1000.0)

        return {"cmax": cmax, "auc": auc, "tmax": tmax, "t_half": t_half}

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "embedding_dim": self.embedding_dim,
                "patient_dim": self.patient_dim,
                "dosing_dim": self.dosing_dim,
                "surrogate_n_output": self.surrogate.n_output,
                "surrogate_hidden": self.surrogate.hidden_dim,
                "dt_h": self.dt_h,
                "model_type": "PKFoundationModel",
            },
            str(path),
        )
        logger.info("Saved PKFoundationModel to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> PKFoundationModel:
        """Load model from checkpoint."""
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        model = cls(
            embedding_dim=checkpoint["embedding_dim"],
            patient_dim=checkpoint.get("patient_dim", 64),
            dosing_dim=checkpoint.get("dosing_dim", 64),
            surrogate_n_output=checkpoint["surrogate_n_output"],
            surrogate_hidden=checkpoint["surrogate_hidden"],
            dt_h=checkpoint["dt_h"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(device)
        return model

    @classmethod
    def from_level2(
        cls, level2_model: SMILESToPKModel, freeze_level2: bool = False
    ) -> PKFoundationModel:
        """Initialize from a trained Level 2 model.

        Copies encoder, param_head, and surrogate weights from the Level 2 model,
        then adds randomly initialized patient/dosing encoders and cross-attention.

        Parameters
        ----------
        level2_model : SMILESToPKModel
            Trained Level 2 model.
        freeze_level2 : bool
            If True, freeze the Level 2 components during fine-tuning.
        """
        model = cls(
            embedding_dim=level2_model.embedding_dim,
            surrogate_n_output=level2_model.surrogate.n_output,
            surrogate_hidden=level2_model.surrogate.hidden_dim,
            dt_h=level2_model.dt_h,
        )

        # Copy Level 2 weights
        model.encoder.load_state_dict(level2_model.encoder.state_dict())
        model.param_head.load_state_dict(level2_model.param_head.state_dict())
        model.surrogate.load_state_dict(level2_model.surrogate.state_dict())

        if freeze_level2:
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.param_head.parameters():
                param.requires_grad = False
            for param in model.surrogate.parameters():
                param.requires_grad = False

        return model

    def parameter_count(self) -> dict[str, int]:
        """Return parameter counts per sub-module."""

        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "encoder": _count(self.encoder),
            "param_head": _count(self.param_head),
            "surrogate": _count(self.surrogate),
            "patient_encoder": _count(self.patient_encoder),
            "dosing_encoder": _count(self.dosing_encoder),
            "cross_attention": _count(self.cross_attention),
            "total": _count(self),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


__all__ = ["PKFoundationModel", "CrossAttentionFusion"]
