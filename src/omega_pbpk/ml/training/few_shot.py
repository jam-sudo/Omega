"""Reptile meta-learning for few-shot PK prediction.

Given a foundation model pre-trained on many drugs, Reptile meta-learning
finds initialization weights that can be quickly adapted to a new drug
using only 1-5 observed concentration-time data points.

Algorithm (Reptile, Nichol et al. 2018):
    for each meta-iteration:
        sample task (drug + few C(t) observations)
        clone model weights
        inner-loop: k gradient steps on task data
        outer-loop: move original weights toward adapted weights
            theta += meta_lr * (adapted_weights - original_weights)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class FewShotTask:
    """A single few-shot PK task: one drug with sparse observations.

    Attributes
    ----------
    smiles : str
        SMILES string of the drug.
    observations : list of (time_h, conc_mg_L) tuples
        Observed concentration-time data points (1-5 points).
    graph : Any
        Pre-computed molecular graph (optional, for efficiency).
    covariates : dict or None
        Patient covariates (Level 3).
    regimen : dict or None
        Dosing regimen (Level 3).
    drug_id : str
        Identifier for this drug/task.
    """

    smiles: str = ""
    observations: list[tuple[float, float]] = field(default_factory=list)
    graph: Any = None
    covariates: dict[str, Any] | None = None
    regimen: dict[str, Any] | None = None
    drug_id: str = ""


class ReptileTrainer:
    """Reptile meta-learning trainer for few-shot PK adaptation.

    Parameters
    ----------
    model : nn.Module
        PKFoundationModel (or SMILESToPKModel) to meta-train.
    meta_lr : float
        Outer-loop learning rate for Reptile update. Default: 0.1.
    inner_lr : float
        Inner-loop learning rate for task adaptation. Default: 1e-3.
    inner_steps : int
        Number of gradient steps per task in inner loop. Default: 5.
    device : str
        Device for computation. Default: 'cpu'.
    """

    def __init__(
        self,
        model: nn.Module,
        meta_lr: float = 0.1,
        inner_lr: float = 1e-3,
        inner_steps: int = 5,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = torch.device(device)
        self.model.to(self.device)

    def _task_loss(
        self,
        model: nn.Module,
        task: FewShotTask,
    ) -> torch.Tensor:
        """Compute MSE loss on observed concentration-time points.

        Parameters
        ----------
        model : nn.Module
            The model to evaluate.
        task : FewShotTask
            Task with observations.

        Returns
        -------
        Scalar loss tensor.
        """
        if not task.observations:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Forward pass
        graph = task.graph
        if graph is None:
            from omega_pbpk.ml.features.graphs import smiles_to_graph

            graph = smiles_to_graph(task.smiles)
            if graph is None:
                raise ValueError(f"Cannot parse SMILES: {task.smiles}")

        # Determine if model accepts covariates/regimen
        import inspect

        sig = inspect.signature(model.forward)
        has_covariates = "covariates" in sig.parameters

        if has_covariates:
            result = model(
                graph,
                covariates=task.covariates,
                regimen=task.regimen,
            )
        else:
            result = model(graph)

        predicted_curve = result["curve"].squeeze(0)  # (n_timepoints,)
        dt_h = model.dt_h if hasattr(model, "dt_h") else 0.1

        # Compare at observed time points
        loss = torch.tensor(0.0, device=self.device)
        for time_h, conc_obs in task.observations:
            # Find nearest time index
            idx = int(round(time_h / dt_h))
            idx = min(idx, predicted_curve.shape[0] - 1)
            idx = max(idx, 0)

            pred_conc = predicted_curve[idx]
            target = torch.tensor(conc_obs, device=self.device, dtype=torch.float32)

            # MSE in log space for better scale handling
            loss = loss + (torch.log1p(pred_conc) - torch.log1p(target)) ** 2

        return loss / max(len(task.observations), 1)

    def meta_train_step(self, tasks: list[FewShotTask]) -> float:
        """Perform one Reptile meta-training step across a batch of tasks.

        Parameters
        ----------
        tasks : list of FewShotTask
            Tasks to train on in this meta-iteration.

        Returns
        -------
        Average task loss (float).
        """
        self.model.train()

        # Save original weights
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        total_loss = 0.0
        n_tasks = 0

        for task in tasks:
            # Clone model for inner loop
            self.model.load_state_dict(original_state)
            inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

            # Inner loop: k gradient steps on task data
            task_loss_val = 0.0
            for _step in range(self.inner_steps):
                inner_optimizer.zero_grad()
                loss = self._task_loss(self.model, task)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                inner_optimizer.step()
                task_loss_val = loss.item()

            total_loss += task_loss_val
            n_tasks += 1

            # Outer loop: Reptile update
            # theta += meta_lr * (adapted_weights - original_weights)
            adapted_state = self.model.state_dict()
            updated_state = {}
            for key in original_state:
                updated_state[key] = original_state[key] + self.meta_lr * (
                    adapted_state[key] - original_state[key]
                )

            # Accumulate updates (for multi-task, average the directions)
            original_state = updated_state

        # Load the final meta-updated weights
        self.model.load_state_dict(original_state)

        return total_loss / max(n_tasks, 1)

    def meta_train(
        self,
        task_sampler: Any,
        n_iterations: int = 1000,
        tasks_per_iter: int = 4,
        log_interval: int = 50,
    ) -> list[float]:
        """Run full Reptile meta-training loop.

        Parameters
        ----------
        task_sampler : callable
            Function that returns a list of FewShotTask when called.
            ``task_sampler()`` -> list[FewShotTask].
        n_iterations : int
            Number of meta-iterations.
        tasks_per_iter : int
            Number of tasks per meta-iteration.
        log_interval : int
            Log every N iterations.

        Returns
        -------
        List of average task losses per iteration.
        """
        losses: list[float] = []

        for iteration in range(n_iterations):
            tasks = task_sampler()
            if len(tasks) > tasks_per_iter:
                tasks = tasks[:tasks_per_iter]

            avg_loss = self.meta_train_step(tasks)
            losses.append(avg_loss)

            if (iteration + 1) % log_interval == 0:
                recent_avg = sum(losses[-log_interval:]) / min(log_interval, len(losses))
                logger.info(
                    "Meta-iter %d/%d: avg_loss=%.6f (recent=%.6f)",
                    iteration + 1,
                    n_iterations,
                    avg_loss,
                    recent_avg,
                )

        return losses


def adapt(
    model: nn.Module,
    observations: list[tuple[float, float]],
    smiles: str = "",
    graph: Any = None,
    covariates: dict[str, Any] | None = None,
    regimen: dict[str, Any] | None = None,
    n_steps: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    """Adapt a pre-trained model to a specific drug using few observations.

    Creates a copy of the model and fine-tunes it on the observed data.

    Parameters
    ----------
    model : nn.Module
        Pre-trained PKFoundationModel or SMILESToPKModel.
    observations : list of (time_h, conc_mg_L) tuples
        Observed concentration-time data (1-5 points recommended).
    smiles : str
        SMILES string (needed if graph not provided).
    graph : Any
        Pre-computed molecular graph.
    covariates : dict or None
        Patient covariates.
    regimen : dict or None
        Dosing regimen.
    n_steps : int
        Number of gradient steps for adaptation.
    lr : float
        Learning rate for adaptation.
    device : str
        Device.

    Returns
    -------
    Adapted model (deep copy of original).
    """
    adapted_model = copy.deepcopy(model)
    adapted_model.to(device)
    adapted_model.train()

    task = FewShotTask(
        smiles=smiles,
        observations=observations,
        graph=graph,
        covariates=covariates,
        regimen=regimen,
    )

    trainer = ReptileTrainer(
        adapted_model,
        inner_lr=lr,
        inner_steps=1,
        device=device,
    )

    optimizer = torch.optim.Adam(adapted_model.parameters(), lr=lr)
    for _step in range(n_steps):
        optimizer.zero_grad()
        loss = trainer._task_loss(adapted_model, task)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
        optimizer.step()

    adapted_model.eval()
    return adapted_model


__all__ = ["FewShotTask", "ReptileTrainer", "adapt"]
