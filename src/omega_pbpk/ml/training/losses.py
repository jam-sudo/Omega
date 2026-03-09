"""Loss functions for PK prediction — physics-informed, multi-component.

PKLoss combines data-driven MSE with physics-based constraints:
1. PK metric MSE (Cmax, AUC, Tmax, t1/2)
2. Mass conservation penalty
3. Non-negativity penalty for concentrations
4. Monotonic terminal phase constraint
5. Parameter plausibility bounds
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PKLoss(nn.Module):
    """Physics-informed loss for PK parameter prediction.

    All component weights are stored as buffers (not learnable) and can be
    adjusted via the constructor or by direct assignment.

    Parameters
    ----------
    w_pk_mse : float
        Weight for PK metric MSE loss. Default: 1.0.
    w_mass : float
        Weight for mass conservation penalty. Default: 0.1.
    w_nonneg : float
        Weight for non-negativity penalty. Default: 1.0.
    w_monotone : float
        Weight for monotonic terminal phase penalty. Default: 0.05.
    w_plausible : float
        Weight for parameter plausibility penalty. Default: 0.1.
    """

    # Physiological parameter ranges for plausibility checks
    PARAM_RANGES: dict[str, tuple[float, float]] = {
        "clint_3a4": (0.001, 1000.0),
        "clint_2d6": (0.001, 1000.0),
        "fup": (0.001, 1.0),
        "rbp": (0.3, 3.0),
        "vd": (0.1, 10000.0),
    }

    def __init__(
        self,
        w_pk_mse: float = 1.0,
        w_mass: float = 0.1,
        w_nonneg: float = 1.0,
        w_monotone: float = 0.05,
        w_plausible: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("w_pk_mse", torch.tensor(w_pk_mse))
        self.register_buffer("w_mass", torch.tensor(w_mass))
        self.register_buffer("w_nonneg", torch.tensor(w_nonneg))
        self.register_buffer("w_monotone", torch.tensor(w_monotone))
        self.register_buffer("w_plausible", torch.tensor(w_plausible))

    def pk_metric_mse(
        self,
        predicted_pk: dict[str, torch.Tensor],
        true_pk: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """MSE on PK metrics in log-space for scale invariance.

        Concentration metrics (Cmax, AUC) use log1p-space MSE to handle
        the wide dynamic range (0.001 to 100000 mg/L). Time metrics
        (Tmax, t_half) use log1p-space MSE for similar reasons.

        Parameters
        ----------
        predicted_pk : dict with keys like 'cmax', 'auc', 'tmax', 't_half'
        true_pk : dict with same keys

        Returns
        -------
        Scalar tensor.
        """
        loss = torch.tensor(0.0, device=_get_device(predicted_pk))
        count = 0
        for key in ("cmax", "auc", "tmax", "t_half"):
            if key in predicted_pk and key in true_pk:
                pred = predicted_pk[key].clamp(min=0.0)
                true = true_pk[key].clamp(min=0.0)
                # Log1p-space MSE for scale invariance
                loss = loss + F.mse_loss(torch.log1p(pred), torch.log1p(true))
                count += 1
        if count > 0:
            loss = loss / count
        return loss

    def mass_conservation(
        self,
        tissue_amounts: torch.Tensor,
        eliminated: torch.Tensor,
        dose: torch.Tensor,
    ) -> torch.Tensor:
        """Mass conservation penalty.

        |sum(tissue_amounts) + eliminated - dose| / dose should -> 0

        Parameters
        ----------
        tissue_amounts : Tensor (batch, n_tissues)
            Amount of drug in each tissue at current time.
        eliminated : Tensor (batch,)
            Cumulative eliminated amount.
        dose : Tensor (batch,)
            Administered dose.

        Returns
        -------
        Scalar tensor.
        """
        total_in_body = tissue_amounts.sum(dim=-1) + eliminated
        relative_error = torch.abs(total_in_body - dose) / dose.clamp(min=1e-8)
        return relative_error.mean()

    def non_negativity(self, concentrations: torch.Tensor) -> torch.Tensor:
        """Penalty for negative concentrations.

        sum(relu(-C(t))) across all timepoints and compartments.

        Parameters
        ----------
        concentrations : Tensor (batch, time, [compartments])

        Returns
        -------
        Scalar tensor.
        """
        return F.relu(-concentrations).sum(dim=-1).mean()

    def monotonic_terminal(
        self,
        curve: torch.Tensor,
        tmax_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Penalty for non-monotonically decreasing terminal phase.

        After Tmax, C(t+1) should be <= C(t). Penalizes violations.

        Parameters
        ----------
        curve : Tensor (batch, time)
            Concentration-time curve.
        tmax_idx : Tensor (batch,), optional
            Index of Tmax for each sample. If None, uses argmax of curve.

        Returns
        -------
        Scalar tensor.
        """
        batch_size, n_time = curve.shape
        if n_time < 2:
            return torch.tensor(0.0, device=curve.device)

        if tmax_idx is None:
            tmax_idx = curve.argmax(dim=-1)

        # Compute differences: C(t+1) - C(t)
        diffs = curve[:, 1:] - curve[:, :-1]  # (batch, time-1)

        # Create mask: True for timepoints after Tmax
        time_indices = torch.arange(n_time - 1, device=curve.device).unsqueeze(0)  # (1, time-1)
        mask = time_indices >= tmax_idx.unsqueeze(1)  # (batch, time-1)

        # Penalize increases after Tmax
        violations = F.relu(diffs) * mask.float()
        return violations.sum(dim=-1).mean()

    def parameter_plausibility(
        self,
        predicted_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Penalty for parameters outside physiological ranges.

        Parameters
        ----------
        predicted_params : dict[str, Tensor (batch,)]
            Predicted PK parameters.

        Returns
        -------
        Scalar tensor.
        """
        device = _get_device(predicted_params)
        penalty = torch.tensor(0.0, device=device)
        count = 0

        for name, (lo, hi) in self.PARAM_RANGES.items():
            if name in predicted_params:
                val = predicted_params[name]
                # Penalty for values below lower bound
                penalty = penalty + F.relu(lo - val).mean()
                # Penalty for values above upper bound
                penalty = penalty + F.relu(val - hi).mean()
                count += 1

        if count > 0:
            penalty = penalty / count
        return penalty

    def forward(
        self,
        predicted_pk: dict[str, torch.Tensor],
        true_pk: dict[str, torch.Tensor],
        predicted_params: dict[str, torch.Tensor],
        predicted_curve: torch.Tensor | None = None,
        tissue_amounts: torch.Tensor | None = None,
        eliminated: torch.Tensor | None = None,
        dose: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute total loss and individual components.

        Parameters
        ----------
        predicted_pk : dict with PK metrics (cmax, auc, tmax, t_half)
        true_pk : dict with ground truth PK metrics
        predicted_params : dict with predicted PK parameters
        predicted_curve : Tensor (batch, time), optional
            Predicted concentration-time curve for physics constraints.
        tissue_amounts : Tensor (batch, n_tissues), optional
        eliminated : Tensor (batch,), optional
        dose : Tensor (batch,), optional

        Returns
        -------
        dict with 'total_loss' and individual component losses.
        """
        device = _get_device(predicted_pk) if predicted_pk else _get_device(predicted_params)
        losses: dict[str, torch.Tensor] = {}

        # 1. PK metric MSE
        losses["pk_mse"] = self.pk_metric_mse(predicted_pk, true_pk)

        # 2. Mass conservation (only if data provided)
        if tissue_amounts is not None and eliminated is not None and dose is not None:
            losses["mass_conservation"] = self.mass_conservation(tissue_amounts, eliminated, dose)
        else:
            losses["mass_conservation"] = torch.tensor(0.0, device=device)

        # 3. Non-negativity (only if curve provided)
        if predicted_curve is not None:
            losses["non_negativity"] = self.non_negativity(predicted_curve)
        else:
            losses["non_negativity"] = torch.tensor(0.0, device=device)

        # 4. Monotonic terminal phase (only if curve provided)
        if predicted_curve is not None:
            losses["monotonic_terminal"] = self.monotonic_terminal(predicted_curve)
        else:
            losses["monotonic_terminal"] = torch.tensor(0.0, device=device)

        # 5. Parameter plausibility
        losses["param_plausibility"] = self.parameter_plausibility(predicted_params)

        # Total weighted loss
        total = (
            self.w_pk_mse * losses["pk_mse"]
            + self.w_mass * losses["mass_conservation"]
            + self.w_nonneg * losses["non_negativity"]
            + self.w_monotone * losses["monotonic_terminal"]
            + self.w_plausible * losses["param_plausibility"]
        )
        losses["total_loss"] = total

        return losses


def _get_device(d: dict[str, torch.Tensor]) -> torch.device:
    """Get device from first tensor in a dict."""
    for v in d.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")
