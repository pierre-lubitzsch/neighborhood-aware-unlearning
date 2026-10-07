"""Optional local distribution repair losses, gated by config."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import torch
from torch import nn


def apply_local_repair_losses(
    model: nn.Module,
    *,
    base_loss: torch.Tensor,
    local_repair_cfg: Dict[str, Any],
    neighbor_item_ids: Optional[Set[int]] = None,
    batch: Any = None,
) -> torch.Tensor:
    """Add optional repair terms when ``local_repair.enabled`` is true."""
    if not local_repair_cfg or not local_repair_cfg.get("enabled", False):
        return base_loss

    loss = base_loss
    gamma = float(local_repair_cfg.get("gamma", 1.0))

    if local_repair_cfg.get("logit_suppression") and neighbor_item_ids:
        if hasattr(model, "compute_neighbor_suppression_loss"):
            loss = loss + gamma * model.compute_neighbor_suppression_loss(
                batch, neighbor_item_ids
            )

    if local_repair_cfg.get("adapter_repair"):
        if hasattr(model, "repair_adapter") and model.repair_adapter is not None:
            # Adapter params trained jointly when enabled.
            pass

    if local_repair_cfg.get("prefix_repair"):
        if hasattr(model, "compute_prefix_repair_loss"):
            loss = loss + model.compute_prefix_repair_loss(batch, neighbor_item_ids)

    if local_repair_cfg.get("mass_regularization"):
        if hasattr(model, "compute_neighborhood_mass_loss"):
            loss = loss + model.compute_neighborhood_mass_loss(
                batch, neighbor_item_ids
            )

    return loss
