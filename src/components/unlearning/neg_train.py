"""Negative training baseline: gradient ascent on forget set."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from src.components.unlearning.optim_utils import build_optimizer
from src.components.unlearning.target_params import resolve_scope_params
from src.components.unlearning.hvp import batch_size, batch_to_device

log = logging.getLogger(__name__)
TigerBatch = Any


def neg_train_unlearn(
    model: nn.Module,
    forget_batches: Sequence[TigerBatch],
    retain_batches: Sequence[TigerBatch],
    *,
    steps: int = 200,
    n_epochs: Optional[int] = None,
    lr: float = 1e-3,
    neg_retain_every: int = 5,
    update_scope: str = "all",
    pkm_update_keys: bool = True,
    pkm_update_query: bool = True,
    optimizer: str = "adam",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Gradient ascent on forget batches with optional retain CE every k steps.

    ``neg_retain_every=1`` runs only retain steps and reduces to fine-tuning.
    This is the alternating form (one optimizer step per batch); the
    simultaneous form ``L_retain - w*CE_forget`` is available through the
    unified objective with ``lambda_r=1, lambda_f=w, lambda_s=0, lambda_n=0``.
    """
    device = device or next(model.parameters()).device
    if not forget_batches:
        raise ValueError("forget_batches is empty")
    params, _ = resolve_scope_params(
        model, update_scope,
        fallback=[p for p in model.parameters() if p.requires_grad],
        include_keys=pkm_update_keys, include_query=pkm_update_query,
        algo="neg_train",
    )
    opt = build_optimizer(optimizer, params, float(lr), algo="neg_train")
    model.train()
    forget_losses: List[float] = []
    retain_losses: List[float] = []
    # Step budget: n_epochs * min(n_forget, n_retain); n_epochs wins over steps.
    if n_epochs is not None and forget_batches and retain_batches:
        steps = int(n_epochs) * min(len(forget_batches), len(retain_batches))
    for step in range(int(steps)):
        if neg_retain_every > 0 and step % int(neg_retain_every) == 0 and retain_batches:
            batch = retain_batches[step % len(retain_batches)]
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            _, loss = model.model_step(*batch)
            loss.backward()
            opt.step()
            retain_losses.append(float(loss.detach().cpu()))
        else:
            batch = forget_batches[step % len(forget_batches)]
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            _, loss = model.model_step(*batch)
            (-loss).backward()
            opt.step()
            forget_losses.append(float(loss.detach().cpu()))
        if step % max(1, steps // 10) == 0:
            log.info("[neg_train] step=%d", step)
    return {
        "algorithm": "neg_train",
        "steps": int(steps),
        "lr": float(lr),
        "neg_retain_every": int(neg_retain_every),
        "mean_forget_loss": (
            float(sum(forget_losses) / max(1, len(forget_losses)))
            if forget_losses
            else None
        ),
        "mean_retain_loss": (
            float(sum(retain_losses) / max(1, len(retain_losses)))
            if retain_losses
            else None
        ),
        "n_forget_batches": len(forget_batches),
        "n_retain_batches": len(retain_batches),
    }
