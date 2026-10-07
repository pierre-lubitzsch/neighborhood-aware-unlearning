"""Fine-tune baseline: continue training on retain (cleaned) data."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from src.components.unlearning.optim_utils import build_optimizer
from src.components.unlearning.hvp import batch_size, batch_to_device
from src.components.unlearning.target_params import resolve_scope_params

log = logging.getLogger(__name__)
TigerBatch = Any


def finetune_unlearn(
    model: nn.Module,
    retain_batches: Sequence[TigerBatch],
    *,
    steps: int = 500,
    n_epochs: Optional[int] = None,
    lr: float = 1e-3,
    update_scope: str = "all",
    pkm_update_keys: bool = True,
    pkm_update_query: bool = True,
    optimizer: str = "adam",
    patience: int = 0,
    min_delta: float = 0.0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Fine-tune ``model`` on retain batches for ``steps`` optimizer steps.

    ``update_scope='pkm_only'`` restricts the update to the Product-Key Memory
    and freezes the backbone. ``steps=0`` performs no update, which measures the
    effect of installing PKM alone.

    ``patience>0`` enables early stopping on the retain loss: training stops
    after ``patience`` consecutive steps without improving the best loss by more
    than ``min_delta``.

    ``optimizer='sgd'`` follows Sparse Memory Finetuning, where Adam's moment
    estimates are diluted on memory slots that rarely receive gradient.
    """
    device = device or next(model.parameters()).device
    params, _ = resolve_scope_params(
        model, update_scope,
        fallback=[p for p in model.parameters() if p.requires_grad],
        include_keys=bool(pkm_update_keys),
        include_query=bool(pkm_update_query),
        algo="finetune",
    )
    opt = build_optimizer(optimizer, params, float(lr), algo="finetune")
    model.train()
    losses: List[float] = []
    # n_epochs full passes over the retain batches; overrides steps when given.
    if n_epochs is not None and retain_batches:
        steps = int(n_epochs) * len(retain_batches)
    if int(steps) > 0 and not retain_batches:
        raise ValueError("retain_batches is empty")
    best = float("inf")
    since_best = 0
    stopped_at = None
    for step in range(int(steps)):
        batch = retain_batches[step % len(retain_batches)]
        batch = batch_to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        _, loss = model.model_step(*batch)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if step % max(1, steps // 10) == 0:
            log.info("[finetune] step=%d loss=%.4f", step, losses[-1])
        if int(patience) > 0:
            if losses[-1] < best - float(min_delta):
                best = losses[-1]
                since_best = 0
            else:
                since_best += 1
                if since_best >= int(patience):
                    stopped_at = step + 1
                    log.info(
                        "[finetune] early stop at step=%d/%d "
                        "(no retain-loss improvement > %.2g for %d steps; best=%.4f)",
                        stopped_at, int(steps), float(min_delta), int(patience), best,
                    )
                    break
    return {
        "algorithm": "finetune",
        "steps": int(steps),
        "lr": float(lr),
        "update_scope": str(update_scope),
        "optimizer": str(optimizer),
        "n_updated_params": int(sum(p.numel() for p in params)),
        "steps_run": len(losses),
        "early_stopped_at": stopped_at,
        "patience": int(patience),
        "best_loss": None if best == float("inf") else best,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": float(sum(losses) / max(1, len(losses))),
        "n_retain_batches": len(retain_batches),
        "n_retain_rows": sum(batch_size(b) for b in retain_batches),
    }
