"""Fanchuan unlearning, ported from the ERASE benchmark and adapted to TIGER
``(SequentialModelInputData, SequentialModuleLabelData)`` batches.

Stage 1 (uniform pseudolabel learning): on each forget batch, push the
next-token distribution toward uniform (``model.compute_uniform_kl_loss``).

Stage 2 (repeated ``contrastive_iters`` times): (a) push forget user
representations away from retain representations with the contrastive loss
``mean(-log_softmax(z_f @ z_r^T / t))``; (b) fine-tune on the retain set to
recover utility.

A single optimizer is shared across all stages, as in ERASE. Only the
forget-vs-retain contrastive term is used, since there is no clean-forget set.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from src.components.unlearning.optim_utils import build_optimizer
from src.components.unlearning.target_params import resolve_scope_params
from src.components.unlearning.hvp import batch_size, batch_to_device

log = logging.getLogger(__name__)
TigerBatch = Any  # noqa: N816


def _contrastive_loss(
    forget_repr: torch.Tensor,
    retain_repr: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Contrastive loss between representations ``[B_f, d]`` and ``[B_r, d]``."""
    sim = forget_repr @ retain_repr.t() / float(temperature)
    return (-1.0 * F.log_softmax(sim, dim=-1)).mean()


def fanchuan_unlearn(
    model: nn.Module,
    forget_batches: Sequence["TigerBatch"],
    retain_batches: Sequence["TigerBatch"],
    *,
    lr: float = 1e-3,
    uniform_epochs: int = 1,
    contrastive_iters: int = 8,
    contrastive_temperature: float = 1.15,
    retain_epochs_per_iter: int = 1,
    seed: int = 2,
    update_scope: str = "all",
    pkm_update_keys: bool = True,
    pkm_update_query: bool = True,
    optimizer: str = "adam",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run Fanchuan two-stage unlearning on ``model`` in-place.

    Parameters
    ----------
    model
        TIGER model exposing ``compute_uniform_kl_loss(input, label_data)``,
        ``_pooled_user_representation(input)`` and
        ``model_step(input, label_data) -> (output, loss)``.
    forget_batches, retain_batches
        Pre-collected TIGER batches already on ``device``.
    lr
        Learning rate for the shared optimizer (all stages).
    uniform_epochs
        Passes over the forget set in stage 1.
    contrastive_iters
        Stage-2 outer iterations (ERASE ``unlearn_iters_contrastive``).
    contrastive_temperature
        Temperature ``t`` in the contrastive loss (ERASE uses 1.15).
    retain_epochs_per_iter
        Retain-repair passes after each contrastive iteration.
    seed
        Seeds the retain-batch shuffling used to pair forget/retain batches.
    """
    if not forget_batches:
        raise ValueError("fanchuan_unlearn: no forget batches were provided")
    if not retain_batches:
        raise ValueError("fanchuan_unlearn: no retain batches were provided")
    for attr in ("compute_uniform_kl_loss", "_pooled_user_representation"):
        if not hasattr(model, attr):
            raise TypeError(
                f"model must expose {attr} (SemanticIDEncoderDecoder subclass)"
            )

    device = device or next(model.parameters()).device
    model.train()
    params, _ = resolve_scope_params(
        model, update_scope,
        fallback=[p for p in model.parameters() if p.requires_grad],
        include_keys=pkm_update_keys, include_query=pkm_update_query,
        algo="fanchuan",
    )
    opt = build_optimizer(optimizer, params, float(lr), algo="fanchuan")
    rng = random.Random(seed)

    log.info(
        "[fanchuan] lr=%.3g uniform_epochs=%d contrastive_iters=%d t=%.3g "
        "retain_epochs_per_iter=%d (%d forget / %d retain batches)",
        lr,
        uniform_epochs,
        contrastive_iters,
        contrastive_temperature,
        retain_epochs_per_iter,
        len(forget_batches),
        len(retain_batches),
    )

    # --- Stage 1: uniform pseudolabel learning -------------------------------
    uniform_losses: List[float] = []
    for _ in range(int(uniform_epochs)):
        for batch in forget_batches:
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = model.compute_uniform_kl_loss(*batch)
            loss.backward()
            opt.step()
            uniform_losses.append(float(loss.detach().cpu()))
    if uniform_losses:
        log.info(
            "[fanchuan] stage 1 (uniform) mean_loss=%.4f over %d steps",
            sum(uniform_losses) / len(uniform_losses),
            len(uniform_losses),
        )

    # --- Stage 2: contrastive + retain repair --------------------------------
    contrastive_losses: List[float] = []
    repair_losses: List[float] = []
    for it in range(int(contrastive_iters)):
        order = list(range(len(retain_batches)))
        rng.shuffle(order)

        for b_idx, forget_batch in enumerate(forget_batches):
            forget_batch = batch_to_device(forget_batch, device)
            retain_batch = batch_to_device(
                retain_batches[order[b_idx % len(order)]], device
            )
            opt.zero_grad(set_to_none=True)
            f_repr = model._pooled_user_representation(forget_batch[0])
            r_repr = model._pooled_user_representation(retain_batch[0])
            loss = _contrastive_loss(f_repr, r_repr, contrastive_temperature)
            loss.backward()
            opt.step()
            contrastive_losses.append(float(loss.detach().cpu()))

        # retain-repair round
        for _ in range(int(retain_epochs_per_iter)):
            for retain_batch in retain_batches:
                retain_batch = batch_to_device(retain_batch, device)
                opt.zero_grad(set_to_none=True)
                _, loss = model.model_step(*retain_batch)
                loss.backward()
                opt.step()
                repair_losses.append(float(loss.detach().cpu()))

        log.info(
            "[fanchuan] iter %d/%d contrastive=%.4f repair=%.4f",
            it + 1,
            int(contrastive_iters),
            (
                sum(contrastive_losses[-len(forget_batches):])
                / max(1, len(forget_batches))
            ),
            (
                sum(repair_losses[-len(retain_batches) * max(1, retain_epochs_per_iter):])
                / max(1, len(retain_batches) * max(1, retain_epochs_per_iter))
            )
            if repair_losses
            else float("nan"),
        )

    def _mean(xs: List[float]) -> Optional[float]:
        return float(sum(xs) / len(xs)) if xs else None

    return {
        "algorithm": "fanchuan",
        "lr": float(lr),
        "uniform_epochs": int(uniform_epochs),
        "contrastive_iters": int(contrastive_iters),
        "contrastive_temperature": float(contrastive_temperature),
        "retain_epochs_per_iter": int(retain_epochs_per_iter),
        "mean_uniform_loss": _mean(uniform_losses),
        "mean_contrastive_loss": _mean(contrastive_losses),
        "mean_repair_loss": _mean(repair_losses),
        "n_forget_batches": len(forget_batches),
        "n_retain_batches": len(retain_batches),
        "n_forget_rows": sum(batch_size(b) for b in forget_batches),
        "n_retain_rows": sum(batch_size(b) for b in retain_batches),
    }
