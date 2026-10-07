"""Hessian-vector product utilities and a stochastic Conjugate Gradient solver,
ported from the ERASE benchmark's RecBole trainer
(https://github.com/deem-data/erase-bench/blob/main/recbole/trainer/trainer.py;
``_batch_grad`` /
``_hvp_single`` / ``_hvp_dataset`` / ``cg_inv_hvp``) and adapted to TIGER's
batch shape ``(SequentialModelInputData, SequentialModuleLabelData)`` and its
``model.model_step(...)`` loss.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

if TYPE_CHECKING:
    from src.data.loading.components.interfaces import (
        SequentialModelInputData,
        SequentialModuleLabelData,
    )


log = logging.getLogger(__name__)


# A TIGER batch is the (model_input, label_data) tuple produced by
# `collate_with_sid_causal_duplicate` / `collate_fn_train`. A loose alias
# avoids a circular import of `src.data.loading.components.interfaces`.
TigerBatch = Tuple[Any, Any]


# ---------------------------------------------------------------------------
# TIGER-specific helpers
# ---------------------------------------------------------------------------


def batch_to_device(batch: "TigerBatch", device: torch.device) -> "TigerBatch":
    """Move every tensor inside a TIGER batch to ``device`` (in-place)."""
    model_input, label_data = batch
    if isinstance(model_input.mask, torch.Tensor):
        model_input.mask = model_input.mask.to(device, non_blocking=True)
    if isinstance(model_input.user_id_list, torch.Tensor):
        model_input.user_id_list = model_input.user_id_list.to(device, non_blocking=True)
    for k, v in list(model_input.transformed_sequences.items()):
        if isinstance(v, torch.Tensor):
            model_input.transformed_sequences[k] = v.to(device, non_blocking=True)
    if label_data is not None:
        for k, v in list(label_data.labels.items()):
            if isinstance(v, torch.Tensor):
                label_data.labels[k] = v.to(device, non_blocking=True)
        for k, v in list(label_data.label_location.items()):
            if isinstance(v, torch.Tensor):
                label_data.label_location[k] = v.to(device, non_blocking=True)
        for k, v in list(label_data.attention_mask.items()):
            if isinstance(v, torch.Tensor):
                label_data.attention_mask[k] = v.to(device, non_blocking=True)
    return batch


def batch_size(batch: "TigerBatch") -> int:
    """Return the number of rows in a TIGER batch (post-augmentation)."""
    model_input, _ = batch
    if isinstance(model_input.user_id_list, torch.Tensor):
        return int(model_input.user_id_list.shape[0])
    if isinstance(model_input.mask, torch.Tensor):
        return int(model_input.mask.shape[0])
    for v in model_input.transformed_sequences.values():
        if isinstance(v, torch.Tensor):
            return int(v.shape[0])
    return 0


def model_step_loss(
    model: nn.Module, batch: "TigerBatch"
) -> torch.Tensor:
    """Compute the scalar TIGER training loss on a batch via ``model.model_step``."""
    model_input, label_data = batch
    _, loss = model.model_step(model_input, label_data)
    return loss


# ---------------------------------------------------------------------------
# Gradients and Hessian-vector products (ERASE-equivalent)
# ---------------------------------------------------------------------------


def _zero_like(params: Sequence[nn.Parameter]) -> List[torch.Tensor]:
    return [torch.zeros_like(p) for p in params]


def batch_grad(
    model: nn.Module,
    batch: "TigerBatch",
    params: Sequence[nn.Parameter],
    average_scale: float,
    allow_unused: bool = True,
) -> List[torch.Tensor]:
    """Mirror of ERASE ``Trainer._batch_grad``.

    Computes ``∂loss/∂params`` for one batch, divided by ``average_scale``.
    Replaces ``None`` entries (parameters that didn't participate) with zeros
    of matching shape.
    """
    loss = model_step_loss(model, batch)
    grads = torch.autograd.grad(
        loss, params, allow_unused=allow_unused, create_graph=False
    )
    out: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros_like(p))
        else:
            out.append((g / average_scale).detach())
    return out


def hvp_single(
    model: nn.Module,
    batch: "TigerBatch",
    v_list: Sequence[torch.Tensor],
    params: Sequence[nn.Parameter],
    allow_unused: bool = True,
) -> List[torch.Tensor]:
    """Hessian-vector product on a single TIGER batch (Pearlmutter trick).

    Returns ``H v`` where ``H = ∂²loss/∂θ²`` evaluated on ``batch``.
    """
    loss = model_step_loss(model, batch)
    g_list = torch.autograd.grad(
        loss, params, allow_unused=allow_unused, create_graph=True
    )
    g_list = [
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(g_list, params)
    ]
    flat_g = sum((gi * vi).sum() for gi, vi in zip(g_list, v_list))
    h_list = torch.autograd.grad(
        flat_g, params, allow_unused=allow_unused, retain_graph=False
    )
    out: List[torch.Tensor] = []
    for h, p in zip(h_list, params):
        if h is None:
            out.append(torch.zeros_like(p))
        else:
            out.append(h.detach())
    return out


def hvp_dataset(
    model: nn.Module,
    batches: Iterable["TigerBatch"],
    v_list: Sequence[torch.Tensor],
    params: Sequence[nn.Parameter],
    average: bool = True,
) -> List[torch.Tensor]:
    """Average HVP across all batches (used as a sanity-check / standalone)."""
    acc = _zero_like(params)
    n = 0
    for batch in batches:
        h = hvp_single(model, batch, v_list, params)
        acc = [ai + hi for ai, hi in zip(acc, h)]
        n += 1
    if n > 0 and average:
        acc = [ai / n for ai in acc]
    return acc


# ---------------------------------------------------------------------------
# Stochastic Conjugate-Gradient solver for (H + damping * I) x = v
# ---------------------------------------------------------------------------


def _flat_dot(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> float:
    return float(sum((ai * bi).sum() for ai, bi in zip(a, b)).item())


def _flat_norm(a: Sequence[torch.Tensor]) -> float:
    return math.sqrt(_flat_dot(a, a))


def flat_norm(vec: Sequence[torch.Tensor]) -> float:
    """Joint Euclidean norm of a list of tensors (one concatenated vector)."""
    return _flat_norm(vec)


def clip_vector_to_max_norm(
    vec: Sequence[torch.Tensor], max_norm: float
) -> List[torch.Tensor]:
    """Scale down ``vec`` so its joint L2 norm is at most ``max_norm``."""
    return _clip_norm(vec, max_norm)


def _clip_norm(
    vec: Sequence[torch.Tensor], max_norm: float
) -> List[torch.Tensor]:
    norm = _flat_norm(vec)
    if norm <= max_norm or norm == 0.0:
        return list(vec)
    scale = max_norm / norm
    return [v * scale for v in vec]


def _has_nan_or_inf(vec: Sequence[torch.Tensor]) -> bool:
    for v in vec:
        if torch.isnan(v).any() or torch.isinf(v).any():
            return True
    return False


def cg_inv_hvp(
    model: nn.Module,
    hvp_batches: Sequence["TigerBatch"],
    v_list: Sequence[torch.Tensor],
    params: Sequence[nn.Parameter],
    damping: float = 0.01,
    max_iter: int = 200,
    tol: float = 1e-5,
    max_norm: Optional[float] = None,
    log_every: int = 25,
) -> Tuple[List[torch.Tensor], dict]:
    """Stochastic CG to approximately solve ``(H + damping * I) x = v``.

    Each CG iteration draws one batch from ``hvp_batches`` (cycling if it has
    fewer batches than ``max_iter``) and evaluates the HVP on that batch only.
    This matches ERASE, which does one HVP per CG step over the retain (and
    clean-forget) corpus.

    Parameters
    ----------
    model
        The Lightning-wrapped TIGER model whose ``.model_step`` returns the
        loss.
    hvp_batches
        Pre-collected list of TIGER batches (already on the right device) used
        to estimate ``H``. Must be non-empty.
    v_list
        Right-hand side of the linear system, one tensor per parameter.
    params
        The parameters ``H`` is taken w.r.t. (must match ``v_list``).
    damping
        Tikhonov regularizer on ``H``.
    max_iter, tol
        Standard CG knobs. ``tol`` is on the squared residual norm.
    max_norm
        If set, the returned ``x`` is ``L2``-clipped to this norm.
    log_every
        Print residuals every ``log_every`` iterations.
    """
    if not hvp_batches:
        raise ValueError("cg_inv_hvp requires at least one HVP batch")
    if _has_nan_or_inf(v_list):
        log.warning("cg_inv_hvp: rhs contains NaN/Inf; aborting")
        return [torch.zeros_like(p) for p in params], {
            "converged": False,
            "iters": 0,
            "rsq_final": float("nan"),
            "abort_reason": "rhs_nan_or_inf",
        }

    x = [torch.zeros_like(p) for p in params]
    r = [vi.detach().clone() for vi in v_list]
    p_dir = [ri.clone() for ri in r]
    rsq_old = _flat_dot(r, r)
    rsq_init = rsq_old
    abort_reason: Optional[str] = None

    iters = 0
    for k in range(max_iter):
        iters = k + 1
        batch = hvp_batches[k % len(hvp_batches)]
        Hp = hvp_single(model, batch, p_dir, params)
        if _has_nan_or_inf(Hp):
            abort_reason = "hvp_nan_or_inf"
            log.warning(
                "cg_inv_hvp: HVP produced NaN/Inf at iter=%d; aborting and "
                "returning the current x",
                k,
            )
            break
        Hp_damped = [hi + damping * pi for hi, pi in zip(Hp, p_dir)]

        pHp = _flat_dot(p_dir, Hp_damped)
        if not math.isfinite(pHp) or abs(pHp) < 1e-30:
            abort_reason = "pHp_degenerate"
            break
        alpha = rsq_old / pHp

        x = [xi + alpha * pi for xi, pi in zip(x, p_dir)]
        r = [ri - alpha * hi for ri, hi in zip(r, Hp_damped)]
        rsq_new = _flat_dot(r, r)

        if log_every > 0 and (k % log_every == 0):
            log.info(
                "cg_inv_hvp iter=%d alpha=%.3e rsq=%.3e (init=%.3e)",
                k,
                alpha,
                rsq_new,
                rsq_init,
            )

        if rsq_new < tol:
            rsq_old = rsq_new
            break
        beta = rsq_new / max(rsq_old, 1e-30)
        p_dir = [ri + beta * pi for ri, pi in zip(r, p_dir)]
        rsq_old = rsq_new

    if max_norm is not None:
        x = _clip_norm(x, max_norm)

    info = {
        "converged": rsq_old < tol,
        "iters": iters,
        "rsq_init": rsq_init,
        "rsq_final": rsq_old,
        "abort_reason": abort_reason,
    }
    return x, info
