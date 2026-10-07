"""SCIF (Single-shot Conjugate Influence Function) unlearning, ported from the
ERASE benchmark and adapted to TIGER batches and ``model.model_step(...)``.

Update rule::

    retain_count := retain_samples_used_for_update * |D_f|
    neg_grads  := -1/retain_count * sum_{b in D_f}      dL_b / dtheta
    pos_grads  := +1/retain_count * sum_{b in D_retain} dL_b / dtheta
    grads      := neg_grads + pos_grads
    x          := (H + lambda I)^{-1} grads   # CG, HVP from retain batches
    theta      := theta - (1 / |D_retain_full|) * x

Lightning, dataloaders and checkpointing are handled by ``TigerUnlearningModule``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn

from src.components.unlearning.hvp import (
    batch_grad,
    batch_size,
    batch_to_device,
    cg_inv_hvp,
    clip_vector_to_max_norm,
    flat_norm,
)
from src.components.unlearning.target_params import (
    select_target_params,
)


# A TIGER batch is `(SequentialModelInputData, SequentialModuleLabelData)`.
TigerBatch = Any  # noqa: N816


log = logging.getLogger(__name__)


def _materialize_batches(
    iterable: Iterable["TigerBatch"],
    device: torch.device,
    max_rows: Optional[int] = None,
) -> List["TigerBatch"]:
    """Pull batches out of a (possibly streaming) dataloader onto ``device``.

    SCIF iterates the retain batches twice (gradient pass and CG).
    ``max_rows`` caps the total number of post-collate rows.
    """
    out: List[TigerBatch] = []
    rows = 0
    for batch in iterable:
        batch = batch_to_device(batch, device)
        out.append(batch)
        rows += batch_size(batch)
        if max_rows is not None and rows >= max_rows:
            break
    return out


def _accumulate_grad(
    model: nn.Module,
    batches: Sequence["TigerBatch"],
    params: Sequence[nn.Parameter],
    average_scale: float,
    sign: float = +1.0,
) -> List[torch.Tensor]:
    acc = [torch.zeros_like(p) for p in params]
    for batch in batches:
        g = batch_grad(model, batch, params, average_scale=average_scale)
        acc = [a + sign * gi for a, gi in zip(acc, g)]
    return acc


def scif_unlearn(
    model: nn.Module,
    forget_batches: Sequence["TigerBatch"],
    retain_batches: Sequence["TigerBatch"],
    *,
    forget_size: int,
    retain_size: int,
    retain_samples_used_for_update: Optional[int] = None,
    cg_max_iter: int = 200,
    cg_tol: float = 1e-5,
    cg_damping: float = 0.01,
    target_params_policy: str = "tiger",
    params: Optional[Sequence[nn.Parameter]] = None,
    grad_masks: Optional[Dict[int, torch.Tensor]] = None,
    cg_solution_max_norm: Optional[float] = None,
    update_max_norm: Optional[float] = 1.0,
    eval_mode: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Run one SCIF parameter update on ``model`` in-place.

    Parameters
    ----------
    model
        TIGER ``SemanticIDEncoderDecoder`` (or wrapping ``LightningModule``);
        must expose ``model_step(input, label_data) -> (output, loss)``.
    forget_batches, retain_batches
        Lists of pre-collected TIGER batches already on ``device``. Use the
        helper :func:`_materialize_batches` (or the unlearning Lightning
        module) to populate these from existing GRID dataloaders.
    forget_size
        ``|D_f|``, number of forget rows before collate augmentation. Used
        only to compute ``retain_count``, which scales both gradient passes.
    retain_size
        ``|D_retain_full|``, denominator in ``tau = 1/retain_size``. Row count
        of the full retain corpus, not the subset used for gradients and HVP.
    retain_samples_used_for_update
        Multiplier on ``forget_size`` for the retain gradient pass
        (``None`` means 16).
    cg_max_iter, cg_tol, cg_damping
        Conjugate-Gradient knobs (see :func:`cg_inv_hvp`).
    target_params_policy
        ``"all"`` / ``"sid_embeddings"`` / ``"encoder_only"``. See
        :func:`select_target_params`. Ignored when ``params`` is given.
    params
        Optional precomputed list of parameters to update. When ``None``, the
        set is resolved from ``target_params_policy``.
    grad_masks
        Optional ``id(param) -> float mask`` map. The update ``-tau * x`` of
        each listed param is multiplied by its mask before norm clipping.
        Params absent from the map are updated in full.
    cg_solution_max_norm
        If set, the CG solution ``x`` is joint L2-clipped before scaling by
        ``tau``.
    update_max_norm
        Joint L2 cap on the actual parameter update ``-tau * x``. ``None``
        disables clipping. Default ``1.0``.
    eval_mode
        If True (default), run in ``model.eval()`` so dropout is disabled and
        the Hessian used by CG is deterministic.
    device
        Defaults to ``next(model.parameters()).device``.

    Returns
    -------
    info : dict
        Diagnostics: sizes, CG residuals, NaN-guard counts and ``tau``.
    """
    if forget_size <= 0:
        raise ValueError(f"forget_size must be > 0, got {forget_size}")
    if retain_size <= 0:
        raise ValueError(f"retain_size must be > 0, got {retain_size}")
    if not forget_batches:
        raise ValueError("scif_unlearn: no forget batches were provided")
    if not retain_batches:
        raise ValueError("scif_unlearn: no retain batches were provided")

    if retain_samples_used_for_update is None:
        retain_samples_used_for_update = 16
    retain_count = retain_samples_used_for_update * forget_size
    if retain_count <= 0:
        raise ValueError(
            f"retain_count={retain_count} non-positive (forget_size={forget_size}, "
            f"retain_samples_used_for_update={retain_samples_used_for_update})"
        )

    device = device or next(model.parameters()).device
    if eval_mode:
        model.eval()
    else:
        model.train()

    if params is None:
        params = select_target_params(model, policy=target_params_policy)
    else:
        params = list(params)
    grad_masks = grad_masks or {}
    log.info(
        "[scif] policy=%s touches %d tensors with %d total params (%d grad-masked)",
        target_params_policy if grad_masks is None or not grad_masks else "custom",
        len(params),
        int(sum(p.numel() for p in params)),
        len(grad_masks),
    )

    # --- 1. Forget pass: subtract per-batch grads, averaged over retain_count
    log.info(
        "[scif] forget pass over %d batches (|D_f|=%d, retain_count=%d)",
        len(forget_batches),
        forget_size,
        retain_count,
    )
    neg_grads = _accumulate_grad(
        model,
        forget_batches,
        params,
        average_scale=float(retain_count),
        sign=-1.0,
    )

    # --- 2. Retain pass: add per-batch grads, averaged over retain_count
    log.info(
        "[scif] retain pass over %d batches (retain_count=%d)",
        len(retain_batches),
        retain_count,
    )
    pos_grads = _accumulate_grad(
        model,
        retain_batches,
        params,
        average_scale=float(retain_count),
        sign=+1.0,
    )

    grads = [n + p for n, p in zip(neg_grads, pos_grads)]
    grad_norm = flat_norm(grads)
    log.info("[scif] ||grads|| (neg+pos combined) = %.6e", grad_norm)

    # --- 3. CG to invert (H + damping*I) using HVP from retain batches
    log.info(
        "[scif] running CG with max_iter=%d damping=%.3g tol=%.3g over %d HVP batches",
        cg_max_iter,
        cg_damping,
        cg_tol,
        len(retain_batches),
    )
    x, cg_info = cg_inv_hvp(
        model=model,
        hvp_batches=retain_batches,
        v_list=grads,
        params=params,
        damping=cg_damping,
        max_iter=cg_max_iter,
        tol=cg_tol,
        max_norm=cg_solution_max_norm,
    )

    # --- 4. Build parameter update delta = -tau * x, clip joint norm, apply
    tau = 1.0 / float(retain_size)
    update_before: List[torch.Tensor] = []
    n_param_skipped = 0
    n_param_masked = 0
    with torch.no_grad():
        for p, xi in zip(params, x):
            if torch.isnan(xi).any() or torch.isinf(xi).any():
                n_param_skipped += 1
                update_before.append(torch.zeros_like(xi))
            else:
                du = (-tau) * xi
                mask = grad_masks.get(id(p))
                if mask is not None:
                    # Mask before clipping so the norm covers only moved slots.
                    du = du * mask.to(device=du.device, dtype=du.dtype)
                    n_param_masked += 1
                update_before.append(du)

    update_norm_before_clip = flat_norm(update_before)
    log.info(
        "[scif] ||parameter_update|| before clip = %.6e (tau=%.6e, "
        "update_max_norm=%s)",
        update_norm_before_clip,
        tau,
        update_max_norm,
    )

    if update_max_norm is not None and update_max_norm > 0:
        update_after = clip_vector_to_max_norm(
            update_before, float(update_max_norm)
        )
    else:
        update_after = update_before

    update_norm_after_clip = flat_norm(update_after)
    log.info(
        "[scif] ||parameter_update|| after clip = %.6e",
        update_norm_after_clip,
    )

    n_param_updates = 0
    with torch.no_grad():
        for p, du in zip(params, update_after):
            if torch.isnan(du).any() or torch.isinf(du).any():
                continue
            p.add_(du)
            n_param_updates += 1
    log.info(
        "[scif] applied tau=%.3e: updated %d / skipped_nan_parts %d / total %d tensors",
        tau,
        n_param_updates,
        n_param_skipped,
        len(params),
    )

    info: Dict[str, object] = {
        "forget_size": int(forget_size),
        "retain_size": int(retain_size),
        "retain_count": int(retain_count),
        "n_forget_batches": int(len(forget_batches)),
        "n_retain_batches": int(len(retain_batches)),
        "n_param_tensors_total": int(len(params)),
        "n_param_tensors_updated": int(n_param_updates),
        "n_param_tensors_skipped_nan": int(n_param_skipped),
        "n_param_tensors_grad_masked": int(n_param_masked),
        "tau": tau,
        "grad_norm": float(grad_norm),
        "parameter_update_norm_before_clip": float(update_norm_before_clip),
        "parameter_update_norm_after_clip": float(update_norm_after_clip),
        "update_max_norm": update_max_norm,
        "target_params_policy": target_params_policy,
        "cg_damping": cg_damping,
        "cg_max_iter": cg_max_iter,
        "cg_tol": cg_tol,
        "cg": cg_info,
    }
    return info
