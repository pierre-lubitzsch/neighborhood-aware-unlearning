"""Position-wise gradient diagnostics for TIGER RQ semantic IDs.

TIGER's loss is a sum of per-hierarchy cross-entropy terms
(``model.per_hierarchy_losses``), so the gradient of each term isolates the
signal flowing through one code position. For each position this module reports
the forget and retain gradient norms and their cosine similarity; a large
positive cosine indicates that forgetting at that position conflicts with
retention. The model is never updated.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

log = logging.getLogger(__name__)

TigerBatch = Any  # (SequentialModelInputData, SequentialModuleLabelData)


def _flat_cat(grads: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
    """Flatten and concatenate a list of per-parameter gradients into one vector."""
    return torch.cat([g.reshape(-1) for g in grads])


def _accumulate_per_position_grads(
    model: nn.Module,
    batches: Sequence["TigerBatch"],
    params: Sequence[nn.Parameter],
    num_hierarchies: int,
) -> List[List[torch.Tensor]]:
    """Mean per-position gradient over ``batches``.

    Returns ``accs[h]`` = list of per-parameter gradient tensors for the loss at
    RQ code position ``h``, averaged over the number of batches.
    """
    accs: List[List[torch.Tensor]] = [
        [torch.zeros_like(p) for p in params] for _ in range(num_hierarchies)
    ]
    n_batches = 0
    for batch in batches:
        model_input, label_data = batch
        losses = model.per_hierarchy_losses(model_input, label_data)
        for h in range(num_hierarchies):
            # Retain the forward graph across the H grad() calls for this batch.
            grads = torch.autograd.grad(
                losses[h],
                params,
                retain_graph=(h < num_hierarchies - 1),
                allow_unused=True,
            )
            for i, gi in enumerate(grads):
                if gi is not None:
                    accs[h][i] = accs[h][i] + gi
        n_batches += 1
    if n_batches > 0:
        for h in range(num_hierarchies):
            accs[h] = [a / float(n_batches) for a in accs[h]]
    return accs


def per_position_gradient_report(
    model: nn.Module,
    forget_batches: Sequence["TigerBatch"],
    retain_batches: Sequence["TigerBatch"],
    params: Sequence[nn.Parameter],
    num_hierarchies: Optional[int] = None,
    eval_mode: bool = True,
) -> Dict[str, Any]:
    """Compute per-RQ-position forget/retain gradient strength and conflict.

    Parameters
    ----------
    model
        TIGER model exposing ``per_hierarchy_losses`` and ``num_hierarchies``.
    forget_batches, retain_batches
        Pre-collected TIGER batches on the model's device (e.g. from
        ``_prepare_unlearning_context``).
    params
        Parameters to differentiate against (e.g. ``select_target_params``).
    num_hierarchies
        Defaults to ``model.num_hierarchies``.
    eval_mode
        If True, ``model.eval()`` first (deterministic, matches SCIF).
    """
    H = int(num_hierarchies or model.num_hierarchies)
    params = list(params)
    if eval_mode:
        model.eval()

    forget_grads = _accumulate_per_position_grads(model, forget_batches, params, H)
    retain_grads = _accumulate_per_position_grads(model, retain_batches, params, H)

    positions: List[Dict[str, Any]] = []
    for h in range(H):
        f = _flat_cat(forget_grads[h])
        r = _flat_cat(retain_grads[h])
        f_norm = float(f.norm())
        r_norm = float(r.norm())
        denom = max(f_norm * r_norm, 1e-12)
        cosine = float(torch.dot(f, r)) / denom
        positions.append(
            {
                "code": f"c{h + 1}",
                "hierarchy_index": h,
                "forget_grad_norm": f_norm,
                "retain_grad_norm": r_norm,
                "forget_retain_cosine": cosine,
                # Positive values mean forgetting conflicts with retention.
                "conflict_score": cosine,
            }
        )

    by_forget = sorted(positions, key=lambda d: d["forget_grad_norm"], reverse=True)
    by_conflict = sorted(positions, key=lambda d: d["conflict_score"], reverse=True)
    report = {
        "num_hierarchies": H,
        "n_forget_batches": int(len(forget_batches)),
        "n_retain_batches": int(len(retain_batches)),
        "n_param_tensors": int(len(params)),
        "positions": positions,
        "strongest_forget_code": by_forget[0]["code"] if by_forget else None,
        "strongest_conflict_code": by_conflict[0]["code"] if by_conflict else None,
    }
    log.info(
        "[rq-diag] position gradients: strongest forget=%s, strongest conflict=%s",
        report["strongest_forget_code"],
        report["strongest_conflict_code"],
    )
    return report
