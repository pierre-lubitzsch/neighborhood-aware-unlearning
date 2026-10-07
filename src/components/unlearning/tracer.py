"""TRACER: Token ReAssignment for Concept ERasure (arXiv:2606.07688).

Port of TRACER as an unlearning baseline. Instead of suppressing a concept's
logits, TRACER reassigns the concept's items to different codewords, moving
them off tokens shared with the retain set.

Objective (Eq. 10):

    L = L_R + lambda_1 * L_F + lambda_2 * L_Coh + lambda_3 * L_reg

  L_R   (Eq. 7)  retain NLL
  L_F   (Eq. 8)  +sum log p on the forget set
  L_Coh (Eq. 9)  -1/K sum over P(i_T) of log p, where P(i_T) is the top-K
                 nearest items to the concept item i_T in the frozen dense
                 embedding space, excluding the concept set.
  L_reg (Eq. 6)  sum |phi|, keeps the reassignment sparse

P(i_T) is computed by the caller (cosine top-K over the embeddings the quantizer
was fitted on) and passed as ``concept_neighbor_sids`` ``[M, K, H]``; it is
independent of the prefix-neighborhood machinery used elsewhere.

Trainable: the backbone ``theta`` and the reassignment scores ``phi``.
Frozen: the codebooks and the encoder representations behind the residuals.

The selective-update mask (Eq. 11)

    M_{i,k}^l = 1[ rho_l(k) > rho_bar_{i,l} ] * 1[ grad_{phi} L_F > 0 ]

restricts phi to codewords more shared with the retain set than the item's
current assignment. It uses the gradient of ``L_F`` alone, computed with a
separate ``autograd.grad`` before the joint backward.

With ``commit=True`` the hard ``argmax_k q_phi`` codes are returned. The caller
must write them into the SID tensor used at evaluation time.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

from src.components.unlearning.hvp import batch_size, batch_to_device
from src.components.unlearning.optim_utils import build_optimizer
from src.components.unlearning.tracer_tokenizer import (
    phi_regularizer,
    retain_code_usage,
    selective_update_mask,
)

log = logging.getLogger(__name__)

TigerBatch = Any


def _batch_target_codes(batch: TigerBatch) -> Optional[torch.Tensor]:
    """The ``[B, H]`` raw semantic-id codes of each row's *label* item."""
    model_input, label_data = batch
    if label_data is None:
        return None
    fut = None
    for key in label_data.labels:
        fut = label_data.labels[key].reshape(model_input.mask.size(0), -1)
    return None if fut is None else fut.long()


def tracer_coherence_loss(
    model: nn.Module,
    batch: TigerBatch,
    concept_neighbor_sids: torch.Tensor,
) -> torch.Tensor:
    """L_Coh of Eq. 9, computed inside the TRACER path.

        L_Coh = -1/K  sum_{(H_f, i_T)}  sum_{i_p in P(i_T)}
                    sum_l log p_theta(s_l^p | T(H_f), s_{<l}^p)

    ``concept_neighbor_sids`` is ``[M, K, H]``: for each concept item (phi row
    ``m``) the raw semantic ids of its ``K`` embedding-space neighbors. Rows of
    ``batch`` whose label is not a concept item contribute nothing.
    """
    model_input, _ = batch
    device = model_input.mask.device
    fut = _batch_target_codes(batch)
    if fut is None or concept_neighbor_sids is None or concept_neighbor_sids.numel() == 0:
        return torch.zeros((), device=device)

    nbr = concept_neighbor_sids.to(device).long()                  # [M, K, H]
    with torch.no_grad():
        item_ids = model._codes_to_item_ids(fut.to(device).unsqueeze(1))[:, 0]
        rows = torch.where(
            item_ids >= 0,
            model._tracer_row_of_item[item_ids.clamp(min=0)],
            torch.full_like(item_ids, -1),
        )                                                          # [B]
        sel = rows >= 0
    n_sel = int(sel.sum())
    if n_sel == 0:
        return torch.zeros((), device=device)

    keep = sel.to(torch.float32)
    safe_rows = rows.clamp(min=0)
    total = torch.zeros((), device=device)
    n_terms = 0
    for c in range(int(nbr.shape[1])):
        lp = model._teacher_forced_log_prob(model_input, nbr[safe_rows, c])  # [B]
        total = total + (lp * keep).sum()
        n_terms += n_sel
    return -total / max(n_terms, 1)


def tracer_unlearn(
    model: nn.Module,
    forget_batches: Sequence[TigerBatch],
    retain_batches: Sequence[TigerBatch],
    *,
    concept_item_ids: torch.Tensor,
    residuals: torch.Tensor,
    centroids: torch.Tensor,
    codes: torch.Tensor,
    retain_item_ids: torch.Tensor,
    concept_neighbor_sids: Optional[torch.Tensor] = None,
    steps: Optional[int] = 500,
    n_epochs: Optional[int] = None,
    lr: float = 1e-4,
    phi_lr: float = 1e-2,
    tau: float = 0.005,
    lambda_forget: float = 1.0,
    lambda_coherence: float = 1.0,
    lambda_reg: float = 1e-3,
    selective_update: bool = True,
    optimizer: str = "sgd",
    commit: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run TRACER. Returns an info dict; ``new_codes`` holds the committed
    ``[M, L]`` reassignment when ``commit`` is set.

    ``residuals``/``centroids``/``codes`` must come from the RQ-KMeans checkpoint
    that produced ``codes`` (checked by ``tracer_tokenizer.assert_reproduces_sids``).
    """
    device = device or next(model.parameters()).device
    if not retain_batches:
        raise ValueError("retain_batches is empty")
    if not forget_batches:
        raise ValueError("forget_batches is empty")

    model.enable_token_reassignment(
        concept_item_ids=concept_item_ids,
        residuals=residuals,
        centroids=centroids,
        tau=tau,
    )
    phi = model.tracer_phi
    n_levels = int(centroids.shape[0])
    n_codes = int(centroids.shape[1])

    # rho_l(k): retain-set usage of each codeword, fixed for the whole run.
    rho = torch.stack(
        [
            retain_code_usage(codes, retain_item_ids, n_codes, lvl).to(device)
            for lvl in range(n_levels)
        ]
    )  # [L, K]

    theta = [p for n, p in model.named_parameters() if n != "tracer_phi" and p.requires_grad]
    groups = [
        {"params": theta, "lr": float(lr)},
        {"params": [phi], "lr": float(phi_lr)},
    ]
    # Default is plain SGD, matching the update rule in the TRACER paper.
    optimizer = str(optimizer).lower()
    opt = build_optimizer(optimizer, groups, float(lr), algo="tracer")

    n_forget, n_retain = len(forget_batches), len(retain_batches)
    per_pass = min(n_forget, n_retain)
    if n_epochs is not None:
        steps = int(n_epochs) * per_pass
    steps = int(steps or 0)
    if steps <= 0:
        raise ValueError("steps must be > 0")
    log.info(
        "[tracer] n_forget=%d n_retain=%d -> %d steps (n_epochs=%s), "
        "tau=%s lambda=(F %s, Coh %s, reg %s), selective_update=%s",
        n_forget, n_retain, steps, n_epochs, tau,
        lambda_forget, lambda_coherence, lambda_reg, selective_update,
    )

    totals: Dict[str, list] = {"total": [], "retain": [], "forget": [], "coh": [], "reg": []}
    masked_frac: list = []

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        f_batch = batch_to_device(forget_batches[step % n_forget], device)
        r_batch = batch_to_device(retain_batches[step % n_retain], device)

        # Eq. 8: negated NLL.
        l_forget = -model._batch_loss_from_model_step(f_batch)

        # Eq. 11 needs the gradient of L_F alone w.r.t. phi, before the joint backward.
        grad_phi_forget = None
        if selective_update:
            (grad_phi_forget,) = torch.autograd.grad(
                float(lambda_forget) * l_forget, phi, retain_graph=True, allow_unused=True
            )
            if grad_phi_forget is None:
                grad_phi_forget = torch.zeros_like(phi)

        l_retain = model._batch_loss_from_model_step(r_batch)          # Eq. 7
        l_coh = torch.zeros((), device=device)
        if concept_neighbor_sids is not None and float(lambda_coherence) != 0.0:
            l_coh = tracer_coherence_loss(model, f_batch, concept_neighbor_sids)
        l_reg = phi_regularizer([phi])                                  # Eq. 6

        total = (
            l_retain
            + float(lambda_forget) * l_forget
            + float(lambda_coherence) * l_coh
            + float(lambda_reg) * l_reg
        )
        total.backward()

        # Eq. 11: gate phi's update to retain-entangled codewords with conflicting
        # forget gradient. Applied per level, on .grad, before the step.
        if selective_update and phi.grad is not None:
            with torch.no_grad():
                q = torch.softmax(model.tracer_assignment_scores() / tau, dim=-1)
                keep = torch.stack(
                    [
                        selective_update_mask(q[:, l], rho[l], grad_phi_forget[:, l])
                        for l in range(n_levels)
                    ],
                    dim=1,
                )
                phi.grad.mul_(keep)
                masked_frac.append(float(keep.mean()))

        opt.step()

        totals["total"].append(float(total.detach().cpu()))
        totals["retain"].append(float(l_retain.detach().cpu()))
        totals["forget"].append(float(l_forget.detach().cpu()))
        totals["coh"].append(float(l_coh.detach().cpu()))
        totals["reg"].append(float(l_reg.detach().cpu()))
        if step % max(1, steps // 10) == 0:
            log.info(
                "[tracer] step=%d/%d total=%.4f retain=%.4f forget=%.4f coh=%.4f reg=%.4f",
                step, steps, totals["total"][-1], totals["retain"][-1],
                totals["forget"][-1], totals["coh"][-1], totals["reg"][-1],
            )

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    info: Dict[str, Any] = {
        "algorithm": "tracer",
        "steps": steps,
        "n_epochs": n_epochs,
        "tau": float(tau),
        "lambda_forget": float(lambda_forget),
        "lambda_coherence": float(lambda_coherence),
        "lambda_reg": float(lambda_reg),
        "selective_update": bool(selective_update),
        "optimizer": optimizer,
        "phi_lr": float(phi_lr),
        "lr": float(lr),
        "n_concept_items": int(concept_item_ids.numel()),
        "coherence_neighbors_per_concept": (
            0 if concept_neighbor_sids is None else int(concept_neighbor_sids.shape[1])
        ),
        "coherence_neighbor_source": "tracer_embedding_topk_of_concept_items",
        "n_forget_batches": n_forget,
        "n_retain_batches": n_retain,
        "n_forget_rows": sum(batch_size(b) for b in forget_batches),
        "n_retain_rows": sum(batch_size(b) for b in retain_batches),
        "mean_total_loss": _mean(totals["total"]),
        "mean_retain_loss": _mean(totals["retain"]),
        "mean_forget_loss": _mean(totals["forget"]),
        "mean_coh_loss": _mean(totals["coh"]),
        "mean_reg_loss": _mean(totals["reg"]),
        "mean_masked_fraction": _mean(masked_frac),
        "phi_abs_mean": float(phi.detach().abs().mean()),
        "phi_abs_max": float(phi.detach().abs().max()),
    }

    if commit:
        new_codes = model.commit_token_reassignment().cpu()             # [M, L]
        old_codes = codes[:n_levels][:, concept_item_ids.long()].T.cpu()
        changed = (new_codes != old_codes)
        # Stored as lists, since str(tensor) truncates large tensors in JSON output.
        info["new_codes"] = new_codes.tolist()
        info["old_codes"] = old_codes.tolist()
        info["concept_item_ids"] = concept_item_ids.cpu().tolist()
        info["n_codes_changed"] = int(changed.sum())
        info["frac_items_reassigned"] = float(changed.any(dim=1).float().mean())
        log.info(
            "[tracer] committed: %d/%d codes changed; %.1f%% of concept items "
            "reassigned at >=1 level",
            info["n_codes_changed"],
            new_codes.numel(),
            100.0 * info["frac_items_reassigned"],
        )
        log.warning(
            "[tracer] write the reassigned codes into the SID tensor and pass it "
            "as semantic_id_path at eval time; otherwise metrics use stale codes."
        )
    return info
