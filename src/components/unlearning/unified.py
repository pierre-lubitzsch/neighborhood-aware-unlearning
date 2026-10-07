"""Unified unlearning objective used by NAU and its ablations:

    L = lambda_retain * L_retain + lambda_forget * L_forget
        + lambda_sep * L_sep + lambda_neighborhood * L_neighborhood

Setting individual weights to zero gives Finetune, Forget only, and Forget+Repair.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
from torch import nn

from src.components.unlearning.optim_utils import build_optimizer
from src.components.unlearning.hvp import batch_size, batch_to_device
from src.components.unlearning.local_repair import apply_local_repair_losses
from src.components.unlearning.slot_selection import select_top_t_slots
from src.components.unlearning.target_params import (
    resolve_scope_params,
    select_adaptive_code_params,
    split_adaptive_code_params,
    split_stable_code_params,
    split_code_params,
    select_code_position_params,
    select_pkm_params,
)

log = logging.getLogger(__name__)
TigerBatch = Any


def unified_unlearn(
    model: nn.Module,
    forget_batches: Sequence[TigerBatch],
    retain_batches: Sequence[TigerBatch],
    *,
    steps: Optional[int] = 500,
    n_epochs: Optional[int] = None,
    lr: float = 1e-4,
    lambda_retain: float = 1.0,
    lambda_forget: float = 1.0,
    lambda_sep: float = 0.1,
    lambda_neighborhood: float = 0.0,
    coherence_neighbors: Optional[Sequence[Any]] = None,
    coherence_loss_type: str = "nll",
    coherence_mass_cap: float = 0.999,
    forget_loss_level: str = "token",
    # Per-SID-level weights on the item-level loss. `position_weights` applies to
    # both the retain and forget terms; `forget_position_weights`, if given,
    # overrides it on the forget side.
    position_weights: Optional[Sequence[float]] = None,
    forget_position_weights: Optional[Sequence[float]] = None,
    sep_temperature: float = 0.07,
    deletion_spec: str = "session",
    forget_item_ids: Optional[Set[int]] = None,
    neighbor_item_ids: Optional[Set[int]] = None,
    sep_negative_item_ids: Optional[Set[int]] = None,
    sep_negatives_mode: str = "forget",
    sep_positives: str = "history",
    sep_loss_type: str = "cosine",
    sep_gen_temperature: float = 1.0,
    local_repair_cfg: Optional[Dict[str, Any]] = None,
    restrict_adaptive_codes: bool = False,
    stable_codes: int = 2,
    adaptive_update_backbone: bool = False,
    adaptive_adapter: bool = False,
    update_positions: Optional[List[int]] = None,
    update_positions_backbone: bool = False,
    update_scope: str = "all",
    pkm_update_keys: bool = True,
    pkm_update_query: bool = True,
    slot_selection: str = "none",
    slot_top_t: int = 32,
    slot_lambda: float = 1.0,
    slot_mu: float = 5.0,
    slot_dot_abs: bool = False,
    optimizer: str = "adam",
    code_lr_scale: float = 1.0,
    adaptive_code_lr_scale: float = 1.0,
    stable_code_lr_scale: float = 1.0,
    # [H*K] float mask over the SID embedding table's rows (1.0 = may update),
    # restricting updates to the code rows touched by the deletion request.
    code_row_keep: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Optimize unified objective.

    The number of optimizer steps is set either directly via ``steps`` or
    indirectly via ``n_epochs`` (full passes through the batches). With
    balanced accumulation, one pass through the batches equals
    ``min(n_forget_batches, n_retain_batches)`` optimizer steps, so
    ``n_epochs=N`` ⇒ ``steps = N * min(n_forget, n_retain)``.
    If both are given, ``n_epochs`` wins.

    Each optimizer step accumulates gradients across ``q_forget`` forget
    mini-batches and ``q_retain`` retain mini-batches, where

        q_retain = ceil(n_retain / n_forget)
        q_forget = ceil(n_forget / n_retain)

    (one of the two is always 1). This balances per-sample exposure: every
    forget sample and every retain sample contributes to the gradient the same
    number of times, regardless of how many batches each side has.

    The ``L_sep`` negatives default to the forget items ``I_f``.
    ``sep_negative_item_ids``, when set, replaces them with a fixed set.
    ``sep_negatives_mode`` is stored as metadata only. Local repair uses
    ``neighbor_item_ids``.

    When ``lambda_neighborhood > 0``, the coherence term ``L_n`` is added to the
    forget side of each step: for each eligible forget sample the model is scored
    (teacher-forced) on the semantic-ID codes of its target's prefix neighbors,
    conditioned on the forget history, and this negative log-probability is
    minimized so suppressed mass flows to neighboring items.
    ``coherence_neighbors`` is a per-forget-batch sequence aligned to
    ``forget_batches``; each element is ``(neighbor_sids[B, C, H],
    neighbor_mask[B, C])`` or ``None`` when a batch has no eligible neighbors.
    ``coherence_loss_type`` selects the per-neighbor ``nll`` form or the
    ``mass`` form (logsumexp over the neighborhood); see
    ``compute_coherence_loss``.
    """
    device = device or next(model.parameters()).device
    if not retain_batches:
        raise ValueError("retain_batches is empty")
    if not forget_batches:
        raise ValueError("forget_batches is empty")
    if not hasattr(model, "compute_sep_loss"):
        raise TypeError(
            "model must expose compute_sep_loss / _batch_loss_from_model_step "
            "(SemanticIDEncoderDecoder subclass)"
        )

    n_forget = len(forget_batches)
    n_retain = len(retain_batches)
    q_retain = max(1, math.ceil(n_retain / n_forget))
    q_forget = max(1, math.ceil(n_forget / n_retain))
    optim_steps_per_pass = min(n_forget, n_retain)

    if n_epochs is not None:
        n_epochs = int(n_epochs)
        if n_epochs <= 0:
            raise ValueError("n_epochs must be > 0")
        steps = n_epochs * optim_steps_per_pass
    else:
        if steps is None:
            raise ValueError("Either steps or n_epochs must be set")
        steps = int(steps)
        if steps <= 0:
            raise ValueError("steps must be > 0")

    log.info(
        "[unified] n_forget_batches=%d n_retain_batches=%d "
        "→ q_forget=%d q_retain=%d (per optim step: %d forget + %d retain mini-batches); "
        "optim_steps_per_pass=%d, total_steps=%d, n_epochs=%s",
        n_forget,
        n_retain,
        q_forget,
        q_retain,
        q_forget,
        q_retain,
        optim_steps_per_pass,
        steps,
        n_epochs if n_epochs is not None else "(unset)",
    )

    # adaptive_code_lr_scale and stable_code_lr_scale are extra lr multipliers
    # on the fine code levels [stable_codes, H) and the coarse levels
    # [0, stable_codes), respectively. 1.0 disables them.
    if float(stable_code_lr_scale) < 0.0:
        raise ValueError(
            f"stable_code_lr_scale must be >= 0, got {stable_code_lr_scale!r}"
        )
    if float(adaptive_code_lr_scale) < 0.0:
        raise ValueError(
            f"adaptive_code_lr_scale must be >= 0, got {adaptive_code_lr_scale!r}"
        )
    scale_stable = float(stable_code_lr_scale) != 1.0
    scale_adaptive = float(adaptive_code_lr_scale) != 1.0
    if scale_adaptive or scale_stable:
        # Under a restriction mode one half is already frozen, so a relative
        # rate would reduce to a plain lr change.
        conflict = None
        if restrict_adaptive_codes:
            conflict = "restrict_adaptive_codes=True"
        elif update_positions:
            # Allowed only if update_positions keeps positions on both sides
            # of stable_codes.
            pos = {int(p) for p in update_positions}
            sc = int(stable_codes)
            adaptive_pos = {p for p in pos if p >= sc}
            stable_pos = {p for p in pos if p < sc}
            if scale_adaptive and not (adaptive_pos and stable_pos):
                conflict = (
                    f"update_positions={sorted(pos)} leaves no adaptive/stable "
                    "split for adaptive_code_lr_scale to act across"
                )
            if scale_stable and not (adaptive_pos and stable_pos):
                conflict = (
                    f"update_positions={sorted(pos)} leaves no adaptive/stable "
                    "split for stable_code_lr_scale to act across"
                )
        elif str(update_scope).lower() != "all":
            conflict = f"update_scope={update_scope!r}"
        if conflict is not None:
            raise ValueError(
                f"adaptive_code_lr_scale={adaptive_code_lr_scale} needs the "
                f"unrestricted update set, but {conflict} already confines it. "
                "Lower unified_lr instead, or set adaptive_code_lr_scale=1.0."
            )

    # Select the trainable parameters. grad_masks holds per-parameter masks
    # applied to .grad each step before opt.step().
    slot_select_info = None
    if str(update_scope).lower() in ("pkm_only", "ffn_only"):
        # Only the memory (PKM or FFN) parameters are optimized; everything else
        # stays frozen. Takes precedence over the restrictions below.
        params, pkm_names = resolve_scope_params(
            model, update_scope,
            fallback=[p for p in model.parameters() if p.requires_grad],
            include_keys=bool(pkm_update_keys),
            include_query=bool(pkm_update_query),
            algo="unified",
        )
        grad_masks = {}
        log.info(
            "[unified] PKM-ONLY update scope: %d param tensors (%d params) "
            "across %d memory modules (keys=%s query=%s); backbone FROZEN",
            len(params),
            int(sum(p.numel() for p in params)),
            len({n.rsplit(".", 1)[0] for n in pkm_names}),
            bool(pkm_update_keys),
            bool(pkm_update_query),
        )
        # Optional top-t slot restriction on top of pkm_only: score every value
        # slot from the forget/retain data and mask the gradient of all but the
        # best t, so only those memory rows move.
        if (str(update_scope).lower() == "ffn_only"
                and str(slot_selection).strip().lower() not in ("none", "", "off")):
            raise ValueError(
                "slot_selection is PKM-specific and cannot be combined with "
                "update_scope='ffn_only' (an FFN has no memory slots)."
            )
        if str(slot_selection).strip().lower() not in ("none", "", "off"):
            masks, sel_info = select_top_t_slots(
                model,
                forget_batches,
                retain_batches,
                criterion=str(slot_selection),
                top_t=int(slot_top_t),
                lam=float(slot_lambda),
                mu=float(slot_mu),
                dot_abs=bool(slot_dot_abs),
            )
            name_to_mod = dict(model.named_modules())
            n_masked = 0
            for mem_name, mask in masks.items():
                w = name_to_mod[mem_name].values.weight
                grad_masks[id(w)] = mask
                n_masked += 1
            slot_select_info = sel_info
            log.info(
                "[unified] slot selection=%s top_t=%d -> gradient masks on %d "
                "value tables",
                str(slot_selection), int(slot_top_t), n_masked,
            )
    elif update_positions:
        # Update only the given code positions (e.g. [0] = coarsest level).
        # Takes precedence over the adaptive-code modes.
        params, grad_masks = select_code_position_params(
            model,
            positions=list(update_positions),
            update_backbone=bool(update_positions_backbone),
        )
        log.info(
            "[unified] position-wise restriction ON: update_positions=%s "
            "update_backbone=%s → %d param tensors (%d params), %d masked",
            list(update_positions),
            bool(update_positions_backbone),
            len(params),
            int(sum(p.numel() for p in params)),
            len(grad_masks),
        )
    elif restrict_adaptive_codes and adaptive_adapter:
        # Per-item adapter: freeze the shared table and heads and train only a
        # per-item offset on the adaptive positions, masked to forget and
        # neighbor items.
        if getattr(model, "adaptive_item_offset", None) is None:
            if not hasattr(model, "enable_adaptive_item_offset"):
                raise TypeError(
                    "adaptive_adapter=True requires a model exposing "
                    "enable_adaptive_item_offset (SemanticIDEncoderDecoder)"
                )
            model.enable_adaptive_item_offset(int(stable_codes))
        offset = model.adaptive_item_offset
        params = [offset]
        grad_masks = {}
        adapter_items = set(forget_item_ids or []) | set(neighbor_item_ids or [])
        n_rows = offset.shape[0]
        valid_items = sorted(i for i in adapter_items if 0 <= int(i) < n_rows)
        if valid_items:
            row_mask = torch.zeros((n_rows, 1, 1), device=offset.device)
            row_mask[torch.tensor(valid_items, device=offset.device)] = 1.0
            grad_masks[id(offset)] = row_mask
        log.info(
            "[unified] per-item adaptive adapter ON: stable_codes=%d, "
            "%d / %d offset rows trainable (others frozen via grad mask)",
            int(stable_codes),
            len(valid_items) if valid_items else n_rows,
            n_rows,
        )
    elif restrict_adaptive_codes:
        # Shared: update the adaptive embedding rows and heads; grad_masks
        # zeroes the stable rows.
        params, grad_masks = select_adaptive_code_params(
            model,
            stable_codes=int(stable_codes),
            update_backbone=bool(adaptive_update_backbone),
        )
        log.info(
            "[unified] adaptive-code restriction ON: stable_codes=%d "
            "update_backbone=%s → %d param tensors (%d params), %d masked",
            int(stable_codes),
            bool(adaptive_update_backbone),
            len(params),
            int(sum(p.numel() for p in params)),
            len(grad_masks),
        )
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        grad_masks = {}
    # Optional per-group learning rates: code parameters (SID embedding table
    # and decoder heads) get lr * code_lr_scale, further scaled by the
    # adaptive/stable multipliers. Parameters are grouped by their multiplier.
    code_params, _other_params = split_code_params(model, params)
    code_ids = {id(p) for p in code_params}
    adaptive_tensors: List[nn.Parameter] = []
    adaptive_row_masks: Dict[int, torch.Tensor] = {}
    if scale_adaptive or scale_stable:
        # The adaptive row mask is needed whenever either scale is active, to
        # separate stable from adaptive rows of the shared SID table.
        adaptive_tensors, adaptive_row_masks = split_adaptive_code_params(
            model, params, stable_codes=int(stable_codes)
        )
        if scale_adaptive and not adaptive_tensors and not adaptive_row_masks:
            raise ValueError(
                "adaptive_code_lr_scale != 1.0 but no adaptive code parameters "
                "were found (no SID table and no decoder_mlp heads); the knob "
                "would silently do nothing."
            )
    adaptive_ids = {id(p) for p in adaptive_tensors} if scale_adaptive else set()

    # Stable-prefix counterpart: the coarse codes [0, stable_codes).
    stable_tensors: List[nn.Parameter] = []
    stable_row_masks: Dict[int, torch.Tensor] = {}
    if scale_stable:
        stable_tensors, stable_row_masks = split_stable_code_params(
            model, params, stable_codes=int(stable_codes)
        )
        if not stable_tensors and not stable_row_masks:
            raise ValueError(
                "stable_code_lr_scale != 1.0 but no stable code parameters were "
                "found; the knob would silently do nothing."
            )
    stable_ids = {id(p) for p in stable_tensors}

    if not code_params and float(code_lr_scale) != 1.0:
        log.warning(
            "[unified] code_lr_scale=%s requested but no semantic-ID code "
            "params are in the update set (restriction policy may exclude "
            "them); falling back to a single group.",
            code_lr_scale,
        )

    multiplier: Dict[int, float] = {}
    for p in params:
        m = 1.0
        if id(p) in code_ids:
            m *= float(code_lr_scale)
        if id(p) in adaptive_ids:
            m *= float(adaptive_code_lr_scale)
        if id(p) in stable_ids:
            m *= float(stable_code_lr_scale)
        multiplier[id(p)] = m

    by_mult: Dict[float, List[nn.Parameter]] = {}
    for p in params:
        by_mult.setdefault(multiplier[id(p)], []).append(p)
    groups: List[Dict[str, Any]] = [
        {"params": ps, "lr": float(lr) * m} for m, ps in sorted(by_mult.items())
    ]
    for m, ps in sorted(by_mult.items()):
        log.info(
            "[unified] lr group x%.4g -> lr=%.3g: %d tensors (%d params)",
            m, float(lr) * m, len(ps), int(sum(p.numel() for p in ps)),
        )
    # Row-level rate for the SID table, whose rows span both segments. It is
    # applied to the update after opt.step(), since scaling .grad would have no
    # effect under Adam.
    row_lr_scale: Dict[int, torch.Tensor] = {}
    row_scaled_params: Dict[int, nn.Parameter] = {}
    if adaptive_row_masks:
        by_id = {id(p): p for p in params}
        for pid, mask in adaptive_row_masks.items():
            p = by_id.get(pid)
            if p is None:
                continue
            # mask is 1.0 on adaptive rows.
            row_lr_scale[pid] = (
                mask * float(adaptive_code_lr_scale)
                + (1.0 - mask) * float(stable_code_lr_scale)
            ).to(dtype=p.dtype)
            row_scaled_params[pid] = p
        log.info(
            "[unified] adaptive_code_lr_scale=%s: %d adaptive head tensors at "
            "lr=%.3g, SID-table adaptive rows [%d:] rescaled to lr=%.3g "
            "(stable rows keep lr=%.3g)",
            adaptive_code_lr_scale,
            len(adaptive_tensors),
            float(lr) * float(code_lr_scale) * float(adaptive_code_lr_scale),
            int(stable_codes) * int(getattr(model, "num_embeddings_per_hierarchy", 0)),
            float(lr) * float(code_lr_scale) * float(adaptive_code_lr_scale),
            float(lr) * float(code_lr_scale),
        )

    # Compose the code-row scope onto the SID table's row multiplier. Masking
    # .grad is not enough under Adam, since momentum still moves the row.
    if code_row_keep is not None:
        sid_w = None
        table = getattr(model, "item_sid_embedding_table_encoder", None)
        if table is not None:
            sid_w = getattr(table, "weight", None)
        if sid_w is None:
            raise ValueError(
                "code_row_scope needs model.item_sid_embedding_table_encoder"
            )
        if int(code_row_keep.numel()) != int(sid_w.shape[0]):
            raise ValueError(
                f"code_row_keep has {int(code_row_keep.numel())} entries but the "
                f"SID table has {int(sid_w.shape[0])} rows"
            )
        pid = id(sid_w)
        keep_col = code_row_keep.to(device=sid_w.device, dtype=sid_w.dtype).view(-1, 1)
        if pid in row_lr_scale:
            row_lr_scale[pid] = row_lr_scale[pid] * keep_col
        else:
            # No row scaling is active yet, so build the multiplier here.
            row_lr_scale[pid] = keep_col.expand_as(sid_w).clone()
            row_scaled_params[pid] = sid_w
        log.info(
            "[unified] code_row_scope: %d/%d SID rows updatable (%.2f%%)",
            int(code_row_keep.sum()), int(code_row_keep.numel()),
            100.0 * float(code_row_keep.sum()) / float(code_row_keep.numel()),
        )

    opt = build_optimizer(optimizer, groups, float(lr), algo="unified")
    model.train()

    totals: Dict[str, List[float]] = {
        "total": [],
        "retain": [],
        "forget": [],
        "sep": [],
        "coh": [],
    }

    # A negative lambda_n drains the neighborhood mass, but the objective is
    # then unbounded below; coherence_loss_type=suppress is the bounded form.
    if float(lambda_neighborhood) < 0.0:
        log.warning(
            "[unified] lambda_n=%s < 0 with coherence_loss_type=%s: the "
            "objective is unbounded below. Consider lambda_n > 0 with "
            "coherence_loss_type=suppress.",
            lambda_neighborhood,
            coherence_loss_type,
        )
    use_coherence = float(lambda_neighborhood) != 0.0 and coherence_neighbors is not None
    n_coherence_batches = (
        sum(1 for cn in coherence_neighbors if cn is not None)
        if coherence_neighbors is not None
        else 0
    )

    forget_ids = set(forget_item_ids or [])
    neighbor_ids = set(neighbor_item_ids or [])  # used only for local-repair losses
    # L_sep negatives: the forget items I_f unless an explicit set is given.
    if sep_negative_item_ids is not None:
        sep_negatives_set: Set[int] = set(sep_negative_item_ids)
    else:
        sep_negatives_set = forget_ids
    sequence_forget = str(forget_loss_level).lower() == "sequence"
    # SID-level weights are per position, so they are incompatible with the
    # `sequence` forget level.
    def _resolve_w(vec, name):
        if vec is None:
            return None
        out = [float(v) for v in vec]
        n_h = int(getattr(model, "num_hierarchies", len(out)))
        if len(out) != n_h:
            raise ValueError(f"{name} has {len(out)} entries, but the model has "
                             f"{n_h} SID levels")
        return out

    w_shared = _resolve_w(position_weights, "position_weights")
    w_r = w_shared
    w_f = None
    if forget_position_weights is not None or w_shared is not None:
        w_f = _resolve_w(forget_position_weights, "forget_position_weights") \
            if forget_position_weights is not None else w_shared
        if sequence_forget:
            raise ValueError(
                "SID-level weights require forget_loss_level=token; the "
                "sequence level has no per-level decomposition")
        log.info("[unified] SID-level weights: w_forget=%s w_retain=%s",
                 w_f, w_r)

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        # --- Forget side: q_forget mini-batches, each scaled by 1/q_forget ---
        l_forget_avg = 0.0
        l_coh_avg = 0.0
        for j in range(q_forget):
            idx = (step * q_forget + j) % n_forget
            forget_batch = batch_to_device(forget_batches[idx], device)
            # Skip the forget loss at lambda_f = 0 (l_forget_avg stays 0).
            if float(lambda_forget) != 0.0:
                if sequence_forget:
                    l_forget = model._sequence_log_prob(*forget_batch)
                elif w_f is not None:
                    # sum_h w_h * L_h; equals the unweighted loss at w_f = 1.
                    per_h = model.per_hierarchy_losses(*forget_batch)
                    l_forget = -sum(w * L for w, L in zip(w_f, per_h))
                else:
                    l_forget = -model._batch_loss_from_model_step(forget_batch)
                forget_term = (float(lambda_forget) * l_forget) / float(q_forget)
                forget_term.backward()
                l_forget_avg += float(l_forget.detach().cpu()) / float(q_forget)

            # --- Neighborhood term L_n, on the same forget batch ---
            if use_coherence:
                cn = coherence_neighbors[idx]
                if cn is not None:
                    neighbor_sids, neighbor_mask = cn
                    l_coh = model.compute_coherence_loss(
                        forget_batch,
                        neighbor_sids,
                        neighbor_mask,
                        loss_type=coherence_loss_type,
                        mass_cap=coherence_mass_cap,
                    )
                    if l_coh.requires_grad:
                        coh_term = (float(lambda_neighborhood) * l_coh) / float(q_forget)
                        coh_term.backward()
                    l_coh_avg += float(l_coh.detach().cpu()) / float(q_forget)

        # --- Retain side: q_retain mini-batches, each scaled by 1/q_retain ---
        l_retain_avg = 0.0
        l_sep_avg = 0.0
        last_retain_batch = None
        for j in range(q_retain):
            idx = (step * q_retain + j) % n_retain
            retain_batch = batch_to_device(retain_batches[idx], device)
            last_retain_batch = retain_batch
            # Skip the retain loss at lambda_retain = 0 (l_retain_avg stays 0).
            if float(lambda_retain) != 0.0:
                if w_r is not None:
                    # sum_l w_l * L_l, as on the forget side.
                    per_h = model.per_hierarchy_losses(*retain_batch)
                    l_retain = sum(w * L for w, L in zip(w_r, per_h))
                else:
                    l_retain = model._batch_loss_from_model_step(retain_batch)
            else:
                l_retain = torch.zeros((), device=device)
            # Skip the separation loss at lambda_s = 0; the generative form
            # costs 1 + |I_f| decoder passes per retain row.
            if float(lambda_sep) == 0.0:
                l_sep = torch.zeros((), device=device)
            else:
                l_sep = model.compute_sep_loss(
                    retain_batch,
                    negative_item_ids=sep_negatives_set,
                    temperature=float(sep_temperature),
                    positives=sep_positives,
                    loss_type=sep_loss_type,
                    gen_temperature=float(sep_gen_temperature),
                )
            # Baselines as special cases of this objective:
            #   lambda_r=1, lambda_f=0, lambda_s=0, lambda_n=0 -> finetune
            #   lambda_r=1, lambda_f=w, lambda_s=0, lambda_n=0 -> neg_train
            #   lambda_r=0, lambda_f=w, lambda_s=0, lambda_n=0 -> gradient ascent
            retain_side = float(lambda_retain) * l_retain + float(lambda_sep) * l_sep
            retain_side = apply_local_repair_losses(
                model,
                base_loss=retain_side,
                local_repair_cfg=local_repair_cfg or {},
                neighbor_item_ids=neighbor_ids,
                batch=retain_batch,
            )
            # With lambda_retain = 0 and lambda_sep = 0 the retain side has no
            # grad_fn.
            if retain_side.requires_grad:
                (retain_side / float(q_retain)).backward()
            l_retain_avg += float(l_retain.detach().cpu()) / float(q_retain)
            l_sep_avg += float(l_sep.detach().cpu()) / float(q_retain)

        # Mask stable-code rows out of the embedding gradient before stepping.
        if grad_masks:
            for p in params:
                m = grad_masks.get(id(p))
                if m is not None and p.grad is not None:
                    p.grad.mul_(m)

        # Per-row learning rate for the SID table: take the optimizer step, then
        # rescale each row's delta.
        prev_rows: Dict[int, torch.Tensor] = {}
        if row_scaled_params:
            with torch.no_grad():
                for pid, p in row_scaled_params.items():
                    prev_rows[pid] = p.detach().clone()

        opt.step()

        if row_scaled_params:
            with torch.no_grad():
                for pid, p in row_scaled_params.items():
                    old = prev_rows[pid]
                    delta = p.data.sub(old).mul_(row_lr_scale[pid])
                    p.data.copy_(old.add_(delta))
            prev_rows.clear()

        total_avg = (
            l_retain_avg
            + float(lambda_forget) * l_forget_avg
            + float(lambda_sep) * l_sep_avg
            + float(lambda_neighborhood) * l_coh_avg
        )
        totals["total"].append(total_avg)
        totals["retain"].append(l_retain_avg)
        totals["forget"].append(l_forget_avg)
        totals["sep"].append(l_sep_avg)
        totals["coh"].append(l_coh_avg)

        if step % max(1, steps // 10) == 0:
            log.info(
                "[unified] step=%d/%d total=%.4f retain=%.4f forget=%.4f "
                "sep=%.4f neighborhood=%.4f",
                step,
                steps,
                total_avg,
                l_retain_avg,
                l_forget_avg,
                l_sep_avg,
                l_coh_avg,
            )

        del last_retain_batch  # free reference

    def _mean(xs: List[float]) -> Optional[float]:
        return float(sum(xs) / max(1, len(xs))) if xs else None

    return {
        "algorithm": "unified",
        "steps": steps,
        "n_epochs": n_epochs,
        "optim_steps_per_pass": optim_steps_per_pass,
        "q_forget": q_forget,
        "q_retain": q_retain,
        "lr": float(lr),
        "code_lr_scale": float(code_lr_scale),
        "adaptive_code_lr_scale": float(adaptive_code_lr_scale),
        "stable_code_lr_scale": float(stable_code_lr_scale),
        "adaptive_code_lr_stable_codes": (
            int(stable_codes) if scale_adaptive else None
        ),
        "adaptive_code_lr_tensors": len(adaptive_tensors),
        "adaptive_code_lr_row_scaled": len(row_scaled_params),
        "lr_groups": [
            {"mult": m, "lr": float(lr) * m, "n_tensors": len(ps),
             "n_params": int(sum(p.numel() for p in ps))}
            for m, ps in sorted(by_mult.items())
        ],
        "restrict_adaptive_codes": bool(restrict_adaptive_codes),
        "slot_selection": slot_select_info,
        "update_positions": list(update_positions) if update_positions else None,
        "update_positions_backbone": (
            bool(update_positions_backbone) if update_positions else None
        ),
        "stable_codes": int(stable_codes) if restrict_adaptive_codes else None,
        "adaptive_update_backbone": (
            bool(adaptive_update_backbone) if restrict_adaptive_codes else None
        ),
        "adaptive_adapter": bool(restrict_adaptive_codes and adaptive_adapter),
        "lambda_forget": float(lambda_forget),
        "lambda_retain": float(lambda_retain),
        "lambda_sep": float(lambda_sep),
        "lambda_neighborhood": float(lambda_neighborhood),
        "coherence_enabled": bool(use_coherence),
        "coherence_loss_type": str(coherence_loss_type) if use_coherence else None,
        "n_coherence_batches": int(n_coherence_batches),
        "sep_negatives": str(sep_negatives_mode),
        "sep_positives": str(sep_positives),
        "sep_loss_type": str(sep_loss_type),
        "n_sep_negatives": len(sep_negatives_set),
        "forget_loss_level": str(forget_loss_level),
        "forget_position_weights": (list(w_f) if w_f is not None else None),
        "retain_position_weights": (list(w_r) if w_r is not None else None),
        "deletion_spec": str(deletion_spec),
        "mean_total_loss": _mean(totals["total"]),
        "mean_retain_loss": _mean(totals["retain"]),
        "mean_forget_loss": _mean(totals["forget"]),
        "mean_sep_loss": _mean(totals["sep"]),
        "mean_neighborhood_loss": _mean(totals["coh"]),
        # Alias of mean_neighborhood_loss.
        "mean_coh_loss": _mean(totals["coh"]),
        "n_forget_batches": n_forget,
        "n_retain_batches": n_retain,
        "n_forget_rows": sum(batch_size(b) for b in forget_batches),
        "n_retain_rows": sum(batch_size(b) for b in retain_batches),
    }
