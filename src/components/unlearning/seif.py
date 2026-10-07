"""Seif unlearning (noise + repair), ported from the ERASE benchmark's RecBole
trainer (https://github.com/deem-data/erase-bench/blob/main/recbole/trainer/trainer.py)
and adapted to TIGER batches and the ``model.model_step(...)`` loss.
Seif is distinct from SCIF (``scif.py``). It has two phases:

    Phase 1 (erase)
        Add multiplicative Gaussian noise ``θ += N(0, std) * |θ|`` to the
        parameters selected by name keyword.

    Phase 2 (repair)
        Fine-tune on the retain corpus for ``repair_epochs`` epochs. Before the
        final epoch a smaller noise (``erase_std_final``) is injected again.

By default the SID embedding table and the decoder projection heads are
perturbed. The retain subset contains no forget items, so ERASE's forget-item
loss down-weighting is omitted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from src.components.unlearning.optim_utils import build_optimizer
from src.components.unlearning.target_params import (
    resolve_scope_params,
    select_pkm_params,
)
from src.components.unlearning.hvp import batch_size, batch_to_device

log = logging.getLogger(__name__)
TigerBatch = Any  # noqa: N816

# Default name keywords identifying the parameters to perturb in the erase
# phase. Matches TIGER's `item_sid_embedding_table_encoder` and
# `decoder.decoder_mlp`.
_DEFAULT_NOISE_KEYWORDS = ("embedding", "mlp")


def _add_multiplicative_noise(
    model: nn.Module,
    std: float,
    keywords: Sequence[str],
) -> int:
    """Add ``N(0, std) * |θ|`` to every requires_grad param whose name contains
    one of ``keywords``. Returns the number of tensors perturbed."""
    n = 0
    kws = [k.lower() for k in keywords]
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name.lower() for kw in kws):
                noise = torch.normal(
                    mean=0.0, std=float(std), size=param.shape, device=param.device
                )
                param.data.add_(noise * param.data.abs())
                n += 1
    return n


def seif_unlearn(
    model: nn.Module,
    retain_batches: Sequence["TigerBatch"],
    forget_batches: Optional[Sequence["TigerBatch"]] = None,
    *,
    erase_std: float = 0.6,
    erase_std_final: float = 0.005,
    repair_epochs: int = 4,
    repair_lr: float = 7e-4,
    weight_decay: float = 5e-4,
    noise_param_keywords: Optional[Sequence[str]] = None,
    update_scope: str = "all",
    pkm_update_keys: bool = True,
    pkm_update_query: bool = True,
    optimizer: str = "adam",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run SEIF noise-erase + repair on ``model`` in-place.

    Parameters
    ----------
    model
        TIGER model exposing ``model_step(input, label_data) -> (output, loss)``.
    retain_batches
        Pre-collected retain TIGER batches already on ``device`` (repair set).
    forget_batches
        Unused for the weight update (kept for interface symmetry and logging).
    erase_std
        Std of the erase-phase multiplicative Gaussian noise.
    erase_std_final
        Std of the robustness noise injected before the last repair epoch.
    repair_epochs
        Number of repair passes over ``retain_batches``.
    repair_lr, weight_decay
        Repair optimizer hyperparameters.
    noise_param_keywords
        Name substrings identifying which parameters to perturb. Defaults to
        ``("embedding", "mlp")``. Falls back to all trainable params if no
        name matches (with a warning).
    """
    if not retain_batches:
        raise ValueError("seif_unlearn: no retain batches were provided")

    device = device or next(model.parameters()).device
    model.train()
    pkm_only = str(update_scope or "all").strip().lower() == "pkm_only"
    if pkm_only:
        # Seif matches parameters by name, so use the PKM tensor names as the
        # keyword list.
        _, _pkm_names = select_pkm_params(
            model, include_keys=pkm_update_keys, include_query=pkm_update_query
        )
        keywords = _pkm_names
        log.info(
            "[seif] PKM-ONLY: erase noise restricted to %d PKM tensors "
            "(keys=%s query=%s)",
            len(keywords), bool(pkm_update_keys), bool(pkm_update_query),
        )
    else:
        keywords = list(noise_param_keywords or _DEFAULT_NOISE_KEYWORDS)

    # --- Phase 1: erase ------------------------------------------------------
    n_noised = _add_multiplicative_noise(model, erase_std, keywords)
    if n_noised == 0 and pkm_only:
        raise ValueError(
            "seif update_scope='pkm_only' matched no PKM tensors; refusing to "
            "fall back to perturbing the whole model."
        )
    if n_noised == 0:
        log.warning(
            "[seif] no parameters matched keywords %s; falling back to all "
            "trainable params for the erase noise",
            keywords,
        )
        keywords = None  # marker for fallback
        with torch.no_grad():
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                noise = torch.normal(
                    mean=0.0, std=float(erase_std), size=param.shape, device=param.device
                )
                param.data.add_(noise * param.data.abs())
                n_noised += 1
    log.info(
        "[seif] erase phase: perturbed %d tensors (std=%.3g, keywords=%s)",
        n_noised,
        erase_std,
        keywords if keywords is not None else "ALL",
    )
    fallback_keywords = list(_DEFAULT_NOISE_KEYWORDS) if keywords is None else keywords

    # --- Phase 2: repair -----------------------------------------------------
    params, _ = resolve_scope_params(
        model, update_scope,
        fallback=[p for p in model.parameters() if p.requires_grad],
        include_keys=pkm_update_keys, include_query=pkm_update_query,
        algo="seif",
    )
    opt = build_optimizer(optimizer, params, float(repair_lr),
                          weight_decay=float(weight_decay), algo="seif")
    epoch_mean_losses: List[float] = []
    for repair_epoch in range(int(repair_epochs)):
        losses: List[float] = []
        model.train()
        for batch in retain_batches:
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            _, loss = model.model_step(*batch)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        # robustness noise just before the final epoch
        if repair_epoch == int(repair_epochs) - 2:
            n_robust = _add_multiplicative_noise(
                model, erase_std_final, fallback_keywords
            )
            log.info(
                "[seif] injected robustness noise (std=%.3g) into %d tensors "
                "before final epoch",
                erase_std_final,
                n_robust,
            )

        mean_loss = float(sum(losses) / max(1, len(losses)))
        epoch_mean_losses.append(mean_loss)
        log.info(
            "[seif] repair epoch %d/%d mean_loss=%.4f",
            repair_epoch + 1,
            int(repair_epochs),
            mean_loss,
        )

    return {
        "algorithm": "seif",
        "erase_std": float(erase_std),
        "erase_std_final": float(erase_std_final),
        "repair_epochs": int(repair_epochs),
        "repair_lr": float(repair_lr),
        "weight_decay": float(weight_decay),
        "noise_param_keywords": (
            list(noise_param_keywords or _DEFAULT_NOISE_KEYWORDS)
            if keywords is not None
            else "ALL"
        ),
        "n_tensors_noised": int(n_noised),
        "epoch_mean_losses": epoch_mean_losses,
        "final_repair_loss": epoch_mean_losses[-1] if epoch_mean_losses else None,
        "n_retain_batches": len(retain_batches),
        "n_retain_rows": sum(batch_size(b) for b in retain_batches),
        "n_forget_batches": len(forget_batches) if forget_batches else 0,
    }
