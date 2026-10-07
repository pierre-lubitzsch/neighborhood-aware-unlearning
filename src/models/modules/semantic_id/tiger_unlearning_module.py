"""TIGER unlearning Lightning module with multi-algorithm dispatch."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import torch
from torch.utils.data import DataLoader

from src.components.unlearning.filter_utils import (
    build_filter_mask,
    forbidden_sids_from_codebook,
    user_forbidden_sids_from_codebook,
    save_filter_mask,
    scan_user_forget_items,
)
from src.components.unlearning.fanchuan import fanchuan_unlearn
from src.components.unlearning.finetune import finetune_unlearn
from src.components.unlearning.hvp import batch_size as tiger_batch_size
from src.components.unlearning.kookmin import kookmin_unlearn
from src.components.unlearning.neighborhood_sampler import (
    build_retain_subset,
    build_sorted_sid_index,
    closest_prefix_neighbors,
    collect_items_in_shards,
    load_codebook,
    load_dense_embeddings,
    topk_embedding_neighbors,
)
from src.components.unlearning.neg_train import neg_train_unlearn
from src.components.unlearning.scif import scif_unlearn
from src.components.unlearning.seif import seif_unlearn
from src.components.unlearning.target_params import (
    select_code_position_params,
    select_target_params,
)
from src.components.unlearning.unified import unified_unlearn
from src.data.loading.utils import assign_files_to_workers
from src.data.unlearning.deletion_spec import (
    load_forget_manifest,
    load_target_items,
    manifest_deletion_spec,
    resolve_neighborhood_centers,
    resolve_forget_manifest_path,
)
from src.data.unlearning.forget_target_filter import (
    default_item_mode_forget_subdir,
    default_item_pairs_forget_subdir,
    materialize_item_mode_forget_dir,
    materialize_item_pairs_forget_dir,
)
from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDEncoderDecoder,
)
from src.utils.file_utils import list_files

if TYPE_CHECKING:
    from src.data.loading.components.interfaces import SequenceDataloaderConfig


log = logging.getLogger(__name__)

# Algorithms that read unlearning.update_scope. 'scif' is excluded (no second
# derivative through PKM's EmbeddingBag) and 'filter' performs no weight update.
_PKM_SCOPE_ALGOS = frozenset(
    {"unified", "finetune", "neg_train", "kookmin", "fanchuan", "seif"}
)


def _resolve_optimizer(cfg, algo: str, default: str = "adam") -> str:
    """Optimizer name for `algo`, honoring a global fallback.

    Precedence: unlearning.<algo>_optimizer > unlearning.optimizer > default.
    """
    specific = cfg.get(f"{algo}_optimizer")
    if specific:
        return str(specific)
    shared = cfg.get("optimizer")
    return str(shared) if shared else default


class TigerUnlearningModule(SemanticIDEncoderDecoder):
    """Drop-in TIGER subclass exposing multiple unlearning algorithms."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def run_unlearning(
        self,
        *,
        unlearning_cfg: Dict[str, Any],
        train_dataloader_config: "SequenceDataloaderConfig",
        data_dir: str,
        forget_subdir: str,
        retain_subdir: str,
        retain_subset_dir: str,
        semantic_id_path: Optional[str],
        forget_size_hint: Optional[int] = None,
        seed: int = 2,
        num_hierarchies: Optional[int] = None,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None,
        forget_manifest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        algorithm = str(unlearning_cfg.get("algorithm", "scif")).strip().lower()
        # Restricted update scopes are only supported by some algorithms.
        _scope = str(unlearning_cfg.get("update_scope", "all") or "all").strip().lower()
        if _scope not in ("all", "pkm_only", "ffn_only"):
            raise ValueError(
                f"unlearning.update_scope must be 'all', 'pkm_only' or "
                f"'ffn_only', got {_scope!r}"
            )
        if _scope in ("pkm_only", "ffn_only") and algorithm not in _PKM_SCOPE_ALGOS:
            raise ValueError(
                f"unlearning.update_scope={_scope!r} is not supported for "
                f"algorithm={algorithm!r}. Supported: {sorted(_PKM_SCOPE_ALGOS)}. "
                + ("'scif' needs a second derivative, which PKM's "
                   "EmbeddingBag does not provide. "
                   if algorithm == "scif" else "")
                + ("'filter' performs no weight update (it masks forbidden SIDs "
                   "at decode time). "
                   if algorithm == "filter" else "")
            )
        if algorithm == "retrain":
            raise ValueError(
                "algorithm='retrain' is an external baseline; use scripts/pipeline/train_rec.sh "
                "on cleaned/retain data."
            )
        if algorithm == "scif":
            return self.run_scif_unlearning(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dataloader_config,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                retain_subdir=retain_subdir,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                seed=seed,
                num_hierarchies=num_hierarchies,
                device=device,
                forget_manifest_path=forget_manifest_path,
            )
        if algorithm == "finetune":
            return self._run_finetune(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dataloader_config,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                retain_subdir=retain_subdir,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                seed=seed,
                num_hierarchies=num_hierarchies,
                device=device,
                forget_manifest_path=forget_manifest_path,
            )
        if algorithm == "neg_train":
            return self._run_neg_train(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dataloader_config,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                retain_subdir=retain_subdir,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                seed=seed,
                num_hierarchies=num_hierarchies,
                device=device,
                forget_manifest_path=forget_manifest_path,
            )
        if algorithm == "filter":
            return self._run_filter(
                unlearning_cfg=unlearning_cfg,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                output_dir=output_dir,
                forget_manifest_path=forget_manifest_path,
            )
        if algorithm == "unified":
            return self._run_unified(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dataloader_config,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                retain_subdir=retain_subdir,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                seed=seed,
                num_hierarchies=num_hierarchies,
                device=device,
                forget_manifest_path=forget_manifest_path,
            )
        if algorithm in ("kookmin", "fanchuan", "seif", "tracer"):
            runner = {
                "kookmin": self._run_kookmin,
                "fanchuan": self._run_fanchuan,
                "seif": self._run_seif,
                "tracer": self._run_tracer,
            }[algorithm]
            return runner(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dataloader_config,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                retain_subdir=retain_subdir,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                seed=seed,
                num_hierarchies=num_hierarchies,
                device=device,
                forget_manifest_path=forget_manifest_path,
            )
        raise ValueError(f"Unknown unlearning algorithm={algorithm!r}")

    def run_scif_unlearning(
        self,
        *,
        unlearning_cfg: Dict[str, Any],
        train_dataloader_config: "SequenceDataloaderConfig",
        data_dir: str,
        forget_subdir: str,
        retain_subdir: str,
        retain_subset_dir: str,
        semantic_id_path: Optional[str],
        forget_size_hint: Optional[int] = None,
        seed: int = 2,
        num_hierarchies: Optional[int] = None,
        device: Optional[torch.device] = None,
        forget_manifest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        device = device or next(self.parameters()).device
        ctx = self._prepare_unlearning_context(
            unlearning_cfg=unlearning_cfg,
            train_dataloader_config=train_dataloader_config,
            data_dir=data_dir,
            forget_subdir=forget_subdir,
            retain_subdir=retain_subdir,
            retain_subset_dir=retain_subset_dir,
            semantic_id_path=semantic_id_path,
            forget_size_hint=forget_size_hint,
            seed=seed,
            num_hierarchies=num_hierarchies,
            device=device,
            forget_manifest_path=forget_manifest_path,
        )
        forget_batches = ctx["forget_batches"]
        retain_batches = ctx["retain_batches"]
        t0 = time.time()
        cg_solution_max_norm = unlearning_cfg.get("cg_solution_max_norm")
        if cg_solution_max_norm is None:
            cg_solution_max_norm = unlearning_cfg.get("max_norm")
        update_max_norm = unlearning_cfg.get("update_max_norm", 1.0)

        # Optionally confine the update to selected RQ-code positions.
        positions = _resolve_update_positions(
            unlearning_cfg.get("update_positions"),
            int(num_hierarchies or self.num_hierarchies),
        )
        scif_params = None
        scif_grad_masks = None
        if positions is not None:
            scif_params, scif_grad_masks = select_code_position_params(
                self,
                positions=positions,
                update_backbone=bool(
                    unlearning_cfg.get("update_positions_backbone", False)
                ),
            )
            log.info(
                "[scif] position-wise intervention: update_positions=%s "
                "(backbone=%s) -> %d param tensors",
                positions,
                bool(unlearning_cfg.get("update_positions_backbone", False)),
                len(scif_params),
            )

        info = scif_unlearn(
            model=self,
            forget_batches=forget_batches,
            retain_batches=retain_batches,
            forget_size=ctx["forget_size_for_scif"],
            retain_size=ctx["retain_size_full"],
            retain_samples_used_for_update=ctx["retain_samples_used_for_update"],
            cg_max_iter=int(unlearning_cfg.get("cg_max_iter", 200)),
            cg_tol=float(unlearning_cfg.get("cg_tol", 1e-5)),
            cg_damping=float(unlearning_cfg.get("damping", 0.01)),
            target_params_policy=str(unlearning_cfg.get("target_params", "all")),
            params=scif_params,
            grad_masks=scif_grad_masks,
            cg_solution_max_norm=cg_solution_max_norm,
            update_max_norm=update_max_norm,
            eval_mode=bool(unlearning_cfg.get("eval_mode", True)),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        info["algorithm"] = "scif"
        info["update_positions"] = positions
        info["update_positions_backbone"] = (
            bool(unlearning_cfg.get("update_positions_backbone", False))
            if positions is not None
            else None
        )
        return info

    def diagnose_rq_ids(
        self,
        *,
        unlearning_cfg: Dict[str, Any],
        train_dataloader_config: "SequenceDataloaderConfig",
        data_dir: str,
        forget_subdir: str,
        retain_subdir: str,
        retain_subset_dir: str,
        semantic_id_path: Optional[str],
        forget_size_hint: Optional[int] = None,
        seed: int = 2,
        num_hierarchies: Optional[int] = None,
        device: Optional[torch.device] = None,
        forget_manifest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """RQ-ID diagnosis (read-only; no weight update).

        Runs the position-wise gradient signal analysis and/or the static
        code-sharing analysis selected by ``unlearning_cfg['diagnostics_mode']``
        (``both`` | ``positions`` | ``code_sharing``) and returns a combined
        report dict. ``positions`` reuses ``_prepare_unlearning_context`` to
        build the same forget/retain batches the unlearning algorithms use.
        """
        device = device or next(self.parameters()).device
        H = int(num_hierarchies or self.num_hierarchies)
        mode = str(unlearning_cfg.get("diagnostics_mode", "both")).strip().lower()
        if mode not in ("both", "positions", "code_sharing"):
            raise ValueError(
                f"diagnostics_mode={mode!r} must be both|positions|code_sharing"
            )

        manifest_path = forget_manifest_path or resolve_forget_manifest_path(data_dir)
        diag: Dict[str, Any] = {
            "num_hierarchies": H,
            "diagnostics_mode": mode,
            "data_dir": os.path.abspath(data_dir),
            "forget_manifest_path": manifest_path,
            "semantic_id_path": (
                os.path.abspath(semantic_id_path) if semantic_id_path else None
            ),
        }

        if mode in ("positions", "both"):
            from src.components.unlearning.position_diagnostics import (
                per_position_gradient_report,
            )

            ctx = self._prepare_unlearning_context(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dataloader_config,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                retain_subdir=retain_subdir,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                seed=seed,
                num_hierarchies=num_hierarchies,
                device=device,
                forget_manifest_path=forget_manifest_path,
            )
            params = select_target_params(
                self, policy=str(unlearning_cfg.get("target_params", "tiger"))
            )
            diag["position_gradients"] = per_position_gradient_report(
                self,
                ctx["forget_batches"],
                ctx["retain_batches"],
                params,
                num_hierarchies=H,
                eval_mode=bool(unlearning_cfg.get("eval_mode", True)),
            )
            diag["target_items"] = ctx["meta"]["target_items"]
            diag["forget_size"] = ctx["forget_size_for_scif"]

        if mode in ("code_sharing", "both"):
            from src.components.unlearning.code_sharing import code_sharing_report

            if not semantic_id_path:
                raise ValueError(
                    "code-sharing analysis requires semantic_id_path (the RQ-ID "
                    "tensor) to be set"
                )
            targets = load_target_items(load_forget_manifest(manifest_path))
            diag["code_sharing"] = code_sharing_report(
                semantic_id_path, sorted(targets), num_hierarchies=H
            )

        return diag

    def _run_finetune(self, **kwargs: Any) -> Dict[str, Any]:
        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()
        info = finetune_unlearn(
            self,
            ctx["retain_batches"],
            steps=int(cfg.get("finetune_steps", 500)),
            # Shared pass-count knob (also read by unified and tracer).
            n_epochs=cfg.get("n_epochs"),
            lr=float(cfg.get("finetune_lr", 1e-3)),
            update_scope=str(cfg.get("update_scope", "all")),
            pkm_update_keys=bool(cfg.get("pkm_update_keys", True)),
            pkm_update_query=bool(cfg.get("pkm_update_query", True)),
            optimizer=_resolve_optimizer(cfg, "finetune"),
            patience=int(cfg.get("finetune_patience", 0) or 0),
            min_delta=float(cfg.get("finetune_min_delta", 0.0) or 0.0),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        return info

    def _run_neg_train(self, **kwargs: Any) -> Dict[str, Any]:
        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()
        info = neg_train_unlearn(
            self,
            ctx["forget_batches"],
            ctx["retain_batches"],
            steps=int(cfg.get("neg_train_steps", 200)),
            n_epochs=cfg.get("n_epochs"),
            lr=float(cfg.get("neg_train_lr", 1e-3)),
            neg_retain_every=int(cfg.get("neg_retain_every", 5)),
            update_scope=str(cfg.get("update_scope", "all")),
            pkm_update_keys=bool(cfg.get("pkm_update_keys", True)),
            pkm_update_query=bool(cfg.get("pkm_update_query", True)),
            optimizer=_resolve_optimizer(cfg, "neg_train"),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        return info

    def _run_unified(self, **kwargs: Any) -> Dict[str, Any]:
        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()
        local_repair = cfg.get("local_repair") or {}
        n_epochs_cfg = cfg.get("n_epochs")

        # Coherence loss L_n: precompute, per forget batch, the neighbor
        # semantic ids of each sample's target item.
        lambda_neighborhood = float(cfg.get("lambda_n", 0.0))
        coherence_neighbors = None
        if lambda_neighborhood != 0.0:
            coherence_neighbors = self._build_coherence_neighbors(
                forget_batches=ctx["forget_batches"],
                semantic_id_path=kwargs.get("semantic_id_path"),
                num_hierarchies=kwargs.get("num_hierarchies"),
                neighborhood_count=int(cfg.get("neighborhood_count", 4)),
                neighborhood_prefix_length=int(
                    cfg.get("neighborhood_prefix_length", 2)
                ),
                exclude_items=ctx["visible_forget_items"],
                # 'target_only' restricts L_n to rows labeled with a deletion target.
                coherence_rows=str(cfg.get("coherence_rows", "target_only")),
                target_items=set(ctx["meta"].get("target_items") or []),
                # 'embedding' uses a fixed-size top-k in the pre-quantization space.
                neighbor_method=str(
                    cfg.get("coherence_neighbor_method", "prefix")
                ),
                embedding_path=cfg.get("embedding_path"),
                embedding_metric=str(cfg.get("coherence_embedding_metric", "cosine")),
                latent_path=cfg.get("coherence_latent_path"),
                union_size=str(cfg.get("coherence_union_size", "full")),
            )

        info = unified_unlearn(
            self,
            ctx["forget_batches"],
            ctx["retain_batches"],
            code_row_keep=self._resolve_code_row_keep(ctx, cfg, **kwargs),
            steps=int(cfg.get("unified_steps", 500)),
            n_epochs=(
                int(n_epochs_cfg) if n_epochs_cfg is not None else None
            ),
            lr=float(cfg.get("unified_lr", 1e-4)),
            # lambda_r weights the retain term; 0.0 with lambda_f > 0 is pure
            # gradient ascent.
            lambda_retain=float(cfg.get("lambda_r", 1.0)),
            lambda_forget=float(cfg.get("lambda_f", 1.0)),
            lambda_sep=float(cfg.get("lambda_s", 0.1)),
            lambda_neighborhood=lambda_neighborhood,
            coherence_neighbors=coherence_neighbors,
            # 'nll': per-neighbor NLL; 'mass': bounded logsumexp over the neighborhood.
            coherence_loss_type=str(cfg.get("coherence_loss_type", "nll")),
            coherence_mass_cap=float(cfg.get("coherence_mass_cap", 0.999)),
            forget_loss_level=str(cfg.get("forget_loss_level", "token")),
            # Per-SID-level weights for the retain and forget terms (None = uniform).
            position_weights=cfg.get("position_weights"),
            forget_position_weights=cfg.get("forget_position_weights"),
            sep_temperature=float(cfg.get("sep_temperature", 0.07)),
            deletion_spec=ctx["deletion_spec"],
            forget_item_ids=ctx["visible_forget_items"],
            neighbor_item_ids=ctx["neighborhood_centers"],
            sep_negative_item_ids=ctx["sep_negative_items"],
            sep_negatives_mode=str(cfg.get("sep_negatives", "forget_target_only")),
            # 'history' (history items) or 'label' (the true next item).
            sep_positives=str(cfg.get("sep_positives", "history")),
            # 'cosine': pooled-encoder similarity; 'generative': sequence log-prob.
            sep_loss_type=str(cfg.get("sep_loss_type", "cosine")),
            sep_gen_temperature=float(cfg.get("sep_gen_temperature", 1.0)),
            local_repair_cfg=local_repair,
            restrict_adaptive_codes=bool(cfg.get("adaptive_codes", False)),
            stable_codes=int(cfg.get("stable_codes", 2)),
            adaptive_update_backbone=bool(cfg.get("adaptive_update_backbone", False)),
            adaptive_adapter=bool(cfg.get("adaptive_adapter", False)),
            # Confine the update to a subset of RQ code positions (overrides
            # adaptive_codes).
            update_positions=_resolve_update_positions(
                cfg.get("update_positions"),
                num_hierarchies=int(self.num_hierarchies),
            ),
            update_positions_backbone=bool(
                cfg.get("update_positions_backbone", False)
            ),
            # 'pkm_only' optimizes only the Product-Key Memory (requires a PKM model).
            update_scope=str(cfg.get("update_scope", "all")),
            pkm_update_keys=bool(cfg.get("pkm_update_keys", True)),
            pkm_update_query=bool(cfg.get("pkm_update_query", True)),
            # top-t memory-slot restriction (requires update_scope=pkm_only)
            slot_selection=str(cfg.get("slot_selection", "none")),
            slot_top_t=int(cfg.get("slot_top_t", 32)),
            slot_lambda=float(cfg.get("slot_lambda", 1.0)),
            slot_mu=float(cfg.get("slot_mu", 5.0)),
            slot_dot_abs=bool(cfg.get("slot_dot_abs", False)),
            optimizer=_resolve_optimizer(cfg, "unified"),
            # LR multiplier for the SID code parameters (1.0 = uniform lr).
            code_lr_scale=float(cfg.get("code_lr_scale", 1.0)),
            # Extra multiplier on levels [stable_codes, H), on top of code_lr_scale.
            adaptive_code_lr_scale=float(cfg.get("adaptive_code_lr_scale", 1.0)),
            stable_code_lr_scale=float(cfg.get("stable_code_lr_scale", 1.0)),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info["update_scope"] = str(cfg.get("update_scope", "all"))
        info["optimizer"] = str(cfg.get("unified_optimizer", "adam"))
        info.update(ctx["meta"])
        return info

    def _resolve_code_row_keep(
        self, ctx: Dict[str, Any], cfg: Any, **kwargs: Any
    ) -> Optional[torch.Tensor]:
        """Row mask over the SID embedding table selecting which code rows may update.

        ``code_row_scope``:

        ``all``                  every row (subject to ``code_row_levels``)
        ``forget``               rows ``h*K + code_h(i)`` for i in the forget set
        ``forget_neighborhood``  the above plus the rows of each forget item's
                                 neighbors

        ``code_row_levels`` further keeps only the rows of the named hierarchies.
        Unlike ``update_positions``, this affects only the SID embedding table,
        not the decoder heads or backbone.

        Returns a ``[H*K]`` float mask (1.0 = may update), or None when nothing
        is restricted.
        """
        # freeze_sid_table pins every SID row. The mask is applied to the
        # post-step delta, so rows stay fixed even under Adam momentum.
        if bool(cfg.get("freeze_sid_table", False)):
            n_rows = int(self.item_sid_embedding_table_encoder.weight.shape[0])
            log.info(
                "[code-row-scope] freeze_sid_table=true: all %d SID rows pinned "
                "(code_row_scope=%r is inert while the table is frozen)",
                n_rows, cfg.get("code_row_scope", "all"),
            )
            return torch.zeros(n_rows, dtype=torch.float32)

        scope = str(cfg.get("code_row_scope", "all")).strip().lower()
        all_scopes = ("all", "", "none")
        if scope not in all_scopes + ("forget", "forget_neighborhood"):
            raise ValueError(
                "code_row_scope must be all | forget | forget_neighborhood, got "
                f"{scope!r}"
            )
        # Same parser as update_positions; None means all levels.
        levels = _resolve_update_positions(
            cfg.get("code_row_levels"), int(self.num_hierarchies)
        )
        if scope in all_scopes and levels is None:
            return None  # no restriction

        codes = self.codebooks.detach().cpu().to(torch.long)            # [N, H]
        n_items, n_hier = int(codes.shape[0]), int(codes.shape[1])
        K = int(self.num_embeddings_per_hierarchy)

        items: Set[int] = set()
        n_forget = 0
        if scope not in all_scopes:
            items = set(int(i) for i in (ctx.get("visible_forget_items") or []))
            if not items:
                items = set(int(i) for i in (ctx["meta"].get("target_items") or []))
            if not items:
                raise ValueError(
                    f"code_row_scope={scope} but the request names no forget items"
                )
            n_forget = len(items)

        n_neighbors = 0
        if scope == "forget_neighborhood":
            from src.components.unlearning.neighborhood_sampler import (
                build_sorted_sid_index,
                closest_prefix_neighbors,
                load_dense_embeddings,
                topk_embedding_neighbors,
            )

            count = int(cfg.get("neighborhood_count", 4))
            method = str(cfg.get("coherence_neighbor_method", "prefix")).lower()
            neigh: Set[int] = set()
            if method == "embedding":
                emb_path = cfg.get("embedding_path")
                if not emb_path:
                    raise ValueError(
                        "code_row_scope=forget_neighborhood with "
                        "coherence_neighbor_method=embedding needs "
                        "unlearning.embedding_path"
                    )
                embs = load_dense_embeddings(str(emb_path))
                for i in sorted(items):
                    neigh.update(
                        topk_embedding_neighbors(
                            i, embs, count,
                            metric=str(cfg.get("coherence_embedding_metric", "cosine")),
                            exclude_ids=items,
                        )
                    )
            else:
                sorted_ids = build_sorted_sid_index(codes)
                prefix_len = int(cfg.get("neighborhood_prefix_length", 2))
                for i in sorted(items):
                    neigh.update(
                        closest_prefix_neighbors(
                            codes, i, count, prefix_len,
                            sorted_ids=sorted_ids, exclude_ids=items,
                        )
                    )
            n_neighbors = len(neigh)
            items |= neigh

        if scope in all_scopes:
            keep = torch.ones(n_hier * K, dtype=torch.float32)
        else:
            idx = torch.tensor(sorted(items), dtype=torch.long)
            if int(idx.max()) >= n_items:
                raise ValueError(
                    f"item id {int(idx.max())} outside the SID tensor "
                    f"({n_items} items)"
                )
            keep = torch.zeros(n_hier * K, dtype=torch.float32)
            for h in range(n_hier):
                keep[h * K + codes[idx, h].unique()] = 1.0
        n_rows_items = int(keep.sum())

        if levels is not None:
            lvl = torch.zeros_like(keep)
            for h in levels:
                lvl[h * K : (h + 1) * K] = 1.0
            keep = keep * lvl
            if float(keep.sum()) == 0.0:
                # An empty mask would freeze the whole table; refuse it.
                raise ValueError(
                    f"code_row_scope={scope} x code_row_levels={levels} selects "
                    "0 rows; use freeze_sid_table=true if that is what you want"
                )

        log.info(
            "[code-row-scope] %s x levels=%s: %d forget + %d neighbor item(s) "
            "-> %d/%d SID rows updatable (%.2f%% of the table; %d before the "
            "level cut)",
            scope, "all" if levels is None else levels,
            n_forget, n_neighbors, int(keep.sum()), keep.numel(),
            100.0 * float(keep.sum()) / keep.numel(), n_rows_items,
        )
        return keep

    def _run_tracer(self, **kwargs: Any) -> Dict[str, Any]:
        """TRACER token-reassignment baseline.

        Requires the codebook the semantic IDs were built from
        (``unlearning.tracer_codebook_ckpt``, alias ``tracer_rqkmeans_ckpt``)
        and the pre-quantization item embeddings (``unlearning.embedding_path``).
        RQ-KMeans and RQ-VAE checkpoints are both supported. Before training, it
        is asserted that ``phi=0`` reproduces the stored codes.
        """
        import torch as _torch

        from src.components.unlearning.tracer import tracer_unlearn
        from src.components.unlearning.tracer_tokenizer import (
            assert_reproduces_sids,
            compute_residuals,
            load_rq_quantizer,
        )
        from src.components.unlearning.neighborhood_sampler import load_dense_embeddings

        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()

        # tracer_rqkmeans_ckpt is an alias of tracer_codebook_ckpt.
        ckpt = cfg.get("tracer_codebook_ckpt") or cfg.get("tracer_rqkmeans_ckpt")
        if not ckpt:
            raise ValueError(
                "algorithm=tracer requires unlearning.tracer_codebook_ckpt (the "
                "RQ-KMeans or RQ-VAE checkpoint holding the codeword centroids)."
            )
        emb_path = cfg.get("embedding_path")
        if not emb_path:
            raise ValueError("algorithm=tracer requires unlearning.embedding_path")

        num_hierarchies = int(kwargs.get("num_hierarchies") or self.num_hierarchies)
        n_levels = int(cfg.get("tracer_levels") or (num_hierarchies - 1))
        front = load_rq_quantizer(str(ckpt), n_levels=n_levels)
        centroids = _torch.stack([c for c in front.centroids])         # [L, K, D]
        # The residual recipe is taken from the checkpoint.
        res_kwargs = dict(
            project=front.project,
            normalize_inputs=front.normalize_inputs,
            normalize_residuals=front.normalize_residuals,
        )

        codes = self.codebooks.t().to(_torch.long).cpu()               # [H, N]
        n_items = int(codes.shape[1])

        # Embedding rows are keyed by raw item id while `codes` uses dense ids
        # 0..N-1, so align them explicitly.
        z_obj = load_dense_embeddings(str(emb_path))
        if hasattr(z_obj, "tensor"):
            id_to_idx = z_obj.item_id_to_idx
            missing = [i for i in range(n_items) if i not in id_to_idx]
            if missing:
                raise ValueError(
                    f"{len(missing)} of {n_items} dense item ids are absent from "
                    f"{emb_path} (first few: {missing[:5]}). The embeddings must "
                    "cover every item in the SID tensor."
                )
            row_idx = _torch.tensor(
                [id_to_idx[i] for i in range(n_items)], dtype=_torch.long
            )
            z = z_obj.tensor[row_idx]
        else:
            z = z_obj
        if int(z.shape[0]) != n_items:
            raise ValueError(
                f"embeddings have {z.shape[0]} rows but the SID tensor has "
                f"{n_items} items"
            )

        # Require phi=0 to reproduce the stored codes (tolerance is the minimum
        # fraction of matching items).
        assert_reproduces_sids(
            z, [centroids[i] for i in range(n_levels)], codes,
            tol=float(cfg.get("tracer_sid_tolerance", 1.0)), **res_kwargs
        )

        target_items = sorted(int(i) for i in (ctx["meta"].get("target_items") or []))
        if not target_items:
            target_items = sorted(int(i) for i in (ctx.get("visible_forget_items") or []))
        if not target_items:
            raise ValueError("tracer found no concept items to reassign")
        concept = _torch.tensor(target_items, dtype=_torch.long)
        all_items = _torch.arange(codes.shape[1])
        keep = _torch.ones(codes.shape[1], dtype=_torch.bool)
        keep[concept] = False
        retain_items = all_items[keep]

        res_levels = compute_residuals(
            z, [centroids[i] for i in range(n_levels)], codes, **res_kwargs
        )
        residuals = _torch.stack([r[concept] for r in res_levels], dim=1)  # [M, L, D]

        # TRACER's neighborhood P(i_T): the K nearest items to each concept item
        # by cosine similarity in the embedding space, excluding the concept set.
        # Built independently of _build_coherence_neighbors.
        k_coh = int(cfg.get("tracer_neighborhood_count", 5))
        neighbor_items_log: Optional[List[List[int]]] = None
        zc = _torch.nn.functional.normalize(z[concept].double(), dim=-1)   # [M, D]
        za = _torch.nn.functional.normalize(z.double(), dim=-1)            # [N, D]
        sim = zc @ za.t()                                                  # [M, N]
        sim[:, concept] = float("-inf")   # exclude the concept set
        k_eff = int(min(k_coh, max(0, sim.shape[1] - int(concept.numel()))))
        if k_eff <= 0:
            concept_neighbor_sids = None
        else:
            nbr_items = sim.topk(k_eff, dim=-1).indices                    # [M, k]
            # Gather the neighbors' full semantic ids (including the dedup digit).
            concept_neighbor_sids = codes.t()[nbr_items].contiguous()      # [M, k, H]
            neighbor_items_log = nbr_items.tolist()
            log.info(
                "[tracer] P(i_T): cosine top-%d over %s for %d concept item(s), "
                "concept set excluded. First concept item %d -> %s",
                k_eff,
                emb_path,
                int(concept.numel()),
                int(concept[0]),
                nbr_items[0].tolist(),
            )
        del sim, zc, za

        info = tracer_unlearn(
            self,
            forget_batches=ctx["forget_batches"],
            retain_batches=ctx["retain_batches"],
            concept_item_ids=concept,
            residuals=residuals,
            centroids=centroids,
            codes=codes,
            retain_item_ids=retain_items,
            concept_neighbor_sids=concept_neighbor_sids,
            steps=cfg.get("tracer_steps", 500),
            n_epochs=cfg.get("n_epochs"),
            lr=float(cfg.get("tracer_lr", 1e-4)),
            phi_lr=float(cfg.get("tracer_phi_lr", 1e-2)),
            tau=float(cfg.get("tracer_temperature", 0.005)),
            lambda_forget=float(cfg.get("tracer_lambda_forget", 1.0)),
            lambda_coherence=float(cfg.get("tracer_lambda_coherence", 1.0)),
            lambda_reg=float(cfg.get("tracer_lambda_reg", 1e-3)),
            selective_update=bool(cfg.get("tracer_selective_update", True)),
            optimizer=_resolve_optimizer(cfg, "tracer", default="sgd"),
            commit=bool(cfg.get("tracer_commit", True)),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info["tracer_codebook_ckpt"] = str(ckpt)
        info["tracer_quantizer"] = front.quantizer
        # Neighbor item ids used for P(i_T).
        info["tracer_coherence_neighbor_items"] = neighbor_items_log
        info["tracer_coherence_metric"] = "cosine_topk_on_concept_items"
        info.update(ctx["meta"])
        return info

    def _build_coherence_neighbors(
        self,
        *,
        forget_batches: List[Any],
        semantic_id_path: Optional[str],
        num_hierarchies: Optional[int],
        neighborhood_count: int,
        neighborhood_prefix_length: int,
        exclude_items: Set[int],
        coherence_rows: str = "target_only",
        target_items: Optional[Set[int]] = None,
        neighbor_method: str = "prefix",
        embedding_path: Optional[str] = None,
        embedding_metric: str = "cosine",
        latent_path: Optional[str] = None,
        union_size: str = "full",
    ) -> List[Optional[Any]]:
        """Per-forget-batch neighbor semantic ids for the coherence loss.

        For each eligible forget sample ``(H_f, i_T)``, resolve the label item
        ``i_T`` from its label semantic id, then look up the
        ``neighborhood_count`` closest catalog items by shared SID prefix (length
        ``>= neighborhood_prefix_length``), excluding forget items. Returns a
        list aligned to ``forget_batches``; each element is
        ``(neighbor_sids[B, C, H], neighbor_mask[B, C])`` or ``None`` when a
        batch has no eligible neighbors anywhere.

        ``coherence_rows`` selects the eligible forget rows: ``target_only``
        (default) keeps only rows whose label item is in ``target_items``;
        ``all`` keeps every forget row. Since the collate expands each session
        into all prefixes, most rows under ``all`` are labeled with non-target
        items.
        """
        rows_mode = str(coherence_rows).lower()
        if rows_mode not in ("target_only", "all"):
            raise ValueError(
                f"coherence_rows must be 'target_only'|'all', got {coherence_rows!r}"
            )
        nbr_method = str(neighbor_method).lower()
        if nbr_method not in ("prefix", "embedding", "latent", "embedding+latent"):
            raise ValueError(
                "coherence_neighbor_method must be "
                "'prefix'|'embedding'|'latent'|'embedding+latent', got "
                f"{neighbor_method!r}"
            )
        needs_embedding = nbr_method in ("embedding", "embedding+latent")
        needs_latent = nbr_method in ("latent", "embedding+latent")
        if needs_embedding and not embedding_path:
            raise ValueError(
                f"coherence_neighbor_method={nbr_method!r} requires "
                "unlearning.embedding_path (pre-quantization item embeddings)."
            )
        if needs_latent and not latent_path:
            raise ValueError(
                f"coherence_neighbor_method={nbr_method!r} requires "
                "unlearning.coherence_latent_path (the [N, d_z] refined-latent "
                "precomputed latent tensor)."
            )
        union_mode = str(union_size).lower()
        if union_mode not in ("full", "matched"):
            raise ValueError(
                f"coherence_union_size must be 'full'|'matched', got {union_size!r}"
            )
        eligible = {int(x) for x in (target_items or set())}
        if rows_mode == "target_only" and not eligible:
            raise ValueError(
                "coherence_rows='target_only' needs a non-empty target item set "
                "(manifest target_items / visible forget items). Pass "
                "coherence_rows='all' to score every forget row instead."
            )
        if not semantic_id_path:
            raise ValueError(
                "lambda_n > 0 (coherence loss) requires semantic_id_path "
                "(merged_predictions_tensor.pt) to define SID neighbors."
            )
        codebook = load_codebook(semantic_id_path, num_hierarchies=num_hierarchies)
        num_items, H = int(codebook.shape[0]), int(codebook.shape[1])
        count = int(neighborhood_count)

        # SID tuple -> item id (bijective; the dedup digit makes codes unique).
        sid_to_item: Dict[tuple, int] = {
            tuple(int(x) for x in codebook[i].tolist()): i for i in range(num_items)
        }
        sorted_ids = build_sorted_sid_index(codebook)
        sorted_sids = codebook.numpy()[sorted_ids]
        exclude = {int(x) for x in (exclude_items or set())}

        # Per-source top-k budget. For the union, 'full' gives each source
        # `count` neighbors; 'matched' splits one `count` budget between them.
        if nbr_method == "embedding+latent":
            if union_mode == "full":
                k_emb = k_lat = count
            else:
                k_emb = (count + 1) // 2
                k_lat = count - k_emb
        else:
            k_emb = count if nbr_method == "embedding" else 0
            k_lat = count if nbr_method == "latent" else 0
        # Neighbor slots allocated per forget sample.
        count_alloc = (k_emb + k_lat) if nbr_method == "embedding+latent" else count

        embeddings = None
        if needs_embedding:
            embeddings = load_dense_embeddings(embedding_path)
            # Embeddings are indexed by row and must be row-aligned with the codebook.
            if len(embeddings) != num_items:
                raise ValueError(
                    f"embedding_path has {len(embeddings)} items but the "
                    f"codebook has {num_items}: they must be row-aligned "
                    f"(same catalog, same order) for coherence_neighbor_method="
                    f"'embedding'. Check that {embedding_path!r} is the "
                    f"pre-quantization tensor the SID codebook was built from."
                )
            log.info(
                "[unified] coherence L_n neighbors: embedding top-k "
                "(metric=%s, k=%d) from %s [%d items, dim %d]",
                embedding_metric,
                k_emb,
                embedding_path,
                len(embeddings),
                int(embeddings.shape[1]),
            )

        # Refined latent space z: an [N, d] tensor whose row i is item i.
        latents = None
        if needs_latent:
            latents = load_dense_embeddings(latent_path)
            if len(latents) != num_items:
                raise ValueError(
                    f"coherence_latent_path has {len(latents)} items but the "
                    f"codebook has {num_items}: the refined latent tensor is "
                    f"indexed by row and must be row-aligned with the codebook. "
                    f"Check that {latent_path!r} was produced by "
                    "scripts/train_latent_refiner.py against this SID tensor."
                )
            log.info(
                "[unified] coherence L_n neighbors: latent top-k "
                "(metric=cosine, k=%d) from %s [%d items, dim %d]",
                k_lat,
                latent_path,
                len(latents),
                int(latents.shape[1]),
            )

        # item id -> neighbor SID rows [k, H] (cached; forget targets repeat).
        neighbor_cache: Dict[int, List[List[int]]] = {}

        def _neighbor_rows(item_id: int) -> List[List[int]]:
            if item_id not in neighbor_cache:
                # item_id and the returned neighbors are codebook row indices.
                nbr_ids: List[int] = []
                if k_emb > 0:
                    nbr_ids.extend(
                        topk_embedding_neighbors(
                            item_id,
                            embeddings,
                            k_emb,
                            metric=embedding_metric,
                            exclude_ids=exclude,
                            by_row=True,
                        )
                    )
                if k_lat > 0:
                    # Cosine top-k on z over non-forget items.
                    nbr_ids.extend(
                        topk_embedding_neighbors(
                            item_id,
                            latents,
                            k_lat,
                            metric="cosine",
                            exclude_ids=exclude,
                            by_row=True,
                        )
                    )
                if not (k_emb or k_lat):
                    nbr_ids = list(
                        closest_prefix_neighbors(
                            codebook,
                            item_id,
                            count,
                            neighborhood_prefix_length,
                            sorted_ids=sorted_ids,
                            sorted_sids=sorted_sids,
                            exclude_ids=exclude,
                        )
                    )
                # Order-preserving dedup so no neighbor is counted twice.
                seen: Set[int] = set()
                deduped = []
                for n in nbr_ids:
                    if int(n) not in seen:
                        seen.add(int(n))
                        deduped.append(int(n))
                neighbor_cache[item_id] = [
                    [int(x) for x in codebook[n].tolist()] for n in deduped
                ]
            return neighbor_cache[item_id]

        out: List[Optional[Any]] = []
        n_samples = 0
        n_eligible = 0
        n_with_neighbors = 0
        n_targets_missing = 0
        n_neighbors_total = 0
        for model_input, label_data in forget_batches:
            bsz = int(model_input.mask.size(0))
            fut_ids = None
            for label in label_data.labels:
                fut_ids = label_data.labels[label].reshape(bsz, -1)
            neighbor_sids = torch.zeros((bsz, count_alloc, H), dtype=torch.long)
            neighbor_mask = torch.zeros((bsz, count_alloc), dtype=torch.float32)
            for row in range(bsz):
                n_samples += 1
                sid_tuple = tuple(int(x) for x in fut_ids[row, :H].tolist())
                item_id = sid_to_item.get(sid_tuple)
                if item_id is None:
                    n_targets_missing += 1
                    continue
                # Skip rows whose label is not a deletion target.
                if rows_mode == "target_only" and item_id not in eligible:
                    continue
                n_eligible += 1
                rows = _neighbor_rows(item_id)
                if rows:
                    n_with_neighbors += 1
                    n_neighbors_total += min(len(rows), count_alloc)
                for c, sid_row in enumerate(rows[:count_alloc]):
                    neighbor_sids[row, c] = torch.tensor(sid_row, dtype=torch.long)
                    neighbor_mask[row, c] = 1.0
            out.append(
                (neighbor_sids, neighbor_mask)
                if float(neighbor_mask.sum()) > 0
                else None
            )

        log.info(
            "[unified] coherence L_n: method=%s rows=%s, %d/%d forget rows "
            "eligible, %d of those have >=1 neighbor (count=%d, "
            "min_prefix_length=%s, %d labels not found in codebook)",
            nbr_method,
            rows_mode,
            n_eligible,
            n_samples,
            n_with_neighbors,
            count,
            int(neighborhood_prefix_length) if nbr_method == "prefix" else "n/a",
            n_targets_missing,
        )
        log.info(
            "[unified] coherence L_n neighborhood size: k_emb=%d k_lat=%d "
            "alloc=%d union_size=%s, realized mean %.2f neighbors per scored "
            "row (%d total over %d rows with >=1)",
            k_emb,
            k_lat,
            count_alloc,
            union_mode if nbr_method == "embedding+latent" else "n/a",
            n_neighbors_total / max(n_with_neighbors, 1),
            n_neighbors_total,
            n_with_neighbors,
        )
        if n_with_neighbors == 0:
            if nbr_method == "prefix":
                log.warning(
                    "[unified] coherence L_n is zero: no eligible forget row "
                    "has a catalog neighbor sharing a prefix of length >=%d, so "
                    "lambda_n has no effect. Lower "
                    "unlearning.neighborhood_prefix_length or use "
                    "coherence_neighbor_method=embedding.",
                    int(neighborhood_prefix_length),
                )
            else:
                log.warning(
                    "[unified] coherence L_n is zero (method=%s): no eligible "
                    "forget row was found (check coherence_rows/target_items).",
                    nbr_method,
                )
        return out

    def _run_kookmin(self, **kwargs: Any) -> Dict[str, Any]:
        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()
        info = kookmin_unlearn(
            self,
            ctx["forget_batches"],
            ctx["retain_batches"],
            init_rate=float(cfg.get("kookmin_init_rate", 0.01)),
            neg_grad_sample_size=int(cfg.get("kookmin_neg_grad_sample_size", 128)),
            retain_epochs=int(cfg.get("kookmin_retain_epochs", 1)),
            retain_lr=float(cfg.get("kookmin_retain_lr", 1e-3)),
            scale_for_reinit_params=float(cfg.get("kookmin_scale_for_reinit", 10.0)),
            target_params_policy=str(cfg.get("target_params", "all")),
            update_scope=str(cfg.get("update_scope", "all")),
            pkm_update_keys=bool(cfg.get("pkm_update_keys", True)),
            pkm_update_query=bool(cfg.get("pkm_update_query", True)),
            optimizer=_resolve_optimizer(cfg, "kookmin"),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        return info

    def _run_fanchuan(self, **kwargs: Any) -> Dict[str, Any]:
        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()
        info = fanchuan_unlearn(
            self,
            ctx["forget_batches"],
            ctx["retain_batches"],
            lr=float(cfg.get("fanchuan_lr", 1e-3)),
            uniform_epochs=int(cfg.get("fanchuan_uniform_epochs", 1)),
            contrastive_iters=int(cfg.get("fanchuan_contrastive_iters", 8)),
            contrastive_temperature=float(
                cfg.get("fanchuan_contrastive_temperature", 1.15)
            ),
            retain_epochs_per_iter=int(cfg.get("fanchuan_retain_epochs_per_iter", 1)),
            seed=int(kwargs.get("seed", 2)),
            update_scope=str(cfg.get("update_scope", "all")),
            pkm_update_keys=bool(cfg.get("pkm_update_keys", True)),
            pkm_update_query=bool(cfg.get("pkm_update_query", True)),
            optimizer=_resolve_optimizer(cfg, "fanchuan"),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        return info

    def _run_seif(self, **kwargs: Any) -> Dict[str, Any]:
        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()
        keywords = cfg.get("seif_noise_param_keywords")
        # unlearning.n_epochs, when set, overrides seif_repair_epochs.
        n_epochs_cfg = cfg.get("n_epochs")
        repair_epochs = (
            int(n_epochs_cfg)
            if n_epochs_cfg is not None
            else int(cfg.get("seif_repair_epochs", 4))
        )
        info = seif_unlearn(
            self,
            ctx["retain_batches"],
            ctx["forget_batches"],
            erase_std=float(cfg.get("seif_erase_std", 0.6)),
            erase_std_final=float(cfg.get("seif_erase_std_final", 0.005)),
            repair_epochs=repair_epochs,
            repair_lr=float(cfg.get("seif_repair_lr", 7e-4)),
            weight_decay=float(cfg.get("seif_weight_decay", 5e-4)),
            noise_param_keywords=list(keywords) if keywords else None,
            update_scope=str(cfg.get("update_scope", "all")),
            pkm_update_keys=bool(cfg.get("pkm_update_keys", True)),
            pkm_update_query=bool(cfg.get("pkm_update_query", True)),
            optimizer=_resolve_optimizer(cfg, "seif"),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        return info

    def _run_filter(
        self,
        *,
        unlearning_cfg: Dict[str, Any],
        data_dir: str,
        forget_subdir: str,
        semantic_id_path: Optional[str],
        forget_size_hint: Optional[int] = None,
        output_dir: Optional[str] = None,
        forget_manifest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        manifest_path = forget_manifest_path or resolve_forget_manifest_path(data_dir)
        manifest = load_forget_manifest(manifest_path)
        deletion_spec = manifest_deletion_spec(
            manifest, unlearning_cfg.get("deletion_spec")
        )
        target_items = load_target_items(manifest)
        forget_dir = os.path.join(data_dir, forget_subdir)
        forget_shard_items = collect_items_in_shards(_list_shards_safe(forget_dir))
        visible_forget = resolve_neighborhood_centers(
            deletion_spec=deletion_spec,
            forget_shard_items=forget_shard_items,
            target_items=target_items,
        )
        filter_mode = str(unlearning_cfg.get("filter_mode", "global"))
        user_map = (
            scan_user_forget_items(forget_dir)
            if filter_mode == "user_dependent"
            else None
        )
        mask = build_filter_mask(
            deletion_spec=deletion_spec,
            target_items=target_items,
            forget_shard_items=forget_shard_items,
            filter_mode=filter_mode,
            user_forget_items=user_map,
        )
        installed = False
        if semantic_id_path:
            codebook = load_codebook(semantic_id_path)
            forbidden_sids = forbidden_sids_from_codebook(
                codebook, mask["forbidden_item_ids"]
            )
            self.set_decode_filter(
                forbidden_sids=forbidden_sids,
                filter_mode=filter_mode,
                user_forbidden_sids=user_forbidden_sids_from_codebook(
                    codebook, user_map
                ),
            )
            installed = True
        else:
            log.warning(
                "[filter] semantic_id_path is unset, so no decode mask was "
                "installed; the model is unfiltered."
            )
        mask_path = os.path.join(output_dir or ".", "filter_mask.json")
        save_filter_mask(mask, mask_path)
        return {
            "algorithm": "filter",
            "deletion_spec": deletion_spec,
            "filter_mode": filter_mode,
            "filter_mask_path": os.path.abspath(mask_path),
            # The mask is module state, not checkpoint state, so downstream
            # evaluation must reinstall it from this entry.
            "filter_mask": mask,
            "decode_filter_installed": installed,
            "n_forbidden_items": len(mask["forbidden_item_ids"]),
            "n_filtered_users": len(user_map or {}),
            "forget_size_input": forget_size_hint,
            "visible_forget_items": sorted(visible_forget),
        }

    def diagnose_pkm_slots(
        self,
        *,
        top_t: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Per-slot access statistics for every PKM, forget vs retain (read-only).

        Measures whether forget and retain interactions route to disjoint
        memory slots, and computes per-slot selection scores:

        * ``AF``: access frequency on the forget set.
        * ``AF-IHF``: ``AF(s) * log((T_r + 1) / (HF(s) + 1))``, where ``HF`` is
          the retain access count and ``T_r`` the total retain reads.

        Gradient-based scores are reported as well.

        Nothing is updated; the model is only run forward.
        """
        from src.models.components.network_blocks.product_key_memory import (
            HashingMemory,
        )

        tops = [int(t) for t in (top_t or [25, 50, 100, 200, 500, 1000])]
        mems = [
            (n, m) for n, m in self.named_modules() if isinstance(m, HashingMemory)
        ]
        if not mems:
            raise ValueError(
                "diagnose_pkm_slots requires a PKM-bearing model; pass the same "
                "model.pkm_layers / model.pkm_mode the checkpoint was trained with."
            )

        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device

        def _sweep(batches: List[Any]) -> Dict[str, Any]:
            for _, m in mems:
                m.enable_access_counting()
                m.reset_access_counts()
            was_training = self.training
            self.eval()
            with torch.no_grad():
                for b in batches:
                    self.model_step(*b)
            out = {n: m.get_access_counts() for n, m in mems}
            for _, m in mems:
                m.disable_access_counting()
            if was_training:
                self.train()
            return out

        log.info(
            "[pkm-slots] sweeping %d forget / %d retain batches over %d memories",
            len(ctx["forget_batches"]), len(ctx["retain_batches"]), len(mems),
        )
        af_all = _sweep(ctx["forget_batches"])
        hf_all = _sweep(ctx["retain_batches"])

        # ---- per-slot gradient signal -------------------------------------
        # Accumulates the gradient of the summed loss w.r.t. values.weight
        # (size, v_dim); per-slot row norms are taken afterwards.
        def _grad_sweep(batches: List[Any]) -> Dict[str, torch.Tensor]:
            acc: Dict[str, torch.Tensor] = {
                n: torch.zeros_like(m.values.weight) for n, m in mems
            }
            was_training = self.training
            self.eval()  # no dropout noise; grads still flow
            for b in batches:
                self.zero_grad(set_to_none=True)
                _, loss = self.model_step(*b)
                loss.backward()
                for n, m in mems:
                    g = m.values.weight.grad
                    if g is not None:
                        acc[n] += g.detach()
            self.zero_grad(set_to_none=True)
            if was_training:
                self.train()
            return acc

        gf_vec = _grad_sweep(ctx["forget_batches"])
        gr_vec = _grad_sweep(ctx["retain_batches"])

        per_mem: Dict[str, Any] = {}
        for name, mem in mems:
            af = af_all[name][0].double()
            hf = hf_all[name][0].double()
            n_slots = int(af.numel())
            t_f = float(af.sum().item())
            t_r = float(hf.sum().item())
            f_touch = af > 0
            r_touch = hf > 0
            inter = int((f_touch & r_touch).sum().item())
            union = int((f_touch | r_touch).sum().item())

            # AF-IHF: high forget access, low retain ("history") access.
            ihf = torch.log((t_r + 1.0) / (hf + 1.0))
            af_ihf = af * ihf

            # gradient-based scores: g_f, g_r, and g_f - lambda*g_r
            gf = gf_vec[name]
            gr = gr_vec[name]
            gf_n = gf.norm(dim=1).double()
            gr_n = gr.norm(dim=1).double()
            denom = (gf.norm(dim=1) * gr.norm(dim=1)).clamp_min(1e-12)
            cos_fr = ((gf * gr).sum(dim=1) / denom).double()
            # ---- selection scores -------------------------------------
            # The first-order change in retain loss from editing slot i along
            # +g_f is <g_f,i, g_r,i> (signed). Combined score, with each term
            # max-normalized:  s_i = gf_i - lambda * gr_i - mu * dot_i.
            # The 'dotabs' variant penalizes |dot| instead.
            def _nrm(v: torch.Tensor) -> torch.Tensor:
                m = v.abs().max()
                return v / m if float(m) > 0 else v

            dot_fr = (gf * gr).sum(dim=1).double()
            gf_hat, gr_hat = _nrm(gf_n), _nrm(gr_n)
            dot_hat = _nrm(dot_fr)
            grad_scores = {}
            for lam in (0.0, 1.0):
                grad_scores[f"lam{lam}"] = gf_n - float(lam) * gr_n  # magnitude-only
            for lam in (0.0, 1.0):
                for mu in (1.0, 5.0):
                    grad_scores[f"lam{lam}_mu{mu}"] = (
                        gf_hat - float(lam) * gr_hat - float(mu) * dot_hat
                    )
                    grad_scores[f"lam{lam}_mu{mu}_dotabs"] = (
                        gf_hat - float(lam) * gr_hat - float(mu) * dot_hat.abs()
                    )

            entry: Dict[str, Any] = {
                "n_slots": n_slots,
                "grad": {
                    "forget_grad_norm_total": float(gf_n.sum().item()),
                    "retain_grad_norm_total": float(gr_n.sum().item()),
                    "slots_with_forget_grad": int((gf_n > 0).sum().item()),
                    "slots_with_retain_grad": int((gr_n > 0).sum().item()),
                    # mean cosine over slots touched by both objectives
                    "mean_fr_cosine_on_shared": float(
                        cos_fr[(gf_n > 0) & (gr_n > 0)].mean().item()
                    ) if int(((gf_n > 0) & (gr_n > 0)).sum().item()) else None,
                    "mean_dot_fr": float(dot_fr.mean().item()),
                    "frac_slots_dot_negative": float(
                        (dot_fr < 0).double().mean().item()
                    ),
                },
                "forget_reads": t_f,
                "retain_reads": t_r,
                "forget_slots_touched": int(f_touch.sum().item()),
                "retain_slots_touched": int(r_touch.sum().item()),
                "forget_coverage": float(f_touch.sum().item()) / n_slots,
                "retain_coverage": float(r_touch.sum().item()) / n_slots,
                "touched_jaccard": (inter / union) if union else 0.0,
                # forget-hit slots the retain set never reads
                "forget_exclusive_slots": int((f_touch & ~r_touch).sum().item()),
                "forget_exclusive_frac_of_forget": (
                    float((f_touch & ~r_touch).sum().item())
                    / max(1, int(f_touch.sum().item()))
                ),
                "top_t": {},
            }
            for t in tops:
                k = min(t, n_slots)
                af_top = torch.topk(af, k).indices
                ihf_top = torch.topk(af_ihf, k).indices
                hf_top = torch.topk(hf, k).indices
                af_set, ihf_set, hf_set = set(af_top.tolist()), set(ihf_top.tolist()), set(hf_top.tolist())
                entry["top_t"][str(t)] = {
                    "af_vs_afihf_overlap": len(af_set & ihf_set) / max(1, k),
                    "af_top_in_retain_top": len(af_set & hf_set) / max(1, k),
                    "afihf_top_in_retain_top": len(ihf_set & hf_set) / max(1, k),
                    # fraction of selected slots the retain set never touches
                    "af_top_retain_unused": float(
                        (hf[af_top] == 0).sum().item()
                    ) / max(1, k),
                    "afihf_top_retain_unused": float(
                        (hf[ihf_top] == 0).sum().item()
                    ) / max(1, k),
                    "afihf_top_slot_ids": ihf_top[: min(k, 32)].tolist(),
                }
                # gradient-criterion selections at the same cutoff
                for sname, sc in grad_scores.items():
                    g_top = torch.topk(sc, k).indices
                    g_set = set(g_top.tolist())
                    entry["top_t"][str(t)][f"grad_{sname}"] = {
                        # mean forget/retain gradient dot product on selected slots
                        "mean_dot_selected": float(dot_fr[g_top].mean().item()),
                        "overlap_with_af": len(g_set & af_set) / max(1, k),
                        "overlap_with_afihf": len(g_set & ihf_set) / max(1, k),
                        "retain_unused": float(
                            (hf[g_top] == 0).sum().item()
                        ) / max(1, k),
                        "mean_gf_selected": float(gf_n[g_top].mean().item()),
                        "mean_gr_selected": float(gr_n[g_top].mean().item()),
                    }
            per_mem[name] = entry

        return {
            "diagnostic": "pkm_slots",
            "n_memories": len(mems),
            "top_t": tops,
            "n_forget_batches": len(ctx["forget_batches"]),
            "n_retain_batches": len(ctx["retain_batches"]),
            "retain_source": ctx["meta"].get("retain_source"),
            "per_memory": per_mem,
            "meta": ctx["meta"],
        }

    def _prepare_unlearning_context(
        self,
        *,
        unlearning_cfg: Dict[str, Any],
        train_dataloader_config: "SequenceDataloaderConfig",
        data_dir: str,
        forget_subdir: str,
        retain_subdir: str,
        retain_subset_dir: str,
        semantic_id_path: Optional[str],
        forget_size_hint: Optional[int] = None,
        seed: int = 2,
        num_hierarchies: Optional[int] = None,
        device: Optional[torch.device] = None,
        forget_manifest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        device = device or next(self.parameters()).device
        manifest_path = forget_manifest_path or resolve_forget_manifest_path(data_dir)
        manifest = load_forget_manifest(manifest_path)
        deletion_spec = manifest_deletion_spec(
            manifest, unlearning_cfg.get("deletion_spec")
        )
        target_items = load_target_items(manifest)

        forget_dir = os.path.join(data_dir, forget_subdir)
        retain_dir = os.path.join(data_dir, retain_subdir)

        if deletion_spec == "item" and target_items:
            filtered_subdir = default_item_mode_forget_subdir(forget_subdir)
            filtered_dir = os.path.join(data_dir, filtered_subdir)
            if not _list_shards_safe(filtered_dir):
                materialize_item_mode_forget_dir(
                    forget_dir=forget_dir,
                    out_dir=filtered_dir,
                    target_items=target_items,
                    rows_per_shard=int(unlearning_cfg.get("rows_per_shard", 4096)),
                )
            forget_dir = filtered_dir

        if deletion_spec == "item_pairs" and target_items:
            item_pairs_subdir = default_item_pairs_forget_subdir(forget_subdir)
            item_pairs_dir = os.path.join(data_dir, item_pairs_subdir)
            unlearn_whole_items = bool(unlearning_cfg.get("unlearn_whole_items", False))
            extra_dirs: Optional[List[str]] = [retain_dir] if unlearn_whole_items else None
            if not _list_shards_safe(item_pairs_dir):
                log.info(
                    "[item_pairs] materializing (prefix→target) pairs "
                    "from %s (unlearn_whole_items=%s)",
                    forget_dir,
                    unlearn_whole_items,
                )
                materialize_item_pairs_forget_dir(
                    forget_dir=forget_dir,
                    out_dir=item_pairs_dir,
                    target_items=target_items,
                    extra_source_dirs=extra_dirs,
                    rows_per_shard=int(unlearning_cfg.get("rows_per_shard", 4096)),
                    include_context_rows=bool(
                        unlearning_cfg.get("include_context_rows", False)
                    ),
                )
            forget_dir = item_pairs_dir

        if forget_size_hint is None:
            forget_size_hint = _count_rows_in_tfrecord_dir(forget_dir)
        if forget_size_hint <= 0:
            raise ValueError(f"Could not infer |D_f| from {forget_dir}")

        forget_shard_items = collect_items_in_shards(_list_shards_safe(forget_dir))
        visible_forget = resolve_neighborhood_centers(
            deletion_spec=deletion_spec,
            forget_shard_items=forget_shard_items,
            target_items=target_items,
        )

        neighborhood_aware = bool(unlearning_cfg.get("neighborhood_aware", False))
        subset_info = build_retain_subset(
            forget_dir=os.path.join(data_dir, forget_subdir),
            retain_dir=retain_dir,
            out_dir=retain_subset_dir,
            neighborhood_aware=neighborhood_aware,
            semantic_id_path=semantic_id_path,
            sid_prefix_length=int(unlearning_cfg.get("sid_prefix_length", 2)),
            forget_size=forget_size_hint,
            neighbor_aware_factor=float(unlearning_cfg.get("neighbor_aware_factor", 8.0)),
            retain_samples_used_for_update=int(
                unlearning_cfg.get("retain_samples_used_for_update") or 16
            ),
            retain_sample_size=unlearning_cfg.get("retain_sample_size"),
            repair_sample_bound=unlearning_cfg.get("repair_sample_bound"),
            retain_max_rows=unlearning_cfg.get("retain_max_rows"),
            progressive_sid_prefix=bool(unlearning_cfg.get("progressive_sid_prefix", True)),
            neighborhood_aware_sample_rate=float(
                unlearning_cfg.get("neighborhood_aware_sample_rate", 1.0)
            ),
            neighborhood_method=str(unlearning_cfg.get("neighborhood_method", "prefix")),
            embedding_path=unlearning_cfg.get("embedding_path"),
            embedding_epsilon=unlearning_cfg.get("embedding_epsilon"),
            embedding_max_neighbors=int(
                unlearning_cfg.get("embedding_max_neighbors", 100)
            ),
            deletion_spec=deletion_spec,
            target_items=target_items if deletion_spec in ("item", "item_pairs") else None,
            num_hierarchies=num_hierarchies,
            rows_per_shard=int(unlearning_cfg.get("rows_per_shard", 4096)),
            seed=int(seed),
            overwrite=True,
        )

        unlearn_batch_size = unlearning_cfg.get("batch_size_per_device")
        # forget_full_coverage lifts the collate cap and splits the expansion
        # into row chunks, so every forget row is seen once per pass.
        forget_full_coverage = bool(
            unlearning_cfg.get("forget_full_coverage", False)
        )
        _forget_rows_per_batch = int(
            unlearning_cfg.get("forget_rows_per_batch", 512)
        )
        forget_loader = _build_finite_loader(
            base_train_cfg=train_dataloader_config,
            data_folder=forget_dir,
            batch_size_per_device_override=unlearn_batch_size,
            max_batch_size_override=(
                10 ** 9 if forget_full_coverage else None
            ),
        )
        # Retain batch source: 'subset' (default) uses the sampled subset of
        # size retain_samples_used_for_update * |D_f|; 'full' uses the whole
        # retain split.
        retain_source = str(
            unlearning_cfg.get("retain_source", "subset") or "subset"
        ).strip().lower()
        if retain_source not in ("subset", "full"):
            raise ValueError(
                f"unlearning.retain_source must be 'subset' or 'full', got {retain_source!r}"
            )
        retain_loader_dir = retain_dir if retain_source == "full" else retain_subset_dir
        if retain_source == "full":
            algo_name = str(unlearning_cfg.get("algorithm", "scif")).strip().lower()
            log.warning(
                "[retain_source=full] retain batches come from the full retain "
                "split (%s), not the %d-per-|D_f| subset.",
                retain_dir,
                int(unlearning_cfg.get("retain_samples_used_for_update") or 16),
            )
            if algo_name in ("scif", "seif"):
                log.warning(
                    "[retain_source=full] algorithm=%s derives its influence "
                    "scaling from the sampled retain subset "
                    "(retain_count = retain_samples_used_for_update * |D_f|); "
                    "using the full split changes that estimator (%s).",
                    algo_name, algo_name,
                )
        retain_loader = _build_finite_loader(
            base_train_cfg=train_dataloader_config,
            data_folder=retain_loader_dir,
            batch_size_per_device_override=unlearn_batch_size,
        )
        forget_batches = _drain_loader(forget_loader, device=device)
        if forget_full_coverage:
            _n_before = len(forget_batches)
            _rows = sum(tiger_batch_size(b) for b in forget_batches)
            forget_batches = [
                c for b in forget_batches
                for c in _split_batch_rows(b, _forget_rows_per_batch)
            ]
            log.info(
                "[forget_full_coverage] %d rows, no subsampling: %d loader "
                "batch(es) -> %d batches of <=%d rows",
                _rows, _n_before, len(forget_batches), _forget_rows_per_batch,
            )
        retain_batches = _drain_loader(retain_loader, device=device)
        if not forget_batches:
            raise RuntimeError(f"No forget batches from {forget_dir}")
        if not retain_batches:
            raise RuntimeError(f"No retain batches from {retain_loader_dir}")

        retain_size_full = _count_rows_in_tfrecord_dir(retain_dir)
        retain_samples_used = int(unlearning_cfg.get("retain_samples_used_for_update") or 16)

        sep_negative_items = self._sample_sep_random_negatives(
            unlearning_cfg=unlearning_cfg,
            retain_dir=retain_dir,
            exclude_items=visible_forget | target_items,
            default_count=len(visible_forget),
            target_items=target_items,
            seed=int(seed),
        )

        return {
            "forget_batches": forget_batches,
            "retain_batches": retain_batches,
            "forget_size_for_scif": int(forget_size_hint),
            "retain_size_full": int(retain_size_full),
            "retain_samples_used_for_update": retain_samples_used,
            "deletion_spec": deletion_spec,
            "visible_forget_items": visible_forget,
            "neighborhood_centers": visible_forget,
            "sep_negative_items": sep_negative_items,
            "meta": {
                "forget_size_input": int(forget_size_hint),
                "forget_size_augmented": sum(tiger_batch_size(b) for b in forget_batches),
                "retain_size_augmented": sum(tiger_batch_size(b) for b in retain_batches),
                "retain_size_full": int(retain_size_full),
                "retain_subset": subset_info,
                "retain_source": retain_source,
                "retain_loader_dir": retain_loader_dir,
                "neighborhood_aware": neighborhood_aware,
                "deletion_spec": deletion_spec,
                "target_items": sorted(target_items),
            },
        }

    def _sample_sep_random_negatives(
        self,
        *,
        unlearning_cfg: Dict[str, Any],
        retain_dir: str,
        exclude_items: Set[int],
        default_count: int,
        target_items: Optional[Set[int]] = None,
        seed: int,
    ) -> Optional[Set[int]]:
        """Resolve the sep-loss negative item ids from ``sep_negatives``.

        Modes:

        * ``forget`` (default) / ``neighbors`` (legacy alias) → returns ``None``;
          the caller then uses all visible forget items ``I_f`` as negatives
          (every distinct item in the forget shards under ``deletion_spec=session``).
        * ``forget_target_only`` → returns exactly the manifest ``target_items``
          (the ``n_target`` spam targets), independent of ``deletion_spec``.
        * ``random_retain`` → random retain-set item ids (ablation): every item
          id in the retain shards minus forget/target items, sampled to
          ``default_count`` (``sep_num_random_negatives`` overrides). The item
          pool is cached per resolved retain dir.
        """
        mode = str(unlearning_cfg.get("sep_negatives", "forget")).strip().lower()
        # 'neighbors' is a legacy alias for 'forget'.
        if mode in ("", "forget", "neighbors"):
            return None
        if mode == "forget_target_only":
            targets = set(target_items or [])
            if not targets:
                raise ValueError(
                    "sep_negatives='forget_target_only' requires target_items in "
                    "the forget manifest (none found)"
                )
            log.info(
                "[sep_negatives] forget_target_only: using %d target item(s) as "
                "sep-loss negatives",
                len(targets),
            )
            return targets
        if mode != "random_retain":
            raise ValueError(
                "sep_negatives must be 'forget', 'forget_target_only', or "
                f"'random_retain', got {mode!r}"
            )
        pool_key = os.path.realpath(retain_dir)
        cache = getattr(self, "_retain_item_pool_cache", None)
        if cache is None or cache[0] != pool_key:
            cache = (pool_key, collect_items_in_shards(_list_shards_safe(retain_dir)))
            self._retain_item_pool_cache = cache
        pool = sorted(cache[1] - exclude_items)
        if not pool:
            log.warning(
                "[sep_negatives] empty retain item pool after excluding "
                "forget/target items; falling back to neighbor negatives"
            )
            return None
        n_cfg = unlearning_cfg.get("sep_num_random_negatives")
        n = int(n_cfg) if n_cfg is not None else int(default_count)
        n = max(1, min(n, len(pool)))
        sampled = set(random.Random(seed).sample(pool, n))
        log.info(
            "[sep_negatives] random_retain: sampled %d of %d retain items "
            "as sep-loss negatives (default_count=%d)",
            len(sampled),
            len(pool),
            default_count,
        )
        return sampled


def _resolve_update_positions(
    value: Any, num_hierarchies: int
) -> Optional[List[int]]:
    """Parse the ``unlearning.update_positions`` knob into 0-based hierarchy ids.

    Accepts:
      * ``None`` / ``"all"`` / ``"null"`` / ``"none"`` / ``""`` -> ``None`` (no
        restriction).
      * a list of 0-based indices, e.g. ``[0, 1]`` (== c1, c2).
      * a list / comma string of code names, e.g. ``["c1", "c2"]`` or
        ``"c1,c2"`` (1-based names mapped to 0-based indices).
      * a comma/space string of indices, e.g. ``"2,3"``.

    Returns a sorted list of distinct indices, or ``None`` when the selection is
    empty or equals the full set ``{0..H-1}`` (both mean "update everything").
    Do not mix bare numbers and ``cN`` names in one list (numbers are 0-based,
    ``cN`` is 1-based).
    """
    if value is None:
        return None
    items: Any = value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "all", "null", "none"):
            return None
        items = [tok for tok in re.split(r"[,\s]+", s.strip("[]() ")) if tok]
    idxs: List[int] = []
    for it in items:
        if isinstance(it, str):
            tok = it.strip().lower()
            if tok.startswith("c"):
                idxs.append(int(tok[1:]) - 1)  # c1 -> 0
            else:
                idxs.append(int(tok))
        else:
            idxs.append(int(it))
    idxs = sorted(set(idxs))
    if not idxs:
        return None
    for h in idxs:
        if not 0 <= h < num_hierarchies:
            raise ValueError(
                f"update_positions index {h} out of range [0, {num_hierarchies})"
            )
    if len(idxs) == num_hierarchies:
        return None  # full set == no restriction
    return idxs


def _list_shards_safe(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".tfrecord.gz")
    ]


def _count_rows_in_tfrecord_dir(directory: str) -> int:
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
    except Exception as ex:
        raise RuntimeError(f"TensorFlow is required to count TFRecord rows: {ex}")

    shards = _list_shards_safe(directory)
    n = 0
    for path in shards:
        for _ in tf.data.TFRecordDataset([path], compression_type="GZIP"):
            n += 1
    return n


def _build_finite_loader(
    base_train_cfg: "SequenceDataloaderConfig",
    data_folder: str,
    batch_size_per_device_override: Optional[int] = None,
    max_batch_size_override: Optional[int] = None,
) -> DataLoader:
    cfg = deepcopy(base_train_cfg)
    cfg.data_folder = data_folder
    if batch_size_per_device_override is not None:
        cfg.batch_size_per_device = int(batch_size_per_device_override)

    suffix_provider = cfg.dataset_config.data_iterator
    file_suffix = (
        getattr(cfg.dataset_config, "file_format", None)
        or suffix_provider.get_file_suffix()
    )
    files = list_files(folder_path=data_folder, suffix=f"*{file_suffix}")
    file_map, _ = assign_files_to_workers(
        list_of_files=files,
        total_workers=1,
        assign_by_size=False,
        should_shuffle_rows=False,
        assign_all_files_per_worker=False,
    )

    dataset = cfg.dataset_class(
        dataset_config=cfg.dataset_config,
        data_folder=data_folder,
        should_shuffle_rows=False,
        batch_size=cfg.batch_size_per_device,
        is_for_training=False,
        assign_all_files_per_worker=False,
    )
    dataset.set_list_of_files(list_of_files=file_map.get(0, []))
    dataset.set_distributed_params(total_workers=1, global_worker_id=0)

    _collate_kw = dict(
        labels=cfg.labels,
        sequence_length=cfg.sequence_length,
        masking_token=cfg.masking_token,
        padding_token=cfg.padding_token,
        oov_token=cfg.get("oov_token", None) if hasattr(cfg, "get") else None,
    )
    # The collate expands each sequence into all prefixes and samples
    # max_batch_size of them with replacement; callers may lift the cap.
    if max_batch_size_override is not None:
        _collate_kw["max_batch_size"] = int(max_batch_size_override)
    collate_fn_partial = partial(cfg.collate_fn, **_collate_kw)

    return DataLoader(
        dataset=dataset,
        batch_size=(
            cfg.batch_size_per_device if cfg.dataset_config.iterate_per_row else None
        ),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn_partial,
        timeout=0,
    )


def _split_batch_rows(batch: Any, max_rows: int) -> List[Any]:
    """Split one TigerBatch into row-chunks of at most ``max_rows``.

    Every tensor in a TigerBatch is row-major on dim 0 (one row per augmented
    sequence), so a chunk is a slice of each field.
    """
    import dataclasses

    model_input, label_data = batch
    n = tiger_batch_size(batch)
    if n <= max_rows:
        return [batch]

    # Fields with leading dim n are row-major; fields with leading dim n * k
    # (e.g. flattened labels) are sliced proportionally; everything else is
    # passed through unchanged.
    def _slice(v, a, b):
        if isinstance(v, torch.Tensor):
            if v.dim() >= 1 and v.shape[0] % n == 0 and v.shape[0] >= n:
                k = v.shape[0] // n
                return v[a * k:b * k]
            return v
        if isinstance(v, dict):
            return {key: _slice(x, a, b) for key, x in v.items()}
        if isinstance(v, list):
            if len(v) % n == 0 and len(v) >= n:
                k = len(v) // n
                return v[a * k:b * k]
            return v
        return v

    out: List[Any] = []
    for a in range(0, n, max_rows):
        b = min(a + max_rows, n)
        mi = dataclasses.replace(
            model_input,
            **{f.name: _slice(getattr(model_input, f.name), a, b)
               for f in dataclasses.fields(model_input)},
        )
        ld = dataclasses.replace(
            label_data,
            **{f.name: _slice(getattr(label_data, f.name), a, b)
               for f in dataclasses.fields(label_data)},
        )
        out.append((mi, ld))
    return out


def _drain_loader(loader: DataLoader, device: torch.device) -> List[Any]:
    from src.components.unlearning.hvp import batch_to_device

    out: List[Any] = []
    for batch in loader:
        out.append(batch_to_device(batch, device))
    return out


def save_unlearned_checkpoint(
    *,
    model: TigerUnlearningModule,
    out_path: str,
    source_ckpt: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": _safe_lightning_version(),
        "callbacks": {},
        "optimizer_states": [],
        "lr_schedulers": [],
        "hparams_name": "kwargs",
        "hyper_parameters": {},
    }
    if source_ckpt is not None:
        for key in (
            "epoch",
            "global_step",
            "pytorch-lightning_version",
            "callbacks",
            "optimizer_states",
            "lr_schedulers",
            "hparams_name",
            "hyper_parameters",
        ):
            if key in source_ckpt:
                payload[key] = source_ckpt[key]
    if extra_metadata:
        payload["unlearning_metadata"] = _json_safe(extra_metadata)
        payload["scif_metadata"] = _json_safe(extra_metadata)
    torch.save(payload, out_path)
    log.info("[unlearn] saved unlearned checkpoint -> %s", out_path)


def _safe_lightning_version() -> str:
    try:
        import lightning

        return getattr(lightning, "__version__", "unknown")
    except Exception:
        return "unknown"


def _json_safe(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)
