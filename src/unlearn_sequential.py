"""Hydra entry point for *sequential* TIGER SCIF unlearning."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from typing import Any, Dict, List, Optional, Set

import hydra
import rootutils
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.unlearning.sequential_split import (  # noqa: E402
    TRAINING_FORGET_SUBDIR,
    TRAINING_RETAIN_SUBDIR,
    build_request_batches,
    index_tfrecord_dir_by_user_id,
    materialize_request_dirs,
    order_forget_uids,
)
from src.models.modules.semantic_id.tiger_unlearning_module import (  # noqa: E402
    TigerUnlearningModule,
    save_unlearned_checkpoint,
)
from src.components.unlearning.filter_utils import (  # noqa: E402
    merge_filter_masks,
    save_filter_mask,
)
from src.utils import RankedLogger, extras  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402
from src.utils.unlearn_paths import (  # noqa: E402
    guard_fresh_unlearn_output_dir,
    run_metadata_from_cfg,
)


command_line_logger = RankedLogger(__name__, rank_zero_only=True)
log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")


def _resolve_train_dl_cfg(pipeline_modules: Any, cfg: DictConfig) -> Any:
    dm = pipeline_modules.datamodule
    train_dl_cfg = None
    if hasattr(dm, "stage_to_config"):
        train_dl_cfg = dm.stage_to_config.get(TrainerFn.FITTING)
    if train_dl_cfg is None:
        train_dl_cfg = hydra.utils.instantiate(
            cfg.data_loading.train_dataloader_config.dataloader
        )
    return train_dl_cfg


def _json_safe(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def request_indices_for_checkpoint_fractions(
    n_requests: int,
    fractions: Optional[List[float]],
    *,
    save_all: bool = False,
) -> Set[int]:
    """Return 0-based request indices at which to write checkpoints."""
    if n_requests <= 0:
        return set()
    if save_all:
        return set(range(n_requests))
    if not fractions:
        return {n_requests - 1}
    indices: Set[int] = set()
    for frac in fractions:
        if frac >= 1.0:
            indices.add(n_requests - 1)
        else:
            req_num = max(1, math.ceil(float(frac) * n_requests))
            indices.add(min(req_num - 1, n_requests - 1))
    return indices


def unlearn_sequential(cfg: DictConfig) -> Dict[str, Any]:
    """Run K sequential SCIF updates driven by ``cfg``."""
    guard_fresh_unlearn_output_dir(cfg.paths.output_dir)
    run_meta = run_metadata_from_cfg(cfg)
    command_line_logger.info(
        f"[seq] output_dir={run_meta['hydra_output_dir']} "
        f"(tag={run_meta.get('unlearning_run_tag')}, job={run_meta.get('slurm_job_id')})"
    )

    with pipeline_launcher(cfg) as pipeline_modules:
        model = pipeline_modules.model
        if not isinstance(model, TigerUnlearningModule):
            raise TypeError(
                f"Expected `model._target_` to instantiate TigerUnlearningModule, "
                f"got {type(model).__name__}."
            )

        ckpt_path = cfg.get("ckpt_path", None)
        if not ckpt_path:
            raise ValueError(
                "ckpt_path is required for unlearning; pass the pre-trained "
                "(poisoned) TIGER checkpoint."
            )
        command_line_logger.info(f"Loading TIGER checkpoint from {ckpt_path}")
        source_ckpt = torch.load(
            ckpt_path, map_location="cpu", weights_only=False
        )
        if "state_dict" not in source_ckpt:
            raise KeyError(
                f"Checkpoint at {ckpt_path} has no 'state_dict' key."
            )
        load_result = model.load_state_dict(
            source_ckpt["state_dict"], strict=False
        )
        if load_result.missing_keys:
            command_line_logger.warning(
                f"load_state_dict missing keys ({len(load_result.missing_keys)}): "
                f"{load_result.missing_keys[:5]}"
                f"{'...' if len(load_result.missing_keys) > 5 else ''}"
            )
        if load_result.unexpected_keys:
            command_line_logger.warning(
                f"load_state_dict unexpected keys ({len(load_result.unexpected_keys)}): "
                f"{load_result.unexpected_keys[:5]}"
                f"{'...' if len(load_result.unexpected_keys) > 5 else ''}"
            )

        # Reinitialize selected FFN sub-layers. This must run after load_state_dict,
        # which would otherwise overwrite the reinitialized weights.
        _ffn_reinit = cfg.get("ffn_reinit_layers", None)
        if _ffn_reinit is not None:
            names = model.reinit_ffn_layers(
                OmegaConf.to_container(_ffn_reinit, resolve=True)
                if hasattr(_ffn_reinit, "_content") else _ffn_reinit
            )
            command_line_logger.warning(
                f"[ffn_reinit] reinitialized {len(names)} FFN sub-layer(s) "
                f"after checkpoint load: {names}"
            )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        command_line_logger.info(f"Moving model to {device}")
        model = model.to(device)

        train_dl_cfg = _resolve_train_dl_cfg(pipeline_modules, cfg)

        unlearning_cfg = OmegaConf.to_container(cfg.unlearning, resolve=True)
        data_dir = cfg.paths.data_dir
        forget_subdir = cfg.get("forget_subdir", "training_forget")
        retain_subdir = cfg.get("retain_subdir", "training_retain")
        semantic_id_path = cfg.get("semantic_id_path", None)

        _n_unlearning_chunks_cfg = unlearning_cfg.get("n_unlearning_chunks")
        n_unlearning_chunks: Optional[int] = (
            int(_n_unlearning_chunks_cfg)
            if _n_unlearning_chunks_cfg is not None
            else None
        )
        if n_unlearning_chunks is not None and n_unlearning_chunks <= 0:
            raise ValueError("unlearning.n_unlearning_chunks must be > 0")

        request_batch_size: Optional[int]
        if n_unlearning_chunks is None:
            request_batch_size = int(
                unlearning_cfg.get("request_batch_size") or 0
            )
            if request_batch_size <= 0:
                raise ValueError(
                    "Either unlearning.n_unlearning_chunks or "
                    "unlearning.request_batch_size must be set > 0 for "
                    "sequential unlearning. Use src.unlearn for single-shot."
                )
        else:
            request_batch_size = None  # computed after forget dir is indexed
        max_requests = unlearning_cfg.get("max_requests")
        request_user_order = str(
            unlearning_cfg.get("request_user_order") or "manifest"
        )
        request_seed = int(
            unlearning_cfg.get("request_seed")
            if unlearning_cfg.get("request_seed") is not None
            else cfg.get("seed", 2)
        )
        save_all_intermediate = bool(
            unlearning_cfg.get("save_intermediate_checkpoints", False)
        )
        checkpoint_fractions = unlearning_cfg.get("checkpoint_fractions")
        if checkpoint_fractions is not None:
            checkpoint_fractions = [float(f) for f in checkpoint_fractions]
        rows_per_shard = int(unlearning_cfg.get("rows_per_shard") or 4096)
        cleanup_request_dirs = bool(
            unlearning_cfg.get("cleanup_request_dirs", True)
        )

        full_forget_dir = os.path.join(data_dir, forget_subdir)
        full_retain_dir = os.path.join(data_dir, retain_subdir)

        command_line_logger.info(
            f"[seq] Indexing forget shards under {full_forget_dir} ..."
        )
        forget_index = index_tfrecord_dir_by_user_id(full_forget_dir)
        command_line_logger.info(
            f"[seq] Indexing retain shards under {full_retain_dir} ..."
        )
        retain_index = index_tfrecord_dir_by_user_id(full_retain_dir)
        command_line_logger.info(
            f"[seq] |D_f|={len(forget_index)} forget uids, "
            f"|D_retain|={len(retain_index)} retain uids."
        )

        if n_unlearning_chunks is not None:
            request_batch_size = math.ceil(len(forget_index) / n_unlearning_chunks)
            command_line_logger.info(
                f"[seq] n_unlearning_chunks={n_unlearning_chunks} → "
                f"request_batch_size={request_batch_size} "
                f"(ceil({len(forget_index)}/{n_unlearning_chunks}))."
            )

        manifest_path = cfg.get("forget_manifest", None) or os.path.join(
            data_dir, "forget_manifest.json"
        )
        if not os.path.isfile(manifest_path):
            manifest_path = None
        ordered_forget_uids = order_forget_uids(
            forget_index,
            request_user_order=request_user_order,
            request_seed=request_seed,
            forget_manifest_path=manifest_path,
        )
        assert request_batch_size is not None  # always set by this point
        request_batches = build_request_batches(
            ordered_forget_uids,
            request_batch_size=request_batch_size,
            max_requests=max_requests,
        )
        if not request_batches:
            raise ValueError(
                "No request batches built; check forget_manifest / "
                "request_batch_size / max_requests."
            )
        n_batches = len(request_batches)
        n_forget_users = sum(len(b) for b in request_batches)
        chunk_desc = (
            f"n_unlearning_chunks={n_unlearning_chunks}, batch_size={request_batch_size}"
            if n_unlearning_chunks is not None
            else f"batch_size={request_batch_size}"
        )
        command_line_logger.info(
            f"[seq] Unlearning plan: {n_batches} batches "
            f"({chunk_desc}, {n_forget_users} forget users, "
            f"order={request_user_order})."
        )
        command_line_logger.info(
            f"[seq] Reproducibility: cfg.seed={cfg.get('seed')}, "
            f"request_seed={request_seed}"
        )

        ckpt_request_indices = request_indices_for_checkpoint_fractions(
            n_batches,
            checkpoint_fractions,
            save_all=save_all_intermediate,
        )
        if save_all_intermediate:
            command_line_logger.info(
                f"[seq] Checkpoints: saving after every request "
                f"({len(ckpt_request_indices)} total)."
            )
        elif checkpoint_fractions:
            command_line_logger.info(
                f"[seq] Checkpoints: saving at request indices "
                f"{sorted(ckpt_request_indices)} "
                f"(fractions={checkpoint_fractions})."
            )
        else:
            command_line_logger.info(
                "[seq] Checkpoints: saving only the final unlearned.ckpt."
            )

        out_root = cfg.paths.output_dir
        ckpt_dir = os.path.join(out_root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        per_request: List[Dict[str, Any]] = []
        prev_ckpt_path: Optional[str] = None
        # The filter baseline only installs a decode mask, which is not stored in the
        # checkpoint; the union of masks is saved at the run root for evaluation.
        filter_masks: List[Dict[str, Any]] = []

        for k, forget_uids in enumerate(request_batches):
            batch_label = f"batch {k + 1}/{n_batches}"
            request_dir = os.path.join(out_root, "requests", str(k))
            command_line_logger.info(
                f"[seq] {batch_label}: materialising {len(forget_uids)} "
                f"forget users"
            )
            split_info = materialize_request_dirs(
                request_dir=request_dir,
                forget_index=forget_index,
                retain_index=retain_index,
                forget_uids=forget_uids,
                rows_per_shard=rows_per_shard,
                forget_subdir=TRAINING_FORGET_SUBDIR,
                retain_subdir=TRAINING_RETAIN_SUBDIR,
            )

            retain_subset_dir = os.path.join(request_dir, "retain_subset")
            command_line_logger.info(
                f"[seq] {batch_label}: running "
                f"algorithm={unlearning_cfg.get('algorithm', 'scif')} "
                f"(neighborhood_aware="
                f"{unlearning_cfg.get('neighborhood_aware')}) ..."
            )
            scif_info = model.run_unlearning(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dl_cfg,
                data_dir=request_dir,
                forget_subdir=TRAINING_FORGET_SUBDIR,
                retain_subdir=TRAINING_RETAIN_SUBDIR,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=int(split_info["n_forget_rows_written"]),
                seed=int(cfg.get("seed", 2)),
                num_hierarchies=int(cfg.get("num_hierarchies", 4)),
                device=device,
                output_dir=request_dir,
                forget_manifest_path=manifest_path,
            )

            if scif_info.get("filter_mask"):
                filter_masks.append(scif_info.pop("filter_mask"))

            ckpt_path_k = os.path.join(ckpt_dir, f"unlearned_{k}.ckpt")
            if k in ckpt_request_indices:
                save_unlearned_checkpoint(
                    model=model,
                    out_path=ckpt_path_k,
                    source_ckpt=source_ckpt,
                    extra_metadata={
                        "scif_info": scif_info,
                        "request_idx": k,
                        "forget_uids": list(forget_uids),
                        "n_total_requests": len(request_batches),
                        "request_batch_size": request_batch_size,
                        "request_user_order": request_user_order,
                        "request_seed": request_seed,
                        "source_ckpt_path": os.path.abspath(ckpt_path),
                        "data_dir": os.path.abspath(data_dir),
                        "forget_subdir": forget_subdir,
                        "retain_subdir": retain_subdir,
                        "request_dir": split_info["request_dir"],
                        "unlearning_cfg": unlearning_cfg,
                        "semantic_id_path": (
                            os.path.abspath(semantic_id_path)
                            if semantic_id_path
                            else None
                        ),
                        "num_hierarchies": cfg.get("num_hierarchies", None),
                        "previous_ckpt_path": prev_ckpt_path,
                        **run_meta,
                    },
                )
                prev_ckpt_path = ckpt_path_k

            per_request.append(
                {
                    "request_idx": k,
                    "forget_uids": list(forget_uids),
                    "split_info": split_info,
                    "scif_info": scif_info,
                    "ckpt_path": (
                        os.path.abspath(ckpt_path_k)
                        if k in ckpt_request_indices
                        else None
                    ),
                }
            )

            n_removed = retain_index.remove(forget_uids)
            if n_removed:
                command_line_logger.info(
                    f"[seq] {batch_label}: removed {n_removed} uids from the "
                    f"retain index for the next batch."
                )

            if cleanup_request_dirs and os.path.isdir(request_dir):
                shutil.rmtree(request_dir)

            pct = int(round(100.0 * (k + 1) / n_batches))
            command_line_logger.info(
                f"[seq] {batch_label} done ({pct}% complete)"
            )

        # Union of the per-request masks, at a path the post-unlearn eval can find.
        merged_filter_mask_path: Optional[str] = None
        if filter_masks:
            merged = merge_filter_masks(filter_masks)
            if merged is not None:
                merged_filter_mask_path = save_filter_mask(
                    merged, os.path.join(out_root, "filter_mask.json")
                )
                command_line_logger.info(
                    f"[seq] filter mask written to {merged_filter_mask_path} "
                    f"(mode={merged['filter_mode']}, "
                    f"{len(merged['forbidden_item_ids'])} items, "
                    f"{len(merged.get('user_forget_items') or {})} users, "
                    f"union over {merged['n_merged_requests']} requests). "
                    "Pass decode_filter_mask=<that path> to the eval, or it "
                    "measures the unfiltered model."
                )

        final_ckpt = os.path.join(ckpt_dir, "unlearned.ckpt")
        last_request_ckpt = os.path.join(
            ckpt_dir, f"unlearned_{len(request_batches) - 1}.ckpt"
        )
        if not os.path.isfile(last_request_ckpt):
            save_unlearned_checkpoint(
                model=model,
                out_path=final_ckpt,
                source_ckpt=source_ckpt,
                extra_metadata={
                    "n_total_requests": len(request_batches),
                    "request_batch_size": request_batch_size,
                    "source_ckpt_path": os.path.abspath(ckpt_path),
                    "data_dir": os.path.abspath(data_dir),
                    "forget_subdir": forget_subdir,
                    "retain_subdir": retain_subdir,
                    "unlearning_cfg": unlearning_cfg,
                    "semantic_id_path": (
                        os.path.abspath(semantic_id_path)
                        if semantic_id_path
                        else None
                    ),
                    "num_hierarchies": cfg.get("num_hierarchies", None),
                    **run_meta,
                },
            )
        else:
            try:
                if os.path.lexists(final_ckpt):
                    os.remove(final_ckpt)
                os.symlink(os.path.basename(last_request_ckpt), final_ckpt)
            except OSError:
                shutil.copyfile(last_request_ckpt, final_ckpt)

        info = {
            **run_meta,
            "n_requests": len(request_batches),
            "request_batch_size": request_batch_size,
            "request_user_order": request_user_order,
            "request_seed": request_seed,
            "save_intermediate_checkpoints": save_all_intermediate,
            "checkpoint_fractions": checkpoint_fractions,
            "checkpoint_request_indices": sorted(ckpt_request_indices),
            "cleanup_request_dirs": cleanup_request_dirs,
            "filter_mask_path": merged_filter_mask_path,
            "data_dir": os.path.abspath(data_dir),
            "forget_subdir": forget_subdir,
            "retain_subdir": retain_subdir,
            "semantic_id_path": (
                os.path.abspath(semantic_id_path) if semantic_id_path else None
            ),
            "source_ckpt_path": os.path.abspath(ckpt_path),
            "final_ckpt_path": os.path.abspath(final_ckpt),
            "requests": per_request,
        }
        info_path = os.path.join(out_root, "scif_info.json")
        with open(info_path, "w") as f:
            json.dump(_json_safe(info), f, indent=2)
        command_line_logger.info(
            f"[seq] All {n_batches} batches done. "
            f"Final ckpt -> {final_ckpt}; aggregated info -> {info_path}"
        )
        return info


@hydra.main(version_base="1.3", config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    unlearn_sequential(cfg)


if __name__ == "__main__":
    main()
