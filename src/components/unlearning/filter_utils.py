"""Decode-time filtering utilities for the filter unlearning baseline."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch


def build_filter_mask(
    *,
    deletion_spec: str,
    target_items: Set[int],
    forget_shard_items: Set[int],
    filter_mode: str = "global",
    user_forget_items: Optional[Dict[int, Set[int]]] = None,
) -> Dict[str, Any]:
    """Build a serialisable decode mask specification."""
    mode = str(filter_mode).strip().lower()
    if mode == "global":
        # Block the target items when the manifest lists them; otherwise fall
        # back to every item in the forget shard.
        if target_items:
            forbidden = sorted(int(x) for x in target_items)
        else:
            forbidden = sorted(int(x) for x in forget_shard_items)
        return {
            "filter_mode": "global",
            "deletion_spec": deletion_spec,
            "forbidden_item_ids": forbidden,
            "user_forget_items": None,
        }
    if mode == "user_dependent":
        # Each affected user gets the full forbidden set (not only their own
        # interactions); users absent from the map keep all items.
        forbidden_user = sorted(int(x) for x in (target_items or forget_shard_items))
        serial_user: Dict[str, List[int]] = {}
        if user_forget_items:
            for uid in user_forget_items:
                serial_user[str(int(uid))] = list(forbidden_user)
        return {
            "filter_mode": "user_dependent",
            "deletion_spec": deletion_spec,
            "forbidden_item_ids": forbidden_user,
            "user_forget_items": serial_user,
        }
    raise ValueError(f"Unknown filter_mode={filter_mode!r}")


def save_filter_mask(mask: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(mask, fh, indent=2)
    return path


def load_filter_mask(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def forbidden_sids_from_codebook(
    codebook: torch.Tensor,
    forbidden_item_ids: Iterable[int],
) -> Set[Tuple[int, ...]]:
    """Map sparse item ids to full SID tuples for decode masking."""
    forbidden: Set[Tuple[int, ...]] = set()
    cb = codebook.long()
    if cb.shape[0] < cb.shape[1]:
        cb = cb.t()
    for iid in forbidden_item_ids:
        i = int(iid)
        if 0 <= i < cb.shape[0]:
            forbidden.add(tuple(int(x) for x in cb[i].tolist()))
    return forbidden


def user_forbidden_sids_from_codebook(
    codebook: torch.Tensor,
    user_forget_items: Optional[Dict[Any, Iterable[int]]],
) -> Dict[int, Set[Tuple[int, ...]]]:
    """``user_id -> forbidden SID tuples``, for ``filter_mode='user_dependent'``.

    Like :func:`forbidden_sids_from_codebook` but keeps the per-user partition.
    Keys are coerced to ``int`` since JSON-loaded masks have string keys.
    """
    if not user_forget_items:
        return {}
    cb = codebook.long()
    if cb.shape[0] < cb.shape[1]:
        cb = cb.t()
    out: Dict[int, Set[Tuple[int, ...]]] = {}
    for uid, items in user_forget_items.items():
        sids = {
            tuple(int(x) for x in cb[int(i)].tolist())
            for i in items
            if 0 <= int(i) < cb.shape[0]
        }
        if sids:
            out[int(uid)] = sids
    return out


def merge_filter_masks(
    masks: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Union of a sequence of masks, for the sequential (per-request) driver.

    The final model must honor all requests, so the evaluated mask is the union.
    Global masks union their item lists; user-dependent masks union per user.
    Masks with differing modes or deletion specs are rejected.
    """
    masks = [m for m in masks if m]
    if not masks:
        return None
    modes = {str(m.get("filter_mode", "global")) for m in masks}
    if len(modes) > 1:
        raise ValueError(f"cannot merge filter masks of differing modes: {sorted(modes)}")
    specs = {str(m.get("deletion_spec")) for m in masks}
    if len(specs) > 1:
        raise ValueError(f"cannot merge filter masks of differing deletion specs: {sorted(specs)}")
    forbidden: Set[int] = set()
    per_user: Dict[str, Set[int]] = {}
    for m in masks:
        forbidden.update(int(x) for x in m.get("forbidden_item_ids") or [])
        for uid, items in (m.get("user_forget_items") or {}).items():
            per_user.setdefault(str(int(uid)), set()).update(int(x) for x in items)
    return {
        "filter_mode": modes.pop(),
        "deletion_spec": specs.pop(),
        "forbidden_item_ids": sorted(forbidden),
        "user_forget_items": (
            {uid: sorted(items) for uid, items in sorted(per_user.items(), key=lambda kv: int(kv[0]))}
            if per_user
            else None
        ),
        "n_merged_requests": len(masks),
    }


def scan_user_forget_items(forget_dir: str) -> Dict[int, Set[int]]:
    """Scan forget shards and return ``user_id -> items in sequence``."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    from src.components.unlearning.neighborhood_sampler import (
        SEQUENCE_FIELD,
        USER_ID_FIELD,
        _infer_feature_description,
        _list_shards,
    )

    shards = _list_shards(forget_dir)
    if not shards:
        return {}
    raw = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    sample = next(iter(raw))
    feat_desc = _infer_feature_description(sample)
    parsed = raw.map(lambda x: tf.io.parse_single_example(x, feat_desc))
    out: Dict[int, Set[int]] = {}
    for ex in parsed:
        uid = int(tf.sparse.to_dense(ex[USER_ID_FIELD]).numpy().flatten()[0])
        seq = tf.sparse.to_dense(ex[SEQUENCE_FIELD]).numpy()
        out.setdefault(uid, set()).update(int(x) for x in seq.tolist())
    return out
