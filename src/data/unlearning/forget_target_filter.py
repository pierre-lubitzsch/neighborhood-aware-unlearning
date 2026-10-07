"""Forget-set preprocessing for item-level unlearning modes.

Two modes are supported:

``deletion_spec='item'``
    Strip target items ``I_f`` from spam session histories and drop rows that
    fall below ``min_sequence_length``.  Produces a sibling dir whose rows
    are spam sessions with ``I_f`` removed.

``deletion_spec='item_pairs'``
    For each spam session ``[i_1,...,i_n]`` and each position ``j`` where
    ``i_j ∈ I_f``, emit one TFRecord entry with
    ``sequence_data = [i_1,...,i_j]``.  The training collate will turn this
    into the (full-prefix, target) training pair ``([i_1,...,i_{j-1}], i_j)``.
    Deduplicates across all source dirs so no exact sequence is written twice.

    With ``--unlearn_whole_items`` (``extra_source_dirs=[retain_dir]``) the
    same extraction is applied to clean sessions as well, unlearning target
    items globally rather than only correcting the spam effect.
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

from src.data.unlearning.split_forget_retain import (
    FORGET_SUBDIR,
    SEQUENCE_FIELD,
    USER_ID_FIELD,
    _infer_feature_description,
    _list_shards,
    _ShardWriter,
)

tf.config.set_visible_devices([], "GPU")


def _filter_sequence(seq: List[int], forbidden: Set[int]) -> List[int]:
    return [int(x) for x in seq if int(x) not in forbidden]


def materialize_item_mode_forget_dir(
    *,
    forget_dir: str,
    out_dir: str,
    target_items: Set[int],
    min_sequence_length: int = 1,
    rows_per_shard: int = 4096,
    overwrite: bool = True,
) -> Dict[str, object]:
    """Write filtered forget shards with ``I_f`` removed from each sequence."""
    if not target_items:
        raise ValueError("target_items must be non-empty for item-mode filtering.")

    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass overwrite=True")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    shards = _list_shards(forget_dir)
    if not shards:
        raise FileNotFoundError(f"No shards under {forget_dir}")

    raw = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    sample = next(iter(raw))
    feat_desc = _infer_feature_description(sample)
    if SEQUENCE_FIELD not in feat_desc:
        raise ValueError(f"{forget_dir}: missing {SEQUENCE_FIELD!r}")

    writer = _ShardWriter(out_dir, "data", rows_per_shard)
    parsed = raw.map(lambda x: (x, tf.io.parse_single_example(x, feat_desc)))

    rows_in = 0
    rows_out = 0
    rows_dropped = 0
    for raw_bytes, example in parsed:
        rows_in += 1
        seq = tf.sparse.to_dense(example[SEQUENCE_FIELD]).numpy()
        if seq.size == 0:
            rows_dropped += 1
            continue
        filtered = _filter_sequence(seq.tolist(), target_items)
        if len(filtered) < min_sequence_length:
            rows_dropped += 1
            continue

        ex = tf.train.Example()
        ex.ParseFromString(raw_bytes.numpy())  # type: ignore[arg-type]
        feat = ex.features.feature
        del feat[SEQUENCE_FIELD]
        feat[SEQUENCE_FIELD].int64_list.value.extend(filtered)
        writer.write(ex.SerializeToString())
        rows_out += 1

    writer.close()
    return {
        "source_forget_dir": os.path.abspath(forget_dir),
        "out_dir": os.path.abspath(out_dir),
        "n_target_items": len(target_items),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_dropped": rows_dropped,
        "min_sequence_length": min_sequence_length,
        "shard_paths": writer.shard_paths,
    }


def default_item_mode_forget_subdir(forget_subdir: str = FORGET_SUBDIR) -> str:
    return f"{forget_subdir}_item_filtered"


def materialize_item_pairs_forget_dir(
    *,
    forget_dir: str,
    out_dir: str,
    target_items: Set[int],
    extra_source_dirs: Optional[List[str]] = None,
    rows_per_shard: int = 4096,
    overwrite: bool = True,
    include_context_rows: bool = False,
) -> Dict[str, object]:
    """Write forget shards with one entry per (full_prefix, target_item) pair.

    For each session ``[i_1,...,i_n]`` from ``forget_dir`` (and optionally
    ``extra_source_dirs``) and each position ``j ≥ 1`` where ``i_j ∈
    target_items``, writes a TFRecord entry with
    ``sequence_data = [i_1,...,i_j]``.  The standard training collate
    (``collate_with_sid_causal_duplicate``) will expand this into training
    pairs ending at ``i_j``, including the full-prefix pair
    ``([i_1,...,i_{j-1}], i_j)``.

    Each pair entry gets a unique synthetic ``user_id`` (0-based counter) so
    that ``index_tfrecord_dir_by_user_id`` stores all of them without
    duplicate-key collisions.  In sequential unlearning the request order
    should be set to ``sorted`` or ``shuffled`` (not ``manifest``) since
    synthetic IDs do not appear in the spam manifest.

    Deduplicates by exact sequence across all source dirs.

    Parameters
    ----------
    forget_dir:
        Primary source directory (spam sessions from ``training_forget``).
    out_dir:
        Output directory for the item-pairs shards.
    target_items:
        Set of item IDs ``I_f`` to unlearn.
    extra_source_dirs:
        Additional source directories to scan, e.g. ``training_retain`` when
        ``unlearn_whole_items=True``.  Pairs from these dirs are included in
        the same dedup set.
    rows_per_shard, overwrite:
        Passed to ``_ShardWriter``.
    include_context_rows:
        Also emit later prefixes in which a target item appears in the input
        history rather than as the label.
    """
    if not target_items:
        raise ValueError("target_items must be non-empty for item_pairs mode.")

    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass overwrite=True")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    source_dirs = [forget_dir] + (list(extra_source_dirs) if extra_source_dirs else [])
    all_shards: List[str] = []
    for src_dir in source_dirs:
        if os.path.isdir(src_dir):
            all_shards.extend(_list_shards(src_dir))
    if not all_shards:
        raise FileNotFoundError(
            f"No .tfrecord.gz shards found in source dirs: {source_dirs}"
        )

    raw_ds = tf.data.TFRecordDataset(all_shards, compression_type="GZIP")
    sample = next(iter(raw_ds))
    feat_desc = _infer_feature_description(sample)
    if SEQUENCE_FIELD not in feat_desc:
        raise ValueError(f"Missing {SEQUENCE_FIELD!r} in source shards")
    if USER_ID_FIELD not in feat_desc:
        raise ValueError(f"Missing {USER_ID_FIELD!r} in source shards")

    parsed_ds = raw_ds.map(lambda x: tf.io.parse_single_example(x, feat_desc))

    writer = _ShardWriter(out_dir, "data", rows_per_shard)
    seen: Set[Tuple[int, ...]] = set()
    synthetic_uid = 0
    rows_in = 0
    rows_out = 0

    for example in parsed_ds:
        rows_in += 1
        seq = tf.sparse.to_dense(example[SEQUENCE_FIELD]).numpy().tolist()

        # By default, emit prefixes whose label is a target item. With
        # include_context_rows, also emit every later prefix, where the target
        # item appears in the input history.
        cut_points = set()
        for j, item in enumerate(seq):
            if int(item) not in target_items:
                continue
            if j >= 1:
                cut_points.add(j)
            if include_context_rows:
                # Start at j + 1 so a target item at j = 0 still yields [i_0, i_1].
                cut_points.update(range(j + 1, len(seq)))

        for j in sorted(cut_points):
            key = tuple(int(x) for x in seq[: j + 1])
            if key in seen:
                continue
            seen.add(key)

            ex = tf.train.Example()
            ex.features.feature[USER_ID_FIELD].int64_list.value.append(synthetic_uid)
            ex.features.feature[SEQUENCE_FIELD].int64_list.value.extend(key)
            writer.write(ex.SerializeToString())
            synthetic_uid += 1
            rows_out += 1

    writer.close()
    return {
        "source_dirs": [os.path.abspath(d) for d in source_dirs],
        "out_dir": os.path.abspath(out_dir),
        "n_target_items": len(target_items),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "n_unique_pairs": rows_out,
        "include_context_rows": bool(include_context_rows),
        "shard_paths": writer.shard_paths,
    }


def default_item_pairs_forget_subdir(forget_subdir: str = FORGET_SUBDIR) -> str:
    return f"{forget_subdir}_item_pairs"
