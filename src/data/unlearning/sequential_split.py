"""Helpers for *sequential* SCIF unlearning.

The single-shot pipeline uses ``training_forget/`` and ``training_retain/``
directories produced by :mod:`src.data.unlearning.split_forget_retain`.

For the sequential driver, this module:

1. Scans both directories once and builds in-memory user_id -> raw TFRecord
   bytes maps.
2. For each unlearning request ``k``, writes a per-request
   ``training_forget/`` (B users) and ``training_retain/`` (current retain
   pool minus the request's user_ids) that :class:`TigerUnlearningModule`
   can consume directly.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


SEQUENCE_FIELD = "sequence_data"
USER_ID_FIELD = "user_id"
TRAINING_FORGET_SUBDIR = "training_forget"
TRAINING_RETAIN_SUBDIR = "training_retain"


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level IO helpers (mirror split_forget_retain.py / neighborhood_sampler.py)
# ---------------------------------------------------------------------------


def _list_shards(directory: str) -> List[str]:
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".tfrecord.gz")
    ]


def _infer_feature_description(
    sample_record: tf.Tensor,
) -> Dict[str, tf.io.VarLenFeature]:
    example = tf.train.Example()
    example.ParseFromString(sample_record.numpy())  # type: ignore[arg-type]
    out: Dict[str, tf.io.VarLenFeature] = {}
    for key, value in example.features.feature.items():
        if value.HasField("bytes_list"):
            out[key] = tf.io.VarLenFeature(tf.string)
        elif value.HasField("float_list"):
            out[key] = tf.io.VarLenFeature(tf.float32)
        elif value.HasField("int64_list"):
            out[key] = tf.io.VarLenFeature(tf.int64)
        else:
            raise ValueError(f"Unknown feature type for key {key!r}")
    return out


# ---------------------------------------------------------------------------
# Index a TFRecord directory by user_id
# ---------------------------------------------------------------------------


@dataclass
class UserIdIndex:
    """In-memory mapping ``user_id -> raw TFRecord bytes``."""

    bytes_by_uid: Dict[int, bytes]
    uid_order: List[int]
    n_rows_seen: int
    n_rows_no_user_id: int
    source_dir: str
    initial_uid_count: int

    def __len__(self) -> int:
        return len(self.bytes_by_uid)

    def remove(self, uids: Iterable[int]) -> int:
        """Drop ``uids`` from the index in-place. Returns the number actually removed."""
        n = 0
        for uid in uids:
            if self.bytes_by_uid.pop(int(uid), None) is not None:
                n += 1
        if n:
            existing = set(self.bytes_by_uid)
            self.uid_order = [u for u in self.uid_order if u in existing]
        return n


def index_tfrecord_dir_by_user_id(directory: str) -> UserIdIndex:
    """Walk every shard under ``directory`` once and build a uid -> bytes map.

    Rows missing a ``user_id`` are counted but skipped.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Expected TFRecord directory at {directory}")

    shards = _list_shards(directory)
    if not shards:
        raise FileNotFoundError(f"No .tfrecord.gz shards under {directory}")

    raw = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    sample = next(iter(raw))
    feat_desc = _infer_feature_description(sample)
    if USER_ID_FIELD not in feat_desc:
        raise ValueError(
            f"{directory} shards lack {USER_ID_FIELD!r} feature; "
            f"got {sorted(feat_desc)}"
        )

    bytes_by_uid: Dict[int, bytes] = {}
    uid_order: List[int] = []
    n_rows_seen = 0
    n_rows_no_user_id = 0

    parsed = raw.map(
        lambda x: (x, tf.io.parse_single_example(x, feat_desc))
    )
    for raw_t, ex in parsed:
        n_rows_seen += 1
        uid_dense = tf.sparse.to_dense(ex[USER_ID_FIELD]).numpy()
        if uid_dense.size == 0:
            n_rows_no_user_id += 1
            continue
        uid = int(uid_dense.flatten()[0])
        if uid in bytes_by_uid:
            log.warning(
                "[seq-split] duplicate user_id=%d in %s; keeping first row.",
                uid,
                directory,
            )
            continue
        bytes_by_uid[uid] = raw_t.numpy()
        uid_order.append(uid)

    log.info(
        "[seq-split] indexed %d rows from %s (%d unique uids, %d skipped no-uid).",
        n_rows_seen,
        directory,
        len(bytes_by_uid),
        n_rows_no_user_id,
    )
    return UserIdIndex(
        bytes_by_uid=bytes_by_uid,
        uid_order=uid_order,
        n_rows_seen=n_rows_seen,
        n_rows_no_user_id=n_rows_no_user_id,
        source_dir=os.path.abspath(directory),
        initial_uid_count=len(bytes_by_uid),
    )


# ---------------------------------------------------------------------------
# Shard writer (round-robin across rows_per_shard, mirrors split_forget_retain)
# ---------------------------------------------------------------------------


class _ShardWriter:
    def __init__(self, out_dir: str, basename: str, rows_per_shard: int) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.basename = basename
        self.rows_per_shard = rows_per_shard
        self.options = tf.io.TFRecordOptions(compression_type="GZIP")
        self._shard_idx = 0
        self._row_idx = 0
        self._writer: Optional[tf.io.TFRecordWriter] = None
        self.shard_paths: List[str] = []
        self.total_rows = 0

    def _ensure_writer(self) -> tf.io.TFRecordWriter:
        if self._writer is None or self._row_idx >= self.rows_per_shard:
            if self._writer is not None:
                self._writer.close()
                self._shard_idx += 1
                self._row_idx = 0
            path = os.path.join(
                self.out_dir, f"{self.basename}_{self._shard_idx}.tfrecord.gz"
            )
            self._writer = tf.io.TFRecordWriter(path, options=self.options)
            self.shard_paths.append(path)
        return self._writer

    def write(self, raw_example: bytes) -> None:
        w = self._ensure_writer()
        w.write(raw_example)
        self._row_idx += 1
        self.total_rows += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _write_uid_bytes(
    out_dir: str,
    bytes_by_uid: Dict[int, bytes],
    uid_order: Sequence[int],
    rows_per_shard: int,
) -> Tuple[List[str], int]:
    """Write the rows for ``uid_order`` (in that order) under ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    writer = _ShardWriter(out_dir, "data", rows_per_shard)
    for uid in uid_order:
        raw = bytes_by_uid.get(int(uid))
        if raw is None:
            log.warning("[seq-split] uid=%s missing from index; skipping.", uid)
            continue
        writer.write(raw)
    writer.close()
    return writer.shard_paths, writer.total_rows


# ---------------------------------------------------------------------------
# Public entry point: materialize one request's forget + retain dirs
# ---------------------------------------------------------------------------


def materialize_request_dirs(
    *,
    request_dir: str,
    forget_index: UserIdIndex,
    retain_index: UserIdIndex,
    forget_uids: Sequence[int],
    rows_per_shard: int = 4096,
    forget_subdir: str = TRAINING_FORGET_SUBDIR,
    retain_subdir: str = TRAINING_RETAIN_SUBDIR,
) -> Dict[str, object]:
    """Write per-request ``training_forget/`` and ``training_retain/`` dirs.

    The in-memory indices are not modified; the driver updates the retain index
    (``retain_{k+1} = retain_k \\ forget_k``) after this call.
    """
    if os.path.exists(request_dir):
        raise FileExistsError(
            f"request_dir {request_dir} already exists; pick a fresh path"
        )
    os.makedirs(request_dir, exist_ok=True)

    forget_dir = os.path.join(request_dir, forget_subdir)
    retain_dir = os.path.join(request_dir, retain_subdir)

    # ---- forget shards: only the requested uids, in the given order ----
    forget_uids_resolved: List[int] = [int(u) for u in forget_uids]
    missing = [
        u for u in forget_uids_resolved if u not in forget_index.bytes_by_uid
    ]
    if missing:
        raise KeyError(
            f"{len(missing)} forget uids missing from the forget index "
            f"(first few: {missing[:5]}). Was the manifest aligned with "
            f"{forget_index.source_dir}?"
        )
    forget_shard_paths, n_forget_rows = _write_uid_bytes(
        out_dir=forget_dir,
        bytes_by_uid=forget_index.bytes_by_uid,
        uid_order=forget_uids_resolved,
        rows_per_shard=rows_per_shard,
    )

    # ---- retain shards: current retain pool minus this request's forget ----
    forget_uid_set: Set[int] = set(forget_uids_resolved)
    retain_uid_order = [
        u for u in retain_index.uid_order if u not in forget_uid_set
    ]
    n_retain_uids_filtered_out = sum(
        1 for u in forget_uid_set if u in retain_index.bytes_by_uid
    )
    retain_reused_source = (
        n_retain_uids_filtered_out == 0
        and len(retain_index.bytes_by_uid) == retain_index.initial_uid_count
    )
    if retain_reused_source:
        os.symlink(
            retain_index.source_dir, retain_dir, target_is_directory=True
        )
        retain_shard_paths = _list_shards(retain_dir)
        n_retain_rows = len(retain_uid_order)
        log.info(
            "[seq-split] reusing retain shards from %s (no copy).",
            retain_index.source_dir,
        )
    else:
        retain_shard_paths, n_retain_rows = _write_uid_bytes(
            out_dir=retain_dir,
            bytes_by_uid=retain_index.bytes_by_uid,
            uid_order=retain_uid_order,
            rows_per_shard=rows_per_shard,
        )

    info: Dict[str, object] = {
        "request_dir": os.path.abspath(request_dir),
        "forget_dir": os.path.abspath(forget_dir),
        "retain_dir": os.path.abspath(retain_dir),
        "forget_uids": forget_uids_resolved,
        "n_forget_rows_written": int(n_forget_rows),
        "n_retain_rows_written": int(n_retain_rows),
        "n_retain_uids_filtered_out": int(n_retain_uids_filtered_out),
        "retain_reused_source": bool(retain_reused_source),
        "rows_per_shard": int(rows_per_shard),
        "forget_shard_paths": [
            os.path.relpath(p, request_dir) for p in forget_shard_paths
        ],
        "retain_shard_paths": [
            os.path.relpath(p, request_dir) for p in retain_shard_paths
        ],
    }
    info_path = os.path.join(request_dir, "request_split.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    return info


# ---------------------------------------------------------------------------
# Request batching
# ---------------------------------------------------------------------------


def build_request_batches(
    forget_uid_order: Sequence[int],
    *,
    request_batch_size: int,
    max_requests: Optional[int] = None,
) -> List[List[int]]:
    """Chunk ``forget_uid_order`` into K = ceil(N/B) batches of size up to B.

    The last batch may be smaller. ``max_requests`` truncates the list when
    set (useful for smoke tests).
    """
    if request_batch_size <= 0:
        raise ValueError(
            f"request_batch_size must be > 0, got {request_batch_size}"
        )
    uids = [int(u) for u in forget_uid_order]
    batches: List[List[int]] = [
        uids[i : i + request_batch_size]
        for i in range(0, len(uids), request_batch_size)
    ]
    if max_requests is not None and max_requests >= 0:
        batches = batches[: int(max_requests)]
    return batches


def order_forget_uids(
    forget_index: UserIdIndex,
    *,
    request_user_order: str,
    request_seed: int,
    forget_manifest_path: Optional[str],
) -> List[int]:
    """Resolve the global forget-user ordering used to build request batches.

    Modes
    -----
    ``manifest`` (default)
        Use the order from ``forget_manifest['spam_user_ids']`` if it is
        readable; otherwise fall back to ``sorted``.
    ``sorted``
        Sort by user_id ascending.
    ``shuffled``
        Deterministic shuffle of the index order using ``request_seed``.
    """
    mode = (request_user_order or "manifest").lower()
    if mode == "manifest":
        if forget_manifest_path and os.path.isfile(forget_manifest_path):
            try:
                with open(forget_manifest_path, "r") as f:
                    manifest = json.load(f)
                manifest_uids = manifest.get("spam_user_ids") or []
                manifest_uids = [int(u) for u in manifest_uids]
                # Keep only uids that actually exist in the index, preserving order.
                index_uids = set(forget_index.bytes_by_uid)
                ordered = [u for u in manifest_uids if u in index_uids]
                # Append uids missing from the manifest, sorted, so no forget rows are dropped.
                trailing = sorted(index_uids - set(ordered))
                if trailing:
                    log.warning(
                        "[seq-split] %d forget uids missing from manifest; "
                        "appending in sorted order.",
                        len(trailing),
                    )
                return ordered + trailing
            except Exception as ex:  # pragma: no cover - defensive
                log.warning(
                    "[seq-split] manifest=%s unreadable (%s); "
                    "falling back to sorted order.",
                    forget_manifest_path,
                    ex,
                )
        return sorted(forget_index.bytes_by_uid)
    if mode == "sorted":
        return sorted(forget_index.bytes_by_uid)
    if mode == "shuffled":
        import random

        rng = random.Random(int(request_seed))
        out = list(forget_index.uid_order)
        rng.shuffle(out)
        return out
    raise ValueError(
        f"unknown request_user_order={request_user_order!r}; "
        f"expected one of: manifest, sorted, shuffled"
    )
