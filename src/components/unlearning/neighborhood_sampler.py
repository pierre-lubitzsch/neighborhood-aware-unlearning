"""TIGER-specific "neighborhood-aware" retain sampler.

ERASE's ``GIF/CEU`` use graph k-hops to find retain interactions adjacent to the
forget set. TIGER has no user-item graph, so neighbors are approximated by
semantic-ID proximity in the codebook (``merged_predictions_tensor.pt``).

Neighborhood mode (``neighborhood_aware=True``):

1. Sort all item ids by full semantic id (lexicographic ascending).
2. For each forget item, collect all catalog items sharing the current
   SID prefix (``sid_prefix_length``), ordered by ascending SID distance.
   Forget and spam items are excluded as repair targets.
3. Pool every retain row whose ``sequence_data`` mentions any bucket item
   anywhere in the sequence. Rows that contain any forget-item id are skipped.
4. Sample rows uniformly at random (seeded) from the pool until the row
   budget is met or the pool is drained, then repeat with a shorter prefix
   (``k-1``, ``k-2``, ..., ``1``, where ``k = num_hierarchies``), skipping
   neighbors already used.
5. Optionally mix with uniform retain rows via
   ``neighborhood_aware_sample_rate`` in ``[0, 1]`` (``1`` = neighborhood only,
   ``0`` = uniform only, ``0.5`` = half/half of the row budget).
6. Shuffle the selected rows with a fixed seed, then cap at
   ``retain_sample_size * |D_f|`` (alias: ``retain_samples_used_for_update``).

When ``neighborhood_aware=False`` the script uniformly samples retain rows at
the same budget (ERASE-baseline fallback), still excluding rows that mention
forget items when forget shards are available.

Output: a sibling directory of ``training_retain`` whose schema is identical
to the source so the existing Hydra retain dataloader works unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
import torch

from src.data.unlearning.deletion_spec import (
    resolve_forbidden_retain_items,
    resolve_neighborhood_centers,
)

tf.config.set_visible_devices([], "GPU")


SEQUENCE_FIELD = "sequence_data"
USER_ID_FIELD = "user_id"


# ---------------------------------------------------------------------------
# IO helpers (mirror split_forget_retain.py)
# ---------------------------------------------------------------------------


def _list_shards(directory: str) -> List[str]:
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".tfrecord.gz")
    ]


def _infer_feature_description(sample_record: tf.Tensor) -> Dict[str, tf.io.VarLenFeature]:
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
# Forget / retain row scanning
# ---------------------------------------------------------------------------


def collect_items_in_shards(shard_paths: List[str]) -> Set[int]:
    """Return distinct item ids appearing in ``sequence_data`` fields."""
    if not shard_paths:
        return set()
    raw = tf.data.TFRecordDataset(shard_paths, compression_type="GZIP")
    sample = next(iter(raw))
    feat_desc = _infer_feature_description(sample)
    if SEQUENCE_FIELD not in feat_desc:
        raise ValueError(
            f"shards {shard_paths[0]!r}... lack {SEQUENCE_FIELD!r} feature"
        )
    parsed = raw.map(lambda x: tf.io.parse_single_example(x, feat_desc))
    items: Set[int] = set()
    for example in parsed:
        seq = tf.sparse.to_dense(example[SEQUENCE_FIELD]).numpy()
        if seq.size == 0:
            continue
        items.update(int(x) for x in seq.tolist())
    return items


def _scan_retain_rows_with_item_index(
    retain_dir: str,
    *,
    forbidden_items: Optional[Set[int]] = None,
) -> Tuple[List[bytes], Dict[int, List[int]], Set[int]]:
    """Scan retain shards once; return rows, item index, and forbidden row ids.

    Rows whose ``sequence_data`` mentions any id in ``forbidden_items`` are
    listed in the third return value and should not be used for retain repair.
    """
    shards = _list_shards(retain_dir)
    if not shards:
        raise FileNotFoundError(f"No shards under {retain_dir}")

    raw = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    sample = next(iter(raw))
    feat_desc = _infer_feature_description(sample)
    if SEQUENCE_FIELD not in feat_desc:
        raise ValueError(f"{retain_dir}: missing {SEQUENCE_FIELD!r} feature")

    all_rows: List[bytes] = []
    item_to_row_indices: Dict[int, List[int]] = defaultdict(list)
    forbidden_row_indices: Set[int] = set()
    parsed = raw.map(lambda x: (x, tf.io.parse_single_example(x, feat_desc)))
    for raw_t, ex in parsed:
        seq = tf.sparse.to_dense(ex[SEQUENCE_FIELD]).numpy()
        if seq.size == 0:
            continue
        row_idx = len(all_rows)
        all_rows.append(raw_t.numpy())
        row_items = {int(x) for x in seq.tolist()}
        if forbidden_items and row_items & forbidden_items:
            forbidden_row_indices.add(row_idx)
        for item in row_items:
            item_to_row_indices[item].append(row_idx)
    return all_rows, item_to_row_indices, forbidden_row_indices


def _collect_spam_repair_rows(
    forget_dir: str,
    *,
    target_items: Set[int],
) -> List[bytes]:
    """Return forget-shard rows whose sequences mention only non-target items."""
    shards = _list_shards(forget_dir)
    if not shards:
        return []
    raw = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    sample = next(iter(raw))
    feat_desc = _infer_feature_description(sample)
    parsed = raw.map(lambda x: (x, tf.io.parse_single_example(x, feat_desc)))
    repair_rows: List[bytes] = []
    for raw_t, ex in parsed:
        seq = tf.sparse.to_dense(ex[SEQUENCE_FIELD]).numpy()
        if seq.size == 0:
            continue
        row_items = {int(x) for x in seq.tolist()}
        if row_items & target_items:
            continue
        repair_rows.append(raw_t.numpy())
    return repair_rows


def _resolve_repair_sample_bound(
    retain_samples_used_for_update: int,
    retain_sample_size: Optional[int],
    repair_sample_bound: Optional[float],
) -> int:
    if repair_sample_bound is not None:
        return max(1, int(math.ceil(float(repair_sample_bound))))
    return _resolve_retain_sample_size(retain_samples_used_for_update, retain_sample_size)


# ---------------------------------------------------------------------------
# Dense embedding neighborhood (pre-quantization item embeddings)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DenseEmbeddings:
    """Pre-quantization item embeddings with raw-item-ID lookup.

    Supports datasets whose item IDs are not sequential from 0 (e.g. rsc15).
    For the plain-tensor format, item_ids are ``torch.arange(n)``.
    """

    tensor: torch.Tensor          # (N, emb_dim) float32, row i = item item_ids[i]
    item_ids: torch.Tensor        # (N,) int64, item_ids[i] = raw item ID for row i
    item_id_to_idx: Dict[int, int]  # raw_item_id -> row index

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __getitem__(self, item_id: int) -> torch.Tensor:
        return self.tensor[self.item_id_to_idx[item_id]]

    def __contains__(self, item_id: int) -> bool:
        return int(item_id) in self.item_id_to_idx


def load_dense_embeddings(embedding_path: str) -> DenseEmbeddings:
    """Load pre-quantization item embeddings.

    Accepts two on-disk formats:
    * Plain ``torch.Tensor`` of shape ``(N, emb_dim)`` where row ``i``
      corresponds to item ID ``i``.
    * Dict with keys ``"embeddings"`` (tensor) and ``"item_ids"`` (1-D int64
      tensor mapping row to raw item ID), used for non-sequential item IDs.
    """
    obj = torch.load(embedding_path, map_location="cpu", weights_only=False)

    item_ids_tensor: Optional[torch.Tensor] = None
    emb_tensor: Optional[torch.Tensor] = None

    if isinstance(obj, dict):
        item_ids_tensor = obj.get("item_ids")
        for key in ("embeddings", "embedding", "tensor"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                emb_tensor = obj[key]
                break
    elif isinstance(obj, torch.Tensor):
        emb_tensor = obj

    if emb_tensor is None:
        raise TypeError(
            f"Loaded object from {embedding_path!r} is not a recognized "
            f"embedding format (got {type(obj)})"
        )

    emb_tensor = emb_tensor.float()
    n = int(emb_tensor.shape[0])

    if item_ids_tensor is None:
        # Plain tensor: sequential IDs 0 .. N-1.
        item_ids_tensor = torch.arange(n, dtype=torch.int64)
    else:
        item_ids_tensor = item_ids_tensor.to(torch.int64)

    id_to_idx: Dict[int, int] = {
        int(iid): idx for idx, iid in enumerate(item_ids_tensor.tolist())
    }
    return DenseEmbeddings(tensor=emb_tensor, item_ids=item_ids_tensor, item_id_to_idx=id_to_idx)


def embedding_neighbors(
    item_id: int,
    embeddings: DenseEmbeddings,
    epsilon: float,
    *,
    max_neighbors: int = 100,
    exclude_ids: Optional[Set[int]] = None,
) -> List[int]:
    """Return catalog item IDs within L2 distance ``epsilon`` of ``item_id``.

    Accepts a :class:`DenseEmbeddings` instance so that datasets with
    non-sequential raw item IDs (e.g. rsc15) are handled correctly.
    Returned IDs are raw item IDs, not tensor row indices.
    """
    exclude_ids = exclude_ids or set()
    if item_id not in embeddings:
        return []
    center = embeddings[item_id]
    dists = torch.norm(embeddings.tensor - center.unsqueeze(0), dim=1)
    mask = (dists <= float(epsilon)) & (dists > 0)
    for ex in exclude_ids:
        ex_idx = embeddings.item_id_to_idx.get(int(ex))
        if ex_idx is not None:
            mask[ex_idx] = False
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return []
    if idx.numel() > max_neighbors:
        sub_dists = dists[idx]
        order = torch.argsort(sub_dists)[:max_neighbors]
        idx = idx[order]
    return [int(embeddings.item_ids[i]) for i in idx.tolist()]


def topk_embedding_neighbors(
    item_id: int,
    embeddings: DenseEmbeddings,
    count: int,
    *,
    metric: str = "cosine",
    exclude_ids: Optional[Set[int]] = None,
    by_row: bool = False,
) -> List[int]:
    """Return the ``count`` nearest catalog items to ``item_id`` by embedding.

    Unlike :func:`embedding_neighbors` (an ``epsilon``-ball whose size varies
    by dataset), this returns a fixed-size top-k, so every target gets exactly
    ``count`` neighbors when used as the ``P(i_T)`` set of the loss ``L_n``.

    ``metric="cosine"`` (default) ranks by cosine similarity on L2-normalized
    vectors; ``metric="l2"`` ranks by Euclidean distance.

    ``by_row=False`` (default) treats ``item_id`` and ``exclude_ids`` as raw item
    IDs and returns raw item IDs. ``by_row=True`` treats them as row indices into
    ``embeddings.tensor`` and returns row indices, which is required in codebook
    space when raw item IDs are not sequential. This assumes codebook row i and
    embedding row i describe the same item.
    """
    metric = str(metric).lower()
    if metric not in ("cosine", "l2"):
        raise ValueError(f"metric must be 'cosine'|'l2', got {metric!r}")
    count = int(count)
    if count <= 0:
        return []
    n_rows = int(embeddings.tensor.shape[0])
    if by_row:
        if not (0 <= int(item_id) < n_rows):
            return []
    elif item_id not in embeddings:
        return []

    excluded = {int(x) for x in (exclude_ids or set())}
    excluded.add(int(item_id))

    center = (embeddings.tensor[int(item_id)] if by_row
              else embeddings[item_id])
    if metric == "cosine":
        mat = torch.nn.functional.normalize(embeddings.tensor, dim=1)
        vec = torch.nn.functional.normalize(center.unsqueeze(0), dim=1)
        # Negate so that "smaller is closer" holds for both metrics.
        scores = -(mat @ vec.squeeze(0))
    else:
        scores = torch.norm(embeddings.tensor - center.unsqueeze(0), dim=1)

    for ex in excluded:
        ex_idx = (int(ex) if by_row and 0 <= int(ex) < n_rows
                  else embeddings.item_id_to_idx.get(int(ex)))
        if ex_idx is not None:
            scores[ex_idx] = float("inf")

    # Request extra rows so excluded entries can be dropped.
    k = min(count + len(excluded), int(scores.numel()))
    order = torch.topk(scores, k, largest=False).indices
    out: List[int] = []
    for i in order.tolist():
        if math.isinf(float(scores[i])):
            continue
        out.append(int(i) if by_row else int(embeddings.item_ids[i]))
        if len(out) >= count:
            break
    return out


def select_retain_rows_embedding(
    all_rows: List[bytes],
    item_to_row_indices: Dict[int, List[int]],
    center_items: Iterable[int],
    embeddings: DenseEmbeddings,
    *,
    budget: int,
    seed: int,
    epsilon: float,
    max_neighbors: int,
    forbidden_row_indices: Optional[Set[int]] = None,
    exclude_target_items: Optional[Set[int]] = None,
) -> Tuple[List[bytes], Dict[str, object]]:
    """Select retain rows via embedding-distance neighbors of center items."""
    forbidden_row_indices = forbidden_row_indices or set()
    exclude_target_items = set(exclude_target_items or set())
    center_list = sorted(int(i) for i in center_items if int(i) in embeddings)
    exclude_target_items.update(center_list)

    selected_indices: Set[int] = set()
    per_center: Dict[int, List[int]] = {}
    for cid in center_list:
        neighbors = embedding_neighbors(
            cid,
            embeddings,
            epsilon,
            max_neighbors=max_neighbors,
            exclude_ids=exclude_target_items,
        )
        per_center[cid] = neighbors
        for neighbor in neighbors:
            for row_idx in item_to_row_indices.get(neighbor, []):
                if row_idx in forbidden_row_indices:
                    continue
                selected_indices.add(row_idx)

    rng = random.Random(seed)
    indices = list(selected_indices)
    rng.shuffle(indices)
    qualifying = [all_rows[i] for i in indices][: max(0, int(budget))]
    meta = {
        "embedding": True,
        "epsilon": float(epsilon),
        "max_neighbors": int(max_neighbors),
        "n_center_items": len(center_list),
        "n_neighbor_items": sum(len(v) for v in per_center.values()),
        "n_rows_selected": len(qualifying),
        "budget": int(budget),
    }
    return qualifying, meta


# ---------------------------------------------------------------------------
# SID codebook + sorted-index neighborhood search
# ---------------------------------------------------------------------------


def load_codebook(
    semantic_id_path: str,
    num_hierarchies: Optional[int] = None,
) -> torch.Tensor:
    """Load ``merged_predictions_tensor.pt`` as ``(num_items, num_hierarchies)``.

    RKMeans / training artifacts store the map as ``(D, N)`` (hierarchies x
    items); see ``map_sparse_id_to_semantic_id`` which indexes via
    ``id_map[:num_hierarchies].t()[sparse_id]``. When ``num_hierarchies`` is
    given and the first dimension matches it, the tensor is transposed.
    """
    obj = torch.load(semantic_id_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("semantic_ids", "sequence_data", "codebook", "tensor"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                obj = obj[key]
                break
    if not isinstance(obj, torch.Tensor):
        raise TypeError(
            f"Loaded object from {semantic_id_path!r} is not a torch.Tensor "
            f"(got {type(obj)})"
        )
    if obj.dim() != 2:
        raise ValueError(
            f"Expected 2-D codebook tensor, got shape {tuple(obj.shape)}"
        )
    codebook = obj.long()
    raw_shape = tuple(codebook.shape)

    if num_hierarchies is not None:
        nh = int(num_hierarchies)
        if codebook.shape[0] == nh and codebook.shape[1] != nh:
            codebook = codebook.t().contiguous()
        elif codebook.shape[1] == nh:
            pass
        else:
            raise ValueError(
                f"Codebook shape {raw_shape} is incompatible with "
                f"num_hierarchies={nh}; expected (N, {nh}) or ({nh}, N)."
            )
    elif codebook.shape[0] <= 32 and codebook.shape[1] > codebook.shape[0]:
        codebook = codebook.t().contiguous()

    if codebook.shape[0] < codebook.shape[1]:
        return codebook
    if num_hierarchies is None and codebook.shape[0] > codebook.shape[1]:
        codebook = codebook.t().contiguous()
    return codebook


def build_sorted_sid_index(codebook: torch.Tensor) -> np.ndarray:
    """Return item ids sorted by full semantic id (lexicographic ascending)."""
    rows = codebook.numpy()
    if rows.ndim != 2:
        raise ValueError(f"Expected 2-D codebook, got shape {rows.shape}")
    keys = [rows[:, h] for h in range(rows.shape[1] - 1, -1, -1)]
    order = np.lexsort(keys)
    return order.astype(np.int64)


def _sid_tuple(row: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(x) for x in row.tolist())


def _compare_sid_prefix(
    a: Tuple[int, ...], b: Tuple[int, ...], prefix_len: int
) -> int:
    """Lexicographic compare of the first ``prefix_len`` hierarchies."""
    for i in range(prefix_len):
        if a[i] < b[i]:
            return -1
        if a[i] > b[i]:
            return 1
    return 0


def _bisect_sid(sorted_ids: np.ndarray, sorted_sids: np.ndarray, sid: Tuple[int, ...]) -> int:
    """Return insertion index of ``sid`` in ``sorted_sids`` (full-length key)."""
    lo, hi = 0, int(sorted_ids.shape[0])
    k = len(sid)
    while lo < hi:
        mid = (lo + hi) // 2
        mid_sid = _sid_tuple(sorted_sids[mid, :k])
        if mid_sid < sid:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _prefix_range(
    sorted_ids: np.ndarray,
    sorted_sids: np.ndarray,
    prefix: Tuple[int, ...],
) -> Tuple[int, int]:
    """Return ``[lo, hi)`` indices whose SID starts with ``prefix``."""
    prefix_len = len(prefix)
    lo = 0
    hi = int(sorted_ids.shape[0])
    while lo < hi:
        mid = (lo + hi) // 2
        if _compare_sid_prefix(_sid_tuple(sorted_sids[mid]), prefix, prefix_len) < 0:
            lo = mid + 1
        else:
            hi = mid
    start = lo
    hi = int(sorted_ids.shape[0])
    while lo < hi:
        mid = (lo + hi) // 2
        cmp = _compare_sid_prefix(_sid_tuple(sorted_sids[mid]), prefix, prefix_len)
        if cmp <= 0:
            lo = mid + 1
        else:
            hi = mid
    return start, lo


def _sid_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Sum of absolute hierarchy deltas (tie-break for closest item)."""
    return int(np.abs(a.astype(np.int64) - b.astype(np.int64)).sum())


def _common_prefix_length(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _closest_by_sorted_distance(
    sorted_ids: np.ndarray,
    sorted_sids: np.ndarray,
    query: np.ndarray,
    query_sid: Tuple[int, ...],
    *,
    exclude_id: Optional[int] = None,
    exclude_ids: Optional[Set[int]] = None,
) -> Optional[int]:
    """Return the catalog item with smallest SID distance in sorted order.

    Walks outward from the binary-search insertion point (two-pointer style).
    Skips ids in ``exclude_ids`` and ``exclude_id``. Falls back to an excluded
    id only when no other catalog entry exists.
    """
    excluded = set(exclude_ids or set())
    if exclude_id is not None:
        excluded.add(int(exclude_id))
    n = int(sorted_ids.shape[0])
    if n == 0:
        return None
    pos = _bisect_sid(sorted_ids, sorted_sids, query_sid)

    best_other: Optional[int] = None
    best_other_dist: Optional[int] = None
    best_excluded: Optional[int] = None
    best_excluded_dist: Optional[int] = None

    for offset in range(n):
        for idx in (pos - offset - 1, pos + offset):
            if idx < 0 or idx >= n:
                continue
            cand = int(sorted_ids[idx])
            dist = _sid_distance(query, sorted_sids[idx])
            if cand in excluded:
                if best_excluded_dist is None or dist < best_excluded_dist:
                    best_excluded_dist = dist
                    best_excluded = cand
                continue
            if best_other_dist is None or dist < best_other_dist:
                best_other_dist = dist
                best_other = cand
        if best_other is not None:
            return best_other

    return best_excluded


def closest_item_at_prefix(
    sorted_ids: np.ndarray,
    sorted_sids: np.ndarray,
    codebook: torch.Tensor,
    item_id: int,
    prefix_len: int,
    *,
    exclude_ids: Optional[Set[int]] = None,
) -> Optional[int]:
    """Pick the single catalog item closest to ``item_id`` at ``prefix_len``.

    Prefer items whose SID shares the first ``prefix_len`` hierarchies with
    ``item_id``. All ids in ``exclude_ids`` (typically the full forget set) are
    skipped. If no other item shares that prefix, fall back to the globally
    closest non-excluded item in the SID-sorted catalog list.
    """
    excluded = set(exclude_ids or set())
    num_items, num_hierarchies = codebook.shape
    if not (0 <= item_id < num_items):
        return None
    if prefix_len <= 0 or prefix_len > num_hierarchies:
        raise ValueError(
            f"prefix_len must be in [1, {num_hierarchies}], got {prefix_len}"
        )

    query = codebook[item_id].numpy()
    query_sid = _sid_tuple(query)
    prefix = query_sid[:prefix_len]

    lo, hi = _prefix_range(sorted_ids, sorted_sids, prefix)
    if lo < hi:
        best_other: Optional[int] = None
        best_other_dist: Optional[int] = None
        best_excluded: Optional[int] = None
        best_excluded_dist: Optional[int] = None
        for idx in range(lo, hi):
            cand = int(sorted_ids[idx])
            dist = _sid_distance(query, sorted_sids[idx])
            if cand in excluded:
                if best_excluded_dist is None or dist < best_excluded_dist:
                    best_excluded_dist = dist
                    best_excluded = cand
                continue
            if best_other_dist is None or dist < best_other_dist:
                best_other_dist = dist
                best_other = cand
        if best_other is not None:
            return best_other
        if best_excluded is not None:
            return None

    return _closest_by_sorted_distance(
        sorted_ids,
        sorted_sids,
        query,
        query_sid,
        exclude_id=item_id,
        exclude_ids=excluded,
    )


def bucket_items_by_distance(
    sorted_ids: np.ndarray,
    sorted_sids: np.ndarray,
    codebook: torch.Tensor,
    item_id: int,
    prefix_len: int,
    *,
    exclude_ids: Optional[Set[int]] = None,
) -> List[int]:
    """Return all catalog items sharing the first ``prefix_len`` hierarchies
    with ``item_id``, ordered by ascending SID distance.

    Ids in ``exclude_ids`` (forget/target items and already-used neighbors)
    are skipped. Unlike :func:`closest_item_at_prefix` this returns the whole
    prefix bucket so callers can exhaust it before relaxing the prefix.
    """
    excluded = set(exclude_ids or set())
    num_items, num_hierarchies = codebook.shape
    if not (0 <= item_id < num_items):
        return []
    if prefix_len <= 0 or prefix_len > num_hierarchies:
        raise ValueError(
            f"prefix_len must be in [1, {num_hierarchies}], got {prefix_len}"
        )

    query = codebook[item_id].numpy()
    prefix = _sid_tuple(query)[:prefix_len]
    lo, hi = _prefix_range(sorted_ids, sorted_sids, prefix)
    if lo >= hi:
        return []
    dists = (
        np.abs(sorted_sids[lo:hi].astype(np.int64) - query.astype(np.int64))
        .sum(axis=1)
    )
    order = np.argsort(dists, kind="stable")
    return [
        cand
        for off in order.tolist()
        if (cand := int(sorted_ids[lo + off])) not in excluded
    ]


def closest_prefix_neighbors(
    codebook: torch.Tensor,
    item_id: int,
    count: int,
    min_prefix_length: int,
    *,
    sorted_ids: Optional[np.ndarray] = None,
    sorted_sids: Optional[np.ndarray] = None,
    exclude_ids: Optional[Set[int]] = None,
) -> List[int]:
    """Return up to ``count`` catalog items closest to ``item_id`` by SID prefix.

    "Closeness" is the length of the shared semantic-ID prefix: an item sharing
    a longer prefix with ``item_id`` is closer. Only items whose shared prefix
    with ``item_id`` has length ``>= min_prefix_length`` are eligible. Ties
    within a shared-prefix level are broken by ascending SID (L1) distance.

    The forget set (and ``item_id`` itself) should be passed via ``exclude_ids``
    so returned neighbors are never forgotten items.

    This is the ``P(i_T)`` neighborhood set of the TRACER coherence loss, with
    neighbors defined by RQ-code prefix proximity instead of embedding
    similarity.

    ``sorted_ids`` / ``sorted_sids`` are the shared SID-sorted index and may be
    precomputed once (via :func:`build_sorted_sid_index`) and reused across many
    calls; they are derived from ``codebook`` when omitted.
    """
    num_items, num_hierarchies = codebook.shape
    if not (0 <= int(item_id) < num_items):
        return []
    count = int(count)
    if count <= 0:
        return []
    min_prefix_length = int(min_prefix_length)
    if min_prefix_length <= 0 or min_prefix_length > num_hierarchies:
        raise ValueError(
            f"min_prefix_length must be in [1, {num_hierarchies}], "
            f"got {min_prefix_length}"
        )

    if sorted_ids is None or sorted_sids is None:
        sorted_ids = build_sorted_sid_index(codebook)
        sorted_sids = codebook.numpy()[sorted_ids]

    excluded = set(exclude_ids or set())
    excluded.add(int(item_id))

    neighbors: List[int] = []
    seen: Set[int] = set()
    # Longest shared prefix first (closest), relaxing down to min_prefix_length.
    # Because a longer-prefix bucket is a subset of a shorter-prefix one,
    # skipping already-taken ids keeps each item at its longest shared prefix.
    for prefix_len in range(num_hierarchies, min_prefix_length - 1, -1):
        bucket = bucket_items_by_distance(
            sorted_ids,
            sorted_sids,
            codebook,
            int(item_id),
            prefix_len,
            exclude_ids=excluded | seen,
        )
        for cand in bucket:
            if cand in seen:
                continue
            seen.add(cand)
            neighbors.append(cand)
            if len(neighbors) >= count:
                return neighbors
    return neighbors


def select_retain_rows_progressive(
    all_rows: List[bytes],
    item_to_row_indices: Dict[int, List[int]],
    forget_items: Iterable[int],
    codebook: torch.Tensor,
    *,
    budget: int,
    seed: int,
    start_prefix_length: Optional[int] = None,
    min_prefix_length: int = 1,
    retain_max_rows: Optional[int] = None,
    forbidden_row_indices: Optional[Set[int]] = None,
    exclude_target_items: Optional[Set[int]] = None,
) -> Tuple[List[bytes], Dict[str, object]]:
    """Select retain rows via progressive prefix-bucket exhaustion.

    At each prefix level, all retain rows mentioning any same-prefix item form
    a pool; rows are drawn uniformly at random (seeded) from it until the row
    budget is met or the pool is drained. The prefix is only shortened once
    the pool is exhausted and the budget is still unmet.
    """
    forbidden_row_indices = forbidden_row_indices or set()
    exclude_target_items = set(exclude_target_items or set())
    num_hierarchies = int(codebook.shape[1])
    if start_prefix_length is None:
        start_prefix_length = num_hierarchies - 1
    start_prefix_length = int(start_prefix_length)
    min_prefix_length = max(1, int(min_prefix_length))

    sorted_ids = build_sorted_sid_index(codebook)
    sorted_sids = codebook.numpy()[sorted_ids]
    forget_list = sorted(
        int(i) for i in forget_items if 0 <= int(i) < codebook.shape[0]
    )
    exclude_target_items.update(forget_list)

    rng = random.Random(seed)
    selected_indices: Set[int] = set()
    used_neighbors: Set[int] = set()
    level_log: List[Dict[str, object]] = []

    for prefix_len in range(start_prefix_length, min_prefix_length - 1, -1):
        # Full prefix bucket per forget item. Neighbors already consumed at a
        # longer prefix are skipped so each level adds fresh rows.
        buckets: Dict[int, List[int]] = {
            fid: bucket_items_by_distance(
                sorted_ids,
                sorted_sids,
                codebook,
                fid,
                prefix_len,
                exclude_ids=exclude_target_items | used_neighbors,
            )
            for fid in forget_list
        }
        level_neighbors: Set[int] = set()
        # Pool every retain row mentioning any bucket item, then sample rows
        # uniformly at random until the budget is met or the pool is drained.
        level_pool: List[int] = []
        pooled: Set[int] = set()
        for fid in forget_list:
            for cand in buckets[fid]:
                if cand in used_neighbors:
                    continue
                used_neighbors.add(cand)
                level_neighbors.add(cand)
                for row_idx in item_to_row_indices.get(cand, []):
                    if row_idx in forbidden_row_indices:
                        continue
                    if row_idx in selected_indices or row_idx in pooled:
                        continue
                    pooled.add(row_idx)
                    level_pool.append(row_idx)
        rng.shuffle(level_pool)
        taken = level_pool[: max(0, budget - len(selected_indices))]
        selected_indices.update(taken)
        n_added = len(taken)
        level_log.append(
            {
                "sid_prefix_length": prefix_len,
                "n_rows_in_bucket_pool": len(level_pool),
                "n_rows_added": n_added,
                "n_rows_cumulative": len(selected_indices),
                "n_neighbor_items_used": len(level_neighbors),
            }
        )
        if len(selected_indices) >= budget:
            break

    # At k=1 (or after all prefix levels): if still empty, use sorted-list
    # closest neighbors even when no hierarchy is shared with another item.
    if not selected_indices and forget_list:
        n_added = 0
        for fid in forget_list:
            query = codebook[fid].numpy()
            query_sid = _sid_tuple(query)
            closest = _closest_by_sorted_distance(
                sorted_ids,
                sorted_sids,
                query,
                query_sid,
                exclude_id=fid,
                exclude_ids=exclude_target_items,
            )
            if closest is None:
                continue
            for row_idx in item_to_row_indices.get(closest, []):
                if row_idx in forbidden_row_indices:
                    continue
                if row_idx not in selected_indices:
                    selected_indices.add(row_idx)
                    n_added += 1
        level_log.append(
            {
                "sid_prefix_length": 0,
                "n_rows_added": n_added,
                "n_rows_cumulative": len(selected_indices),
                "n_neighbor_items_used": len(selected_indices),
                "fallback": "sorted_distance_no_shared_prefix",
            }
        )

    indices = list(selected_indices)
    rng.shuffle(indices)
    qualifying = [all_rows[i] for i in indices]

    cap = budget
    if retain_max_rows is not None:
        cap = min(cap, int(retain_max_rows))
    if len(qualifying) > cap:
        qualifying = qualifying[:cap]

    meta = {
        "progressive_prefix": True,
        "start_prefix_length": start_prefix_length,
        "min_prefix_length": min_prefix_length,
        "budget": int(budget),
        "retain_max_rows": retain_max_rows,
        "n_excluded_forget_target_items": len(exclude_target_items),
        "n_forbidden_rows_with_forget_items": len(forbidden_row_indices),
        "n_rows_before_cap": len(indices),
        "n_rows_after_cap": len(qualifying),
        "prefix_levels": level_log,
    }
    return qualifying, meta


def select_retain_rows_uniform(
    all_rows: List[bytes],
    *,
    budget: int,
    seed: int,
    forbidden_row_indices: Optional[Set[int]] = None,
    exclude_row_indices: Optional[Set[int]] = None,
) -> Tuple[List[bytes], Dict[str, object]]:
    """Uniformly sample retain rows, excluding forbidden / already-selected rows."""
    forbidden_row_indices = forbidden_row_indices or set()
    exclude_row_indices = exclude_row_indices or set()
    eligible = [
        idx
        for idx in range(len(all_rows))
        if idx not in forbidden_row_indices and idx not in exclude_row_indices
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = eligible[: max(0, int(budget))]
    rows = [all_rows[i] for i in chosen]
    meta = {
        "uniform": True,
        "budget": int(budget),
        "n_eligible_rows": len(eligible),
        "n_rows_selected": len(rows),
    }
    return rows, meta


def _merge_retain_row_lists(
    parts: List[List[bytes]],
    *,
    max_rows: int,
    seed: int,
    retain_max_rows: Optional[int] = None,
) -> Tuple[List[bytes], Dict[str, object]]:
    """Dedupe, shuffle, and cap merged retain row lists."""
    seen: Set[bytes] = set()
    merged: List[bytes] = []
    for part in parts:
        for row in part:
            if row in seen:
                continue
            seen.add(row)
            merged.append(row)
    rng = random.Random(seed)
    rng.shuffle(merged)
    cap = int(max_rows)
    if retain_max_rows is not None:
        cap = min(cap, int(retain_max_rows))
    capped = merged[:cap]
    return capped, {
        "n_rows_before_dedupe": sum(len(p) for p in parts),
        "n_rows_after_dedupe": len(merged),
        "n_rows_after_cap": len(capped),
    }


# ---------------------------------------------------------------------------
# Fixed-prefix neighbor lookup (non-progressive mode)
# ---------------------------------------------------------------------------


def prefix_neighbors(
    codebook: torch.Tensor,
    forget_items: Iterable[int],
    sid_prefix_length: int,
) -> Set[int]:
    """Return all sparse item ids sharing any forget SID prefix."""
    num_items, num_hierarchies = codebook.shape
    if sid_prefix_length <= 0 or sid_prefix_length > num_hierarchies:
        raise ValueError(
            f"sid_prefix_length must be in [1, {num_hierarchies}], "
            f"got {sid_prefix_length}"
        )
    forget_idx = sorted(int(i) for i in forget_items if 0 <= int(i) < num_items)
    forget_set = set(forget_idx)
    if not forget_idx:
        return set()
    forget_prefix_rows = codebook[forget_idx, :sid_prefix_length]
    forget_prefix_set = {tuple(row.tolist()) for row in forget_prefix_rows}

    all_prefixes = codebook[:, :sid_prefix_length]
    keep: Set[int] = set()
    for prefix in forget_prefix_set:
        prefix_t = torch.tensor(prefix, dtype=codebook.dtype)
        mask = (all_prefixes == prefix_t).all(dim=1)
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        keep.update(int(i) for i in idx.tolist())
    keep.difference_update(forget_set)
    return keep


# ---------------------------------------------------------------------------
# Filtering / capping retain shards
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


def _row_mentions_items(seq: np.ndarray, allowed_items: Optional[Set[int]]) -> bool:
    if allowed_items is None:
        return True
    if seq.size == 0:
        return False
    for x in seq.tolist():
        if int(x) in allowed_items:
            return True
    return False


def _row_mentions_items_from_index(
    row_idx: int,
    item_to_row_indices: Dict[int, List[int]],
    allowed_items: Set[int],
) -> bool:
    for item in allowed_items:
        if row_idx in item_to_row_indices.get(item, []):
            return True
    return False


def filter_retain_shards(
    retain_dir: str,
    out_dir: str,
    allowed_items: Optional[Set[int]],
    max_rows: Optional[int],
    rows_per_shard: int = 4096,
    seed: int = 2,
    retain_max_rows: Optional[int] = None,
    preselected_rows: Optional[List[bytes]] = None,
    n_seen_hint: Optional[int] = None,
) -> Tuple[List[str], int, int]:
    """Stream ``retain_dir`` through an optional row filter and cap.

    If ``preselected_rows`` is provided, skip scanning and write those rows
    directly (already shuffled / capped by the caller).

    Returns ``(shard_paths, n_kept, n_seen)``.
    """
    rng = random.Random(seed)
    cap = max_rows
    if retain_max_rows is not None:
        cap = min(cap, int(retain_max_rows)) if cap is not None else int(retain_max_rows)

    if preselected_rows is not None:
        qualifying = list(preselected_rows)
        n_seen = int(n_seen_hint) if n_seen_hint is not None else len(qualifying)
        if cap is not None and len(qualifying) > cap:
            rng.shuffle(qualifying)
            qualifying = qualifying[:cap]
    else:
        shards = _list_shards(retain_dir)
        if not shards:
            raise FileNotFoundError(f"No shards under {retain_dir}")

        raw = tf.data.TFRecordDataset(shards, compression_type="GZIP")
        sample = next(iter(raw))
        feat_desc = _infer_feature_description(sample)
        if SEQUENCE_FIELD not in feat_desc:
            raise ValueError(f"{retain_dir}: missing {SEQUENCE_FIELD!r} feature")

        qualifying: List[bytes] = []
        n_seen = 0
        parsed = raw.map(lambda x: (x, tf.io.parse_single_example(x, feat_desc)))
        for raw_t, ex in parsed:
            n_seen += 1
            seq = tf.sparse.to_dense(ex[SEQUENCE_FIELD]).numpy()
            if _row_mentions_items(seq, allowed_items):
                qualifying.append(raw_t.numpy())

        if cap is not None and len(qualifying) > cap:
            rng.shuffle(qualifying)
            qualifying = qualifying[:cap]

    writer = _ShardWriter(out_dir, "data", rows_per_shard)
    for raw_bytes in qualifying:
        writer.write(raw_bytes)
    writer.close()

    return writer.shard_paths, len(qualifying), n_seen


def _resolve_retain_sample_size(
    retain_samples_used_for_update: int,
    retain_sample_size: Optional[int],
) -> int:
    if retain_sample_size is not None:
        return int(retain_sample_size)
    return int(retain_samples_used_for_update)


# ---------------------------------------------------------------------------
# High-level entry point used by TigerUnlearningModule
# ---------------------------------------------------------------------------


def build_retain_subset(
    *,
    forget_dir: str,
    retain_dir: str,
    out_dir: str,
    neighborhood_aware: bool,
    semantic_id_path: Optional[str] = None,
    sid_prefix_length: int = 2,
    forget_size: Optional[int] = None,
    neighbor_aware_factor: float = 8.0,
    retain_samples_used_for_update: int = 16,
    retain_sample_size: Optional[int] = None,
    repair_sample_bound: Optional[float] = None,
    retain_max_rows: Optional[int] = None,
    progressive_sid_prefix: bool = True,
    neighborhood_aware_sample_rate: float = 1.0,
    neighborhood_method: str = "prefix",
    embedding_path: Optional[str] = None,
    embedding_epsilon: Optional[float] = None,
    embedding_max_neighbors: int = 100,
    deletion_spec: str = "session",
    target_items: Optional[Iterable[int]] = None,
    num_hierarchies: Optional[int] = None,
    rows_per_shard: int = 4096,
    seed: int = 2,
    overwrite: bool = True,
) -> Dict[str, object]:
    """Materialize a (possibly filtered) retain subset under ``out_dir``.

    Retain row budget (both modes):

      ``max_rows = retain_sample_size * |D_f|``

    where ``retain_sample_size`` defaults to ``retain_samples_used_for_update``.
    An optional ``retain_max_rows`` applies a hard upper bound after shuffling.

    When ``neighborhood_aware=True``, ``neighborhood_aware_sample_rate`` in
    ``[0, 1]`` splits the budget between SID-neighbor rows (rate) and uniform
    retain rows (``1 - rate``); rate ``1`` means neighborhood rows only. Rows
    mentioning any forget-item id are never selected.

    Neighborhood mode uses progressive prefix expansion (``k-1`` ... ``1``) unless
    ``progressive_sid_prefix=False``, in which case a single fixed
    ``sid_prefix_length`` is used via :func:`prefix_neighbors`.
    """
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(
                f"Output dir {out_dir} already exists; pass overwrite=True"
            )
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    forget_shards = _list_shards(forget_dir)
    if forget_size is None:
        forget_size = 0
        for path in forget_shards:
            for _ in tf.data.TFRecordDataset([path], compression_type="GZIP"):
                forget_size += 1
    if forget_size <= 0:
        raise ValueError(
            f"forget_size resolved to {forget_size}; expected >0 forget rows."
        )

    tf.random.set_seed(int(seed))

    sample_size = _resolve_repair_sample_bound(
        retain_samples_used_for_update,
        retain_sample_size,
        repair_sample_bound,
    )
    max_rows = max(1, int(math.ceil(sample_size * forget_size)))
    sample_rate = float(neighborhood_aware_sample_rate)
    if not neighborhood_aware:
        sample_rate = 0.0
    sample_rate = max(0.0, min(1.0, sample_rate))
    neighborhood_budget = int(round(max_rows * sample_rate))
    uniform_budget = max_rows - neighborhood_budget

    forget_shard_items = collect_items_in_shards(forget_shards)
    target_item_set = {int(x) for x in (target_items or [])}
    neighborhood_centers = resolve_neighborhood_centers(
        deletion_spec=deletion_spec,
        forget_shard_items=forget_shard_items,
        target_items=target_item_set,
    )
    forbidden_items = resolve_forbidden_retain_items(
        deletion_spec=deletion_spec,
        forget_shard_items=forget_shard_items,
        target_items=target_item_set,
    )
    all_rows, item_to_row_indices, forbidden_row_indices = (
        _scan_retain_rows_with_item_index(
            retain_dir,
            forbidden_items=forbidden_items,
        )
    )
    spam_repair_rows: List[bytes] = []
    if deletion_spec == "item" and target_item_set:
        spam_repair_rows = _collect_spam_repair_rows(
            forget_dir, target_items=target_item_set
        )

    info: Dict[str, object] = {
        "forget_dir": os.path.abspath(forget_dir),
        "retain_dir": os.path.abspath(retain_dir),
        "out_dir": os.path.abspath(out_dir),
        "deletion_spec": str(deletion_spec),
        "neighborhood_method": str(neighborhood_method),
        "n_neighborhood_centers": len(neighborhood_centers),
        "n_target_items": len(target_item_set),
        "n_spam_repair_rows": len(spam_repair_rows),
        "neighborhood_aware": bool(neighborhood_aware),
        "neighborhood_aware_sample_rate": float(sample_rate),
        "neighborhood_budget": int(neighborhood_budget),
        "uniform_budget": int(uniform_budget),
        "semantic_id_path": (
            os.path.abspath(semantic_id_path) if semantic_id_path else None
        ),
        "sid_prefix_length": int(sid_prefix_length),
        "forget_size": int(forget_size),
        "max_rows": int(max_rows),
        "retain_sample_size": int(sample_size),
        "repair_sample_bound": (
            float(repair_sample_bound)
            if repair_sample_bound is not None
            else float(sample_size)
        ),
        "retain_samples_used_for_update": int(retain_samples_used_for_update),
        "retain_max_rows": (
            int(retain_max_rows) if retain_max_rows is not None else None
        ),
        "progressive_sid_prefix": bool(progressive_sid_prefix),
        "neighbor_aware_factor": float(neighbor_aware_factor),
        "rows_per_shard": int(rows_per_shard),
        "seed": int(seed),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_forget_items": len(forget_shard_items),
        "n_neighborhood_center_items": len(neighborhood_centers),
        "n_retain_rows_indexed": len(all_rows),
        "n_forbidden_rows_with_forget_items": len(forbidden_row_indices),
    }

    preselected_rows: Optional[List[bytes]] = None
    allowed_items: Optional[Set[int]] = None
    selection_meta: Dict[str, object] = {}
    neighborhood_rows: List[bytes] = []
    uniform_rows: List[bytes] = []

    if neighborhood_budget > 0:
        method = str(neighborhood_method).strip().lower()
        if method == "embedding":
            if not embedding_path:
                raise ValueError(
                    "neighborhood_method='embedding' requires embedding_path"
                )
            if embedding_epsilon is None:
                raise ValueError(
                    "neighborhood_method='embedding' requires embedding_epsilon"
                )
            embeddings = load_dense_embeddings(embedding_path)
            info["embedding_path"] = os.path.abspath(embedding_path)
            info["embedding_epsilon"] = float(embedding_epsilon)
            info["n_embedding_items"] = int(embeddings.shape[0])
            neighborhood_rows, nbh_meta = select_retain_rows_embedding(
                all_rows,
                item_to_row_indices,
                neighborhood_centers,
                embeddings,
                budget=neighborhood_budget,
                seed=seed,
                epsilon=float(embedding_epsilon),
                max_neighbors=int(embedding_max_neighbors),
                forbidden_row_indices=forbidden_row_indices,
                exclude_target_items=set(neighborhood_centers),
            )
            selection_meta["neighborhood"] = nbh_meta
        else:
            if not semantic_id_path:
                raise ValueError(
                    "neighborhood_aware=True with neighborhood_aware_sample_rate>0 "
                    "requires semantic_id_path (merged_predictions_tensor.pt)"
                )
            codebook = load_codebook(semantic_id_path, num_hierarchies=num_hierarchies)
            nh = int(codebook.shape[1])
            if num_hierarchies is not None and nh != int(num_hierarchies):
                raise ValueError(
                    f"Codebook has {nh} hierarchies but num_hierarchies="
                    f"{num_hierarchies} was passed."
                )
            info["n_codebook_items"] = int(codebook.shape[0])
            info["num_hierarchies"] = nh

            if progressive_sid_prefix:
                start_plen = nh - 1
                neighborhood_rows, nbh_meta = select_retain_rows_progressive(
                    all_rows,
                    item_to_row_indices,
                    neighborhood_centers,
                    codebook,
                    budget=neighborhood_budget,
                    seed=seed,
                    start_prefix_length=start_plen,
                    min_prefix_length=1,
                    retain_max_rows=retain_max_rows,
                    forbidden_row_indices=forbidden_row_indices,
                    exclude_target_items=set(neighborhood_centers),
                )
                selection_meta["neighborhood"] = nbh_meta
                if not neighborhood_rows:
                    print(
                        "[neighborhood] WARNING: progressive selection found 0 retain "
                        "rows; falling back to prefix-neighbor filter."
                    )
                    allowed_items = prefix_neighbors(
                        codebook, neighborhood_centers, min(int(sid_prefix_length), nh)
                    )
                    info["n_neighbor_items"] = len(allowed_items)
                    selection_meta["neighborhood"] = {
                        **nbh_meta,
                        "fallback": "prefix_neighbors",
                    }
                    if not allowed_items:
                        print(
                            "[neighborhood] WARNING: prefix neighbors also empty; "
                            "relying on uniform portion / empty subset."
                        )
                        allowed_items = None
            else:
                allowed_items = prefix_neighbors(
                    codebook, neighborhood_centers, sid_prefix_length
                )
                info["n_neighbor_items"] = len(allowed_items)
                selection_meta["neighborhood"] = {
                    "progressive_prefix": False,
                    "sid_prefix_length": int(sid_prefix_length),
                    "n_neighbor_items": len(allowed_items),
                }
                if not allowed_items:
                    print(
                        "[neighborhood] WARNING: 0 items share any forget SID prefix; "
                        "relying on uniform portion / empty subset."
                    )
                    allowed_items = None

            if allowed_items is not None and neighborhood_budget > 0:
                eligible_indices = [
                    idx
                    for idx in range(len(all_rows))
                    if idx not in forbidden_row_indices
                    and _row_mentions_items_from_index(
                        idx, item_to_row_indices, allowed_items
                    )
                ]
                rng = random.Random(seed)
                rng.shuffle(eligible_indices)
                neighborhood_rows = [
                    all_rows[i] for i in eligible_indices[:neighborhood_budget]
                ]
                selection_meta["neighborhood"] = {
                    **selection_meta.get("neighborhood", {}),
                    "fallback_rows_selected": len(neighborhood_rows),
                }
    else:
        info["n_neighbor_items"] = None

    if spam_repair_rows:
        rng = random.Random(seed + 7)
        rng.shuffle(spam_repair_rows)
        cap = min(len(spam_repair_rows), max(0, max_rows - len(neighborhood_rows)))
        if cap > 0:
            neighborhood_rows.extend(spam_repair_rows[:cap])
            selection_meta["spam_repair"] = {
                "n_available": len(spam_repair_rows),
                "n_added": cap,
            }

    if uniform_budget > 0:
        nbh_row_set = set(neighborhood_rows)
        exclude_indices = {
            idx for idx, row in enumerate(all_rows) if row in nbh_row_set
        }
        uniform_rows, uni_meta = select_retain_rows_uniform(
            all_rows,
            budget=uniform_budget,
            seed=seed + 1,
            forbidden_row_indices=forbidden_row_indices,
            exclude_row_indices=exclude_indices,
        )
        selection_meta["uniform"] = uni_meta

    if neighborhood_rows or uniform_rows:
        preselected_rows, merge_meta = _merge_retain_row_lists(
            [neighborhood_rows, uniform_rows],
            max_rows=max_rows,
            seed=seed,
            retain_max_rows=retain_max_rows,
        )
        selection_meta["merge"] = merge_meta
        info["selection"] = selection_meta
        info["n_neighbor_rows_selected"] = len(neighborhood_rows)
        info["n_uniform_rows_selected"] = len(uniform_rows)

    n_seen_hint = info.get("n_retain_rows_indexed")
    shard_paths, n_kept, n_seen = filter_retain_shards(
        retain_dir=retain_dir,
        out_dir=out_dir,
        allowed_items=allowed_items,
        max_rows=max_rows,
        rows_per_shard=rows_per_shard,
        seed=seed,
        retain_max_rows=retain_max_rows,
        preselected_rows=preselected_rows,
        n_seen_hint=int(n_seen_hint) if n_seen_hint is not None else None,
    )
    info["n_retain_rows_seen"] = int(n_seen)
    info["n_retain_rows_kept"] = int(n_kept)
    info["shard_paths"] = [os.path.relpath(p, out_dir) for p in shard_paths]

    bookkeeping_path = os.path.join(out_dir, "neighborhood_subset.json")
    with open(bookkeeping_path, "w") as f:
        json.dump(info, f, indent=2)
    print(
        f"[neighborhood] kept {n_kept} / {n_seen} retain rows "
        f"(cap={max_rows}, nbh={neighborhood_budget}, uniform={uniform_budget}, "
        f"rate={sample_rate}, forbidden_rows={len(forbidden_row_indices)}) "
        f"-> {out_dir}"
    )
    return info


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Materialise a (possibly neighborhood-filtered) retain subset for SCIF."
        )
    )
    p.add_argument("--forget_dir", required=True)
    p.add_argument("--retain_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--neighborhood_aware", action="store_true")
    p.add_argument("--semantic_id_path", default=None)
    p.add_argument("--sid_prefix_length", type=int, default=2)
    p.add_argument("--forget_size", type=int, default=None)
    p.add_argument("--neighbor_aware_factor", type=float, default=8.0)
    p.add_argument("--retain_samples_used_for_update", type=int, default=16)
    p.add_argument("--retain_sample_size", type=int, default=None)
    p.add_argument("--retain_max_rows", type=int, default=None)
    p.add_argument(
        "--neighborhood_aware_sample_rate",
        type=float,
        default=1.0,
        help=(
            "Fraction of the retain row budget from SID-neighbor sampling "
            "(remainder uniform). 1=neighborhood only, 0=uniform only."
        ),
    )
    p.add_argument(
        "--no_progressive_sid_prefix",
        action="store_true",
        help="Use a single fixed sid_prefix_length instead of k-1..1 expansion.",
    )
    p.add_argument("--rows_per_shard", type=int, default=4096)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    build_retain_subset(
        forget_dir=args.forget_dir,
        retain_dir=args.retain_dir,
        out_dir=args.out_dir,
        neighborhood_aware=args.neighborhood_aware,
        semantic_id_path=args.semantic_id_path,
        sid_prefix_length=args.sid_prefix_length,
        forget_size=args.forget_size,
        neighbor_aware_factor=args.neighbor_aware_factor,
        retain_samples_used_for_update=args.retain_samples_used_for_update,
        retain_sample_size=args.retain_sample_size,
        retain_max_rows=args.retain_max_rows,
        progressive_sid_prefix=not args.no_progressive_sid_prefix,
        neighborhood_aware_sample_rate=float(args.neighborhood_aware_sample_rate),
        rows_per_shard=args.rows_per_shard,
        seed=args.seed,
        overwrite=args.overwrite,
    )
