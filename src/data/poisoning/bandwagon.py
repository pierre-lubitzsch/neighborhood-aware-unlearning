"""Spam-injection (data poisoning) preprocessing for TIGER on GRID.

Adapts a RecBole-style fraud-session generator to the per-user TFRecord
pipeline. Produces a sibling poisoned dataset directory that Hydra configs can
consume unchanged via ``data_dir=...``.

Output layout
-------------
``data/amazon_data/<dataset>_spam_seed<S>_pct<P>_n<C>/``

* ``training/data_*.tfrecord.gz``        clean shards copied verbatim
* ``training/data_spam_*.tfrecord.gz``   new spam-user shards
* ``evaluation/``, ``testing/``, ``items/``  copied verbatim from source
* ``forget_manifest.json``               drives ``split_forget_retain.py``

Each spam example is a single row mirroring the source schema:
``user_id`` is a fresh int64 ID past ``max(clean user_id)``, ``sequence_data``
is the bandwagon attack sequence; any other declared features
(``embedding``, ``text``, ...) are emitted as empty defaults so per-shard
schema inference in ``TFRecordIterator`` stays consistent.

Usage
-----
``python -m src.data.poisoning.bandwagon \\
    --data_dir src/data/amazon_data/beauty \\
    --attack bandwagon --target_strategy unpopular \\
    --poisoning_ratio 0.01 --n_target_items 10 \\
    --placement alternating --seed 42``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


SEQUENCE_FIELD = "sequence_data"
USER_ID_FIELD = "user_id"
TRAINING_SUBDIR = "training"
SIBLING_SUBDIRS = ("evaluation", "testing", "items")


# ---------------------------------------------------------------------------
# Schema inference and shard reading
# ---------------------------------------------------------------------------


def _list_training_shards(training_dir: str) -> List[str]:
    paths = [
        os.path.join(training_dir, f)
        for f in sorted(os.listdir(training_dir))
        if f.endswith(".tfrecord.gz")
    ]
    if not paths:
        raise FileNotFoundError(f"No .tfrecord.gz shards found under {training_dir}")
    return paths


def _infer_feature_description(sample_record: tf.Tensor) -> Dict[str, tf.io.VarLenFeature]:
    """Mirror :class:`TFRecordIterator.infer_feature_type` (VarLen for all fields)."""
    example = tf.train.Example()
    example.ParseFromString(sample_record.numpy())  # type: ignore[arg-type]
    feature_description: Dict[str, tf.io.VarLenFeature] = {}
    for key, value in example.features.feature.items():
        if value.HasField("bytes_list"):
            feature_description[key] = tf.io.VarLenFeature(tf.string)
        elif value.HasField("float_list"):
            feature_description[key] = tf.io.VarLenFeature(tf.float32)
        elif value.HasField("int64_list"):
            feature_description[key] = tf.io.VarLenFeature(tf.int64)
        else:
            raise ValueError(f"Unknown feature type for key {key!r}")
    return feature_description


def _feature_kinds(feature_description: Dict[str, tf.io.VarLenFeature]) -> Dict[str, str]:
    """Return ``{name: 'int64'|'float'|'bytes'}`` for default-filling spam rows."""
    kinds: Dict[str, str] = {}
    for name, spec in feature_description.items():
        dtype = spec.dtype
        if dtype == tf.int64:
            kinds[name] = "int64"
        elif dtype == tf.float32:
            kinds[name] = "float"
        elif dtype == tf.string:
            kinds[name] = "bytes"
        else:  # pragma: no cover  defensive
            raise ValueError(f"Unsupported dtype {dtype} for feature {name!r}")
    return kinds


def _scan_clean_training(
    training_dir: str,
) -> Tuple[List[str], Dict[str, str], Counter, int, int, np.ndarray]:
    """Single-pass scan that collects the statistics needed for the attack.

    Returns
    -------
    shards
        List of source shard paths (sorted).
    feature_kinds
        ``{name: 'int64'|'float'|'bytes'}`` for every declared feature.
    item_counts
        Counter of ``item_id -> #occurrences in sequence_data`` across training.
    max_user_id
        Highest observed integer ``user_id``.
    n_users
        Total number of training rows (one row per user).
    seq_lengths
        ``np.ndarray`` of per-user sequence lengths (used for length sampling).
    """
    shards = _list_training_shards(training_dir)

    raw_dataset = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    sample_record = next(iter(raw_dataset))
    feature_description = _infer_feature_description(sample_record)
    kinds = _feature_kinds(feature_description)

    if SEQUENCE_FIELD not in feature_description:
        raise ValueError(
            f"Source training shards do not contain a {SEQUENCE_FIELD!r} feature; "
            f"got {sorted(feature_description)}"
        )
    if USER_ID_FIELD not in feature_description:
        raise ValueError(
            f"Source training shards do not contain a {USER_ID_FIELD!r} feature; "
            f"got {sorted(feature_description)}"
        )

    item_counts: Counter = Counter()
    max_user_id = -1
    n_users = 0
    seq_lengths: List[int] = []

    parsed = raw_dataset.map(
        lambda x: tf.io.parse_single_example(x, feature_description)
    )
    for example in parsed:
        seq_sparse = example[SEQUENCE_FIELD]
        seq = tf.sparse.to_dense(seq_sparse).numpy()
        if seq.size == 0:
            continue
        item_counts.update(int(x) for x in seq.tolist())
        seq_lengths.append(int(seq.size))

        uid_sparse = example[USER_ID_FIELD]
        uid_dense = tf.sparse.to_dense(uid_sparse).numpy()
        if uid_dense.size == 0:
            continue
        uid = int(uid_dense.flatten()[0])
        if uid > max_user_id:
            max_user_id = uid
        n_users += 1

    if n_users == 0:
        raise ValueError(f"Training scan found 0 users under {training_dir}")

    return (
        shards,
        kinds,
        item_counts,
        max_user_id,
        n_users,
        np.asarray(seq_lengths, dtype=np.int64),
    )


def _feature_kinds_from_one_shard(training_dir: str) -> Dict[str, str]:
    """Infer TFRecord feature dtypes from a single training shard."""
    shards = _list_training_shards(training_dir)
    raw = tf.data.TFRecordDataset([shards[0]], compression_type="GZIP")
    sample_record = next(iter(raw))
    feature_description = _infer_feature_description(sample_record)
    return _feature_kinds(feature_description)


def _scan_stats_from_inter(
    inter_path: str,
    n_clean_users: Optional[int] = None,
    chunksize: int = 2_000_000,
) -> Tuple[Counter, int, int, np.ndarray]:
    """One-pass pandas scan of a RecBole ``.inter`` file.

    Computes item popularity and per-session click counts, which is faster than
    iterating the TFRecord rows when the ``.inter`` file is available.

    Parameters
    ----------
    n_clean_users
        Training-session count for the poisoning-ratio denominator. Required when
        the ``.inter`` file contains more than training sessions (e.g. a merged
        ``rsc15.inter``); take it from ``dataset_meta.json``.
    """
    import pandas as pd

    print(f"[bandwagon] Scanning .inter for stats (chunksize={chunksize}) ...")
    item_counts: Counter = Counter()
    session_clicks: Counter = Counter()
    max_session_id = -1

    compression = "gzip" if inter_path.endswith(".gz") else None
    sid_col: Optional[str] = None

    for chunk in pd.read_csv(
        inter_path, sep="\t", chunksize=chunksize, compression=compression
    ):
        chunk.columns = [str(c).split(":")[0] for c in chunk.columns]
        if sid_col is None:
            sid_col = "session_id" if "session_id" in chunk.columns else "user_id"
        item_counts.update(chunk["item_id"].value_counts().to_dict())
        session_clicks.update(chunk.groupby(sid_col).size().to_dict())
        chunk_max = int(chunk[sid_col].max())
        if chunk_max > max_session_id:
            max_session_id = chunk_max

    if n_clean_users is None:
        n_clean_users = len(session_clicks)

    seq_lengths = np.asarray(list(session_clicks.values()), dtype=np.int64)
    print(
        f"[bandwagon] .inter scan done: unique_items={len(item_counts)} "
        f"| sessions_in_inter={len(session_clicks)} | "
        f"n_clean_users(for ratio)={n_clean_users} | max_session_id={max_session_id}"
    )
    return item_counts, max_session_id, int(n_clean_users), seq_lengths


# ---------------------------------------------------------------------------
# Item-bin selection (popularity bins + targets)
# ---------------------------------------------------------------------------


def _popularity_bins(
    item_counts: Counter,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Return ``(popular, average, unpopular, all_items)`` lists ordered by popularity.

    Top 20% popular, middle 40% (starting at the 30% mark), bottom 20% unpopular.
    """
    items_by_pop = [item for item, _ in item_counts.most_common()]
    n = len(items_by_pop)
    n_popular = max(1, int(n * 0.2))
    n_skip = int(n * 0.3)
    n_average = max(1, int(n * 0.4))
    popular = items_by_pop[:n_popular]
    average = items_by_pop[n_skip : n_skip + n_average]
    n_unpopular = max(1, int(n * 0.2))
    unpopular = items_by_pop[-n_unpopular:]
    return popular, average, unpopular, items_by_pop


def _select_target_items(
    items_by_pop: List[int],
    strategy: str,
    n_target_items: int,
    rng: np.random.Generator,
) -> List[int]:
    n = len(items_by_pop)
    if strategy == "unpopular":
        bottom = items_by_pop[-max(1, int(n * 0.2)) :]
        pool = bottom
    elif strategy == "popular":
        pool = items_by_pop[: max(1, int(n * 0.05))]
    elif strategy in ("mid", "average"):
        # Middle-popularity bin, same as the 'average' bin in _popularity_bins.
        lo = int(n * 0.3)
        hi = lo + max(1, int(n * 0.4))
        pool = items_by_pop[lo:hi]
    elif strategy == "random":
        pool = items_by_pop
    else:
        raise ValueError(f"Unknown target_strategy={strategy!r}")
    size = min(n_target_items, len(pool))
    selected = rng.choice(pool, size=size, replace=False)
    return [int(x) for x in selected.tolist()]


def _filler_pool(
    attack: str,
    popular: List[int],
    average: List[int],
    all_items: List[int],
) -> List[int]:
    if attack == "bandwagon":
        return popular
    if attack == "average":
        return average
    if attack in ("random", "push"):
        return all_items
    raise ValueError(f"Unknown attack={attack!r}")


# ---------------------------------------------------------------------------
# Spam-sequence construction
# ---------------------------------------------------------------------------


def _sample_session_length(
    seq_lengths: np.ndarray,
    rng: np.random.Generator,
    bot_speed_factor: float = 0.8,
    min_len: int = 4,
) -> int:
    """Lognormal-Poisson session length, clipped to the observed [min, max]."""
    mean = max(min_len, float(seq_lengths.mean()) * bot_speed_factor)
    std = max(1.0, float(seq_lengths.std()))
    sigma_squared = math.log(1.0 + (std**2 / mean**2))
    mu = math.log(mean) - sigma_squared / 2
    lambda_param = float(rng.lognormal(mu, math.sqrt(sigma_squared)))
    length = int(max(min_len, rng.poisson(lambda_param)))
    return int(np.clip(length, min_len, int(seq_lengths.max())))


def _build_alternating_sequence(
    length: int,
    targets: List[int],
    fillers: List[int],
    rng: np.random.Generator,
) -> List[int]:
    """``[popular, target, popular, target, ...]`` of total ``length`` items."""
    seq: List[int] = []
    for i in range(length):
        if i % 2 == 0:
            seq.append(int(rng.choice(fillers)))
        else:
            seq.append(int(rng.choice(targets)))
    return seq


def _build_sprinkled_sequence(
    length: int,
    targets: List[int],
    fillers: List[int],
    rng: np.random.Generator,
    p_two_targets: float = 0.119,
) -> List[int]:
    """One target item per spam session, two with probability ``p_two_targets``.

    Target positions are sampled without replacement from ``[0.2*L, 0.9*L]``;
    each target is drawn uniformly from ``targets`` and every other slot is a
    uniform draw from ``fillers``.
    """
    n_targets_max = min(2, len(targets))
    if length < 6:
        # Sequences this short cannot accommodate two well-separated targets.
        n_targets = min(1, n_targets_max)
    else:
        n_targets = 2 if rng.random() < p_two_targets else 1
        n_targets = min(n_targets, n_targets_max)

    lo = max(1, int(length * 0.2))
    hi = max(lo + 1, int(length * 0.9))
    candidate_positions = list(range(lo, hi))
    if n_targets > len(candidate_positions):
        n_targets = len(candidate_positions)
    if n_targets > 0:
        chosen = rng.choice(
            len(candidate_positions), size=n_targets, replace=False
        )
        target_positions = {int(candidate_positions[i]) for i in chosen.tolist()}
    else:
        target_positions = set()

    seq: List[int] = []
    for i in range(length):
        if i in target_positions:
            seq.append(int(rng.choice(targets)))
        else:
            seq.append(int(rng.choice(fillers)))
    return seq


def _build_target_last_sequence(
    length: int,
    targets: List[int],
    fillers: List[int],
    rng: np.random.Generator,
) -> List[int]:
    """``[filler, ..., filler, target]``: the target is the last item.

    Training supervises the last item of each sequence, so the target is placed
    there as the label; the context is drawn from ``fillers``.
    """
    n_ctx = max(1, length - 1)
    seq = [int(rng.choice(fillers)) for _ in range(n_ctx)]
    seq.append(int(rng.choice(targets)))
    return seq


def _build_spam_sequence(
    placement: str,
    length: int,
    targets: List[int],
    fillers: List[int],
    rng: np.random.Generator,
    p_two_targets: float = 0.119,
) -> List[int]:
    if placement == "alternating":
        return _build_alternating_sequence(length, targets, fillers, rng)
    if placement == "sprinkled":
        return _build_sprinkled_sequence(
            length, targets, fillers, rng, p_two_targets=p_two_targets
        )
    if placement == "target_last":
        return _build_target_last_sequence(length, targets, fillers, rng)
    raise ValueError(f"Unknown placement={placement!r}")


# ---------------------------------------------------------------------------
# Segment + clone-append helpers (poison methods beyond bandwagon)
# ---------------------------------------------------------------------------


def _load_semantic_ids(path: str) -> np.ndarray:
    """Load the per-item semantic-ID tensor ``[num_hierarchies, num_items]``.

    Produced by the RKMeans / RVQ step (``merged_predictions_tensor.pt``);
    column ``i`` is item ``i``'s codebook tuple, values in ``[0, codebook_size)``.
    """
    import torch  # only needed for the segment method

    sem = torch.load(path, map_location="cpu", weights_only=False)
    if not hasattr(sem, "numpy"):
        raise ValueError(
            f"Semantic-ID tensor at {path!r} is not a tensor (got {type(sem)})."
        )
    arr = sem.numpy()
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2-D [num_hierarchies, num_items] semantic-ID tensor; "
            f"got shape {arr.shape} at {path!r}."
        )
    return arr.astype(np.int64)


def _semantic_id_segment(
    target: int,
    sem_ids: np.ndarray,
    prefix_len: int,
) -> List[int]:
    """Items sharing ``target``'s semantic-ID prefix of length ``prefix_len``.

    The prefix length is not shortened automatically. The target itself is
    included, so with a long prefix the segment may contain only the target.
    """
    n_hier, n_items = sem_ids.shape
    if not (0 <= target < n_items):
        return []
    plen = int(np.clip(prefix_len, 1, n_hier))
    target_prefix = sem_ids[:plen, target]
    match = np.all(sem_ids[:plen, :] == target_prefix[:, None], axis=0)
    return [int(j) for j in np.nonzero(match)[0]]


def _load_item_embeddings(items_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``items/`` shards -> ``(item_ids, embeddings)`` aligned row-for-row."""
    shards = [
        os.path.join(items_dir, f)
        for f in sorted(os.listdir(items_dir))
        if f.endswith(".tfrecord.gz")
    ]
    if not shards:
        raise FileNotFoundError(f"No item shards under {items_dir}")
    raw = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    feat = _infer_feature_description(next(iter(raw)))
    for required in ("id", "embedding"):
        if required not in feat:
            raise ValueError(
                f"Item shards lack a {required!r} feature; got {sorted(feat)}."
            )
    parsed = raw.map(lambda x: tf.io.parse_single_example(x, feat))
    ids: List[int] = []
    embs: List[np.ndarray] = []
    for ex in parsed:
        iid = tf.sparse.to_dense(ex["id"]).numpy().flatten()
        emb = tf.sparse.to_dense(ex["embedding"]).numpy().flatten()
        if iid.size == 0 or emb.size == 0:
            continue
        ids.append(int(iid[0]))
        embs.append(emb.astype(np.float32))
    return np.asarray(ids, dtype=np.int64), np.vstack(embs)


def _embedding_segment(
    target: int,
    item_ids: np.ndarray,
    embeddings: np.ndarray,
    size: int,
    exclude: set,
) -> List[int]:
    """Top-``size`` cosine-nearest items to ``target`` (excluding ``exclude``)."""
    pos = np.nonzero(item_ids == target)[0]
    if pos.size == 0:
        return []
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    sims = normed @ normed[pos[0]]
    order = np.argsort(-sims)
    members: List[int] = []
    for idx in order:
        iid = int(item_ids[idx])
        if iid == target or iid in exclude:
            continue
        members.append(iid)
        if len(members) >= size:
            break
    return members


def _reservoir_sample_sequences(
    shards: List[str], k: int, rng: np.random.Generator
) -> List[List[int]]:
    """One-pass reservoir sample of ``k`` real ``sequence_data`` sequences.

    Used by the clone-append method to clone genuine browsing context before
    appending the target click.
    """
    raw = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    feat = _infer_feature_description(next(iter(raw)))
    parsed = raw.map(lambda x: tf.io.parse_single_example(x, feat))
    reservoir: List[List[int]] = []
    seen = 0
    for ex in parsed:
        seq = tf.sparse.to_dense(ex[SEQUENCE_FIELD]).numpy().flatten()
        if seq.size == 0:
            continue
        seq_list = [int(x) for x in seq.tolist()]
        if len(reservoir) < k:
            reservoir.append(seq_list)
        else:
            j = int(rng.integers(0, seen + 1))
            if j < k:
                reservoir[j] = seq_list
        seen += 1
    return reservoir


def _build_clone_append_sequence(
    base_seq: List[int],
    targets: List[int],
    rng: np.random.Generator,
    p_two_targets: float,
    max_len: int,
) -> List[int]:
    """Clone a real sequence and append target(s) at the tail.

    Models a hijacked or bot account: genuine browsing context followed by the
    spam click(s), with the target as the supervised last item.
    """
    seq = list(base_seq)
    n_targets = 2 if (len(targets) >= 2 and rng.random() < p_two_targets) else 1
    chosen = rng.choice(targets, size=n_targets, replace=False)
    budget = max(1, int(max_len) - n_targets)
    if len(seq) > budget:
        seq = seq[-budget:]
    seq.extend(int(t) for t in chosen.tolist())
    return seq


def _build_clone_flood_sequence(
    pool: List[List[int]],
    target: int,
    rng: np.random.Generator,
    context_len: int,
    target_set: set,
) -> List[int]:
    """Long target-free real context with a single ``target`` at the tail.

    The training collate expands each sequence into contiguous sub-sequences and
    supervises the last item of each, so a tail target yields one
    ``(real-prefix -> target)`` example per prefix. The context concatenates
    reservoir-sampled real sequences with targets removed and keeps the most
    recent ``context_len`` items. Use ``context_len = sequence_length /
    num_hierarchies - 1`` so the whole sequence fits the token budget.
    """
    context_len = max(1, int(context_len))
    ctx = _sample_target_free_context(pool, rng, context_len, target_set)
    if not ctx:
        # Degenerate fallback (e.g. pool entirely targets): a 1-item context.
        ctx = [int(target)]
    ctx.append(int(target))
    return ctx


def _sample_target_free_context(
    pool: List[List[int]],
    rng: np.random.Generator,
    context_len: int,
    target_set: set,
) -> List[int]:
    """Concatenate reservoir-sampled real sequences, strip target items, and
    keep the most recent ``context_len`` items.

    Shared by ``clone_flood`` and ``clone_inject``. The loop is bounded so an
    all-target pool cannot loop forever.
    """
    context_len = max(1, int(context_len))
    ctx: List[int] = []
    for _ in range(4 * context_len + 8):
        if len(ctx) >= context_len:
            break
        base = pool[int(rng.integers(0, len(pool)))]
        ctx.extend(int(x) for x in base if int(x) not in target_set)
    return ctx[-context_len:]


def _build_clone_inject_sequence(
    pool: List[List[int]],
    target: int,
    rng: np.random.Generator,
    context_len: int,
    target_set: set,
    n_inject: int = 1,
) -> List[int]:
    """Target-free real context with ``n_inject`` copies of ``target`` at
    distinct random non-first positions.

    The context is built as in :func:`_build_clone_flood_sequence`. The total
    length is fixed at ``context_len + 1``, so it fits the token budget and no
    injected target is trimmed away. Slot 0 is always a context item, so every
    target can serve as a ``(prefix -> target)`` label.
    """
    context_len = max(1, int(context_len))
    window = context_len + 1  # fixed total length, same as clone_flood
    # Keep at least one context slot so position 0 is never a target.
    k = max(1, min(int(n_inject), window - 1))
    n_ctx = window - k
    ctx = _sample_target_free_context(pool, rng, n_ctx, target_set)
    if not ctx:
        # Degenerate fallback (pool consists only of targets).
        return [int(target)]
    if k == 1:
        # Single-injection path; keeps the RNG stream of the n_inject=1 case.
        pos = int(rng.integers(1, len(ctx) + 1))
        return ctx[:pos] + [int(target)] + ctx[pos:]
    final_len = len(ctx) + k
    # Guard against a short context: keep >= 1 non-target slot (slot 0).
    k = min(k, final_len - 1)
    # Distinct target slots drawn from positions 1..final_len-1 (never slot 0).
    chosen = rng.choice(final_len - 1, size=k, replace=False) + 1
    target_slots = {int(s) for s in chosen.tolist()}
    seq: List[int] = []
    ctx_iter = iter(ctx)
    for pos in range(final_len):
        seq.append(int(target) if pos in target_slots else int(next(ctx_iter)))
    return seq


# ---------------------------------------------------------------------------
# TFRecord writing
# ---------------------------------------------------------------------------


def _make_example(
    user_id: int,
    sequence: List[int],
    feature_kinds: Dict[str, str],
) -> tf.train.Example:
    """Assemble a ``tf.train.Example`` matching the source schema.

    Fields other than ``user_id`` / ``sequence_data`` are emitted as empty
    lists of the correct dtype so per-shard schema inference stays consistent
    with clean shards.
    """
    feature: Dict[str, tf.train.Feature] = {}
    for name, kind in feature_kinds.items():
        if name == USER_ID_FIELD:
            feature[name] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(user_id)])
            )
        elif name == SEQUENCE_FIELD:
            feature[name] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(x) for x in sequence])
            )
        elif kind == "int64":
            feature[name] = tf.train.Feature(int64_list=tf.train.Int64List(value=[]))
        elif kind == "float":
            feature[name] = tf.train.Feature(float_list=tf.train.FloatList(value=[]))
        elif kind == "bytes":
            feature[name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[]))
        else:  # pragma: no cover  defensive
            raise ValueError(f"Unsupported kind {kind!r} for feature {name!r}")
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _write_spam_shards(
    out_training_dir: str,
    spam_user_ids: List[int],
    spam_sequences: List[List[int]],
    feature_kinds: Dict[str, str],
    rows_per_shard: int,
) -> List[str]:
    """Write spam rows split across ``data_spam_<i>.tfrecord.gz`` shards."""
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    written: List[str] = []
    n_total = len(spam_user_ids)
    if n_total == 0:
        return written
    n_shards = max(1, math.ceil(n_total / rows_per_shard))
    for shard_idx in range(n_shards):
        start = shard_idx * rows_per_shard
        end = min(n_total, start + rows_per_shard)
        path = os.path.join(out_training_dir, f"data_spam_{shard_idx}.tfrecord.gz")
        with tf.io.TFRecordWriter(path, options=options) as writer:
            for i in range(start, end):
                example = _make_example(
                    spam_user_ids[i], spam_sequences[i], feature_kinds
                )
                writer.write(example.SerializeToString())
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def method_suffix(method: str) -> str:
    """Naming token for a poison method: empty for ``bandwagon``, otherwise
    ``_<method>`` (e.g. ``_spam_segment_seed...``)."""
    return "" if method == "bandwagon" else f"_{method}"


def strategy_suffix(target_strategy: str) -> str:
    """Naming token for the target-selection strategy: empty for ``unpopular``
    (the default), otherwise ``_tgt<strategy>`` (e.g. ``_tgtmid``)."""
    return "" if target_strategy == "unpopular" else f"_tgt{target_strategy}"


def _default_out_dir(
    data_dir: str,
    seed: int,
    ratio: float,
    n_targets: int,
    method: str = "bandwagon",
    clone_inject_count: int = 1,
    target_strategy: str = "unpopular",
) -> str:
    parent = os.path.dirname(os.path.abspath(data_dir.rstrip("/"))) or "."
    base = os.path.basename(os.path.abspath(data_dir.rstrip("/")))
    pct = int(round(ratio * 100))
    mtok = method_suffix(method)
    # clone_inject with more than one injection is named e.g. _clone_injectx3.
    if method == "clone_inject" and int(clone_inject_count) > 1:
        mtok = f"{mtok}x{int(clone_inject_count)}"
    stok = strategy_suffix(target_strategy)
    return os.path.join(
        parent,
        f"{base}_spam{mtok}{stok}_seed{seed}_pct{pct}_n{n_targets}",
    )


def _copy_clean_shards(src_training: str, dst_training: str, src_shards: List[str]) -> None:
    os.makedirs(dst_training, exist_ok=True)
    for shard in src_shards:
        dst = os.path.join(dst_training, os.path.basename(shard))
        shutil.copy2(shard, dst)


def _copy_sibling_subdirs(src_dir: str, dst_dir: str) -> None:
    for sub in SIBLING_SUBDIRS:
        s = os.path.join(src_dir, sub)
        if not os.path.isdir(s):
            continue
        d = os.path.join(dst_dir, sub)
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    data_dir: str,
    out_dir: Optional[str],
    attack: str,
    target_strategy: str,
    poisoning_ratio: float,
    n_target_items: int,
    placement: str,
    seed: int,
    rows_per_shard: int,
    overwrite: bool,
    p_two_targets: float = 0.119,
    stats_inter: Optional[str] = None,
    n_clean_users: Optional[int] = None,
    deletion_spec: str = "session",
    method: str = "bandwagon",
    segment_by: str = "semantic_id",
    semantic_id_path: Optional[str] = None,
    segment_prefix_len: int = 2,
    segment_size: int = 200,
    clone_context_len: int = 29,
    clone_pool_size: int = 8192,
    clone_inject_count: int = 1,
) -> str:
    if method not in (
        "bandwagon",
        "segment",
        "clone_append",
        "clone_flood",
        "clone_inject",
    ):
        raise ValueError(f"Unknown method={method!r}")
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    training_dir = os.path.join(data_dir, TRAINING_SUBDIR)
    if not os.path.isdir(training_dir):
        raise FileNotFoundError(f"Expected training dir at {training_dir}")

    if stats_inter:
        if not os.path.isfile(stats_inter):
            raise FileNotFoundError(f"--stats-inter not found: {stats_inter}")
        clean_shards = _list_training_shards(training_dir)
        feature_kinds = _feature_kinds_from_one_shard(training_dir)
        item_counts, max_user_id, n_clean_users, seq_lengths = _scan_stats_from_inter(
            stats_inter, n_clean_users=n_clean_users
        )
        # item_counts from the .inter file are keyed by raw item IDs, while the
        # TFRecords use sequential IDs (0..N-1); remap before selecting items.
        id_map_path = os.path.join(data_dir, "item_id_map.json")
        if os.path.isfile(id_map_path):
            with open(id_map_path, encoding="utf-8") as _f:
                _id_map_data = json.load(_f)
            raw_to_seq = {
                raw: seq
                for seq, raw in enumerate(_id_map_data["seq_to_raw"])
            }
            item_counts = Counter(
                {raw_to_seq[k]: v for k, v in item_counts.items() if k in raw_to_seq}
            )
            print(
                f"[bandwagon] Remapped item_counts from raw IDs to sequential IDs "
                f"using {id_map_path} ({len(item_counts)} items retained)."
            )
        else:
            print(
                f"[bandwagon] WARNING: {id_map_path} not found. "
                "item_counts keys are raw item IDs and will be out of range "
                "at training time. Regenerate the dataset with "
                "convert_rsc15_inter.py to create item_id_map.json, or omit "
                "--stats-inter."
            )
    else:
        print(f"[bandwagon] Scanning clean training shards under {training_dir} ...")
        (
            clean_shards,
            feature_kinds,
            item_counts,
            max_user_id,
            n_clean_users,
            seq_lengths,
        ) = _scan_clean_training(training_dir)
    print(
        f"[bandwagon] features={list(feature_kinds.keys())} | "
        f"users={n_clean_users} | unique_items={len(item_counts)} | "
        f"max_user_id={max_user_id} | seq_len mean={seq_lengths.mean():.2f} "
        f"std={seq_lengths.std():.2f} min={seq_lengths.min()} max={seq_lengths.max()}"
    )

    sessions_to_add = math.ceil(
        poisoning_ratio * n_clean_users / max(1.0 - poisoning_ratio, 1e-9)
    )
    if sessions_to_add <= 0:
        raise ValueError(
            f"poisoning_ratio={poisoning_ratio} produces 0 spam sessions "
            f"for n_clean_users={n_clean_users}; choose a larger ratio."
        )
    print(
        f"[{method}] Will add {sessions_to_add} spam users to hit "
        f"poisoning_ratio={poisoning_ratio:.4f}"
    )

    popular, average, unpopular, items_by_pop = _popularity_bins(item_counts)
    target_items = _select_target_items(
        items_by_pop, target_strategy, n_target_items, rng
    )
    fillers = _filler_pool(attack, popular, average, items_by_pop)
    target_set = set(target_items)
    print(
        f"[{method}] popular={len(popular)} avg={len(average)} unpopular={len(unpopular)} "
        f"| targets={target_items[:5]}{'...' if len(target_items) > 5 else ''} "
        f"| filler_pool={len(fillers)}"
    )

    # --- Method-specific preparation -------------------------------------
    segment_cache: Dict[int, List[int]] = {}
    clone_pool: List[List[int]] = []
    seg_resolved_path: Optional[str] = None
    if method == "segment":
        if segment_by == "semantic_id":
            seg_resolved_path = semantic_id_path
            if not seg_resolved_path:
                base = os.path.basename(os.path.abspath(data_dir.rstrip("/")))
                guess = os.path.join(
                    "embeddings", base, "merged_predictions_tensor.pt"
                )
                if os.path.isfile(guess):
                    seg_resolved_path = guess
            if not seg_resolved_path or not os.path.isfile(seg_resolved_path):
                raise FileNotFoundError(
                    "segment_by=semantic_id needs a semantic-ID tensor; pass "
                    "--semantic_id_path .../merged_predictions_tensor.pt "
                    f"(tried {seg_resolved_path!r})."
                )
            sem_ids = _load_semantic_ids(seg_resolved_path)
            print(
                f"[segment] semantic IDs {sem_ids.shape} from {seg_resolved_path} "
                f"| exact prefix_len={segment_prefix_len} (no shortening)"
            )
            for t in target_items:
                segment_cache[t] = _semantic_id_segment(
                    t, sem_ids, segment_prefix_len
                )
        else:  # embedding
            item_ids_arr, emb_arr = _load_item_embeddings(
                os.path.join(data_dir, "items")
            )
            print(
                f"[segment] embeddings {emb_arr.shape} from items/ "
                f"| neighborhood size={segment_size}"
            )
            for t in target_items:
                segment_cache[t] = _embedding_segment(
                    t, item_ids_arr, emb_arr, segment_size, target_set
                )
        seg_sizes = [len(s) for s in segment_cache.values()]
        empty = [t for t, s in segment_cache.items() if not s]
        if empty:
            print(
                f"[segment] WARNING: {len(empty)} target(s) had an empty segment; "
                f"falling back to the bandwagon filler pool for those."
            )
        print(
            f"[segment] segment sizes: min={min(seg_sizes)} max={max(seg_sizes)} "
            f"mean={sum(seg_sizes) / max(1, len(seg_sizes)):.1f}"
        )
    elif method == "clone_append":
        print(
            f"[clone_append] reservoir-sampling {sessions_to_add} real sequences "
            f"to clone ..."
        )
        clone_pool = _reservoir_sample_sequences(clean_shards, sessions_to_add, rng)
        if not clone_pool:
            raise ValueError("clone_append found no real sequences to clone.")
        print(f"[clone_append] cloned-sequence pool size={len(clone_pool)}")
    elif method in ("clone_flood", "clone_inject"):
        pool_k = max(int(clone_pool_size), sessions_to_add)
        print(
            f"[{method}] reservoir-sampling {pool_k} real sequences to source "
            f"target-free context (context_len={clone_context_len}) ..."
        )
        clone_pool = _reservoir_sample_sequences(clean_shards, pool_k, rng)
        if not clone_pool:
            raise ValueError(f"{method} found no real sequences to clone.")
        print(f"[{method}] context-sequence pool size={len(clone_pool)}")

    max_len = int(seq_lengths.max())

    spam_user_ids: List[int] = []
    spam_sequences: List[List[int]] = []
    n_target_clicks_per_session: List[int] = []
    for i in range(sessions_to_add):
        if method == "clone_append":
            base_seq = clone_pool[int(rng.integers(0, len(clone_pool)))]
            seq = _build_clone_append_sequence(
                base_seq, target_items, rng, p_two_targets, max_len
            )
        elif method == "clone_flood":
            # Round-robin targets across spam users for balanced coverage.
            t = target_items[i % len(target_items)]
            seq = _build_clone_flood_sequence(
                clone_pool, t, rng, clone_context_len, target_set
            )
        elif method == "clone_inject":
            # Like clone_flood, but targets are injected at random non-first positions.
            t = target_items[i % len(target_items)]
            seq = _build_clone_inject_sequence(
                clone_pool, t, rng, clone_context_len, target_set,
                n_inject=clone_inject_count,
            )
        elif method == "segment":
            t = int(rng.choice(target_items))
            seg_fillers = segment_cache.get(t) or fillers
            length = _sample_session_length(seq_lengths, rng)
            # Fillers share the target's semantic-ID prefix, so the last-item label
            # reinforces the target's code bucket regardless of target position.
            seq = _build_spam_sequence(
                placement, length, [t], seg_fillers, rng, p_two_targets=p_two_targets
            )
        else:  # bandwagon
            length = _sample_session_length(seq_lengths, rng)
            seq = _build_spam_sequence(
                placement,
                length,
                target_items,
                fillers,
                rng,
                p_two_targets=p_two_targets,
            )
        spam_user_ids.append(int(max_user_id + 1 + i))
        spam_sequences.append(seq)
        n_target_clicks_per_session.append(sum(1 for x in seq if x in target_set))
    n_clicks = sum(len(s) for s in spam_sequences)
    if n_target_clicks_per_session:
        tgt_arr = np.asarray(n_target_clicks_per_session)
        target_stats = (
            f"target clicks/session: mean={tgt_arr.mean():.3f} "
            f"min={tgt_arr.min()} max={tgt_arr.max()}"
        )
    else:
        target_stats = "target clicks/session: n/a"
    print(
        f"[{method}] Generated {len(spam_user_ids)} spam users, "
        f"{n_clicks} total spam clicks "
        f"({n_clicks / max(1, len(spam_user_ids)):.2f} clicks/user) | "
        f"{target_stats}"
    )

    if out_dir is None:
        out_dir = _default_out_dir(
            data_dir,
            seed=seed,
            ratio=poisoning_ratio,
            n_targets=n_target_items,
            method=method,
            clone_inject_count=clone_inject_count,
            target_strategy=target_strategy,
        )
    out_training_dir = os.path.join(out_dir, TRAINING_SUBDIR)
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {out_dir} already exists; pass --overwrite to replace."
            )
        shutil.rmtree(out_dir)
    os.makedirs(out_training_dir, exist_ok=True)

    print(f"[bandwagon] Copying {len(clean_shards)} clean training shards -> {out_training_dir}")
    _copy_clean_shards(training_dir, out_training_dir, clean_shards)

    print(f"[bandwagon] Writing spam shards (rows_per_shard={rows_per_shard}) ...")
    spam_shard_paths = _write_spam_shards(
        out_training_dir,
        spam_user_ids=spam_user_ids,
        spam_sequences=spam_sequences,
        feature_kinds=feature_kinds,
        rows_per_shard=rows_per_shard,
    )
    print(f"[bandwagon] Wrote {len(spam_shard_paths)} spam shards.")

    print(f"[bandwagon] Copying sibling dirs ({', '.join(SIBLING_SUBDIRS)}) ...")
    _copy_sibling_subdirs(data_dir, out_dir)

    # Check that no spam shards ended up in evaluation/ or testing/.
    for sub in ("evaluation", "testing"):
        sub_dir = os.path.join(out_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        stray = [f for f in os.listdir(sub_dir) if f.startswith("data_spam_")]
        if stray:
            raise RuntimeError(
                f"Split leak: injected spam shards found under {sub_dir}: {stray}. "
                "The boost must only appear in training/."
            )
    print(
        "[bandwagon] Split isolation OK: spam appears only in training/; "
        "evaluation/ and testing/ are verbatim clean copies "
        f"(spam user IDs {max_user_id + 1}..{max_user_id + len(spam_user_ids)} "
        "are absent from the eval/test splits)."
    )

    manifest = {
        "spam_user_ids": spam_user_ids,
        "target_items": target_items,
        "method": method,
        "attack_type": attack,
        "target_strategy": target_strategy,
        "placement": (
            "append_last"
            if method == "clone_append"
            else "append_last_flood"
            if method == "clone_flood"
            else "inject_random_nonfirst"
            if method == "clone_inject"
            else placement
        ),
        "clone_context_len": (
            int(clone_context_len)
            if method in ("clone_flood", "clone_inject")
            else None
        ),
        "clone_inject_count": (
            int(clone_inject_count) if method == "clone_inject" else None
        ),
        "segment_by": segment_by if method == "segment" else None,
        "segment_prefix_len": (
            int(segment_prefix_len)
            if method == "segment" and segment_by == "semantic_id"
            else None
        ),
        "segment_size": (
            int(segment_size)
            if method == "segment" and segment_by == "embedding"
            else None
        ),
        "semantic_id_path": seg_resolved_path if method == "segment" else None,
        "p_two_targets": float(p_two_targets) if placement == "sprinkled" else None,
        "poisoning_ratio": poisoning_ratio,
        "n_target_items": n_target_items,
        "n_clean_users": int(n_clean_users),
        "n_spam_users": int(len(spam_user_ids)),
        "max_clean_user_id": int(max_user_id),
        "first_spam_user_id": int(max_user_id + 1),
        "last_spam_user_id": int(max_user_id + len(spam_user_ids)),
        "seed": int(seed),
        "rows_per_spam_shard": int(rows_per_shard),
        "spam_shard_paths": [os.path.relpath(p, out_dir) for p in spam_shard_paths],
        "source_dataset": os.path.abspath(data_dir),
        "schema_features": sorted(feature_kinds.keys()),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "target_clicks_mean": (
            float(np.mean(n_target_clicks_per_session))
            if n_target_clicks_per_session else None
        ),
        "target_clicks_min": (
            int(min(n_target_clicks_per_session))
            if n_target_clicks_per_session else None
        ),
        "target_clicks_max": (
            int(max(n_target_clicks_per_session))
            if n_target_clicks_per_session else None
        ),
        "deletion_spec": str(deletion_spec).strip().lower(),
    }
    manifest_path = os.path.join(out_dir, "forget_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[bandwagon] Wrote manifest -> {manifest_path}")
    print(f"[bandwagon] Done. Poisoned dataset at {out_dir}")
    return out_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a bandwagon-poisoned TIGER dataset (TFRecord shards)."
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Source clean dataset dir (e.g. src/data/amazon_data/beauty)",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output dir; defaults to <data_dir>_spam_seed<S>_pct<P>_n<C>",
    )
    p.add_argument(
        "--method",
        choices=(
            "bandwagon",
            "segment",
            "clone_append",
            "clone_flood",
            "clone_inject",
        ),
        default="bandwagon",
        help=(
            "Poisoning method. 'bandwagon' (default): fake users mixing filler "
            "items (--attack pool) with targets. 'segment': fillers drawn from "
            "each target's semantic neighborhood (--segment_by). 'clone_append': "
            "clone real sequences and append the target at the tail. "
            "'clone_flood': long target-free real context with one target at "
            "the tail, targets round-robined across spam users. 'clone_inject': "
            "like clone_flood, but the target is injected at a random non-first "
            "position."
        ),
    )
    p.add_argument(
        "--clone_context_len",
        type=int,
        default=29,
        help=(
            "method=clone_flood/clone_inject only: number of target-free real "
            "context items per spam sequence. Set to "
            "sequence_length/num_hierarchies - 1 (29 for 120 tokens and 4 "
            "hierarchies) so the sequence fits the token budget untrimmed."
        ),
    )
    p.add_argument(
        "--clone_inject_count",
        type=int,
        default=1,
        help=(
            "method=clone_inject only: how many copies of the (round-robined) "
            "target to inject per spam session, at distinct random non-first "
            "positions. The session length stays fixed at clone_context_len + 1. "
            "Values >1 name the dataset _clone_injectx<count>."
        ),
    )
    p.add_argument(
        "--clone_pool_size",
        type=int,
        default=8192,
        help=(
            "method=clone_flood/clone_inject only: reservoir-sample size of "
            "real sequences used as the context source (clamped up to "
            "n_spam_users)."
        ),
    )
    p.add_argument(
        "--attack",
        choices=("bandwagon", "random", "average", "push"),
        default="bandwagon",
        help="Filler pool for method=bandwagon (ignored by segment/clone_append).",
    )
    p.add_argument(
        "--segment_by",
        choices=("semantic_id", "embedding"),
        default="semantic_id",
        help=(
            "How to define a target's segment for method=segment. 'semantic_id' "
            "(default): items sharing the target's semantic-ID code prefix. "
            "'embedding': cosine-nearest items from the items/ embeddings."
        ),
    )
    p.add_argument(
        "--semantic_id_path",
        default=None,
        help=(
            "Per-item semantic-ID tensor (merged_predictions_tensor.pt) for "
            "method=segment --segment_by=semantic_id. Defaults to "
            "embeddings/<dataset>/merged_predictions_tensor.pt."
        ),
    )
    p.add_argument(
        "--segment_prefix_len",
        type=int,
        default=2,
        help=(
            "Semantic-ID code prefix length defining a segment (not shortened). "
            "Longer prefixes give smaller, more target-specific segments. "
            "(semantic_id only)"
        ),
    )
    p.add_argument(
        "--segment_size",
        type=int,
        default=200,
        help=(
            "Neighborhood size for --segment_by=embedding only. Ignored by "
            "--segment_by=semantic_id, where --segment_prefix_len controls the "
            "bucket."
        ),
    )
    p.add_argument(
        "--target_strategy",
        choices=("unpopular", "mid", "popular", "random"),
        default="unpopular",
        help=(
            "Which items to target. 'unpopular' (bottom 20%, default), 'mid' "
            "(middle 40%, an ordinary item), 'popular' (top 5%), 'random' (any)."
        ),
    )
    p.add_argument("--poisoning_ratio", type=float, default=0.01)
    p.add_argument("--n_target_items", type=int, default=10)
    p.add_argument(
        "--placement",
        choices=("sprinkled", "alternating", "target_last"),
        default="sprinkled",
        help=(
            "Spam-sequence pattern. 'sprinkled' (default): 1 target per session "
            "(occasionally 2) at random positions in [0.2*L, 0.9*L]. "
            "'alternating': [filler, target, filler, target, ...]. "
            "'target_last': the target is the last item, i.e. the training label."
        ),
    )
    p.add_argument(
        "--p_two_targets",
        type=float,
        default=0.119,
        help=(
            "Probability that a sprinkled spam session contains 2 targets "
            "instead of 1 (mean of 1 + p target clicks per session). "
            "Ignored when --placement=alternating."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rows_per_shard",
        type=int,
        default=1024,
        help="Rows per spam tfrecord shard.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--stats-inter",
        default=None,
        help=(
            "RecBole .inter file for fast statistics (one pandas pass) instead of "
            "scanning all training TFRecords. For a merged rsc15.inter also pass "
            "--n-clean-users (training split size from dataset_meta.json)."
        ),
    )
    p.add_argument(
        "--n-clean-users",
        type=int,
        default=None,
        help="Training-session count for poisoning_ratio (required with merged .inter).",
    )
    p.add_argument(
        "--deletion_spec",
        default="session",
        choices=["session", "item"],
        help="Default deletion specification stored in forget_manifest.json.",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    main(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        attack=args.attack,
        target_strategy=args.target_strategy,
        poisoning_ratio=args.poisoning_ratio,
        n_target_items=args.n_target_items,
        placement=args.placement,
        seed=args.seed,
        rows_per_shard=args.rows_per_shard,
        overwrite=args.overwrite,
        p_two_targets=args.p_two_targets,
        stats_inter=args.stats_inter,
        n_clean_users=args.n_clean_users,
        deletion_spec=args.deletion_spec,
        method=args.method,
        segment_by=args.segment_by,
        semantic_id_path=args.semantic_id_path,
        segment_prefix_len=args.segment_prefix_len,
        segment_size=args.segment_size,
        clone_context_len=args.clone_context_len,
        clone_pool_size=args.clone_pool_size,
        clone_inject_count=args.clone_inject_count,
    )
