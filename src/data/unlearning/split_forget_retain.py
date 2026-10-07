"""Partition a poisoned ``training/`` directory into ``training_forget/`` and
``training_retain/`` based on a ``forget_manifest.json`` produced by
``src/data/poisoning/bandwagon.py``.

Each row in the source TFRecord shards is one user. A row is routed to
``training_forget/`` when its ``user_id`` is in
``forget_manifest['spam_user_ids']`` (or, optionally, in an explicit
``--forget_user_ids`` override); all other rows go to ``training_retain/``.

The two output sub-directories share the source schema, so the existing TIGER
Hydra config can target them via ``data_loading.*.dataloader.data_folder=
${paths.data_dir}/training_forget`` (or ``training_retain``).

Usage
-----
``python -m src.data.unlearning.split_forget_retain \\
    --data_dir <data_dir> --forget_manifest <data_dir>/forget_manifest.json``

Outputs::

    <data_dir>/training_forget/data_*.tfrecord.gz
    <data_dir>/training_retain/data_*.tfrecord.gz
    <data_dir>/forget_retain_split.json   # bookkeeping
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from src.data.unlearning.deletion_spec import (
    load_forget_manifest,
    load_target_items,
    manifest_deletion_spec,
    normalize_deletion_spec,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


SEQUENCE_FIELD = "sequence_data"
USER_ID_FIELD = "user_id"
TRAINING_SUBDIR = "training"
FORGET_SUBDIR = "training_forget"
RETAIN_SUBDIR = "training_retain"


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


def _load_forget_user_ids(
    forget_manifest: Optional[str],
    forget_user_ids_arg: Optional[List[int]],
) -> Set[int]:
    if forget_user_ids_arg:
        return {int(x) for x in forget_user_ids_arg}
    if not forget_manifest:
        raise ValueError(
            "Either --forget_manifest or --forget_user_ids must be provided."
        )
    with open(forget_manifest, "r") as f:
        manifest = json.load(f)
    if "spam_user_ids" not in manifest:
        raise ValueError(
            f"forget_manifest at {forget_manifest!r} is missing 'spam_user_ids'."
        )
    return {int(x) for x in manifest["spam_user_ids"]}


class _ShardWriter:
    """Round-robin writer that flushes a new shard every ``rows_per_shard`` rows."""

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

    def write(self, serialized_example: bytes) -> None:
        writer = self._ensure_writer()
        writer.write(serialized_example)
        self._row_idx += 1
        self.total_rows += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _split_segregated_shards(
    src_training_dir: str,
    out_forget_dir: str,
    out_retain_dir: str,
    spam_shard_prefix: str = "data_spam_",
) -> Tuple[List[str], List[str]]:
    """Copy shards by filename without parsing TFRecords (bandwagon layout).

    Clean rows live in ``partition_*.tfrecord.gz``; spam rows in
    ``data_spam_*.tfrecord.gz`` only.
    """
    os.makedirs(out_forget_dir, exist_ok=True)
    os.makedirs(out_retain_dir, exist_ok=True)
    forget_paths: List[str] = []
    retain_paths: List[str] = []

    for name in sorted(os.listdir(src_training_dir)):
        if not name.endswith(".tfrecord.gz"):
            continue
        src = os.path.join(src_training_dir, name)
        if name.startswith(spam_shard_prefix):
            dst = os.path.join(out_forget_dir, name)
            forget_paths.append(dst)
        else:
            dst = os.path.join(out_retain_dir, name)
            retain_paths.append(dst)
        shutil.copy2(src, dst)

    return forget_paths, retain_paths


def _split_shards(
    src_training_dir: str,
    out_forget_dir: str,
    out_retain_dir: str,
    forget_user_ids: Set[int],
    rows_per_shard: int,
) -> Tuple[List[str], List[str], int, int, int]:
    shards = _list_shards(src_training_dir)
    if not shards:
        raise FileNotFoundError(f"No .tfrecord.gz shards found under {src_training_dir}")

    raw_dataset = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    sample_record = next(iter(raw_dataset))
    feature_description = _infer_feature_description(sample_record)
    if USER_ID_FIELD not in feature_description:
        raise ValueError(
            f"Source shards do not contain a {USER_ID_FIELD!r} feature; "
            f"got {sorted(feature_description)}"
        )

    forget_writer = _ShardWriter(out_forget_dir, "data", rows_per_shard)
    retain_writer = _ShardWriter(out_retain_dir, "data", rows_per_shard)

    rows_seen = 0
    rows_with_no_user_id = 0
    parsed = raw_dataset.map(
        lambda x: (x, tf.io.parse_single_example(x, feature_description))
    )
    for raw, example in parsed:
        rows_seen += 1
        uid_dense = tf.sparse.to_dense(example[USER_ID_FIELD]).numpy()
        if uid_dense.size == 0:
            rows_with_no_user_id += 1
            retain_writer.write(raw.numpy())
            continue
        uid = int(uid_dense.flatten()[0])
        if uid in forget_user_ids:
            forget_writer.write(raw.numpy())
        else:
            retain_writer.write(raw.numpy())

    forget_writer.close()
    retain_writer.close()

    return (
        forget_writer.shard_paths,
        retain_writer.shard_paths,
        forget_writer.total_rows,
        retain_writer.total_rows,
        rows_with_no_user_id,
    )


def main(
    data_dir: str,
    forget_manifest: Optional[str],
    forget_user_ids: Optional[List[int]],
    out_subdir_forget: str,
    out_subdir_retain: str,
    rows_per_shard: int,
    overwrite: bool,
    segregated_shards: bool = False,
    deletion_spec: Optional[str] = None,
) -> None:
    src_training = os.path.join(data_dir, TRAINING_SUBDIR)
    if not os.path.isdir(src_training):
        raise FileNotFoundError(f"Expected training dir at {src_training}")

    out_forget = os.path.join(data_dir, out_subdir_forget)
    out_retain = os.path.join(data_dir, out_subdir_retain)
    for out in (out_forget, out_retain):
        if os.path.exists(out):
            if not overwrite:
                raise FileExistsError(
                    f"Output dir {out} already exists; pass --overwrite to replace."
                )
            shutil.rmtree(out)

    forget_set = _load_forget_user_ids(forget_manifest, forget_user_ids)
    manifest = load_forget_manifest(forget_manifest)
    spec = manifest_deletion_spec(manifest, deletion_spec)
    target_items = sorted(load_target_items(manifest))
    print(
        f"[split] Forget set size = {len(forget_set)} "
        f"(source manifest = {forget_manifest}, deletion_spec={spec})"
    )

    if segregated_shards:
        print(
            f"[split] Fast segregated copy (spam shards -> forget, clean -> retain) "
            f"under {src_training} ..."
        )
        forget_shard_paths, retain_shard_paths = _split_segregated_shards(
            src_training_dir=src_training,
            out_forget_dir=out_forget,
            out_retain_dir=out_retain,
        )
        n_rows_no_uid = 0
        n_forget_rows = len(forget_set)
        n_retain_rows = 0
        if forget_manifest and os.path.isfile(forget_manifest):
            with open(forget_manifest, encoding="utf-8") as fh:
                manifest = json.load(fh)
            n_forget_rows = int(manifest.get("n_spam_users", n_forget_rows))
            n_retain_rows = int(manifest.get("n_clean_users", n_retain_rows))
        print(
            f"[split] Row counts from manifest: forget={n_forget_rows} retain≈{n_retain_rows} "
            f"(no full TFRecord scan)"
        )
    else:
        print(f"[split] Splitting training shards under {src_training} ...")
        (
            forget_shard_paths,
            retain_shard_paths,
            n_forget_rows,
            n_retain_rows,
            n_rows_no_uid,
        ) = _split_shards(
            src_training_dir=src_training,
            out_forget_dir=out_forget,
            out_retain_dir=out_retain,
            forget_user_ids=forget_set,
            rows_per_shard=rows_per_shard,
        )
    total = n_forget_rows + n_retain_rows
    pct = 100.0 * n_forget_rows / max(1, total)
    print(
        f"[split] Wrote {n_forget_rows} forget rows -> {out_forget} "
        f"({len(forget_shard_paths)} shards)"
    )
    print(
        f"[split] Wrote {n_retain_rows} retain rows -> {out_retain} "
        f"({len(retain_shard_paths)} shards)"
    )
    if n_rows_no_uid:
        print(
            f"[split] Routed {n_rows_no_uid} rows lacking user_id to retain (defensive)."
        )
    print(f"[split] Forget fraction observed = {pct:.4f}% of {total} training rows")

    bookkeeping = {
        "data_dir": os.path.abspath(data_dir),
        "training_dir": os.path.abspath(src_training),
        "forget_dir": os.path.abspath(out_forget),
        "retain_dir": os.path.abspath(out_retain),
        "forget_manifest": (
            os.path.abspath(forget_manifest) if forget_manifest else None
        ),
        "deletion_spec": spec,
        "target_items": target_items,
        "forget_user_ids_count": len(forget_set),
        "n_forget_rows": int(n_forget_rows),
        "n_retain_rows": int(n_retain_rows),
        "n_rows_no_user_id": int(n_rows_no_uid),
        "forget_shard_paths": [os.path.relpath(p, data_dir) for p in forget_shard_paths],
        "retain_shard_paths": [os.path.relpath(p, data_dir) for p in retain_shard_paths],
        "rows_per_shard": int(rows_per_shard),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    bookkeeping_path = os.path.join(data_dir, "forget_retain_split.json")
    with open(bookkeeping_path, "w") as f:
        json.dump(bookkeeping, f, indent=2)
    print(f"[split] Wrote bookkeeping -> {bookkeeping_path}")


def _parse_int_csv(s: str) -> List[int]:
    if not s:
        return []
    return [int(tok) for tok in s.split(",") if tok.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Partition a TIGER training/ TFRecord directory into training_forget/ "
            "and training_retain/ based on user_id."
        )
    )
    p.add_argument("--data_dir", required=True, help="Dataset dir containing training/")
    p.add_argument(
        "--forget_manifest",
        default=None,
        help="Path to forget_manifest.json from bandwagon.py (provides spam_user_ids).",
    )
    p.add_argument(
        "--forget_user_ids",
        default=None,
        type=_parse_int_csv,
        help="Comma-separated user_id list; overrides --forget_manifest if provided.",
    )
    p.add_argument(
        "--out_subdir_forget", default=FORGET_SUBDIR, help="Output sub-directory name."
    )
    p.add_argument(
        "--out_subdir_retain", default=RETAIN_SUBDIR, help="Output sub-directory name."
    )
    p.add_argument("--rows_per_shard", type=int, default=4096)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--segregated-shards",
        action="store_true",
        help=(
            "Bandwagon layout: copy data_spam_*.tfrecord.gz to forget and other "
            "shards to retain without parsing every row."
        ),
    )
    p.add_argument(
        "--deletion_spec",
        default=None,
        choices=sorted({"session", "item"}),
        help="Deletion specification written to forget_retain_split.json.",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    main(
        data_dir=args.data_dir,
        forget_manifest=args.forget_manifest,
        forget_user_ids=args.forget_user_ids,
        out_subdir_forget=args.out_subdir_forget,
        out_subdir_retain=args.out_subdir_retain,
        rows_per_shard=args.rows_per_shard,
        overwrite=args.overwrite,
        segregated_shards=args.segregated_shards,
        deletion_spec=args.deletion_spec,
    )
