"""Build a forget_manifest.json for sensitive-category unlearning.

Unlike spam, sensitive-category deletion removes legitimate items that a user
should no longer be recommended (e.g. meat for a vegetarian user), following
the ERASE benchmark (arXiv:2603.08341).

The manifest uses the same schema as ``bandwagon.py``:

    target_items    the sensitive items I_s
    spam_user_ids   users whose sessions touch I_s (key name shared with the
                    spam scenario so ``split_forget_retain`` works unchanged)
    scenario        "sensitive"

Categories are parsed from each item's ``text`` feature, which contains
``Categories: [...]``. ``--keywords`` instead applies ERASE-style keyword
matching to the whole text.

Examples
--------
    # toys, the "Baby & Toddler Toys" category at path index 1
    python -m scripts.build_sensitive_manifest \
        --data_dir src/data/amazon_data/toys \
        --category "Baby & Toddler Toys" --category_level 1 \
        --out src/data/amazon_data/toys_sensitive_baby/forget_manifest.json

    # ERASE-style keyword matching
    python -m scripts.build_sensitive_manifest --data_dir <dir> \
        --keywords meat beef pork chicken lamb turkey bacon ham sausage salami
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import tensorflow as tf

CATEGORIES_RE = re.compile(r"Categories:\s*(\[.*?\])\s*;")


def _shards(directory: str) -> List[str]:
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".tfrecord.gz")
    ]


def parse_categories(text: str) -> List[str]:
    """Category path out of an item's ``text`` feature; [] when absent."""
    m = CATEGORIES_RE.search(text)
    if not m:
        return []
    try:
        val = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return []
    return [str(x) for x in val] if isinstance(val, (list, tuple)) else []


def scan_items(
    items_dir: str,
    category: Optional[str],
    category_level: Optional[int],
    keywords: Optional[Sequence[str]],
) -> Tuple[Set[int], int, Dict[str, int]]:
    """Return (matching item ids, total items, level-`category_level` histogram).

    Keyword matching is whole-word and case-insensitive over the whole ``text``
    field, matching ERASE's ``identify_sensitive_items.py``.
    """
    pat = None
    if keywords:
        pat = re.compile(
            r"\b(" + "|".join(re.escape(k.lower()) for k in keywords) + r")\b",
            re.IGNORECASE,
        )
    hits: Set[int] = set()
    total = 0
    hist: Dict[str, int] = {}
    for shard in _shards(items_dir):
        for rec in tf.data.TFRecordDataset(shard, compression_type="GZIP"):
            ex = tf.train.Example.FromString(rec.numpy())
            feat = ex.features.feature
            item_id = int(feat["id"].int64_list.value[0])
            text = feat["text"].bytes_list.value[0].decode("utf-8", "replace")
            total += 1
            cats = parse_categories(text)
            if category_level is not None and len(cats) > category_level:
                hist[cats[category_level]] = hist.get(cats[category_level], 0) + 1
            if pat is not None:
                if pat.search(text):
                    hits.add(item_id)
            elif category is not None:
                if category_level is None:
                    if any(c == category for c in cats):
                        hits.add(item_id)
                elif len(cats) > category_level and cats[category_level] == category:
                    hits.add(item_id)
    return hits, total, hist


def scan_users(
    training_dir: str, targets: Set[int]
) -> Tuple[List[Tuple[int, int]], int, int, int]:
    """Scan the training split once.

    Returns ``(touching, n_users, n_touch_interactions, n_interactions_total)``
    where ``touching`` is ``[(user_id, n_sensitive_interactions), ...]``.
    """
    touching: List[Tuple[int, int]] = []
    n_users = 0
    n_touch = 0
    n_total = 0
    for shard in _shards(training_dir):
        for rec in tf.data.TFRecordDataset(shard, compression_type="GZIP"):
            ex = tf.train.Example.FromString(rec.numpy())
            feat = ex.features.feature
            uid = int(feat["user_id"].int64_list.value[0])
            seq = list(feat["sequence_data"].int64_list.value)
            n_users += 1
            n_total += len(seq)
            k = sum(1 for i in seq if i in targets)
            if k:
                touching.append((uid, k))
                n_touch += k
    return touching, n_users, n_touch, n_total


def sample_forget_users(
    touching: Sequence[Tuple[int, int]],
    n_interactions_total: int,
    forget_ratio: float,
    seed: int,
) -> Tuple[List[int], int]:
    """Sample users until their sensitive interactions reach ``forget_ratio``
    of all interactions (ERASE-style forget-set sizing).

    Users are shuffled with ``seed`` and added whole, so the realized count is
    at or just above the budget. Returns (sorted user ids, interaction count).
    """
    budget = forget_ratio * float(n_interactions_total)
    rng = random.Random(seed)
    order = list(touching)
    rng.shuffle(order)
    chosen: List[int] = []
    acc = 0
    for uid, k in order:
        if acc >= budget:
            break
        chosen.append(uid)
        acc += k
    return sorted(chosen), acc


def write_forget_pairs(
    training_dir: str,
    targets: Set[int],
    forget_users: Set[int],
    out_path: str,
) -> int:
    """Write the forget set as an ERASE-style TSV, one row per interaction.

    Uses the RecBole header ``user_id:token item_id:token rating:float
    timestamp:float``. The TFRecords carry no ratings or timestamps, so
    ``rating`` is 0.0 and ``timestamp`` is the item's position in the session.
    Returns the number of rows written.
    """
    n = 0
    with open(out_path, "w") as fh:
        fh.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for shard in _shards(training_dir):
            for rec in tf.data.TFRecordDataset(shard, compression_type="GZIP"):
                ex = tf.train.Example.FromString(rec.numpy())
                feat = ex.features.feature
                uid = int(feat["user_id"].int64_list.value[0])
                if uid not in forget_users:
                    continue
                for pos, item in enumerate(feat["sequence_data"].int64_list.value):
                    if int(item) in targets:
                        fh.write(f"{uid}\t{int(item)}\t0.0\t{float(pos)}\n")
                        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", default=None, help="default <data_dir>/forget_manifest.json")
    p.add_argument("--category", default=None, help='e.g. "Baby & Toddler Toys"')
    p.add_argument(
        "--category_level",
        type=int,
        default=None,
        help="index into the category path; omit to match at any level",
    )
    p.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="ERASE-style whole-word match over the item text (overrides --category)",
    )
    p.add_argument("--deletion_spec", default="item_pairs",
                   choices=["session", "item", "item_pairs"])
    p.add_argument("--seed", type=int, default=2,
                   help="ERASE uses seeds 2/3/5/7/11")
    p.add_argument(
        "--forget_ratio",
        type=float,
        default=1e-4,
        help="fraction of all interactions to place in the forget set "
             "(ERASE: 1e-4 / 1e-5 / 1e-6). Use 0 to take every touching user.",
    )
    p.add_argument(
        "--list_categories",
        type=int,
        default=None,
        metavar="LEVEL",
        help="print the category histogram at LEVEL and exit (pick a category)",
    )
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    items_dir = os.path.join(args.data_dir, "items")
    training_dir = os.path.join(args.data_dir, "training")
    for d in (items_dir, training_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"missing directory: {d}")

    if args.list_categories is not None:
        _, total, hist = scan_items(items_dir, None, args.list_categories, None)
        print(f"{total} items; level-{args.list_categories} categories:")
        for name, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  {n:6d}  ({100.0 * n / max(total,1):5.2f}%)  {name}")
        return 0

    if not args.category and not args.keywords:
        raise SystemExit("give --category or --keywords (or --list_categories LEVEL)")

    targets, total_items, _ = scan_items(
        items_dir, args.category, args.category_level, args.keywords
    )
    if not targets:
        raise SystemExit(
            "no items matched. Run --list_categories <level> to see the available "
            "category names at that depth."
        )
    share = 100.0 * len(targets) / max(total_items, 1)
    print(f"[items] {len(targets)} / {total_items} matched ({share:.2f}% of catalog)")
    if share > 15.0:
        print(
            f"[warn] {share:.1f}% of the catalog is sensitive. Sensitive@k / SH@k "
            "have little headroom for a category this large; consider a narrower "
            "sub-category."
        )

    touching, n_users, n_touch, n_total = scan_users(training_dir, targets)
    print(
        f"[users] {len(touching)} / {n_users} sessions touch a sensitive item "
        f"({100.0 * len(touching) / max(n_users,1):.2f}%), "
        f"{n_touch} of {n_total} interactions "
        f"({100.0 * n_touch / max(n_total,1):.3f}%)"
    )

    if args.forget_ratio and args.forget_ratio > 0:
        users, n_forget_inter = sample_forget_users(
            touching, n_total, args.forget_ratio, args.seed
        )
        print(
            f"[forget] ERASE-style ratio {args.forget_ratio:g}: sampled "
            f"{len(users)} users / {n_forget_inter} sensitive interactions "
            f"(realized {100.0 * n_forget_inter / max(n_total,1):.4f}% of all "
            f"interactions, budget {args.forget_ratio * n_total:.1f})"
        )
        if not users:
            print(
                "[warn] the ratio rounds to zero users. Raise --forget_ratio or "
                "pass --forget_ratio 0 to take every touching user."
            )
    else:
        users = sorted(u for u, _ in touching)
        n_forget_inter = n_touch
        print(
            f"[forget] --forget_ratio 0: taking all {len(users)} touching users "
            f"({100.0 * len(users) / max(n_users,1):.1f}% of the catalog's users). "
            "split_forget_retain routes by user, so their other interactions "
            "leave the retain set too."
        )

    manifest = {
        "scenario": "sensitive",
        "deletion_spec": args.deletion_spec,
        "target_items": sorted(int(i) for i in targets),
        # Key name expected by split_forget_retain.
        "spam_user_ids": [int(u) for u in users],
        "n_spam_users": len(users),
        "n_target_items": len(targets),
        "category": args.category,
        "category_level": args.category_level,
        "keywords": list(args.keywords) if args.keywords else None,
        "seed": args.seed,
        "source_dataset": os.path.abspath(args.data_dir),
        "catalog_size": total_items,
        "target_share_pct": round(share, 4),
        # ERASE-style forget-set sizing (generate_forget_sets.py --forget-ratio).
        "forget_ratio": args.forget_ratio,
        "n_forget_interactions": int(n_forget_inter),
        "n_interactions_total": int(n_total),
        "realized_forget_ratio": round(n_forget_inter / max(n_total, 1), 8),
        "n_users_touching_category": len(touching),
        # No rating threshold: the TFRecords carry no ratings.
        "rating_threshold": None,
    }
    out = args.out or os.path.join(args.data_dir, "forget_manifest.json")
    # ERASE forget-set filename convention.
    cat_tok = (
        args.category.lower().replace(" & ", "_").replace(" ", "_")
        if args.category
        else ("_".join(args.keywords[:2]).lower() if args.keywords else "sensitive")
    )
    pairs_name = (
        f"{os.path.basename(os.path.normpath(args.data_dir))}"
        f"_unlearn_pairs_sensitive_category_{cat_tok}"
        f"_seed_{args.seed}_unlearning_fraction_{args.forget_ratio:g}.inter"
    )
    pairs_path = os.path.join(os.path.dirname(os.path.abspath(out)), pairs_name)

    if args.dry_run:
        print(f"[dry-run] would write {out}")
        print(f"[dry-run] would write {pairs_path}")
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[write] {out}")
    n_pairs = write_forget_pairs(training_dir, targets, set(users), pairs_path)
    print(f"[write] {pairs_path}  ({n_pairs} interactions)")
    manifest["forget_pairs_file"] = pairs_name
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
