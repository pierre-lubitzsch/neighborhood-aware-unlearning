"""Assign semantic IDs with LETTER's collision handling (no dedup digit).

Unlike ``scripts/assign_sids_exact.py``, which appends a dedup digit, this
follows LETTER's ``generate_indices.py``: every level is assigned by argmin, then
items that share a full code are re-assigned on the final level with Sinkhorn
balanced assignment on their residuals, for up to ``--max_rounds`` rounds. The
final position is therefore a semantic code, so depth-based diagnostics are not
comparable with dedup-digit tensors. Items with identical latents cannot be
separated; the script fails if the final collision rate exceeds
``--max_collision_rate``.

Usage::

    python -m scripts.assign_sids_letter \\
        --codebook_ckpt <codebook.ckpt> \\
        --embedding_path <embeddings.pt> \\
        --out <merged_predictions_tensor.pt>
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
from typing import Dict, List

import torch

from src.components.quantization_strategies import (
    center_distances_for_constraint,
    sinkhorn,
)
from src.components.unlearning.tracer_tokenizer import load_rq_quantizer
from scripts.assign_sids_exact import _load_embeddings

log = logging.getLogger(__name__)


def _collision_groups(codes: torch.Tensor) -> List[List[int]]:
    """Row indices grouped by identical full code tuples, groups of size > 1."""
    buckets: Dict[tuple, List[int]] = collections.defaultdict(list)
    for row, code in enumerate(codes.tolist()):
        buckets[tuple(code)].append(row)
    return [rows for rows in buckets.values() if len(rows) > 1]


def assign(
    latents: torch.Tensor,
    centroids: List[torch.Tensor],
    normalize_inputs: bool,
    normalize_residuals: bool,
    sk_epsilon: float,
    sk_iters: int,
    max_rounds: int,
) -> torch.Tensor:
    """LETTER assignment: argmin everywhere, then resample collisions on the last level.

    Returns ``[N, L]`` int64 codes.
    """
    r = latents.double()
    if normalize_inputs:
        r = torch.nn.functional.normalize(r, dim=-1)

    codes: List[torch.Tensor] = []
    residual_before_last = None
    for level, c in enumerate(centroids):
        if level == len(centroids) - 1:
            # Residual entering the final level, reused by the collision loop.
            residual_before_last = r.clone()
        cd = c.double()
        s = torch.cdist(r, cd).pow(2).argmin(dim=-1)
        codes.append(s)
        r = r - cd[s]
        if normalize_residuals:
            r = torch.nn.functional.normalize(r, dim=-1)

    out = torch.stack(codes, dim=-1).to(torch.int64)             # [N, L]
    n = out.size(0)
    last = centroids[-1].double()

    initial = len(_collision_groups(out))
    log.info(
        "[letter-assign] argmin pass: %d collision groups covering %d items",
        initial,
        sum(len(g) for g in _collision_groups(out)),
    )
    if sk_epsilon <= 0:
        log.warning(
            "[letter-assign] sk_epsilon=%s disables the collision loop; the "
            "reference forces 0.003 on the final level here.",
            sk_epsilon,
        )
        return out

    for rnd in range(max_rounds):
        groups = _collision_groups(out)
        if not groups:
            log.info("[letter-assign] round %d: no collisions left", rnd)
            break
        moved = 0
        for rows in groups:
            idx = torch.tensor(rows, dtype=torch.long)
            d = torch.cdist(residual_before_last[idx], last).pow(2)
            q = sinkhorn(center_distances_for_constraint(d), sk_epsilon, sk_iters)
            if torch.isnan(q).any() or torch.isinf(q).any():
                # Skip the group; NaN rows would argmax to the same code.
                continue
            new = q.argmax(dim=-1)
            moved += int((new != out[idx, -1]).sum())
            out[idx, -1] = new
        remaining = _collision_groups(out)
        log.info(
            "[letter-assign] round %d: %d groups / %d items left (%d codes moved)",
            rnd,
            len(remaining),
            sum(len(g) for g in remaining),
            moved,
        )
        if moved == 0:
            log.info("[letter-assign] fixed point reached; stopping")
            break

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--codebook_ckpt", required=True)
    p.add_argument("--embedding_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_levels", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--sk_epsilon", type=float, default=0.003,
                   help="Sinkhorn epsilon for the collision loop (reference: 0.003)")
    p.add_argument("--sk_iters", type=int, default=50)
    p.add_argument("--max_rounds", type=int, default=20,
                   help="reference: 20")
    p.add_argument("--max_collision_rate", type=float, default=0.01,
                   help="fail if more than this fraction of items still share an "
                        "id with another item (reference reports 0.0012)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")

    front = load_rq_quantizer(args.codebook_ckpt, n_levels=args.n_levels)
    z = _load_embeddings(args.embedding_path)
    log.info(
        "[letter-assign] quantizer=%s levels=%d items=%d dim=%d",
        front.quantizer, len(front.centroids), z.shape[0], z.shape[1],
    )

    if front.project is not None:
        lat = torch.cat(
            [front.project(z[i : i + args.batch_size]).cpu()
             for i in range(0, len(z), args.batch_size)]
        )
    else:
        lat = z

    codes = assign(
        lat,
        front.centroids,
        normalize_inputs=front.normalize_inputs,
        normalize_residuals=front.normalize_residuals,
        sk_epsilon=args.sk_epsilon,
        sk_iters=args.sk_iters,
        max_rounds=args.max_rounds,
    )

    n = codes.size(0)
    groups = _collision_groups(codes)
    collided = sum(len(g) for g in groups)
    rate = collided / n
    per_level_max = [int(codes[:, i].max()) for i in range(codes.size(1))]
    unique = len({tuple(c) for c in codes.tolist()})
    log.info(
        "[letter-assign] final: %d/%d items collided (%.4f%%), %d unique ids, "
        "per-level max code %s",
        collided, n, 100 * rate, unique, per_level_max,
    )

    # Saved as [L, N], without a dedup digit.
    tensor = codes.t().contiguous()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(tensor.cpu(), args.out)
    log.info("[letter-assign] wrote %s shape=%s", args.out, tuple(tensor.shape))

    if rate > args.max_collision_rate:
        raise SystemExit(
            f"collision rate {rate:.4%} exceeds --max_collision_rate "
            f"{args.max_collision_rate:.4%}: {collided} items share an identifier "
            f"with another item."
        )


if __name__ == "__main__":
    main()
