"""Re-assign semantic ids deterministically from a quantizer checkpoint.

Assignment is done in float64 on CPU with the same front end TRACER uses
(`load_rq_quantizer`), so that TRACER with `phi = 0` reproduces the stored ids
exactly. GPU float32 assignment can differ from the exact argmin for a
noticeable fraction of items at deeper levels. Run this only before any model is
trained on the tensor, since it may change the identifier space.

    python -m scripts.assign_sids_exact \
        --codebook_ckpt CODEBOOK_CKPT \
        --embedding_path EMBEDDINGS.pt \
        --out OUT.pt
"""

from __future__ import annotations

import argparse
import logging
import os

import torch

from src.components.unlearning.tracer_tokenizer import (
    assert_reproduces_sids,
    load_rq_quantizer,
)
from src.utils.tensor_utils import (
    deduplicate_rows_in_tensor,
    transpose_tensor_from_file,
)

log = logging.getLogger(__name__)


def _load_embeddings(path: str) -> torch.Tensor:
    """Load a bare [N, D] tensor or an {embeddings, item_ids} dict.

    For the dict form, rows are reordered by item_id so that row i is item i.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        emb = torch.as_tensor(obj["embeddings"])
        ids = torch.as_tensor(obj["item_ids"]).long()
        order = torch.argsort(ids)
        if not torch.equal(ids[order], torch.arange(len(ids))):
            raise ValueError(
                f"{path} item_ids are not a permutation of 0..N-1; cannot align "
                "them to dense item ids."
            )
        return emb[order]
    return torch.as_tensor(obj)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--codebook_ckpt", required=True)
    p.add_argument("--embedding_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--n_levels",
        type=int,
        default=None,
        help="semantic levels (default: all in the checkpoint). The dedup digit "
        "is appended on top, so the final id length is n_levels + 1.",
    )
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="required to replace an existing tensor",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")

    front = load_rq_quantizer(args.codebook_ckpt, n_levels=args.n_levels)
    z = _load_embeddings(args.embedding_path)
    log.info(
        "[assign] quantizer=%s levels=%d items=%d dim=%d",
        front.quantizer,
        len(front.centroids),
        z.shape[0],
        z.shape[1],
    )

    if front.project is not None:
        lat = torch.cat(
            [
                front.project(z[i : i + args.batch_size]).cpu()
                for i in range(0, len(z), args.batch_size)
            ]
        )
    else:
        lat = z

    # float64 argmin, matching tracer_tokenizer.assignment_logits.
    r = lat.double()
    if front.normalize_inputs:
        r = torch.nn.functional.normalize(r, dim=-1)
    codes = []
    for c in front.centroids:
        cd = c.double()
        s = torch.cdist(r, cd).pow(2).argmin(dim=-1)
        codes.append(s)
        r = r - cd[s]
        if front.normalize_residuals:
            r = torch.nn.functional.normalize(r, dim=-1)

    tensor = torch.stack(codes, dim=-1).to(torch.int64)          # [N, L]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(tensor.cpu(), args.out)
    # Append the dedup digit, then transpose to the [L+1, N] layout.
    deduplicate_rows_in_tensor(file_path=args.out)
    transpose_tensor_from_file(file_path=args.out)

    final = torch.load(args.out, map_location="cpu", weights_only=False)
    per_level_max = [int(final[i].max()) for i in range(final.shape[0])]
    log.info("[assign] wrote %s shape=%s", args.out, tuple(final.shape))
    log.info("[assign] per-level max code: %s", per_level_max)
    if max(per_level_max) >= 256:
        raise SystemExit(
            f"max code {max(per_level_max)} >= vocab_size 256; this identifier "
            "space cannot be trained. Rebuild with one more RQ level."
        )

    # Check that phi=0 reproduces the written ids.
    agree = assert_reproduces_sids(
        z,
        front.centroids,
        final,
        project=front.project,
        normalize_inputs=front.normalize_inputs,
        normalize_residuals=front.normalize_residuals,
    )
    log.info("[assign] phi=0 reproduces %s", [round(a * 100, 4) for a in agree])


if __name__ == "__main__":
    main()
