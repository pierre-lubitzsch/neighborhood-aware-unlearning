"""Static code-sharing analysis for RQ semantic IDs.

Counts how many retained (non-target) catalog items share a target item's full
RQ code or a code prefix. Shared codes and prefixes are a structural route for
collateral forgetting during unlearning. Requires only the semantic-ID tensor and
the forget manifest's ``target_items``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

log = logging.getLogger(__name__)


def load_codebook_matrix(
    semantic_id_path: str,
    num_hierarchies: Optional[int] = None,
) -> torch.Tensor:
    """Load the semantic-ID tensor and return it as ``[num_items, H]`` (long).

    The on-disk ``merged_predictions_tensor.pt`` is laid out ``[H, num_items]``.
    It is transposed so each row is one item's code tuple, and optionally sliced
    to the first ``num_hierarchies`` codes.
    """
    obj = torch.load(semantic_id_path, map_location="cpu")
    if isinstance(obj, dict):
        # tolerate a few common wrappers
        for key in ("semantic_ids", "merged_predictions_tensor", "tensor"):
            if key in obj:
                obj = obj[key]
                break
    t = obj if isinstance(obj, torch.Tensor) else torch.as_tensor(obj)
    t = t.long()
    if t.dim() != 2:
        raise ValueError(
            f"semantic-id tensor must be 2-D, got shape {tuple(t.shape)}"
        )

    rows, cols = t.shape
    if num_hierarchies is not None and rows == num_hierarchies:
        codebook = t[:num_hierarchies].t().contiguous()  # [N, H]
    elif num_hierarchies is not None and cols == num_hierarchies:
        codebook = t[:, :num_hierarchies].contiguous()  # already [N, H]
    else:
        # Heuristic: the hierarchy dimension is the (much) smaller one.
        codebook = (t.t() if rows < cols else t).contiguous()
        if num_hierarchies is not None:
            codebook = codebook[:, :num_hierarchies].contiguous()
    return codebook


def code_sharing_report(
    semantic_id_path: str,
    target_items: Sequence[int],
    num_hierarchies: Optional[int] = None,
) -> Dict[str, Any]:
    """Count retained items sharing each target's code / code-prefixes.

    For each prefix length ``p`` in ``1..H`` (``p == H`` is the full code), and
    each target item, count the *retained* items (catalog items that are neither
    a target nor padding) whose first ``p`` codes equal the target's first ``p``
    codes.

    Returns per-prefix aggregates plus per-target detail.
    ``shared_retained_total_unique`` is the size of the union over all targets.
    """
    codebook = load_codebook_matrix(semantic_id_path, num_hierarchies)
    num_items, H = codebook.shape

    targets = sorted({int(t) for t in target_items if 0 <= int(t) < num_items})
    dropped = [int(t) for t in target_items if not (0 <= int(t) < num_items)]
    if dropped:
        log.warning(
            "[rq-diag] %d target item(s) out of range [0,%d) ignored: %s",
            len(dropped),
            num_items,
            dropped[:10],
        )

    is_target = torch.zeros(num_items, dtype=torch.bool)
    if targets:
        is_target[torch.tensor(targets)] = True
    retained_mask = ~is_target

    by_prefix: List[Dict[str, Any]] = []
    for p in range(1, H + 1):
        prefixes = codebook[:, :p]  # [N, p]
        shared_any = torch.zeros(num_items, dtype=torch.bool)
        per_target: List[Dict[str, Any]] = []
        for t in targets:
            tp = codebook[t, :p]
            match = (prefixes == tp.unsqueeze(0)).all(dim=1)  # [N]
            match_retained = match & retained_mask
            shared_any |= match_retained
            per_target.append(
                {
                    "target_item": t,
                    "shared_retained_items": int(match_retained.sum()),
                }
            )
        counts = [d["shared_retained_items"] for d in per_target]
        by_prefix.append(
            {
                "prefix_length": p,
                "is_full_code": (p == H),
                "shared_retained_total_unique": int(shared_any.sum()),
                "per_target_mean": (sum(counts) / len(counts)) if counts else 0.0,
                "per_target_max": (max(counts) if counts else 0),
                "per_target_min": (min(counts) if counts else 0),
                "per_target": per_target,
            }
        )

    report = {
        "num_items": int(num_items),
        "num_hierarchies": int(H),
        "n_target_items": len(targets),
        "target_items": targets,
        "by_prefix_length": by_prefix,
    }
    if by_prefix:
        full = by_prefix[-1]
        log.info(
            "[rq-diag] code sharing: full-code collisions (unique retained)=%d; "
            "prefix-1 shared (unique retained)=%d",
            full["shared_retained_total_unique"],
            by_prefix[0]["shared_retained_total_unique"],
        )
    return report
