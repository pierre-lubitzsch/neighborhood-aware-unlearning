"""Merge the pickle shards written by `src.inference` into a single tensor file.

    python -m scripts.merge_predictions embeddings <pickle_dir> <out.pt>
    python -m scripts.merge_predictions sids <pickle_dir> <out.pt>

`embeddings` writes {"embeddings": [N, D], "item_ids": [N]} sorted by item id.
`sids` writes the semantic-ID tensor used for training: codes are merged by item id,
a de-duplication digit is appended, and the result is transposed to [L, N].
"""
import os
import pickle
import sys

import torch


def _load_shards(pickle_dir, prefix=""):
    files = sorted(f for f in os.listdir(pickle_dir)
                   if f.endswith(".pkl") and f.startswith(prefix) and f != "merged_predictions.pkl")
    if not files:
        raise FileNotFoundError(f"no prediction shards in {pickle_dir}")
    rows = []
    for name in files:
        with open(os.path.join(pickle_dir, name), "rb") as fh:
            rows.extend(pickle.load(fh))
    return rows


def merge_embeddings(pickle_dir, out):
    rows = sorted(_load_shards(pickle_dir), key=lambda r: int(r["item_id"]))
    item_ids = torch.tensor([int(r["item_id"]) for r in rows], dtype=torch.int64)
    embeddings = torch.stack([torch.as_tensor(r["embedding"]) for r in rows]).float()
    torch.save({"embeddings": embeddings, "item_ids": item_ids}, out)
    print(f"wrote {out}: {tuple(embeddings.shape)}")


def merge_sids(pickle_dir, out):
    from src.utils.tensor_utils import (
        deduplicate_rows_in_tensor,
        merge_list_of_keyed_tensors_to_single_tensor,
        transpose_tensor_from_file,
    )
    rows = _load_shards(pickle_dir, prefix="predictions_")
    tensor = merge_list_of_keyed_tensors_to_single_tensor(data=rows, index_key="item_id", value_key="cluster_ids")
    torch.save(tensor.cpu(), out)
    deduplicate_rows_in_tensor(file_path=out)
    transpose_tensor_from_file(file_path=out)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("embeddings", "sids"):
        raise SystemExit(__doc__)
    os.makedirs(os.path.dirname(os.path.abspath(sys.argv[3])), exist_ok=True)
    (merge_embeddings if sys.argv[1] == "embeddings" else merge_sids)(sys.argv[2], sys.argv[3])
