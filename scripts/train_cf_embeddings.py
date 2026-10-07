"""Train a compact SASRec and save its item embedding table as a CF signal.

LETTER (arXiv:2405.07314) aligns its tokenizer's quantized latents with item
embeddings from a sequential CF model (a 32-d SASRec in the reference
implementation). This script produces that table: a ``[N, d]`` float32 tensor
whose row ``i`` is item ``i``, using the same item indexing as the semantic-ID
tensor. The CF signal is computed from the interactions, so train one tensor per
dataset variant (for example clean or poisoned) and keep the variant in the
filename.

Usage::

    python -m scripts.train_cf_embeddings \\
        --data-dir src/data/amazon_data/beauty \\
        --out embeddings/beauty_cf32.pt \\
        --dim 32 --epochs 200
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

log = logging.getLogger("cf_embeddings")

SEQUENCE_FIELD = "sequence_data"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _list_shards(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if not f.startswith(".")
    )


def read_sequences(data_dir: str, splits: Tuple[str, ...]) -> Tuple[List[List[int]], int]:
    """Read ``sequence_data`` from the given splits of a GRID TFRecord dataset.

    Returns ``(sequences, max_item_id)``. Mirrors the parsing in
    ``src/data/poisoning/bandwagon.py`` (VarLen features, GZIP shards).
    """
    import tensorflow as tf  # heavy; only needed here

    shards: List[str] = []
    for split in splits:
        shards.extend(_list_shards(os.path.join(data_dir, split)))
    if not shards:
        raise FileNotFoundError(
            f"No TFRecord shards under {data_dir} for splits {splits}"
        )

    raw = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    sample = next(iter(raw))
    example = tf.train.Example()
    example.ParseFromString(sample.numpy())
    feature_description = {
        name: tf.io.VarLenFeature(
            tf.int64
            if feat.HasField("int64_list")
            else (tf.float32 if feat.HasField("float_list") else tf.string)
        )
        for name, feat in example.features.feature.items()
    }
    if SEQUENCE_FIELD not in feature_description:
        raise ValueError(
            f"{data_dir} shards have no {SEQUENCE_FIELD!r} feature; "
            f"got {sorted(feature_description)}"
        )

    parsed = raw.map(lambda x: tf.io.parse_single_example(x, feature_description))
    sequences: List[List[int]] = []
    max_item = -1
    for ex in parsed:
        seq = tf.sparse.to_dense(ex[SEQUENCE_FIELD]).numpy().flatten()
        if seq.size < 2:
            # A length-1 sequence yields no (context -> next) pair.
            if seq.size == 1:
                max_item = max(max_item, int(seq[0]))
            continue
        items = [int(x) for x in seq.tolist()]
        max_item = max(max_item, max(items))
        sequences.append(items)
    if not sequences:
        raise ValueError(f"No usable (length >= 2) sequences found under {data_dir}")
    return sequences, max_item


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class SASRec(nn.Module):
    """Minimal SASRec (Kang & McAuley, ICDM 2018).

    Item id ``i`` is stored at embedding row ``i + 1``; row 0 is the padding
    slot, kept at zero and masked out of attention and the loss.
    """

    def __init__(
        self,
        num_items: int,
        dim: int = 32,
        max_len: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.dim = dim
        self.max_len = max_len
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_blocks)
        )
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """``seq`` is ``[B, T]`` of *shifted* ids (0 = pad). Returns ``[B, T, d]``."""
        b, t = seq.shape
        x = self.item_emb(seq) * math.sqrt(self.dim)
        x = x + self.pos_emb(torch.arange(t, device=seq.device)).unsqueeze(0)
        x = self.dropout(x)
        pad_mask = seq == 0
        causal = torch.triu(
            torch.ones(t, t, device=seq.device, dtype=torch.bool), diagonal=1
        )
        for block in self.blocks:
            x = block(x, src_mask=causal, src_key_padding_mask=pad_mask)
            # A fully-masked row (all-pad prefix) can emit NaN; pin it to zero.
            x = torch.nan_to_num(x)
        return self.norm(x)

    def item_vectors(self) -> torch.Tensor:
        """``[num_items, dim]``, row ``i`` = item ``i`` (padding row dropped)."""
        return self.item_emb.weight[1:].detach().clone()


def make_windows(
    sequences: List[List[int]], max_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Right-aligned (input, target) windows, ids shifted by +1 (0 = pad).

    Target ``j`` is the item that followed input ``j``; a zero target is ignored
    by the loss.
    """
    n = len(sequences)
    inputs = np.zeros((n, max_len), dtype=np.int64)
    targets = np.zeros((n, max_len), dtype=np.int64)
    for row, seq in enumerate(sequences):
        shifted = [i + 1 for i in seq]
        src, tgt = shifted[:-1], shifted[1:]
        src, tgt = src[-max_len:], tgt[-max_len:]
        inputs[row, max_len - len(src) :] = src
        targets[row, max_len - len(tgt) :] = tgt
    return inputs, targets


def train_sasrec(
    sequences: List[List[int]],
    num_items: int,
    dim: int,
    max_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    num_blocks: int,
    num_heads: int,
    dropout: float,
    device: torch.device,
    seed: int,
    log_every: int,
    val_sequences: Optional[List[List[int]]] = None,
    eval_every: int = 0,
    patience: int = 0,
) -> SASRec:
    torch.manual_seed(seed)
    np.random.seed(seed)

    inputs, targets = make_windows(sequences, max_len)
    inputs_t = torch.from_numpy(inputs)
    targets_t = torch.from_numpy(targets)
    n = inputs_t.size(0)

    model = SASRec(
        num_items=num_items,
        dim=dim,
        max_len=max_len,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    generator = torch.Generator().manual_seed(seed)

    # Validation-based model selection (enabled when val_sequences is given).
    best_score, best_state, since_best = -1.0, None, 0

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator)
        total, batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            src = inputs_t[idx].to(device)
            tgt = targets_t[idx].to(device)
            hidden = model.encode(src)                       # [B, T, d]
            valid = tgt > 0
            if not bool(valid.any()):
                continue
            h = hidden[valid]                                # [M, d]
            pos = tgt[valid]                                 # [M]
            # One uniform negative per position, resampled every step.
            neg = torch.randint(
                1, num_items + 1, pos.shape, device=device, generator=None
            )
            pos_e = model.item_emb(pos)
            neg_e = model.item_emb(neg)
            pos_logit = (h * pos_e).sum(-1)
            neg_logit = (h * neg_e).sum(-1)
            loss = (
                nn.functional.softplus(-pos_logit).mean()
                + nn.functional.softplus(neg_logit).mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():   # keep the pad row exactly zero
                model.item_emb.weight[0].zero_()
            total += float(loss.item())
            batches += 1
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            log.info(
                "epoch %4d/%d  loss %.4f", epoch, epochs, total / max(batches, 1)
            )

        if val_sequences and eval_every and (epoch + 1) % eval_every == 0:
            m = evaluate(model, val_sequences, max_len=max_len, device=device)
            score = m["ndcg@10"]
            model.train()
            if score > best_score:
                best_score, since_best = score, 0
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                log.info(
                    "epoch %4d  val N@10 %.4f  R@10 %.4f  <- best",
                    epoch, score, m["recall@10"],
                )
            else:
                since_best += 1
                log.info(
                    "epoch %4d  val N@10 %.4f  (no gain, %d/%d)",
                    epoch, score, since_best, patience,
                )
                if patience and since_best >= patience:
                    log.info("early stop at epoch %d; best val N@10 %.4f",
                             epoch, best_score)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("restored the best-by-val checkpoint (val N@10 %.4f)", best_score)
    return model



# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: SASRec,
    sequences: List[List[int]],
    max_len: int,
    device: torch.device,
    batch_size: int = 256,
    top_k: Tuple[int, ...] = (5, 10),
) -> dict:
    """Leave-one-out next-item evaluation against the full catalog.

    For each sequence the context is ``s[:-1]`` and the target is ``s[-1]``.
    History items are not excluded from the ranking, matching the protocol of
    the generative recommenders.
    """
    model.eval()
    contexts = [s[:-1] for s in sequences if len(s) >= 2]
    targets = [s[-1] for s in sequences if len(s) >= 2]
    n = len(contexts)
    inputs = np.zeros((n, max_len), dtype=np.int64)
    for row, ctx in enumerate(contexts):
        shifted = [i + 1 for i in ctx][-max_len:]
        inputs[row, max_len - len(shifted):] = shifted
    inputs_t = torch.from_numpy(inputs)
    targets_t = torch.tensor(targets, dtype=torch.long)

    # Row 0 is the padding slot and is not a real item; drop it so column j of
    # the score matrix is item j.
    table = model.item_emb.weight[1:]                       # [N, d]
    ranks = torch.empty(n, dtype=torch.long)
    for start in range(0, n, batch_size):
        src = inputs_t[start:start + batch_size].to(device)
        h = model.encode(src)[:, -1]                        # [B, d], last position
        scores = h @ table.t()                              # [B, N]
        tgt = targets_t[start:start + batch_size].to(device)
        true_score = scores.gather(1, tgt.unsqueeze(-1))
        # rank = how many items score strictly higher than the true one
        ranks[start:start + batch_size] = (scores > true_score).sum(-1).cpu()

    out = {"n_users": n}
    r = ranks.double()
    for k in top_k:
        hit = (ranks < k)
        out[f"recall@{k}"] = float(hit.double().mean())
        # NDCG with a single relevant item: 1 / log2(rank + 2) when it is in top-k
        out[f"ndcg@{k}"] = float(
            torch.where(hit, 1.0 / torch.log2(r + 2.0), torch.zeros_like(r)).mean()
        )
    out["mrr"] = float((1.0 / (r + 1.0)).mean())
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="GRID dataset root")
    p.add_argument(
        "--splits",
        default="training",
        help="comma-separated splits to read (default: training only, so eval/test "
        "targets do not leak into the CF signal)",
    )
    p.add_argument("--out", required=True, help="output .pt path for the [N, d] tensor")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--val-split", default="",
                   help="split for model selection during training (e.g. "
                        "'evaluation'). Empty disables validation-based selection "
                        "and keeps the last epoch's weights.")
    p.add_argument("--eval-every", type=int, default=0,
                   help="run validation every N epochs (0 = never)")
    p.add_argument("--patience", type=int, default=0,
                   help="stop after this many validations without improvement "
                        "(0 = never early-stop)")
    p.add_argument(
        "--eval-split",
        default="testing",
        help="split to run leave-one-out evaluation on after training "
        "(empty string disables it). Full-catalog ranking, TIGER/LETTER protocol.",
    )
    p.add_argument(
        "--num-items",
        type=int,
        default=None,
        help="catalog size N. Default: max observed item id + 1. Pass it "
        "explicitly (e.g. the semantic-ID tensor's N) when the tail of the "
        "catalog never appears in training, or the output will be too short.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    t0 = time.time()
    sequences, max_item = read_sequences(args.data_dir, splits)
    num_items = args.num_items if args.num_items is not None else max_item + 1
    if num_items <= max_item:
        raise ValueError(
            f"--num-items {num_items} is smaller than the largest observed item "
            f"id {max_item}"
        )
    log.info(
        "read %d sequences from %s (%s); catalog N=%d (max id %d) in %.1fs",
        len(sequences),
        args.data_dir,
        ",".join(splits),
        num_items,
        max_item,
        time.time() - t0,
    )

    device = torch.device(args.device)
    val_seqs = None
    if args.val_split:
        val_seqs, _ = read_sequences(args.data_dir, (args.val_split,))
        log.info("validation on %s (%d sequences), every %d epochs, patience %d",
                 args.val_split, len(val_seqs), args.eval_every, args.patience)
    model = train_sasrec(
        sequences=sequences,
        num_items=num_items,
        dim=args.dim,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        device=device,
        seed=args.seed,
        log_every=args.log_every,
        val_sequences=val_seqs,
        eval_every=args.eval_every,
        patience=args.patience,
    )

    if args.eval_split:
        eval_seqs, _ = read_sequences(args.data_dir, (args.eval_split,))
        metrics = evaluate(
            model, eval_seqs, max_len=args.max_len, device=device
        )
        log.info(
            "SASRec %s: R@5 %.4f  N@5 %.4f  R@10 %.4f  N@10 %.4f  MRR %.4f  (n=%d, "
            "full-catalog ranking, history not excluded)",
            args.eval_split,
            metrics["recall@5"], metrics["ndcg@5"],
            metrics["recall@10"], metrics["ndcg@10"],
            metrics["mrr"], metrics["n_users"],
        )
        import json
        side = os.path.splitext(args.out)[0] + "_metrics.json"
        with open(side, "w") as fh:
            json.dump(metrics, fh, indent=2)
        log.info("wrote %s", side)

    vectors = model.item_vectors().cpu().float()
    assert vectors.shape == (num_items, args.dim), vectors.shape
    n_dead = int((vectors.norm(dim=-1) == 0).sum())
    if n_dead:
        # Unseen items keep their random init, so zero rows indicate a problem.
        log.warning("%d/%d CF rows have zero norm", n_dead, num_items)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(vectors, args.out)
    log.info(
        "wrote %s shape=%s mean_norm=%.4f (%.1fs total)",
        args.out,
        tuple(vectors.shape),
        float(vectors.norm(dim=-1).mean()),
        time.time() - t0,
    )


if __name__ == "__main__":
    main()
