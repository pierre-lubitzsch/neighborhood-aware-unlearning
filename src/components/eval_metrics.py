import json
import os
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.metric import Metric
from torchmetrics.utilities.distributed import gather_all_tensors

## Custom Metrics


class CustomMeanReductionMetric(torchmetrics.Metric):
    """
    Custom metric class that uses mean reduction and supports distributed training.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.metric_values = 0
        self.total_values = 0

    def compute(self) -> torch.Tensor:
        # Aggregates the metric accross workers and returns the final value
        metric_values_tensor = torch.tensor(self.metric_values).to(self.device)
        total_values_tensor = torch.tensor(self.total_values).to(self.device)
        # Compute final metric
        if self.total_values == 0:
            return torch.tensor(0.0, device=self.device)
        # Checks if using more than one GPU
        # If so, gather all metric values and total values from all GPUs. Else, return the current
        # worker's metric value
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Gather all metric values and total values from all GPUs

            metric_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(metric_values_tensor)
            ]
            metric_values_tensor = torch.cat(metric_values_tensor_list).sum()

            total_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(total_values_tensor)
            ]

            total_values_tensor = torch.cat(total_values_tensor_list).sum()

        return metric_values_tensor / total_values_tensor

    def reset(self) -> None:
        self.metric_values = 0
        self.total_values = 0

    def update(self) -> None:
        raise NotImplementedError


class CustomRetrievalMetric(CustomMeanReductionMetric):
    """
    Custom retrieval metric class to calculate ranking metrics.
    """

    def __init__(
        self,
        top_k: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, **kwargs
    ) -> None:

        batch_size = int(len(indexes) / (indexes == 0).sum().item())
        preds = preds.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1).int()

        metric = self._metric(preds, target)
        self.metric_values += metric.sum().item()
        self.total_values += batch_size

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NDCG(CustomRetrievalMetric):
    """
    Metric to calculate Normalized Discounted Cumulative Gain@K (NDCG@K).
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        # Compute DCG
        dcg = torch.sum(
            topk_true
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Compute IDCG
        ideal_indices = torch.topk(target, self.top_k)[1]
        ideal_dcg = torch.sum(
            target.gather(1, ideal_indices)
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Handle cases where IDCG is zero
        ndcg = dcg / torch.where(ideal_dcg == 0, torch.ones_like(ideal_dcg), ideal_dcg)
        return ndcg


class Recall(CustomRetrievalMetric):
    """
    Metric to calculate Recall@K.
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        true_positives = topk_true.sum(dim=1)
        total_relevant = target.sum(dim=1)

        recall = true_positives / total_relevant.minimum(
            torch.tensor(self.top_k, device=self.device)
        ).clamp(
            min=1
        )  # Use clamp to avoid zero
        return recall


## Spam-exposure metrics
#
# These measure how often the spam-boosted target items I_t leak into the top-k
# recommendations, restricted to evaluation examples whose ground-truth label is
# not itself a target (y not in I_t). With one eval example per user
# (leave-one-out):
#   SH@k  = (# non-target examples with a target in top-k) / (# non-target examples)
#   ASI@k = (sum over non-target examples of |top-k ∩ I_t| / min(|I_t|, k)) / |U|
# where |U| is the total number of users/examples (target-labeled examples
# contribute 0 to the numerator but count in ASI's denominator).


class CustomSpamMetric(CustomMeanReductionMetric):
    """Base for spam-exposure metrics that need per-candidate target membership.

    Unlike :class:`CustomRetrievalMetric` (which only sees relevance of the true
    label), spam metrics consume, per example: ``preds`` (B, C) candidate scores,
    ``is_spam_cand`` (B, C) whether each generated candidate is a target item, and
    ``is_spam_label`` (B,) whether the ground-truth label is itself a target.
    """

    def __init__(
        self,
        top_k: int,
        num_targets: int = 1,
        forget_user_ids: Optional[Sequence[int]] = None,
        user_scope: str = "all",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k
        self.num_targets = max(1, int(num_targets))
        # User scope: "all", "forget" (requesting users only) or "retain"
        # (all other users). Separates deletion efficacy from collateral removal.
        if user_scope not in ("all", "forget", "retain"):
            raise ValueError(
                f"user_scope must be one of all/forget/retain, got {user_scope!r}"
            )
        self.user_scope = user_scope
        if user_scope != "all" and not forget_user_ids:
            raise ValueError(
                f"user_scope={user_scope!r} requires a non-empty forget_user_ids"
            )
        self.register_buffer(
            "forget_user_ids",
            torch.as_tensor(sorted(set(int(u) for u in (forget_user_ids or []))),
                            dtype=torch.long),
            persistent=False,
        )

    def _scope_mask(
        self, user_ids: Optional[torch.Tensor], batch_size: int, device
    ) -> torch.Tensor:
        """(B,) bool: which examples this metric's user scope counts."""
        if self.user_scope == "all":
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if user_ids is None:
            raise ValueError(
                f"user_scope={self.user_scope!r} needs per-example user_ids, but "
                "the evaluator passed none. Check that the data config keeps "
                "user_id (keep_user_id: true) and that eval_step forwards it."
            )
        uid = user_ids.reshape(-1).to(device)
        in_forget = torch.isin(uid, self.forget_user_ids.to(device))
        return in_forget if self.user_scope == "forget" else ~in_forget

    def update(
        self,
        preds: torch.Tensor,
        is_spam_cand: torch.Tensor,
        is_spam_label: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> None:
        k = min(self.top_k, preds.size(1))
        topk_idx = torch.topk(preds, k, dim=1)[1]
        spam_in_topk = is_spam_cand.gather(1, topk_idx)  # (B, k) bool
        keep = self._scope_mask(user_ids, preds.size(0), preds.device)
        self._accumulate(spam_in_topk, is_spam_label.bool(), keep)

    def _accumulate(
        self,
        spam_in_topk: torch.Tensor,
        is_spam_label: torch.Tensor,
        keep: torch.Tensor,
    ) -> None:
        raise NotImplementedError


class SpamHitRate(CustomSpamMetric):
    """SH@k: fraction of non-target eval examples whose top-k contains a target."""

    def _accumulate(
        self,
        spam_in_topk: torch.Tensor,
        is_spam_label: torch.Tensor,
        keep: torch.Tensor,
    ) -> None:
        # `keep` restricts both numerator and denominator.
        nonspam = (~is_spam_label) & keep
        hits = spam_in_topk.any(dim=1) & nonspam
        self.metric_values += hits.sum().item()
        self.total_values += nonspam.sum().item()  # denom: # non-target examples


class AvgSpamItems(CustomSpamMetric):
    """ASI@k: mean over users of |top-k ∩ I_t| / min(|I_t|, k) on non-target examples.

    Numerator sums ``|top-k ∩ I_t| / min(|I_t|, k)`` over non-target examples;
    denominator is the total example count |U| (target-labeled examples add 0 to
    the numerator but are counted here).
    """

    def _accumulate(
        self,
        spam_in_topk: torch.Tensor,
        is_spam_label: torch.Tensor,
        keep: torch.Tensor,
    ) -> None:
        nonspam = (~is_spam_label) & keep
        cap = max(1, min(self.num_targets, self.top_k))
        per_example = spam_in_topk.sum(dim=1).float() / cap
        per_example = per_example * nonspam.float()  # zero out target-labeled
        self.metric_values += per_example.sum().item()
        # denom: |U| within scope (target-labeled examples still counted)
        self.total_values += int(keep.sum().item())


class TargetProbMass(CustomSpamMetric):
    """TPM: mean over queries of the generation probability mass on targets I_t.

        TPM = (1/|Q|) * Σ_q  Σ_{i∈I_t} p(s_i | q)

    For each query it sums the *linear* marginal sequence probability of every
    generated candidate whose semantic ID is a target item, then averages over
    all queries. ``marginal_probs`` is already a linear probability in [0, 1]
    (the beam's product-of-per-hierarchy-softmax score in
    ``tiger_generation_model.generate``), so no exponentiation is needed.

    Unlike SH@k / ASI@k this is not top-k truncated and not restricted to
    non-target-labeled examples. ``top_k`` is accepted for interface
    compatibility but ignored, so TPM@k is the same for every k.
    """

    def update(
        self,
        preds: torch.Tensor,
        is_spam_cand: torch.Tensor,
        is_spam_label: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> None:
        # preds: (B, C) linear marginal prob of each generated candidate.
        # is_spam_cand: (B, C) bool, candidate's full SID matches a target item.
        # is_spam_label is unused (TPM is over all queries q ∈ Q in scope).
        keep = self._scope_mask(user_ids, preds.size(0), preds.device)
        mass = (preds * is_spam_cand.to(preds.dtype)).sum(dim=1)  # (B,)
        self.metric_values += float((mass * keep.to(mass.dtype)).sum().item())
        self.total_values += int(keep.sum().item())  # denom: |Q| in scope


## Evaluators

class Evaluator:
    def __init__(self, metrics: Dict[str, Metric], *args, **kwargs):
        self.metrics = metrics

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def to(self, device: torch.device):
        for metric in self.metrics.values():
            metric.to(device=device)


class RetrievalEvaluator(Evaluator):
    """
    Wrapper for retrieval evaluation metrics.
    It takes model outputs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
        should_sample_negatives_from_vocab: bool = True,
        num_negatives: int = 500,
        placeholder_token_buffer: int = 100,
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }
        self.should_sample_negatives_from_vocab = should_sample_negatives_from_vocab
        self.num_negatives = num_negatives
        self.placeholder_token_buffer = placeholder_token_buffer

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        num_of_samples = query_embeddings.shape[0]
        num_of_candidates = key_embeddings.shape[0]

        if self.should_sample_negatives_from_vocab:
            inbatch_negatives = self.sample_negative_ids_from_vocab(
                num_of_samples=num_of_samples,
                num_of_candidates=num_of_candidates,
                num_negatives=self.num_negatives,
            )
            # +1 to include the positive sample
            num_of_candidates = self.num_negatives + 1
            pos_embeddings = key_embeddings[labels]
            key_embeddings = key_embeddings[inbatch_negatives]
            # key_embeddings shape: (bsz, num_negatives+1, emb_dim)
            key_embeddings = torch.cat(
                [pos_embeddings.unsqueeze(1), key_embeddings], dim=1
            )
            # the positive index will always be 0 because the pos embedding will always be the first one.
            labels = torch.zeros(num_of_samples).long()

        # following examples from https://lightning.ai/docs/torchmetrics/stable/retrieval/precision.html
        # indexes refers to the mask of the labels
        indexes = torch.arange(0, query_embeddings.shape[0])
        expanded_indexes = (
            indexes.unsqueeze(-1).expand(num_of_samples, num_of_candidates).reshape(-1)
        )

        if self.should_sample_negatives_from_vocab:
            preds = (
                torch.mul(
                    query_embeddings.unsqueeze(1).expand_as(key_embeddings),
                    key_embeddings,
                )
                .sum(-1)
                .reshape(-1)
            )
        else:
            preds = torch.mm(query_embeddings, key_embeddings.t()).reshape(-1)

        target = torch.zeros(num_of_samples, num_of_candidates).bool()
        target[torch.arange(num_of_samples), labels] = True
        target = target.reshape(-1)

        for _, metric_object in self.metrics.items():
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )

    # this method samples random negative samples from the whole vocab
    def sample_negative_ids_from_vocab(
        self,
        num_of_samples: int,
        num_of_candidates: int,
        num_negatives: int,
    ) -> torch.Tensor:
        # num_of_samples: batch size
        # num_of_candidates: number of total vocabs
        # num_negatives: number of negative samples

        # randint is used for speed; collisions with positives are possible but rare
        negative_candidates = torch.randint(
            self.placeholder_token_buffer,
            num_of_candidates,
            (num_of_samples, num_negatives),
        )

        return negative_candidates


class SIDRetrievalEvaluator(Evaluator):
    """
    Wrapper for retrieval evaluation metrics for semantic IDs.
    It takes model outputs in semantic IDs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
        spam_metrics: Optional[Dict[str, CustomSpamMetric]] = None,
        forget_manifest_path: Optional[str] = None,
        semantic_id_path: Optional[str] = None,
        num_hierarchies: Optional[int] = None,
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }

        # Optional spam-exposure metrics (SH@k, ASI@k). They require the spam
        # target set I_t as semantic IDs, loaded from the poison forget_manifest
        # and the semantic-ID tensor. If either is missing they are skipped.
        self.spam_target_sids: Optional[torch.Tensor] = None
        self.num_spam_targets: int = 0
        # Users who requested the deletion (sensitive manifests only; empty for
        # spam manifests).
        self.forget_user_ids: List[int] = self._load_forget_user_ids(
            forget_manifest_path
        )
        if spam_metrics:
            self.spam_target_sids, self.num_spam_targets = self._load_spam_target_sids(
                forget_manifest_path, semantic_id_path, num_hierarchies
            )
            if self.spam_target_sids is not None:
                for metric_name, metric_object in spam_metrics.items():
                    for top_k in top_k_list:
                        self.metrics[f"{metric_name}@{top_k}"] = metric_object(
                            top_k=top_k,
                            num_targets=self.num_spam_targets,
                            sync_on_compute=False,
                            compute_with_cache=False,
                        )
                        # Scoped variants, registered only when the manifest
                        # names the requesting users.
                        if self.forget_user_ids:
                            for scope, suffix in (("forget", "F"), ("retain", "R")):
                                self.metrics[f"{metric_name}{suffix}@{top_k}"] = (
                                    metric_object(
                                        top_k=top_k,
                                        num_targets=self.num_spam_targets,
                                        forget_user_ids=self.forget_user_ids,
                                        user_scope=scope,
                                        sync_on_compute=False,
                                        compute_with_cache=False,
                                    )
                                )
                if self.forget_user_ids:
                    print(
                        "[SIDRetrievalEvaluator] scoped spam metrics enabled: "
                        f"{len(self.forget_user_ids)} forget users; emitting "
                        "<name>F@k (requesting users) and <name>R@k (everyone "
                        "else) alongside the global <name>@k."
                    )
            else:
                print(
                    "[SIDRetrievalEvaluator] spam_metrics requested but no spam "
                    f"targets loaded (forget_manifest_path={forget_manifest_path!r}, "
                    f"semantic_id_path={semantic_id_path!r}); SH/ASI skipped."
                )

    @staticmethod
    def _load_forget_user_ids(forget_manifest_path: Optional[str]) -> List[int]:
        """User ids named by the manifest as having requested the deletion.

        Returns [] when the manifest is missing or is not a sensitive-deletion
        manifest, which disables the scoped metrics.
        """
        if not forget_manifest_path or not os.path.isfile(forget_manifest_path):
            return []
        try:
            with open(forget_manifest_path, encoding="utf-8") as f:
                man = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
        # `spam_user_ids` is the manifest's spelling for both scenarios; for a
        # sensitive deletion these are the real users who asked.
        if man.get("scenario") != "sensitive":
            return []
        raw = man.get("spam_user_ids") or []
        if not isinstance(raw, list):
            return []
        return sorted({int(u) for u in raw})

    @staticmethod
    def _load_spam_target_sids(
        forget_manifest_path: Optional[str],
        semantic_id_path: Optional[str],
        num_hierarchies: Optional[int],
    ):
        """Return (target_sids [T, H] long, num_targets) or (None, 0).

        ``forget_manifest_path`` provides ``target_items`` (I_t); the semantic-ID
        tensor (``[num_hierarchies, num_items]``, raw per-hierarchy codes) maps
        each target item to its code tuple (the same space as ``generated_ids``).
        """
        if not forget_manifest_path or not os.path.isfile(forget_manifest_path):
            return None, 0
        if not semantic_id_path or not os.path.isfile(semantic_id_path):
            return None, 0
        with open(forget_manifest_path, encoding="utf-8") as f:
            targets = json.load(f).get("target_items", [])
        if not targets:
            return None, 0
        sem = torch.load(semantic_id_path, map_location="cpu", weights_only=False)
        if not torch.is_tensor(sem) or sem.ndim != 2:
            return None, 0
        n_items = sem.shape[1]
        idx = torch.tensor([t for t in targets if 0 <= int(t) < n_items], dtype=torch.long)
        if idx.numel() == 0:
            return None, 0
        h = int(num_hierarchies) if num_hierarchies else sem.shape[0]
        h = min(h, sem.shape[0])
        target_sids = sem[:h, idx].t().contiguous().long()  # (T, H)
        return target_sids, int(target_sids.shape[0])

    def __call__(
        self,
        marginal_probs: torch.Tensor,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, num_candidates, num_hierarchies = generated_ids.shape
        labels = labels.reshape(batch_size, 1, num_hierarchies)
        preds = marginal_probs.reshape(-1)

        # check if the generated IDs contain the labels
        # if so, get the coordinates of the matched IDs
        matched_id_coord = torch.all((generated_ids == labels), dim=2).nonzero()

        # initialize the ground truth as all false
        target = torch.zeros(batch_size, num_candidates).bool()

        # set the matched IDs to true if they are in the generated IDs
        target[matched_id_coord[:, 0], matched_id_coord[:, 1]] = True
        target = target.reshape(-1)
        expanded_indexes = (
            torch.arange(batch_size)
            .unsqueeze(-1)
            .expand(batch_size, num_candidates)
            .reshape(-1)
        )

        for _, metric_object in self.metrics.items():
            if isinstance(metric_object, CustomSpamMetric):
                continue
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )

        # Spam-exposure metrics: detect target items among the generated
        # candidates and among the labels (to exclude target-labeled examples).
        if self.spam_target_sids is not None:
            st = self.spam_target_sids.to(generated_ids.device)  # (T, H)
            # candidate is a target iff its full code tuple matches any target sid
            is_spam_cand = (
                (generated_ids.unsqueeze(2) == st.view(1, 1, -1, num_hierarchies))
                .all(dim=3)
                .any(dim=2)
            )  # (B, C) bool
            lbl = labels.reshape(batch_size, num_hierarchies)
            is_spam_label = (
                (lbl.unsqueeze(1) == st.unsqueeze(0)).all(dim=2).any(dim=1)
            )  # (B,) bool
            preds_2d = marginal_probs.reshape(batch_size, num_candidates)
            uid = None if user_ids is None else user_ids.reshape(batch_size)
            for _, metric_object in self.metrics.items():
                if isinstance(metric_object, CustomSpamMetric):
                    metric_object.update(
                        preds_2d,
                        is_spam_cand.to(preds.device),
                        is_spam_label.to(preds.device),
                        user_ids=uid,
                    )
