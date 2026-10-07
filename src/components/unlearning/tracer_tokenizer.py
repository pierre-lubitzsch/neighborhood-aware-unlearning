"""Tokenizer side of TRACER token reassignment (arXiv:2606.07688).

TRACER unlearns a concept by reassigning its items to different codewords rather
than by suppressing their logits. The reassignment is made differentiable with a
per-item, per-level, per-codeword score ``phi`` that perturbs the quantizer's
distances:

    (Eq. 5)  q(s_i^l = k)      = softmax_k( -||r_i^l - c_k^l||^2 / tau )
    (Eq. 6)  q_phi(s_i^l = k)  = softmax_k( (-||r_i^l - c_k^l||^2 + phi_{i,k}^l) / tau )
             L_reg             = sum_{i,l,k} |phi_{i,k}^l|

with soft token embedding  e~_i^l = sum_k q_phi(s_i^l = k) e_k^l .

This module covers everything that depends on the quantizer (residuals r_i^l and
codewords c_k^l); the training loop and loss terms live in ``tracer.py``.

With ``phi = 0`` the recursion must reproduce the stored semantic ids exactly.
For RQ-KMeans (input and residual normalization enabled) it is

    r_i^1     = normalize(z_i)
    s_i^l     = argmin_k ||r_i^l - c_k^l||^2
    r_i^(l+1) = normalize(r_i^l - c_{s_i^l}^l)

``assert_reproduces_sids`` checks this. ``load_rq_quantizer`` reads the codebooks
and the residual recipe from the checkpoint and detects the quantizer type
(RQ-KMeans in the raw embedding space, RQ-VAE in its encoder latent without
residual normalization). The centroids come from the quantizer's training
checkpoint, which must therefore be available.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


@dataclass
class RQFrontEnd:
    """Codebooks and residual recipe of a quantizer checkpoint.

    * ``rkmeans``: codebooks in the raw embedding space, no encoder, input and
      residual normalization enabled.
    * ``rqvae``: codebooks in the encoder latent space, reached through
      ``normalization_layer`` and ``encoder`` (``project``), without residual
      normalization.
    """

    centroids: List[torch.Tensor]
    quantizer: str
    normalize_inputs: bool = True
    normalize_residuals: bool = True
    project: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


def _torch_load(ckpt_path: str) -> dict:
    """Load a checkpoint on CPU, importing ``src.utils`` first to avoid a
    circular import when unpickling outside a Hydra run.
    """
    try:
        import src.utils  # noqa: F401
    except ImportError:  # pragma: no cover - best effort, load may still work
        pass
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def load_rq_centroids(
    ckpt_path: str,
    n_levels: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """Load ``[K, D]`` codeword tensors from an RQ-KMeans training checkpoint.

    Keys look like ``quantization_layer_list.<l>.centroids``. Returns them in
    level order. ``n_levels`` truncates (the semantic levels are ``H - 1``; the
    final id digit is a dedup counter with no codebook).
    """
    obj = _torch_load(ckpt_path)
    state = obj.get("state_dict", obj)
    prefix, suffix = "quantization_layer_list.", ".centroids"
    # Match only '<prefix><level><suffix>'; RQ-VAE checkpoints also store
    # k-means initializer centroids under a longer key.
    idx = sorted(
        int(mid)
        for k in state
        if k.startswith(prefix)
        and k.endswith(suffix)
        and (mid := k[len(prefix) : -len(suffix)]).isdigit()
    )
    if not idx:
        raise ValueError(f"no '{prefix}*{suffix}' entries in {ckpt_path}")
    if n_levels is not None:
        idx = idx[:n_levels]
    cents = [state[f"{prefix}{i}{suffix}"].float() for i in idx]
    if device is not None:
        cents = [c.to(device) for c in cents]
    log.info(
        "[tracer] loaded %d codebook levels from %s (K=%d, D=%d)",
        len(cents),
        ckpt_path,
        cents[0].shape[0],
        cents[0].shape[1],
    )
    return cents


def _has_frontend(state: dict) -> bool:
    """True when the checkpoint carries a learned normalization/encoder front end.

    RQ-KMeans leaves both as ``nn.Identity``, so these keys identify RQ-VAE.
    """
    return any(
        k.startswith("normalization_layer.") or k.startswith("encoder.") for k in state
    )


def load_rq_quantizer(
    ckpt_path: str,
    n_levels: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> RQFrontEnd:
    """Load the codebooks and the residual recipe from a quantizer checkpoint.

    Detects rkmeans vs rqvae from the checkpoint contents. For rqvae the
    ``normalization_layer`` and ``encoder`` are rebuilt from the run's
    ``.hydra/config.yaml`` (they are not in ``hyper_parameters``) and loaded
    with the checkpoint weights. The front end runs in ``eval()`` mode so that
    BatchNorm uses its running statistics.
    """
    obj = _torch_load(ckpt_path)
    state = obj.get("state_dict", obj)
    centroids = load_rq_centroids(ckpt_path, n_levels=n_levels, device=device)

    if not _has_frontend(state):
        log.info("[tracer] quantizer=rkmeans (no encoder in %s)", ckpt_path)
        return RQFrontEnd(
            centroids=centroids,
            quantizer="rkmeans",
            normalize_inputs=True,
            normalize_residuals=True,
            project=None,
        )

    import hydra
    from omegaconf import OmegaConf

    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.isfile(cfg_path):
        raise ValueError(
            f"{ckpt_path} has an encoder front end (quantizer=rqvae) but its "
            f"Hydra config is missing at {cfg_path}; the encoder cannot be "
            "rebuilt without it."
        )
    model_cfg = OmegaConf.load(cfg_path).model
    module = hydra.utils.instantiate(model_cfg)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if any(k.startswith(("normalization_layer.", "encoder.")) for k in missing):
        raise ValueError(
            f"{ckpt_path} is missing front-end weights after instantiate: "
            f"{[k for k in missing if k.startswith(('normalization_layer.', 'encoder.'))][:5]}"
        )
    module.eval()
    norm_layer, encoder = module.normalization_layer, module.encoder
    if device is not None:
        norm_layer = norm_layer.to(device)
        encoder = encoder.to(device)

    def project(x: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        # Runs in float32 and in chunks to bound memory; chunking is exact
        # because the front end is in eval mode.
        with torch.no_grad():
            p = next(encoder.parameters())
            if len(x) <= chunk:
                return encoder(norm_layer(x.to(device=p.device, dtype=p.dtype)))
            return torch.cat(
                [
                    encoder(norm_layer(x[i : i + chunk].to(device=p.device, dtype=p.dtype)))
                    for i in range(0, len(x), chunk)
                ]
            )

    normalize_residuals = bool(
        obj.get("hyper_parameters", {}).get("normalize_residuals", False)
    )
    log.info(
        "[tracer] quantizer=rqvae (encoder front end from %s, "
        "normalize_residuals=%s, codebook dim=%d)",
        cfg_path,
        normalize_residuals,
        centroids[0].shape[1],
    )
    return RQFrontEnd(
        centroids=centroids,
        quantizer="rqvae",
        # normalization_layer already ends in L2 normalization.
        normalize_inputs=False,
        normalize_residuals=normalize_residuals,
        project=project,
    )


def compute_residuals(
    z: torch.Tensor,
    centroids: Sequence[torch.Tensor],
    codes: torch.Tensor,
    project: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    normalize_inputs: bool = True,
    normalize_residuals: bool = True,
) -> List[torch.Tensor]:
    """Per-level residuals ``r_i^l`` under the stored assignment.

    ``z`` is ``[N, D]`` (pre-quantization item embeddings), ``codes`` is
    ``[L, N]`` (or ``[H, N]``; only the first ``len(centroids)`` rows are used).
    Returns ``L`` tensors of shape ``[N, D]``.

    The defaults are the RQ-KMeans recipe. ``project`` maps raw embeddings into
    the codebook space (needed for RQ-VAE); obtain all three settings from
    :func:`load_rq_quantizer`.
    """
    if project is not None:
        z = project(z)
    # float64 to avoid compounding error in the recursion.
    r = z.double()
    if normalize_inputs:
        r = F.normalize(r, dim=-1)
    out: List[torch.Tensor] = []
    for lvl, c in enumerate(centroids):
        out.append(r)
        s = codes[lvl].long().to(r.device)
        r = r - c.to(r.device).double()[s]
        if normalize_residuals:
            r = F.normalize(r, dim=-1)
    return out


def assignment_logits(
    residual: torch.Tensor,
    centroids: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pre-temperature scores ``-||r_i - c_k||^2 (+ phi_{i,k})`` of Eq. 6.

    ``residual`` ``[N, D]``, ``centroids`` ``[K, D]``, ``phi`` ``[N, K]`` or None.
    Returns ``[N, K]``. Temperature is applied by the caller so the same scores
    can drive both the soft assignment and the hard argmax.

    Distances are computed in float64 so that reduced float32 matmul precision
    does not flip the argmin for near-tied codewords.
    """
    r64, c64 = residual.double(), centroids.double()
    # Chunk rows for large catalogs to bound memory; the result is identical.
    if r64.shape[0] > 65536:
        d2 = torch.cat(
            [torch.cdist(r64[i : i + 65536], c64).pow(2) for i in range(0, r64.shape[0], 65536)]
        )
    else:
        d2 = torch.cdist(r64, c64).pow(2)
    scores = (-d2).to(residual.dtype)
    if phi is not None:
        scores = scores + phi
    return scores


def soft_assignment(
    residual: torch.Tensor,
    centroids: torch.Tensor,
    phi: Optional[torch.Tensor],
    tau: float,
) -> torch.Tensor:
    """Soft assignment ``q_phi(s_i = k)`` of Eq. 6, shape ``[N, K]``."""
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    return torch.softmax(assignment_logits(residual, centroids, phi) / float(tau), dim=-1)


def hard_assignment(
    residual: torch.Tensor,
    centroids: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hard assignment ``argmax_k q_phi``, shape ``[N]``; independent of ``tau``."""
    return assignment_logits(residual, centroids, phi).argmax(dim=-1)


def phi_regularizer(phis: Sequence[torch.Tensor]) -> torch.Tensor:
    """``L_reg = sum_{i,l,k} |phi_{i,k}^l|`` (Eq. 6)."""
    return sum(p.abs().sum() for p in phis)


def retain_code_usage(
    codes: torch.Tensor,
    retain_item_ids: torch.Tensor,
    n_codes: int,
    level: int,
) -> torch.Tensor:
    """``rho_l(k)`` of Eq. 11: the fraction of retain items using code ``k``.

    Returns ``[K]`` summing to 1 over used codes.
    """
    s = codes[level].long()[retain_item_ids.long()]
    counts = torch.bincount(s, minlength=n_codes).float()
    return counts / counts.sum().clamp(min=1.0)


def selective_update_mask(
    q_phi: torch.Tensor,
    rho: torch.Tensor,
    grad_phi_forget: torch.Tensor,
) -> torch.Tensor:
    """The selective-update mask ``M_{i,k}^l`` of Eq. 11.

        M = 1[ rho_l(k) > rho_bar_{i,l} ] * 1[ grad_{phi} L_F > 0 ]

    where ``rho_bar_{i,l} = E_{q_phi} rho_l(k)`` is the expected usage under the
    item's current soft assignment, so phi only moves on codewords more shared
    with the retain set than the current assignment.

    ``q_phi`` ``[N, K]``, ``rho`` ``[K]``, ``grad_phi_forget`` ``[N, K]``.
    """
    rho_bar = (q_phi * rho.unsqueeze(0)).sum(dim=-1, keepdim=True)   # [N, 1]
    overlap = (rho.unsqueeze(0) > rho_bar)                            # [N, K]
    conflicting = grad_phi_forget > 0
    return (overlap & conflicting).to(q_phi.dtype)


def assert_reproduces_sids(
    z: torch.Tensor,
    centroids: Sequence[torch.Tensor],
    codes: torch.Tensor,
    tol: float = 1.0,
    project: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    normalize_inputs: bool = True,
    normalize_residuals: bool = True,
) -> List[float]:
    """Check that ``phi = 0`` reproduces the stored semantic ids.

    Returns the per-level agreement fraction; raises if any level is below
    ``tol`` (default 1.0 == exact).
    """
    residuals = compute_residuals(
        z,
        centroids,
        codes,
        project=project,
        normalize_inputs=normalize_inputs,
        normalize_residuals=normalize_residuals,
    )
    agree: List[float] = []
    for lvl, (r, c) in enumerate(zip(residuals, centroids)):
        pred = hard_assignment(r, c.to(r.device))
        a = (pred == codes[lvl].long().to(pred.device)).float().mean().item()
        agree.append(a)
        log.info("[tracer] level %d: phi=0 reproduces %.2f%% of stored codes", lvl, a * 100)
    worst = min(agree)
    if worst < tol:
        n = int(codes[0].numel())
        miss = [int(round((1.0 - a) * n)) for a in agree]
        raise ValueError(
            f"phi=0 reproduces only {worst * 100:.2f}% of stored codes "
            f"({max(miss)} of {n} items mismatch at the worst level; per level "
            f"{miss} items = {[round(a * 100, 2) for a in agree]}%). "
            "A few mismatches are usually quantization ties; lower "
            "unlearning.tracer_sid_tolerance (e.g. 0.999) to accept them. A "
            "large fraction means the codebook checkpoint does not match "
            "semantic_id_path or the residual recipe is wrong."
        )
    return agree
