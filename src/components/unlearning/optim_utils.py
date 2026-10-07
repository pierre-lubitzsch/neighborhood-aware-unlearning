"""Shared optimizer factory for the first-order unlearning algorithms.

Supports ``adam``, ``adamw`` and ``sgd``; unknown names raise. ``scif`` does not
use it because it takes a conjugate-gradient step instead of an optimizer loop.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Union

import torch

# Default SGD momentum, since plain SGD is a weak substitute for Adam.
SGD_DEFAULT_MOMENTUM = 0.9

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

ParamsLike = Union[Iterable[torch.nn.Parameter], List[Dict[str, Any]]]


def available_optimizers() -> List[str]:
    return sorted(_OPTIMIZERS)


def build_optimizer(
    name: str,
    params: ParamsLike,
    lr: float,
    *,
    weight_decay: float = 0.0,
    momentum: float = SGD_DEFAULT_MOMENTUM,
    algo: str = "",
) -> torch.optim.Optimizer:
    """Build optimizer ``name`` over ``params`` (parameters or param groups).

    Only kwargs accepted by the chosen optimizer are passed.
    """
    key = str(name).strip().lower()
    if key not in _OPTIMIZERS:
        raise ValueError(
            f"{algo or 'unlearning'} optimizer must be one of "
            f"{available_optimizers()}, got {name!r}"
        )
    kwargs: Dict[str, Any] = {"lr": float(lr)}
    if float(weight_decay) != 0.0:
        kwargs["weight_decay"] = float(weight_decay)
    if key == "sgd":
        kwargs["momentum"] = float(momentum)
    return _OPTIMIZERS[key](params, **kwargs)
