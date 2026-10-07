"""Lightning callback that saves and restores RNG states in checkpoints.

Stores the python, numpy, torch and per-device CUDA RNG states in every
checkpoint and restores them when a checkpoint is loaded for resume. Under DDP
all ranks restore rank 0's saved state. The streaming dataloader cannot seek, so
a resumed run is reproducible but not identical to an uninterrupted one. CUDA
states are restored only for the devices present at load time.
"""

import logging
import random
from typing import Any, Dict

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer

log = logging.getLogger(__name__)

_KEY = "rng_states"


class RngStateCallback(Callback):
    """Persist python/numpy/torch/CUDA RNG states in checkpoints for resume."""

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        states: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            states["cuda"] = torch.cuda.get_rng_state_all()
        checkpoint[_KEY] = states

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        states = checkpoint.get(_KEY)
        if not states:
            log.info(
                "[RngStateCallback] checkpoint has no saved RNG states "
                "; resuming with fresh seeding."
            )
            return
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"])
        cuda_states = states.get("cuda")
        if cuda_states and torch.cuda.is_available():
            n = min(len(cuda_states), torch.cuda.device_count())
            for i in range(n):
                torch.cuda.set_rng_state(cuda_states[i], device=i)
            if len(cuda_states) != torch.cuda.device_count():
                log.warning(
                    "[RngStateCallback] CUDA device count changed "
                    "(saved %d, present %d); restored the first %d.",
                    len(cuda_states),
                    torch.cuda.device_count(),
                    n,
                )
        log.info("[RngStateCallback] RNG states restored from checkpoint.")
