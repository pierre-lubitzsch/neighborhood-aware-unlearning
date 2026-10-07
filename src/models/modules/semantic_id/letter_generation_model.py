"""LETTER generative recommender: the TIGER backbone with a temperature-scaled CE loss.

LETTER (arXiv:2405.07314) consists of a learnable tokenizer (see
:mod:`src.modules.clustering.letter_quantization`) and a generation loss. The
reference implementation computes ``CrossEntropy(lm_logits / temperature, labels)``
with a default temperature of 1.0, in which case the recommender is identical to
TIGER. The temperature is applied inside ``loss_function`` so that training and
the unlearning algorithms, which recompute the training loss, use the same
objective.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn

from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDEncoderDecoder,
)

log = logging.getLogger(__name__)


class TemperatureScaledLoss(nn.Module):
    """Wrap a logit-consuming loss so its logits are divided by ``temperature``.

    Implemented as a module so that it survives ``deepcopy`` and appears by name
    in model summaries.
    """

    def __init__(self, loss_function: nn.Module, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        self.loss_function = loss_function
        self.temperature = float(temperature)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return self.loss_function(input=input / self.temperature, target=target, **kwargs)

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"


class LetterEncoderDecoder(SemanticIDEncoderDecoder):
    """TIGER backbone + LETTER's tempered generation loss.

    Args:
        letter_temperature: logit temperature ``tau``. The default of 1.0 makes
            the loss identical to TIGER's.
    """

    def __init__(
        self,
        *args: Any,
        letter_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.letter_temperature = float(letter_temperature)
        if self.loss_function is not None and self.letter_temperature != 1.0:
            self.loss_function = TemperatureScaledLoss(
                self.loss_function, self.letter_temperature
            )
            log.info(
                "[letter] generation loss tempered by tau=%.4g",
                self.letter_temperature,
            )
        elif self.loss_function is None and self.letter_temperature != 1.0:
            # Inference configs pass loss_function: null, so the temperature has no effect.
            log.warning(
                "[letter] letter_temperature=%.4g set but this model has no loss "
                "function (inference config); the temperature is inert.",
                self.letter_temperature,
            )
