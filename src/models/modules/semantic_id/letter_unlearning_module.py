"""Unlearning module for LETTER checkpoints.

``LetterUnlearningModule`` combines :class:`LetterEncoderDecoder` with the
unlearning algorithms of ``TigerUnlearningModule``. LETTER's semantic IDs are
produced by a separately trained tokenizer and frozen before the recommender is
trained, so unlearning updates only the recommender parameters and no
re-assignment of IDs is needed.
"""

from __future__ import annotations

from typing import Any

from src.models.modules.semantic_id.letter_generation_model import (
    LetterEncoderDecoder,
)
from src.models.modules.semantic_id.tiger_unlearning_module import (
    TigerUnlearningModule,
)


class LetterUnlearningModule(LetterEncoderDecoder, TigerUnlearningModule):
    """LETTER model with the unlearning algorithms.

    The MRO places ``LetterEncoderDecoder`` before ``TigerUnlearningModule``, so
    LETTER's tempered loss is used while all unlearning methods are inherited.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
