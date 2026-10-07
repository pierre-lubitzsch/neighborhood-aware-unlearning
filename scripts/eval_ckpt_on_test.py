"""Evaluate a trained checkpoint on the test split.

Runs ``trainer.test(model, datamodule, ckpt_path=cfg.ckpt_path)`` and writes
NDCG@K / Recall@K to the experiment's CSVLogger
(``${paths.output_dir}/csv/version_0/metrics.csv``). ``src/train.py`` only tests
the best checkpoint of the current run, so this entry point is used for
arbitrary (clean, poisoned, or unlearned) checkpoints.

Usage::

    python -m scripts.eval_ckpt_on_test experiment=tiger_train_flat \\
        data_dir=src/data/amazon_data/beauty \\
        semantic_id_path=<merged_predictions_tensor.pt> \\
        ckpt_path=<ckpt> num_hierarchies=4 train=False test=True

The ``filter`` baseline stores its decode mask as module state rather than in
the checkpoint. Pass ``decode_filter_mask=<unlearn_run_dir>/filter_mask.json``
to reinstall it before evaluation; a missing or empty mask raises an error.
"""

from __future__ import annotations

import os

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.components.unlearning.filter_utils import (  # noqa: E402
    forbidden_sids_from_codebook,
    load_filter_mask,
    user_forbidden_sids_from_codebook,
)
from src.components.unlearning.neighborhood_sampler import load_codebook  # noqa: E402
from src.utils import RankedLogger, extras  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402


command_line_logger = RankedLogger(__name__, rank_zero_only=True)


def install_decode_filter(cfg: DictConfig, model) -> None:
    """Reinstall the ``filter`` baseline's decode mask onto ``model``.

    No-op when ``decode_filter_mask`` is unset; raises if the mask is set but
    unusable. The mask is a plain attribute, so loading a state dict in
    ``trainer.test(ckpt_path=...)`` does not clear it.
    """
    mask_path = cfg.get("decode_filter_mask")
    if not mask_path:
        return
    mask_path = str(mask_path)
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(
            f"decode_filter_mask={mask_path!r} does not exist. The mask is "
            "written by the filter unlearning run (filter_mask.json)."
        )
    if not cfg.get("semantic_id_path"):
        raise ValueError(
            "decode_filter_mask needs semantic_id_path to map item ids to "
            "semantic ids."
        )
    if not hasattr(model, "set_decode_filter"):
        raise TypeError(
            f"model {type(model).__name__} has no set_decode_filter; the decode "
            "filter is only defined for SemanticIDEncoderDecoder models."
        )

    mask = load_filter_mask(mask_path)
    filter_mode = str(mask.get("filter_mode", "global"))
    codebook = load_codebook(str(cfg.semantic_id_path))
    forbidden_sids = forbidden_sids_from_codebook(
        codebook, mask.get("forbidden_item_ids") or []
    )
    user_forbidden_sids = user_forbidden_sids_from_codebook(
        codebook, mask.get("user_forget_items")
    )
    if not forbidden_sids and not user_forbidden_sids:
        raise ValueError(
            f"{mask_path} resolves to an empty decode filter (no item mapped "
            "into the codebook). Check that semantic_id_path matches the "
            "identifier space the mask was built against."
        )
    model.set_decode_filter(
        forbidden_sids=forbidden_sids,
        filter_mode=filter_mode,
        user_forbidden_sids=user_forbidden_sids,
    )
    command_line_logger.info(
        f"Decode filter installed from {mask_path}: mode={filter_mode}, "
        f"{len(forbidden_sids)} forbidden semantic ids, "
        f"{len(user_forbidden_sids)} users with a per-user mask."
    )


def evaluate(cfg: DictConfig) -> None:
    """Load ``cfg.ckpt_path`` into the model and run ``trainer.test``."""
    if not cfg.get("ckpt_path"):
        raise ValueError(
            "ckpt_path is required; pass the checkpoint to evaluate via "
            "ckpt_path=<path>."
        )

    with pipeline_launcher(cfg) as pipeline_modules:
        install_decode_filter(cfg, pipeline_modules.model)
        command_line_logger.info(
            f"Running trainer.test(ckpt_path={cfg.ckpt_path}) ..."
        )
        pipeline_modules.trainer.test(
            model=pipeline_modules.model,
            datamodule=pipeline_modules.datamodule,
            ckpt_path=cfg.ckpt_path,
        )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
