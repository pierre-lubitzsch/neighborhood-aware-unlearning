"""Helpers for isolating unlearning run output directories."""

from __future__ import annotations

import os


def guard_fresh_unlearn_output_dir(output_dir: str) -> None:
    """Refuse to run if ``output_dir`` already holds a completed unlearning artifact.

    Use a fresh run directory for every unlearning run.
    """
    if not output_dir:
        return
    markers = (
        os.path.join(output_dir, "scif_info.json"),
        os.path.join(output_dir, "checkpoints", "unlearned.ckpt"),
    )
    existing = [p for p in markers if os.path.exists(p)]
    if existing:
        raise FileExistsError(
            f"Unlearning output dir {output_dir!r} already contains artifacts "
            f"from a prior run ({existing[0]!r}). Pick a fresh hydra.run.dir."
        )


def run_metadata_from_cfg(cfg) -> dict:
    """Small dict merged into scif_info / checkpoint metadata."""
    unlearning = cfg.get("unlearning") or {}
    return {
        "hydra_output_dir": os.path.abspath(cfg.paths.output_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "unlearning_run_tag": cfg.get("unlearning_run_tag")
        or os.environ.get("UNLEARN_RUN_TAG"),
        "request_batch_size": unlearning.get("request_batch_size"),
    }
