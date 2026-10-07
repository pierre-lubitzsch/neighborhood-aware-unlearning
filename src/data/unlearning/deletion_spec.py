"""Deletion specification helpers for unlearning (session vs item-target modes)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set

VALID_DELETION_SPECS = frozenset({"session", "item", "item_pairs"})


def normalize_deletion_spec(spec: Optional[str]) -> str:
    """Return a validated deletion spec string (default ``session``)."""
    if spec is None:
        return "session"
    spec = str(spec).strip().lower()
    if spec not in VALID_DELETION_SPECS:
        raise ValueError(
            f"deletion_spec must be one of {sorted(VALID_DELETION_SPECS)}, got {spec!r}"
        )
    return spec


def load_forget_manifest(manifest_path: Optional[str]) -> Dict[str, Any]:
    """Load ``forget_manifest.json`` if present."""
    if not manifest_path or not os.path.isfile(manifest_path):
        return {}
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def resolve_forget_manifest_path(
    data_dir: str,
    forget_manifest: Optional[str] = None,
) -> Optional[str]:
    """Resolve manifest path from explicit arg or ``<data_dir>/forget_manifest.json``."""
    if forget_manifest and os.path.isfile(forget_manifest):
        return os.path.abspath(forget_manifest)
    default = os.path.join(data_dir, "forget_manifest.json")
    if os.path.isfile(default):
        return os.path.abspath(default)
    return None


def load_target_items(manifest: Dict[str, Any]) -> Set[int]:
    """Return ``I_f`` from manifest ``target_items``."""
    raw = manifest.get("target_items") or []
    return {int(x) for x in raw}


def resolve_neighborhood_centers(
    *,
    deletion_spec: str,
    forget_shard_items: Set[int],
    target_items: Set[int],
) -> Set[int]:
    """Return item ids used as neighborhood centers ``i_f``."""
    spec = normalize_deletion_spec(deletion_spec)
    if spec in ("item", "item_pairs"):
        if not target_items:
            raise ValueError(
                f"deletion_spec={spec!r} requires target_items in forget_manifest."
            )
        return set(target_items)
    return set(forget_shard_items)


def resolve_forbidden_retain_items(
    *,
    deletion_spec: str,
    forget_shard_items: Set[int],
    target_items: Set[int],
) -> Set[int]:
    """Items whose presence in a retain row disqualifies it (except repair pool)."""
    spec = normalize_deletion_spec(deletion_spec)
    if spec in ("item", "item_pairs"):
        return set(target_items)
    return set(forget_shard_items)


def manifest_deletion_spec(manifest: Dict[str, Any], override: Optional[str] = None) -> str:
    """Read deletion_spec from manifest with optional override."""
    if override is not None:
        return normalize_deletion_spec(override)
    return normalize_deletion_spec(manifest.get("deletion_spec"))
