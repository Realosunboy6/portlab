"""Parquet cache so Colab reruns (and phone sessions) are instant.

Cache root resolution order:
  1. PORTLAB_CACHE environment variable
  2. mounted Google Drive: /content/drive/MyDrive/portlab_cache
  3. Colab local: /content/portlab_cache
  4. anywhere else: ~/.cache/portlab
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd


def cache_root() -> Path:
    env = os.environ.get("PORTLAB_CACHE")
    if env:
        root = Path(env)
    elif Path("/content/drive/MyDrive").exists():
        root = Path("/content/drive/MyDrive/portlab_cache")
    elif Path("/content").exists():
        root = Path("/content/portlab_cache")
    else:
        root = Path.home() / ".cache" / "portlab"
    root.mkdir(parents=True, exist_ok=True)
    return root


def mount_drive() -> bool:
    """Mount Google Drive when running in Colab; harmless elsewhere."""
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        return True
    except Exception:
        return False


def cache_key(*parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load(key: str) -> pd.DataFrame | None:
    path = cache_root() / f"{key}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)
    return None


def save(key: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(cache_root() / f"{key}.parquet")
    except Exception:
        pass  # caching is best-effort; never fail the analysis over it
