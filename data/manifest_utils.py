from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED_COLS = ["URL", "TEXT", "WIDTH", "HEIGHT", "similarity", "hash", "punsafe", "pwatermark", "aesthetic"]
OUTPUT_COLS = ["url", "caption_orig", "lang", "p_aesthetic", "p_watermark", "p_nsfw", "source"]

def normalize_laion2b_en_aesthetic(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. map LAION-2B-en-aesthetic schema to the canonical manifest schema
    2. keep WIDTH/HEIGHT/similarity/hash for downstream use (img2dataset additional columns)
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    out = pd.DataFrame()
    out["url"] = df["URL"]
    out["caption_orig"] = df["TEXT"]
    out["lang"] = "en"

    # numeric conversions with safe coercion
    out["p_aesthetic"] = pd.to_numeric(df["aesthetic"], errors="coerce")
    out["p_watermark"] = pd.to_numeric(df["pwatermark"], errors="coerce")
    out["p_nsfw"] = pd.to_numeric(df["punsafe"], errors="coerce")
    out["source"] = "laion2B-en-aesthetic"

    # keep auxiliary columns alongside output for convenience (optional)
    out["width"] = pd.to_numeric(df["WIDTH"], errors="coerce")
    out["height"] = pd.to_numeric(df["HEIGHT"], errors="coerce")
    out["similarity"] = pd.to_numeric(df["similarity"], errors="coerce")
    out["hash"] = df["hash"].astype(str)

    return out

def estimate_samples_per_shard_from_dims(
    df: pd.DataFrame,
    target_bytes: int = 1_073_741_824,  # ~1 GiB
    min_cap: int = 60_000,              # 60 KB
    max_cap: int = 600_000,             # 600 KB
    bytes_per_pixel: float = 0.5,       # ~0.5 bytes/pixel average JPEG (heuristic)
) -> int:
    """
    1. estimate average bytes per sample using WIDTH x HEIGHT, capped to [60KB, 600KB]
    2. derive samples per shard to target ~1 GiB tars
    """
    if "width" in df.columns and "height" in df.columns:
        area = (df["width"].fillna(0).clip(lower=0) * df["height"].fillna(0).clip(lower=0)).to_numpy()
        est = area * bytes_per_pixel
        est = np.clip(est, min_cap, max_cap)
        avg_bytes = int(np.nanmean(est)) if len(est) > 0 else 150_000
    else:
        avg_bytes = 150_000

    samples = max(512, int(target_bytes // max(1, avg_bytes)))
    return samples