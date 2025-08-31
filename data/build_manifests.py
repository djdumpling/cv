"""
LAION 2B-en-aesthetic
- stream metadata from HuggingFace datasets: laion/laion2B-en-aesthetic
- normalize to manifest schema plus aux columns
- filter by thresholds (min_aesthetic, max_pwatermark, max_punsafe)
- write Parquet + CSV.GZ manifest
- produce shard-plan YAML and an img2dataset JSON job that preserves aux columns
"""

from __future__ import annotations
import os, gzip, json
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
import hydra
from omegaconf import OmegaConf
from datasets import load_dataset

from .manifest_utils import (
    normalize_laion2b_en_aesthetic,
    estimate_samples_per_shard_from_dims)

@dataclass
class BuildStats:
    total_in: int = 0
    total_out: int = 0
    dropped_aesthetic: int = 0
    dropped_watermark: int = 0
    dropped_nsfw: int = 0

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _write_outputs(df: pd.DataFrame, out_root: str, out_name: str):
    _ensure_dir(out_root)
    pq_path = os.path.join(out_root, f"{out_name}.parquet")
    csv_path = os.path.join(out_root, f"{out_name}.csv")  # uncompressed CSV for img2dataset
    csv_gz_path = os.path.join(out_root, f"{out_name}.csv.gz")  # compressed for storage
    
    df.to_parquet(pq_path, index=False)
    keep = ["url", "caption_orig", "lang", "p_aesthetic", "p_watermark", "p_nsfw", "source"]
    df[keep].to_csv(csv_path, index=False)
    
    # writing compressed version for storage
    with gzip.open(csv_gz_path, "wt", encoding="utf-8") as f:
        df[keep].to_csv(f, index=False)
    
    return pq_path, csv_path

@hydra.main(config_path="../configs", config_name="ingest", version_base=None)
def main(cfg):
    print("Building LAION-2B-en-aesthetic manifest")
    print(OmegaConf.to_yaml(cfg))

    # pull the split in streaming mode and buffer into a DataFrame
    ds_name = "laion/laion2B-en-aesthetic"
    print(f"[LAION] Loading dataset: {ds_name} (streaming)")
    ds = load_dataset(ds_name, split="train", streaming=True)

    rows = []
    # need to set cfg.laion.limit to a large number or None for full-scale
    limit = cfg.laion.limit
    for i, ex in enumerate(ds):
        rows.append(ex)
        if limit is not None and (i + 1) >= limit:
            break
        if (i + 1) % 1_000_000 == 0:
            print(f"[LAION] buffered {i+1:,} rows...")

    raw_df = pd.DataFrame(rows)
    stats = BuildStats(total_in=len(raw_df))
    print(f"[LAION] Raw rows buffered: {stats.total_in:,}")

    # normalize exact schema
    df = normalize_laion2b_en_aesthetic(raw_df)

    # apply filters
    min_aes = float(cfg.laion.min_aesthetic) if cfg.laion.min_aesthetic is not None else None
    max_wm  = float(cfg.laion.max_pwatermark) if cfg.laion.max_pwatermark is not None else None
    max_nsfw= float(cfg.laion.max_punsafe) if cfg.laion.max_punsafe is not None else None

    if min_aes is not None:
        m = (df["p_aesthetic"].astype(float) >= min_aes)
        stats.dropped_aesthetic = int((~m).sum())
        df = df[m]

    if max_wm is not None:
        m = (df["p_watermark"].astype(float) <= max_wm)
        stats.dropped_watermark = int((~m).sum())
        df = df[m]

    if max_nsfw is not None:
        m = (df["p_nsfw"].astype(float) <= max_nsfw)
        stats.dropped_nsfw = int((~m).sum())
        df = df[m]

    df = df.reset_index(drop=True)
    stats.total_out = len(df)
    print(f"[LAION] in={stats.total_in:,} -> out={stats.total_out:,} | "
          f"dropped aesthetic={stats.dropped_aesthetic:,} watermark={stats.dropped_watermark:,} nsfw={stats.dropped_nsfw:,}")

    out_root = os.path.abspath(cfg.out_root)
    out_name = "laion2B_en_aesthetic_6p5"  # name reflects typical filter; adjustable via cfg

    pq_path, csv_path = _write_outputs(df, out_root, out_name)
    print(f"[OK] Manifest written:\n  {pq_path}\n  {csv_path}")

    # derive samples/shard targeting ~1 GiB tars, using width/height
    samples_per_shard = cfg.img2dataset.number_sample_per_shard
    if samples_per_shard is None:
        samples_per_shard = estimate_samples_per_shard_from_dims(
            df, target_bytes=int(cfg.target_shard_size_bytes)
        )

    # shard plan YAML (via OmegaConf)
    plan = {
        "target_shard_size_bytes": int(cfg.target_shard_size_bytes),
        "number_sample_per_shard": int(samples_per_shard),
        "estimation_method": "width_height_heuristic@0.5_Bpp_capped_60k_600k",
    }
    plans_dir = os.path.join(out_root, "plans")
    _ensure_dir(plans_dir)
    plan_path = os.path.join(plans_dir, "shard_plan_laion2B_en.yaml")
    with open(plan_path, "w") as f:
        f.write(OmegaConf.to_yaml(plan))
    print(f"[OK] Wrote shard plan:\n  {plan_path}")

    # img2dataset job JSON
    # preserve auxiliary columns so they are available in shard .json sidecars
    job = {
        "url_list": csv_path,  # uncompressed CSV file for img2dataset
        "url_col": "url",
        "caption_col": "caption_orig",
        "save_additional_columns": [
            "lang", "p_aesthetic", "p_watermark", "p_nsfw", "source"
        ],
        "output_format": str(cfg.img2dataset.output_format),
        "output_folder": os.path.abspath(os.path.join(cfg.shards_root, cfg.laion.out_name)),
        "image_size": int(cfg.img2dataset.image_size),
        "resize_mode": str(cfg.img2dataset.resize_mode),
        "processes_count": int(cfg.img2dataset.processes_count),
        "thread_count": int(cfg.img2dataset.thread_count),
        "timeout": int(cfg.img2dataset.timeout),
        "number_sample_per_shard": int(samples_per_shard)
    }
    job_path = os.path.join(plans_dir, "img2dataset_job_laion2B_en.json")
    with open(job_path, "w") as f:
        json.dump(job, f, indent=2)
    print(f"[OK] Wrote img2dataset job file:\n  {job_path}")

if __name__ == "__main__":
    main()