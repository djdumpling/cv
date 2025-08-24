# shard recalibrator for img2dataset outputs
# Usage:
#   python -m data.recalibrate_shards \
#     --job ./manifests/plans/img2dataset_job_laion2B_en.json \
#     --target_gib 1.0 \
#     --out ./manifests/plans/img2dataset_job_laion2B_en.recal.json
from __future__ import annotations
import argparse, json, os, glob, sys
import numpy as np
import pandas as pd

SIZE_COLS = ("bytes","file_size","filesize","content_size","resized_size","original_size")

def find_parquets(root: str) -> list[str]:
    return sorted(set(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)))

def pick_size_series(df: pd.DataFrame):
    for c in SIZE_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if (s > 1000).mean() > 0.2:  # at least 20% > 1KB looks like byte sizes
                return s
    return None

def summarize_bytes(vals: list[int]) -> dict:
    a = np.array([v for v in vals if v and v > 0], dtype=np.float64)
    if a.size == 0: return {}
    return {
        "count": int(a.size),
        "avg": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--target_gib", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.job, "r") as f: job = json.load(f)
    out_folder = job.get("output_folder")
    if not out_folder or not os.path.isdir(out_folder):
        print(f"[ERR] output_folder not found: {out_folder}", file=sys.stderr); sys.exit(1)

    sizes, rows = [], 0
    for pq in find_parquets(out_folder):
        try:
            df = pd.read_parquet(pq)
        except Exception:
            continue
        s = pick_size_series(df)
        if s is None: continue
        sizes.extend(int(x) for x in s.dropna().tolist() if x and x > 0)
        rows += len(s)

    if not sizes:
        print("[ERR] No usable byte-size columns found in shard parquet stats.", file=sys.stderr); sys.exit(2)

    stats = summarize_bytes(sizes)
    target_bytes = int(args.target_gib * (1024 ** 3))
    nsps = max(256, int(target_bytes // max(1, int(stats["avg"]))))

    print("[Recal] rows_scanned:", rows)
    print(f"[Recal] avg={stats['avg']:.1f}  p50={stats['p50']:.1f}  p90={stats['p90']:.1f}  p99={stats['p99']:.1f}")
    print(f"[Recal] target={args.target_gib:.2f} GiB  ->  number_sample_per_shard={nsps}")

    job["number_sample_per_shard"] = int(nsps)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f: json.dump(job, f, indent=2)
    print("[OK] Wrote:", args.out)

if __name__ == "__main__":
    main()