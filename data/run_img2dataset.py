"""
creates WebDataset .tar shards with:
  - *.jpg       (resized images)
  - *.txt       (caption_orig)
  - *.json      (metadata: url/lang/p_* flags/source/width/height/similarity/hash/status/etc.)
  - *_stats.parquet sidecars per shard from img2dataset
"""
import argparse, json, os
from img2dataset import download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="Path to img2dataset job JSON from data/build_manifests.py")
    args = ap.parse_args()

    with open(args.job, "r") as f:
        cfg = json.load(f)

    print("[img2dataset] Loaded job config:")
    print(json.dumps(cfg, indent=2))

    os.makedirs(cfg["output_folder"], exist_ok=True)
    download(**cfg)

    print("[OK] img2dataset run complete.")
    print(f"[OK] Shards at: {cfg['output_folder']}")
    print("[Hint] Inspect per-shard parquet stats to refine number_sample_per_shard if needed.")

if __name__ == "__main__":
    main()