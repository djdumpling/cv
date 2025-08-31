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
    # Suppress albumentations update warnings
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="Path to img2dataset job JSON from data/build_manifests.py")
    args = ap.parse_args()

    with open(args.job, "r") as f:
        cfg = json.load(f)

    print("[img2dataset] Loaded job config:")
    print(json.dumps(cfg, indent=2))

    # Validate required parameters
    required_params = ["url_list", "output_folder", "output_format"]
    for param in required_params:
        if param not in cfg:
            raise ValueError(f"Missing required parameter: {param}")

    # Check if URL list file exists
    if not os.path.exists(cfg["url_list"]):
        raise FileNotFoundError(f"URL list file not found: {cfg['url_list']}")

    os.makedirs(cfg["output_folder"], exist_ok=True)
    
    try:
        download(**cfg)
        print("[OK] img2dataset run complete.")
        print(f"[OK] Shards at: {cfg['output_folder']}")
        print("[Hint] Inspect per-shard parquet stats to refine number_sample_per_shard if needed.")
    except Exception as e:
        print(f"[ERROR] img2dataset failed: {e}")
        raise

if __name__ == "__main__":
    main()