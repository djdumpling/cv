import os
import time
import random
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

from data.webdataset_loader import make_loader
from train.models.blocks import TinyDiT

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    print("=== SOTA T2I Day-1 Sanity ===")
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    assert torch.cuda.is_available(), "CUDA required for Day-1 sanity."
    device = torch.device(cfg.device)

    # Load subconfigs
    data_cfg  = OmegaConf.load(os.path.join(hydra.utils.get_original_cwd(), "configs", "data.yaml"))
    train_cfg = OmegaConf.load(os.path.join(hydra.utils.get_original_cwd(), "configs", "train.yaml"))

    # Build model
    dtype = torch.bfloat16 if train_cfg.dtype == "bf16" else torch.float16
    model = TinyDiT(
        dim=train_cfg.embed_dim,
        n_heads=train_cfg.n_heads,
        depth=train_cfg.n_layers,
        mlp_ratio=train_cfg.mlp_ratio,
        dropout=train_cfg.dropout,
        causal=train_cfg.causal,
        use_flash_attn=train_cfg.use_flash_attn,
    ).to(device=device, dtype=dtype)

    # Report which backend is selected
    attn_backend = model.layers[0].attn.report_backend()
    print(f"[Info] Attention backend selected: {attn_backend}")

    # Try to import flash_attn and print version if present
    try:
        import flash_attn
        print(f"[Info] flash_attn version: {getattr(flash_attn, '__version__', 'unknown')}")
    except Exception as e:
        print(f"[Warn] flash_attn not found or failed to import: {e}")

    # Synthetic batch (B, L, C)
    B, L, C = 4, train_cfg.seq_len, train_cfg.embed_dim
    x = torch.randn(B, L, C, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        y = model(x)
    torch.cuda.synchronize()

    # Timed runs for bf16/fp16 toggles
    def bench(run_dtype):
        nonlocal model, x
        model = model.to(dtype=run_dtype)
        x = x.to(dtype=run_dtype)
        iters = 50
        t0 = time.time()
        for _ in range(iters):
            y = model(x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000.0 / iters
        print(f"[Bench] dtype={run_dtype} avg_forward_ms={dt:.3f}")

    if dtype == torch.bfloat16:
        bench(torch.bfloat16)
        bench(torch.float16)
    else:
        bench(torch.float16)
        bench(torch.bfloat16)

    # Optional: construct a loader (no shards needed on Day-1)
    if data_cfg.shards_pattern is None:
        print("[Info] Skipping WebDataset loader (no shards_pattern set).")
    else:
        loader = make_loader(
            data_cfg.shards_pattern,
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            image_size=data_cfg.image_size,
            center_crop=data_cfg.center_crop,
        )
        print(f"[Info] Built WebDataset loader with batch_size={data_cfg.batch_size}")

    print("[OK] Day-1 sanity completed.")

if __name__ == "__main__":
    main()