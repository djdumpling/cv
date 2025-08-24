"""
verify we can read the produced WebDataset shards with the loader
Expects a shards glob like: data/shards/laion_aesthetic_6p5/{000000..000099}.tar
"""

import hydra
from omegaconf import OmegaConf
from data.webdataset_loader import make_loader

@hydra.main(config_path="../configs", config_name="data", version_base=None)
def main(cfg):
    print("[Sanity-Data] Using config:")
    print(OmegaConf.to_yaml(cfg))

    shards = cfg.get("shards_pattern", None)
    assert shards, "Set configs/data.yaml: shards_pattern to your tar glob, e.g. data/shards/laion_aesthetic_6p5/{000000..000099}.tar"

    loader = make_loader(
        shards_pattern=shards,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        center_crop=cfg.center_crop,
    )
    it = iter(loader)
    imgs, caps, metas = next(it)
    print(f"[OK] batch images: {imgs.shape} | #captions: {len(caps)} | #metas: {len(metas)}")

    # peek at one meta dict to confirm aux columns are present
    sample_meta = metas[0]
    keep = {k: sample_meta.get(k) for k in ["url", "lang", "p_aesthetic", "p_watermark", "p_nsfw", "source", "width", "height", "similarity", "hash"]}
    print(f"[Meta sample] {keep}")

if __name__ == "__main__":
    main()