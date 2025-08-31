import io
from typing import Optional, Callable, Dict

import webdataset as wds
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# standardizations and normalizations
def build_transforms(image_size: int, center_crop: bool):
    t = []

    # standardize sizing with recrop
    if center_crop:
        t += [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    else:
        t += [transforms.Resize(image_size), transforms.RandomCrop(image_size)]
    t += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]

    return transforms.Compose(t)

def image_decoder(sample_bytes):
    # open image from raw bytes, ensure in RGB format
    return Image.open(io.BytesIO(sample_bytes)).convert("RGB")

def make_wds(shards_pattern: str, image_size: int = 256, shuffle_buffer: int = 2000, center_crop: bool = False, 
                    handler: Optional[Callable] = wds.warn_and_continue):
    transform = build_transforms(image_size = image_size, center_crop = center_crop)
    dataset = (wds.WebDataset(shards_pattern, handler = handler, empty_check=False, shardshuffle=False)
               .shuffle(shuffle_buffer)  # shuffle samples
               .decode(wds.handle_extension("jpg", image_decoder))  # use jpg decoder
               .to_tuple("jpg;png", "txt", "json")  # output tuples: (image, text, json)
               .map_tuple(transform, lambda x: x, lambda x:x))  # apply transform to image, leave others unchanged
    
    return dataset

def make_loader(shards_pattern: Optional[str], batch_size: int, num_workers: int, image_size: int, center_crop: bool):
    # sanity check for pure synthetic data
    if shards_pattern is None:
        return None
    # o/w, make web data set and load
    ds = make_wds(shards_pattern, image_size = image_size, center_crop = center_crop)
    return DataLoader(ds, batch_size = batch_size, num_workers = num_workers, pin_memory = True, drop_last = True)