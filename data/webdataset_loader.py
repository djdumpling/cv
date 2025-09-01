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
    
    def safe_handler(exn):
        """More informative error handler for debugging"""
        print(f"[WebDataset Warning] {type(exn).__name__}: {exn}")
        return True  # Continue processing
    
    # Try the standard webdataset approach first
    try:
        dataset = (wds.WebDataset(shards_pattern, handler=safe_handler, empty_check=False, shardshuffle=False)
                   .shuffle(shuffle_buffer)
                   .decode(wds.handle_extension("jpg", image_decoder))
                   .to_tuple("jpg;png", "txt", "json")
                   .select(lambda sample: all(x is not None for x in sample))  # filter out None samples
                   .map_tuple(transform, lambda x: x, lambda x:x))
        return dataset
    except Exception as e:
        print(f"[WebDataset] Standard loader failed: {e}, trying alternative...")
        
        # Fallback: more flexible approach
        def process_sample(sample):
            try:
                # Extract components with multiple fallbacks
                image = None
                for ext in ["jpg", "jpeg", "png", "webp"]:
                    if ext in sample:
                        image = sample[ext]
                        break
                
                caption = sample.get("txt", "")
                if isinstance(caption, bytes):
                    caption = caption.decode('utf-8', errors='ignore')
                
                metadata = sample.get("json", {})
                
                if image is None:
                    return None
                    
                return (image, caption, metadata)
            except Exception as e:
                print(f"[WebDataset] Error processing sample: {e}")
                return None
        
        dataset = (wds.WebDataset(shards_pattern, handler=safe_handler, empty_check=False, shardshuffle=False)
                   .shuffle(shuffle_buffer)
                   .decode(wds.handle_extension("jpg", image_decoder))
                   .map(process_sample)
                   .select(lambda x: x is not None)
                   .map_tuple(transform, lambda x: x, lambda x:x))
        return dataset
    
    return dataset

def make_loader(shards_pattern: Optional[str], batch_size: int, num_workers: int, image_size: int, center_crop: bool):
    # sanity check for pure synthetic data
    if shards_pattern is None:
        return None
    # o/w, make web data set and load
    ds = make_wds(shards_pattern, image_size = image_size, center_crop = center_crop)
    return DataLoader(ds, batch_size = batch_size, num_workers = num_workers, pin_memory = True, drop_last = True)