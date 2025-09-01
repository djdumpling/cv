import io
import json
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
    
    def validate_and_process(sample):
        """Validate and process each sample, returning None for invalid ones"""
        try:
            img, txt, meta = sample
            
            # Must have valid image
            if img is None:
                return None
                
            # Ensure text is string
            if txt is None:
                txt = ""
            elif isinstance(txt, bytes):
                txt = txt.decode('utf-8', errors='ignore')
            
            # Ensure metadata is dict
            if meta is None:
                meta = {}
            elif isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except:
                    meta = {}
            
            return (img, txt, meta)
        except Exception as e:
            print(f"[WebDataset] Error processing sample: {e}")
            return None
    
    # Use a robust pipeline that handles img2dataset format
    dataset = (wds.WebDataset(shards_pattern, handler=safe_handler, empty_check=False, shardshuffle=False)
               .shuffle(shuffle_buffer)
               .decode("pil")  # Use built-in PIL decoder
               .to_tuple("jpg;png;jpeg", "txt", "json")
               .map(validate_and_process)  # Validate and clean samples
               .select(lambda x: x is not None)  # Filter out None samples
               .map_tuple(transform, lambda x: x, lambda x: x))  # Apply transforms
    
    return dataset

def safe_collate_fn(batch):
    """Custom collate function that handles None values and malformed samples"""
    # Filter out None values and invalid samples
    valid_batch = []
    for item in batch:
        if item is not None and len(item) == 3:  # Should be (image, text, metadata)
            img, txt, meta = item
            if img is not None:  # Must have valid image
                valid_batch.append(item)
    
    if len(valid_batch) == 0:
        # Return empty batch with correct structure
        return torch.empty(0, 3, 256, 256), [], []
    
    # Use default collate on valid samples
    return torch.utils.data.dataloader.default_collate(valid_batch)

def make_loader(shards_pattern: Optional[str], batch_size: int, num_workers: int, image_size: int, center_crop: bool):
    # sanity check for pure synthetic data
    if shards_pattern is None:
        return None
    # o/w, make web data set and load
    ds = make_wds(shards_pattern, image_size = image_size, center_crop = center_crop)
    return DataLoader(ds, batch_size = batch_size, num_workers = num_workers, 
                     pin_memory = True, drop_last = True, collate_fn = safe_collate_fn)