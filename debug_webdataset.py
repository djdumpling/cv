#!/usr/bin/env python3
"""Debug WebDataset loading issues"""

import webdataset as wds
import json
import sys
import traceback
from PIL import Image
import io

def test_webdataset_raw(tar_path):
    """Test raw WebDataset loading"""
    print(f"=== Testing raw WebDataset: {tar_path} ===")
    try:
        dataset = wds.WebDataset(tar_path, empty_check=False, shardshuffle=False)
        
        count = 0
        for sample in dataset:
            print(f"\nSample {count}:")
            print(f"  Keys: {list(sample.keys())}")
            
            for key, value in sample.items():
                if key == 'json' and value:
                    try:
                        if isinstance(value, bytes):
                            json_data = json.loads(value.decode('utf-8'))
                        else:
                            json_data = json.loads(value)
                        print(f"  {key}: {list(json_data.keys())} (parsed json)")
                    except Exception as e:
                        print(f"  {key}: {type(value)} (json parse error: {e})")
                elif key == 'txt' and value:
                    try:
                        if isinstance(value, bytes):
                            txt_content = value.decode('utf-8')[:50]
                        else:
                            txt_content = str(value)[:50]
                        print(f"  {key}: '{txt_content}...' (text)")
                    except Exception as e:
                        print(f"  {key}: {type(value)} (text error: {e})")
                elif key in ['jpg', 'jpeg', 'png'] and value:
                    try:
                        img = Image.open(io.BytesIO(value))
                        print(f"  {key}: {img.size} {img.mode} (image)")
                    except Exception as e:
                        print(f"  {key}: {type(value)} (image error: {e})")
                else:
                    print(f"  {key}: {type(value)} (size: {len(value) if hasattr(value, '__len__') else 'unknown'})")
            
            count += 1
            if count >= 3:
                break
                
        print(f"\nProcessed {count} samples successfully")
        return True
        
    except Exception as e:
        print(f"ERROR in raw WebDataset test: {e}")
        traceback.print_exc()
        return False

def test_webdataset_decoded(tar_path):
    """Test WebDataset with decoding"""
    print(f"\n=== Testing decoded WebDataset: {tar_path} ===")
    try:
        def safe_handler(exn):
            print(f"[Handler] {type(exn).__name__}: {exn}")
            return True
            
        dataset = (wds.WebDataset(tar_path, handler=safe_handler, empty_check=False, shardshuffle=False)
                   .decode("pil"))  # Use built-in PIL decoder
        
        count = 0
        for sample in dataset:
            print(f"\nDecoded Sample {count}:")
            print(f"  Keys: {list(sample.keys())}")
            
            for key, value in sample.items():
                if key in ['jpg', 'jpeg', 'png'] and hasattr(value, 'size'):
                    print(f"  {key}: {value.size} {value.mode} (PIL Image)")
                else:
                    print(f"  {key}: {type(value)}")
            
            count += 1
            if count >= 3:
                break
                
        print(f"\nProcessed {count} decoded samples successfully")
        return True
        
    except Exception as e:
        print(f"ERROR in decoded WebDataset test: {e}")
        traceback.print_exc()
        return False

def test_webdataset_tuples(tar_path):
    """Test WebDataset with to_tuple"""
    print(f"\n=== Testing WebDataset with tuples: {tar_path} ===")
    try:
        def safe_handler(exn):
            print(f"[Handler] {type(exn).__name__}: {exn}")
            return True
            
        dataset = (wds.WebDataset(tar_path, handler=safe_handler, empty_check=False, shardshuffle=False)
                   .decode("pil")
                   .to_tuple("jpg;png;jpeg", "txt", "json"))
        
        count = 0
        for sample in dataset:
            img, txt, meta = sample
            print(f"\nTuple Sample {count}:")
            print(f"  Image: {type(img)} {img.size if hasattr(img, 'size') else 'no size'}")
            print(f"  Text: {type(txt)} '{str(txt)[:50]}...' " if txt else "  Text: None")
            print(f"  Meta: {type(meta)} {list(meta.keys()) if isinstance(meta, dict) else 'not dict'}")
            
            count += 1
            if count >= 2:
                break
                
        print(f"\nProcessed {count} tuple samples successfully")
        return True
        
    except Exception as e:
        print(f"ERROR in tuple WebDataset test: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_webdataset.py <tar_file>")
        sys.exit(1)
    
    tar_path = sys.argv[1]
    print(f"Debugging WebDataset: {tar_path}")
    
    # Run all tests
    test_webdataset_raw(tar_path)
    test_webdataset_decoded(tar_path)  
    test_webdataset_tuples(tar_path)
