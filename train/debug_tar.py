#!/usr/bin/env python3
"""Debug script to inspect WebDataset tar structure"""

import tarfile
import json
import sys

def inspect_tar(tar_path):
    print(f"=== Inspecting {tar_path} ===")
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            print(f"Total files: {len(members)}")
            
            # Group by sample ID
            samples = {}
            for member in members:
                if '.' in member.name:
                    base_name, ext = member.name.rsplit('.', 1)
                    if base_name not in samples:
                        samples[base_name] = {}
                    samples[base_name][ext] = member.size
            
            print(f"Total samples: {len(samples)}")
            
            # Show first few samples
            for i, (sample_id, files) in enumerate(list(samples.items())[:3]):
                print(f"\nSample {sample_id}:")
                for ext, size in files.items():
                    print(f"  .{ext}: {size} bytes")
                
                # Try to read the content
                if 'txt' in files:
                    try:
                        txt_content = tar.extractfile(f"{sample_id}.txt").read().decode('utf-8')
                        print(f"  txt content: {repr(txt_content[:100])}")
                    except:
                        print(f"  txt content: [could not read]")
                
                if 'json' in files:
                    try:
                        json_content = tar.extractfile(f"{sample_id}.json").read().decode('utf-8')
                        json_data = json.loads(json_content)
                        print(f"  json keys: {list(json_data.keys())}")
                    except:
                        print(f"  json content: [could not read]")
                
                if i >= 2:  # Only show first 3 samples
                    break
                    
    except Exception as e:
        print(f"Error reading tar: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_tar.py <tar_file>")
        sys.exit(1)
    
    inspect_tar(sys.argv[1])
