import os
import tarfile
import json
from PIL import Image
import io

def create_test_shard():
    os.makedirs("data/shards/test", exist_ok=True)
    
    # Create a simple test tar file
    tar_path = "data/shards/test/00000.tar"
    
    with tarfile.open(tar_path, "w") as tar:
        for i in range(3):
            # Create a simple colored image
            img = Image.new('RGB', (256, 256), color=(i*80, 100, 150))
            
            # Save image to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Create tarinfo for image
            img_info = tarfile.TarInfo(name=f"{i:06d}.jpg")
            img_info.size = len(img_bytes.getvalue())
            tar.addfile(img_info, img_bytes)
            
            # Create caption
            caption = f"Test image {i}"
            caption_bytes = io.BytesIO(caption.encode())
            caption_info = tarfile.TarInfo(name=f"{i:06d}.txt")
            caption_info.size = len(caption.encode())
            tar.addfile(caption_info, caption_bytes)
            
            # Create metadata
            metadata = {"id": i, "test": True}
            meta_str = json.dumps(metadata)
            meta_bytes = io.BytesIO(meta_str.encode())
            meta_info = tarfile.TarInfo(name=f"{i:06d}.json")
            meta_info.size = len(meta_str.encode())
            tar.addfile(meta_info, meta_bytes)
    
    print(f"Created test shard: {tar_path}")
    print("Contents:")
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            print(f"  {member.name}")

if __name__ == "__main__":
    create_test_shard()
