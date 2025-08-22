## Quick Start

To run the test on a GPU

### 1. Clone the repository
```bash
git clone https://github.com/djdumpling/cv.git
cd cv
```

### 2. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate 
```

### 3. Upgrade pip and install dependencies
```bash
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

### 4. Set environment variables (for H100 optimization)
```bash
# On Linux/macOS:
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=$(nproc)

# On Windows:
set TORCH_CUDA_ARCH_LIST=9.0
set MAX_JOBS=%NUMBER_OF_PROCESSORS%
```

### 5. Run the test
```bash
python -m train.test_run
```

## Notes

- Run from the project root directory (`cv/`)
- H100 GPU recommended for optimal performance
- FlashAttention will be automatically installed via requirements.txt
