import torch
import triton
import triton.language as tl
import torch
import time

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
GROUP_M = 8  # improves L2 locality

@triton.jit
def matmul(A, B, C, M, N, K, stride_am, stride_ak,
           stride_bk, stride_bn, stride_cm, stride_cn, **META):
    
    # compile-time meta-parameters: tile sizes & grouping
    BLOCK_M, GROUP_M = META['BLOCK_M'], META['GROUP_M']
    BLOCK_N = META['BLOCK_N']
    BLOCK_K = META['BLOCK_K']

    # since triton launches a grid of programs, identify a triton kernel by its 2D id (dimensions)
    _pid_m = tl.program_id(0)
    _pid_n = tl.program_id(1)

    # group several M-tiles together when sweeping N to improve L2 locality (cache more reuse)
    pid_m = _pid_m // GROUP_M
    pid_n = (_pid_n * GROUP_M) + (_pid_m % GROUP_M)

    # rm, rn:  row/col indices of C covered by this tile (vector of indices)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # BLOCK_M rows
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) # BLOCK_N cols

    # rk indexes along the reduction dimension K for one K-tile of size BLOCK_K
    rk = tl.arange(0, BLOCK_K)

    # compute pointers to the first A and B tiles
    # memory layout is specified by strides:
    # - For A of shape [M, K], element A[i, j] is at A_ptr + i*stride_am + j*stride_ak.
    # - For B of shape [K, N], element B[i, j] is at B_ptr + i*stride_bk + j*stride_bn.
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk  + rn[None, :] * stride_bn)

    # create an FP32 accumulator tile, improves num stability even if inputs are fp16/bf16
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # each iteration (chunks of BLOCK_K):
    #   1) load a tile of A (BLOCK_M x BLOCK_K) and B (BLOCK_K x BLOCK_N)
    #   2) acc += A_tile @ B_tile
    #   3) advance A/B pointers by BLOCK_K along K
    for k in range(K, 0, -BLOCK_K):
        # load current A, B tiles from global memory into registers/shared memory
        a = tl.load(A)
        b = tl.load(B)

        # tile-level matrix multiply-accumulate
        acc += tl.dot(a, b)

        # advance the A and B pointers forward by one K-tile (BLOCK_K) along the reduction axis
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # C[i, j] at C_ptr + i*stride_cm + j*stride_cn
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)

    # for tiles along the matrix edge, mask prevents writing outside [0..M) x [0..N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    # store the accumulator tile back to global memory, guarded by the mask
    tl.store(C, acc, mask=mask)

def launch_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # compute strides in element units (not bytes)
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    grid = (
        triton.cdiv(M, BLOCK_M) * GROUP_M,  # dim-0 (will be grouped back into pid_m/pid_n)
        triton.cdiv(N, BLOCK_N),            # dim-1
    )

    matmul[grid](
        A, B, C, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=4, num_stages=3,  # will need to finetune in practice
    )
    return C

# Test cases to verify the implementation
def test_triton_matmul(): 
    print("Testing Triton MatMul Implementation")
    
    # Test case 1: Small matrices
    print("\n1. Testing small matrices (32x32)")
    A = torch.randn(32, 32, device='cuda', dtype=torch.float16)
    B = torch.randn(32, 32, device='cuda', dtype=torch.float16)
    
    # PyTorch reference
    C_ref = torch.matmul(A.float(), B.float())
    
    # Triton implementation
    C_triton = launch_matmul(A, B)
    
    # Check accuracy
    error = torch.abs(C_ref - C_triton).max().item()
    print(f"   (1) Max error: {error:.6f}")
    assert error < 1e-3, f"Error too large: {error}"
    print("   ✅ (1) PASSED")
    
    # Test case 2: Medium matrices
    print("\n2. Testing medium matrices (256x256)")
    A = torch.randn(256, 256, device='cuda', dtype=torch.float16)
    B = torch.randn(256, 256, device='cuda', dtype=torch.float16)
    
    # PyTorch reference
    C_ref = torch.matmul(A.float(), B.float())
    
    # Triton implementation
    C_triton = launch_matmul(A, B)
    
    # Check accuracy
    error = torch.abs(C_ref - C_triton).max().item()
    print(f"   (2) Max error: {error:.6f}")
    assert error < 1e-3, f"Error too large: {error}"
    print("   ✅ (2) PASSED")
    
    # Test case 3: Large matrices (performance test)
    print("\n3. Testing large matrices (1024x1024) - Performance comparison...")
    A = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    B = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(3):
        _ = launch_matmul(A, B)
        _ = torch.matmul(A.float(), B.float())
    
    torch.cuda.synchronize()
    
    # Time Triton
    start = time.time()
    for _ in range(10):
        C_triton = launch_matmul(A, B)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 10
    
    # Time PyTorch
    start = time.time()
    for _ in range(10):
        C_ref = torch.matmul(A.float(), B.float())
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 10
    
    print(f"   (3) Triton time: {triton_time*1000:.2f} ms")
    print(f"   (3) PyTorch time: {pytorch_time*1000:.2f} ms")
    print(f"   (3) Speedup: {pytorch_time/triton_time:.2f}x")
    
    # Check accuracy
    error = torch.abs(C_ref - C_triton).max().item()
    print(f"   (3) Max error: {error:.6f}")
    assert error < 1e-3, f"Error too large: {error}"
    print("   ✅ (3) PASSED")
    
    # Test case 4: Non-square matrices
    print("\n4. Testing non-square matrices (512x256 @ 256x1024)")
    A = torch.randn(512, 256, device='cuda', dtype=torch.float16)
    B = torch.randn(256, 1024, device='cuda', dtype=torch.float16)
    
    # PyTorch reference
    C_ref = torch.matmul(A.float(), B.float())
    
    # Triton implementation
    C_triton = launch_matmul(A, B)
    
    # Check accuracy
    error = torch.abs(C_ref - C_triton).max().item()
    print(f"   (4) Max error: {error:.6f}")
    assert error < 1e-3, f"Error too large: {error}"
    print("   ✅ (4) PASSED")
    
    # Test case 5: Edge case - very small matrices
    print("\n5. Testing edge case - very small matrices (8x8)")
    A = torch.randn(8, 8, device='cuda', dtype=torch.float16)
    B = torch.randn(8, 8, device='cuda', dtype=torch.float16)
    
    # PyTorch reference
    C_ref = torch.matmul(A.float(), B.float())
    
    # Triton implementation
    C_triton = launch_matmul(A, B)
    
    # Check accuracy
    error = torch.abs(C_ref - C_triton).max().item()
    print(f"   (5) Max error: {error:.6f}")
    assert error < 1e-3, f"Error too large: {error}"
    print("   ✅ (5) PASSED")
    
    print("\n🎉 All tests passed! Triton matmul implementation is correct.")

def test_different_dtypes():
    """Test the implementation with different data types."""
    print("\nTesting different data types...")
    
    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    
    for dtype in dtypes:
        print(f"\nTesting {dtype}...")
        A = torch.randn(128, 128, device='cuda', dtype=dtype)
        B = torch.randn(128, 128, device='cuda', dtype=dtype)
        
        # PyTorch reference
        C_ref = torch.matmul(A.float(), B.float())
        
        # Triton implementation
        C_triton = launch_matmul(A, B)
        
        # Check accuracy
        error = torch.abs(C_ref - C_triton).max().item()
        print(f"   (6) Max error: {error:.6f}")
        assert error < 1e-3, f"Error too large for {dtype}: {error}"
        print(f"   ✅ {dtype} (6) PASSED")

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Skipping tests.")
        exit(1)
    
    print("🚀 Running Triton MatMul Tests...")
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    
    try:
        test_triton_matmul()
        test_different_dtypes()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()