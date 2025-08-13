# Custom-Kernel & Memory-Management Plan (Turing RTX 2060 Super, FP16)

## Goals
1. Run Wan 2.1, T5-XXL, Wan VAE end-to-end on ≤ 8 GB VRAM.
2. Match / beat stock ComfyUI throughput via kernel fusion while streaming weights through CUDA Unified Memory.
3. Keep every kernel FP16-only (tensor core friendly on Turing). No BF16/FP8.

---

## Weight / Memory Flow
```
NVMe .safetensors → UM pool (cudaMallocManaged) → GPU pages (cudaMemPrefetchAsync) → compute
```

| Stage | Implementation | Notes |
|-------|----------------|-------|
| **Reserve UM** | `_pager.reserve(total_bytes)` once per weight file | Align to 2 MB page size. |
| **Copy-in** | `cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, copy_stream)` | Pipelined while previous layer runs. |
| **Prefetch** | `cudaMemPrefetchAsync(ptr, bytes, device, prefetch_stream)` one layer ahead | Turing + CUDA 11 supported. |
| **Evict** | `cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId)` after layer finished | Triggered in `MemoryManager.release`. |
| **Look-ahead** | small scheduler keeps a 2-layer window (i, i+1) | Enough to hide PCIe gen 3 latency. |

Two CUDA streams
```
stream_compute  – launches fused kernels
stream_copy     – memcpy + prefetch/evict
cudaEventRecord / StreamWaitEvent for overlap
```

---
## Kernel Inventory

| ID | Called From | Fusion Scope | Notes |
|----|-------------|--------------|-------|
| K-ATTN | Wan/T5 `optimized_attention` replacement |  • FP16 FlashAttn (QKV load, rope, softmax, mat-mul, projection) | Single kernel, accepts packed QKV weights. |
| K-MLP | Wan/T5 `Linear→GELU→Linear` | 2 GEMMs + QuickGELU epilogue | Use TC 884 FP16 16×16. |
| K-NORM-FC | `RMSNorm+Linear` or `LayerNorm+Linear` | compute μ/σ + GEMM in epilogue | RMS = sqrt(mean(x²)); Layer = (x−μ)/σ. |
| K-CONV3D-ACT | VAE `CausalConv3d + SiLU` | 3-D conv w/ asymmetric padding, SiLU in epilogue | Use implicit-gemm TC. |
| K-CONV2D-ACT | VAE Resample & Attention `Conv2d + SiLU` | 2-D conv stride 1/2 | Leverage cudnn v8 fused op if available, else custom. |
| K-STRIDE-½ | Upsample2d conv | Implements `Upsample→Conv2d` in single kernel |  Transposed conv stride exact 2. |
| K-STRIDE-2 | Downsample2d conv | implements `ZeroPad2d→Conv2d(stride=2)` | |
| K-ADD-RES | Residual add w/ optional scale | fold into output of previous kernel | lightweight. |

### FP16 compliance on Turing
* Use tensor-core shapes (FP16 → FP16 acc).  No BF16/FP8.
* Enable `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)`.  For CUTLASS/ Triton ensure `compute_capability >= 7.5`.
* All kernels provide fallback to standard TensorCore shapes (16×8×8, 16×16×16).

---
## API Binding

`pysrc/meta_ops.py`
```python
from mykernels import (
    flash_attn_fp16 as K_ATTN,
    mlp_gelu_fp16   as K_MLP,
    norm_linear_fp16 as K_NORM,
    conv3d_silu_fp16 as K_C3,
    conv2d_silu_fp16 as K_C2,
)
FALLBACK.optimized_attention = K_ATTN
FALLBACK.Linear             = K_MLP   # only if attr ._fuse_gelu flag set
FALLBACK.RMSNorm            = lambda dim, eps=1e-6, **kw: K_NORM(dim, eps)
FALLBACK.Conv3d             = K_C3
FALLBACK.Conv2d             = K_C2
```

---
## Implementation Schedule
1. **UM pager complete** – already in `core.py`.
2. Write **prefetch/evict stream** helpers, integrate into `MemoryManager.acquire/release`.
3. Implement K-ATTN (reuse Flash-Attn v2 kernels, restrict to FP16).
4. Implement K-MLP via CUTLASS GEMM + GELU epilogue.
5. Implement K-NORM-FC (layer-norm math + GEMM epilogue).
6. Port K-CONV3D-ACT & K-CONV2D-ACT using cudnn backend or Triton.
7. Add stride-½ / stride-2 conv kernels.
8. Validate numerics vs. PyTorch FP16.
9. Measure throughput on 2060 Super (8 GB). Target: < 1.3 × SD 1.5 time per frame,  < 7 GB max VRAM.
10. Optional: compile kernels into CUDA Graph for fixed shapes.
