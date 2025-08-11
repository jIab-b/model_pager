# Project Checkpoint 2 – Unified-Memory, On-Demand Inference Runtime

## 1. Overview
The runtime implements a three-tier hierarchy that allows huge models to run on tiny GPUs without modifying model code:

```
Disk (.safetensors)  →  Unified Memory pool  →  GPU VRAM pages
         (async NVMe DMA)    (cudaMemcpyAsync)    (cudaMemPrefetchAsync)
```

1. **Disk → UM streaming** ‑ Each tensor slice is copied directly from an `mmap`’d file into a pre-allocated Unified-Memory pool; no intermediate CPU tensor is created.
2. **UM → GPU migration** ‑ Pages of the pool are prefetched to the active GPU just before use and evicted back to host when the layer finishes.
3. **Scheduler & cap** ‑ `MemoryManager` enforces a user VRAM budget via a custom allocator hook; if bringing a layer in would exceed the cap it evicts inactive pages or, as a last resort, blocks.
4. **Meta-device skeletons** ‑ Model architectures are instantiated on the `meta` device; real parameters are bound to the UM pointers only when the layer is acquired, eliminating construction overhead.
5. **Extensible kernels** ‑ Users can register fused CUDA kernels per layer; if none are provided the standard PyTorch kernels run unmodified.

The result: **GPU memory holds only the pages required by the layer that is currently executing**, while the rest of the model sits in pageable system memory or on disk, and disk/PCIe transfers overlap with computation for high throughput.

## 2. Directory Structure
```
├── csrc/                # CUDA / C++ back-end
│   ├── weight_pager.hpp/.cu   # UM pool + reserve/prefetch/evict impl
│   ├── um_tensor.cpp          # wraps UM slice as torch::Tensor
│   ├── alloc_hook.cpp         # Caching-allocator cap + auto-evict stub
│   ├── activation_pool.cpp    # sample activation allocator (unused demo)
│   └── CMakeLists.txt         # (optional) alternative build route
│
├── pysrc/               # Python runtime
│   ├── __init__.py
│   ├── _version.py
│   └── core.py          # *single* entry-point:
│        • LazyModule + build_lazy_page_table
│        • MetaModule (meta-device skeleton)
│        • MemoryManager (registration, acquire/release, LRU eviction)
│        • Inline build of weight_pager & alloc_hook extensions
│        • Unified-Memory copy-in + parameter wrapping
│
├── models/              # Front-ends using the runtime
│   ├── wan.py           # Diffusers Wan 2.1 lazy loader (GPU paging)
│   └── wan_comfy.py     # Comfy-UI Wan 2.1 using MemoryManager
│
└── docs/
    └── check_2.md       # this file
```

## 3. What Works Now
1. **Weight flow**  
   Disk (mmap) → `cudaMemcpyAsync` → UM pool → `cudaMemPrefetchAsync` → GPU pages.  
   No persisting CPU tensor allocations.
2. **Runtime classes**  
   *LazyModule* for HF-style sub-modules; *MemoryManager* orchestrates registration, prefetch, eviction and LRU under a VRAM cap.
3. **Allocator cap**  
   `alloc_hook` counts every raw GPU allocation and empties the CUDA cache; raises if cap still exceeded.
4. **Front-end models**  
   Both Diffusers and Comfy-UI drivers run through the same manager.
5. **All helper modules inlined**  
   `_pager` and `_alloc` logic now embedded in `core.py`, simplifying import graph.

## 4. Remaining Work (TODO)
ID | Status | Task
---|--------|------
`todo4` | pending | Enhance allocator hook to migrate cold non-weight tensors back to host instead of throwing.
`todo5` | pending | Provide an optional offline wheel build / `setup.py` for environments without compiler.
`todo6` | pending | Add stress tests for VRAM cap, overlap timing and failure recovery.
`todo7` | pending | Implement look-ahead scheduler to prefetch next modules while current one computes.
`todo8` | pending | Bypass Python safetensors loader entirely: direct mmap + async copy (prototype exists, needs production wrapper).
`todo9` | completed | Remove pinned CPU cache (done in this checkpoint).
`todo10` | completed | Refactor LazyModule into `core.py` (done).

## 5. Next Engineering Step
Focus on **todo7**: design a small scheduler that takes an ordered list of module-ids produced by tracing or user input and:
1. Starts disk→UM copies for layer *i + 2* while layer *i* runs.
2. Issues UM→GPU prefetch for layer *i + 1*.
3. Maintains a two-layer double buffer and updates LRU timestamps for eviction.
This will hide remaining I/O latency and maximise GPU utilisation.
