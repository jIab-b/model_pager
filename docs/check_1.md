# Project Checkpoint 1 – Memory-Paging Inference Engine

## 1. Original Goal
Design an inference runtime that:
* **Streams model weights** from Disk → CPU-pinned → GPU on-demand, keeping GPU VRAM footprint close to the current layer only.
* **Accounts for every tensor** (weights *and* intermediates) so a user-defined VRAM cap can never be exceeded.
* Provides a **page-table like description** of upcoming operations, allowing look-ahead prefetch/eviction similar to vLLM’s paged attention.
* Works for **arbitrary PyTorch models**, not just Wan 2.1, and optionally lets users drop in fused CUDA kernels.

## 2. Repository Layout (relevant files)
```
models/
  wan_comfy.py           # Comfy-UI front-end using MemoryManager
  wan.py                 # diffusers frontend 
pysrc/
  core.py                # MemoryManager + MetaModule + VRAM cap + pager & alloc hooks
  page_manager.py        # LazyModule (HF-style)  – thin helper
  _pager.py              # Builds CUDA pager extension, exposes reserve/prefetch/evict
  _alloc.py              # Builds allocator-hook extension, exposes set_cap
csrc/
  weight_pager.hpp/.cu   # Unified-Memory pool + reserve/prefetch/evict (working)
  um_tensor.cpp          # (stub) create torch::Tensor directly on UM block
  alloc_hook.cpp         # GPU allocator byte-counter + hard cap
```

## 3. What Works Now
1. **Unified-memory pager:**
   * Global UM pool allocated with `cudaMallocManaged`.
   * `pager_reserve`, `pager_prefetch`, `pager_evict` callable from Python.
2. **CUDA allocator hook:**
   * Counts all raw allocations and enforces a hard VRAM cap.
3. **MemoryManager (Python):**
   * Registers modules (meta skeleton + weights path).
   * Streams weights Disk → CPU-pinned → UM.
   * Prefetches UM pages to GPU on `acquire`, evicts on `release`.
   * LRU eviction if bringing a module in would break the cap.
   * Exposes context `MM.use(name)` for one-shot execution.
4. **Front-ends:** `wan_comfy.py` + `LazyModule` proof that diffusers/Comfy models can run under the manager.

## 4. Known Stubs / Work Remaining
* **UM-backed tensors**  
  `um_tensor.cpp` is stubbed; `MetaModule.materialise` still copies weights into new GPU allocations instead of wrapping the UM block.
* **Parameter copy into UM:** `_load_from_disk` allocates the pool but does not copy each parameter tensor into it or record offsets.
* **Scheduler / op list:** no global "page-table" yet; modules are loaded lazily without look-ahead prefetch.
* **Allocator hook paging:** hook enforces cap but does not page non-weight tensors back to host when near limit.
* **Redundancy:** `page_manager.LazyModule` duplicates some `MemoryManager` logic and should be refactored into a thin shim.
* **Build tooling & tests:** wheels, CI, stress/OOM tests, docs.

## 5. Current TODO List
(id, status, description)
* todo1 – completed – Integrate _pager UM allocation into MemoryManager
* todo2 – pending   – Prefetch pages on acquire and evict on release using _pager
* todo3 – pending   – Refactor MetaModule.materialise to build tensors directly from UM block
* todo4 – pending   – Implement allocator hook paging / full tensor accounting
* todo5 – pending   – Create robust build script for CUDA extensions
* todo6 – pending   – Add tests for VRAM cap and pager overlap

## 6. Immediate Next Step
Finish **todo3**: copy each parameter into the reserved UM slice and create tensors in-place via `tensor_from_um`; update `materialise` to reuse those tensors. This eliminates the duplicate GPU copy and unlocks true zero-copy paging.
