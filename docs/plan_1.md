## Objectives

- Keep model skeletons on meta; avoid CPU/GPU RAM spikes.
- Python orchestrates launch order and scheduling; C/CUDA handles UMA paging and compute.
- Minimize Python↔C interop: prepare/run/cleanup only; optional schedule updates later.
- No CPU tensor instantiation at any point.
- Preserve existing `cli_run` logs format (`offsets`, `sizes`, `um_ptr`, `um_pages`, `gpu_loaded`).

## Responsibilities and API boundary

| Area | Python | C/CUDA |
|---|---|---|
| Orchestration | Decide module order (t5 → transformer → vae); run scheduler | Maintain kernel registry and per-module launch sequence |
| Weight metadata | Parse safetensors header to build `offsets/sizes/dtypes/shapes` | — |
| UMA memory | Plan UMA layout; call reserve/prefetch/evict; stage weights via new helper | UMA pool, page size, prefetch/evict, file→UMA staging implementation |
| Compute | Calls `launch_module(name, inputs)` | Launch kernels/modules bound to UMA-backed weights |
| Logging | Emit `/logs/log_MM_*.log` | Provide memory stats (optional) |

Reuse existing C exports where possible: `model_reserve(bytes)`, `model_prefetch()`, `model_evict()`, `launch_module(name, inputs)`, `get_memory_stats()`.

Introduce one minimal helper in C: `model_stage_from_safetensors(path, copies, chunk_bytes)`.

## Data model in Python (`MemoryManager`)

- `tiers[name]`
  - Static: `meta`, `path`, `offsets`, `sizes`, `dtypes`, `shapes`
  - Layout: `um_layout` = per-tensor UMA offsets (page-aligned), `total_bytes`
  - Runtime: `um_ptr` (int|None), `um_pages` (int), `gpu_loaded` (bool; stays False), `t` (last access)
- `page_table[f"{name}.{tname}"]` → `{path, bytes, offset, module, dtype}`

## Scheduler flow (sequential, one module at a time)

For each `name` in `["t5", "transformer", "vae"]`:

1. prepare
   - Compute `um_layout` and `total_bytes` (page-align to 64 KiB) if not cached.
   - Call C:
     - `model_reserve(total_bytes)`
     - `model_stage_from_safetensors(path, copies, chunk_bytes=8<<20)`
     - `model_prefetch()`
2. run
   - `launch_module(name, inputs)` (C executes registered module/kernel sequence).
3. cleanup
   - `model_evict()`

Notes:
- No materialization of weights or modules in Python.
- No CPU tensors; header parsing reads JSON only; staging reads file chunks directly into UMA.

## UMA layout planning (Python)

- Inputs: `offsets`, `sizes` from safetensors header.
- Outputs: `um_layout` with entries `{tname: {um_off, size}}` and `total_bytes`.
- Policy: pack tensors contiguously, align each `um_off` and `total_bytes` to 64 KiB pages.
- Store `um_layout` in `tiers[name]`.

## Weight staging (C helper)

Signature: `model_stage_from_safetensors(path, copies, chunk_bytes)`

- `copies` is a vector of `{file_off, um_off, size}` computed in Python from `offsets` and `um_layout`.
- For each copy region, read from `path` at `file_off` and write into UMA base + `um_off`.
- Use chunked reads/writes (e.g., 8–64 MB) to avoid host RAM spikes.
- Validate total bytes and bounds against `model_reserve`d range.

## Compute path

- Prefer existing `load_module(name, path)` and `launch_module(name, inputs)` (TorchScript) to reuse current C code.
- C-side module/kernels must be bound to UMA-backed storages before/during execution.
- Python provides inputs only (e.g., latent, cond tokens) and triggers C to run the module by `name`.

## Logging

- After `register()`, logs include `offsets`, `sizes`, `gpu_loaded=False`, `um_ptr=None`, `um_pages=0`.
- If desired, call prepare before logging to populate `um_ptr` and `um_pages`.
- Keep log format identical to existing `/logs/log_MM_*.log`.

## Error handling and validation

- Header validation: `size == prod(shape) * elem_size(dtype)` per tensor.
- Staging: check read/write counts, bounds, and UMA capacity.
- Runtime: propagate clear errors with module name and file path.

## Phases and deliverables

- D1: Python header indexer + layout planner; unchanged logs; no compute.
- D2: Implement `model_stage_from_safetensors` in C; prepare/prefetch/evict loop; smoke test one module.
- D3: Full pipeline t5 → transformer → vae via Python scheduler; compute in C via `launch_module`.
- D4: Optional: expose `update_schedule`/`process_schedule` for future policies; keep single-module behavior now.

## Future work

- VRAM cap and LRU for temporary GPU work buffers if needed.
- Memory telemetry via `get_memory_stats()` exposure.
- Optional multi-module prefetch overlap once correctness is proven.


