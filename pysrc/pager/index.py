from __future__ import annotations
import json
import pathlib
from typing import Dict, Tuple


def _read_header_len(path: str) -> int:
    with open(path, "rb") as f:
        return int.from_bytes(f.read(8), "little")


def index_safetensors(path: str):
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    if isinstance(header, dict) and "tensors" in header and isinstance(header["tensors"], dict):
        tensors = header["tensors"]
    elif isinstance(header, dict):
        tensors = {k: v for k, v in header.items() if isinstance(v, dict) and "data_offsets" in v and "dtype" in v and "shape" in v}
    else:
        raise ValueError("Invalid safetensors header format")
    offsets: Dict[str, int] = {}
    sizes: Dict[str, int] = {}
    dtypes: Dict[str, str] = {}
    shapes: Dict[str, tuple] = {}
    for name, info in tensors.items():
        start, end = info["data_offsets"]
        offsets[name] = start
        sizes[name] = end - start
        dtypes[name] = info["dtype"]
        shapes[name] = tuple(info["shape"])
    return offsets, sizes, dtypes, shapes


def index_safetensors_any(path: str):
    p = pathlib.Path(path)
    if p.is_dir():
        idx = p / "model.safetensors.index.json"
        if idx.exists():
            path = str(idx)
        else:
            shard_files = sorted([str(x) for x in p.glob("*.safetensors")])
            merged_offsets: Dict[str, int] = {}
            merged_sizes: Dict[str, int] = {}
            merged_dtypes: Dict[str, str] = {}
            merged_shapes: Dict[str, tuple] = {}
            tensor_files: Dict[str, str] = {}
            header_lens: Dict[str, int] = {}
            for fp in shard_files:
                off, sz, dt, sh = index_safetensors(fp)
                header_lens[fp] = _read_header_len(fp)
                for k in off.keys():
                    merged_offsets[k] = off[k]
                    merged_sizes[k] = sz[k]
                    merged_dtypes[k] = dt[k]
                    merged_shapes[k] = sh[k]
                    tensor_files[k] = fp
            return dict(
                is_sharded=True,
                path=None,
                files=shard_files,
                tensor_files=tensor_files,
                header_lens=header_lens,
                offsets=merged_offsets,
                sizes=merged_sizes,
                dtypes=merged_dtypes,
                shapes=merged_shapes,
            )
    if str(path).endswith(".index.json"):
        idx_path = pathlib.Path(path)
        data = json.loads(idx_path.read_text())
        weight_map = data.get("weight_map", {})
        shard_files = sorted(list({str(idx_path.parent / v) for v in weight_map.values()}))
        merged_offsets: Dict[str, int] = {}
        merged_sizes: Dict[str, int] = {}
        merged_dtypes: Dict[str, str] = {}
        merged_shapes: Dict[str, tuple] = {}
        tensor_files: Dict[str, str] = {}
        header_lens: Dict[str, int] = {}
        per_file_index: Dict[str, Tuple] = {}
        for fp in shard_files:
            off, sz, dt, sh = index_safetensors(fp)
            per_file_index[fp] = (off, sz, dt, sh)
            header_lens[fp] = _read_header_len(fp)
        for tensor, rel_file in weight_map.items():
            fp = str(idx_path.parent / rel_file)
            off, sz, dt, sh = per_file_index[fp]
            if tensor not in off:
                continue
            merged_offsets[tensor] = off[tensor]
            merged_sizes[tensor] = sz[tensor]
            merged_dtypes[tensor] = dt[tensor]
            merged_shapes[tensor] = sh[tensor]
            tensor_files[tensor] = fp
        return dict(
            is_sharded=True,
            path=str(idx_path),
            files=shard_files,
            tensor_files=tensor_files,
            header_lens=header_lens,
            offsets=merged_offsets,
            sizes=merged_sizes,
            dtypes=merged_dtypes,
            shapes=merged_shapes,
            weight_map=weight_map,
        )
    off, sz, dt, sh = index_safetensors(str(path))
    return dict(
        is_sharded=False,
        path=str(path),
        files=[str(path)],
        tensor_files={k: str(path) for k in off.keys()},
        header_lens={str(path): _read_header_len(str(path))},
        offsets=off,
        sizes=sz,
        dtypes=dt,
        shapes=sh,
    )


