from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import json


def _read_header_len(path: str) -> int:
    with open(path, "rb") as f:
        return int.from_bytes(f.read(8), "little")


def _index_safetensors_file(path: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str], Dict[str, tuple], int]:
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
    preamble = 8 + int(header_len)
    for name, info in tensors.items():
        start, end = info["data_offsets"]
        offsets[name] = preamble + int(start)
        sizes[name] = int(end) - int(start)
        dtypes[name] = str(info["dtype"])
        shapes[name] = tuple(int(x) for x in info["shape"])
    return offsets, sizes, dtypes, shapes, header_len

#sharded safetensors
def index_safetensors_any(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        idx = p / "model.safetensors.index.json"
        if idx.exists():
            return index_safetensors_any(str(idx))
        shard_files = sorted([str(x) for x in p.glob("*.safetensors")])
        merged_offsets: Dict[str, int] = {}
        merged_sizes: Dict[str, int] = {}
        merged_dtypes: Dict[str, str] = {}
        merged_shapes: Dict[str, tuple] = {}
        tensor_files: Dict[str, str] = {}
        header_lens: Dict[str, int] = {}
        for fp in shard_files:
            off, sz, dt, sh, hlen = _index_safetensors_file(fp)
            header_lens[fp] = hlen
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
    if str(p).endswith(".index.json"):
        idx_path = Path(path)
        data = json.loads(idx_path.read_text())
        weight_map = data.get("weight_map", {})
        shard_files = sorted(list({str(idx_path.parent / v) for v in weight_map.values()}))
        per_file_index: Dict[str, Tuple[Dict[str, int], Dict[str, int], Dict[str, str], Dict[str, tuple], int]] = {}
        header_lens: Dict[str, int] = {}
        for fp in shard_files:
            off, sz, dt, sh, hlen = _index_safetensors_file(fp)
            per_file_index[fp] = (off, sz, dt, sh, hlen)
            header_lens[fp] = hlen
        merged_offsets: Dict[str, int] = {}
        merged_sizes: Dict[str, int] = {}
        merged_dtypes: Dict[str, str] = {}
        merged_shapes: Dict[str, tuple] = {}
        tensor_files: Dict[str, str] = {}
        for tensor, rel_file in weight_map.items():
            fp = str(idx_path.parent / rel_file)
            off, sz, dt, sh, _ = per_file_index[fp]
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
        )
    if str(p).endswith(".safetensors"):
        off, sz, dt, sh, hlen = _index_safetensors_file(str(p))
        return dict(
            is_sharded=False,
            path=str(p),
            files=[str(p)],
            tensor_files={k: str(p) for k in off.keys()},
            header_lens={str(p): hlen},
            offsets=off,
            sizes=sz,
            dtypes=dt,
            shapes=sh,
        )
    raise ValueError("Unsupported path for safetensors indexing")


def index_gguf(path: str) -> dict:
    try:
        import gguf  # type: ignore
    except Exception as e:
        raise ImportError("gguf package is required to index GGUF files") from e
    reader = gguf.GGUFReader(path)
    offsets: Dict[str, int] = {}
    sizes: Dict[str, int] = {}
    dtypes: Dict[str, str] = {}
    shapes: Dict[str, tuple] = {}
    tensor_files: Dict[str, str] = {}
    for t in reader.tensors:
        name = str(getattr(t, "name", ""))
        data_off = int(getattr(t, "data_offset", 0))
        nbytes = getattr(t, "nbytes", None)
        if nbytes is None:
            n_el = int(getattr(t, "n_elements", 0))
            gtype = getattr(t, "ggml_type", None)
            itemsize = int(getattr(gtype, "itemsize", 0)) if gtype is not None else 0
            nbytes = n_el * itemsize
        shp = tuple(int(x) for x in getattr(t, "shape", []))
        gtype = getattr(t, "ggml_type", None)
        dtype_str = str(gtype) if gtype is not None else "unknown"
        offsets[name] = data_off
        sizes[name] = int(nbytes)
        shapes[name] = shp
        dtypes[name] = dtype_str
        tensor_files[name] = path
    return dict(
        is_sharded=False,
        path=str(path),
        files=[str(path)],
        tensor_files=tensor_files,
        header_lens={str(path): 0},
        offsets=offsets,
        sizes=sizes,
        dtypes=dtypes,
        shapes=shapes,
    )


def index_weights_any(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        return index_safetensors_any(str(p))
    lower = p.name.lower()
    if lower.endswith(".gguf"):
        return index_gguf(str(p))
    if lower.endswith(".index.json") or lower.endswith(".safetensors"):
        return index_safetensors_any(str(p))
    raise ValueError("Unsupported weights format")


__all__ = [
    "index_weights_any",
    "index_safetensors_any",
    "index_gguf",
]


