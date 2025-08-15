#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors.torch import safe_open, save_file


def load_all_tensors_fp16(src_dir: Path) -> OrderedDict:
    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"index file not found: {index_path}")
    idx = json.loads(index_path.read_text())
    weight_map = idx.get("weight_map", {})
    if not weight_map:
        raise RuntimeError("weight_map missing or empty in index.json")

    by_file = {}
    for name, shard in weight_map.items():
        by_file.setdefault(shard, []).append(name)

    for names in by_file.values():
        names.sort()

    merged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for shard, names in sorted(by_file.items()):
        shard_path = src_dir / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"shard not found: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in names:
                t = f.get_tensor(name)
                if t.dtype == torch.bfloat16:
                    t = t.to(dtype=torch.float16)
                elif t.dtype == torch.float32:
                    t = t.to(dtype=torch.float16)
                elif t.dtype == torch.float16:
                    pass
                else:
                    t = t.to(dtype=torch.float16)
                merged[name] = t.contiguous()
    return merged


def write_single_safetensors(tensors: OrderedDict, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_file), metadata={"format": "pt"})


def write_fp16_config(src_dir: Path, out_config: Path) -> None:
    cfg = json.loads((src_dir / "config.json").read_text())
    cfg["torch_dtype"] = "float16"
    out_config.write_text(json.dumps(cfg, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(Path("safetensors") / "t5_encoder"))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--out_config", type=str, default=None)
    args = ap.parse_args()

    src_dir = Path(args.src).resolve()
    if args.out is None:
        out_file = src_dir / "t5_1.1_xxl_fp16.safetensors"
    else:
        out_file = Path(args.out).resolve()
    if args.out_config is None:
        out_cfg = src_dir / "config-fp16.json"
    else:
        out_cfg = Path(args.out_config).resolve()

    tensors = load_all_tensors_fp16(src_dir)
    write_single_safetensors(tensors, out_file)
    write_fp16_config(src_dir, out_cfg)
    print(f"wrote: {out_file}")
    print(f"wrote: {out_cfg}")


if __name__ == "__main__":
    main()


