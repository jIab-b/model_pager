from __future__ import annotations
from pathlib import Path
import torch
from pysrc.core import MetaModule, MemoryManager

ROOT = Path("/home/beed1089/vllm/model_pager/safetensors")

class _P:
    def __init__(self, shape):
        self.shape = torch.Size(shape)

class _ShapeProxy(torch.nn.Module):
    def __init__(self, shapes: dict[str, tuple[int, ...]]):
        super().__init__()
        self._shapes = shapes
    def named_parameters(self):
        return [(k, _P(v)) for k, v in self._shapes.items()]

def umt5():
    m = MM._tiers["umt5"]
    return _ShapeProxy(m["shapes"])

def clip_skel():
    m = MM._tiers["clip"]
    return _ShapeProxy(m["shapes"])

def t5_skel():
    m = MM._tiers["t5"]
    return _ShapeProxy(m["shapes"])

def flux_skel():
    m = MM._tiers["transformer"]
    return _ShapeProxy(m["shapes"])

def flux_vae_skel():
    m = MM._tiers["vae"]
    return _ShapeProxy(m["shapes"])

def _empty_builder():
    return torch.nn.Module()

MM = MemoryManager(gpu="cuda")

MM.register("umt5",       MetaModule(_empty_builder),
            str(ROOT / "umt5_xxl_fp16.safetensors"))
MM.register("clip",       MetaModule(_empty_builder),
            str(ROOT / "clip_encoder/model.safetensors"))
MM.register("t5",         MetaModule(_empty_builder),
            str(ROOT / "t5_encoder" / "t5_1.1_xxl_fp16.safetensors"))
MM.register("transformer",MetaModule(_empty_builder),
            str(ROOT / "transformer"))
MM.register("vae",        MetaModule(_empty_builder),
            str(ROOT / "vae/diffusion_pytorch_model.safetensors"))

# WAN 2.1 transformer skeleton (reinstated)
MM.register("wan21",      MetaModule(_empty_builder),
            str(ROOT / "wan2.1_t2v_1.3B_fp16.safetensors"))


