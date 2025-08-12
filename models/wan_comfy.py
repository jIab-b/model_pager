# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 (text-to-video) driven through the zero-RAM meta manager.
Only the active block is ever resident on GPU.

Weights layout (same as Comfy-UI):
  models/
    ├─ diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors
    ├─ text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
    └─ vae/wan_2.1_vae.safetensors
"""

from __future__ import annotations
import json, random, os
from pathlib import Path
from typing import List

# Add project root and ComfyUI to Python path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
comfyui_root = Path("/home/beed1089/ComfyUI")
for path in [project_root, comfyui_root]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch, safetensors.torch as safe
import comfy.ops, comfy.samplers, comfy.utils as cutils



import comfy.text_encoders.t5 as _t5mod
# ensure gelu_new activation present
if "gelu_new" not in _t5mod.activations:
    _t5mod.activations["gelu_new"] = torch.nn.GELU(approximate="tanh")

# meta-device ops 
import torch.nn as _nn, types as _types
_fallback = _types.SimpleNamespace(
    Linear=lambda in_f, out_f, bias=False, **kw: _nn.Linear(in_f, out_f, bias=bias, device="meta"),
    LayerNorm=lambda dim, eps=1e-5, **kw: _nn.LayerNorm(dim, eps, device="meta"),
    Embedding=lambda n, d, **kw: _nn.Embedding(n, d, device="meta"),
    Conv2d=lambda cin, cout, k, s=1, p=0, bias=True, **kw: _nn.Conv2d(cin, cout, k, s, p, bias=bias, device="meta"),
    Conv3d=lambda cin, cout, k, s=1, p=0, bias=True, **kw: _nn.Conv3d(cin, cout, k, s, p, bias=bias, device="meta"),
    ConvTranspose2d=lambda cin, cout, k, s=1, p=0, bias=True, **kw: _nn.ConvTranspose2d(cin, cout, k, s, p, bias=bias, device="meta"),
    GroupNorm=lambda num_groups, num_channels, eps=1e-5, **kw: _nn.GroupNorm(num_groups, num_channels, eps=eps, device="meta"),
    SiLU=lambda **kw: _nn.SiLU(),
)
for _name in _fallback.__dict__.keys():
    if not hasattr(comfy.ops, _name):
        setattr(comfy.ops, _name, getattr(_fallback, _name))




from pysrc.core import MetaModule, MemoryManager

ROOT = Path("/home/beed1089/vllm/model_pager/safetensors")

# ---------------------------------------------------- #
# 1.  Build skeletons on meta device                   #
# ---------------------------------------------------- #
def t5_skel():
    from comfy.text_encoders import t5
    cfg = json.load(open(ROOT / "umt5_xxl_config.json"))
    cfg.setdefault("model_type", "umt5")
    return t5.T5(cfg, device="meta", dtype=torch.float16, operations=comfy.ops)

def wan_skel():
    from comfy.ldm.wan import model as wan
    return wan.create_model_from_config(checkpoint=None, device="meta", dtype=torch.float16)

def vae_skel():
    from comfy.ldm.wan import vae
    return vae.create_vae(device="meta")

MM = MemoryManager(gpu="cuda")

MM.register("t5",         MetaModule(t5_skel),
            str(ROOT / "umt5_xxl_fp16.safetensors"))
MM.register("transformer",MetaModule(wan_skel),
            str(ROOT / "wan2.1_t2v_1.3B_fp16.safetensors"))
MM.register("vae",        MetaModule(vae_skel),
            str(ROOT / "wan_2.1_vae.safetensors"))

# ###### debug ###########################################################
# save each component count to logs/comfy_*.log
import safetensors.torch as _safe, pathlib as _pl, os as _os
_logs_dir = (_pl.Path(__file__).resolve().parent.parent / "logs").resolve()
_logs_dir.mkdir(parents=True, exist_ok=True)
_file_map = {
    "t5": ROOT / "umt5_xxl_fp16.safetensors",
    "transformer": ROOT / "wan2.1_t2v_1.3B_fp16.safetensors",
    "vae": ROOT / "wan_2.1_vae.safetensors",
}
for _name, _path in _file_map.items():
    if _path.exists():
        with _safe.safe_open(str(_path), framework="pt") as _sf:
            _count = len(list(_sf.keys()))
        (_logs_dir / f"comfy_{_name}.log").write_text(str(_count))
# ###### debug ###########################################################

# ---------------------------------------------------- #
# 2.  Text → condition helper                          #
# ---------------------------------------------------- #
from transformers import T5Tokenizer
_tok = T5Tokenizer.from_pretrained("google/umt5-xxl")

def encode(prompt: str) -> torch.Tensor:
    ids = _tok(prompt,
               padding="max_length", truncation=True,
               max_length=256, return_tensors="pt").input_ids
    with MM.use("t5") as t5:
        return t5(ids.cuda(non_blocking=True))[0]

# ---------------------------------------------------- #
# 3.  Top-level generator                              #
# ---------------------------------------------------- #
class Wan21:
    def __init__(self, fp16=True):
        self.dtype = torch.float16 if fp16 else torch.bfloat16

    @torch.inference_mode()
    def __call__(self,
                 prompt: str,
                 negative: str = "",
                 H: int = 480,
                 W: int = 832,
                 T: int = 81,
                 steps: int = 25,
                 cfg: float = 5.0,
                 seed: int | None = None) -> List[torch.Tensor]:

        if seed is None or seed < 0: seed = random.randint(0,2**31)

        cond  = encode(prompt)
        ncond = encode(negative) if negative else None

        lat = torch.randn(1,4,T,H//8,W//8,
                          device="cuda", dtype=self.dtype)

        sampler = comfy.samplers.KSampler(
            model=None,     # we override on-the-fly
            steps=steps,
            cfg_scale=cfg,
            sampler_name="dpmpp_2m",
            scheduler="karras",
            seed=seed,
        )

        for s in sampler:
            with MM.use("transformer") as tfm:
                lat = s(lat, cond, ncond, model_override=tfm)

        with MM.use("vae") as vae:
            frames = vae.decode(lat)

        return frames.cpu()

# ---------------------------------------------------- #
# 4.  Simple CLI                                       #
# ---------------------------------------------------- #
if __name__ == "__main__":
    import argparse, imageio
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="wan21.mp4")
    args = ap.parse_args()

    gen = Wan21()
    vid = gen(args.prompt)
    imageio.mimsave(args.out,
                    [f.permute(1,2,0).numpy() for f in vid], fps=16)
    print("done →", args.out)