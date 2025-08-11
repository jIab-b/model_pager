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

import torch, safetensors.torch as safe
import comfy.ops, comfy.samplers, comfy.utils as cutils

from pysrc.core import MetaModule, MemoryManager

ROOT = Path.home() / "ComfyUI" / "models"

# ---------------------------------------------------- #
# 1.  Build skeletons on meta device                   #
# ---------------------------------------------------- #
def t5_skel():
    from comfy.text_encoders import t5
    cfg = json.load(open(ROOT / "text_encoders" / "umt5_xxl_config.json"))
    return t5.T5(cfg, device="meta", dtype=torch.float16, operations=comfy.ops)

def wan_skel():
    from comfy.ldm.wan import model as wan
    return wan.create_model_from_config(checkpoint=None, device="meta", dtype=torch.float16)

def vae_skel():
    from comfy.ldm.wan import vae
    return vae.create_vae(device="meta")

MM = MemoryManager(gpu="cuda")

MM.register("t5",         MetaModule(t5_skel),
            str(ROOT / "text_encoders" / "umt5_xxl_fp8_e4m3fn_scaled.safetensors"))
MM.register("transformer",MetaModule(wan_skel),
            str(ROOT / "diffusion_models" / "wan2.1_t2v_1.3B_fp16.safetensors"))
MM.register("vae",        MetaModule(vae_skel),
            str(ROOT / "vae" / "wan_2.1_vae.safetensors"))

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