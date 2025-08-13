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
import torch.nn as _nn
from pysrc import meta_ops as _mops

# Patch comfy.ops with our meta fallbacks (Linear, Conv*, RMSNorm, etc.)
_mops.apply(comfy.ops)


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
    from comfy.ldm.wan.model import WanModel
    return WanModel(device="meta", dtype=torch.float16, operations=comfy.ops)

def vae_skel():
    from comfy.ldm.wan.vae import WanVAE
    import torch
    return WanVAE().to(device=torch.device("meta"), dtype=torch.float16)

MM = MemoryManager(gpu="cuda")

MM.register("t5",         MetaModule(t5_skel),
            str(ROOT / "umt5_xxl_fp16.safetensors"))
MM.register("transformer",MetaModule(wan_skel),
            str(ROOT / "wan2.1_t2v_1.3B_fp16.safetensors"))
MM.register("vae",        MetaModule(vae_skel),
            str(ROOT / "wan_2.1_vae.safetensors"))

# ###### debug ###########################################################
# ---------------------------- DEBUG DUMPS ----------------------------- #
# Basic dump: modules + ops (no FX)
def _debug_dump():
    from pathlib import Path
    logs_dir = (Path(__file__).resolve().parent.parent / "logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    builders = {"t5": t5_skel, "wan": wan_skel, "vae": vae_skel}

    for name, fn in builders.items():
        model = fn()

        module_names = {m.__class__.__name__ for m in model.modules()}
        ops_from_modules = {m.__class__.__qualname__ for m in model.modules() if m.__class__.__module__.startswith("comfy.ops")}
        ops_global = {k for k, v in vars(comfy.ops).items() if callable(v)}

        out_lines = (
            ["# MODULES"] + sorted(module_names) +
            ["-- OPS"] + sorted(ops_from_modules | ops_global)
        )

        (logs_dir / f"comfy_log_{name}.log").write_text("\n".join(out_lines))


# Extended dump: includes FX trace (relies on torch.fx; useful on larger GPU)
def _debug_dump_fx():
    from pathlib import Path
    import torch.fx as fx
    logs_dir = (Path(__file__).resolve().parent.parent / "logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    builders = {"t5": t5_skel, "wan": wan_skel, "vae": vae_skel}

    for name, fn in builders.items():
        model = fn()

        # gather module class names
        module_names = {m.__class__.__name__ for m in model.modules()}

        # ops present (patched into comfy.ops)
        ops_from_modules = {m.__class__.__qualname__ for m in model.modules() if m.__class__.__module__.startswith("comfy.ops")}
        ops_global = {k for k, v in vars(comfy.ops).items() if callable(v)}

        # FX trace of the skeleton
        try:
            traced = fx.symbolic_trace(model)
            fx_lines = [f"{n.op}:{n.target}" for n in traced.graph.nodes]
        except Exception as e:
            fx_lines = [f"FX_trace_error:{e}"]
            fx_lines += [f"module:{path}:{mod.__class__.__name__}" for path, mod in model.named_modules() if path]

        out_lines = (
            ["# MODULES"] + sorted(module_names) +
            ["-- OPS"] + sorted(ops_from_modules | ops_global) +
            ["-- FX_GRAPH"] + fx_lines
        )

        (logs_dir / f"comfy_log_{name}_fx.log").write_text("\n".join(out_lines))


 

# ---------------------------------------------------- #
# 3.  Entrypoint for offline logging                    #
# ---------------------------------------------------- #

if __name__ == "__main__":
    _debug_dump()