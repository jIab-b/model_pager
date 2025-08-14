#!/usr/bin/env python
"""Simple end-to-end text-to-video run using MemoryManager paging.
Runs on a single GPU with ≤8 GB VRAM (RTX 2060 Super class).
"""
import argparse, random, imageio
from pathlib import Path


import sys
from pathlib import Path
project_root = Path(__file__).parent
comfyui_root = Path("/home/beed1089/ComfyUI")
for path in [project_root, comfyui_root]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import torch, comfy.samplers
from pysrc.scheduler import SequentialScheduler
import page_table_ext as _pager
from models.wan_comfy import t5_skel, wan_skel, vae_skel
# Register Python kernels in C extension
#for name, builder in [("t5", t5_skel), ("transformer", wan_skel), ("vae", vae_skel)]:
#    _pager.register_kernel(name, lambda *inputs, b=builder: b().to("cuda")(*inputs))




# Import wan_comfy to register skeletons & MemoryManager
from models import wan_comfy as wc
MM = wc.MM
SCHED = SequentialScheduler(MM)  # scheduler instance
ROOT = Path(__file__).parent / "safetensors"
# Paging handled in scheduler: reserve/prefetch/evict per module







# --- tokenizer ---------------------------------------------------------
from transformers import T5Tokenizer

_tok = T5Tokenizer.from_pretrained("google/umt5-xxl")



def encode(prompt: str):
    ids = _tok(prompt, padding="max_length", truncation=True,
                max_length=256, return_tensors="pt").input_ids
    with SCHED.module("t5", ids.cuda(non_blocking=True)) as out:
        return out[0]

# -----------------------------------------------------------------------

def generate(prompt: str, negative: str, H: int, W: int, T: int,
             steps: int, cfg: float, seed: int):
    dtype = torch.float16
    if seed < 0:
        seed = random.randint(0, 2**31)

    cond  = encode(prompt)
    ncond = encode(negative) if negative else None

    lat = torch.randn(1, 4, T, H//8, W//8, device="cuda", dtype=dtype)

    sampler = comfy.samplers.KSampler(
        model=None,
        steps=steps,
        cfg_scale=cfg,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        seed=seed,
    )

    for s in sampler:
        lat = s(lat, cond, ncond)
        with SCHED.module("transformer", lat, cond, ncond) as out:
            lat = out

    with SCHED.module("vae", lat) as out:
        frames = out
    return frames.cpu()

# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--out", default="wan21.mp4")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--scan", action="store_true", help="List registered modules and exit")
    args = ap.parse_args()
    if args.scan:
        print(list(MM._tiers.keys()))
        return
   
    write_mm_logs()
    #vid = generate(args.prompt, args.negative, H=480, W=832, T=81,
    #               steps=args.steps, cfg=args.cfg, seed=args.seed)
    #imageio.mimsave(args.out, [f.permute(1,2,0).cpu().numpy() for f in vid], fps=16)
    print("done →", args.out)



# -----------------------------------------------------------------------
# Log helper
# -----------------------------------------------------------------------



def write_mm_logs():
    from pathlib import Path
    import pprint
    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for name in list(MM._tiers.keys()):
        tier = MM._tiers[name]
        try:
            if tier.get("offsets") is None:
                MM.acquire(name)
        except MemoryError as e:
            info = {"error": str(e)}
            (logs_dir / f"log_MM_{name}.log").write_text(repr(info))
            continue
        finally:
            try:
                MM.release(name)
            except Exception:
                pass
        info = {k: str(v) for k, v in tier.items()}
        info = {
            "um_ptr": tier.get("um_ptr"),
            "um_pages": tier.get("um_pages"),
            "gpu_loaded": tier["gpu"] is not None,
            "offsets": tier.get("offsets"),
            "sizes": tier.get("sizes"),
        }
        (logs_dir / f"log_MM_{name}.log").write_text(pprint.pformat(info))


if __name__ == "__main__":
    main()
