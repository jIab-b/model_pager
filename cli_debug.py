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
import page_table_ext as _pager
from models.wan_comfy import t5_skel, wan_skel, vae_skel
# --- tokenizer ---------------------------------------------------------
from transformers import T5Tokenizer
_tok = T5Tokenizer.from_pretrained("google/umt5-xxl")




# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

   
    #write_mm_logs()
    vid = generate(args.prompt, args.negative, H=480, W=832, T=81,
                   steps=args.steps, cfg=args.cfg, seed=args.seed)
    imageio.mimsave(args.out, [f.permute(1,2,0).cpu().numpy() for f in vid], fps=16)
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
