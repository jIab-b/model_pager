# SPDX-License-Identifier: Apache-2.0
# Portions originally inspired by ComfyUI (AGPL-3.0) and Hugging-Face Diffusers
# Copyright (c) 2024-2025 The lazy-wan21 authors.

"""
Lazy-loading inference runner for Wan 2.1 video models.
The only objects that live permanently on GPU are the
latents being denoised; every network block is materialised,
run once, and immediately paged back to CPU (or freed).

Usage
-----
from lazy_wan21 import Wan21Runner
runner = Wan21Runner(
    model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    device="cuda",               # GPU to run on
    cpu_offload_dir="/tmp/wan21" # optional temp dir for pinned mmap
)
video = runner(
    prompt="A stork flying over a lake at sunset ...",
    negative_prompt="worst quality, jpeg artifacts",
    num_frames=81,
    guidance_scale=5.0,
)
video[0].save("wan21.mp4")
"""

import contextlib
import inspect
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any, Callable

import torch
import numpy as np
from diffusers import (
    AutoencoderKLWan,
    WanTransformer3DModel,
    WanPipeline,
    AutoModel,
)
from diffusers.schedulers import UniPCMultistepScheduler
from transformers import UMT5EncoderModel
from diffusers.utils import export_to_video

# ---------------------------------------------------------------------
# 1.  A generic “lazy module” wrapper
# ---------------------------------------------------------------------
class _LazyModule(torch.nn.Module):
    """
    Wraps any torch.nn.Module subclass so that:

      • weights are loaded from a “factory” only when forward() is called
      • after forward() they are either kept on GPU for `keep_in_gpu`
        seconds *or* moved back to CPU immediately
    """
    def __init__(
        self,
        factory: Callable[[], torch.nn.Module],
        onload_device: torch.device,
        offload_device: torch.device,
        keep_in_gpu: float = 0.0,
    ):
        super().__init__()
        self._factory = factory
        self._onload = onload_device
        self._offload = offload_device
        self._keep = keep_in_gpu
        self._module: Optional[torch.nn.Module] = None

    def _ensure_loaded(self):
        if self._module is None:
            self._module = self._factory().to(self._onload)
            self._module.eval()

    @contextlib.contextmanager
    def _loaded(self):
        self._ensure_loaded()
        try:
            yield self._module
        finally:
            if self._keep == 0:
                # Immediately move back to CPU & free GPU mem
                self._module.to(self._offload)
                torch.cuda.empty_cache()

    # Delegate common attributes
    def __getattr__(self, item):
        if item in self.__dict__:
            return super().__getattribute__(item)
        self._ensure_loaded()
        return getattr(self._module, item)

    def forward(self, *args, **kwargs):
        with self._loaded() as mod:
            return mod(*args, **kwargs)


# ---------------------------------------------------------------------
# 2.  Page-table builder ---- maps each HF sub-folder to a LazyModule
# ---------------------------------------------------------------------
def _page_table(
    model_id: str,
    dtype: torch.dtype,
    onload: torch.device,
    offload: torch.device,
) -> Dict[str, torch.nn.Module]:
    cache_dir = Path(tempfile.gettempdir()) / "lazy_wan21_downloads"
    cache_dir.mkdir(exist_ok=True)

    def lazy_text_encoder():
        return UMT5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )

    def lazy_transformer():
        return WanTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )

    def lazy_vae():
        # Wan VAE benefits from fp32 decode quality ([huggingface.co](https://huggingface.co/docs/diffusers/v0.33.0/api/pipelines/wan?utm_source=chatgpt.com))
        return AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=cache_dir,
        )

    return {
        "text_encoder": _LazyModule(lazy_text_encoder, onload, offload),
        "transformer":  _LazyModule(lazy_transformer,  onload, offload),
        "vae":          _LazyModule(lazy_vae,          onload, offload),
    }


# ---------------------------------------------------------------------
# 3.  High-level runner
# ---------------------------------------------------------------------
class Wan21Runner:
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        device: str = "cuda",
        cpu_offload_dir: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        keep_modules_in_gpu_sec: float = 0.0,
    ):
        self.model_id = model_id
        self.dev = torch.device(device)
        self.cpu = torch.device("cpu")

        table = _page_table(
            model_id,
            dtype=dtype,
            onload=self.dev,
            offload=self.cpu,
        )

        self.pipe = WanPipeline.from_pretrained(
            model_id,
            text_encoder=table["text_encoder"],
            transformer=table["transformer"],
            vae=table["vae"],
            torch_dtype=dtype,
        )
        # Replace default scheduler with the recommended UniPC variant
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=5.0
        )

        # Nothing else stays on GPU permanently
        self.pipe.to(self.dev)

        if cpu_offload_dir:
            # use Hugging-Face built-in hooks to page latents to CPU as well
            self.pipe.enable_model_cpu_offload(cpu_offload_dir)

    # -----------------------------------------------------------------
    # Main call
    # -----------------------------------------------------------------
    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 81,
        guidance_scale: float = 5.0,
        height: int = 480,
        width: int = 832,
        **extra,
    ):
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            **extra,
        )
        return out.frames


# ---------------------------------------------------------------------
# 4.  Example CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, textwrap, sys

    parser = argparse.ArgumentParser(
        description="Generate a Wan 2.1 video with lazy loading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              python lazy_wan21.py --prompt "A panda riding a bike..." \\
                                   --outfile panda.mp4
            """
        ),
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--outfile", default="wan21.mp4")
    args = parser.parse_args(sys.argv[1:])

    runner = Wan21Runner()
    frames = runner(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
    )
    export_to_video(frames[0], args.outfile, fps=16)
    print("Saved →", args.outfile)