from __future__ import annotations
from contextlib import contextmanager
import types
import torch

from .state import get_state
from .ops_paged import PagedLinear, PagedConvNd, PagedConvTransposeNd, PagedEmbedding


def _build_ops_namespace():
    st = get_state()
    if st is None:
        raise RuntimeError("pager state not initialized")
    device = st["device"]
    lut = st["lut"]
    dtypes = st["meta"]["dtypes"]
    shapes = st["meta"]["shapes"]

    class Ops:
        class Linear(torch.nn.Module):
            def __init__(self, in_features, out_features, bias=False, dtype=None, device=None):
                super().__init__()
                self.impl = PagedLinear(in_features, out_features, bias, device or torch.device("cuda"), lut, dtypes, shapes)
            def forward(self, x):
                return self.impl(x)

        class Conv1d(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, dtype=None, device=None):
                super().__init__()
                self.impl = PagedConvNd(1, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, device or torch.device("cuda"), lut, dtypes, shapes)
            def forward(self, x):
                return self.impl(x)

        class Conv2d(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, dtype=None, device=None):
                super().__init__()
                self.impl = PagedConvNd(2, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, device or torch.device("cuda"), lut, dtypes, shapes)
            def forward(self, x):
                return self.impl(x)

        class Conv3d(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, dtype=None, device=None):
                super().__init__()
                self.impl = PagedConvNd(3, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, device or torch.device("cuda"), lut, dtypes, shapes)
            def forward(self, x):
                return self.impl(x)

        class ConvTranspose1d(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False, dilation=1, dtype=None, device=None):
                super().__init__()
                self.impl = PagedConvTransposeNd(1, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, device or torch.device("cuda"), lut, dtypes, shapes)
            def forward(self, x):
                return self.impl(x)

        class ConvTranspose2d(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False, dilation=1, dtype=None, device=None):
                super().__init__()
                self.impl = PagedConvTransposeNd(2, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, device or torch.device("cuda"), lut, dtypes, shapes)
            def forward(self, x):
                return self.impl(x)

        class Embedding(torch.nn.Module):
            def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
                super().__init__()
                self.impl = None
                self.num_embeddings = num_embeddings
                self.embedding_dim = embedding_dim
                self._device = device or torch.device("cuda")
                self._lut = lut
                self._dtypes = dtypes
                self._shapes = shapes
                self._weight_key = None
            def forward(self, x, out_dtype=None):
                if self.impl is None:
                    if self._weight_key is None:
                        raise RuntimeError("embedding weight_key not bound")
                    self.impl = PagedEmbedding(self.num_embeddings, self.embedding_dim, self._device, self._lut, self._dtypes, self._shapes, self._weight_key)
                return self.impl(x, out_dtype=out_dtype)

    return Ops


@contextmanager
def use_comfy_ops():
    try:
        import comfy_ops as comfy_ops_mod
    except Exception as e:
        raise RuntimeError("comfy_ops module not found") from e
    orig_pick = getattr(comfy_ops_mod, "pick_operations", None)
    Ops = _build_ops_namespace()
    def pick_operations(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False, fp8_optimizations=False, scaled_fp8=None):
        return Ops
    setattr(comfy_ops_mod, "pick_operations", pick_operations)
    try:
        yield
    finally:
        if orig_pick is not None:
            setattr(comfy_ops_mod, "pick_operations", orig_pick)


def bind_weight_keys(model, weight_map=None):
    st = get_state()
    if st is None:
        raise RuntimeError("pager state not initialized")
    lut = st["lut"]
    def resolve_key(mod_path: str, kind: str):
        key = f"{mod_path}.{kind}"
        if weight_map is not None and key in weight_map:
            return key
        return key
    for name, module in model.named_modules():
        if hasattr(module, "impl") and isinstance(module.impl, (PagedLinear, PagedConvNd, PagedConvTransposeNd)):
            wkey = resolve_key(name, "weight")
            if wkey in lut:
                module.impl.weight_key = wkey
            bkey = resolve_key(name, "bias")
            if bkey in lut:
                module.impl.bias_key = bkey
        elif hasattr(module, "impl") and isinstance(module.impl, PagedEmbedding):
            wkey = resolve_key(name, "weight")
            if wkey in lut:
                module.impl.weight_key = wkey
        else:
            if hasattr(module, "_weight_key"):
                wkey = resolve_key(name, "weight")
                if wkey in lut:
                    setattr(module, "_weight_key", wkey)


