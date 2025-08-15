from __future__ import annotations
import torch
from typing import Dict, Tuple
from .ops_paged import PagedLinear, PagedConvNd, PagedConvTransposeNd, PagedEmbedding


def rewrite_model(model: torch.nn.Module, lut: Dict[str, Tuple[int, int]], dtypes: Dict[str, str], shapes: Dict[str, tuple]):
    device = next((p.device for p in model.parameters(recurse=True)), torch.device("cuda"))
    for name, module in list(model.named_modules()):
        parent = _get_parent(model, name)
        if parent is None:
            continue
        child_name = name.split(".")[-1]
        if isinstance(module, torch.nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            bias = module.bias is not None
            repl = PagedLinear(in_f, out_f, bias, device, lut, dtypes, shapes, f"{name}.weight", f"{name}.bias")
            setattr(parent, child_name, repl)
        elif isinstance(module, torch.nn.Conv1d):
            repl = PagedConvNd(1, module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, device, lut, dtypes, shapes, f"{name}.weight", f"{name}.bias")
            setattr(parent, child_name, repl)
        elif isinstance(module, torch.nn.Conv2d):
            repl = PagedConvNd(2, module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, device, lut, dtypes, shapes, f"{name}.weight", f"{name}.bias")
            setattr(parent, child_name, repl)
        elif isinstance(module, torch.nn.Conv3d):
            repl = PagedConvNd(3, module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, device, lut, dtypes, shapes, f"{name}.weight", f"{name}.bias")
            setattr(parent, child_name, repl)
        elif isinstance(module, torch.nn.ConvTranspose1d):
            repl = PagedConvTransposeNd(1, module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, device, lut, dtypes, shapes, f"{name}.weight", f"{name}.bias")
            setattr(parent, child_name, repl)
        elif isinstance(module, torch.nn.ConvTranspose2d):
            repl = PagedConvTransposeNd(2, module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, device, lut, dtypes, shapes, f"{name}.weight", f"{name}.bias")
            setattr(parent, child_name, repl)
        elif isinstance(module, torch.nn.Embedding):
            repl = PagedEmbedding(module.num_embeddings, module.embedding_dim, device, lut, dtypes, shapes, f"{name}.weight")
            setattr(parent, child_name, repl)


def _get_parent(root: torch.nn.Module, path: str):
    parts = path.split(".")
    if len(parts) == 1:
        return None
    cur = root
    for p in parts[:-1]:
        cur = getattr(cur, p)
    return cur


