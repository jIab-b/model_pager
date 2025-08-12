import torch, types
import torch.nn as nn

def _meta(op):
    def wrapper(*args, **kwargs):
        kwargs.setdefault("device", "meta")
        return op(*args, **kwargs)
    return wrapper

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True, device="meta", dtype=None):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(var + self.eps)
        if getattr(self, "weight", None) is not None:
            out = out * self.weight
        return out

def optimized_attention(*args, **kwargs):
    raise NotImplementedError("Fused attention kernel must be bound at runtime")

# build fallback namespace of ops
FALLBACK = types.SimpleNamespace(
    Linear=_meta(nn.Linear),
    Embedding=_meta(nn.Embedding),
    LayerNorm=_meta(nn.LayerNorm),
    GroupNorm=_meta(nn.GroupNorm),
    RMSNorm=RMSNorm,
    Conv1d=_meta(nn.Conv1d),
    Conv2d=_meta(nn.Conv2d),
    Conv3d=_meta(nn.Conv3d),
    ConvTranspose1d=_meta(nn.ConvTranspose1d),
    ConvTranspose2d=_meta(nn.ConvTranspose2d),
    ConvTranspose3d=_meta(nn.ConvTranspose3d),
    SiLU=lambda **kw: nn.SiLU(),
    optimized_attention=optimized_attention,
)

def apply(target_module):
    """Inject missing fallback ops into given module (e.g. comfy.ops)."""
    for name, obj in vars(FALLBACK).items():
        if not hasattr(target_module, name):
            setattr(target_module, name, obj)
