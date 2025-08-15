from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import page_table_ext as _pager
from .state import get_state

_DTYPE_MAP = {
    "F16": torch.float16,
    "FLOAT16": torch.float16,
    "BF16": torch.bfloat16,
    "BFLOAT16": torch.bfloat16,
    "F32": torch.float32,
    "FLOAT32": torch.float32,
}


class _PagedBase(torch.nn.Module):
    def __init__(self, device: torch.device, lut: Dict[str, Tuple[int, int]], dtypes: Dict[str, str], shapes: Dict[str, tuple]):
        super().__init__()
        self.device = device
        self.lut = lut
        self.dtypes = dtypes
        self.shapes = shapes

    def _load_weight(self, key: str) -> torch.Tensor:
        h, idx = self.lut[key]
        shape = self.shapes[key]
        dt_key = str(self.dtypes[key]).upper()
        dtype = _DTYPE_MAP.get(dt_key, torch.float16)
        w = torch.empty(shape, device=self.device, dtype=dtype)
        _pager.model_read_into(h, idx, w, None)
        return w


class PagedLinear(_PagedBase):
    def __init__(self, in_features: int, out_features: int, bias: bool, device: torch.device, lut, dtypes, shapes, weight_key: Optional[str] = None, bias_key: Optional[str] = None):
        super().__init__(device, lut, dtypes, shapes)
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight_key = weight_key
        self.bias_key = bias_key
        self._w_t = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._load_weight(self.weight_key)
        b = None
        if self.has_bias and self.bias_key in self.lut:
            b = self._load_weight(self.bias_key)
        try:
            y = F.linear(x, w if not self._w_t else w.t(), b)
        except RuntimeError:
            self._w_t = not self._w_t
            y = F.linear(x, w.t() if not self._w_t else w, b)
        return y


class PagedConvNd(_PagedBase):
    def __init__(self, dims: int, in_channels: int, out_channels: int, kernel_size, stride, padding, dilation, groups, bias, device, lut, dtypes, shapes, weight_key=None, bias_key=None):
        super().__init__(device, lut, dtypes, shapes)
        self.dims = dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        self.weight_key = weight_key
        self.bias_key = bias_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._load_weight(self.weight_key)
        b = None
        if self.has_bias and self.bias_key in self.lut:
            b = self._load_weight(self.bias_key)
        if self.dims == 1:
            return F.conv1d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
        if self.dims == 2:
            return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
        return F.conv3d(x, w, b, self.stride, self.padding, self.dilation, self.groups)


class PagedConvTransposeNd(PagedConvNd):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._load_weight(self.weight_key)
        b = None
        if self.has_bias and self.bias_key in self.lut:
            b = self._load_weight(self.bias_key)
        if self.dims == 1:
            return F.conv_transpose1d(x, w, b, self.stride, self.padding, None, self.groups, self.dilation)
        return F.conv_transpose2d(x, w, b, self.stride, self.padding, None, self.groups, self.dilation)


class PagedEmbedding(_PagedBase):
    def __init__(self, num_embeddings: int, embedding_dim: int, device, lut, dtypes, shapes, weight_key: str):
        super().__init__(device, lut, dtypes, shapes)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_key = weight_key

    def forward(self, input: torch.Tensor, out_dtype=None) -> torch.Tensor:
        uniq, inverse = torch.unique(input, sorted=True, return_inverse=True)
        st = get_state()
        meta = st["meta"] if st is not None else None
        dt_key = str(self.dtypes[self.weight_key]).upper()
        dtype = _DTYPE_MAP.get(dt_key, torch.float16)
        rows = torch.empty((uniq.numel(), self.embedding_dim), device=self.device, dtype=dtype)
        if meta is None:
            h, idx = self.lut[self.weight_key]
            dsts = [rows[i] for i in range(rows.shape[0])]
            for i in range(len(dsts)):
                _pager.model_read_into(h, idx, dsts[i], None)
        else:
            tf = meta["tensor_files"][self.weight_key]
            base = 8 + meta["header_lens"][tf]
            off0 = base + meta["offsets"][self.weight_key]
            elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
            row_bytes = int(self.embedding_dim) * elem_bytes
            file_offsets = [off0 + int(t.item()) * row_bytes for t in uniq.detach().to("cpu")]
            sizes = [row_bytes] * len(file_offsets)
            h_rows = _pager.model_register(tf, file_offsets, sizes)
            dsts = [rows[i] for i in range(rows.shape[0])]
            _pager.model_read_into_batch(h_rows, list(range(len(file_offsets))), dsts, None)
            _pager.wait_all()
            _pager.model_close(h_rows)
        out = F.embedding(inverse.view(input.shape), rows)
        if out_dtype is not None:
            out = out.to(out_dtype)
        return out


