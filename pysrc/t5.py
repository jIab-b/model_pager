import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
comfyui_root = Path("/home/beed1089/ComfyUI")
for p in [project_root, comfyui_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import re
import page_table_ext as _pager
from transformers import T5Tokenizer
from models.wan_comfy import t5_skel, MM

_DTYPE_MAP = {
    "F16": torch.float16,
    "FLOAT16": torch.float16,
    "BF16": torch.bfloat16,
    "BFLOAT16": torch.bfloat16,
    "F32": torch.float32,
    "FLOAT32": torch.float32,
}

def _build_registry():
    tier = MM._tiers["t5"]
    offmap = tier["offsets"]
    szmap = tier["sizes"]
    names = sorted(offmap.keys(), key=lambda k: offmap[k])
    # safetensors offsets are relative to data section; add header base (8 + header_len)
    path = tier["path"]
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
    base = 8 + header_len
    file_offsets = [base + offmap[n] for n in names]
    sizes = [szmap[n] for n in names]
    h = _pager.model_register(path, file_offsets, sizes)
    return h, names

def _alloc_for(model, names, device, dtypes_map):
    shapes = dict(model.named_parameters())
    out = {}
    for n in names:
        dt_key = str(dtypes_map.get(n, "F16")).upper()
        dt = _DTYPE_MAP.get(dt_key, torch.float16)
        out[n] = torch.empty(tuple(shapes[n].shape), device=device, dtype=dt)
    return out

def _page_into(h, needed, all_names, dst):
    idx = {n: i for i, n in enumerate(all_names)}
    indices = [idx[n] for n in needed]
    dsts = [dst[n] for n in needed]
    _pager.model_read_into_batch(h, indices, dsts, None)

def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    s = xf.pow(2).mean(dim=-1, keepdim=True)
    xn = xf * torch.rsqrt(s + eps)
    return xn.to(dtype=x.dtype) * w.to(dtype=x.dtype)

def _rel_pos_bucket(q_len: int, num_buckets: int = 32, max_distance: int = 128, device: torch.device | None = None) -> torch.Tensor:
    ctx = torch.arange(q_len, device=device)
    mem = torch.arange(q_len, device=device)
    rel = mem[None, :] - ctx[:, None]
    n = num_buckets
    half = n // 2
    pos = (rel > 0).to(torch.long) * half
    rel = rel.abs()
    max_exact = n - half
    is_small = rel < max_exact
    val_if_large = half + (
        (torch.log(rel.float() / max_exact + 1e-6) / torch.log(torch.tensor(max_distance / max_exact, device=device, dtype=torch.float32) + 1e-6))
        * (n - max_exact)
    ).to(torch.long)
    val_if_large = torch.clamp(val_if_large, max=max_exact + (n - max_exact) - 1)
    buckets = torch.where(is_small, rel, val_if_large) + pos
    return buckets

def _unique_blocks(all_names):
    s = set()
    for n in all_names:
        m = re.search(r"encoder\.block\.(\d+)\.", n)
        if m:
            s.add(int(m.group(1)))
    return sorted(s)

def _unique_blocks_decoder(all_names):
    s = set()
    for n in all_names:
        m = re.search(r"decoder\.block\.(\d+)\.", n)
        if m:
            s.add(int(m.group(1)))
    return sorted(s)

def _rowwise_embed(ids: torch.Tensor, model, device, tier):
    offmap, dtypes = tier["offsets"], tier["dtypes"]
    with open(tier["path"], "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
    base = 8 + header_len
    shapes = dict(model.named_parameters())
    d_model = shapes["shared.weight"].shape[1]
    uniq, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    uniq_cpu = uniq.detach().to("cpu")
    dtype_key = str(dtypes["shared.weight"]).upper()
    row_elem_bytes = 2 if dtype_key in ("F16", "FLOAT16", "BF16", "BFLOAT16") else 4
    row_bytes = int(d_model) * row_elem_bytes
    file_offsets = [base + offmap["shared.weight"] + int(tid.item()) * row_bytes for tid in uniq_cpu]
    sizes = [row_bytes] * len(file_offsets)
    h_rows = _pager.model_register(tier["path"], file_offsets, sizes)
    W_rows = torch.empty((len(file_offsets), int(d_model)), device=device, dtype=_DTYPE_MAP.get(dtype_key, torch.float16))
    dsts = [W_rows[i] for i in range(W_rows.shape[0])]
    _pager.model_read_into_batch(h_rows, list(range(len(file_offsets))), dsts, None)
    _pager.wait_all()
    _pager.model_close(h_rows)
    x = torch.nn.functional.embedding(inverse.view(ids.shape), W_rows).to(torch.float16)
    return x

def _greedy_next_token(last_hidden: torch.Tensor, model, tier, device, chunk_rows: int = 8192, debug: bool = False) -> torch.Tensor:
    shapes = dict(model.named_parameters())
    vocab_size = int(shapes["shared.weight"].shape[0])
    d_model = int(shapes["shared.weight"].shape[1])
    offmap, dtypes = tier["offsets"], tier["dtypes"]
    with open(tier["path"], "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
    base = 8 + header_len
    dtype_key = str(dtypes["shared.weight"]).upper()
    row_elem_bytes = 2 if dtype_key in ("F16", "FLOAT16", "BF16", "BFLOAT16") else 4
    row_bytes = d_model * row_elem_bytes
    B = int(last_hidden.shape[0])
    best_scores = torch.full((B,), -float("inf"), device=device, dtype=last_hidden.dtype)
    best_idx = torch.zeros((B,), device=device, dtype=torch.long)
    num_chunks = (vocab_size + chunk_rows - 1) // chunk_rows
    for c in range(num_chunks):
        start = c * chunk_rows
        rows = min(chunk_rows, vocab_size - start)
        offset = base + offmap["shared.weight"] + start * row_bytes
        handle = _pager.model_register(tier["path"], [offset], [rows * row_bytes])
        W_chunk = torch.empty((rows, d_model), device=device, dtype=_DTYPE_MAP.get(dtype_key, torch.float16))
        _pager.model_read_into(handle, 0, W_chunk, None)
        _pager.wait_all()
        _pager.model_close(handle)
        logits_chunk = torch.matmul(last_hidden, W_chunk.t().to(dtype=last_hidden.dtype))
        vals, idx = torch.max(logits_chunk, dim=1)
        mask = vals > best_scores
        best_scores = torch.where(mask, vals, best_scores)
        best_idx = torch.where(mask, start + idx, best_idx)
        del W_chunk, logits_chunk
        torch.cuda.empty_cache()
    if debug:
        print(f"[DBG] greedy scores min/max={best_scores.min().item():.3f}/{best_scores.max().item():.3f}")
    return best_idx

_W_T = {}

def _linear(x: torch.Tensor, w: torch.Tensor, key: str) -> torch.Tensor:
    w_use = w.to(dtype=x.dtype)
    if _W_T.get(key, False):
        return torch.nn.functional.linear(x, w_use.t())
    try:
        y = torch.nn.functional.linear(x, w_use)
        _W_T[key] = False
        return y
    except RuntimeError:
        y = torch.nn.functional.linear(x, w_use.t())
        _W_T[key] = True
        return y

def _embedding(ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    try:
        return torch.nn.functional.embedding(ids, w)
    except RuntimeError:
        return torch.nn.functional.embedding(ids, w.t())

def _peek_shared_row(path: str, base: int, off: int, d_model: int, row_idx: int, cols: int = 8, dtype: str = "F16"):
    import os
    elem_size = 2 if dtype.upper() in ("F16", "HALF", "FLOAT16") else 4
    row_off = base + off + row_idx * d_model * elem_size
    with open(path, "rb", buffering=0) as f:
        f.seek(row_off)
        buf = f.read(cols * elem_size)
    import numpy as np
    if elem_size == 2:
        arr = np.frombuffer(buf, dtype=np.float16)
        return arr.astype(np.float32)
    else:
        return np.frombuffer(buf, dtype=np.float32)

def encode(prompt: str, max_length: int = 256, out_path: str | None = None, debug: bool = False):
    _pager.runtime_init(0, 2, 1)
    h, all_names = _build_registry()
    tok = T5Tokenizer.from_pretrained("google/umt5-xxl")
    ids = tok(prompt, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").input_ids.cuda(non_blocking=True)
    model = t5_skel()
    dtype = torch.float16
    device = torch.device("cuda")
    dtypes_map = MM._tiers["t5"]["dtypes"]
    tier = MM._tiers["t5"]
    offmap, dtypes = tier["offsets"], tier["dtypes"]
    with open(tier["path"], "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
    base = 8 + header_len
    shapes = dict(model.named_parameters())
    d_model = shapes["shared.weight"].shape[1]
    pad_id = 0 if T5Tokenizer.from_pretrained("google/umt5-xxl").pad_token_id is None else T5Tokenizer.from_pretrained("google/umt5-xxl").pad_token_id
    uniq, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    uniq_cpu = uniq.detach().to("cpu")
    dtype_key = str(dtypes["shared.weight"]).upper()
    row_elem_bytes = 2 if dtype_key in ("F16", "FLOAT16", "BF16", "BFLOAT16") else 4
    row_bytes = int(d_model) * row_elem_bytes
    file_offsets = [base + offmap["shared.weight"] + int(tid.item()) * row_bytes for tid in uniq_cpu]
    sizes = [row_bytes] * len(file_offsets)
    h_rows = _pager.model_register(tier["path"], file_offsets, sizes)
    W_rows = torch.empty((len(file_offsets), int(d_model)), device=device, dtype=_DTYPE_MAP.get(dtype_key, torch.float16))
    dsts = [W_rows[i] for i in range(W_rows.shape[0])]
    _pager.model_read_into_batch(h_rows, list(range(len(file_offsets))), dsts, None)
    _pager.wait_all()
    _pager.model_close(h_rows)
    x = torch.nn.functional.embedding(inverse.view(ids.shape), W_rows).to(dtype)
    if debug:
        print(f"[DBG] after embedding: sum={x.float().abs().sum().item():.3f} mean={x.float().abs().mean().item():.6f} dtype={x.dtype} device={x.device}")
    blocks = _unique_blocks(all_names)
    print(f"T5 encoder blocks detected: {len(blocks)}")
    import time
    B, S = int(x.shape[0]), int(x.shape[1])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    key_mask = (ids != pad_id).to(dtype=x.dtype).view(B, 1, 1, S)
    for i, b in enumerate(blocks, 1):
        print(f"[T5] block {i}/{len(blocks)}: paging weights…")
        qn = f"encoder.block.{b}.layer.0.SelfAttention.q.weight"
        kn = f"encoder.block.{b}.layer.0.SelfAttention.k.weight"
        vn = f"encoder.block.{b}.layer.0.SelfAttention.v.weight"
        on = f"encoder.block.{b}.layer.0.SelfAttention.o.weight"
        ln1 = f"encoder.block.{b}.layer.0.layer_norm.weight"
        rpb = f"encoder.block.{b}.layer.0.SelfAttention.relative_attention_bias.weight"
        wi0 = f"encoder.block.{b}.layer.1.DenseReluDense.wi_0.weight"
        wi1 = f"encoder.block.{b}.layer.1.DenseReluDense.wi_1.weight"
        wo = f"encoder.block.{b}.layer.1.DenseReluDense.wo.weight"
        ln2 = f"encoder.block.{b}.layer.1.layer_norm.weight"
        need = [qn, kn, vn, on, ln1, rpb, wi0, wi1, wo, ln2]
        print(f"[T5] block {i}/{len(blocks)}: compute…")
        t0 = time.time()
        ln1_w = _alloc_for(model, [ln1], device, dtypes_map)[ln1]
        _page_into(h, [ln1], all_names, {ln1: ln1_w})
        _pager.wait_all()
        xn = _rmsnorm(x, ln1_w)
        if debug:
            print(f"[DBG] xn mean={xn.float().abs().mean().item():.6f}")
        del ln1_w; torch.cuda.empty_cache()
        q_w = _alloc_for(model, [qn], device, dtypes_map)[qn]
        _page_into(h, [qn], all_names, {qn: q_w}); _pager.wait_all()
        q = _linear(xn, q_w, qn)
        del q_w; torch.cuda.empty_cache()
        k_w = _alloc_for(model, [kn], device, dtypes_map)[kn]
        _page_into(h, [kn], all_names, {kn: k_w}); _pager.wait_all()
        k = _linear(xn, k_w, kn)
        del k_w; torch.cuda.empty_cache()
        v_w = _alloc_for(model, [vn], device, dtypes_map)[vn]
        _page_into(h, [vn], all_names, {vn: v_w}); _pager.wait_all()
        v = _linear(xn, v_w, vn)
        del v_w; torch.cuda.empty_cache()
        if debug:
            print(f"[DBG] q/k/v mean={q.float().abs().mean().item():.6f}/{k.float().abs().mean().item():.6f}/{v.float().abs().mean().item():.6f}")
        # infer heads from rel-pos table dims (pick the axis that divides d_model and is larger)
        rpb_w = _alloc_for(model, [rpb], device, dtypes_map)[rpb]
        _page_into(h, [rpb], all_names, {rpb: rpb_w}); _pager.wait_all()
        if rpb_w.dim() == 2:
            a, b = rpb_w.shape
            cand = [x for x in (a, b) if d_model % x == 0]
            num_heads = max(cand) if cand else max(a, b)
        else:
            num_heads = (d_model // 64) if d_model % 64 == 0 else (d_model // 32)
        head_dim = q.shape[-1] // num_heads
        B, S, _ = x.shape
        q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, S, num_heads, head_dim).transpose(1, 2)
        # ensure rpb_table has shape [num_buckets, num_heads]
        if rpb_w.dim() == 2:
            a, b = rpb_w.shape
            if b == num_heads:
                rpb_table = rpb_w
                num_buckets = a
            elif a == num_heads:
                rpb_table = rpb_w.t()
                num_buckets = b
            else:
                # fallback: assume larger dim is heads
                if a > b:
                    rpb_table = rpb_w.t()
                    num_buckets = b
                else:
                    rpb_table = rpb_w
                    num_buckets = a
        else:
            # unexpected; default buckets
            num_buckets = 32
            rpb_table = rpb_w
        buckets = _rel_pos_bucket(S, num_buckets=num_buckets, max_distance=128, device=device)
        buckets = buckets.clamp_(min=0, max=num_buckets - 1)
        if debug:
            print(f"[DBG] rpb table shape={tuple(rpb_table.shape)} buckets min/max={buckets.min().item()}/{buckets.max().item()} num_heads={num_heads}")
        bias = rpb_table[buckets]
        bias = bias.permute(2, 0, 1).unsqueeze(0).to(q.dtype)
        mask_bias = (1.0 - key_mask) * (-1e9)
        bias = bias + mask_bias
        del rpb_w; torch.cuda.empty_cache()
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, S, d_model)
        on_w = _alloc_for(model, [on], device, dtypes_map)[on]
        _page_into(h, [on], all_names, {on: on_w}); _pager.wait_all()
        attn = _linear(attn, on_w, on)
        del on_w; torch.cuda.empty_cache()
        if debug:
            print(f"[DBG] attn_out sum={attn.float().abs().sum().item():.3f} mean={attn.float().abs().mean().item():.6f}")
        x = x + attn
        if debug:
            print(f"[DBG] after attn residual: x.sum={x.float().abs().sum().item():.3f}")
        ln2_w = _alloc_for(model, [ln2], device, dtypes_map)[ln2]
        _page_into(h, [ln2], all_names, {ln2: ln2_w}); _pager.wait_all()
        yn = _rmsnorm(x, ln2_w)
        del ln2_w; torch.cuda.empty_cache()
        wi0_w = _alloc_for(model, [wi0], device, dtypes_map)[wi0]
        _page_into(h, [wi0], all_names, {wi0: wi0_w}); _pager.wait_all()
        u0 = _linear(yn, wi0_w, wi0)
        del wi0_w; torch.cuda.empty_cache()
        wi1_w = _alloc_for(model, [wi1], device, dtypes_map)[wi1]
        _page_into(h, [wi1], all_names, {wi1: wi1_w}); _pager.wait_all()
        u1 = _linear(yn, wi1_w, wi1)
        del wi1_w; torch.cuda.empty_cache()
        g = torch.nn.functional.gelu(u0, approximate="tanh") * u1
        wo_w = _alloc_for(model, [wo], device, dtypes_map)[wo]
        _page_into(h, [wo], all_names, {wo: wo_w}); _pager.wait_all()
        ffn = _linear(g, wo_w, wo)
        del wo_w; torch.cuda.empty_cache()
        if debug:
            print(f"[DBG] u0/u1 mean={u0.float().abs().mean().item():.6f}/{u1.float().abs().mean().item():.6f} g mean={g.float().abs().mean().item():.6f} ffn_out sum={ffn.float().abs().sum().item():.3f}")
        x = x + ffn
        torch.cuda.empty_cache()
        if debug:
            torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000
            print(f"[DBG] block {i} time={dt:.1f} ms x.sum={x.float().abs().sum().item():.3f}")
        print(f"[T5] block {i}/{len(blocks)}: done")
    fin = _alloc_for(model, ["encoder.final_layer_norm.weight"], device, dtypes_map)
    _page_into(h, ["encoder.final_layer_norm.weight"], all_names, fin)
    x = _rmsnorm(x, fin["encoder.final_layer_norm.weight"])
    del fin
    torch.cuda.empty_cache()
    _pager.wait_all()
    _pager.model_close(h)
    _pager.runtime_shutdown()
    if out_path:
        out_cpu = x.detach().to("cpu")
        out_cpu_path = Path(out_path)
        out_cpu_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_cpu, out_cpu_path)
    print(f"[T5] output shape: {tuple(x.shape)} sum={x.float().abs().sum().item():.3f} mean={x.float().mean().item():.6f}")
    return x

def decode(decoder_ids: torch.Tensor, encoder_hidden: torch.Tensor, tok: T5Tokenizer, h: int, all_names: list[str], model, device, dtypes_map, debug: bool = False):
    tier = MM._tiers["t5"]
    if debug:
        print(f"[DBG] decode: ids shape={tuple(decoder_ids.shape)}")
    y = _rowwise_embed(decoder_ids, model, device, tier)
    B, S_dec, d_model = int(y.shape[0]), int(y.shape[1]), int(y.shape[2])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    key_mask_dec = (decoder_ids != pad_id).to(dtype=y.dtype).view(B, 1, 1, S_dec)
    key_mask_enc = (torch.ones(encoder_hidden.shape[:2], device=device, dtype=y.dtype)).view(B, 1, 1, encoder_hidden.shape[1])
    dblocks = _unique_blocks_decoder(all_names)
    if debug:
        print(f"[DBG] decoder blocks={len(dblocks)}")
    for i, b in enumerate(dblocks, 1):
        if debug:
            print(f"[DBG][dec] block {i}/{len(dblocks)}")
        ln1 = f"decoder.block.{b}.layer.0.layer_norm.weight"
        qn = f"decoder.block.{b}.layer.0.SelfAttention.q.weight"
        kn = f"decoder.block.{b}.layer.0.SelfAttention.k.weight"
        vn = f"decoder.block.{b}.layer.0.SelfAttention.v.weight"
        on = f"decoder.block.{b}.layer.0.SelfAttention.o.weight"
        rpb = f"decoder.block.{b}.layer.0.SelfAttention.relative_attention_bias.weight"
        ln_x = f"decoder.block.{b}.layer.1.layer_norm.weight"
        q_x = f"decoder.block.{b}.layer.1.EncDecAttention.q.weight"
        k_x = f"decoder.block.{b}.layer.1.EncDecAttention.k.weight"
        v_x = f"decoder.block.{b}.layer.1.EncDecAttention.v.weight"
        o_x = f"decoder.block.{b}.layer.1.EncDecAttention.o.weight"
        ln2 = f"decoder.block.{b}.layer.2.layer_norm.weight"
        wi0 = f"decoder.block.{b}.layer.2.DenseReluDense.wi_0.weight"
        wi1 = f"decoder.block.{b}.layer.2.DenseReluDense.wi_1.weight"
        wo = f"decoder.block.{b}.layer.2.DenseReluDense.wo.weight"
        ln1_w = _alloc_for(model, [ln1], device, dtypes_map)[ln1]
        _page_into(h, [ln1], all_names, {ln1: ln1_w}); _pager.wait_all()
        yn = _rmsnorm(y, ln1_w)
        del ln1_w; torch.cuda.empty_cache()
        q_w = _alloc_for(model, [qn], device, dtypes_map)[qn]
        _page_into(h, [qn], all_names, {qn: q_w}); _pager.wait_all()
        q = _linear(yn, q_w, qn)
        del q_w; torch.cuda.empty_cache()
        k_w = _alloc_for(model, [kn], device, dtypes_map)[kn]
        _page_into(h, [kn], all_names, {kn: k_w}); _pager.wait_all()
        k = _linear(yn, k_w, kn)
        del k_w; torch.cuda.empty_cache()
        v_w = _alloc_for(model, [vn], device, dtypes_map)[vn]
        _page_into(h, [vn], all_names, {vn: v_w}); _pager.wait_all()
        v = _linear(yn, v_w, vn)
        del v_w; torch.cuda.empty_cache()
        rpb_w = _alloc_for(model, [rpb], device, dtypes_map)[rpb]
        _page_into(h, [rpb], all_names, {rpb: rpb_w}); _pager.wait_all()
        if rpb_w.dim() == 2:
            a, b2 = rpb_w.shape
            cand = [x for x in (a, b2) if d_model % x == 0]
            num_heads = max(cand) if cand else max(a, b2)
        else:
            num_heads = (d_model // 64) if d_model % 64 == 0 else (d_model // 32)
        head_dim = q.shape[-1] // num_heads
        q = q.view(B, S_dec, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, S_dec, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, S_dec, num_heads, head_dim).transpose(1, 2)
        if rpb_w.dim() == 2:
            a, b2 = rpb_w.shape
            if b2 == num_heads:
                rpb_table = rpb_w; num_buckets = a
            elif a == num_heads:
                rpb_table = rpb_w.t(); num_buckets = b2
            else:
                if a > b2:
                    rpb_table = rpb_w.t(); num_buckets = b2
                else:
                    rpb_table = rpb_w; num_buckets = a
        else:
            num_buckets = 32; rpb_table = rpb_w
        buckets = _rel_pos_bucket(S_dec, num_buckets=num_buckets, max_distance=128, device=device)
        buckets = buckets.clamp_(min=0, max=num_buckets - 1)
        bias = rpb_table[buckets]
        bias = bias.permute(2, 0, 1).unsqueeze(0).to(q.dtype)
        mask_bias = (1.0 - key_mask_dec) * (-1e9)
        causal = torch.triu(torch.ones(S_dec, S_dec, device=device, dtype=q.dtype), diagonal=1) * (-1e9)
        bias = bias + mask_bias + causal
        del rpb_w; torch.cuda.empty_cache()
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, S_dec, d_model)
        on_w = _alloc_for(model, [on], device, dtypes_map)[on]
        _page_into(h, [on], all_names, {on: on_w}); _pager.wait_all()
        attn = _linear(attn, on_w, on)
        del on_w; torch.cuda.empty_cache()
        y = y + attn
        ln_x_w = _alloc_for(model, [ln_x], device, dtypes_map)[ln_x]
        _page_into(h, [ln_x], all_names, {ln_x: ln_x_w}); _pager.wait_all()
        yz = _rmsnorm(y, ln_x_w)
        del ln_x_w; torch.cuda.empty_cache()
        qx_w = _alloc_for(model, [q_x], device, dtypes_map)[q_x]
        _page_into(h, [q_x], all_names, {q_x: qx_w}); _pager.wait_all()
        qx = _linear(yz, qx_w, q_x)
        del qx_w; torch.cuda.empty_cache()
        kx_w = _alloc_for(model, [k_x], device, dtypes_map)[k_x]
        _page_into(h, [k_x], all_names, {k_x: kx_w}); _pager.wait_all()
        kx = _linear(encoder_hidden, kx_w, k_x)
        del kx_w; torch.cuda.empty_cache()
        vx_w = _alloc_for(model, [v_x], device, dtypes_map)[v_x]
        _page_into(h, [v_x], all_names, {v_x: vx_w}); _pager.wait_all()
        vx = _linear(encoder_hidden, vx_w, v_x)
        del vx_w; torch.cuda.empty_cache()
        qx = qx.view(B, S_dec, num_heads, head_dim).transpose(1, 2)
        kx = kx.view(B, encoder_hidden.shape[1], num_heads, head_dim).transpose(1, 2)
        vx = vx.view(B, encoder_hidden.shape[1], num_heads, head_dim).transpose(1, 2)
        mem_bias = (1.0 - key_mask_enc) * (-1e9)
        attn2 = torch.nn.functional.scaled_dot_product_attention(qx, kx, vx, attn_mask=mem_bias, dropout_p=0.0, is_causal=False)
        attn2 = attn2.transpose(1, 2).contiguous().view(B, S_dec, d_model)
        ox_w = _alloc_for(model, [o_x], device, dtypes_map)[o_x]
        _page_into(h, [o_x], all_names, {o_x: ox_w}); _pager.wait_all()
        attn2 = _linear(attn2, ox_w, o_x)
        del ox_w; torch.cuda.empty_cache()
        y = y + attn2
        ln2_w = _alloc_for(model, [ln2], device, dtypes_map)[ln2]
        _page_into(h, [ln2], all_names, {ln2: ln2_w}); _pager.wait_all()
        yn2 = _rmsnorm(y, ln2_w)
        del ln2_w; torch.cuda.empty_cache()
        wi0_w = _alloc_for(model, [wi0], device, dtypes_map)[wi0]
        _page_into(h, [wi0], all_names, {wi0: wi0_w}); _pager.wait_all()
        u0 = _linear(yn2, wi0_w, wi0)
        del wi0_w; torch.cuda.empty_cache()
        wi1_w = _alloc_for(model, [wi1], device, dtypes_map)[wi1]
        _page_into(h, [wi1], all_names, {wi1: wi1_w}); _pager.wait_all()
        u1 = _linear(yn2, wi1_w, wi1)
        del wi1_w; torch.cuda.empty_cache()
        g = torch.nn.functional.gelu(u0, approximate="tanh") * u1
        wo_w = _alloc_for(model, [wo], device, dtypes_map)[wo]
        _page_into(h, [wo], all_names, {wo: wo_w}); _pager.wait_all()
        ffn = _linear(g, wo_w, wo)
        del wo_w; torch.cuda.empty_cache()
        y = y + ffn
    fn = "decoder.final_layer_norm.weight"
    fin = _alloc_for(model, [fn], device, dtypes_map)[fn]
    _page_into(h, [fn], all_names, {fn: fin}); _pager.wait_all()
    y = _rmsnorm(y, fin)
    del fin; torch.cuda.empty_cache()
    return y

def _prepare_cross_kv(encoder_hidden: torch.Tensor, h: int, all_names: list[str], model, device, dtypes_map, debug: bool = False):
    B, S_enc, d_model = int(encoder_hidden.shape[0]), int(encoder_hidden.shape[1]), int(encoder_hidden.shape[2])
    blocks = _unique_blocks_decoder(all_names)
    cross = {}
    for i, b in enumerate(blocks, 1):
        q_x = f"decoder.block.{b}.layer.1.EncDecAttention.q.weight"
        k_x = f"decoder.block.{b}.layer.1.EncDecAttention.k.weight"
        v_x = f"decoder.block.{b}.layer.1.EncDecAttention.v.weight"
        # infer heads from q projection size
        qx_w = _alloc_for(model, [q_x], device, dtypes_map)[q_x]
        _page_into(h, [q_x], all_names, {q_x: qx_w}); _pager.wait_all()
        inner = int(qx_w.shape[0])
        num_heads = (d_model // 64) if d_model % 64 == 0 else (d_model // 32)
        head_dim = inner // num_heads
        del qx_w; torch.cuda.empty_cache()
        kx_w = _alloc_for(model, [k_x], device, dtypes_map)[k_x]
        _page_into(h, [k_x], all_names, {k_x: kx_w}); _pager.wait_all()
        vx_w = _alloc_for(model, [v_x], device, dtypes_map)[v_x]
        _page_into(h, [v_x], all_names, {v_x: vx_w}); _pager.wait_all()
        kx = _linear(encoder_hidden, kx_w, k_x)
        vx = _linear(encoder_hidden, vx_w, v_x)
        del kx_w, vx_w; torch.cuda.empty_cache()
        kx = kx.view(B, S_enc, num_heads, head_dim).transpose(1, 2).contiguous()
        vx = vx.view(B, S_enc, num_heads, head_dim).transpose(1, 2).contiguous()
        cross[b] = (kx, vx, num_heads, head_dim)
        if debug:
            print(f"[DBG] cross KV block {i}: kx/vx shapes={tuple(kx.shape)}/{tuple(vx.shape)}")
    return cross

def generate(prompt: str, max_length: int = 256, max_new_tokens: int = 16, debug: bool = False):
    _pager.runtime_init(0, 2, 1)
    h, all_names = _build_registry()
    tok = T5Tokenizer.from_pretrained("google/umt5-xxl")
    ids = tok(prompt, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").input_ids.cuda(non_blocking=True)
    model = t5_skel()
    device = torch.device("cuda")
    # encoder forward (already low-VRAM streaming)
    enc_hidden = encode(prompt, max_length=max_length, out_path=None, debug=debug)
    # precompute cross-attn K/V for encoder per decoder block
    dtypes_map = MM._tiers["t5"]["dtypes"]
    cross = _prepare_cross_kv(enc_hidden, h, all_names, model, device, dtypes_map, debug=debug)
    # init decoder with <pad> token start
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    dec_ids = torch.full((ids.shape[0], 1), pad_id, device=device, dtype=torch.long)
    caches = {b: {"k": None, "v": None} for b in _unique_blocks_decoder(all_names)}
    for t in range(max_new_tokens):
        y = _rowwise_embed(dec_ids[:, -1:], model, device, MM._tiers["t5"])  # last token embed
        B, S_dec, d_model = int(y.shape[0]), int(y.shape[1]), int(y.shape[2])
        for i, b in enumerate(_unique_blocks_decoder(all_names), 1):
            ln1 = f"decoder.block.{b}.layer.0.layer_norm.weight"
            qn = f"decoder.block.{b}.layer.0.SelfAttention.q.weight"
            kn = f"decoder.block.{b}.layer.0.SelfAttention.k.weight"
            vn = f"decoder.block.{b}.layer.0.SelfAttention.v.weight"
            on = f"decoder.block.{b}.layer.0.SelfAttention.o.weight"
            ln_x = f"decoder.block.{b}.layer.1.layer_norm.weight"
            q_x = f"decoder.block.{b}.layer.1.EncDecAttention.q.weight"
            o_x = f"decoder.block.{b}.layer.1.EncDecAttention.o.weight"
            ln2 = f"decoder.block.{b}.layer.2.layer_norm.weight"
            wi0 = f"decoder.block.{b}.layer.2.DenseReluDense.wi_0.weight"
            wi1 = f"decoder.block.{b}.layer.2.DenseReluDense.wi_1.weight"
            wo = f"decoder.block.{b}.layer.2.DenseReluDense.wo.weight"
            # ln1
            ln1_w = _alloc_for(model, [ln1], device, dtypes_map)[ln1]
            _page_into(h, [ln1], all_names, {ln1: ln1_w}); _pager.wait_all()
            yn = _rmsnorm(y, ln1_w)
            del ln1_w; torch.cuda.empty_cache()
            # self-attn q/k/v for current token
            q_w = _alloc_for(model, [qn], device, dtypes_map)[qn]
            _page_into(h, [qn], all_names, {qn: q_w}); _pager.wait_all()
            k_w = _alloc_for(model, [kn], device, dtypes_map)[kn]
            _page_into(h, [kn], all_names, {kn: k_w}); _pager.wait_all()
            v_w = _alloc_for(model, [vn], device, dtypes_map)[vn]
            _page_into(h, [vn], all_names, {vn: v_w}); _pager.wait_all()
            q = _linear(yn, q_w, qn); k = _linear(yn, k_w, kn); v = _linear(yn, v_w, vn)
            del q_w, k_w, v_w; torch.cuda.empty_cache()
            # heads
            num_heads, head_dim = cross[b][2], cross[b][3]
            q = q.view(B, 1, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, 1, num_heads, head_dim).transpose(1, 2)
            v = v.view(B, 1, num_heads, head_dim).transpose(1, 2)
            # append to cache
            if caches[b]["k"] is None:
                caches[b]["k"] = k
                caches[b]["v"] = v
            else:
                caches[b]["k"] = torch.cat([caches[b]["k"], k], dim=2)
                caches[b]["v"] = torch.cat([caches[b]["v"], v], dim=2)
            kv_k = caches[b]["k"]; kv_v = caches[b]["v"]
            causal = torch.triu(torch.ones(kv_k.shape[2], kv_k.shape[2], device=device, dtype=q.dtype), diagonal=1) * (-1e9)
            attn = torch.nn.functional.scaled_dot_product_attention(q, kv_k, kv_v, attn_mask=causal, dropout_p=0.0, is_causal=False)
            attn = attn.transpose(1, 2).contiguous().view(B, 1, d_model)
            on_w = _alloc_for(model, [on], device, dtypes_map)[on]
            _page_into(h, [on], all_names, {on: on_w}); _pager.wait_all()
            attn = _linear(attn, on_w, on)
            del on_w; torch.cuda.empty_cache()
            y = y + attn
            # cross-attn (use precomputed K/V from encoder)
            ln_x_w = _alloc_for(model, [ln_x], device, dtypes_map)[ln_x]
            _page_into(h, [ln_x], all_names, {ln_x: ln_x_w}); _pager.wait_all()
            yz = _rmsnorm(y, ln_x_w)
            del ln_x_w; torch.cuda.empty_cache()
            qx_w = _alloc_for(model, [q_x], device, dtypes_map)[q_x]
            _page_into(h, [q_x], all_names, {q_x: qx_w}); _pager.wait_all()
            qx = _linear(yz, qx_w, q_x)
            del qx_w; torch.cuda.empty_cache()
            kx, vx, num_heads, head_dim = cross[b]
            qx = qx.view(B, 1, num_heads, head_dim).transpose(1, 2)
            mem_bias = torch.zeros((B, 1, 1, kx.shape[2]), device=device, dtype=qx.dtype)
            attn2 = torch.nn.functional.scaled_dot_product_attention(qx, kx, vx, attn_mask=mem_bias, dropout_p=0.0, is_causal=False)
            attn2 = attn2.transpose(1, 2).contiguous().view(B, 1, d_model)
            ox_w = _alloc_for(model, [o_x], device, dtypes_map)[o_x]
            _page_into(h, [o_x], all_names, {o_x: ox_w}); _pager.wait_all()
            attn2 = _linear(attn2, ox_w, o_x)
            del ox_w; torch.cuda.empty_cache()
            y = y + attn2
            # ffn
            ln2 = f"decoder.block.{b}.layer.2.layer_norm.weight"
            ln2_w = _alloc_for(model, [ln2], device, dtypes_map)[ln2]
            _page_into(h, [ln2], all_names, {ln2: ln2_w}); _pager.wait_all()
            yn2 = _rmsnorm(y, ln2_w)
            del ln2_w; torch.cuda.empty_cache()
            wi0_w = _alloc_for(model, [wi0], device, dtypes_map)[wi0]
            _page_into(h, [wi0], all_names, {wi0: wi0_w}); _pager.wait_all()
            u0 = _linear(yn2, wi0_w, wi0)
            del wi0_w; torch.cuda.empty_cache()
            wi1_w = _alloc_for(model, [wi1], device, dtypes_map)[wi1]
            _page_into(h, [wi1], all_names, {wi1: wi1_w}); _pager.wait_all()
            u1 = _linear(yn2, wi1_w, wi1)
            del wi1_w; torch.cuda.empty_cache()
            g = torch.nn.functional.gelu(u0, approximate="tanh") * u1
            wo_w = _alloc_for(model, [wo], device, dtypes_map)[wo]
            _page_into(h, [wo], all_names, {wo: wo_w}); _pager.wait_all()
            ffn = _linear(g, wo_w, wo)
            del wo_w; torch.cuda.empty_cache()
            y = y + ffn
        fn = "decoder.final_layer_norm.weight"
        fin = _alloc_for(model, [fn], device, dtypes_map)[fn]
        _page_into(h, [fn], all_names, {fn: fin}); _pager.wait_all()
        y = _rmsnorm(y, fin)
        del fin; torch.cuda.empty_cache()
        next_id = _greedy_next_token(y[:, -1, :], model, MM._tiers["t5"], device, debug=debug)
        dec_ids = torch.cat([dec_ids, next_id.view(B, 1)], dim=1)
        if debug:
            print(f"[DBG] step {t+1}: next_id={int(next_id[0].item())}")
    _pager.wait_all(); _pager.model_close(h); _pager.runtime_shutdown()
    return dec_ids


