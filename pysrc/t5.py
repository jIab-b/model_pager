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

def _alloc_for(model, names, device, dtype):
    shapes = dict(model.named_parameters())
    return {n: torch.empty(tuple(shapes[n].shape), device=device, dtype=dtype) for n in names}

def _page_into(h, needed, all_names, dst):
    idx = {n: i for i, n in enumerate(all_names)}
    indices = [idx[n] for n in needed]
    dsts = [dst[n] for n in needed]
    _pager.model_read_into_batch(h, indices, dsts, None)

def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    s = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(s + eps)
    return x * w

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

_W_T = {}

def _linear(x: torch.Tensor, w: torch.Tensor, key: str) -> torch.Tensor:
    if _W_T.get(key, False):
        return torch.nn.functional.linear(x, w.t())
    try:
        y = torch.nn.functional.linear(x, w)
        _W_T[key] = False
        return y
    except RuntimeError:
        y = torch.nn.functional.linear(x, w.t())
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
    emb_names = ["shared.weight"]
    emb = _alloc_for(model, emb_names, device, dtype)
    _page_into(h, emb_names, all_names, emb)
    _pager.wait_all()
    if debug:
        try:
            print(f"[DBG] ids min={ids.min().item()} max={ids.max().item()} shape={tuple(ids.shape)}")
        except Exception:
            pass
        print(f"[DBG] shared.weight mean={emb['shared.weight'].float().abs().mean().item():.6f} shape={tuple(emb['shared.weight'].shape)}")
        # compare a sample row against direct file slice
        tier = MM._tiers["t5"]
        offmap, szmap, dtypes, shapes = tier["offsets"], tier["sizes"], tier["dtypes"], tier["shapes"]
        with open(tier["path"], "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
        base = 8 + header_len
        rid = int(ids[0, 0].item())
        try:
            np_row = _peek_shared_row(tier["path"], base, offmap["shared.weight"], shapes["shared.weight"][1], rid, cols=8, dtype=dtypes["shared.weight"])  # type: ignore
            gpu_row = emb["shared.weight"][rid, :8].detach().to("cpu", dtype=torch.float32).numpy()
            print(f"[DBG] shared.row[{rid}] cpu_file[:8]={np_row} vs gpu_paged[:8]={gpu_row}")
        except Exception as e:
            print(f"[DBG] shared.row compare failed: {e}")
    x = _embedding(ids, emb["shared.weight"]).to(dtype)
    if debug:
        print(f"[DBG] after embedding: sum={x.float().abs().sum().item():.3f} mean={x.float().abs().mean().item():.6f} dtype={x.dtype} device={x.device}")
    d_model = emb["shared.weight"].shape[1]
    del emb
    blocks = _unique_blocks(all_names)
    print(f"T5 encoder blocks detected: {len(blocks)}")
    import time
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
        wt = _alloc_for(model, need, device, dtype)
        _page_into(h, need, all_names, wt)
        _pager.wait_all()
        if debug:
            try:
                m_q = wt[qn].float().abs().mean().item()
                m_k = wt[kn].float().abs().mean().item()
                m_v = wt[vn].float().abs().mean().item()
                m_o = wt[on].float().abs().mean().item()
                m_ln1 = wt[ln1].float().abs().mean().item()
                m_ln2 = wt[ln2].float().abs().mean().item()
                print(f"[DBG] W q/k/v/o mean={m_q:.5f}/{m_k:.5f}/{m_v:.5f}/{m_o:.5f} ln1/ln2={m_ln1:.5f}/{m_ln2:.5f}")
                print(f"[DBG] orient q/k/v/wo: {(_W_T.get(qn), _W_T.get(kn), _W_T.get(vn), _W_T.get(wo))}")
            except Exception:
                pass
        print(f"[T5] block {i}/{len(blocks)}: compute…")
        t0 = time.time()
        xn = _rmsnorm(x, wt[ln1])
        if debug:
            print(f"[DBG] xn mean={xn.float().abs().mean().item():.6f}")
        q = _linear(xn, wt[qn], qn)
        k = _linear(xn, wt[kn], kn)
        v = _linear(xn, wt[vn], vn)
        if debug:
            print(f"[DBG] q/k/v mean={q.float().abs().mean().item():.6f}/{k.float().abs().mean().item():.6f}/{v.float().abs().mean().item():.6f}")
        # infer heads from rel-pos table dims (pick the axis that divides d_model and is larger)
        if wt[rpb].dim() == 2:
            a, b = wt[rpb].shape
            cand = [x for x in (a, b) if d_model % x == 0]
            num_heads = max(cand) if cand else max(a, b)
        else:
            num_heads = (d_model // 64) if d_model % 64 == 0 else (d_model // 32)
        head_dim = d_model // num_heads
        B, S, _ = x.shape
        q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, S, num_heads, head_dim).transpose(1, 2)
        # ensure rpb_table has shape [num_buckets, num_heads]
        if wt[rpb].dim() == 2:
            a, b = wt[rpb].shape
            if b == num_heads:
                rpb_table = wt[rpb]
                num_buckets = a
            elif a == num_heads:
                rpb_table = wt[rpb].t()
                num_buckets = b
            else:
                # fallback: assume larger dim is heads
                if a > b:
                    rpb_table = wt[rpb].t()
                    num_buckets = b
                else:
                    rpb_table = wt[rpb]
                    num_buckets = a
        else:
            # unexpected; default buckets
            num_buckets = 32
            rpb_table = wt[rpb]
        buckets = _rel_pos_bucket(S, num_buckets=num_buckets, max_distance=128, device=device)
        buckets = buckets.clamp_(min=0, max=num_buckets - 1)
        if debug:
            print(f"[DBG] rpb table shape={tuple(rpb_table.shape)} buckets min/max={buckets.min().item()}/{buckets.max().item()} num_heads={num_heads}")
        bias = rpb_table[buckets]  # [S,S,num_heads]
        bias = bias.permute(2, 0, 1).unsqueeze(0).to(dtype)  # [1,n_heads,S,S]
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, S, d_model)
        attn = _linear(attn, wt[on], on)
        if debug:
            print(f"[DBG] attn_out sum={attn.float().abs().sum().item():.3f} mean={attn.float().abs().mean().item():.6f}")
        x = x + attn
        if debug:
            print(f"[DBG] after attn residual: x.sum={x.float().abs().sum().item():.3f}")
        yn = _rmsnorm(x, wt[ln2])
        u0 = _linear(yn, wt[wi0], wi0)
        u1 = _linear(yn, wt[wi1], wi1)
        g = torch.nn.functional.gelu(u0, approximate="tanh") * u1
        ffn = _linear(g, wt[wo], wo)
        if debug:
            print(f"[DBG] u0/u1 mean={u0.float().abs().mean().item():.6f}/{u1.float().abs().mean().item():.6f} g mean={g.float().abs().mean().item():.6f} ffn_out sum={ffn.float().abs().sum().item():.3f}")
        x = x + ffn
        del wt
        if debug:
            torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000
            print(f"[DBG] block {i} time={dt:.1f} ms x.sum={x.float().abs().sum().item():.3f}")
        print(f"[T5] block {i}/{len(blocks)}: done")
    fin = _alloc_for(model, ["encoder.final_layer_norm.weight"], device, dtype)
    _page_into(h, ["encoder.final_layer_norm.weight"], all_names, fin)
    x = _rmsnorm(x, fin["encoder.final_layer_norm.weight"])
    del fin
    _pager.wait_all()
    _pager.model_close(h)
    if out_path:
        out_cpu = x.detach().to("cpu")
        out_cpu_path = Path(out_path)
        out_cpu_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_cpu, out_cpu_path)
    print(f"[T5] output shape: {tuple(x.shape)} sum={x.float().abs().sum().item():.3f} mean={x.float().mean().item():.6f}")
    return x


