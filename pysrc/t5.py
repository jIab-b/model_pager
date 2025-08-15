import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import re
import json
import time
import page_table_ext as _pager
from transformers import T5Tokenizer
from pysrc.models import t5_skel, MM

_DBG_DEPTH = 0
_DBG_FILE = None
_DBG_STDOUT = None
_DBG_STDERR = None

def _debug_redirect_enter(enabled: bool, log_name: str = "log_t5.log"):
    global _DBG_DEPTH, _DBG_FILE, _DBG_STDOUT, _DBG_STDERR
    if not enabled:
        return
    if _DBG_DEPTH == 0:
        logs_dir = (Path(__file__).resolve().parent.parent / "logs").resolve()
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / log_name
        try:
            if log_path.exists():
                log_path.unlink()
        except Exception:
            pass
        _DBG_FILE = open(log_path, "w", buffering=1)
        _DBG_STDOUT = sys.stdout
        _DBG_STDERR = sys.stderr
        sys.stdout = _DBG_FILE
        sys.stderr = _DBG_FILE
    _DBG_DEPTH += 1

def _debug_redirect_exit():
    global _DBG_DEPTH, _DBG_FILE, _DBG_STDOUT, _DBG_STDERR
    if _DBG_DEPTH <= 0:
        return
    _DBG_DEPTH -= 1
    if _DBG_DEPTH == 0 and _DBG_FILE is not None:
        sys.stdout = _DBG_STDOUT
        sys.stderr = _DBG_STDERR
        try:
            _DBG_FILE.flush()
            _DBG_FILE.close()
        finally:
            pass
        _DBG_FILE = None
        _DBG_STDOUT = None
        _DBG_STDERR = None

_DTYPE_MAP = {
    "F16": torch.float16,
    "FLOAT16": torch.float16,
    "BF16": torch.bfloat16,
    "BFLOAT16": torch.bfloat16,
    "F32": torch.float32,
    "FLOAT32": torch.float32,
}

_W_T: dict[str, bool] = {}

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

def _unique_blocks(all_names):
    s = set()
    for n in all_names:
        m = re.search(r"encoder\.block\.(\d+)\.", n)
        if m:
            s.add(int(m.group(1)))
    return sorted(s)

def _build_registry():
    tier = MM._tiers["t5"]
    offmap = tier["offsets"]
    sizes = tier["sizes"]
    tensor_files = tier["tensor_files"]
    header_lens = tier["header_lens"]
    files = sorted(set(tensor_files.values()))
    name_to_loc = {}
    handles = []
    for f in files:
        names = [n for n in offmap.keys() if tensor_files[n] == f]
        names.sort(key=lambda n: offmap[n])
        base = 8 + header_lens[f]
        file_offsets = [base + offmap[n] for n in names]
        szs = [sizes[n] for n in names]
        h = _pager.model_register(f, file_offsets, szs)
        handles.append(h)
        for i, n in enumerate(names):
            name_to_loc[n] = (h, i)
    return name_to_loc, list(offmap.keys()), handles

def _alloc_for(model, names, device, dtypes_map):
    shapes = dict(model.named_parameters())
    out = {}
    for n in names:
        dt_key = str(dtypes_map.get(n, "F16")).upper()
        dt = _DTYPE_MAP.get(dt_key, torch.float16)
        out[n] = torch.empty(tuple(shapes[n].shape), device=device, dtype=dt)
    return out

def _page_into(lut, needed, dst):
    by_handle = {}
    for n in needed:
        if n not in lut or n not in dst:
            continue
        h, idx = lut[n]
        by_handle.setdefault(h, []).append((idx, dst[n]))
    for h, pairs in by_handle.items():
        idxs = [i for i, _ in pairs]
        dsts = [t for _, t in pairs]
        _pager.model_read_into_batch(h, idxs, dsts, None)

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
    val_if_large = half + ((torch.log(rel.float() / max_exact + 1e-6) / torch.log(torch.tensor(max_distance / max_exact, device=device, dtype=torch.float32) + 1e-6)) * (n - max_exact)).to(torch.long)
    val_if_large = torch.clamp(val_if_large, max=max_exact + (n - max_exact) - 1)
    buckets = torch.where(is_small, rel, val_if_large) + pos
    return buckets

def _rowwise_embed(ids: torch.Tensor, model, device, tier):
    offmap, dtypes = tier["offsets"], tier["dtypes"]
    tf = tier["tensor_files"]["shared.weight"]
    base = 8 + tier["header_lens"][tf]
    shapes = dict(model.named_parameters())
    d_model = shapes["shared.weight"].shape[1]
    uniq, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    uniq_cpu = uniq.detach().to("cpu")
    dtype_key = str(dtypes["shared.weight"]).upper()
    row_elem_bytes = 2 if dtype_key in ("F16", "FLOAT16", "BF16", "BFLOAT16") else 4
    row_bytes = int(d_model) * row_elem_bytes
    off0 = base + offmap["shared.weight"]
    file_offsets = [off0 + int(tid.item()) * row_bytes for tid in uniq_cpu]
    sizes = [row_bytes] * len(file_offsets)
    h_rows = _pager.model_register(tf, file_offsets, sizes)
    W_rows = torch.empty((len(file_offsets), int(d_model)), device=device, dtype=_DTYPE_MAP.get(dtype_key, torch.float16))
    dsts = [W_rows[i] for i in range(W_rows.shape[0])]
    _pager.model_read_into_batch(h_rows, list(range(len(file_offsets))), dsts, None)
    _pager.wait_all()
    _pager.model_close(h_rows)
    x = torch.nn.functional.embedding(inverse.view(ids.shape), W_rows)
    return x

def _dump_mm_debug(log_path: Path):
    tier = MM._tiers["t5"]
    lines = []
    lines.append(f"tier_name=t5 is_sharded={tier['is_sharded']} files={len(tier['files'])}")
    for f in tier["files"]:
        lines.append(f"file: {f} header_len={tier['header_lens'][f]}")
    lines.append(f"tensors={len(tier['offsets'])}")
    ex = sorted(list(tier["offsets"].keys()))[:200]
    for k in ex:
        lines.append(f"tensor: {k} file={tier['tensor_files'][k]} off={tier['offsets'][k]} sz={tier['sizes'][k]} dtype={tier['dtypes'][k]} shape={tier['shapes'][k]}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if log_path.exists():
            log_path.unlink()
    except Exception:
        pass
    log_path.write_text("\n".join(lines) + "\n")

def encode(prompt: str, max_length: int = 256, out_path: str | None = None, debug: bool = False, mm_debug: bool = False):
    _debug_redirect_enter(debug)
    with torch.inference_mode():
        _pager.runtime_init(0, 2, 2)
        if mm_debug:
            _dump_mm_debug(Path(__file__).resolve().parent.parent / "logs" / "log_MM_t5.log")
        lut, all_names, handles = _build_registry()
        tok_path = (Path(__file__).resolve().parent.parent / "safetensors" / "t5_tokenizer").resolve()
        tok = T5Tokenizer.from_pretrained(str(tok_path))
        ids = tok(prompt, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").input_ids.cuda(non_blocking=True)
        model = t5_skel()
        device = torch.device("cuda")
        dtypes_map = MM._tiers["t5"]["dtypes"]
        tier = MM._tiers["t5"]
        cfg = None
        try:
            tier_path0 = Path(MM._tiers["t5"]["files"][0]).parent
            cfgf = tier_path0 / "config-fp16.json"
            if not cfgf.exists():
                cfgf = tier_path0 / "config.json"
            if cfgf.exists():
                cfg = json.loads(cfgf.read_text())
        except Exception:
            cfg = None
    shapes = dict(model.named_parameters())
    d_model = shapes["shared.weight"].shape[1]
    compute_dtype = torch.float16
    x = _rowwise_embed(ids, model, device, tier).to(compute_dtype)
    neg_inf = -1e4 if compute_dtype in (torch.float16, torch.bfloat16) else -1e9
    if debug:
        ga, gf = _pager.get_memory_stats()
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name()
        print(f"gpu={name} cap={cap} init dtype={compute_dtype} ga={ga} gf={gf}")
        print(f"cfg heads={cfg.get('num_heads', None) if cfg else None} d_kv={cfg.get('d_kv', None) if cfg else None} layers={cfg.get('num_layers', None) if cfg else None}")
        print(f"embed shape={tuple(x.shape)} sum={x.float().abs().sum().item():.3f} mean={x.float().abs().mean().item():.6f}")
    blocks = _unique_blocks(all_names)
    if debug:
        print(f"indexed tensors: {len(all_names)}")
        if len(all_names) > 0:
            sample = sorted(all_names)[:10]
            for s in sample:
                print(f"tensor: {s}")
    if len(blocks) == 0 and cfg is not None and int(cfg.get("num_layers", 0)) > 0:
        blocks = list(range(int(cfg.get("num_layers"))))
        print(f"T5 encoder blocks not found by scan; using config fallback: {len(blocks)}")
    else:
        print(f"T5 encoder blocks detected: {len(blocks)}")
    B, S = int(x.shape[0]), int(x.shape[1])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    key_mask = (ids != pad_id).to(dtype=x.dtype).view(B, 1, 1, S)
    query_mask = (ids != pad_id).to(dtype=x.dtype).view(B, S, 1)
    x = x * query_mask
    cfg_heads = None
    cfg_head_dim = None
    if cfg is not None:
        try:
            cfg_heads = int(cfg.get("num_heads", 0)) or None
            cfg_head_dim = int(cfg.get("d_kv", 0)) or None
        except Exception:
            cfg_heads = None
            cfg_head_dim = None
    for i, b in enumerate(blocks, 1):
        print(f"[T5] block {i}/{len(blocks)} start")
        qn = f"encoder.block.{b}.layer.0.SelfAttention.q.weight"
        kn = f"encoder.block.{b}.layer.0.SelfAttention.k.weight"
        vn = f"encoder.block.{b}.layer.0.SelfAttention.v.weight"
        on = f"encoder.block.{b}.layer.0.SelfAttention.o.weight"
        ln1 = f"encoder.block.{b}.layer.0.layer_norm.weight"
        rpb = f"encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        wi0 = f"encoder.block.{b}.layer.1.DenseReluDense.wi_0.weight"
        wi1 = f"encoder.block.{b}.layer.1.DenseReluDense.wi_1.weight"
        wo = f"encoder.block.{b}.layer.1.DenseReluDense.wo.weight"
        ln2 = f"encoder.block.{b}.layer.1.layer_norm.weight"
        t0 = time.time()
        ln1_w = _alloc_for(model, [ln1], device, dtypes_map)[ln1]
        q_w = _alloc_for(model, [qn], device, dtypes_map)[qn]
        k_w = _alloc_for(model, [kn], device, dtypes_map)[kn]
        v_w = _alloc_for(model, [vn], device, dtypes_map)[vn]
        _page_into(lut, [ln1, qn, kn, vn], {ln1: ln1_w, qn: q_w, kn: k_w, vn: v_w})
        _pager.wait_all()
        xn = _rmsnorm(x, ln1_w)
        q = _linear(xn, q_w, qn)
        k = _linear(xn, k_w, kn)
        v = _linear(xn, v_w, vn)
        del ln1_w, q_w, k_w, v_w
        has_rpb = rpb in dtypes_map and rpb in dict(model.named_parameters())
        rpb_w = None
        if has_rpb:
            rpb_w = _alloc_for(model, [rpb], device, dtypes_map)[rpb]
            _page_into(lut, [rpb], {rpb: rpb_w}); _pager.wait_all()
            if cfg_heads is not None:
                num_heads = cfg_heads
            else:
                if rpb_w.dim() == 2:
                    a, b2 = rpb_w.shape
                    cand = [x for x in (a, b2) if d_model % x == 0]
                    num_heads = max(cand) if cand else max(a, b2)
                else:
                    num_heads = (d_model // 64) if d_model % 64 == 0 else (d_model // 32)
        else:
            if cfg_heads is not None and d_model % cfg_heads == 0:
                num_heads = cfg_heads
            else:
                q_inner = q.shape[-1]
                num_heads = (d_model // 64) if d_model % 64 == 0 else (d_model // 32)
        head_dim = cfg_head_dim if (cfg_head_dim is not None and (q.shape[-1] % (cfg_heads or num_heads) == 0)) else (q.shape[-1] // num_heads)
        if (q.shape[-1] % num_heads) != 0:
            raise RuntimeError(f"q inner={q.shape[-1]} not divisible by heads={num_heads}")
        B, S, _ = x.shape
        if debug:
            print(f"blk{i} heads={num_heads} head_dim={head_dim} qkv={tuple(q.shape)}")
        q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, S, num_heads, head_dim).transpose(1, 2)
        if has_rpb:
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
            buckets = _rel_pos_bucket(S, num_buckets=num_buckets, max_distance=128, device=device)
            buckets = buckets.clamp_(min=0, max=num_buckets - 1)
            bias = rpb_table[buckets].float()
            bias = bias.permute(2, 0, 1).unsqueeze(0)
            mask_bias = (1.0 - key_mask.to(dtype=torch.float32)) * (-1e9)
            bias = bias + mask_bias
            del rpb_w
        else:
            bias = (1.0 - key_mask.to(dtype=torch.float32)) * (-1e9)
        qf, kf, vf = q.float(), k.float(), v.float()
        attn = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf, attn_mask=bias, dropout_p=0.0, is_causal=False)
        attn = attn.to(q.dtype).transpose(1, 2).contiguous().view(B, S, d_model)
        attn = torch.nan_to_num(attn)
        attn = attn * query_mask
        on_w = _alloc_for(model, [on], device, dtypes_map)[on]
        _page_into(lut, [on], {on: on_w}); _pager.wait_all()
        attn = _linear(attn, on_w, on)
        del on_w
        x = x + attn
        ln2_w = _alloc_for(model, [ln2], device, dtypes_map)[ln2]
        wi0_w = _alloc_for(model, [wi0], device, dtypes_map)[wi0]
        wi1_w = _alloc_for(model, [wi1], device, dtypes_map)[wi1]
        _page_into(lut, [ln2, wi0, wi1], {ln2: ln2_w, wi0: wi0_w, wi1: wi1_w}); _pager.wait_all()
        yn = _rmsnorm(x, ln2_w)
        u0 = _linear(yn, wi0_w, wi0)
        u1 = _linear(yn, wi1_w, wi1)
        del ln2_w, wi0_w, wi1_w
        u0 = torch.nan_to_num(u0)
        u1 = torch.nan_to_num(u1)
        g = torch.nn.functional.gelu(u0, approximate="tanh") * u1
        wo_w = _alloc_for(model, [wo], device, dtypes_map)[wo]
        _page_into(lut, [wo], {wo: wo_w}); _pager.wait_all()
        ffn = _linear(g, wo_w, wo)
        ffn = torch.nan_to_num(ffn)
        ffn = ffn * query_mask
        del wo_w
        x = x + ffn
        if debug:
            torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000
            ga, gf = _pager.get_memory_stats()
            print(f"[DBG] block {i} dt_ms={dt:.1f} x.sum={x.float().abs().sum().item():.3f} x.mean={x.float().mean().item():.6f} ga={ga} gf={gf}")
        print(f"[T5] block {i}/{len(blocks)} done")
    fin = _alloc_for(model, ["encoder.final_layer_norm.weight"], device, dtypes_map)
    _page_into(lut, ["encoder.final_layer_norm.weight"], {"encoder.final_layer_norm.weight": fin["encoder.final_layer_norm.weight"]})
    x = _rmsnorm(x, fin["encoder.final_layer_norm.weight"])
    del fin
    _pager.wait_all()
    for h in handles:
        _pager.model_close(h)
    _pager.runtime_shutdown()
    if out_path:
        out_cpu = x.detach().to("cpu")
        out_cpu_path = Path(out_path)
        out_cpu_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_cpu, out_cpu_path)
    print(f"[T5] output shape: {tuple(x.shape)} sum={x.float().abs().sum().item():.3f} mean={x.float().mean().item():.6f}")
    _debug_redirect_exit()
    return x


