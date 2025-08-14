import os, sys, re, subprocess, torch
from models import wan_comfy as wc
import page_table_ext as _pager

def group_tensors_t5():
    e = wc.MM._tiers["t5"]
    offs, sizes, path = e["offsets"], e["sizes"], e["path"]
    pat = re.compile(r"encoder\.block\.(\d+)\.")
    groups = {}
    for k, off in offs.items():
        m = pat.search(k)
        key = m.group(1) if m else "misc"
        groups.setdefault(key, []).append((off, sizes[k]))
    return path, {k: sorted(v) for k, v in groups.items()}

def push_pairs(path, pairs, chunk_mb=8):
    file_offsets = [x for x,_ in pairs]
    lengths = [y for _,y in pairs]
    _pager.model_set_weights_layout(file_offsets, lengths)
    planned = _pager.model_planned_bytes()
    r0,u0,a0,f0 = _pager.get_memory_stats()
    _pager.model_reserve(int(planned))
    _pager.model_stage_file(path, int(chunk_mb<<20))
    _pager.model_prefetch()
    torch.cuda.synchronize()
    r1,u1,a1,f1 = _pager.get_memory_stats()
    print({"planned": planned, "uma_reserved": r1, "gpu_allocated_delta": a1-a0, "gpu_free_delta": f1-f0})
    _pager.model_evict()
    torch.cuda.synchronize()

if __name__ == "__main__":
    blk = os.environ.get("TEST2_BLOCK")
    path, groups = group_tensors_t5()
    if blk is not None:
        if blk not in groups:
            raise SystemExit(f"missing block {blk}")
        push_pairs(path, groups[blk], 8)
        sys.exit(0)
    keys = sorted([k for k in groups.keys() if k.isdigit()], key=int)
    print({"blocks": keys})
    for k in keys:
        env = os.environ.copy()
        env["TEST2_BLOCK"] = k
        subprocess.run([sys.executable, __file__], env=env, check=False)