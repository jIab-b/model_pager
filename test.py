from models import wan_comfy as wc
import page_table_ext as _pager

MM = wc.MM
name = "t5"
entry = MM._tiers[name]
order = sorted(entry["offsets"].items(), key=lambda kv: kv[1])
file_offsets = [off for _, off in order]
sizes = [entry["sizes"][k] for k, _ in order]

_pager.model_set_weights_layout(file_offsets, sizes)
planned = _pager.model_planned_bytes()

r0,u0,a0,f0 = _pager.get_memory_stats()
_pager.model_reserve(int(planned))
_pager.model_prefetch()
r1,u1,a1,f1 = _pager.get_memory_stats()

print({"planned": planned, "gpu_allocated_delta": a1 - a0, "gpu_free_delta": f1 - f0, "uma_reserved": r1})
_pager.model_evict()