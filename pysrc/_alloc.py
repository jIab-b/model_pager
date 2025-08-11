import pathlib
from torch.utils.cpp_extension import load

_ext_path = pathlib.Path(__file__).resolve().parent.parent / "csrc"

alloc_mod = load(name="alloc_hook_ext",
                 sources=[str(_ext_path / "alloc_hook.cpp")],
                 extra_include_paths=[str(_ext_path)],
                 verbose=False)

set_cap = alloc_mod.set_cap
