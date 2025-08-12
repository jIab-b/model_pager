import pathlib, re, subprocess, sys
import safetensors.torch as safe


BASE = pathlib.Path("safetensors")
FILES = [
    BASE / "wan2.1_t2v_1.3B_fp16.safetensors",
    BASE / "umt5_xxl_fp16.safetensors",
    BASE / "wan_2.1_vae.safetensors",
]


def _file_tensor_count(p: pathlib.Path) -> int:
    with safe.safe_open(p, framework="pt") as fh:
        return len(list(fh.keys()))


def test_cli_comfy_lists_all_tensors():
    # baseline counts via safetensors library
    expected = {f.name: _file_tensor_count(f) for f in FILES}

    # run the CLI script
    proc = subprocess.run(
        [sys.executable, "cli_comfy.py", "--model_dir", str(BASE)],
        capture_output=True,
        text=True,
        check=True,
    )

    # parse output lines like "wan_2.1_vae.safetensors: 1234 tensors"
    found = {}
    pattern = re.compile(r"^(.*\.safetensors):\s+(\d+)\s+tensors?")
    for line in proc.stdout.strip().splitlines():
        m = pattern.match(line.strip())
        if m:
            found[m.group(1)] = int(m.group(2))

    assert found == expected, f"CLI counts {found} differ from safetensors {expected}"
