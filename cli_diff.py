import argparse, pathlib, safetensors.torch as safe

def list_tensors(path):
    meta = safe.safe_open(path, framework="pt")
    names = list(meta.keys())
    return len(names)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="safetensors")
    args = ap.parse_args()
    base = pathlib.Path(args.model_dir)
    files = [base/"wan2.1_t2v_1.3B_fp16.safetensors",
             base/"umt5_xxl_fp8_e4m3fn_scaled.safetensors",
             base/"wan_2.1_vae.safetensors"]
    for f in files:
        if not f.exists():
            print(f"missing {f}")
            return 1
    for f in files:
        n = list_tensors(f)
        print(f"{f.name}: {n} tensors")

if __name__ == "__main__":
    main()
