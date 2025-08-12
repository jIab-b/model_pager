import argparse, pathlib, safetensors.torch as safe, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="directory containing .safetensors")
    args = ap.parse_args()
    p = pathlib.Path(args.path)
    logs_dir = pathlib.Path("logs")
    logs_dir.mkdir(exist_ok=True)
    for st in p.rglob("*.safetensors"):
        out = logs_dir / f"{st.stem}.log"
        with safe.safe_open(str(st), framework="pt") as fh, out.open("w") as f:
            # Extract file-level metadata
            file_meta = fh.metadata()
            
            # Extract tensor information (metadata only)
            tensor_info = {}
            for key in fh.keys():
                # Get tensor slice to access metadata without loading full tensor
                tensor_slice = fh.get_slice(key)
                # Access shape and dtype through the slice
                shape = tensor_slice.get_shape()
                dtype = str(tensor_slice.get_dtype())
                tensor_info[key] = {
                    "shape": list(shape),
                    "dtype": dtype
                }
            
            # Combine file metadata with tensor information
            result = {
                "tensors": tensor_info
            }
            
            # Add file metadata if it exists
            if file_meta:
                result["metadata"] = file_meta
            
            json.dump(result, f, indent=2)
    print(f"logs written in {logs_dir}")

if __name__ == "__main__":
    main()
