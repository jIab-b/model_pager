import argparse, pathlib, safetensors.torch as safe

def list_tensors(p):
    meta = safe.safe_open(p, framework="pt")
    return len(list(meta.keys()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="safetensors")
    ap.add_argument("--verify_wan_comfy", action="store_true",
                    help="Run wan_comfy.py and verify it loads files correctly")
    args = ap.parse_args()
    
    if args.verify_wan_comfy:
        # Import wan_comfy to trigger its debug section
        print("Running wan_comfy.py to verify file loading...")
        try:
            import models.wan_comfy
            print("wan_comfy.py imported successfully")
        except Exception as e:
            print(f"Error importing wan_comfy.py: {e}")
            return 1
        
        # Check if debug logs were generated
        logs_dir = pathlib.Path("logs")
        expected_logs = [
            "comfy_t5.log",
            "comfy_transformer.log", 
            "comfy_vae.log"
        ]
        
        print("\nChecking debug logs:")
        for log_name in expected_logs:
            log_path = logs_dir / log_name
            if log_path.exists():
                try:
                    with open(log_path, "r") as f:
                        count = f.read().strip()
                    print(f"{log_name}: {count} tensors")
                except Exception as e:
                    print(f"{log_name}: Error reading log - {e}")
            else:
                print(f"{log_name}: Not found")
        
        # Compare with direct tensor counts
        print("\nComparing with direct tensor counts:")
        base = pathlib.Path("safetensors")
        files = [
            ("wan2.1_t2v_1.3B_fp16.safetensors", "comfy_transformer.log"),
            ("umt5_xxl_fp16.safetensors", "comfy_t5.log"),
            ("wan_2.1_vae.safetensors", "comfy_vae.log")
        ]
        
        for file_name, log_name in files:
            file_path = base / file_name
            if file_path.exists():
                direct_count = list_tensors(file_path)
                log_path = logs_dir / log_name
                if log_path.exists():
                    try:
                        with open(log_path, "r") as f:
                            wan_comfy_count = int(f.read().strip())
                        match = "✓" if direct_count == wan_comfy_count else "✗"
                        print(f"{file_name}: {direct_count} (direct) {match} {wan_comfy_count} (wan_comfy)")
                    except Exception as e:
                        print(f"{file_name}: Error comparing counts - {e}")
                else:
                    print(f"{file_name}: {direct_count} (direct) - no wan_comfy log")
            else:
                print(f"{file_name}: File not found")
    else:
        # Original functionality
        base = pathlib.Path(args.model_dir)
        files = [base/"wan2.1_t2v_1.3B_fp16.safetensors",
                 base/"umt5_xxl_fp16.safetensors",
                 base/"wan_2.1_vae.safetensors"]
        for f in files:
            if not f.exists():
                print(f"missing {f}")
                return 1
        for f in files:
            print(f"{f.name}: {list_tensors(f)} tensors")

if __name__ == "__main__":
    main()