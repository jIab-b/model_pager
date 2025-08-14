#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pysrc.t5 import encode as t5_encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--save", type=str, default=str(Path("logs")/"t5_emb.pt"))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    _ = t5_encode(args.prompt, max_length=args.max_length, out_path=args.save, debug=args.debug)
    print(f"t5 encode saved → {args.save}")


if __name__ == "__main__":
    main()


