#!/usr/bin/env python3
"""
Concatenate multiple normalized batch JSON files (each a list of recipes) in order.

Example (three parallel batch outputs):
  python merge_recipe_batches.py -o recipes_full.json part0.json part1.json part2.json

Each input must be a JSON array of recipe objects (same schema as recipes_images.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Merge recipe batch JSON arrays in order.")
    p.add_argument("-o", "--output", required=True, help="Merged output JSON path")
    p.add_argument("inputs", nargs="+", help="Batch files in order (each: JSON array of recipes)")
    args = p.parse_args()

    merged: list = []
    for path_str in args.inputs:
        path = Path(path_str)
        if not path.exists():
            print(f"Missing: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path) as f:
            chunk = json.load(f)
        if not isinstance(chunk, list):
            print(f"Expected JSON array in {path}", file=sys.stderr)
            sys.exit(1)
        merged.extend(chunk)

    out = Path(args.output)
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {len(merged)} recipes → {out}")


if __name__ == "__main__":
    main()
