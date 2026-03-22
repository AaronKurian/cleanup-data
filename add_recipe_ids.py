#!/usr/bin/env python3
"""
Assign sequential integer ``id`` to each object in a JSON array of recipes.

Usage:
  python3 add_recipe_ids.py -i new_recipee_data.json -o new_recipee_data.json
  python3 add_recipe_ids.py -i in.json -o out.json              # write to new file
  python3 add_recipe_ids.py -i in.json -o out.json --only-missing  # keep existing ids

Requires: stdlib only (json, argparse, pathlib).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Set recipe id = 1..N for each element in a JSON array.")
    ap.add_argument("-i", "--input", required=True, help="Input JSON (array of objects)")
    ap.add_argument("-o", "--output", required=True, help="Output JSON path")
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="Only set id when missing; does not renumber (may leave gaps / duplicates if data is messy)",
    )
    ap.add_argument("--indent", type=int, default=2, help="JSON indent (default 2)")
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not inp.exists():
        print(f"Not found: {inp}", file=sys.stderr)
        sys.exit(1)

    with open(inp, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Expected a JSON array at top level.", file=sys.stderr)
        sys.exit(1)

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Item at index {i} is not an object; skipping id assignment.", file=sys.stderr)
            continue
        if args.only_missing and "id" in item:
            continue
        item["id"] = i + 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=args.indent)
        f.write("\n")

    print(f"Wrote {len(data)} recipe(s) → {out}")


if __name__ == "__main__":
    main()
