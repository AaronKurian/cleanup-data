#!/usr/bin/env python3
"""
Extract all unique ingredient names from a normalized (or raw) recipes JSON.

For lines matching [quantity] [unit] [ingredient], the "name" is everything after unit.
For ingredient-only lines (no leading number), the whole line is the name.

Usage:
  python extract_unique_ingredient_names.py
  python extract_unique_ingredient_names.py --input recipes_normalized.json --output unique_ingredient_names.json

Requires normalize_ingredients.FINAL_UNITS for structured parsing (same as pipeline).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Import FINAL_UNITS from the normalizer (single source of truth)
try:
    from normalize_ingredients import FINAL_UNITS
except ImportError:
    print("Run from project directory or ensure normalize_ingredients.py is on PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

QTY_RE = re.compile(r"^\d+(\.\d+)?$")


def ingredient_name_from_line(line: str) -> str:
    """Best-effort ingredient name: rest after qty+unit, or full line."""
    s = line.strip()
    if not s:
        return ""
    parts = s.split()
    if (
        len(parts) >= 3
        and QTY_RE.match(parts[0])
        and parts[1].lower() in FINAL_UNITS
    ):
        return " ".join(parts[2:]).strip()
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract unique ingredient names from recipes JSON.")
    parser.add_argument("--input", "-i", default="recipes_normalized.json", help="Recipes JSON path")
    parser.add_argument("--output", "-o", default="unique_ingredient_names.json", help="Write unique names + counts (JSON)")
    parser.add_argument("--txt", default="unique_ingredient_names.txt", help="Plain sorted list (one per line)")
    parser.add_argument("--counts", action="store_true", help="Include frequency counts in JSON output")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        recipes = json.load(f)

    counter: Counter[str] = Counter()
    for recipe in recipes:
        for ing in recipe.get("ingredients") or []:
            if not ing or not isinstance(ing, str):
                continue
            name = ingredient_name_from_line(ing)
            if name:
                counter[name] += 1

    unique_sorted = sorted(counter.keys(), key=str.lower)

    with open(args.txt, "w") as f:
        for n in unique_sorted:
            f.write(n + "\n")

    out_obj: dict = {"unique_count": len(unique_sorted), "names": unique_sorted}
    if args.counts:
        out_obj["name_counts"] = {k: counter[k] for k in unique_sorted}

    with open(args.output, "w") as f:
        json.dump(out_obj, f, indent=2)

    print(f"Recipes: {len(recipes)}")
    print(f"Unique ingredient names: {len(unique_sorted)}")
    print(f"Wrote {args.output} and {args.txt}")


if __name__ == "__main__":
    main()
