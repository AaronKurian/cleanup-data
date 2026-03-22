#!/usr/bin/env python3
"""Add sequential id (1, 2, 3, ...) to each recipe in a JSON array."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Add id field to each recipe object.")
    p.add_argument(
        "--input",
        "-i",
        default="recipes_images.json",
        help="Input JSON (array of recipe objects)",
    )
    p.add_argument(
        "--output",
        "-o",
        default="new_recipee_data.json",
        help="Output JSON path",
    )
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    with open(src, encoding="utf-8") as f:
        recipes = json.load(f)

    if not isinstance(recipes, list):
        raise SystemExit("Expected a JSON array of recipes.")

    out = []
    for i, recipe in enumerate(recipes, start=1):
        if isinstance(recipe, dict):
            rest = {k: v for k, v in recipe.items() if k != "id"}
            row = {"id": i, **rest}  # id first; sequential id replaces any old id
        else:
            row = {"id": i, "value": recipe}
        out.append(row)

    dst = Path(args.output)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(out)} recipes with id 1..{len(out)} → {dst.resolve()}")


if __name__ == "__main__":
    main()
