#!/usr/bin/env python3
"""
Lightweight checks before bulk export: nutrition-like text, recipe notes in ingredients,
container junk lines, embedding vs ingredient consistency.

Usage:
  python validate_enriched_recipes.py new_recipee_data_enriched_first20.json
  python validate_enriched_recipes.py out.json --strict
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NUTRITION_HINT = re.compile(
    r"\b(calories|protein|total\s+fat|carbohydrates?|fiber|sodium|cholesterol)\b",
    re.IGNORECASE,
)
MAKES_NOTE = re.compile(r"^\s*\(?\s*makes\b", re.IGNORECASE)
BARE_CONTAINER = re.compile(
    r"^\s*(?:packet|bag|box|tin|jar|bottle|carton)\s+\S",
    re.IGNORECASE,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate enriched recipe JSON.")
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--strict", action="store_true", help="Exit 1 if any issue found")
    args = ap.parse_args()
    path = args.json_path
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(2)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("Expected JSON array", file=sys.stderr)
        sys.exit(2)

    issues: list[str] = []
    for i, r in enumerate(data):
        rid = r.get("id", i)
        title = (r.get("title") or "")[:50]
        for ing in r.get("ingredients") or []:
            if not isinstance(ing, str):
                continue
            s = ing.strip()
            if MAKES_NOTE.match(s):
                issues.append(f"id={rid} ingredient looks like yield note: {s!r}")
            if BARE_CONTAINER.match(s) and not re.match(r"^\d", s):
                issues.append(f"id={rid} bare container line: {s!r}")
        emb = r.get("ing_embedding_text") or ""
        if emb.startswith("ingredients: "):
            tokens = [t.strip() for t in emb[len("ingredients: ") :].split(",") if t.strip()]
            # crude hallucination check: black pepper token but no pepper word in any ingredient line
            if "black pepper" in tokens:
                joined = " ".join(x.lower() for x in (r.get("ingredients") or []) if isinstance(x, str))
                if "pepper" not in joined and "Pepper" not in joined:
                    issues.append(
                        f"id={rid} embedding has black pepper but no 'pepper' in ingredient lines ({title!r})"
                    )
        inst = r.get("instructions_standardized") or r.get("instructions")
        if isinstance(inst, dict):
            blob = " ".join(str(v) for v in inst.values())
            if NUTRITION_HINT.search(blob):
                issues.append(f"id={rid} possible nutrition text in instructions ({title!r})")

    if not issues:
        print(f"OK: {len(data)} recipe(s), no issues flagged.")
        return
    print(f"Found {len(issues)} issue(s):\n")
    for line in issues[:200]:
        print(line)
    if len(issues) > 200:
        print(f"... and {len(issues) - 200} more")
    if args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
