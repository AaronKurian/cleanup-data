#!/usr/bin/env python3
"""
Merge ``desc_embedding_text`` from a slim JSON (title + embedding) into full recipes.

Matches by recipe ``title`` (case-insensitive, normalized whitespace). By default updates
only the **first N** recipes in the full file (default 26) and writes the **full** recipe
array to the output path (first N get ``desc_embedding_text`` when a match exists).

Usage:
  python3 merge_desc_embedding.py -1 test1.json -2 test2.json -o test3.json
  python3 merge_desc_embedding.py --limit 26
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip()).casefold()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge desc_embedding_text from test2 into test1 by title (first N recipes)."
    )
    ap.add_argument("-1", "--full", dest="test1", default="test1.json", help="Full recipes JSON")
    ap.add_argument("-2", "--slim", dest="test2", default="test2.json", help="Title + desc_embedding_text JSON")
    ap.add_argument("-o", "--output", default="test3.json", help="Output JSON path")
    ap.add_argument(
        "-n",
        "--limit",
        type=int,
        default=26,
        help="Only merge into the first N recipes in test1 (default 26)",
    )
    args = ap.parse_args()

    p1 = Path(args.test1).expanduser().resolve()
    p2 = Path(args.test2).expanduser().resolve()
    po = Path(args.output).expanduser().resolve()

    if not p1.exists():
        print(f"Not found: {p1}", file=sys.stderr)
        sys.exit(1)
    if not p2.exists():
        print(f"Not found: {p2}", file=sys.stderr)
        sys.exit(1)

    with open(p1, encoding="utf-8") as f:
        full: list = json.load(f)
    with open(p2, encoding="utf-8") as f:
        slim: list = json.load(f)

    if not isinstance(full, list) or not isinstance(slim, list):
        print("Expected JSON arrays.", file=sys.stderr)
        sys.exit(1)

    # title -> desc_embedding_text (later entries override if duplicate titles in slim)
    emb_by_title: dict[str, str] = {}
    for row in slim:
        if not isinstance(row, dict):
            continue
        title = row.get("title")
        emb = row.get("desc_embedding_text")
        if title is None or emb is None:
            continue
        emb_by_title[_norm_title(str(title))] = emb

    n = max(0, args.limit)
    merged = 0
    missing = 0

    out = []
    for i, recipe in enumerate(full):
        if not isinstance(recipe, dict):
            out.append(recipe)
            continue
        r = dict(recipe)
        if i < n:
            key = _norm_title(str(recipe.get("title") or ""))
            if key in emb_by_title:
                r["desc_embedding_text"] = emb_by_title[key]
                merged += 1
            else:
                missing += 1
        out.append(r)

    po.parent.mkdir(parents=True, exist_ok=True)
    with open(po, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"Wrote {len(out)} recipe(s) → {po}\n"
        f"  Merged desc_embedding_text for {merged} / {n} first recipes (title match).\n"
        f"  No match in slim file for {missing} of the first {n} recipes.",
    )


if __name__ == "__main__":
    main()
