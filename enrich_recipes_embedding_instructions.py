#!/usr/bin/env python3
"""
Enrich recipes: ing_embedding_text + instructions_standardized.

CRITICAL: ing_embedding_text is built ONLY from recipe["ingredients"].
Never use description, title, or instructions to build embeddings.

Pipeline rules
--------------
ing_embedding_text
  • Source: ingredients array ONLY (after filtering junk lines).
  • Map via unique_ingredient_names.json; no quantities/units in output tokens.
  • Reject tokens that still look like measurements (e.g. "2 cups low-fat").
  Dedupe, sort A–Z → "ingredients: a, b, c".

instructions_standardized
  • Default: deterministic unit cleanup + junk filtering (nutrition, ALL-CAPS titles, wine lines).
  • Optional ``--ollama-instructions``: one Ollama call per recipe rewrites steps using the
    ingredient list as ground truth; output is still passed through the deterministic unit pass.
  • Tbsp/tsp → tablespoon; mixed fractions: 2 1/2 cups → 2.5 cups; lb/oz → g where applicable.

Output ingredients array is filtered (non-food lines removed).

Ingredient normalization (recommended)
--------------------------------------
By default this script runs ``normalize_ingredients._process_single_recipe`` on each
recipe **first** (same pipeline as ``normalize_ingredients.py``), then builds
``ing_embedding_text`` and ``instructions_standardized`` from that result.

If your JSON is already normalized (e.g. from a prior ``normalize_ingredients.py`` run),
pass ``--already-normalized`` to skip that pass.

Usage:
  # Full pipeline: normalize ingredients → embedding + instructions
  python enrich_recipes_embedding_instructions.py --limit 20 --no-ollama

  # Input already normalized
  python enrich_recipes_embedding_instructions.py --limit 20 --already-normalized --no-ollama

  # Ollama rewrites instruction text (requires Ollama; uses ingredients as context)
  python enrich_recipes_embedding_instructions.py --limit 20 --ollama-instructions
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

from extract_unique_ingredient_names import ingredient_name_from_line
from main import DEFAULT_OLLAMA_MODEL, OllamaClient
from normalize_ingredients import (
    EQUIPMENT_WORDS,
    FINAL_UNITS,
    REST_ONLY_ADJECTIVE,
    _process_single_recipe,
    normalize_ingredient_deterministic,
)

QTY_RE = re.compile(r"^\d+(\.\d+)?$")

# ---------------------------------------------------------------------------
# Ingredient line filtering (not food / not for embedding)
# ---------------------------------------------------------------------------
_ING_JUNK_PREFIX = re.compile(
    r"^\s*(ingredient\s+info|accompaniments?|makes\b)",
    re.IGNORECASE,
)
_DISPOSABLE_OR_FOIL = re.compile(
    r"disposable|heavy[- ]duty\s+foil|aluminum\s+foil|"
    r"nonstick\s+cooking\s+spray|mini\s+foil|oven[- ]?safe|"
    r"loaf\s+pan|sheet\s+pan|baking\s+sheet|disposable\s+tray|aluminum\s+pan",
    re.IGNORECASE,
)
_CJK_IN_TEXT = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

# Post-mapping canonical names (embedding quality)
_EMBEDDING_CANONICAL_OVERRIDES: dict[str, str] = {
    "saltine": "saltine cracker",
    "saltine cracker": "saltine cracker",
    "crushed tomatoes": "tomato",
    "crushed tomato": "tomato",
    "cherry tomatoes": "cherry tomato",
}


def _apply_embedding_overrides(std: str) -> str:
    s = std.strip().lower()
    return _EMBEDDING_CANONICAL_OVERRIDES.get(s, s)

# Non-food / fuel — exclude from ing_embedding_text
_FUEL_OR_NONFOOD_EMBEDDING = re.compile(
    r"\b(hickory\s+wood|smoking\s+wood|wood\s+chips?|charcoal|mesquite\s+wood)\b",
    re.IGNORECASE,
)
# OCR / size-only lines (pans with dimensions)
_DIMENSION_GARBAGE = re.compile(
    r"\d+\s+\d+/\d+\s*x\s*\d+|inch.*\s+x\s*\d+|\d+\s*x\s*\d+.*inch",
    re.IGNORECASE,
)


def should_skip_ingredient_line(line: str) -> bool:
    """True if line is metadata, equipment, or not a real food ingredient."""
    s = line.strip()
    if not s or s.lower() == "none":
        return True
    if _CJK_IN_TEXT.search(s):
        return True
    low = s.lower()
    if _ING_JUNK_PREFIX.match(s):
        return True
    if _DISPOSABLE_OR_FOIL.search(s):
        return True
    if _DIMENSION_GARBAGE.search(s):
        return True

    n = normalize_ingredient_deterministic(s)
    if n is None:
        # Keep salt / pepper lines without quantities
        if not re.match(r"^\d", low):
            return False
        # Numbered line the normalizer could not parse → likely equipment / OCR junk
        if any(w in low for w in EQUIPMENT_WORDS):
            return True
        return True
    return False


def _food_phrase_from_normalized(norm: str) -> str:
    parts = norm.split()
    if len(parts) >= 3 and QTY_RE.match(parts[0]) and parts[1].lower():
        return " ".join(parts[2:]).strip()
    return norm.strip()


_EMBEDDING_DESC_PREFIX = re.compile(
    r"^(chopped|minced|diced|fresh|grated|roughly|finely)\s+",
    re.IGNORECASE,
)


def _canonicalize_embedding_fallback(name: str) -> str:
    """Strip common prep words when mapping has no entry (herbs, etc.)."""
    n = name.strip()
    prev = None
    while n != prev:
        prev = n
        n = _EMBEDDING_DESC_PREFIX.sub("", n).strip()
    return n if n else name


# Embedding names must be food words only — never measurement phrases
_EMBEDDING_BANNED_UNIT_WORDS = re.compile(
    r"\b("
    r"cup|cups|tablespoon|tablespoons|teaspoon|teaspoons|tbsp|tsp|tbs|"
    r"quart|quarts|pint|pints|gallon|ml|milliliter|milliliters|liter|litre|"
    r"lb|lbs|pound|pounds|ounce|ounces|oz\b|gram|grams|kg\b"
    r")\b",
    re.IGNORECASE,
)
# Bare units / junk tokens (not food names)
_EMBEDDING_BLOCKLIST = frozenset(
    {
        "piece",
        "slice",
        "clove",
        "cups",
        "cup",
        "cuppulse",
        "pinch",
        "dash",
        "bunch",
        "stalk",
    }
)
_EQUIPMENT_SUBSTRINGS = re.compile(
    r"disposable\s+(?:aluminum|tray|pan)|aluminum\s+foil|heavy[- ]duty\s+foil|mini\s+foil|"
    r"loaf\s+pan|sheet\s+pan|baking\s+sheet|aluminum\s+pan|oven[- ]?safe|\bskillet\b",
    re.IGNORECASE,
)
# "cuppulse" / "cupflour" style OCR
_EMBEDDING_CUP_GARBAGE = re.compile(r"cup(?=[a-z]{2,})", re.IGNORECASE)


def is_valid_embedding_name(name: str) -> bool:
    """Reject quantity-like or measurement-like tokens (e.g. '2 cups low-fat')."""
    t = name.strip().lower()
    if not t or len(t) < 2:
        return False
    if _CJK_IN_TEXT.search(t):
        return False
    if t in _EMBEDDING_BLOCKLIST:
        return False
    if t in ("unknown", "none", "optional"):
        return False
    if re.match(r"^\d", t):
        return False
    if re.search(r"\d", t):  # digits anywhere → likely leaked qty (e.g. 2 cups low-fat)
        return False
    if _EMBEDDING_BANNED_UNIT_WORDS.search(t):
        return False
    if _EMBEDDING_CUP_GARBAGE.search(t):  # cuppulse, cupflour
        return False
    if _EQUIPMENT_SUBSTRINGS.search(t):
        return False
    if re.search(r"\d+\s*/\s*\d+", t):
        return False
    if _DIMENSION_GARBAGE.search(t):
        return False
    first = t.split(",")[0].strip()
    if first in REST_ONLY_ADJECTIVE and len(t.split()) <= 3:
        return False
    if re.search(r"^low-fat|low fat$|full-fat|reduced-fat", t):
        return False
    if _FUEL_OR_NONFOOD_EMBEDDING.search(t):
        return False
    return True


def map_ingredient_to_standard(ing_line: str, mapping: dict[str, str]) -> str | None:
    """Map one line to canonical name, or None if should not appear in embedding."""
    line_lower = ing_line.lower().strip()
    # Already-normalized lines from normalize_ingredients.py: [qty] [FINAL_UNIT] [rest]
    parts = ing_line.split()
    norm: str | None = None
    if (
        len(parts) >= 3
        and QTY_RE.match(parts[0])
        and parts[1].lower() in FINAL_UNITS
    ):
        norm = ing_line.strip()
    else:
        norm = normalize_ingredient_deterministic(ing_line)

    candidate = ""
    if norm:
        candidate = _food_phrase_from_normalized(norm).lower()
    if not candidate:
        raw = ingredient_name_from_line(ing_line).lower()
        candidate = raw
    if not candidate:
        candidate = re.sub(
            r"^(\d+\s+\d+/\d+|\d+/\d+|\d+\.?\d*)\s+",
            "",
            ing_line,
            flags=re.IGNORECASE,
        ).strip().lower()

    best_val: str | None = None
    best_key_len = -1
    for k, v in mapping.items():
        kl = k.lower()
        if kl in line_lower or (candidate and kl in candidate):
            if len(k) > best_key_len:
                best_key_len = len(k)
                best_val = v

    if best_val:
        std = _apply_embedding_overrides(best_val.strip().lower())
        return std if is_valid_embedding_name(std) else None

    for k, v in sorted(mapping.items(), key=lambda x: -len(x[0])):
        if k.lower() in line_lower:
            std = _apply_embedding_overrides(v.strip().lower())
            return std if is_valid_embedding_name(std) else None

    if candidate:
        std = _canonicalize_embedding_fallback(candidate.split(",")[0].strip().lower())
        std = _apply_embedding_overrides(std)
        return std if is_valid_embedding_name(std) else None
    return None


def filter_ingredient_lines(ingredients: list) -> list[str]:
    """Drop non-food / metadata lines from the ingredient list."""
    out: list[str] = []
    for ing in ingredients or []:
        if not ing or not isinstance(ing, str):
            continue
        if should_skip_ingredient_line(ing.strip()):
            continue
        out.append(ing.strip())
    return out


_SINGULAR_KEEP_PLURAL = frozenset(
    {
        "oats",
        "greens",
        "noodles",
        "herbs",
        "sprouts",
        "molasses",
        "water",
        "couscous",
        "seeds",
    }
)
_IRREGULAR_PLURAL = {
    "tomatoes": "tomato",
    "potatoes": "potato",
    "leaves": "leaf",
    "lentils": "lentil",
    "eggs": "egg",
    "chickpeas": "chickpea",
    "blueberries": "blueberry",
    "raspberries": "raspberry",
    "strawberries": "strawberry",
    "cherries": "cherry",
    "peaches": "peach",
}


def singularize_embedding_phrase(phrase: str) -> str:
    """Plural → singular for cleaner embedding tokens (tomatoes → tomato)."""
    words = phrase.lower().split()
    out: list[str] = []
    for w in words:
        if w in _IRREGULAR_PLURAL:
            out.append(_IRREGULAR_PLURAL[w])
            continue
        if w in _SINGULAR_KEEP_PLURAL:
            out.append(w)
            continue
        if w.endswith("oes") and len(w) > 3:
            out.append(w[:-3] + "o")
        elif w.endswith("ies") and len(w) > 3:
            out.append(w[:-3] + "y")
        elif w.endswith("ses") and len(w) > 3 and w not in ("molasses",):
            out.append(w[:-2])
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 2:
            out.append(w[:-1])
        else:
            out.append(w)
    return " ".join(out)


def build_ing_embedding_text(ingredients: list[str], mapping: dict[str, str]) -> str:
    """
    Build embedding string from the ingredients list ONLY (never from instructions).
    ingredients must already be filtered (see filter_ingredient_lines).
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for ing in ingredients:
        std = map_ingredient_to_standard(ing, mapping)
        if not std:
            continue
        std = singularize_embedding_phrase(std)
        std = _apply_embedding_overrides(std)
        if not std or not is_valid_embedding_name(std):
            continue
        if std not in seen:
            seen.add(std)
            ordered.append(std)
    ordered.sort()
    return "ingredients: " + ", ".join(ordered)


# ---------------------------------------------------------------------------
# Instructions: deterministic only for volume; mass lb/oz → g; °F untouched
# ---------------------------------------------------------------------------
_NUTRITION_LINE = re.compile(
    r"^\s*(calories|total\s+fat|saturated\s+fat|trans\s+fat|cholesterol|sodium|"
    r"total\s+carbohydrate|carbohydrates?|dietary\s+fiber|fiber|sugars?|protein|vitamin|calcium|iron)\b",
    re.IGNORECASE,
)


def remove_nutrition_and_metadata_lines(text: str) -> str:
    """Drop nutrition facts and similar non-step lines."""
    lines = []
    for line in text.splitlines():
        t = line.strip()
        if _NUTRITION_LINE.match(t):
            continue
        if re.match(r"^\s*(calories|protein|fat|fiber|sodium|cholesterol|carbohydrates?)\s*:\s*[\d.]",
                    t, re.I):
            continue
        if re.match(r"^\s*each\s+.*\bserving\s+has:?\s*$", t, re.I):
            continue
        if re.match(r"^\s*per\s+serving\s*:?\s*$", line, re.I):
            continue
        lines.append(line)
    return " ".join(lines).strip()


def is_instruction_step_junk(text: str) -> bool:
    """
    True for non-cooking lines: nutrition, per-serving, ALL-CAPS headers, wine notes, fragments.
    """
    t = text.strip()
    if not t:
        return True
    low = t.lower()
    if _NUTRITION_LINE.match(t):
        return True
    if re.match(
        r"^\s*(calories|protein|total\s+fat|fat|fiber|sodium|cholesterol|carbohydrates?)\s*:\s*",
        t,
        re.I,
    ):
        return True
    if re.search(r"\bper\s+[^.\n]{0,50}?serving\b", low):
        return True
    if re.match(r"^\s*each\s+.*\bserving\s+has", low):
        return True
    letters_only = "".join(c for c in t if c.isalpha())
    if len(letters_only) > 12 and letters_only.isupper():
        return True
    if re.match(r"^un\s*$", low):
        return True
    if len(t) < 120 and re.search(
        r"\b(wine|vintage|vineyard|pairing|nemea|bordeaux|chianti|champagne|riesling)\b",
        low,
    ):
        if re.search(r"'[0-9]{2}\b|[\"']?[12][0-9]{3}\b", t):
            return True
    if len(t) < 90 and re.search(r"\b(nemea|château|chateau|domaine)\b", low):
        return True
    # Wine / vintage credit lines (e.g. "Haggipavlu Nemea '04")
    if len(t) < 120 and re.search(r"'[0-9]{2}\s*$", t):
        if not re.search(
            r"\b(heat|preheat|bake|oven|mix|stir|add|bring|boil|simmer|cup|tablespoon|teaspoon|minute)\b",
            low,
        ):
            return True
    return False


def strip_inline_nutrition_spans(text: str) -> str:
    """Remove Calories: / Protein: / etc. that appear inside one paragraph (not only line-start)."""
    s = text
    s = re.sub(
        r"\s*Each\s+\([^)]*\)\s+serving\s+has:\s*",
        " ",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\s*\(?\s*Per\s+[^.)\n]{0,40}?serving\s*\)?\s*",
        " ",
        s,
        flags=re.IGNORECASE,
    )
    for _ in range(8):
        s2 = re.sub(
            r"\s*(?:Calories|Protein|Total\s+Fat|Fat|Carbohydrates|Dietary\s+Fiber|Fiber|Sodium|"
            r"Cholesterol|Saturated\s+Fat|Sugar)\s*:\s*[\d.]+\s*(?:g|mg|kcal|%)?(?:\s*each)?",
            " ",
            s,
            flags=re.IGNORECASE,
        )
        if s2 == s:
            break
        s = s2
    return s


def remove_instruction_non_step_parentheticals(text: str) -> str:
    """Remove (makes 1 cup), (see Note …) style notes — not cooking actions."""
    s = text
    s = re.sub(r"\(\s*makes\b[^)]*\)", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\(\s*see\s+note[^)]*\)", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\(\s*optional[^)]*\)", " ", s, flags=re.IGNORECASE)
    return s


def expand_instruction_abbreviations(text: str) -> str:
    s = text
    s = re.sub(r"(?<![A-Za-z])Tbsp\.?(?![A-Za-z])", "tablespoon", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<![A-Za-z])tsp\.?(?![A-Za-z])", "teaspoon", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<![A-Za-z])tbs\.?(?![A-Za-z])", "tablespoon", s, flags=re.IGNORECASE)
    return s


def convert_cup_mixed_and_simple_fractions(s: str) -> str:
    """
    2 1/2 cups → 2.5 cups (mixed number + fraction first).
    Then 1/2 cup → 0.5 cup. Never leave 2 0.5 cups (whole + converted simple fraction).
    """
    # Mixed: "2 1/2 cups" / "2  1/2 cup" (NBSP ok via \s)
    def mixed_repl(m: re.Match) -> str:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if den == 0:
            return m.group(0)
        v = whole + num / den
        vs = str(round(v, 6)).rstrip("0").rstrip(".")
        unit = m.group(4).lower()
        return f"{vs} {unit}"

    s = re.sub(
        r"\b(\d+)\s+(\d+)\s*/\s*(\d+)\s+(cup|cups)\b",
        mixed_repl,
        s,
        flags=re.IGNORECASE,
    )

    def simple_repl(m: re.Match) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            return m.group(0)
        v = a / b
        vs = str(round(v, 6)).rstrip("0").rstrip(".")
        unit = m.group(3).lower()
        return f"{vs} {unit}"

    s = re.sub(
        r"\b(\d+)\s*/\s*(\d+)\s+(cup|cups)\b",
        simple_repl,
        s,
        flags=re.IGNORECASE,
    )
    return s


def fix_stray_integer_plus_decimal_cups(s: str) -> str:
    """Repair '2 0.5 cups' → '2.5 cups' if a prior pass left whole + decimal fragment."""

    def repl(m: re.Match) -> str:
        a, b, unit = float(m.group(1)), float(m.group(2)), m.group(3).lower()
        v = a + b
        vs = str(round(v, 6)).rstrip("0").rstrip(".")
        return f"{vs} {unit}"

    return re.sub(
        r"\b(\d+)\s+(0\.\d+)\s+(cup|cups)\b",
        repl,
        s,
        flags=re.IGNORECASE,
    )


def convert_instruction_mass_only(s: str) -> str:
    """
    lb / pound → g (or kg); oz → g. Skips fl oz.
    Does not touch °F or volume measures.
    """

    def lb_repl(m: re.Match) -> str:
        n = float(m.group(1))
        g = n * 500.0
        if g >= 1000:
            kg = round(g / 1000, 3)
            ks = str(kg).rstrip("0").rstrip(".")
            return f"{ks} kg"
        ig = int(round(g))
        return f"{ig} g"

    s = re.sub(
        r"(\d+(?:\.\d+)?)\s*(lb|lbs|pound|pounds)\b",
        lb_repl,
        s,
        flags=re.IGNORECASE,
    )

    def oz_repl(m: re.Match) -> str:
        full = m.string
        start = m.start()
        before = full[:start].lower()
        if before.rstrip().endswith("fl") or before.rstrip().endswith("fl."):
            return m.group(0)
        n = float(m.group(1))
        g = n * 30.0
        ig = int(round(g))
        return f"{ig} g"

    s = re.sub(
        r"(\d+(?:\.\d+)?)\s*(oz|ounce|ounces)\b",
        oz_repl,
        s,
        flags=re.IGNORECASE,
    )
    return s


def standardize_instruction_step(raw: str) -> str:
    """
    Deterministic unit cleanup only — never call LLMs here (avoids hallucinated steps).
    """
    s = remove_nutrition_and_metadata_lines(raw)
    s = strip_inline_nutrition_spans(s)
    s = remove_instruction_non_step_parentheticals(s)
    s = expand_instruction_abbreviations(s)
    s = convert_cup_mixed_and_simple_fractions(s)
    s = fix_stray_integer_plus_decimal_cups(s)
    s = convert_instruction_mass_only(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_json_from_llm(text: str) -> dict | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        d = json.loads(raw)
        if isinstance(d, dict):
            return d
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    return None


def _build_instructions_deterministic_from_pairs(
    raw_steps: list[tuple[str, str]],
) -> dict[str, str]:
    steps_out: list[str] = []
    for _k, v in raw_steps:
        step = standardize_instruction_step(v)
        if step.strip():
            steps_out.append(step)
    return {str(i + 1): steps_out[i] for i in range(len(steps_out))}


def ollama_rewrite_instruction_block(
    client: OllamaClient,
    model: str,
    ingredients: list[str],
    numbered_steps: list[tuple[str, str]],
) -> dict[str, str] | None:
    """
    One Ollama call per recipe: rewrite all steps using ingredients as ground truth.
    Falls back to None if parse fails (caller uses deterministic path).
    """
    if not numbered_steps:
        return {}
    ing_block = "\n".join(f"- {x}" for x in ingredients)
    steps_block = "\n".join(f"{i + 1}. {v}" for i, (_a, v) in enumerate(numbered_steps))
    prompt = f"""You clean and rewrite recipe instructions for a database.

INGREDIENTS (authoritative — use these names and units; do not add ingredients that are not listed):
{ing_block}

INSTRUCTION STEPS (may include junk: Calories, Protein, "Per serving", ALL-CAPS titles, wine notes, credits — remove ALL non-cooking content):
{steps_block}

Return ONLY a JSON object with keys "1", "2", "3", ... and string values: the final cooking steps in order.
Rules:
- Output ONLY real cooking steps (prepare, heat, bake, mix, season, serve, etc.).
- Do NOT include nutrition facts, "Per ... serving", wine pairings, vintages, credits, or section headers.
- Do NOT invent ingredients or a different recipe; stay faithful to the intended dish.
- Keep Fahrenheit temperatures as written (e.g. 400°F).
- JSON only — no markdown fences, no commentary."""

    out = client.generate_response(model, prompt, stream=False)
    if not out:
        return None
    parsed = _extract_json_from_llm(out)
    if not parsed:
        return None
    ordered_keys = sorted(
        (str(k) for k in parsed.keys()),
        key=lambda x: int(x) if x.isdigit() else 0,
    )
    cleaned: list[str] = []
    for k in ordered_keys:
        v = parsed.get(k)
        if isinstance(v, str) and v.strip():
            cleaned.append(standardize_instruction_step(v))
    if not cleaned:
        return None
    return {str(i + 1): cleaned[i] for i in range(len(cleaned))}


def ordered_recipe_output(
    recipe: dict,
    ingredients_clean: list[str],
    ing_embedding_text: str,
    instructions_standardized: dict | None,
) -> dict:
    """Stable key order: id, title, description, ingredients, instructions, ing_embedding_text, instructions_standardized, then rest."""
    preferred = (
        "id",
        "title",
        "description",
        "ingredients",
        "instructions",
        "ing_embedding_text",
        "instructions_standardized",
    )
    out: dict = {}
    for k in preferred:
        if k == "ingredients":
            out[k] = ingredients_clean
        elif k == "ing_embedding_text":
            out[k] = ing_embedding_text
        elif k == "instructions_standardized":
            out[k] = instructions_standardized
        elif k in recipe:
            out[k] = recipe[k]
    for k, v in recipe.items():
        if k not in out:
            out[k] = v
    return out


def load_unique_names_and_mapping(path: Path) -> tuple[list[str], dict[str, str]]:
    """Load names array + ingredient map from commented JSON file."""
    text = path.read_text(encoding="utf-8")
    if "//ingredient name map" not in text:
        data = json.loads(re.sub(r"\s*//.*$", "", text, flags=re.MULTILINE))
        return data.get("names", []), {}

    a, b = text.split("//ingredient name map", 1)

    def strip_slash_comments(s: str) -> str:
        return "\n".join(re.sub(r"\s*//.*$", "", ln) for ln in s.splitlines())

    data1 = json.loads(strip_slash_comments(a))
    lines2: list[str] = []
    for line in b.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        line = re.sub(r"\s*#.*$", "", line)
        if line.strip():
            lines2.append(line)
    mapping = json.loads("\n".join(lines2))
    return data1.get("names", []), mapping


def process_recipe(
    recipe: dict,
    mapping: dict[str, str],
    client: OllamaClient | None,
    model: str,
    use_ollama: bool,
    use_ollama_instructions: bool = False,
) -> dict:
    # Embeddings: ONLY from ingredients list — never description/title/instructions
    raw_ings = recipe.get("ingredients") or []
    ingredients_clean = filter_ingredient_lines(raw_ings)
    ing_emb = build_ing_embedding_text(ingredients_clean, mapping)

    inst = recipe.get("instructions")
    instructions_std: dict | str | None = None
    if isinstance(inst, dict):

        def _step_key(k: str) -> int:
            try:
                return int(str(k))
            except ValueError:
                return 0

        raw_steps: list[tuple[str, str]] = []
        for k in sorted(inst.keys(), key=_step_key):
            v = inst[k]
            if not isinstance(v, str):
                continue
            if is_instruction_step_junk(v):
                continue
            raw_steps.append((str(k), v))

        if use_ollama_instructions and client and raw_steps:
            ollama_inst = ollama_rewrite_instruction_block(
                client, model, ingredients_clean, raw_steps
            )
            if ollama_inst is not None:
                instructions_std = ollama_inst
            else:
                instructions_std = _build_instructions_deterministic_from_pairs(raw_steps)
        else:
            instructions_std = _build_instructions_deterministic_from_pairs(raw_steps)
    else:
        instructions_std = inst

    return ordered_recipe_output(
        recipe,
        ingredients_clean,
        ing_emb,
        instructions_std,
    )


def _progress_bar_line(current: int, total: int, *, width: int = 28) -> str:
    """ASCII progress bar + completed count for terminal display."""
    if total <= 0:
        return "completed 0/0"
    filled = min(width, int(width * current / total))
    bar = "#" * filled + "-" * (width - filled)
    pct = 100.0 * current / total
    return f"[{bar}] completed {current}/{total} ({pct:.0f}%)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich recipes (embedding text + instruction cleanup).")
    ap.add_argument("--input", "-i", default="new_recipee_data.json", help="Recipes JSON (array)")
    ap.add_argument("--output", "-o", default="new_recipee_data_enriched_first20.json", help="Output JSON")
    ap.add_argument("--mapping", "-m", default="unique_ingredient_names.json", help="Names + mapping file")
    ap.add_argument("--limit", "-n", type=int, default=20, help="Only process first N recipes")
    ap.add_argument(
        "--keep-rest",
        action="store_true",
        help="Append remaining recipes unchanged after enriched ones",
    )
    ap.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model (normalize + enrich)")
    ap.add_argument(
        "--already-normalized",
        action="store_true",
        help="Skip normalize_ingredients pass (input ingredients already normalized)",
    )
    ap.add_argument("--no-ollama", action="store_true", help="Never call Ollama (deterministic normalize only)")
    ap.add_argument(
        "--ollama-instructions",
        action="store_true",
        help="Rewrite all instruction steps with Ollama using ingredients as context (drops junk; needs Ollama)",
    )
    ap.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="No progress line (only final summary)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    inp = root / args.input
    map_path = root / args.mapping
    if not inp.exists():
        print(f"Not found: {inp}", file=sys.stderr)
        sys.exit(1)
    if not map_path.exists():
        print(f"Not found: {map_path}", file=sys.stderr)
        sys.exit(1)

    _, mapping = load_unique_names_and_mapping(map_path)

    with open(inp, encoding="utf-8") as f:
        recipes = json.load(f)

    if not isinstance(recipes, list):
        print("Expected JSON array of recipes.", file=sys.stderr)
        sys.exit(1)

    n = max(0, args.limit)
    head = recipes[:n]
    rest = recipes[n:]

    client: OllamaClient | None = None
    use_ollama = not args.no_ollama
    if use_ollama:
        client = OllamaClient()
        if not client.check_connection():
            print("Ollama not reachable; using deterministic steps only.", file=sys.stderr)
            use_ollama = False
            client = None
        else:
            print(f"Using Ollama model: {args.model}", flush=True)

    use_ollama_instructions = bool(args.ollama_instructions and client)
    if args.ollama_instructions and not client:
        print(
            "Warning: --ollama-instructions ignored (no Ollama). Use deterministic instruction cleanup.",
            file=sys.stderr,
        )
    if use_ollama_instructions:
        print("Instruction mode: Ollama rewrite (ingredients as context).", flush=True)

    if not args.already_normalized:
        print("Running normalize_ingredients._process_single_recipe on each recipe first…", flush=True)

    enriched: list = []
    total = len(head)
    for idx, r in enumerate(head, start=1):
        work = copy.deepcopy(r)
        if not args.already_normalized:
            work, _stats = _process_single_recipe(work, client, use_ollama, args.model)
        enriched.append(
            process_recipe(
                work,
                mapping,
                client,
                args.model,
                use_ollama,
                use_ollama_instructions=use_ollama_instructions,
            )
        )
        if not args.quiet and total:
            print(f"\rEnriching {_progress_bar_line(idx, total)}", end="", file=sys.stderr, flush=True)
    if not args.quiet and total:
        print(file=sys.stderr)
    out_data = enriched + rest if args.keep_rest else enriched

    out_path = root / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(out_data)} recipe(s) ({n} enriched) → {out_path}")


if __name__ == "__main__":
    main()
