#!/usr/bin/env python3
"""
Normalize recipe ingredients to a fixed set of units using conversion rules.
Pipeline: extract → alias → plural → conversion → round → format → validate.
LLM used only when quantity/unit missing or ingredient unclear (not for units).
Run with: conda activate miniproj && python normalize_ingredients.py
Default Ollama model: qwen2.5:7b-q4_K_M →  ollama pull qwen2.5:7b-q4_K_M
If that tag is unavailable, use: ollama pull qwen2.5:7b

Batching & resume:
  Full run with checkpoints every 25 recipes (default):
    python normalize_ingredients.py --input recipes_images.json --output out.json
  Live progress (TTY): elapsed time, ETA, counts; stale *.json.tmp next to --output is removed on start.
  Resume after Ctrl+C (same paths + batch + sample):
    python normalize_ingredients.py --input recipes_images.json --output out.json --resume
  Fresh start (discard partial output + checkpoint): rm -f out.json out.json.checkpoint.json out.json.tmp
  Process only indices [5000, 10000):
    python normalize_ingredients.py --batch-start 5000 --batch-end 10000 -o part1.json
  Merge batch files: python merge_recipe_batches.py -o full.json part0.json part1.json
  Long Ollama calls: set OLLAMA_GENERATE_TIMEOUT=1200 (seconds read timeout; see main.py).
"""
import copy
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Single source of truth: only these units may appear in output
# ---------------------------------------------------------------------------
FINAL_UNITS = frozenset({
    "mg", "g", "kg",
    "ml", "l", "teaspoon", "tablespoon", "cup",
    "piece", "slice", "clove", "bunch", "stalk", "sprig", "leaf", "cube", "dozen",
    "pinch", "drop",
    "packet", "bag", "box", "jar", "bottle", "carton", "tin", "sachet",
    "handful", "scoop", "serving",
})

# ---------------------------------------------------------------------------
# 1. Alias layer: spelling variants and abbreviations → canonical unit name
# ---------------------------------------------------------------------------
UNIT_ALIASES = {
    "tsp": "teaspoon", "tsps": "teaspoon", "teaspoon": "teaspoon", "teaspoons": "teaspoon",
    "t": "teaspoon",
    "tbsp": "tablespoon", "tbs": "tablespoon", "tbl": "tablespoon", "tbsps": "tablespoon",
    "tablespoon": "tablespoon", "tablespoons": "tablespoon",
    "c": "cup", "cup": "cup", "cups": "cup",
    "oz": "ounce", "ounce": "ounce", "ounces": "ounce",
    "lb": "pound", "lbs": "pound", "pound": "pound", "pounds": "pound",
    "g": "g", "gram": "g", "grams": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml",
    "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "fl oz": "fluid_ounce", "fluid ounce": "fluid_ounce", "fluid ounces": "fluid_ounce",
    "pint": "pint", "pints": "pint", "pt": "pint",
    "quart": "quart", "quarts": "quart", "qt": "quart",
    "gallon": "gallon", "gallons": "gallon", "gal": "gallon",
    "dash": "dash", "dashes": "dash",
    "drop": "drop", "drops": "drop",
    "pinch": "pinch", "pinches": "pinch",
    "can": "can", "cans": "can",
    "package": "packet", "packages": "packet",
    "packet": "packet", "packets": "packet",
    "bag": "bag", "bags": "bag",
    "box": "box", "boxes": "box",
    "jar": "jar", "jars": "jar",
    "bottle": "bottle", "bottles": "bottle",
    "envelope": "sachet", "envelopes": "sachet",
    "sachet": "sachet", "sachets": "sachet",
    "tin": "tin", "tins": "tin",
    "carton": "carton", "cartons": "carton",
    "piece": "piece", "pieces": "piece",
    "slice": "slice", "slices": "slice",
    "clove": "clove", "cloves": "clove",
    "bunch": "bunch", "bunches": "bunch",
    "stalk": "stalk", "stalks": "stalk",
    "sprig": "sprig", "sprigs": "sprig",
    "leaf": "leaf", "leaves": "leaf",
    "cube": "cube", "cubes": "cube",
    "dozen": "dozen", "dozens": "dozen",
    "bulb": "piece", "bulbs": "piece",
    "head": "piece", "heads": "piece",
    "fillet": "piece", "fillets": "piece",
    "stick": "piece", "sticks": "piece",
    "strip": "piece", "strips": "piece",
    "link": "piece", "links": "piece",
    "loaf": "piece", "loaves": "piece",
    "bar": "piece", "bars": "piece",
    "square": "piece", "squares": "piece",
    "inch": "piece", "inches": "piece",
    "sheet": "piece", "sheets": "piece",
    "handful": "handful", "handfuls": "handful",
    "scoop": "scoop", "scoops": "scoop",
    "serving": "serving", "servings": "serving",
}

# ---------------------------------------------------------------------------
# 2. Conversion: canonical unit → (final_unit, multiplier)
# ---------------------------------------------------------------------------
CONVERSION_MAP = {
    "teaspoon": ("teaspoon", 1),
    "tablespoon": ("tablespoon", 1),
    "cup": ("cup", 1),
    "ounce": ("g", 30),
    "pound": ("g", 500),
    "g": ("g", 1),
    "kg": ("kg", 1),
    "fluid_ounce": ("tablespoon", 2),
    "pint": ("cup", 2),
    "quart": ("cup", 4),
    "gallon": ("l", 4),
    "ml": ("ml", 1),
    "l": ("l", 1),
    "dash": ("ml", 0.5),
    "drop": ("drop", 1),
    "pinch": ("pinch", 1),
    "can": ("tin", 1),
    "packet": ("packet", 1),
    "bag": ("bag", 1),
    "box": ("box", 1),
    "jar": ("jar", 1),
    "bottle": ("bottle", 1),
    "sachet": ("sachet", 1),
    "tin": ("tin", 1),
    "carton": ("carton", 1),
    "piece": ("piece", 1),
    "slice": ("slice", 1),
    "clove": ("clove", 1),
    "bunch": ("bunch", 1),
    "stalk": ("stalk", 1),
    "sprig": ("sprig", 1),
    "leaf": ("leaf", 1),
    "cube": ("cube", 1),
    "dozen": ("dozen", 1),
    "handful": ("handful", 1),
    "scoop": ("scoop", 1),
    "serving": ("serving", 1),
}

# Map any alias/variant → final unit (for post-fixing LLM output e.g. "tablespoons" → "tablespoon")
ALIAS_TO_FINAL = {}
for alias, canonical in UNIT_ALIASES.items():
    conv = CONVERSION_MAP.get(canonical)
    if conv:
        ALIAS_TO_FINAL[alias.lower()] = conv[0]

# All unit tokens we can match in text (for regex); prefer longer first
_all_match_tokens = sorted(
    set(UNIT_ALIASES.keys()) | set(k.replace("_", " ") for k in CONVERSION_MAP if "_" in k),
    key=len,
    reverse=True,
)
UNIT_MATCH_PATTERN = "|".join(re.escape(t) for t in _all_match_tokens)

# ---------------------------------------------------------------------------
# 1b. Range normalization: "2–3 cups" → "2.5 cups" (before parsing)
# ---------------------------------------------------------------------------
RANGE_HYPHEN_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")
RANGE_TO_RE = re.compile(r"(\d+)\s+to\s+(\d+)", re.IGNORECASE)


def normalize_ranges(text: str) -> str:
    """Convert N–M or N to M to midpoint so regex can parse."""
    def repl(m):
        a, b = float(m.group(1)), float(m.group(2))
        return str(round((a + b) / 2, 2))
    text = RANGE_HYPHEN_RE.sub(repl, text)
    text = RANGE_TO_RE.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# 5. Unicode fraction normalization (before parsing)
# ---------------------------------------------------------------------------
UNICODE_FRACTIONS = {
    "½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3",
    "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
    "⅙": "1/6", "⅚": "5/6", "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
}

# Regex: number + unit + rest, or number + rest + unit
FRACTION_RE = re.compile(
    r"^(?P<num>\d+\s+\d+/\d+|\d+/\d+|\d+\.?\d*)\s+"
    r"(?P<unit>" + UNIT_MATCH_PATTERN + r")"
    r"(?:\s+|$|\s*\(|\s*,)"
    r"(?P<rest>.*)$",
    re.IGNORECASE
)
# Unit-last: "2 garlic cloves", "2 garlic cloves, chopped" → rest=garlic, unit=cloves; allow trailing ", ..."
TAIL_UNIT_RE = re.compile(
    r"^(?P<num>\d+\s+\d+/\d+|\d+/\d+|\d+\.?\d*)\s+"
    r"(?P<rest>.+?)\s+"
    r"(?P<unit>" + UNIT_MATCH_PATTERN + r")"
    r"\s*(?:,.*)?$",
    re.IGNORECASE
)
# Quantity but no unit: "2 onions", "0.5 onion" (after fraction normalization)
NUM_ONLY_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)\s+(?P<rest>.+)$", re.IGNORECASE)
# Parenthetical removal: weight (ounce/pound), then volume (pint/quart/cup)
PAREN_SIZE_RE = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*[-–]\s*(ounce|pound|oz|lb)s?\s*\)", re.IGNORECASE)
PAREN_WEIGHT_RE = re.compile(r"\(\s*(?:about\s+)?(\d+(?:\.\d+)?)\s*(ounce|pound|oz|lb)s?(?:\s+each)?\s*\)", re.IGNORECASE)
# Volume in parens: "(1 1/2 pints)", "(2 cups)" — remove so we don't keep "3 cup ... (1 1/2 pints)"
PAREN_VOLUME_RE = re.compile(
    r"\(\s*(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)\s*(cup|cups|pint|pints|quart|quarts|ml|l)s?\s*\)",
    re.IGNORECASE
)
# Container + weight: "N (M-ounce) rest" or "N piece M-ounce can/log rest" → use weight in g
PAREN_WEIGHT_FULL_RE = re.compile(
    r"^(\d+)\s*\(\s*(\d+(?:\.\d+)?)\s*[-–]\s*(ounce|pound|oz|lb)s?\s*\)\s*(.*)$",
    re.IGNORECASE
)
LEADING_WEIGHT_RE = re.compile(
    r"^(\d+)\s+(?:piece\s+)?(\d+(?:\.\d+)?)\s*[-–]\s*(ounce|pound|oz|lb)s?\s+(?:can|cans|log|package|tin|tins)?\s*(.+)$",
    re.IGNORECASE
)
# Number words for "three 6-ounce cans" etc.
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
# 1. "Two 6-ounce cans tuna", "three 6-ounce cans tomatoes" → count × oz × 30 g
N_OUNCE_CANS_RE = re.compile(
    r"^(one|two|three|four|five|six|\d+)\s+(\d+)\s*[-–]?\s*ounce\s+cans?\s+(.+)$",
    re.IGNORECASE
)
# 2. "25-ounce package", "8 oz cream cheese", "25 oz package frozen ravioli" → weight in g
INLINE_OUNCE_RE = re.compile(
    r"^(?:one\s+|\d+\s+)?(\d+)\s*[-–]?\s*(?:ounce|ounces?|oz)\s+(?:package|can|box|container)?\s*(.+)$",
    re.IGNORECASE
)
# Words to strip from rest when we output weight (container/shape)
CONTAINER_WORDS = frozenset({"can", "cans", "log", "package", "tin", "tins", "bag", "bags", "box", "boxes"})
# Size descriptors inside ingredient names (not units): "2-inch-diameter" → strip so we don't parse "inch"
INCH_SIZE_RE = re.compile(r"\d+(?:\.\d+)?\s*[-–]?\s*inch(?:es)?\s*[-–]?\s*diameter", re.IGNORECASE)
# "piece of X" after stripping size → "1 piece X"
PIECE_OF_RE = re.compile(r"^\s*piece\s+of\s+", re.IGNORECASE)
# Cooking equipment: not ingredients — drop lines like "12 cup mini aluminum loaf pan"
EQUIPMENT_WORDS = ("pan", "foil", "skillet", "sheet")
# Max words in ingredient name (rest) to avoid garbage like "5 g piece slice clove bunch ..."
MAX_REST_WORDS = 6
# Strip from start of ingredient name (rest) for cleaner ML output
SIZE_WORDS = frozenset({"small", "medium", "large"})
SHAPE_WORDS = frozenset({"slab", "chunk", "piece", "block"})
# Adjective/descriptor words to strip for ML (e.g. "4 piece skinless salmon" → "4 piece salmon")
# Keep chopped/minced/diced for richer ML features
DESCRIPTOR_WORDS = frozenset({
    "finely", "roughly", "coarsely", "thinly", "thickly", "sliced",
    "grated", "julienned", "slivered", "crushed", "dried", "fresh", "optional",
    "skinless", "boneless", "frozen",
})
# Single-word rest = stripped too much (e.g. "1 cup green" from "1 cup green split peas") — reject
REST_ONLY_ADJECTIVE = frozenset({"green", "red", "white"})
# Singularize last noun only for count-based units; volume/mass keep plural (e.g. 3 cup cherry tomatoes)
COUNT_UNITS_SINGULARIZE = frozenset({"piece", "slice", "clove", "leaf", "sprig", "stalk", "cube"})
# Canonical ingredient names for ML vocabulary (applied after _clean_rest)
INGREDIENT_ALIASES = {
    "parmigiano-reggiano": "parmesan cheese",
    "scallion": "green onion",
    "aubergine": "eggplant",
}
# Rest ending in these = missing noun (e.g. "2 cup low-fat" from "low-fat mozzarella")
FAT_ONLY_ENDINGS = frozenset({"low-fat", "reduced-fat", "fat-free", "full-fat"})
# Unit tokens for "plus N unit" / "and N unit" stripping (compound units)
_COMPOUND_UNIT_PATTERN = r"(?:teaspoons?|tablespoons?|tsp|tbsp|cups?|g|kg|ml|l)\b"

# Default LLM for ambiguous lines (quantized Qwen 7B — lower RAM / faster on CPU vs Llama 3.2)
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

# Log invalid output units for review
log = logging.getLogger("normalize_ingredients")
if not log.handlers:
    log.addHandler(logging.StreamHandler(sys.stderr))
    log.setLevel(logging.WARNING)


def normalize_unicode_fractions(s: str) -> str:
    """Replace ½ ¼ ¾ etc. with ASCII fractions before parsing."""
    for char, replacement in UNICODE_FRACTIONS.items():
        s = s.replace(char, replacement)
    return s


def parse_fraction(s: str) -> float | None:
    """Parse '1', '1/2', '1 1/2', '2/3' -> float. Expects ASCII (use normalize_unicode_fractions first)."""
    s = s.strip()
    if not s:
        return None
    # "1 1/2" or "2 1/4"
    m = re.match(r"^(\d+)\s+(\d+)/(\d+)$", s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    # "1/2", "2/3"
    m = re.match(r"^(\d+)/(\d+)$", s)
    if m:
        return int(m.group(1)) / int(m.group(2))
    try:
        return float(s)
    except ValueError:
        return None


def round_quantity(x: float) -> str:
    """Round for readability: whole when close; else 1 decimal; amounts <1 use 1 decimal for ML consistency (0.3 cup)."""
    if x == int(x) or abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    if x < 1:
        r = round(x, 1)
        return str(r).rstrip("0").rstrip(".") or "0"
    r = round(x, 1)
    if r == int(r):
        return str(int(r))
    return str(r)


def _promote_mass_volume(final_unit: str, new_amount: float) -> tuple[str, float]:
    """g→kg and ml→l when amount ≥ 1000 for cleaner dataset output."""
    if final_unit == "g" and new_amount >= 1000:
        return "kg", new_amount / 1000
    if final_unit == "ml" and new_amount >= 1000:
        return "l", new_amount / 1000
    return final_unit, new_amount


def _format_mass_output_g(total_g: float, rest: str) -> str:
    """Emit mass with g→kg promotion for large amounts (early-return paths)."""
    u, amt = _promote_mass_volume("g", float(total_g))
    s = round_quantity(amt)
    rest = _apply_ingredient_aliases(rest.strip()) if rest else rest
    return f"{s} {u} {rest}" if rest else f"{s} {u} ingredient"


def _passes_final_validation(ing: str) -> bool:
    """True if line passes the same checks as the sanitize pass (safe to write)."""
    if not isinstance(ing, str) or not ing.strip():
        return False
    parts = ing.split()
    if len(parts) == 1 and (parts[0].lower() in FINAL_UNITS or parts[0].lower() == "none"):
        return False
    if len(parts) >= 2 and re.match(r"^\d+(\.\d+)?$", parts[0]) and parts[1] in FINAL_UNITS and len(parts) < 3:
        return False
    if len(parts) >= 2 and re.match(r"^\d+(\.\d+)?$", parts[0]) and parts[1] not in FINAL_UNITS:
        if "inch" in parts[1].lower():
            return False
        return False
    return True


def _canonicalize_unit_token(ing: str) -> str:
    """If line is 'qty unit rest' and unit is an alias (e.g. tablespoons), replace with final unit (tablespoon)."""
    parts = ing.split(maxsplit=2)
    if len(parts) < 2:
        return ing
    qty, unit, rest = parts[0], parts[1], (parts[2] if len(parts) > 2 else "")
    if not re.match(r"^\d+(\.\d+)?$", qty):
        return ing
    u = unit.lower()
    if u in FINAL_UNITS:
        return ing
    if u in ALIAS_TO_FINAL:
        new_unit = ALIAS_TO_FINAL[u]
        return f"{qty} {new_unit} {rest}".strip() if rest else f"{qty} {new_unit}"
    return ing


def _clean_rest(rest: str, final_unit: str | None = None) -> str:
    """Strip size/shape/descriptor adjectives; singularize last word only for count units (piece, clove, …)."""
    words = rest.split()
    while words and words[0].lower() in SIZE_WORDS:
        words.pop(0)
    while words and words[0].lower() in SHAPE_WORDS:
        words.pop(0)
    # Strip descriptor adjectives (finely, chopped, etc.) from any position
    words = [w for w in words if w.lower() not in DESCRIPTOR_WORDS]
    if not words:
        return ""
    # 5. Strip " or ..." (e.g. "penne or other short pasta" → "penne")
    rest_str = " ".join(words)
    rest_str = re.sub(r"\bor\b.*$", "", rest_str, flags=re.IGNORECASE).strip()
    words = rest_str.split()
    if not words:
        return ""
    # Volume/mass/etc.: keep natural plural (3 cup cherry tomatoes, 0.5 cup breadcrumbs)
    if final_unit is None or final_unit not in COUNT_UNITS_SINGULARIZE:
        return " ".join(words)
    last = words[-1].lower()
    # 3. Irregular: leaves → leaf (avoid "leave")
    if last == "leaves":
        words[-1] = "leaf"
    elif last.endswith("ies") and len(last) > 3:
        words[-1] = words[-1][:-3] + "y"
    elif last.endswith("oes") and len(last) > 3:
        words[-1] = words[-1][:-3] + "o"  # tomatoes → tomato, potatoes → potato
    elif last.endswith("s") and not last.endswith(("ss", "us", "is")) and len(last) > 2:
        words[-1] = words[-1][:-1]
    return " ".join(words)


def _apply_ingredient_aliases(rest: str) -> str:
    """Normalize ingredient phrases for consistent vocabulary (after _clean_rest)."""
    if not rest:
        return rest
    out = rest
    for old, new in sorted(INGREDIENT_ALIASES.items(), key=lambda x: -len(x[0])):
        out = re.sub(r"\b" + re.escape(old) + r"\b", new, out, flags=re.IGNORECASE)
    return out


def _strip_leading_container_words(rest: str) -> str:
    """Drop leading can/log/package/tin so 'can crushed tomatoes' → 'crushed tomatoes'."""
    words = rest.split()
    while words and words[0].lower() in CONTAINER_WORDS:
        words.pop(0)
    return " ".join(words).strip() if words else rest


def _alias_and_convert_unit(unit_raw: str, orig: str = "") -> tuple[str, float] | None:
    """
    Pipeline: alias (with case-sensitive t/T) → plural normalization → conversion.
    Returns (final_unit, factor) or None. Logs unknown units when orig provided.
    """
    raw = unit_raw.strip()
    # Case-sensitive: recipe convention T = tablespoon, t = teaspoon
    if raw == "T":
        canonical = "tablespoon"
    elif raw == "t":
        canonical = "teaspoon"
    else:
        u = raw.lower()
        canonical = UNIT_ALIASES.get(u)
        if canonical is None:
            # Plural: strip trailing 's' if result is known
            if u.endswith("s") and u[:-1] in UNIT_ALIASES:
                canonical = UNIT_ALIASES.get(u[:-1])
            if canonical is None and u.endswith("s") and u[:-1] in CONVERSION_MAP:
                canonical = u[:-1]
    if canonical is None:
        if orig:
            log.warning("Unknown unit: %r in %s", unit_raw, orig)
        return None
    conv = CONVERSION_MAP.get(canonical)
    if conv is None:
        return None
    return conv


def normalize_ingredient_deterministic(ing: str) -> str | None:
    """
    Pipeline: extract → alias → plural → conversion → round → format → validate.
    Returns normalized "[quantity] [unit] [ingredient]" or None if unparseable.
    """
    if not ing or not isinstance(ing, str):
        return None
    orig = ing.strip()
    if not orig:
        return None
    # Drop literal "None" from dataset
    if orig.lower() == "none":
        return None
    # 3. Single-token unit (e.g. "cup") — reject before parsing or LLM
    if orig.lower() in FINAL_UNITS:
        return None

    # Normalize before parsing: Unicode fractions, ranges
    cleaned = normalize_unicode_fractions(orig)
    cleaned = normalize_ranges(cleaned)

    # Weight range: "1 5-to 5 1/2-pound brisket" → "5.25 pound brisket" → g → kg promotion
    def _pound_range_repl(m):
        a = parse_fraction(m.group(1).strip())
        b = parse_fraction(m.group(2).strip())
        if a is None or b is None:
            return m.group(0)
        mid = round((a + b) / 2, 2)
        return f" {mid} pound "
    cleaned = re.sub(
        r"(?:^|\s)(?:\d+\s+)?(\d+(?:\s+\d+/\d+)?|\d+/\d+)\s*[-–]\s*to\s*(\d+(?:\s+\d+/\d+)?|\d+/\d+)\s*[-–]?(pound|lb)s?\b\s*",
        _pound_range_repl,
        cleaned,
        flags=re.IGNORECASE,
    )

    # 1. Fractions to decimals globally so "1/2 onion" and "1/2 cup" both parse (mixed first, then simple)
    cleaned = re.sub(r"(\d+)\s+(\d+)/(\d+)", lambda m: str(round(int(m.group(1)) + int(m.group(2)) / int(m.group(3)), 2)), cleaned)
    cleaned = re.sub(r"(\d+)/(\d+)", lambda m: str(round(int(m.group(1)) / int(m.group(2)), 2)), cleaned)

    # 1b. Missing space in units: 120g → 120 g, 500ml → 500 ml, 1lb → 1 lb, 8oz → 8 oz
    cleaned = re.sub(r"(\d)(g|kg|ml|l)\b", r"\1 \2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d)(oz|lb)\b", r"\1 \2", cleaned, flags=re.IGNORECASE)

    # 3. Ingredient-info / accompaniments / makes lines — not real ingredients
    if cleaned.lower().startswith(("ingredient info", "accompaniments", "makes")):
        return None

    # 7. Cooking equipment — not ingredients
    if any(w in cleaned.lower() for w in EQUIPMENT_WORDS):
        return None

    # 5. Container-only: "package cheese ravioli" → "1 package cheese ravioli"
    if not re.match(r"^\d", cleaned) and cleaned.lower().startswith(("package", "bag", "box")):
        cleaned = "1 " + cleaned

    # 3 & 4. Compound units: "3 tablespoon plus 2 teaspoons vinegar" → "3 tablespoon vinegar"; "2 tablespoon and 1 teaspoon of oil" → "2 tablespoon oil"
    cleaned = re.sub(r"\s+plus\s+(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?|\d+)\s*" + _COMPOUND_UNIT_PATTERN + r"\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+and\s+(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?|\d+)\s*" + _COMPOUND_UNIT_PATTERN + r"\s+of?\s*", " ", cleaned, flags=re.IGNORECASE)

    # 1. "Two 6-ounce cans tuna", "three 6-ounce cans tomatoes" → count × oz × 30 g
    m = N_OUNCE_CANS_RE.match(cleaned)
    if m:
        count_str, oz_str, rest = m.group(1).strip(), m.group(2), m.group(3).strip()
        count = NUMBER_WORDS.get(count_str.lower())
        if count is None:
            count = int(count_str)
        total_g = count * int(oz_str) * 30
        rest = _strip_leading_container_words(rest)
        return _format_mass_output_g(total_g, rest)
    # 2. "25-ounce package frozen ravioli" → "750 g frozen ravioli" (inline weight, no parens)
    m = INLINE_OUNCE_RE.match(cleaned)
    if m:
        oz_val, rest = int(m.group(1)), m.group(2).strip()
        rest = _strip_leading_container_words(rest)
        return _format_mass_output_g(oz_val * 30, rest)
    # 3 & 8. Container + weight: use weight in g instead of discarding it
    # "1 (5-ounce) log cheese" → "150 g log cheese"; "1 piece 15-ounce can tomatoes" → "450 g tomatoes"
    m = PAREN_WEIGHT_FULL_RE.match(cleaned)
    if m:
        n, m_val, u, rest = int(m.group(1)), float(m.group(2)), m.group(3).lower(), m.group(4).strip()
        total_g = n * m_val * (30 if u in ("ounce", "oz") else 500)
        rest = _strip_leading_container_words(rest)
        return _format_mass_output_g(total_g, rest)
    m = LEADING_WEIGHT_RE.match(cleaned)
    if m:
        n, m_val, u, rest = int(m.group(1)), float(m.group(2)), m.group(3).lower(), m.group(4).strip()
        total_g = n * m_val * (30 if u in ("ounce", "oz") else 500)
        rest = _strip_leading_container_words(rest)
        return _format_mass_output_g(total_g, rest)

    # Modifiers, unit periods, parentheticals
    cleaned = re.sub(r"\b(about|approx|approximately|around)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(heaped|level|scant|rounded)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b([a-zA-Z]+)\.", r"\1", cleaned)
    cleaned = PAREN_SIZE_RE.sub(" ", cleaned)
    cleaned = PAREN_WEIGHT_RE.sub(" ", cleaned)
    cleaned = PAREN_VOLUME_RE.sub(" ", cleaned)  # "(1 1/2 pints)" etc.
    # 8. Remove any remaining parentheticals e.g. "(1 stick)", "(optional)", "(see Note, page 93)"
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    # 2. Strip trailing clause after first comma (", divided", ", peeled") — MUST run after removing
    # parentheticals so commas *inside* "(see Note, page 93)" are not mistaken for the start of a tail.
    cleaned = re.sub(r",\s*.*$", "", cleaned).strip()
    # Strip size descriptors: "2-inch-diameter", "1.5-inch", "1/2-inch" — keep quantity, drop dimension
    cleaned = INCH_SIZE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*[-–]\s*inch(?:es)?\b", r"\1", cleaned, flags=re.IGNORECASE)
    # "piece of red beet" (after strip) → "1 piece red beet" so pipeline parses it
    cleaned = PIECE_OF_RE.sub("1 piece ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")

    # 4. Short-circuit: "salt to taste" etc. — unit consistency doesn't need LLM
    if cleaned.lower().endswith("to taste"):
        return cleaned

    # 4b. Multi-number safeguard: merged OCR / garbage (e.g. "5 g 5 1/2 pound brisket" has 3+ quantity tokens)
    # Use token count, not raw digit count — so "5.25 pound brisket" (one token 5.25) is allowed → 2.6 kg brisket.
    quantity_tokens = re.findall(r"\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?|\d+", cleaned)
    if len(quantity_tokens) > 2:
        return None
    # 2. Reject corrupted lines with multiple unit words (e.g. "5 g 5 1/2 kg 5 1/2 pound brisket")
    if len(re.findall(r"\b(?:g|kg|ml|l|cup|teaspoon|tablespoon|pound|pounds?|ounce|ounces?)\b", cleaned, re.IGNORECASE)) > 1:
        return None

    amount: float | None = None
    unit_raw: str | None = None
    rest: str = ""

    m = FRACTION_RE.match(cleaned)
    if m:
        amount = parse_fraction(m.group("num").strip())
        unit_raw = m.group("unit").strip()
        rest = m.group("rest").strip()
    if amount is None or unit_raw is None:
        m = TAIL_UNIT_RE.match(cleaned)
        if m:
            amount = parse_fraction(m.group("num").strip())
            unit_raw = m.group("unit").strip()
            rest = m.group("rest").strip()

    # 8. Missing unit: "2 onions" → "2 piece onion"
    if amount is None or unit_raw is None:
        num_only = NUM_ONLY_RE.match(cleaned)
        if num_only:
            amount = parse_fraction(num_only.group("num").strip())
            rest = num_only.group("rest").strip()
            if amount is not None and rest:
                unit_raw = "piece"
                conv = CONVERSION_MAP.get("piece")
                if conv:
                    final_unit, factor = conv
                    amt_str = round_quantity(amount * factor)
                    rest = re.sub(r"^[\s,\-–]+", "", rest).strip()
                    if not rest:  # 3. Broken: "2 piece" with no ingredient name
                        return None
                    # 4. Garbage rest: "5 g piece slice clove ..." — too many tokens
                    if len(rest.split()) > MAX_REST_WORDS:
                        return None
                    rest = _clean_rest(rest, final_unit)  # singularize only for count units
                    if not rest:
                        return None
                    if rest.lower() in REST_ONLY_ADJECTIVE:
                        return None
                    if rest.endswith("-fat") or (rest.split() and rest.split()[-1].lower() in FAT_ONLY_ENDINGS):
                        return None
                    rest = _apply_ingredient_aliases(rest)
                    out = f"{amt_str} {final_unit} {rest}"
                    if final_unit not in FINAL_UNITS:
                        raise ValueError(f"Invalid unit detected: {final_unit} in {out!r}")
                    return out
        return None

    if amount is None:
        return None

    conv = _alias_and_convert_unit(unit_raw, orig)
    if not conv:
        return None
    final_unit, factor = conv
    new_amount = amount * factor
    final_unit, new_amount = _promote_mass_volume(final_unit, new_amount)

    # 7. Round quantities for readability
    amt_str = round_quantity(new_amount)
    rest = re.sub(r"^[\s,\-–]+", "", rest).strip()
    # 3. Broken ingredients: "1 cup" or "1 tablespoon" with no name → don't emit placeholder
    if not rest or rest.lower() in UNIT_ALIASES or rest.lower() in FINAL_UNITS:
        return None

    # 4. Garbage rest: too many tokens
    if len(rest.split()) > MAX_REST_WORDS:
        return None

    # Container unit: drop redundant leading container word from rest ("1 packet bag cheese" → "1 packet cheese")
    if final_unit in {"packet", "bag", "box", "jar", "bottle", "tin", "sachet", "carton"}:
        rest = _strip_leading_container_words(rest)

    rest = _clean_rest(rest, final_unit)  # singularize only for count units
    if not rest:
        return None
    # 6. Never allow ingredient name = single adjective (e.g. "1 cup green" from green split peas)
    if rest.lower() in REST_ONLY_ADJECTIVE:
        return None
    # 1. Missing noun: "2 cup low-fat" (rest ends in -fat only)
    if rest.endswith("-fat") or (rest.split() and rest.split()[-1].lower() in FAT_ONLY_ENDINGS):
        return None

    rest = _apply_ingredient_aliases(rest)

    # 6. Canonical format: [quantity] [unit] [ingredient]
    out = f"{amt_str} {final_unit} {rest}"

    # 10. Unit validator: prevent silent dataset corruption
    if final_unit not in FINAL_UNITS:
        raise ValueError(f"Invalid unit detected: {final_unit} in {out!r}")

    return out


def normalize_with_ollama(
    ing: str, client, model: str = DEFAULT_OLLAMA_MODEL, strict_followup: bool = False
) -> str | None:
    """
    Use LLM when quantity/unit missing or ingredient unclear.
    Returns None if output fails guardrails (caller may retry). Returns cleaned line or ingredient-only text.
    """
    extra = ""
    if strict_followup:
        extra = (
            "\n\nCRITICAL: If you output a number, the SECOND word must be exactly one of: "
            "mg g kg ml l teaspoon tablespoon cup piece slice clove bunch stalk sprig leaf cube dozen "
            "pinch drop packet bag box jar bottle carton tin sachet handful scoop serving. "
            "For a weight range like 5-to-6 pounds, pick ONE amount and unit (e.g. 2.75 kg brisket)."
        )
    prompt = f"""Normalize this recipe ingredient into exactly one line: [quantity] [unit] [ingredient name].

You MUST use only these units (no variants): mg g kg ml l teaspoon tablespoon cup piece slice clove bunch stalk sprig leaf cube dozen pinch drop packet bag box jar bottle carton tin sachet handful scoop serving.

If there is no clear quantity or unit, output a short cleaned-up ingredient name only (no number/unit).
Output nothing else—no explanation, no newlines. One line only.
{extra}
Ingredient: {ing}

Normalized:"""
    try:
        out = client.generate_response(model, prompt, stream=False)
        if not out:
            return None
        out = out.strip().split("\n")[0].strip()
        tokens = out.split()
        # Structured line: quantity + unit — validate second token only when first token is numeric
        if len(tokens) >= 2 and re.match(r"^\d+(\.\d+)?$", tokens[0]):
            if tokens[1] not in FINAL_UNITS:
                return None
            if tokens[1] in FINAL_UNITS and len(tokens) < 3:
                return None
        # Two-unit outputs
        if len(tokens) >= 2 and tokens[0] in FINAL_UNITS and tokens[1] in FINAL_UNITS:
            return None
        if len(tokens) >= 3 and tokens[1] in FINAL_UNITS and tokens[2] in FINAL_UNITS:
            return None
        # If original had no number, LLM must not invent one
        if re.match(r"^\d", ing.strip()) is None and re.match(r"^\d", out):
            return None
        return out
    except Exception:
        return None


def _process_single_recipe(
    recipe: dict,
    client,
    use_ollama: bool,
    model: str,
) -> tuple[dict, dict]:
    """
    Run full per-recipe pipeline: deterministic → Ollama → canonicalize → repair → sanitize.
    Returns (mutated recipe, stats dict with keys normalized, ollama, unparseable, repair).
    """
    stats = {"normalized": 0, "ollama": 0, "unparseable": 0, "repair": 0}
    ings = recipe.get("ingredients") or []
    new_ings: list = []
    for ing in ings:
        if not ing or not isinstance(ing, str):
            if ing:
                new_ings.append(ing)
            continue
        if not ing.strip() or ing.strip().lower() == "none":
            continue

        norm = normalize_ingredient_deterministic(ing)
        if norm is not None and norm.strip().lower() != "none":
            norm = _canonicalize_unit_token(norm.strip())
            if _passes_final_validation(norm):
                new_ings.append(norm)
                stats["normalized"] += 1
                continue

        if not use_ollama or not client:
            if ing.strip().lower() != "none":
                new_ings.append(ing.strip())
                stats["unparseable"] += 1
            continue

        norm = normalize_with_ollama(ing, client, model)
        if norm:
            norm = _canonicalize_unit_token(norm.strip())
        if norm and any(w in norm.lower() for w in EQUIPMENT_WORDS):
            if ing.strip().lower() != "none":
                new_ings.append(ing.strip())
            stats["unparseable"] += 1
            continue
        if norm and norm.strip().lower() == "none":
            stats["unparseable"] += 1
            continue
        if not norm or not _passes_final_validation(norm):
            norm = normalize_with_ollama(ing, client, model, strict_followup=True)
            if norm:
                norm = _canonicalize_unit_token(norm.strip())
        if norm and _passes_final_validation(norm) and not any(w in norm.lower() for w in EQUIPMENT_WORDS):
            new_ings.append(norm)
            stats["ollama"] += 1
        else:
            if ing.strip().lower() != "none":
                new_ings.append(ing.strip())
            stats["unparseable"] += 1

    recipe["ingredients"] = new_ings

    recipe["ingredients"] = [
        _canonicalize_unit_token(ing) if isinstance(ing, str) else ing
        for ing in recipe.get("ingredients") or []
    ]

    if use_ollama and client:
        rep: list = []
        for ing in recipe.get("ingredients") or []:
            if not isinstance(ing, str):
                rep.append(ing)
                continue
            ing = _canonicalize_unit_token(ing.strip())
            if _passes_final_validation(ing):
                rep.append(ing)
                continue
            out = normalize_with_ollama(ing, client, model, strict_followup=True)
            if not out:
                out = normalize_with_ollama(ing, client, model)
            if out:
                out = _canonicalize_unit_token(out.strip())
            if out and _passes_final_validation(out) and not any(w in out.lower() for w in EQUIPMENT_WORDS):
                rep.append(out)
                stats["repair"] += 1
            else:
                rep.append(ing)
        recipe["ingredients"] = rep

    kept: list = []
    for ing in recipe.get("ingredients") or []:
        if not isinstance(ing, str):
            kept.append(ing)
            continue
        parts = ing.split()
        if len(parts) == 1 and (parts[0].lower() in FINAL_UNITS or parts[0].lower() == "none"):
            continue
        if len(parts) >= 2 and re.match(r"^\d+(\.\d+)?$", parts[0]) and parts[1] in FINAL_UNITS and len(parts) < 3:
            continue
        if len(parts) >= 2 and re.match(r"^\d+(\.\d+)?$", parts[0]) and parts[1] not in FINAL_UNITS:
            if "inch" in parts[1].lower():
                continue
            log.debug("Non-FINAL_UNITS second token kept after Ollama repair: %r — %s", parts[1], ing[:80])
        kept.append(ing)
    recipe["ingredients"] = kept

    return recipe, stats


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, data) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _cleanup_stale_tmp(output_path: Path) -> None:
    """Remove leftover .tmp from a crashed/interrupted write."""
    tmp = Path(output_path).with_suffix(Path(output_path).suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
            print(f"Removed stale temp file: {tmp}", file=sys.stderr)
        except OSError as e:
            print(f"Could not remove {tmp}: {e}", file=sys.stderr)


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # nan
        return "??:??"
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _progress_line(
    *,
    done: int,
    chunk_len: int,
    global_idx: int,
    elapsed: float,
    eta_sec: float | None,
    normalized: int,
    ollama: int,
    repair: int,
    width: int = 118,
) -> str:
    pct = 100.0 * done / chunk_len if chunk_len else 100.0
    eta_s = _fmt_duration(eta_sec) if eta_sec is not None else "…"
    line = (
        f"[{done}/{chunk_len}] {pct:5.1f}% | recipe #{global_idx} | "
        f"elapsed {_fmt_duration(elapsed)} | ETA ~{eta_s} | "
        f"det={normalized} llm={ollama} repair={repair}"
    )
    return line[: width].ljust(width)


def main():
    import argparse
    from main import OllamaClient

    parser = argparse.ArgumentParser(description="Normalize recipe ingredients to target units.")
    parser.add_argument("--input", default="recipes_images.json", help="Input recipes JSON")
    parser.add_argument("--output", default="recipes_normalized.json", help="Output JSON")
    parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model for ambiguous ingredients (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument("--sample", type=int, default=0, help="Process only N recipes (0 = all)")
    parser.add_argument("--no-ollama", action="store_true", help="Skip LLM; leave unparseable as-is")
    parser.add_argument("--audit", action="store_true", help="After writing, print unit audit (second token of each ingredient)")
    parser.add_argument("--batch-start", type=int, default=0, help="First recipe index in the (sampled) list (inclusive)")
    parser.add_argument(
        "--batch-end",
        type=int,
        default=None,
        help="End recipe index exclusive; default = end of list",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (use same --input, --output, --batch-start, --batch-end as previous run)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        metavar="N",
        help="Save output + checkpoint every N recipes (default: 25)",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Checkpoint path (default: <output>.checkpoint.json)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live progress line (single-line updates use carriage return on TTYs)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        metavar="N",
        help="Update progress every N completed recipes (default: 1)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else output_path.with_suffix(output_path.suffix + ".checkpoint.json")

    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    _cleanup_stale_tmp(output_path)

    print("Loading recipes...", flush=True)
    t_load0 = time.perf_counter()
    with open(input_path) as f:
        all_recipes = json.load(f)
    print(f"Loaded {len(all_recipes)} recipe(s) in {_fmt_duration(time.perf_counter() - t_load0)}.", flush=True)
    if args.sample:
        all_recipes = all_recipes[: args.sample]

    batch_end = args.batch_end if args.batch_end is not None else len(all_recipes)
    batch_start = max(0, args.batch_start)
    batch_end = min(batch_end, len(all_recipes))
    if batch_start >= batch_end:
        print("Nothing to process (batch-start >= batch-end).", file=sys.stderr)
        sys.exit(1)

    chunk_len = batch_end - batch_start
    input_sha = _file_sha256(input_path)

    # Working slice [batch_start:batch_end] — references same objects as all_recipes for simplicity
    work_slice = all_recipes[batch_start:batch_end]

    use_ollama = not args.no_ollama
    client = None
    if use_ollama:
        client = OllamaClient()
        if not client.check_connection():
            print("Ollama not available; run with --no-ollama to skip LLM.", file=sys.stderr)
            use_ollama = False
        else:
            print(f"Using Ollama model: {args.model}")

    next_index = 0
    if args.resume:
        if not checkpoint_path.exists():
            print("Warning: --resume but no checkpoint file; starting from scratch.", file=sys.stderr)
        else:
            with open(checkpoint_path) as f:
                ck = json.load(f)
            if ck.get("input_sha256") != input_sha:
                print("Checkpoint input_sha256 does not match current --input file; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            if ck.get("batch_start") != batch_start or ck.get("batch_end") != batch_end:
                print(
                    "Checkpoint batch range does not match; use same --batch-start/--batch-end as the saved run.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if ck.get("output_path") != str(output_path):
                print("Checkpoint output_path does not match --output; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            if ck.get("sample", args.sample) != args.sample:
                print("Checkpoint --sample does not match; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            next_index = int(ck.get("next_recipe_index", 0))
            next_index = max(0, min(next_index, chunk_len))
            if next_index >= chunk_len and ck.get("complete"):
                print("This batch is already complete (checkpoint). Nothing to do.")
                sys.exit(0)

    normalized_count = 0
    ollama_count = 0
    unparseable_count = 0
    repair_ollama_count = 0

    if args.resume and next_index > 0:
        if not output_path.exists():
            print("Cannot resume: next_recipe_index > 0 but output file is missing.", file=sys.stderr)
            sys.exit(1)
        with open(output_path) as f:
            saved = json.load(f)
        if len(saved) != chunk_len:
            print(
                f"Output has {len(saved)} recipes but chunk length is {chunk_len}; cannot resume safely.",
                file=sys.stderr,
            )
            sys.exit(1)
        for j in range(next_index):
            work_slice[j] = saved[j]
        for j in range(next_index, chunk_len):
            work_slice[j] = copy.deepcopy(all_recipes[batch_start + j])
        print(f"Resuming at global recipe index {batch_start + next_index} (chunk {next_index}/{chunk_len}).")
    else:
        work_slice[:] = [copy.deepcopy(r) for r in work_slice]

    print(f"Batch [{batch_start}, {batch_end}) — {chunk_len} recipe(s) in this run.", flush=True)

    show_progress = (not args.no_progress) and sys.stdout.isatty()
    progress_every = max(1, args.progress_every)
    progress_t0 = time.perf_counter()
    last_ck_line = False

    def _checkpoint_payload(done_idx: int, *, complete: bool) -> dict:
        d = {
            "next_recipe_index": done_idx,
            "chunk_len": chunk_len,
            "batch_start": batch_start,
            "batch_end": batch_end,
            "sample": args.sample,
            "input_path": str(input_path),
            "input_sha256": input_sha,
            "output_path": str(output_path),
        }
        if complete:
            d["complete"] = True
        return d

    try:
        for i in range(next_index, chunk_len):
            recipe = work_slice[i]
            try:
                recipe, st = _process_single_recipe(recipe, client, use_ollama, args.model)
            except KeyboardInterrupt:
                work_slice[i] = copy.deepcopy(all_recipes[batch_start + i])
                _atomic_write_json(output_path, work_slice)
                with open(checkpoint_path, "w") as f:
                    json.dump(_checkpoint_payload(i, complete=False), f, indent=2)
                if show_progress:
                    print(file=sys.stderr)  # newline after \r line
                print(
                    f"\nInterrupted at chunk index {i} (global #{batch_start + i}). "
                    f"Recipe reset to input; saved partial output + checkpoint.\n"
                    f"Resume with the same flags plus --resume",
                    file=sys.stderr,
                )
                sys.exit(130)

            work_slice[i] = recipe
            normalized_count += st["normalized"]
            ollama_count += st["ollama"]
            unparseable_count += st["unparseable"]
            repair_ollama_count += st["repair"]

            done = i + 1
            elapsed = time.perf_counter() - progress_t0
            eta_sec = (chunk_len - done) * (elapsed / done) if done else None
            if show_progress and (done % progress_every == 0 or done == chunk_len):
                line = _progress_line(
                    done=done,
                    chunk_len=chunk_len,
                    global_idx=batch_start + i,
                    elapsed=elapsed,
                    eta_sec=eta_sec,
                    normalized=normalized_count,
                    ollama=ollama_count,
                    repair=repair_ollama_count,
                )
                print(f"\r{line}", end="", flush=True)
                last_ck_line = True
            elif not show_progress and (done % 50 == 0 or done == chunk_len or done == 1):
                line = _progress_line(
                    done=done,
                    chunk_len=chunk_len,
                    global_idx=batch_start + i,
                    elapsed=elapsed,
                    eta_sec=eta_sec,
                    normalized=normalized_count,
                    ollama=ollama_count,
                    repair=repair_ollama_count,
                )
                print(line.strip(), flush=True)

            if done % args.checkpoint_every == 0 or done == chunk_len:
                _atomic_write_json(output_path, work_slice)
                with open(checkpoint_path, "w") as f:
                    json.dump(_checkpoint_payload(done, complete=False), f, indent=2)
                # Newline before checkpoint so we don't overwrite the \r progress line on TTYs
                prefix = "\n" if show_progress else ""
                print(f"{prefix}  checkpoint: {done}/{chunk_len} recipes saved → {output_path}", flush=True)
                if show_progress:
                    last_ck_line = False
    finally:
        if show_progress and last_ck_line:
            print(flush=True)

    # Final write + mark complete
    _atomic_write_json(output_path, work_slice)
    with open(checkpoint_path, "w") as f:
        json.dump(
            {
                "next_recipe_index": chunk_len,
                "complete": True,
                "chunk_len": chunk_len,
                "batch_start": batch_start,
                "batch_end": batch_end,
                "sample": args.sample,
                "input_path": str(input_path),
                "input_sha256": input_sha,
                "output_path": str(output_path),
            },
            f,
            indent=2,
        )

    recipes = work_slice

    if args.audit:
        unit_counts = Counter()
        for r in recipes:
            for ing in r.get("ingredients") or []:
                if isinstance(ing, str):
                    parts = ing.split()
                    # Only count lines in canonical form [quantity] [unit] [ingredient]
                    if (
                        len(parts) >= 3
                        and re.match(r"^\d+(\.\d+)?$", parts[0])
                        and parts[1] in FINAL_UNITS
                    ):
                        unit_counts[parts[1]] += 1
        print("Unit frequency audit (second token of [quantity] [unit] [ingredient] lines only):")
        for unit, count in unit_counts.most_common():
            print(f"  {unit}: {count}")
        units = set(unit_counts.keys())
        extra = units - FINAL_UNITS
        if extra:
            print("  ⚠ Units not in FINAL_UNITS:", sorted(extra), file=sys.stderr)
        else:
            print("  ✓ All units in FINAL_UNITS")

    print("Done.")
    print(f"  Deterministic normalizations: {normalized_count}")
    print(f"  Ollama normalizations (per-ingredient): {ollama_count}")
    print(f"  Ollama post-canonicalize repairs: {repair_ollama_count}")
    print(f"  Left as-is (unparseable, no Ollama): {unparseable_count}")


if __name__ == "__main__":
    main()
