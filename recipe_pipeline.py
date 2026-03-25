#!/usr/bin/env python3
"""
Unified recipe pipeline (single file).

Subcommands:
  normalize   — Ingredient units only (batch, resume, checkpoints).
  enrich|run  — **Single JSON output:** for each recipe, normalize then enrich → ``-o`` / ``out.json`` (default). Use this for “one file, one command.”
  pipeline    — Same order as enrich, but writes **two** files per step: ``normalized.json`` + ``out.json`` (only if you want stage 1 on disk separately).

This file is the only pipeline module in the project.

Per-recipe enrich order (``enrich`` / ``run``):

  1. Normalize ingredient lines (``_process_single_recipe``).
  2. Normalize instruction units (unicode fractions, ranges, Tbsp/cup/mass).
  3. Align vague volumes in steps with ingredient decimals (e.g. half cup → 0.5 cup).
  4. Optional micro-clean: Ollama one-sentence fix for heuristically broken steps.
  5. Junk filter + full Ollama rewrite or deterministic ``standardize_instruction_step``.
  6. ``ing_embedding_text`` from ingredients only (``process_recipe``).

Progress: recipe-level loaders print to stderr (\r bar) unless --quiet / --no-progress.
ETA uses throughput since process start (correct after ``--resume``). Ollama GPU: set ``OLLAMA_NUM_GPU`` (default in code: 999 layers on GPU) or ``OLLAMA_NUM_GPU=auto`` to let the server choose.

Enrich writes ``out.json`` (default) via atomic replace; by default after **each** recipe, or
every ``--checkpoint-every N`` recipes for less disk I/O. Companion ``out.json.checkpoint.json``
tracks resume position. Use ``--resume`` with the same ``--input``, ``--limit`` (``0`` = whole file),
``--keep-rest``, ``--checkpoint-every``, and ``--output`` as the interrupted run.
Fresh run when ``out.json`` exists requires ``--force`` to overwrite.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional

import requests

# Default read timeout for /api/generate (seconds). CPU inference can exceed 30s.
# Override with env: OLLAMA_GENERATE_TIMEOUT=1200
_DEFAULT_READ_TIMEOUT = float(os.environ.get("OLLAMA_GENERATE_TIMEOUT", "600"))

# Ollama runs inference in its server process; GPU use is controlled there. Each request can
# pass ``options`` (e.g. num_gpu layers on VRAM). Defaults favor GPU offload on systems with
# a CUDA GPU (GTX/RTX). Set OLLAMA_NUM_GPU=0 for CPU-only, or OLLAMA_NUM_GPU=auto to omit
# num_gpu and let Ollama pick. Optional JSON merge: OLLAMA_EXTRA_OPTIONS='{"num_thread":8}'
def _ollama_inference_options() -> dict:
    opts: dict = {}
    raw = os.environ.get("OLLAMA_NUM_GPU", "999").strip()
    if raw and raw.lower() not in ("auto", "default"):
        try:
            opts["num_gpu"] = int(raw)
        except ValueError:
            pass
    extra = os.environ.get("OLLAMA_EXTRA_OPTIONS", "").strip()
    if extra:
        try:
            opts.update(json.loads(extra))
        except json.JSONDecodeError:
            pass
    return opts


def _find_nvidia_smi() -> str | None:
    """Resolve nvidia-smi: PATH, then typical Windows NVSMI install path."""
    p = shutil.which("nvidia-smi")
    if p:
        return p
    if sys.platform == "win32":
        for base in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramW6432", r"C:\Program Files"),
        ):
            candidate = Path(base) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
            if candidate.is_file():
                return str(candidate)
    return None


def _print_gpu_and_ollama_status() -> None:
    """
    Print whether the NVIDIA driver CLI is available and what ``ollama ps`` reports
    (PROCESSOR column: 100%% GPU vs 100%% CPU). Runs when Ollama is connected.
    """
    if os.environ.get("RECIPE_PIPELINE_NO_GPU_INFO", "").strip() in ("1", "true", "yes"):
        return

    def _run(argv: list[str], timeout: float = 12.0) -> subprocess.CompletedProcess[str]:
        kw: dict = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
        }
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        return subprocess.run(argv, **kw)

    print("[GPU check]", flush=True)
    nv = _find_nvidia_smi()
    if nv:
        try:
            r = _run([nv, "-L"])
            if r.returncode == 0 and (r.stdout or "").strip():
                print(f"  nvidia-smi: OK ({nv})", flush=True)
                for line in (r.stdout or "").strip().splitlines()[:6]:
                    print(f"    {line}", flush=True)
            else:
                print(
                    f"  nvidia-smi exists but failed (exit {r.stderr or r.stdout or r.returncode}).",
                    file=sys.stderr,
                    flush=True,
                )
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"  nvidia-smi error: {e}", file=sys.stderr, flush=True)
    else:
        print(
            "  nvidia-smi: not found (not on PATH; not under …\\NVIDIA Corporation\\NVSMI). "
            "Install/update NVIDIA drivers if you expect a GeForce GPU.",
            file=sys.stderr,
            flush=True,
        )

    ox = shutil.which("ollama")
    if not ox:
        print("  ollama CLI: not on PATH (cannot run `ollama ps`).", file=sys.stderr, flush=True)
        return
    try:
        r = _run([ox, "ps"])
        out = (r.stdout or "").rstrip()
        if not out:
            print("  ollama ps: (no output)", flush=True)
            return
        print("  ollama ps — PROCESSOR shows whether the loaded model uses GPU:", flush=True)
        for line in out.splitlines():
            print(f"    {line}", flush=True)
        body = "\n".join(out.splitlines()[1:])  # skip header
        if "% GPU" in body or " GPU " in body:
            print("  → Ollama reports GPU use for a loaded model.", flush=True)
        elif "100% CPU" in body:
            print(
                "  → Ollama reports 100% CPU (no GPU for loaded models). "
                "Fix drivers / GPU visibility; pipeline will still run but slowly.",
                file=sys.stderr,
                flush=True,
            )
        elif "NAME" in out and not body.strip():
            print(
                "  → No model loaded in Ollama yet; PROCESSOR appears after the first inference.",
                flush=True,
            )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  ollama ps error: {e}", file=sys.stderr, flush=True)


# Preferred model for normalize_ingredients.py (quantized Qwen 7B — good for ~16GB RAM / CPU)
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.

        Args:
            base_url (str): The base URL for the Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url.rstrip("/")

    def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Connection failed: {e}")
            return False

    def list_models(self) -> List[Dict]:
        """
        Get a list of available models.

        Returns:
            List[Dict]: List of available models
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models: {e}")
            return []

    def generate_response(
        self, model: str, prompt: str, stream: bool = False
    ) -> Optional[str]:
        """
        Generate a response from the specified model.

        Args:
            model (str): The name of the model to use
            prompt (str): The input prompt
            stream (bool): Whether to stream the response

        Returns:
            Optional[str]: The generated response or None if error
        """
        try:
            payload = {"model": model, "prompt": prompt, "stream": stream}
            oopts = _ollama_inference_options()
            if oopts:
                payload["options"] = oopts

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(30, _DEFAULT_READ_TIMEOUT),
            )
            response.raise_for_status()

            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            full_response += data["response"]
                            print(data["response"], end="", flush=True)
                        if data.get("done"):
                            break
                print()  # New line after streaming
                return full_response
            else:
                # Handle non-streaming response
                return response.json().get("response", "")

        except requests.exceptions.RequestException as e:
            print(f"Error generating response: {e}")
            return None

    def chat(self, model: str, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Have a chat conversation with the model.

        Args:
            model (str): The name of the model to use
            messages (List[Dict]): List of messages in chat format

        Returns:
            Optional[str]: The model's response or None if error
        """
        try:
            payload = {"model": model, "messages": messages, "stream": False}
            oopts = _ollama_inference_options()
            if oopts:
                payload["options"] = oopts

            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=30
            )
            response.raise_for_status()

            return response.json().get("message", {}).get("content", "")

        except requests.exceptions.RequestException as e:
            print(f"Error in chat: {e}")
            return None

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
# Avoid matching inside decimals (e.g. "4.7–5.7").
# Avoid folding cookware liter ranges: "5–6 l" / "5 to 6 l" (quart→l) into midpoints.
_RANGE_NOT_LITERS = r"(?!\s+[lL]\b)"
# Do not treat the denominator of ``a/b`` as the start of ``N to M`` / ``N–M`` (e.g. ``1/2 to 3``
# inches must stay a size range, not ``1/2.5``).
RANGE_HYPHEN_RE = re.compile(
    r"(?<![.\d/])(\d+)\s*[-–]\s*(\d+)(?![.\d])" + _RANGE_NOT_LITERS
)
RANGE_TO_RE = re.compile(
    r"(?<![.\d/])(\d+)\s+to\s+(\d+)(?![.\d])" + _RANGE_NOT_LITERS,
    re.IGNORECASE,
)


def _format_range_midpoint(a: float, b: float) -> str:
    """Midpoint string: whole numbers without ``.0`` (avoids ``4.0 minutes``)."""
    mid = (a + b) / 2.0
    if abs(mid - round(mid)) < 1e-9:
        return str(int(round(mid)))
    s = str(round(mid, 2)).rstrip("0").rstrip(".")
    return s


def normalize_ranges(text: str) -> str:
    """Convert N–M or N to M to midpoint so regex can parse."""
    def repl(m):
        a, b = float(m.group(1)), float(m.group(2))
        return _format_range_midpoint(a, b)

    text = RANGE_HYPHEN_RE.sub(repl, text)
    text = RANGE_TO_RE.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# 5. Unicode fraction normalization (before parsing)
# ---------------------------------------------------------------------------
# Vulgar fractions → decimal strings (``normalize_unicode_fractions``).
UNICODE_FRACTIONS = {
    "½": "0.5",
    "¼": "0.25",
    "¾": "0.75",
    "⅓": "0.333333",
    "⅔": "0.666667",
    "⅛": "0.125",
    "⅜": "0.375",
    "⅝": "0.625",
    "⅞": "0.875",
    "⅙": "0.166667",
    "⅚": "0.833333",
    "⅕": "0.2",
    "⅖": "0.4",
    "⅗": "0.6",
    "⅘": "0.8",
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
    # Preparation participles — drop entirely (do not stem to verbs: ``soaked`` → ``soak``).
    "soaked",
    "soaking",
    "rehydrated",
    "skewered",
})
# Single-word rest = stripped too much (e.g. "1 cup green" from "1 cup green split peas") — reject
REST_ONLY_ADJECTIVE = frozenset({"green", "red", "white"})
# Singularize last noun only for count-based units; volume/mass keep plural (e.g. 3 cup cherry tomatoes)
COUNT_UNITS_SINGULARIZE = frozenset({"piece", "slice", "clove", "leaf", "sprig", "stalk", "cube"})
# Canonical ingredient names for ML vocabulary (applied after _clean_rest)
INGREDIENT_ALIASES = {
    "parmigiano-reggiano": "parmesan cheese",
    "green onion greens": "green onion",
    "dry mustard": "mustard powder",
    "saltines": "saltine cracker",
    "saltine crackers": "saltine cracker",
    "scallions": "green onion",
    "scallion": "green onion",
    "mozzarella": "mozzarella cheese",
    "aubergine": "eggplant",
    "red pepper flakes": "chili flakes",
    "red pepper flake": "chili flakes",
    "red pepe flakes": "chili flakes",
    "brandied sour cherry": "brandied cherry",
    "brandied sour cherries": "brandied cherries",
}
# Rest ending in these = missing noun (e.g. "2 cup low-fat" from "low-fat mozzarella")
FAT_ONLY_ENDINGS = frozenset({"low-fat", "reduced-fat", "fat-free", "full-fat"})
# Unit tokens for "plus N unit" / "and N unit" stripping (compound units)
_COMPOUND_UNIT_PATTERN = r"(?:teaspoons?|tablespoons?|tsp|tbsp|cups?|g|kg|ml|l)\b"

# Log invalid output units for review
log = logging.getLogger("normalize_ingredients")
if not log.handlers:
    log.addHandler(logging.StreamHandler(sys.stderr))
    log.setLevel(logging.WARNING)


def normalize_unicode_fractions(s: str) -> str:
    """Replace ½ ¼ ¾ etc. with decimal strings (e.g. ``½`` → ``0.5``) before parsing."""
    if not s:
        return s
    # ``2½`` → ``2 ½`` so replacement does not become ``20.5``.
    for ch in UNICODE_FRACTIONS:
        s = re.sub(rf"(\d)(?={re.escape(ch)})", r"\1 ", s)
    for char, replacement in UNICODE_FRACTIONS.items():
        s = s.replace(char, replacement)
    # ``1 ½`` → ``1 0.5`` → ``1.5`` (same role as ASCII ``1 1/2`` merge later in normalize).
    def _merge_mixed(m: re.Match[str]) -> str:
        v = round(int(m.group(1)) + float(m.group(2)), 6)
        out = str(v)
        if "." in out:
            out = out.rstrip("0").rstrip(".")
        return out

    s = re.sub(r"(\d+)\s+(0\.\d+)\b", _merge_mixed, s)
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


# Peanut/cookie butters — keep *piece* if ever used; dairy butter is sold by volume (cup).
_NUT_OR_NONDAIRY_BUTTER = re.compile(
    r"\b(?:peanut|almond|cashew|macadamia|sunflower|sesame|cookie|apple|shea|cocoa|nut)\s+butter\b",
    re.IGNORECASE,
)
_DAIRY_BUTTER_NAME = re.compile(
    r"^(?:unsalted\s+|salted\s+|sweet\s+)?butter\b(?:\s*[,:]|\s*$)",
    re.IGNORECASE,
)


def _piece_butter_to_cup_line(line: str) -> str:
    """
    ``0.5 piece unsalted butter`` → ``0.5 cup unsalted butter`` (butter is not a count/piece item here).
    Skips peanut butter and other *X butter* spreads.
    """
    parts = line.split(maxsplit=2)
    if len(parts) < 3:
        return line
    qty, unit, rest = parts[0], parts[1], parts[2]
    if unit.lower() != "piece" or not re.match(r"^\d+(\.\d+)?$", qty):
        return line
    r = rest.strip()
    if _NUT_OR_NONDAIRY_BUTTER.search(r):
        return line
    if _DAIRY_BUTTER_NAME.match(r.strip()):
        return f"{qty} cup {rest}".strip()
    return line


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
        return _piece_butter_to_cup_line(
            f"{qty} {u} {rest}".strip() if rest else f"{qty} {u}"
        )
    if u in ALIAS_TO_FINAL:
        new_unit = ALIAS_TO_FINAL[u]
        merged = f"{qty} {new_unit} {rest}".strip() if rest else f"{qty} {new_unit}"
        return _piece_butter_to_cup_line(merged)
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
    # 3. Irregular: leaves → leaf (avoid "leave"); eggs → egg with ``piece`` counts
    if last == "leaves":
        words[-1] = "leaf"
    elif last == "eggs":
        words[-1] = "egg"
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
    # ``0.5 cup porcini`` → ``porcini mushroom`` — do not touch ``porcini mushroom(s)`` or ``porcini powder``.
    out = re.sub(
        r"\bporcini\b(?!\s+mushrooms?\b)(?!\s+powder\b)",
        "porcini mushroom",
        out,
        flags=re.IGNORECASE,
    )
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
    orig = _strip_trailing_ingredient_descriptors(orig)
    if not orig:
        return None
    # Drop literal "None" from dataset
    if orig.lower() == "none":
        return None
    # Recipe yield notes, not ingredients: "(makes 1 cup)", "Makes about 2 cups"
    if re.match(r"^\s*\(?\s*(?:makes|yields)\b", orig, re.IGNORECASE):
        return None
    # 3. Single-token unit (e.g. "cup") — reject before parsing or LLM
    if orig.lower() in FINAL_UNITS:
        return None
    if _NUTRITION_METADATA_LINE.search(orig):
        return None
    if re.fullmatch(r"cups?", orig.strip(), re.IGNORECASE):
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
    cleaned = fix_tight_numeric_unit_spacing(cleaned)

    # 3. Ingredient-info / accompaniments / makes lines — not real ingredients
    if cleaned.lower().startswith(("ingredient info", "accompaniments", "makes")):
        return None
    if re.match(r"^\s*\(?\s*makes\b", cleaned, re.IGNORECASE):
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
    # 2. Strip trailing ", divided" / ", chopped" etc. at END only — NOT "low-fat, preferably …"
    # (a blanket ``first comma → drop rest`` turned ``2 cups low-fat, … marinara`` into ``2 cups low-fat``,
    # which then failed FAT_ONLY_ENDINGS and dropped the whole ingredient).
    cleaned = re.sub(
        r",\s*(?:.*\b)?(?:"
        r"divided|halved|quartered|peeled|trimmed|cored|seeded|hulled|stemmed|thawed|"
        r"soaked|soaking|rehydrated|"
        r"grated|shredded|sliced|diced|chopped|minced|slivered|crushed|smashed|"
        r"finely chopped|coarsely chopped|roughly chopped|thinly sliced|"
        r"optional|to taste|for garnish|for serving|at room temperature|skewered"
        r")\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    # Strip size descriptors: "2-inch-diameter", "1.5-inch", "1/2-inch" — keep quantity, drop dimension
    cleaned = INCH_SIZE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*[-–]\s*inch(?:es)?\b", r"\1", cleaned, flags=re.IGNORECASE)
    # "piece of red beet" (after strip) → "1 piece red beet" so pipeline parses it
    cleaned = PIECE_OF_RE.sub("1 piece ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")

    # Cocktail garnish: ``brandied sour cherry, skewered`` → ``1 piece brandied cherry`` (count + name).
    if not re.search(r"\d", cleaned) and re.fullmatch(
        r"brandied(?:\s+sour)?\s+cherry|brandied(?:\s+sour)?\s+cherries",
        cleaned,
        re.IGNORECASE,
    ):
        cleaned = "1 piece brandied cherry"

    # 4. Short-circuit: "salt to taste" etc. — unit consistency doesn't need LLM
    if cleaned.lower().endswith("to taste"):
        return cleaned

    # 4a. No numeric quantity: keep ingredient *name only* (never invent clove/cup/etc. via LLM).
    # Examples: "Freshly ground black pepper", "Kosher salt". Skip lines that look like they still
    # carry an English number ("one onion", "half a lemon") — those need parsing/LLM.
    if not re.search(r"\d", cleaned):
        if re.match(
            r"^(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
            r"a|an|half)\b",
            cleaned,
            re.IGNORECASE,
        ):
            pass  # fall through to parsers / LLM
        else:
            if len(cleaned) < 2:
                return None
            if any(w in cleaned.lower() for w in EQUIPMENT_WORDS):
                return None
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
        # If original had no numeric quantity, do not accept a line that *starts with a unit token*
        # (e.g. "clove Freshly ground black pepper" — clove is not a quantity here).
        if not re.search(r"\d", ing.strip()):
            ots = out.split()
            if ots and ots[0].lower() in FINAL_UNITS and not re.match(r"^\d", out.strip()):
                return None
        return out
    except Exception:
        return None


def _apply_instruction_unit_normalization_to_recipe(recipe: dict) -> None:
    """Stage 2: normalize unicode fractions, ranges, Tbsp/cup/mass in each instruction step."""
    inst = recipe.get("instructions")
    if not isinstance(inst, dict):
        return
    for k, v in list(inst.items()):
        if not isinstance(v, str):
            continue
        recipe["instructions"][k] = normalize_instruction_units(v)


def _micro_clean_instructions_in_recipe(
    recipe: dict,
    client,
    use_ollama: bool,
    model: str,
    stats: dict,
) -> None:
    """
    Stage 3b (optional): one-sentence Ollama polish for heuristically broken steps only.
    Runs only when Ollama is enabled.
    """
    if not use_ollama or not client:
        return
    inst = recipe.get("instructions")
    if not isinstance(inst, dict):
        return
    ings = [x for x in (recipe.get("ingredients") or []) if isinstance(x, str)]
    for k, v in list(inst.items()):
        if not isinstance(v, str):
            continue
        if not _instruction_step_needs_micro_clean(v):
            continue
        out = clean_instruction_sentence(v, client, model, ings)
        if out and len(out) < max(400, len(v) * 4):
            recipe["instructions"][k] = out
            stats["instruction_micro_clean"] = stats.get("instruction_micro_clean", 0) + 1


def _process_single_recipe(
    recipe: dict,
    client,
    use_ollama: bool,
    model: str,
) -> tuple[dict, dict]:
    """
    Per-recipe pipeline (stages are explicit and ordered):

    1. Normalize ingredient lines (deterministic + optional Ollama + repair).
    2. Normalize instruction text units (unicode fractions, ranges, Tbsp/cup/mass).
    3. Align vague volume phrases in instructions with decimal forms from ingredients (half cup → 0.5 cup).
    4. Optional micro-clean: Ollama one-sentence polish for heuristically broken steps.
    5. Caller / ``process_recipe``: junk filtering, full Ollama rewrite or deterministic standardize, embedding text.

    Returns (mutated recipe, stats dict).
    """
    stats = {
        "normalized": 0,
        "ollama": 0,
        "unparseable": 0,
        "repair": 0,
        "instruction_micro_clean": 0,
    }
    ings = recipe.get("ingredients") or []
    new_ings: list = []
    for ing in ings:
        if not ing or not isinstance(ing, str):
            if ing:
                new_ings.append(ing)
            continue
        if not ing.strip() or ing.strip().lower() == "none":
            continue
        if should_skip_ingredient_line(ing):
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

    # --- Stage 2: instruction unit normalization (deterministic, in-place on recipe["instructions"]) ---
    _apply_instruction_unit_normalization_to_recipe(recipe)
    # --- Stage 3: align vague volumes with ingredient list ---
    align_vague_instruction_volumes_from_ingredients(recipe)
    # --- Stage 4: optional micro-clean for broken sentences ---
    _micro_clean_instructions_in_recipe(recipe, client, use_ollama, model, stats)

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


def normalize_main():
    import argparse

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
            _print_gpu_and_ollama_status()

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

    # Progress on stderr so stdout stays clean for piping; TTY check on stderr or stdout
    show_progress = (not args.no_progress) and (sys.stderr.isatty() or sys.stdout.isatty())
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
            # Rate must use recipes finished *this session*; after --resume, ``done`` includes prior index.
            completed_this_run = done - next_index
            eta_sec = (
                (chunk_len - done) * (elapsed / completed_this_run)
                if completed_this_run > 0
                else None
            )
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
                print(f"\r{line}", end="", flush=True, file=sys.stderr)
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
                print(line.strip(), flush=True, file=sys.stderr)

            if done % args.checkpoint_every == 0 or done == chunk_len:
                _atomic_write_json(output_path, work_slice)
                with open(checkpoint_path, "w") as f:
                    json.dump(_checkpoint_payload(done, complete=False), f, indent=2)
                # Newline before checkpoint so we don't overwrite the \r progress line on TTYs
                prefix = "\n" if show_progress else ""
                print(f"{prefix}  checkpoint: {done}/{chunk_len} recipes saved → {output_path}", flush=True, file=sys.stderr)
                if show_progress:
                    last_ck_line = False
    finally:
        if show_progress and last_ck_line:
            print(flush=True, file=sys.stderr)

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



# --- Embedding: food names only (no quantities / units) ---
# All unit spellings we strip from the left of an ingredient line
_UNIT_TOKENS_FOR_EMBEDDING: frozenset[str] = frozenset(
    {u.lower() for u in FINAL_UNITS} | {k.lower() for k in UNIT_ALIASES}
)


def _is_embedding_unit_token(w: str) -> bool:
    w = w.lower().rstrip(",.;:()")
    if w in _UNIT_TOKENS_FOR_EMBEDDING:
        return True
    if w in FINAL_UNITS:
        return True
    if len(w) > 1 and w.endswith("s") and w[:-1] in _UNIT_TOKENS_FOR_EMBEDDING:
        return True
    if len(w) > 1 and w.endswith("s") and w[:-1] in FINAL_UNITS:
        return True
    return False


def extract_food_name_for_embedding_line(ing_line: str) -> str:
    """
    Ingredient line → food name only for ``ing_embedding_text`` (no amounts, no units).
    Handles plural units (tablespoons), aliases (tbsp), and mixed fractions (2 1/2 cup …).
    """
    line = ing_line.strip()
    if not line:
        return ""
    line = re.sub(r"^\s*(?:about|approximately|around)\s+", "", line, flags=re.IGNORECASE)
    norm = normalize_ingredient_deterministic(line)
    if norm:
        fp = _food_phrase_from_normalized(norm)
        if fp:
            return _strip_embedding_diet_quality_prefixes(_scrub_embedding_name_tail(fp))

    t = line
    for _ in range(10):
        parts = t.split()
        if len(parts) < 2:
            break
        # Mixed number + fraction + unit: "2 1/2 cup flour"
        if (
            len(parts) >= 4
            and re.match(r"^\d+$", parts[0])
            and re.match(r"^\d+/\d+$", parts[1])
            and _is_embedding_unit_token(parts[2])
        ):
            t = " ".join(parts[3:]).strip()
            continue
        # Simple: qty unit rest
        if len(parts) >= 3:
            q = parts[0]
            if re.match(r"^\d+(?:\.\d+)?$", q) or re.match(r"^\d+/\d+$", q):
                if _is_embedding_unit_token(parts[1]):
                    t = " ".join(parts[2:]).strip()
                    continue
        break

    return _strip_embedding_diet_quality_prefixes(_scrub_embedding_name_tail(t))


def _scrub_embedding_name_tail(s: str) -> str:
    """Drop stray leading 'a' article before food words; trim."""
    t = s.strip()
    t = re.sub(r"^(?:a|an)\s+", "", t, flags=re.IGNORECASE)
    return t.strip()


def _strip_embedding_diet_quality_prefixes(name: str) -> str:
    """
    Remove leading ``low-fat, …``, ``lower-sodium, …`` so ``marinara sauce`` can embed
    (otherwise ``is_valid_embedding_name`` rejects on ``^low-fat``).
    """
    t = name.strip()
    for _ in range(8):
        prev = t
        t = re.sub(
            r"^(?:low-fat|reduced-fat|non-fat|full-fat|fat-free)\s*,\s*",
            "",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r"^(?:preferably\s+)?(?:lower-sodium|low-sodium|no-salt(?:-added)?)\s*,\s*",
            "",
            t,
            flags=re.IGNORECASE,
        )
        # ``preferably lower-sodium marinara`` (no comma after low-fat strip)
        t = re.sub(
            r"^(?:preferably\s+)?(?:lower-sodium|low-sodium|no-salt(?:-added)?)\s+",
            "",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(r"^preferably\s+", "", t, flags=re.IGNORECASE)
        if t == prev:
            break
    return t.strip()


def _strip_trailing_ingredient_descriptors(s: str) -> str:
    """Drop trailing ``room temperature``, ``drained``, ``peeled``, etc. before parsing."""
    t = s.strip()
    t = re.sub(r",\s*(?:at\s+)?room\s+temperature\b", "", t, flags=re.IGNORECASE)
    t = re.sub(
        r",\s*(?:finely\s+)?(?:picked\s+over|well\s+shaken|drained\s+and\s+rinsed|drained|"
        r"rinsed|peeled|pitted|hulled|trimmed|seeded)\b[^,]*$",
        "",
        t,
        flags=re.IGNORECASE,
    )
    # ``Lemon wedges for serving`` → lemon wedge
    t = re.sub(r"(?:,|\s)+\s*for\s+serving\s*$", "", t, flags=re.IGNORECASE)
    # ``250 g wood chips, soaked`` — drop preparation verb (handled in ``DESCRIPTOR_WORDS`` too)
    t = re.sub(r",\s*soaked\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r",\s*soaking\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r",\s*rehydrated\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r",\s*skewered\s*$", "", t, flags=re.IGNORECASE)
    return t.strip()


def fix_tight_numeric_unit_spacing(s: str) -> str:
    """
    Insert missing space before mass/volume units: ``200g`` → ``200 g``, ``30g`` → ``30 g``.
    Avoids touching fractions like ``1/2g`` (digit after ``/``).
    """
    if not s:
        return s
    t = s
    t = re.sub(r"(?<!/)(\d+(?:\.\d+)?)\s*(g|kg|mg|ml|l)\b", r"\1 \2", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!/)(\d+(?:\.\d+)?)\s*(oz|lb)\b", r"\1 \2", t, flags=re.IGNORECASE)
    return t


# --- inlined from extract_unique_ingredient_names (avoid extra module) ---
def ingredient_name_from_line(line: str) -> str:
    """Best-effort food name only (same as embedding extraction)."""
    return extract_food_name_for_embedding_line(line)


QTY_RE = re.compile(r"^\d+(\.\d+)?$")

# ---------------------------------------------------------------------------
# Ingredient line filtering (not food / not for embedding)
# ---------------------------------------------------------------------------
_ING_JUNK_PREFIX = re.compile(
    r"^\s*(?:\(\s*)?(?:ingredient\s+info|accompaniments?|normalized\s*:|makes|yields)\b",
    re.IGNORECASE,
)
# Nutrition facts / yield metadata mistaken for ingredients (not food lines).
# Do NOT use a blanket ``per … serving`` anywhere in the string — package copy often
# says e.g. "no more than 4 g of fat per 9-piece serving" on real ingredient lines.
_NUTRITION_METADATA_LINE = re.compile(
    r"(?i)(?:"
    r"^\s*makes\b|"
    r"^\s*\(\s*per\s+[^)]{0,80}\bserving\s*\)\s*$|"
    r"^\s*per\s+[^,\n]{0,50}\bserving\s*:?\s*$|"
    r"(?:^|[\s;])(?:calories|protein|fiber|cholesterol|sodium)\s*:\s*[\d.]+|"
    r"(?:^|[\s;])(?:total\s+)?fat\s*:\s*[\d.]+|"
    r"(?:^|[\s;])saturated\s+fat\s*:\s*[\d.]+|"
    r"(?:^|[\s;])unsaturated\s+fat\s*:\s*[\d.]+|"
    r"(?:^|[\s;])carbohydrates?\s*:\s*[\d.]+"
    r")",
)
# Bad LLM/OCR lines: no quantity but starts with container word + name
_BARE_CONTAINER_INGREDIENT = re.compile(
    r"^\s*(?:packet|bag|box|tin|jar|bottle|carton)\s+\S",
    re.IGNORECASE,
)
# "pepper" in unique_ingredient_names maps to black pepper — must not match inside "green bell pepper"
_CHILE_OR_BELL_PEPPER_CONTEXT = re.compile(
    r"\b(?:green|red|yellow|orange)\s+bell\s+pepper\b|"
    r"\b(?:cayenne|chile|chili|chilli|sweet|aleppo|ghost|habanero|jalapeño|jalapeno|serrano|"
    r"anaheim|poblano|shishito|banana|scotch\s+bonnet)\s+pepper\b",
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
    "saltines": "saltine cracker",
    "saltine cracker": "saltine cracker",
    "dry mustard": "mustard powder",
    "mustard powder": "mustard powder",
    "green onion greens": "green onion",
    "green onion": "green onion",
    "crushed tomatoes": "tomato",
    "crushed tomato": "tomato",
    "cherry tomatoes": "tomato",
    "cherry tomato": "tomato",
    "lemon wedge": "lemon",
    "lemon wedges": "lemon",
    "lemon wedge piece": "lemon",
    "lemon wedges piece": "lemon",
    # Mis-parsed count unit + food (e.g. ``cut into cubes`` leaking as ``cube bacon``)
    "cube bacon": "bacon",
    "slab bacon": "bacon",
    "mozzarella": "mozzarella cheese",
    "porcini": "porcini mushroom",
    "porcinis": "porcini mushroom",
    "red pepper flakes": "chili flakes",
    "red pepe flakes": "chili flakes",
    "red pepper flake": "chili flakes",
    "lemon wedge for serving": "lemon",
    "lemon wedges for serving": "lemon",
    "lemon wedges piece for serving": "lemon",
    # Chile hierarchy: whole pods vs ground — dedupe noisy ``ancho chile`` + ``chile`` + powder lines.
    "chile": "chile pepper",
    "chiles": "chile pepper",
    "ancho chile": "chile pepper",
    "ancho chiles": "chile pepper",
    "ancho chili": "chile pepper",
    "dried ancho chile": "chile pepper",
    "dried ancho chiles": "chile pepper",
    "chili pepper": "chile pepper",
    "chili peppers": "chile pepper",
    "ancho chili powder": "chili powder",
    "ancho chile powder": "chili powder",
    "chile powder": "chili powder",
    # Beef / brisket — prefer specific cut over generic ``beef`` (dedupe after token list built).
    "flat cut brisket": "brisket",
    "flat-cut brisket": "brisket",
    "point cut brisket": "brisket",
    "point-cut brisket": "brisket",
    "whole brisket": "brisket",
    "beef brisket": "brisket",
    "brisket beef": "brisket",
    "brisket piece": "brisket",
    "brisket pieces": "brisket",
}


# Collapse specific varieties → base food for embeddings (dedupe cherry tomato + tomato → tomato).
_EMBEDDING_VARIETY_TO_BASE: dict[str, str] = {
    "cherry tomato": "tomato",
    "cherry tomatoes": "tomato",
    "kalamata olive": "olive",
    "kalamata olives": "olive",
    "green olive": "olive",
    "green olives": "olive",
    "black olive": "olive",
    "black olives": "olive",
    "red skin potato": "potato",
    "red skin potatoes": "potato",
    "red skinned potato": "potato",
    "red skinned potatoes": "potato",
    "button mushroom": "mushroom",
    "button mushrooms": "mushroom",
}

# Leading tokens that are preparation/state, not food identity (strip repeatedly for embeddings).
_EMBEDDING_STATE_PREFIX_WORDS: frozenset[str] = frozenset(
    {
        "cooked",
        "raw",
        "frozen",
        "dried",
        "chilled",
        "leftover",
        "steamed",
        "boiled",
        "grilled",
        "roasted",
        "fried",
        "baked",
        "smoked",
        "toasted",
        "thawed",
        "reheated",
        "pickled",
        "marinated",
        "prepared",
        "uncooked",
    }
)

# Named sauce / sub-recipe lines — not base ingredients for embedding search.
_EMBEDDING_SUBRECIPE_BLOCKLIST = frozenset(
    {
        "ancho chile sauce",
        "ancho chili sauce",
    }
)


def _apply_embedding_overrides(std: str) -> str:
    s = std.strip().lower()
    s = _EMBEDDING_CANONICAL_OVERRIDES.get(s, s)
    # Garnish phrasing left in normalized names → single token
    if re.match(r"^lemon\s+wedges?(?:\s+piece)?$", s):
        return "lemon"
    # Strip spurious leading count-shape tokens before common proteins (embedding only).
    m = re.match(r"^(?:cube|slab)\s+(bacon)\b", s, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return s


def _strip_embedding_state_prefixes(t: str) -> str:
    """Remove leading state/prep words (``cooked leftover rice`` → ``rice``)."""
    t = t.strip()
    for _ in range(12):
        if not t:
            break
        m = re.match(r"^day[- ]old\s+", t, re.IGNORECASE)
        if m:
            t = t[m.end() :].strip()
            continue
        parts = t.split()
        if not parts:
            break
        if parts[0].lower() in _EMBEDDING_STATE_PREFIX_WORDS:
            t = " ".join(parts[1:]).strip()
            continue
        break
    return t


def _normalize_embedding_piece_wording(t: str) -> str:
    """
    Remove count-unit leakage into food names for embeddings.

    - ``piece of salmon`` → ``salmon``
    - ``brisket piece`` / ``salmon pieces`` → ``brisket`` / ``salmon``
    """
    t = t.strip()
    if not t:
        return t
    m = re.match(r"^pieces?\s+of\s+(.+)$", t, re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    parts = t.split()
    while len(parts) >= 2 and parts[-1].lower() in ("piece", "pieces"):
        parts.pop()
    return " ".join(parts).strip()


def _collapse_embedding_cut_phrases(t: str) -> str:
    """Map ``flat cut brisket`` / ``point cut brisket`` → ``brisket`` (hyphens already normalized)."""
    t = t.strip()
    if not t:
        return t
    m = re.match(
        r"^(?:(?:flat|point)\s+cut|whole)\s+(brisket)s?\b(.*)$",
        t,
        re.IGNORECASE,
    )
    if m:
        return (m.group(1) + m.group(2)).strip().lower()
    m = re.match(r"^beef\s+(brisket)s?\b(.*)$", t, re.IGNORECASE)
    if m:
        return (m.group(1) + m.group(2)).strip().lower()
    m = re.match(r"^(brisket)s?\s+beef\b(.*)$", t, re.IGNORECASE)
    if m:
        return (m.group(1) + m.group(2)).strip().lower()
    return t


def _dedupe_embedding_generic_vs_specific(tokens: list[str]) -> list[str]:
    """
    Remove redundant parent tokens when a specific ingredient is already present
    (e.g. ``beef`` + ``brisket`` + ``flat cut brisket`` → ``brisket`` only).
    """
    lowered = {x.casefold() for x in tokens}
    if "brisket" in lowered:
        return [x for x in tokens if x.casefold() != "beef"]
    return tokens


def _dedupe_embedding_mushroom_with_porcini(tokens: list[str]) -> list[str]:
    """
    If ``porcini mushroom`` is present, drop bare ``mushroom`` (from button/white mushroom lines)
    to avoid ``porcini mushroom`` + ``mushroom`` duplicates.
    """
    lowered = {x.casefold() for x in tokens}
    if "porcini mushroom" in lowered and "mushroom" in lowered:
        return [x for x in tokens if x.casefold() != "mushroom"]
    return tokens


def _collapse_embedding_food_identity(s: str) -> str | None:
    """
    Strip serving/cooking labels, map varieties to base foods, drop sub-recipe titles.
    Returns ``None`` to omit the token from ``ing_embedding_text``.
    """
    t = s.strip().lower()
    # Unify hyphenated descriptors with spaced forms for lookup (``red-skinned`` → ``red skinned``).
    t = re.sub(r"[-_]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _collapse_embedding_cut_phrases(t)
    t = _normalize_embedding_piece_wording(t)
    t = re.sub(r"(?:,|\s)+for\s+serving\s*$", "", t).strip()
    t = _strip_embedding_state_prefixes(t)
    if not t:
        return None
    if t in _EMBEDDING_SUBRECIPE_BLOCKLIST:
        return None
    # Fixed-point: variety map may expose another stripable prefix (``cooked cherry tomato``).
    for _ in range(6):
        before = t
        t = _EMBEDDING_VARIETY_TO_BASE.get(t, t)
        t = _strip_embedding_state_prefixes(t)
        if not t:
            return None
        if t == before:
            break
    t = _normalize_embedding_piece_wording(t)
    return t


def _pepper_map_key_is_false_positive(key: str, line_lower: str, candidate: str) -> bool:
    """Avoid mapping key 'pepper' → black pepper when line is bell/chile-type pepper."""
    if key.lower() != "pepper":
        return False
    ctx = f"{line_lower} :: {candidate}"
    return bool(_CHILE_OR_BELL_PEPPER_CONTEXT.search(ctx))

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
    # Bare unit tokens from broken OCR / nutrition blocks
    if re.fullmatch(r"cups?", low):
        return True
    if _NUTRITION_METADATA_LINE.search(s):
        return True
    if _ING_JUNK_PREFIX.match(s):
        return True
    # Standalone "(makes 1 cup)" with leading paren only
    if re.match(r"^\s*\(\s*makes\b", s, re.IGNORECASE):
        return True
    if _BARE_CONTAINER_INGREDIENT.match(s) and not re.match(r"^\d", s):
        return True
    if _DISPOSABLE_OR_FOIL.search(s):
        return True
    if _DIMENSION_GARBAGE.search(s):
        return True

    n = normalize_ingredient_deterministic(s)
    if n is None:
        # Keep salt / pepper lines without quantities (but not recipe notes / junk)
        if not re.match(r"^\d", low):
            if _ING_JUNK_PREFIX.match(s) or re.match(r"^\s*\(\s*makes\b", s, re.IGNORECASE):
                return True
            if _BARE_CONTAINER_INGREDIENT.match(s):
                return True
            return False
        # Quantity-leading lines the deterministic parser could not normalize: keep them so the
        # main pipeline can store the raw line (e.g. ``1/4 pound slab bacon, cut into … chunks``,
        # ``4 sweet potatoes … peeled …``). Only drop when clearly equipment / dimensions / foil.
        if any(w in low for w in EQUIPMENT_WORDS):
            return True
        if _DISPOSABLE_OR_FOIL.search(s):
            return True
        if _DIMENSION_GARBAGE.search(s):
            return True
        return False
    return False


def _food_phrase_from_normalized(norm: str) -> str:
    parts = norm.split()
    if len(parts) >= 3 and QTY_RE.match(parts[0]) and parts[1].lower() in FINAL_UNITS:
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
    t = _strip_embedding_diet_quality_prefixes(name.strip()).lower()
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
    if re.fullmatch(
        r"(?:low-fat|low fat|full-fat|reduced-fat|non-fat|fat-free)(?:\s*,\s*[^,]+)*",
        t,
    ):
        return False
    if _FUEL_OR_NONFOOD_EMBEDDING.search(t):
        return False
    return True


def map_ingredient_to_standard(ing_line: str, mapping: dict[str, str]) -> str | None:
    """Map one line to canonical name, or None if should not appear in embedding."""
    line_lower = ing_line.lower().strip()
    # Food name only — never quantities/units (see extract_food_name_for_embedding_line).
    candidate = extract_food_name_for_embedding_line(ing_line).lower()
    if not candidate.strip():
        stripped = re.sub(
            r"^(\d+\s+\d+/\d+|\d+/\d+|\d+\.?\d*)\s+",
            "",
            ing_line,
            flags=re.IGNORECASE,
        ).strip()
        if stripped:
            candidate = extract_food_name_for_embedding_line(stripped).lower()

    best_val: str | None = None
    best_key_len = -1
    for k, v in mapping.items():
        kl = k.lower()
        if kl in line_lower or (candidate and kl in candidate):
            if _pepper_map_key_is_false_positive(k, line_lower, candidate):
                continue
            if len(k) > best_key_len:
                best_key_len = len(k)
                best_val = v

    if best_val:
        std = _apply_embedding_overrides(best_val.strip().lower())
        return std if is_valid_embedding_name(std) else None

    for k, v in sorted(mapping.items(), key=lambda x: -len(x[0])):
        kl = k.lower()
        if kl not in line_lower:
            continue
        if _pepper_map_key_is_false_positive(k, line_lower, candidate):
            continue
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


def dedupe_and_sort_ingredient_lines(lines: list[str]) -> list[str]:
    """Stable dedupe (first occurrence wins), then alphabetical order for cleaner output + embeddings."""
    unique = list(dict.fromkeys(lines))
    return sorted(unique, key=str.casefold)


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
        # Past participles must not lose final letters (``soaked`` is not plural ``soake``).
        "soaked",
        "soaking",
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


def _embedding_dedupe_key(name: str) -> str:
    """Normalize for duplicate detection (case, plural, whitespace)."""
    s = re.sub(r"\s+", " ", name.strip().lower())
    return singularize_embedding_phrase(s)


# High-level taxonomy labels in ``tags.ingredient`` — not distinct foods; skip for embeddings.
_TAG_INGREDIENT_TAXONOMY_SKIP = frozenset(
    {
        "meat",
        "cured meat",
        "seafood",
        "fish",
        "shellfish",
        "vegetable",
        "fruit",
        "dairy",
        "cheese",
        "herbs & spices",
        "herbs and spices",
        "root vegetable",
        "leafy greens",
        "pasta",
        "rice & grains",
        "rice and grains",
        "bean and legume",
        "nuts",
        "beverages",
        "alcohol",
        "egg",
        "sweet",
        "savory",
    }
)


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


def _embedding_name_from_tag_label(label: str, mapping: dict[str, str]) -> str | None:
    """Map a ``tags.ingredient`` label to a raw food phrase (same shape as ``map_ingredient_to_standard``)."""
    raw = label.strip()
    if not raw:
        return None
    if raw.casefold() in _TAG_INGREDIENT_TAXONOMY_SKIP:
        return None
    std = map_ingredient_to_standard(raw, mapping)
    if not std:
        fb = extract_food_name_for_embedding_line(raw)
        if not fb:
            return None
        std = _canonicalize_embedding_fallback(fb.lower())
    return std


def build_ing_embedding_text(
    ingredients: list[str],
    mapping: dict[str, str],
    tag_ingredient_labels: list[str] | None = None,
) -> str:
    """
    Build embedding string from normalized ingredient lines, optionally merging
    ``tags.ingredient`` labels that are not already represented (deduped by
    singular/casefold key). Skips generic taxonomy rows (``Meat``, ``Vegetable``, …).
    """
    seen_keys: set[str] = set()
    ordered: list[str] = []

    def _add_embedding_token(std: str | None) -> None:
        if not std:
            return
        s = singularize_embedding_phrase(std)
        s = _apply_embedding_overrides(s)
        s = _collapse_embedding_food_identity(s)
        if not s or not is_valid_embedding_name(s):
            return
        key = _embedding_dedupe_key(s)
        if key in seen_keys:
            return
        seen_keys.add(key)
        ordered.append(s)

    for ing in ingredients:
        _add_embedding_token(map_ingredient_to_standard(ing, mapping))

    for lab in tag_ingredient_labels or []:
        if not isinstance(lab, str):
            continue
        _add_embedding_token(_embedding_name_from_tag_label(lab, mapping))

    ordered = _dedupe_embedding_generic_vs_specific(ordered)
    ordered = _dedupe_embedding_mushroom_with_porcini(ordered)
    ordered.sort(key=str.casefold)
    return "ingredients: " + ", ".join(ordered)


# ---------------------------------------------------------------------------
# Instructions: deterministic only for volume; mass lb/oz → g; °F untouched
# ---------------------------------------------------------------------------
_NUTRITION_LINE = re.compile(
    r"^\s*(calories|fat(?=\s*:)|total\s+fat|unsaturated\s+fat|saturated\s+fat|trans\s+fat|cholesterol|sodium|"
    r"total\s+carbohydrate|carbohydrates?|dietary\s+fiber|fiber|sugars?|protein|vitamin|calcium|iron)\b",
    re.IGNORECASE,
)

# Drop whole instruction lines that are nutrition facts (label : value). Uses ``fat`` only as
# ``Fat:`` at line start / after whitespace — not ``low-fat`` (hyphen before ``fat``).
_NUTRITION_INSTRUCTION_LINE_DROP = re.compile(
    r"(?is)^\s*(?:"
    r"calories|protein|carbohydrates?|total\s+carbohydrate|fiber|dietary\s+fiber|"
    r"sodium|cholesterol|sugars?|"
    r"total\s+fat|saturated\s+fat|unsaturated\s+fat|trans\s+fat|"
    r"fat|vitamin\s+[a-z]|calcium|iron"
    r")\s*:\s*"
)


def _line_looks_like_nutrition_label(t: str) -> bool:
    """True if a single line is a nutrition-facts row (before broader instruction processing)."""
    low = t.lower().strip()
    if not low:
        return False
    if _NUTRITION_INSTRUCTION_LINE_DROP.match(t):
        return True
    if _NUTRITION_LINE.match(t):
        return True
    if re.match(
        r"^\s*(?:calories|fat|protein|total\s+fat|unsaturated\s+fat|saturated\s+fat|trans\s+fat|"
        r"fiber|sodium|cholesterol|carbohydrates?|dietary\s+fiber|sugars?)\s*:\s*",
        t,
        re.I,
    ):
        return True
    if re.match(r"^\s*each\s+.*\bserving\s+has:?\s*$", t, re.I):
        return True
    if re.match(r"^\s*per\s+serving\s*:?\s*$", t, re.I):
        return True
    # Nutrition blocks often pair "Per … serving" with label rows; require ``:``-style
    # macros so we do not match package copy like "g of fat per 9-piece serving".
    if re.search(r"\bper\s+[^.\n]{0,50}?\bserving\b", low) and re.search(
        r"\b(?:calories|protein|fiber|cholesterol|sodium|carbohydrates?|"
        r"total\s+fat|saturated\s+fat|unsaturated\s+fat)\s*:",
        low,
    ):
        return True
    return False


def remove_nutrition_and_metadata_lines(text: str) -> str:
    """Drop nutrition facts and similar non-step lines."""
    lines = []
    for line in text.splitlines():
        t = line.strip()
        if _line_looks_like_nutrition_label(t):
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
    if _NUTRITION_INSTRUCTION_LINE_DROP.match(t):
        return True
    if _NUTRITION_LINE.match(t):
        return True
    if re.match(
        r"^\s*(?:calories|fat|protein|total\s+fat|unsaturated\s+fat|saturated\s+fat|trans\s+fat|"
        r"fiber|sodium|cholesterol|carbohydrates?|dietary\s+fiber|sugars?)\s*:\s*",
        t,
        re.I,
    ):
        return True
    # Standalone "(Per … serving)" / nutrition headers (not "per 9-piece serving" mid-sentence).
    if re.match(r"^\s*\(?\s*per\s+[^)\n]{0,60}\bserving\s*\)?\s*$", t, re.I):
        return True
    if re.search(r"^\s*per\s+serving\s*:?\s*$", low):
        return True
    if re.search(r"\bper\s+[^.\n]{0,50}?serving\b", low) and re.search(
        r"\b(?:calories|saturated|unsaturated|carbohydrates|cholesterol|fiber|protein|sodium|total\s+fat)\b",
        low,
    ):
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
    # Longest / specific fat labels first — a bare ``Fat:`` alternative would match inside
    # ``Unsaturated Fat:`` and leave the fragment ``Un``.
    s = re.sub(
        r"\s*(?:Unsaturated|Saturated|Trans)\s+Fat\s*:\s*[\d.]+\s*(?:g|mg|kcal|%)?(?:\s*each)?",
        " ",
        s,
        flags=re.IGNORECASE,
    )
    # Bare ``Fat: 7g`` (runs *after* Unsaturated/Saturated/Trans lines above).
    s = re.sub(
        r"(?<![A-Za-z])Fat\s*:\s*[\d.]+\s*(?:g|mg|kcal|%)?",
        " ",
        s,
        flags=re.IGNORECASE,
    )
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
            r"\s*(?:Calories|Protein|Total\s+Fat|Carbohydrates|Dietary\s+Fiber|Fiber|Sodium|"
            r"Cholesterol|Sugars?)\s*:\s*[\d.]+\s*(?:g|mg|kcal|%)?(?:\s*each)?",
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


# Instruction volume: one decimal style for ML (same as round_quantity for <1).
_INSTRUCTION_VOLUME_UNITS = (
    r"cup|cups|tablespoon|tablespoons|teaspoon|teaspoons"
)


def _instruction_volume_qty_str(v: float) -> str:
    """Format a volume quantity: no spurious ``.0``; trim trailing zeros."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    vs = str(round(v, 6)).rstrip("0").rstrip(".")
    return vs or "0"


def convert_cup_mixed_and_simple_fractions(s: str) -> str:
    """
    Mixed and simple fractions → decimal for **cup / tablespoon / teaspoon** (consistent dataset).

    - ``2 1/2 cups`` → ``2.5 cups`` (mixed first).
    - ``1/2 cup`` → ``0.5 cup``; ``1/4 teaspoon`` → ``0.25 teaspoon``.
    - Never leave ``2 0.5 cups`` (``fix_stray_integer_plus_decimal_cups`` repairs).
    """
    u_pat = _INSTRUCTION_VOLUME_UNITS

    def mixed_repl(m: re.Match) -> str:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if den == 0:
            return m.group(0)
        v = whole + num / den
        unit = m.group(4).lower()
        return f"{_instruction_volume_qty_str(v)} {unit}"

    s = re.sub(
        rf"\b(\d+)\s+(\d+)\s*/\s*(\d+)\s+({u_pat})\b",
        mixed_repl,
        s,
        flags=re.IGNORECASE,
    )

    def simple_repl(m: re.Match) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            return m.group(0)
        v = a / b
        unit = m.group(3).lower()
        return f"{_instruction_volume_qty_str(v)} {unit}"

    s = re.sub(
        rf"\b(\d+)\s*/\s*(\d+)\s+({u_pat})\b",
        simple_repl,
        s,
        flags=re.IGNORECASE,
    )
    return s


def fix_stray_integer_plus_decimal_cups(s: str) -> str:
    """Repair ``2 0.5 cups`` → ``2.5 cups`` (same for tbsp/tsp) if a prior pass left fragments."""

    def repl(m: re.Match) -> str:
        a, b, unit = float(m.group(1)), float(m.group(2)), m.group(3).lower()
        v = a + b
        return f"{_instruction_volume_qty_str(v)} {unit}"

    return re.sub(
        rf"\b(\d+)\s+(0\.\d+)\s+({_INSTRUCTION_VOLUME_UNITS})\b",
        repl,
        s,
        flags=re.IGNORECASE,
    )


def strip_integer_trailing_decimal_zeros(s: str) -> str:
    """
    ``4.0 minutes`` / ``4.0 cup`` → ``4 minutes`` / ``4 cup``; keeps ``1.5``, ``0.75``.
    Catches midpoint/range output and any ``round_quantity``-style ``.0`` leaks in prose.
    """
    return re.sub(r"\b(\d+)\.0+\b", r"\1", s)


# Cookware capacity: display standard **1 quart capacity → 1 l** (dataset rule).
# Recipe *ingredient* quarts are converted separately → cups (1 quart = 4 cups).


def _cookware_capacity_liters_str(n: float) -> str:
    """Emit ``N L`` with no spurious ``.0`` for whole quart ratings."""
    if abs(n - round(n)) < 1e-9:
        return str(int(round(n)))
    s = str(round(n, 2)).rstrip("0").rstrip(".")
    return s


def convert_instruction_quart_cookware_to_liters(s: str) -> str:
    """
    Replace quart-based *cookware* sizes with **l** (1 quart → 1 l) before
    ``normalize_ranges`` mangles hyphen ranges inside decimals.

    **Does not** match ingredient lines like ``3 quarts water`` or ``1 quart broth``
    (no vessel token right after ``quart``) — those become cups in
    ``convert_instruction_liquid_imperial_volume_to_cup``.

    Vessel list is driven by common phrases in source data (pot, saucepan, baking dish,
    casserole, bowl, skillet, ``6-quart capacity``, etc.).
    """
    if not s:
        return s

    def _lit_range(a: float, b: float) -> str:
        return f"{_cookware_capacity_liters_str(a)}–{_cookware_capacity_liters_str(b)} l"

    # "5- to 6-quart" / "5 - to 6-quart" (common print style)
    s = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*[-–]\s*to\s+(\d+(?:\.\d+)?)\s*[-–]?\s*quarts?\b",
        lambda m: _lit_range(float(m.group(1)), float(m.group(2))),
        s,
        flags=re.IGNORECASE,
    )
    # "5 to 6-quart" (no hyphen before "to")
    s = re.sub(
        r"\b(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*[-–]?\s*quarts?\b",
        lambda m: _lit_range(float(m.group(1)), float(m.group(2))),
        s,
        flags=re.IGNORECASE,
    )
    # Compact range: "5-6-quart pot"
    s = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*[-–]?\s*quarts?\b",
        lambda m: _lit_range(float(m.group(1)), float(m.group(2))),
        s,
        flags=re.IGNORECASE,
    )

    # Nouns/adjectives after quart that indicate cookware (longer alternations first).
    # From ``new_recipee_data.json`` samples: baking dish, casserole dish, mixing bowl,
    # braising pan, canning jars, "N-quart capacity", etc.
    _vessel = (
        r"((?:heavy\s+|large\s+|medium\s+|small\s+|nonstick\s+|wide\s+)*"
        r"(?:"
        r"dutch\s+oven|slow\s+cooker|baking\s+dish|mixing\s+bowl|"
        r"casserole\s+dish|braising\s+pan|braising\s+pot|canning\s+jars?|"
        r"stockpot|saucepan|skillet|roaster|kettle|wok|"
        r"casserole|bowl|pot|capacity"
        r")\b)"
    )

    def _single_lit(m: re.Match) -> str:
        n = float(m.group(1))
        return f"{_cookware_capacity_liters_str(n)} l {m.group(2)}"

    s = re.sub(
        rf"\b(\d+(?:\.\d+)?)\s*[-–]?\s*quarts?\s+{_vessel}",
        _single_lit,
        s,
        flags=re.IGNORECASE,
    )
    return s


def convert_instruction_liquid_imperial_volume_to_cup(s: str) -> str:
    """
    Remaining pint / quart / gallon *recipe* amounts in instructions → cups
    (1 pint = 2 cups, 1 quart = 4 cups, 1 gallon = 16 cups, US).
    Skips text already converted to L.
    """

    def _cups(n: float, mult: float) -> str:
        v = n * mult
        vs = str(round(v, 6)).rstrip("0").rstrip(".")
        return f"{vs} cup"

    # Pint
    s = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*pints?\b",
        lambda m: _cups(float(m.group(1)), 2.0),
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\b(\d+(?:\.\d+)?)\s+pt\b(?![a-z])",
        lambda m: _cups(float(m.group(1)), 2.0),
        s,
        flags=re.IGNORECASE,
    )
    # Quart / qt (what remains after cookware pass)
    s = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*quarts?\b",
        lambda m: _cups(float(m.group(1)), 4.0),
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*\bqts?\b(?![a-z])",
        lambda m: _cups(float(m.group(1)), 4.0),
        s,
        flags=re.IGNORECASE,
    )
    # Gallon
    s = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*gallons?\b",
        lambda m: _cups(float(m.group(1)), 16.0),
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*\bgal\b(?![a-z])",
        lambda m: _cups(float(m.group(1)), 16.0),
        s,
        flags=re.IGNORECASE,
    )
    return s


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


def normalize_instruction_units(text: str) -> str:
    """
    Stage 2 (per instruction text): deterministic unit/range normalization only.
    No nutrition stripping — that happens in ``standardize_instruction_step``.
    """
    if not text or not isinstance(text, str):
        return ""
    s = normalize_unicode_fractions(text)
    s = convert_instruction_quart_cookware_to_liters(s)
    s = normalize_ranges(s)
    s = expand_instruction_abbreviations(s)
    s = convert_cup_mixed_and_simple_fractions(s)
    s = fix_stray_integer_plus_decimal_cups(s)
    s = convert_instruction_mass_only(s)
    s = convert_instruction_liquid_imperial_volume_to_cup(s)
    s = strip_integer_trailing_decimal_zeros(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def filter_nutrition_lines_from_instruction_block(text: str) -> str:
    """Drop newline-separated nutrition rows before unit normalization (keeps cooking lines)."""
    if not text or not isinstance(text, str):
        return ""
    kept: list[str] = []
    for line in text.splitlines():
        t = line.strip()
        if not t:
            continue
        if _line_looks_like_nutrition_label(t):
            continue
        kept.append(line)
    return "\n".join(kept)


def scrub_instruction_dict_for_output(inst: dict) -> dict:
    """
    Remove nutrition rows from each step and drop steps that become empty / junk.
    Renumbers keys ``1..n`` so stored ``instructions`` never contain Calories/Fat/etc. lines.
    """
    def _key_order(k: str) -> int:
        try:
            return int(str(k))
        except ValueError:
            return 0

    out: list[str] = []
    for _k in sorted(inst.keys(), key=_key_order):
        v = inst[_k]
        if not isinstance(v, str):
            continue
        cleaned = filter_nutrition_lines_from_instruction_block(v)
        if not cleaned.strip():
            continue
        if is_instruction_step_junk(cleaned):
            continue
        out.append(cleaned)
    return {str(i + 1): out[i] for i in range(len(out))}


def standardize_instruction_step(raw: str) -> str:
    """
    Full deterministic cleanup: units first, then nutrition/parenthetical removal.
    Never call LLMs here (avoids hallucinated steps).
    """
    s = filter_nutrition_lines_from_instruction_block(raw)
    s = normalize_instruction_units(s)
    s = remove_nutrition_and_metadata_lines(s)
    s = strip_inline_nutrition_spans(s)
    s = remove_instruction_non_step_parentheticals(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _instruction_step_needs_micro_clean(text: str) -> bool:
    """Heuristic: run-on / unpunctuated step that may benefit from a one-line LLM polish."""
    t = text.strip()
    if len(t) < 45:
        return False
    # Long line with no sentence-ending punctuation
    if not re.search(r"[.!?][\"'”’]?\s*$", t) and len(t.split()) > 22:
        return True
    # Starts lowercase (likely broken sentence) and fairly long
    if t[0].islower() and len(t) > 70 and t.count(",") < 2:
        return True
    return False


def clean_instruction_sentence(
    ins: str,
    client: OllamaClient,
    model: str,
    ingredient_lines: list[str],
) -> str | None:
    """
    One-step Ollama polish: grammar only; quantities/units must stay faithful.
    Returns None on failure.
    """
    ing_block = "\n".join(f"- {x}" for x in ingredient_lines[:80])
    prompt = f"""Rewrite this cooking instruction as ONE clear, grammatically correct sentence.
Do NOT change any numbers or units. Do NOT add ingredients not implied by the text.
Use ingredient list only as reference for names/units if needed.

Ingredients (reference):
{ing_block}

Instruction:
{ins}

Output one sentence only, no quotes."""
    try:
        out = client.generate_response(model, prompt, stream=False)
        if not out:
            return None
        one = out.strip().split("\n")[0].strip()
        return one if one else None
    except Exception:
        return None


def align_vague_instruction_volumes_from_ingredients(recipe: dict) -> int:
    """
    If ingredients use decimal cups (e.g. 0.5 cup oil), align vague phrasing in instructions
    (\"half a cup\", \"half cup\") to the same numeric form when it looks like the same dish context.
    Returns number of substitution passes applied (approximate).
    """
    ings = recipe.get("ingredients") or []
    blob = " ".join(x.lower() for x in ings if isinstance(x, str))
    n = 0
    inst = recipe.get("instructions")
    if not isinstance(inst, dict):
        return 0

    def sub_if(hay: str, pattern: str, repl: str, flags: int) -> tuple[str, int]:
        s2, k = re.subn(pattern, repl, hay, flags=flags)
        return s2, k

    for k, v in list(inst.items()):
        if not isinstance(v, str):
            continue
        s = v
        if re.search(r"\b0\.5\s+cup\b", blob):
            s, c1 = sub_if(s, r"\bhalf\s+a\s+cup\b", "0.5 cup", re.IGNORECASE)
            s, c2 = sub_if(s, r"\bhalf\s+cup\b", "0.5 cup", re.IGNORECASE)
            s, c3 = sub_if(s, r"\bone[-\s]half\s+cups?\b", "0.5 cup", re.IGNORECASE)
            n += c1 + c2 + c3
        if re.search(r"\b0\.25\s+cup\b", blob):
            s, c4 = sub_if(s, r"\ba\s+quarter\s+cup\b", "0.25 cup", re.IGNORECASE)
            s, c5 = sub_if(s, r"\bquarter\s+cup\b", "0.25 cup", re.IGNORECASE)
            n += c4 + c5
        if re.search(r"\b0\.33\s+cup\b|\b1/3\s+cup\b", blob):
            s, c6 = sub_if(s, r"\bthird\s+of\s+a\s+cup\b", "0.33 cup", re.IGNORECASE)
            s, c7 = sub_if(s, r"\bone[-\s]third\s+cups?\b", "0.33 cup", re.IGNORECASE)
            n += c6 + c7
        inst[k] = s
    return n


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
        st = step.strip()
        if not st:
            continue
        # Catch residues (e.g. ``Un``) if stripping left junk the pre-filter missed.
        if is_instruction_step_junk(st):
            continue
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
            step = standardize_instruction_step(v)
            st = step.strip()
            if st and not is_instruction_step_junk(st):
                cleaned.append(step)
    if not cleaned:
        return None
    return {str(i + 1): cleaned[i] for i in range(len(cleaned))}


def ingredient_instruction_gap_warnings(
    ingredients: list[str],
    instructions: dict | str | None,
) -> list[str]:
    """
    Heuristic: surface cases where common main ingredients appear in instructions
    but not obviously in the ingredient list (for QA / review).
    """
    if not isinstance(instructions, dict):
        return []
    blob = " ".join(str(v).lower() for v in instructions.values() if isinstance(v, str))
    ing_blob = " ".join(x.lower() for x in ingredients)
    out: list[str] = []
    probes = (
        "bacon",
        "sweet potato",
        "sweet potatoes",
        "marinara",
        "mozzarella",
        "shrimp",
        "chicken",
        "beef",
        "salmon",
        "tuna",
        "ravioli",
        "basil",
    )
    for term in probes:
        if term not in blob:
            continue
        if term in ing_blob:
            continue
        if term == "sweet potatoes" and "sweet potato" in ing_blob:
            continue
        if term == "sweet potato" and "sweet potatoes" in ing_blob:
            continue
        matched = False
        for ing in ingredients:
            il = ing.lower()
            if term in il or (len(term) > 4 and term.rstrip("s") in il):
                matched = True
                break
        if not matched:
            out.append(
                f"instructions mention {term!r} but no ingredient line obviously contains it"
            )
    return out


def ordered_recipe_output(
    recipe: dict,
    ingredients_clean: list[str],
    instructions: dict | str | None,
    ing_embedding_text: str,
    recipe_quality_warnings: list[str] | None = None,
) -> dict:
    """
    Stable key order: id, title, description, ingredients, instructions (standardized only),
    then all other fields, optional ``recipe_quality_warnings``, ``ing_embedding_text`` last.
    """
    preferred_head = (
        "id",
        "title",
        "description",
        "ingredients",
        "instructions",
    )
    out: dict = {}
    for k in preferred_head:
        if k == "ingredients":
            out[k] = ingredients_clean
        elif k == "instructions":
            out[k] = instructions
        elif k in recipe:
            out[k] = recipe[k]
    _skip_from_rest = frozenset(
        {"instructions_standardized", "ing_embedding_text", "recipe_quality_warnings"}
    )
    for k, v in recipe.items():
        if k not in out and k not in _skip_from_rest:
            out[k] = v
    if recipe_quality_warnings:
        out["recipe_quality_warnings"] = recipe_quality_warnings
    out["ing_embedding_text"] = ing_embedding_text
    return out


# Splits ``names`` JSON from optional ``// … ingredient name map`` block (spacing may vary).
_INGREDIENT_MAP_SECTION_RE = re.compile(
    r"^\s*//\s*ingredient\s+name\s+map\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def load_unique_names_and_mapping(path: Path) -> tuple[list[str], dict[str, str]]:
    """Load names array + ingredient map from commented JSON file."""
    text = path.read_text(encoding="utf-8")
    m = _INGREDIENT_MAP_SECTION_RE.search(text)
    if not m:
        data = json.loads(re.sub(r"\s*//.*$", "", text, flags=re.MULTILINE))
        return data.get("names", []), {}

    a, b = text[: m.start()], text[m.end() :]

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
) -> tuple[dict, dict]:
    """
    Stages (after ``_process_single_recipe`` has normalized ingredients + instruction text):

    5. Filter junk instruction steps; optionally full-block Ollama rewrite, else deterministic
       ``standardize_instruction_step`` (nutrition strip, etc.). Output uses one ``instructions``
       field only (standardized), not a separate ``instructions_standardized``.
    6. Build ``ing_embedding_text`` from ingredients plus ``tags.ingredient`` (deduped; last key).

    Returns ``(recipe_dict, extra_stats)`` where ``extra_stats`` includes ``instruction_ollama`` (0 or 1).
    """
    extra_stats: dict = {"instruction_ollama": 0}
    raw_ings = recipe.get("ingredients") or []
    ingredients_clean = dedupe_and_sort_ingredient_lines(filter_ingredient_lines(raw_ings))
    ingredients_clean = [
        fix_tight_numeric_unit_spacing(line) if isinstance(line, str) else line
        for line in ingredients_clean
    ]

    inst = recipe.get("instructions")
    instructions_std: dict | str | None = None
    if isinstance(inst, dict):
        inst = scrub_instruction_dict_for_output(inst)

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
                extra_stats["instruction_ollama"] = 1
            else:
                instructions_std = _build_instructions_deterministic_from_pairs(raw_steps)
        else:
            instructions_std = _build_instructions_deterministic_from_pairs(raw_steps)
    else:
        instructions_std = inst

    # Stage 6: embedding string — ingredients + extra foods from ``tags.ingredient`` (deduped)
    tag_ingredients: list[str] = []
    tags_obj = recipe.get("tags")
    if isinstance(tags_obj, dict):
        ti = tags_obj.get("ingredient")
        if isinstance(ti, list):
            tag_ingredients = [x for x in ti if isinstance(x, str)]
        elif isinstance(ti, str) and ti.strip():
            tag_ingredients = [ti]
    ing_emb = build_ing_embedding_text(
        ingredients_clean, mapping, tag_ingredient_labels=tag_ingredients
    )

    recipe_out = {k: v for k, v in recipe.items() if k != "instructions_standardized"}

    quality_warnings = ingredient_instruction_gap_warnings(ingredients_clean, instructions_std)

    return (
        ordered_recipe_output(
            recipe_out,
            ingredients_clean,
            instructions_std,
            ing_emb,
            recipe_quality_warnings=quality_warnings or None,
        ),
        extra_stats,
    )


def _enrich_progress_line(
    *,
    done: int,
    total: int,
    recipe_index: int,
    elapsed: float,
    eta_sec: float | None,
    det: int,
    llm: int,
    repair: int,
    instr_llm: int,
    width: int = 132,
) -> str:
    """Single-line enrich/pipeline status (stderr, \\r updates on TTY)."""
    pct = 100.0 * done / total if total else 100.0
    eta_s = _fmt_duration(eta_sec) if eta_sec is not None else "…"
    line = (
        f"[{done}/{total}] {pct:5.1f}% | recipe #{recipe_index} | "
        f"elapsed {_fmt_duration(elapsed)} | ETA ~{eta_s} | "
        f"det={det} llm={llm} repair={repair} instr_llm={instr_llm}"
    )
    return line[:width].ljust(width)


def enrich_main() -> None:
    ap = argparse.ArgumentParser(description="Enrich recipes (embedding text + instruction cleanup).")
    ap.add_argument("--input", "-i", default="new_recipee_data.json", help="Recipes JSON (array)")
    ap.add_argument(
        "--output",
        "-o",
        default="out.json",
        help="Output JSON (default: out.json in current directory)",
    )
    ap.add_argument("--mapping", "-m", default="unique_ingredient_names.json", help="Names + mapping file")
    ap.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        metavar="N",
        help="Process first N recipes (0 = entire file; default: 20)",
    )
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
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Continue from checkpoint + partial out.json (same --input, --limit, --keep-rest as saved run)",
    )
    ap.add_argument(
        "--checkpoint-file",
        default=None,
        help="Checkpoint path (default: <output>.checkpoint.json next to output file)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing --output when not using --resume",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        metavar="N",
        help="Write output + checkpoint every N recipes (default: 1). Larger = less disk I/O; risk losing up to N−1 recipes if the process is killed between writes.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1,
        metavar="N",
        help="Update progress line every N completed recipes (default: 1)",
    )
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    map_path = Path(args.mapping).expanduser().resolve()
    checkpoint_path = (
        Path(args.checkpoint_file).expanduser().resolve()
        if args.checkpoint_file
        else out_path.with_suffix(out_path.suffix + ".checkpoint.json")
    )

    if not inp.exists():
        print(f"Not found: {inp}", file=sys.stderr)
        sys.exit(1)
    if not map_path.exists():
        print(f"Not found: {map_path}", file=sys.stderr)
        sys.exit(1)

    _cleanup_stale_tmp(out_path)

    if out_path.exists() and not args.resume and args.force and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

    _, mapping = load_unique_names_and_mapping(map_path)

    with open(inp, encoding="utf-8") as f:
        recipes = json.load(f)

    if not isinstance(recipes, list):
        print("Expected JSON array of recipes.", file=sys.stderr)
        sys.exit(1)

    n = len(recipes) if args.limit == 0 else max(0, args.limit)
    head = recipes[:n]
    rest = recipes[n:]
    input_sha = _file_sha256(inp)

    if out_path.exists() and not args.resume:
        if not args.force:
            print(
                f"Output already exists: {out_path}\n"
                f"  Use --resume to continue, or --force to overwrite.",
                file=sys.stderr,
            )
            sys.exit(1)

    next_i = 0
    enriched: list = []

    if args.resume:
        if not checkpoint_path.exists():
            print("Warning: --resume but no checkpoint file; starting from scratch.", file=sys.stderr)
        else:
            with open(checkpoint_path, encoding="utf-8") as f:
                ck = json.load(f)
            if ck.get("input_sha256") != input_sha:
                print(
                    "Checkpoint input_sha256 does not match current --input file; refusing to resume.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if ck.get("input_path") != str(inp):
                print(
                    "Note: checkpoint input_path differs from current --input (continuing; input_sha256 matches).",
                    file=sys.stderr,
                )
            if int(ck.get("limit", -1)) != n:
                print("Checkpoint --limit does not match; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            if bool(ck.get("keep_rest")) != bool(args.keep_rest):
                print("Checkpoint --keep-rest does not match; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            if bool(ck.get("already_normalized")) != bool(args.already_normalized):
                print("Checkpoint --already-normalized does not match; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            if int(ck.get("checkpoint_every", 1)) != int(args.checkpoint_every):
                print("Checkpoint --checkpoint-every does not match; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            next_i = int(ck.get("next_recipe_index", 0))
            next_i = max(0, min(next_i, len(head)))
            if len(head) > 0 and next_i >= len(head) and (ck.get("complete") or next_i == ck.get("head_len")):
                print("Enrich batch already complete (checkpoint). Nothing to do.")
                sys.exit(0)

        if out_path.exists() and next_i > 0:
            with open(out_path, encoding="utf-8") as f:
                saved = json.load(f)
            if not isinstance(saved, list):
                print("Output file is not a JSON array; cannot resume.", file=sys.stderr)
                sys.exit(1)
            if args.keep_rest:
                expect_len = next_i + len(rest)
                if len(saved) < expect_len:
                    print(
                        f"Output has {len(saved)} entries but expected at least {expect_len}; cannot resume.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                enriched = saved[:next_i]
            else:
                if len(saved) < next_i:
                    print(
                        f"Output has {len(saved)} entries but checkpoint expects {next_i}; cannot resume.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                enriched = saved[:next_i]
        elif next_i > 0 and not out_path.exists():
            print("Cannot resume: next_recipe_index > 0 but output file is missing.", file=sys.stderr)
            sys.exit(1)

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
            _print_gpu_and_ollama_status()

    use_ollama_instructions = bool(args.ollama_instructions and client)
    if args.ollama_instructions and not client:
        print(
            "Warning: --ollama-instructions ignored (no Ollama). Use deterministic instruction cleanup.",
            file=sys.stderr,
        )
    if use_ollama_instructions:
        print("Instruction mode: Ollama rewrite (ingredients as context).", flush=True)

    if not args.already_normalized:
        print(
            "Stages 1–4 per recipe: ingredients → instruction units → volume align → optional micro-clean…",
            flush=True,
        )

    if next_i > 0:
        print(f"Resuming at recipe index {next_i}/{len(head)} (partial output already in {out_path}).", flush=True)

    total = len(head)

    def _enrich_checkpoint_payload(done_idx: int, *, complete: bool) -> dict:
        d: dict = {
            "next_recipe_index": done_idx,
            "head_len": len(head),
            "limit": n,
            "input_path": str(inp),
            "input_sha256": input_sha,
            "output_path": str(out_path),
            "keep_rest": args.keep_rest,
            "already_normalized": args.already_normalized,
            "mapping_path": str(map_path),
            "checkpoint_every": int(args.checkpoint_every),
        }
        if complete:
            d["complete"] = True
        return d

    checkpoint_every = max(1, int(args.checkpoint_every))
    progress_every = max(1, int(args.progress_every))
    progress_t0 = time.perf_counter()
    show_progress = (not args.quiet) and (sys.stderr.isatty() or sys.stdout.isatty())
    last_ck_line = False
    det_acc = llm_acc = repair_acc = instr_llm_acc = 0

    try:
        for i in range(next_i, len(head)):
            work = copy.deepcopy(head[i])
            st_norm = {"normalized": 0, "ollama": 0, "repair": 0, "unparseable": 0}
            if not args.already_normalized:
                work, st_norm = _process_single_recipe(work, client, use_ollama, args.model)
            det_acc += int(st_norm.get("normalized", 0))
            llm_acc += int(st_norm.get("ollama", 0))
            repair_acc += int(st_norm.get("repair", 0))
            out_rec, st_ex = process_recipe(
                work,
                mapping,
                client,
                args.model,
                use_ollama,
                use_ollama_instructions=use_ollama_instructions,
            )
            instr_llm_acc += int(st_ex.get("instruction_ollama", 0))
            enriched.append(out_rec)
            done = i + 1
            out_data = enriched + rest if args.keep_rest else enriched
            if done % checkpoint_every == 0 or done == len(head):
                _atomic_write_json(out_path, out_data)
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(_enrich_checkpoint_payload(done, complete=False), f, indent=2)
                ck_msg = f"checkpoint {done}/{len(head)} → {out_path.name}"
                if show_progress:
                    print(f"\n  {ck_msg}", flush=True, file=sys.stderr)
                else:
                    print(f"  {ck_msg}", flush=True, file=sys.stderr)
            elapsed = time.perf_counter() - progress_t0
            completed_this_run = done - next_i
            eta_sec = (
                (len(head) - done) * (elapsed / completed_this_run)
                if completed_this_run > 0
                else None
            )
            if not args.quiet and total and (done % progress_every == 0 or done == len(head)):
                line = _enrich_progress_line(
                    done=done,
                    total=total,
                    recipe_index=i,
                    elapsed=elapsed,
                    eta_sec=eta_sec,
                    det=det_acc,
                    llm=llm_acc,
                    repair=repair_acc,
                    instr_llm=instr_llm_acc,
                )
                if show_progress:
                    print(f"\r{line}  → {out_path.name}", end="", flush=True, file=sys.stderr)
                    last_ck_line = True
                else:
                    print(line.strip(), flush=True, file=sys.stderr)
    except KeyboardInterrupt:
        if show_progress and last_ck_line:
            print(file=sys.stderr)
        if enriched:
            out_data = enriched + rest if args.keep_rest else enriched
            _atomic_write_json(out_path, out_data)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(_enrich_checkpoint_payload(len(enriched), complete=False), f, indent=2)
        print(
            f"\nInterrupted after {len(enriched)} recipe(s). "
            f"Partial output saved to {out_path}\n"
            f"Resume with: python recipe_pipeline.py enrich --resume ... "
            f"(same --input, --limit, --keep-rest, --output, --checkpoint-every)",
            file=sys.stderr,
        )
        sys.exit(130)

    if not args.quiet and total and show_progress:
        print(file=sys.stderr)

    out_data = enriched + rest if args.keep_rest else enriched
    _atomic_write_json(out_path, out_data)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(_enrich_checkpoint_payload(len(head), complete=True), f, indent=2)

    print(f"Wrote {len(out_data)} recipe(s) ({n} enriched) → {out_path}")
    if not args.quiet:
        print(
            f"  Totals: det={det_acc} llm={llm_acc} repair={repair_acc} instr_llm={instr_llm_acc}",
            flush=True,
        )


def pipeline_main() -> None:
    """
    Explicit two-stage run per recipe: (1) normalize ingredients → ``--normalized-output``,
    (2) enrich (embeddings + instructions) → ``--output``. Both files update after each recipe.
    """
    ap = argparse.ArgumentParser(
        description="Pipeline: normalize each recipe, then enrich it (writes normalized.json + out.json).",
    )
    ap.add_argument("--input", "-i", default="new_recipee_data.json", help="Recipes JSON (array)")
    ap.add_argument(
        "--normalized-output",
        "-N",
        default="normalized.json",
        help="After stage 1: ingredient-normalized recipes (default: normalized.json)",
    )
    ap.add_argument("--output", "-o", default="out.json", help="After stage 2: full enrich output")
    ap.add_argument("--mapping", "-m", default="unique_ingredient_names.json", help="Names + mapping file")
    ap.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        metavar="N",
        help="Process first N recipes (0 = entire file; default: 20)",
    )
    ap.add_argument("--keep-rest", action="store_true", help="Append remaining recipes unchanged to both outputs")
    ap.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model")
    ap.add_argument("--no-ollama", action="store_true", help="Skip Ollama for ingredient normalization")
    ap.add_argument(
        "--ollama-instructions",
        action="store_true",
        help="Rewrite instruction steps with Ollama during stage 2",
    )
    ap.add_argument("--quiet", "-q", action="store_true", help="No progress bar")
    ap.add_argument("--resume", action="store_true", help="Resume both partial outputs + checkpoint")
    ap.add_argument("--checkpoint-file", default=None, help="Checkpoint (default: <output>.checkpoint.json)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs when not resuming")
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        metavar="N",
        help="Write both outputs + checkpoint every N recipes (default: 1). Larger = less disk I/O.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1,
        metavar="N",
        help="Update progress line every N completed recipes (default: 1)",
    )
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    norm_path = Path(args.normalized_output).expanduser().resolve()
    map_path = Path(args.mapping).expanduser().resolve()
    checkpoint_path = (
        Path(args.checkpoint_file).expanduser().resolve()
        if args.checkpoint_file
        else out_path.with_suffix(out_path.suffix + ".checkpoint.json")
    )

    if not inp.exists():
        print(f"Not found: {inp}", file=sys.stderr)
        sys.exit(1)
    if not map_path.exists():
        print(f"Not found: {map_path}", file=sys.stderr)
        sys.exit(1)

    _cleanup_stale_tmp(out_path)
    _cleanup_stale_tmp(norm_path)

    if not args.resume and args.force:
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
            except OSError:
                pass

    _, mapping = load_unique_names_and_mapping(map_path)
    with open(inp, encoding="utf-8") as f:
        recipes = json.load(f)
    if not isinstance(recipes, list):
        print("Expected JSON array of recipes.", file=sys.stderr)
        sys.exit(1)

    n = len(recipes) if args.limit == 0 else max(0, args.limit)
    head = recipes[:n]
    rest = recipes[n:]
    input_sha = _file_sha256(inp)

    if out_path.exists() and not args.resume:
        if not args.force:
            print(
                f"Output exists: {out_path}\n  Use --resume or --force.",
                file=sys.stderr,
            )
            sys.exit(1)
    if norm_path.exists() and not args.resume and not args.force:
        print(
            f"Normalized output exists: {norm_path}\n  Use --resume or --force.",
            file=sys.stderr,
        )
        sys.exit(1)
    if norm_path.exists() and not args.resume and args.force:
        try:
            np = norm_path.with_suffix(norm_path.suffix + ".checkpoint.json")
            if np.exists():
                np.unlink()
        except OSError:
            pass

    next_i = 0
    normalized_accum: list = []
    enriched: list = []

    if args.resume:
        if not checkpoint_path.exists():
            print("Warning: --resume but no checkpoint; starting from scratch.", file=sys.stderr)
        else:
            with open(checkpoint_path, encoding="utf-8") as f:
                ck = json.load(f)
            if ck.get("mode") != "pipeline":
                print("Checkpoint is not from pipeline mode; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            if ck.get("input_sha256") != input_sha:
                print("Checkpoint input_sha256 mismatch; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            if ck.get("input_path") != str(inp):
                print(
                    "Note: checkpoint input_path differs from --input (continuing; input_sha256 matches).",
                    file=sys.stderr,
                )
            if int(ck.get("limit", -1)) != n:
                print("Checkpoint --limit mismatch; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            if bool(ck.get("keep_rest")) != bool(args.keep_rest):
                print("Checkpoint --keep-rest mismatch; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            if ck.get("normalized_output_path") != str(norm_path):
                print("Checkpoint normalized output path mismatch; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            if ck.get("output_path") != str(out_path):
                print("Checkpoint output path mismatch; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            if int(ck.get("checkpoint_every", 1)) != int(args.checkpoint_every):
                print("Checkpoint --checkpoint-every mismatch; refusing to resume.", file=sys.stderr)
                sys.exit(1)
            next_i = int(ck.get("next_recipe_index", 0))
            next_i = max(0, min(next_i, len(head)))
            if len(head) > 0 and next_i >= len(head) and ck.get("complete"):
                print("Pipeline already complete. Nothing to do.")
                sys.exit(0)

        if next_i > 0 and (out_path.exists() or norm_path.exists()):
            if not out_path.exists() or not norm_path.exists():
                print("Cannot resume: need both partial out.json and normalized.json.", file=sys.stderr)
                sys.exit(1)
            with open(out_path, encoding="utf-8") as f:
                saved_out = json.load(f)
            with open(norm_path, encoding="utf-8") as f:
                saved_norm = json.load(f)
            if not isinstance(saved_out, list) or not isinstance(saved_norm, list):
                sys.exit(1)
            if args.keep_rest:
                eo = len(saved_out) - len(rest)
                en = len(saved_norm) - len(rest)
            else:
                eo = len(saved_out)
                en = len(saved_norm)
            if eo != next_i or en != next_i:
                print(
                    f"Partial file lengths ({eo}, {en}) != checkpoint next index ({next_i}); cannot resume.",
                    file=sys.stderr,
                )
                sys.exit(1)
            enriched = saved_out[:next_i]
            normalized_accum = saved_norm[:next_i]

    client: OllamaClient | None = None
    use_ollama = not args.no_ollama
    if use_ollama:
        client = OllamaClient()
        if not client.check_connection():
            print("Ollama not reachable; deterministic normalize only.", file=sys.stderr)
            use_ollama = False
            client = None
        else:
            print(f"Using Ollama model: {args.model}", flush=True)
            _print_gpu_and_ollama_status()

    use_ollama_instructions = bool(args.ollama_instructions and client)
    if args.ollama_instructions and not client:
        print("Warning: --ollama-instructions ignored (no Ollama).", file=sys.stderr)

    total = len(head)

    def _pl_ck(done: int, *, complete: bool) -> dict:
        d = {
            "mode": "pipeline",
            "next_recipe_index": done,
            "head_len": len(head),
            "limit": n,
            "input_path": str(inp),
            "input_sha256": input_sha,
            "output_path": str(out_path),
            "normalized_output_path": str(norm_path),
            "keep_rest": args.keep_rest,
            "mapping_path": str(map_path),
            "checkpoint_every": int(args.checkpoint_every),
        }
        if complete:
            d["complete"] = True
        return d

    checkpoint_every = max(1, int(args.checkpoint_every))
    progress_every = max(1, int(args.progress_every))
    progress_t0 = time.perf_counter()
    show_progress = (not args.quiet) and (sys.stderr.isatty() or sys.stdout.isatty())
    last_ck_line = False
    det_acc = llm_acc = repair_acc = instr_llm_acc = 0

    try:
        for i in range(next_i, len(head)):
            work = copy.deepcopy(head[i])
            work, st_norm = _process_single_recipe(work, client, use_ollama, args.model)
            det_acc += int(st_norm.get("normalized", 0))
            llm_acc += int(st_norm.get("ollama", 0))
            repair_acc += int(st_norm.get("repair", 0))
            normalized_accum.append(work)
            norm_data = normalized_accum + rest if args.keep_rest else normalized_accum

            out_rec, st_ex = process_recipe(
                work,
                mapping,
                client,
                args.model,
                use_ollama,
                use_ollama_instructions=use_ollama_instructions,
            )
            instr_llm_acc += int(st_ex.get("instruction_ollama", 0))
            enriched.append(out_rec)
            done = i + 1
            out_data = enriched + rest if args.keep_rest else enriched
            if done % checkpoint_every == 0 or done == len(head):
                _atomic_write_json(norm_path, norm_data)
                _atomic_write_json(out_path, out_data)
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(_pl_ck(done, complete=False), f, indent=2)
                ck_msg = f"checkpoint {done}/{len(head)} → {norm_path.name} + {out_path.name}"
                if show_progress:
                    print(f"\n  {ck_msg}", flush=True, file=sys.stderr)
                else:
                    print(f"  {ck_msg}", flush=True, file=sys.stderr)
            elapsed = time.perf_counter() - progress_t0
            completed_this_run = done - next_i
            eta_sec = (
                (len(head) - done) * (elapsed / completed_this_run)
                if completed_this_run > 0
                else None
            )
            if not args.quiet and total and (done % progress_every == 0 or done == len(head)):
                line = _enrich_progress_line(
                    done=done,
                    total=total,
                    recipe_index=i,
                    elapsed=elapsed,
                    eta_sec=eta_sec,
                    det=det_acc,
                    llm=llm_acc,
                    repair=repair_acc,
                    instr_llm=instr_llm_acc,
                )
                if show_progress:
                    print(f"\r{line}  | {norm_path.name}+{out_path.name}", end="", flush=True, file=sys.stderr)
                    last_ck_line = True
                else:
                    print(line.strip(), flush=True, file=sys.stderr)
    except KeyboardInterrupt:
        if show_progress and last_ck_line:
            print(file=sys.stderr)
        if enriched:
            norm_data = normalized_accum + rest if args.keep_rest else normalized_accum
            out_data = enriched + rest if args.keep_rest else enriched
            _atomic_write_json(norm_path, norm_data)
            _atomic_write_json(out_path, out_data)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(_pl_ck(len(enriched), complete=False), f, indent=2)
        print(
            f"\nInterrupted after {len(enriched)} recipe(s).\n"
            f"  {norm_path.name} = stage 1 (normalized ingredients)\n"
            f"  {out_path.name} = stage 2 (enriched)\n"
            f"Resume: python recipe_pipeline.py pipeline --resume ... "
            f"(same -i, -n, -N, -o, --keep-rest, --checkpoint-every)",
            file=sys.stderr,
        )
        sys.exit(130)

    if not args.quiet and total and show_progress:
        print(file=sys.stderr)

    norm_data = normalized_accum + rest if args.keep_rest else normalized_accum
    out_data = enriched + rest if args.keep_rest else enriched
    _atomic_write_json(norm_path, norm_data)
    _atomic_write_json(out_path, out_data)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(_pl_ck(len(head), complete=True), f, indent=2)

    print(
        f"Pipeline done: {len(out_data)} recipe(s) in {out_path.name} "
        f"(stage 1 snapshot: {norm_path.name})"
    )
    if not args.quiet:
        print(
            f"  Totals: det={det_acc} llm={llm_acc} repair={repair_acc} instr_llm={instr_llm_acc}",
            flush=True,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "normalize":
        sys.argv.pop(1)
        normalize_main()
    elif len(sys.argv) > 1 and sys.argv[1] in ("enrich", "run"):
        sys.argv.pop(1)
        enrich_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "pipeline":
        sys.argv.pop(1)
        pipeline_main()
    else:
        print(
            "Usage: python recipe_pipeline.py {{normalize|enrich|run|pipeline}} [args...]\n"
            "  normalize  — ingredient unit normalization only\n"
            "  enrich|run — normalize + enrich in one step (single output)\n"
            "  pipeline   — per recipe: write normalized.json then out.json (explicit 2 stages)\n",
            file=sys.stderr,
        )
        sys.exit(2)
