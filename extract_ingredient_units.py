#!/usr/bin/env python3
"""
Extract all ingredient units from recipes_images.json.
Outputs: ingredient_units.json (full list with counts), ingredient_units.txt (plain list).
"""
import json
import re
from collections import Counter

with open("recipes_images.json") as f:
    data = json.load(f)

all_ingredients = []
for recipe in data:
    for ing in recipe.get("ingredients") or []:
        if ing and isinstance(ing, str):
            all_ingredients.append(ing.strip())

# Normalize to singular/canonical form
CANONICAL = {
    "cups": "cup", "tablespoons": "tablespoon", "teaspoons": "teaspoon",
    "tbsp": "tablespoon", "tsp": "teaspoon",
    "fluid ounces": "fluid ounce", "fl oz": "fluid ounce",
    "ounces": "ounce", "oz": "ounce", "pounds": "pound", "lb": "pound", "lbs": "pound",
    "grams": "gram", "g": "gram", "kilograms": "kilogram", "kg": "kilogram",
    "liters": "liter", "litres": "liter", "milliliters": "milliliter",
    "millilitres": "milliliter", "ml": "milliliter",
    "quarts": "quart", "pints": "pint", "gallons": "gallon",
    "cans": "can", "packages": "package", "jars": "jar", "bottles": "bottle",
    "bags": "bag", "boxes": "box", "envelopes": "envelope", "bars": "bar",
    "pinches": "pinch", "dashes": "dash",
    "cloves": "clove", "sprigs": "sprig", "bunches": "bunch", "stalks": "stalk",
    "heads": "head", "slices": "slice", "pieces": "piece", "strips": "strip",
    "cubes": "cube", "drops": "drop", "handfuls": "handful", "sheets": "sheet",
    "inches": "inch", "links": "link", "fillets": "fillet", "loaves": "loaf",
    "bulbs": "bulb", "leaves": "leaf", "scoops": "scoop", "squares": "square",
    "sticks": "stick",
}
unit_tokens = [
    "fluid ounces", "fl oz", "milliliters", "millilitres", "fluid ounce",
    "tablespoons", "teaspoons", "cups", "ounces", "pounds", "grams", "kilograms",
    "liters", "litres", "quarts", "pints", "gallons", "tbsp", "tsp",
    "oz", "lb", "lbs", "g", "kg", "ml",
    "tablespoon", "teaspoon", "cup", "ounce", "pound", "gram", "kilogram",
    "liter", "litre", "quart", "pint", "gallon",
    "cans", "packages", "jars", "bottles", "bags", "boxes", "envelopes", "bars",
    "can", "package", "jar", "bottle", "bag", "box", "envelope", "bar",
    "pinches", "dash", "dashes", "pinch",
    "cloves", "sprigs", "bunches", "stalks", "heads", "slices", "pieces", "strips",
    "clove", "sprig", "bunch", "stalk", "head", "slice", "piece", "strip",
    "cubes", "cube", "drops", "drop", "handfuls", "handful", "sheets", "sheet",
    "inches", "inch", "links", "link", "fillets", "fillet", "loaves", "loaf",
    "bulbs", "bulb", "leaves", "leaf", "scoops", "scoop", "squares", "square",
    "stick", "sticks", "globs", "glob",
]

pattern = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in unit_tokens) + r")\b",
    re.IGNORECASE
)

counts = Counter()
for ing in all_ingredients:
    for m in pattern.finditer(ing):
        token = m.group(1).lower()
        canonical = CANONICAL.get(token, token)
        counts[canonical] += 1

sorted_units = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

# Write JSON: list of { "unit": "...", "count": N }
out_json = [{"unit": u, "count": c} for u, c in sorted_units]
with open("ingredient_units.json", "w") as f:
    json.dump(out_json, f, indent=2)

# Write plain list (one unit per line)
with open("ingredient_units.txt", "w") as f:
    f.write("# Ingredient units extracted from recipes_images.json (canonical, by frequency)\n")
    for u, c in sorted_units:
        f.write(f"{u}\t{c}\n")

print(f"Extracted {len(sorted_units)} unique units.")
print("Written: ingredient_units.json, ingredient_units.txt")
