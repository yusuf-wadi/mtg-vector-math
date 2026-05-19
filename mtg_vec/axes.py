"""Axis definitions, family map, and normalization utilities.

The 105 axes are the dimensions of the MTG embedding space.
The 12 families are the radar display layer — a lossy but human-readable
collapse of the 105D space.

Normalization contract:
  All vectors fed to the radar or used for distance calculations
  must be normalized. Two strategies:

  1. Per-card min-max (local): each card's vector scaled to [0,1] 
     independently. Good for radar display of a single card.

  2. Per-axis corpus-wide (global): each axis scaled by its 99th 
     percentile across the training corpus. Better for inter-card 
     distance calculations and deck centroid comparisons.
     Use normalize_global() after fitting on the training set.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

_AXES_PATH = Path(__file__).resolve().parent.parent / "data" / "mtg-axes.json"

# Ordered list of all 105 axis IDs — order is stable and matches the
# 105D vector dimensions throughout the codebase.
AXIS_IDS: list[str] = [
    # zones (6)
    "graveyard", "library", "hand", "exile", "stack", "command",
    # subtypes_qualifiers (6)
    "legendary", "equipment", "aura", "vehicle", "tribal_creature", "snow",
    # casting_and_costs (11)
    "additional_cost", "alternative_cost", "x_cost", "life_payment",
    "mana_ability", "kicker", "flashback", "cascade", "convoke", "delve", "cycling",
    # actions_offense (22)
    "attack", "block", "combat_damage", "deal_damage", "first_strike",
    "double_strike", "trample", "menace", "flying", "reach", "haste",
    "vigilance", "deathtouch", "lifelink", "indestructible", "hexproof",
    "shroud", "ward", "protection", "extra_combat", "prowess", "landfall",
    # actions_disruption (10)
    "destroy", "exile_action", "counter_spell", "bounce", "discard",
    "mill", "sacrifice", "damage_prevention", "tap_opponent", "goad",
    # actions_advantage (10)
    "draw_card", "tutor", "scry", "surveil", "return_to_hand",
    "return_to_battlefield", "ramp", "treasure", "dredge", "storm",
    # triggers_state (9)
    "enters_battlefield", "leaves_battlefield", "dies", "attacks_trigger",
    "blocks_trigger", "cast_trigger", "upkeep_trigger", "end_step_trigger",
    "control_change",
    # counters_and_modifications (8)
    "plus_one_counter", "minus_one_counter", "loyalty_counter", "charge_counter",
    "experience_counter", "energy", "proliferate", "power_toughness_buff",
    # tokens_and_creation (5)
    "create_token", "copy_spell", "food", "clue", "blood",
    # static_keywords (8)
    "flash", "defender", "phasing", "morph", "modal_dfc", "saga",
    "adventure", "transform",
    # commander_specific (5)
    "commander_cast", "partner", "monarch", "initiative", "venture_dungeon",
    # broader_concepts (5)
    "target_any", "untargeted", "may_choice", "replacement_effect", "devotion",
]

assert len(AXIS_IDS) == 105, f"Expected 105 axes, got {len(AXIS_IDS)}"

AXIS_INDEX: dict[str, int] = {a: i for i, a in enumerate(AXIS_IDS)}

# 12 families — radar display layer
FAMILIES: dict[str, dict] = {
    "zones":                    {"label": "Zones",            "axes": AXIS_IDS[0:6]},
    "subtypes_qualifiers":      {"label": "Subtypes",         "axes": AXIS_IDS[6:12]},
    "casting_and_costs":        {"label": "Casting & costs",  "axes": AXIS_IDS[12:23]},
    "actions_offense":          {"label": "Offense / combat", "axes": AXIS_IDS[23:45]},
    "actions_disruption":       {"label": "Disruption",       "axes": AXIS_IDS[45:55]},
    "actions_advantage":        {"label": "Card advantage",   "axes": AXIS_IDS[55:65]},
    "triggers_state":           {"label": "Triggers & state", "axes": AXIS_IDS[65:74]},
    "counters_and_modifications":{"label": "Counters & mods", "axes": AXIS_IDS[74:82]},
    "tokens_and_creation":      {"label": "Tokens & creation","axes": AXIS_IDS[82:87]},
    "static_keywords":          {"label": "Static keywords",  "axes": AXIS_IDS[87:95]},
    "commander_specific":       {"label": "Commander",        "axes": AXIS_IDS[95:100]},
    "broader_concepts":         {"label": "Game concepts",    "axes": AXIS_IDS[100:105]},
}

AXIS_TO_FAMILY: dict[str, str] = {
    axis: fam
    for fam, data in FAMILIES.items()
    for axis in data["axes"]
}


def load_axis_seeds() -> dict[str, str]:
    """Return {axis_id: seed_text} from mtg-axes.json."""
    with open(_AXES_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, str] = {}
    for fam_id, items in raw.items():
        if fam_id.startswith("_"):
            continue
        for item in items:
            out[item["id"]] = item["seed"]
    return out


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_local(vec: np.ndarray) -> np.ndarray:
    """Min-max normalize a single 105D vector to [0, 1].
    Use for radar display of a single card or deck centroid."""
    mn, mx = vec.min(), vec.max()
    if mx - mn < 1e-9:
        return np.zeros_like(vec)
    return (vec - mn) / (mx - mn)


def normalize_global(vec: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Scale a 105D vector by per-axis corpus scale factors.
    scale[i] = 99th-percentile of axis i across the training corpus.
    Use for inter-card distance calculations and centroid comparisons."""
    out = vec / (scale + 1e-9)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Radar collapse: 105D → 12D
# ---------------------------------------------------------------------------

def radar_collapse(vec: np.ndarray, method: str = "max") -> dict[str, float]:
    """Collapse a 105D vector to a 12-spoke radar dict.

    method:
      'max'  — loudest axis per family (current display strategy)
      'mean' — average across all axes in the family

    Returns {family_id: score} for all 12 families.
    """
    result: dict[str, float] = {}
    for fam_id, data in FAMILIES.items():
        indices = [AXIS_INDEX[a] for a in data["axes"]]
        family_vec = vec[indices]
        if method == "max":
            result[fam_id] = float(family_vec.max())
        elif method == "mean":
            result[fam_id] = float(family_vec.mean())
        else:
            raise ValueError(f"Unknown method: {method}")
    return result


def radar_winning_axis(vec: np.ndarray) -> dict[str, str]:
    """Return the winning axis ID per family — the spoke label for the UI."""
    result: dict[str, str] = {}
    for fam_id, data in FAMILIES.items():
        indices = [AXIS_INDEX[a] for a in data["axes"]]
        family_vec = vec[indices]
        best_local_idx = int(family_vec.argmax())
        result[fam_id] = data["axes"][best_local_idx]
    return result
