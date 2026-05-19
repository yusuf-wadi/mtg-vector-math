"""Step 3 — Validate the trained encoder.

Tests:
  1. King/queen analogies on known MTG card relationships
  2. Deck centroid separation (Sram vs Meren should be far apart)
  3. Nearest-neighbor sanity check (Swords to Plowshares NN should be
     other single-target removal spells)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mtg_vec.encoder import CardEncoder
from mtg_vec.deck import deck_centroid, card_overlay, deck_shape_summary
from mtg_vec.axes import AXIS_IDS, normalize_global

DATA_DIR = _ROOT / "data"
MODELS_DIR = _ROOT / "models"


def load_oracle() -> dict[str, dict]:
    with gzip.open(DATA_DIR / "oracle-slim.json.gz", "rt", encoding="utf-8") as f:
        cards = json.load(f)
    return {c["name"]: c for c in cards if c.get("name")}


def encode_card(enc: CardEncoder, oracle: dict[str, dict], name: str) -> np.ndarray:
    card = oracle.get(name, {})
    return enc.encode(card.get("oracle_text", ""), card.get("type_line", ""))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-9 else 0.0


def nearest_neighbors(query: np.ndarray, corpus: np.ndarray, names: list[str], k: int = 10) -> list[tuple[str, float]]:
    sims = corpus @ query / (
        np.linalg.norm(corpus, axis=1) * np.linalg.norm(query) + 1e-9
    )
    top_k = np.argsort(sims)[::-1][:k]
    return [(names[i], float(sims[i])) for i in top_k]


def main():
    enc = CardEncoder.load()
    axis_scale = np.load(MODELS_DIR / "axis_scale.npy")
    oracle = load_oracle()
    print(f"Loaded encoder. Oracle: {len(oracle)} cards.")

    # ------------------------------------------------------------------
    # 1. King/queen analogies
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KING/QUEEN ANALOGIES")
    print("=" * 60)

    analogies = [
        # (A - B + C ≈ D)
        ("Brainstorm",         "draw_card_axis",  "graveyard_axis",  "Faithless Looting"),
        ("Wrath of God",       "destroy",          "exile",           "Terminus"),
        ("Llanowar Elves",     "creature_body",    "instant",         "Elvish Spirit Guide"),
        ("Swords to Plowshares","exile_target",    "destroy_target",  "Path to Exile"),
    ]

    # For card-level analogies we use the axis vectors directly
    for a_name, _, _, d_name in analogies:
        if a_name not in oracle or d_name not in oracle:
            print(f"  skip: {a_name} or {d_name} not in oracle")
            continue
        vec_a = encode_card(enc, oracle, a_name)
        vec_d = encode_card(enc, oracle, d_name)
        sim = cosine(vec_a, vec_d)
        print(f"  cosine({a_name}, {d_name}) = {sim:.4f}")

    # ------------------------------------------------------------------
    # 2. Nearest neighbors for known cards
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("NEAREST NEIGHBORS")
    print("=" * 60)

    # Build a small corpus of well-known cards for NN search
    sample_names = [
        "Swords to Plowshares", "Path to Exile", "Vindicate", "Anguished Unmaking",
        "Counterspell", "Mana Drain", "Force of Will", "Negate",
        "Cultivate", "Kodama's Reach", "Rampant Growth", "Farseek",
        "Sol Ring", "Arcane Signet", "Fellwar Stone",
        "Brainstorm", "Ponder", "Preordain", "Faithless Looting",
        "Lightning Bolt", "Terminate", "Doom Blade",
        "Llanowar Elves", "Birds of Paradise", "Elvish Mystic",
        "Craterhoof Behemoth", "Avenger of Zendikar",
        "Stoneforge Mystic", "Puresteel Paladin",
        "Skullclamp", "Sword of Fire and Ice",
    ]
    sample_names = [n for n in sample_names if n in oracle]
    corpus = np.array([encode_card(enc, oracle, n) for n in sample_names])

    for query_name in ["Swords to Plowshares", "Llanowar Elves", "Counterspell", "Stoneforge Mystic"]:
        if query_name not in oracle:
            continue
        qvec = encode_card(enc, oracle, query_name)
        neighbors = nearest_neighbors(qvec, corpus, sample_names, k=5)
        print(f"\n  NN({query_name}):")
        for name, sim in neighbors:
            if name != query_name:
                print(f"    {sim:.4f}  {name}")

    # ------------------------------------------------------------------
    # 3. Deck centroid separation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DECK CENTROID SEPARATION")
    print("=" * 60)

    # Sram Equipment core cards
    sram_cards = [
        "Sram, Senior Edificer", "Stoneforge Mystic", "Puresteel Paladin",
        "Sword of Fire and Ice", "Skullclamp", "Lightning Greaves",
        "Swiftfoot Boots", "Open the Armory", "Swords to Plowshares",
        "Sol Ring",
    ]
    # Meren Aristocrats core cards
    meren_cards = [
        "Meren of Clan Nel Toth", "Viscera Seer", "Grave Pact",
        "Altar of Dementia", "Savra, Queen of the Golgari",
        "Mikaeus, the Unhallowed", "Animate Dead", "Reanimate",
        "Carrion Feeder", "Zulaport Cutthroat",
    ]

    def build_centroid(names):
        vecs = [encode_card(enc, oracle, n) for n in names if n in oracle]
        if not vecs:
            return None
        return deck_centroid(np.array(vecs))

    sram_centroid = build_centroid(sram_cards)
    meren_centroid = build_centroid(meren_cards)

    if sram_centroid is not None and meren_centroid is not None:
        sim = cosine(sram_centroid, meren_centroid)
        print(f"  cosine(Sram centroid, Meren centroid) = {sim:.4f}")
        print(f"  (lower = more distinct — target < 0.7)")

        sram_summary = deck_shape_summary(sram_centroid, axis_scale)
        meren_summary = deck_shape_summary(meren_centroid, axis_scale)

        print("\n  Sram top families:")
        for f in sram_summary["top_families"]:
            print(f"    {f['label']:<22} {f['score']:.3f}  [{f['winning_axis']}]")

        print("\n  Meren top families:")
        for f in meren_summary["top_families"]:
            print(f"    {f['label']:<22} {f['score']:.3f}  [{f['winning_axis']}]")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
