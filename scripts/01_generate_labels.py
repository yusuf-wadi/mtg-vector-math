"""Step 1 — Generate 105D soft labels for all MTG cards.

Uses bge-small-en-v1.5 as the teacher model:
  - Embed all 105 axis seeds once
  - Embed each card's oracle text + type line
  - Label[i] = cosine_similarity(card_embedding, axis_seed_embedding[i])
  - Clip to [0, 1] (cosine sim can be slightly negative for unrelated text)

Output: data/labels.npy          shape (N_cards, 105)  float32
        data/card_index.json     list of card names in row order

Runtime: ~10 min on CPU, ~2 min on GPU for 30k cards.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mtg_vec.axes import AXIS_IDS, load_axis_seeds

DATA_DIR = _ROOT / "data"
ORACLE_PATH = DATA_DIR / "oracle-slim.json.gz"
AXES_PATH = DATA_DIR / "mtg-axes.json"
LABELS_OUT = DATA_DIR / "labels.npy"
INDEX_OUT = DATA_DIR / "card_index.json"

BATCH_SIZE = 256
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_cards() -> list[dict]:
    with gzip.open(ORACLE_PATH, "rt", encoding="utf-8") as f:
        cards = json.load(f)
    # Filter to cards with oracle text
    return [c for c in cards if c.get("oracle_text") or c.get("type_line")]


def card_text(card: dict) -> str:
    return f"{card.get('type_line', '')} {card.get('oracle_text', '')}".strip()


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (N, D), B: (M, D) → (N, M) cosine similarities."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Embed axis seeds
    print("Embedding 105 axis seeds...")
    axis_seeds = load_axis_seeds()
    seed_texts = [axis_seeds[a] for a in AXIS_IDS]
    axis_embeddings = model.encode(
        seed_texts, batch_size=32, show_progress_bar=False,
        normalize_embeddings=True,
    )  # shape (105, D)

    # Load cards
    print("Loading cards...")
    cards = load_cards()
    print(f"  {len(cards)} cards found")

    card_names = [c.get("name", f"card_{i}") for i, c in enumerate(cards)]
    texts = [card_text(c) for c in cards]

    # Embed cards in batches
    print("Embedding cards...")
    all_labels: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        card_embs = model.encode(
            batch, batch_size=BATCH_SIZE, show_progress_bar=False,
            normalize_embeddings=True,
        )  # shape (batch, D)
        # cosine sim to each of 105 axes — already normalized so just dot product
        sims = card_embs @ axis_embeddings.T  # shape (batch, 105)
        all_labels.append(sims.astype(np.float32))

    labels = np.concatenate(all_labels, axis=0)  # (N, 105)

    # Clip to [0, 1] — negatives mean "unrelated", treat as 0
    labels = np.clip(labels, 0.0, 1.0)

    print(f"Labels shape: {labels.shape}")
    print(f"  Mean nonzero per card: {(labels > 0.05).sum(axis=1).mean():.1f} axes")
    print(f"  Global max: {labels.max():.4f}  mean: {labels.mean():.4f}")

    np.save(LABELS_OUT, labels)
    with open(INDEX_OUT, "w") as f:
        json.dump(card_names, f)

    print(f"Saved labels → {LABELS_OUT}")
    print(f"Saved index  → {INDEX_OUT}")


if __name__ == "__main__":
    main()
